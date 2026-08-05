"""
FER_strong_cv10fold.py
----------------------
STRATIFIED 10-FOLD CROSS-VALIDATION of the strong single-stream ViT FER recipe,
run on the TRAINING CSV ONLY. This is the cross-validation counterpart of
FER_strong_baseline.py.

WHAT CHANGED vs FER_strong_baseline.py
  * No fixed TRAIN/TEST evaluation. Instead StratifiedKFold(n_splits=10) partitions
    the TRAINING csv so every row is held out exactly once; the held-out fold is that
    fold's evaluation set. TEST_CSV is NOT read at all.
  * NO MODEL IS EVER SAVED. The final trainer.save_model()/image_processor.save_pretrained()
    calls are removed. Per-fold checkpoints still exist transiently (they are required for
    best-checkpoint selection), but they are written to node-local SCRATCH
    ($SLURM_TMPDIR / $TMPDIR / /tmp) and deleted after every fold. Nothing under
    RESULTS_DIR is a model file -- only .npy arrays, .json reports, and .csv summaries.
  * Best-checkpoint selection stays clean: a small stratified INNER validation slice is
    carved from the 9 training folds and used for load_best_model_at_end (macro-F1). The
    held-out test fold is used strictly for final scoring, never for selection.

WHAT IT REPORTS
  (1) Per-fold accuracy / macro-F1, then mean +/- SD across the 10 folds.
  (2) POOLED out-of-fold metrics: folds are disjoint and cover every training row once,
      so concatenating each fold's predictions yields one prediction per row -> a single
      overall accuracy / macro-F1 / confusion matrix over the whole training set.

COMPUTE NOTE: trains a fresh model 10 times (~10x the single-run cost). With EPOCHS=20
that is 200 epochs end to end; lower EPOCHS for a faster CV estimate.

Launch is unchanged from the baseline (DDP is fine here -- all writes are guarded to the
main process):
    accelerate launch --num_processes=2 --mixed_precision=fp16 FER_strong_cv10fold.py
"""
import os
import gc
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed,
)
from accelerate import Accelerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ------------------------------------------------------------------ Accelerator
accelerator = Accelerator()


def mprint(*args, **kwargs):
    """Print only on the main process (avoids duplicate lines under DDP)."""
    if accelerator.is_main_process:
        print(*args, **kwargs)


mprint(f"Detected {accelerator.num_processes} GPU(s). Strong FER 10-fold CV (train only).")

# ------------------------------------------------------------------ Config
# --- Point TRAIN_CSV at ONE dataset's training split (RAF-DB or AffectNet). ---
# TEST_CSV is intentionally NOT used: cross-validation runs on the training data only.
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

# ---- Cross-validation ----
N_FOLDS = 10                 # stratified K-fold on the training CSV
SPLIT_SEED = 42              # controls the fold partition (reproducible)
INNER_VAL_FRACTION = 0.05    # stratified slice carved from the 9 training folds for
                             # best-ckpt selection. Set 0.0 to select on the test fold
                             # instead (simpler but mildly optimistic).

# ---- Backbone: face-domain pretrained ViT is a much better init than ImageNet ViT ----
MODEL_CHECKPOINT = "trpakov/vit-face-expression"
# Alternatives to try (one-line swap):
#   "microsoft/swin-base-patch4-window7-224"   (strong ImageNet backbone)
#   "google/vit-base-patch16-224-in21k"        (your current baseline)

IMG_SIZE = 224
BATCH_SIZE = 64           # effective batch = 64 * num_gpus; drop to 32 on OOM
EPOCHS = 20               # augmentation lets you train longer without overfitting
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True  # keep True for AffectNet; optional for RAF-DB
OUTPUT_DIR = "./fer_strong_cv10fold_results"   # RESULTS ONLY -- never a model

# Transient training checkpoints go here, then get deleted after each fold.
SCRATCH_ROOT = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or tempfile.gettempdir()
)

set_seed(SPLIT_SEED)
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
accelerator.wait_for_everyone()

# ------------------------------------------------------------------ Data (train CSV only)
mprint("Loading TRAINING CSV (test CSV is not used)...")
train_df = pd.read_csv(TRAIN_CSV)
train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].astype(int)
train_df = train_df.reset_index(drop=True)

labels = sorted(train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(l): i for i, l in enumerate(labels)}
id2label = {i: int(l) for i, l in enumerate(labels)}

# Contiguous class-id array over the whole training set -> drives stratification.
y_all = train_df[LABEL_COLUMN].map(label2id).values

mprint("=" * 70)
mprint("RUN CONFIG (stratified K-fold CV on TRAIN only)")
mprint("=" * 70)
mprint(f"Backbone             : {MODEL_CHECKPOINT}")
mprint(f"Train CSV            : {TRAIN_CSV}")
mprint(f"Rows / classes       : {len(train_df)} / {labels}")
mprint(f"Folds                : {N_FOLDS}   (split seed {SPLIT_SEED})")
mprint(f"Inner val fraction   : {INNER_VAL_FRACTION}  (best-ckpt selection; test fold untouched)")
mprint(f"Epochs / fold        : {EPOCHS}")
mprint(f"Results dir          : {os.path.abspath(OUTPUT_DIR)}")
mprint(f"Scratch (checkpoints): {SCRATCH_ROOT}   (deleted after each fold)")
mprint("Model weights kept   : NO  (checkpoints are transient scratch, then removed)")
mprint("Independent test CSV : NOT used (CV on training data only)")

# ------------------------------------------------------------------ Transforms
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

# Training augmentation. Face-safe: mild crop, flip (expressions ~symmetric),
# RandAugment for photometric/geometric variety, random erasing for occlusion robustness.
train_ops = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandAugment(num_ops=2, magnitude=7),     # magnitude 7 is gentle enough for faces
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
eval_ops = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


def make_transform(ops):
    def _t(batch):
        imgs = [Image.open(p).convert("RGB") for p in batch[IMAGE_COLUMN]]
        pixel_values = torch.stack([ops(im) for im in imgs])
        return {
            "pixel_values": pixel_values,
            "labels": [label2id[l] for l in batch[LABEL_COLUMN]],
        }
    return _t


# ------------------------------------------------------------------ Custom loss
class WeightedSmoothTrainer(Trainer):
    """Cross-entropy with class weights + label smoothing."""
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = F.cross_entropy(
            logits, labels, weight=w, label_smoothing=self._label_smoothing
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, gold = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": float(np.mean(preds == gold)),
        "macro_f1": float(f1_score(gold, preds, average="macro")),
    }


def print_confusion_matrix(cm, class_names, title="Confusion matrix (rows=true, cols=pred)"):
    mprint(f"\n{title}")
    col_w = max(6, max(len(str(c)) for c in class_names) + 1)
    header = " " * col_w + "".join(f"{str(c):>{col_w}}" for c in class_names)
    mprint(header)
    for i, row in enumerate(cm):
        mprint(f"{str(class_names[i]):>{col_w}}" + "".join(f"{int(v):>{col_w}}" for v in row))
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = cm.diagonal() / cm.sum(axis=1)
    mprint("Per-class recall: " + ", ".join(
        f"{class_names[i]}={r:.3f}" for i, r in enumerate(recalls)
    ))


# ------------------------------------------------------------------ Single-fold run
def run_one_fold(fold_idx, train_idx, test_idx):
    mprint("\n" + "=" * 70)
    mprint(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    mprint("=" * 70)

    fold_seed = SPLIT_SEED + fold_idx
    set_seed(fold_seed)

    ckpt_dir = os.path.join(SCRATCH_ROOT, f"fer_ckpt_fold{fold_idx}")

    # ---- inner stratified val slice for best-ckpt selection (test fold untouched) ----
    if INNER_VAL_FRACTION and INNER_VAL_FRACTION > 0.0:
        tr_idx, val_idx = train_test_split(
            train_idx,
            test_size=INNER_VAL_FRACTION,
            stratify=y_all[train_idx],
            random_state=fold_seed,
            shuffle=True,
        )
    else:
        tr_idx, val_idx = train_idx, test_idx   # mildly optimistic fallback

    tr_idx = np.asarray(tr_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    mprint(f"  train / inner-val / test sizes : {len(tr_idx)} / {len(val_idx)} / {len(test_idx)}")

    # Build per-fold DataFrames (positional indices into train_df)
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)
    te_df = train_df.iloc[test_idx].reset_index(drop=True)

    # Class weights from this fold's training slice
    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=y_all[tr_idx])
    class_weights_t = torch.tensor(cw, dtype=torch.float) if USE_CLASS_WEIGHTS else None

    train_ds = Dataset.from_pandas(tr_df).with_transform(make_transform(train_ops))
    val_ds = Dataset.from_pandas(va_df).with_transform(make_transform(eval_ops))
    test_ds = Dataset.from_pandas(te_df).with_transform(make_transform(eval_ops))

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=ckpt_dir,                   # <-- scratch; cleaned up after the fold
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",                 # required for best-ckpt selection (scratch)
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,           # keep the best epoch, not the overfit last one
        metric_for_best_model="macro_f1",      # right target under class imbalance
        greater_is_better=True,
        save_total_limit=1,                    # at most one transient checkpoint on scratch
        logging_steps=50,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        seed=fold_seed,
        data_seed=fold_seed,
        report_to="none",
    )

    trainer = WeightedSmoothTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,                   # selection on inner val, NOT the test fold
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        class_weights=class_weights_t,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer.train()

    # ---- final evaluation on this fold's held-out slice (collective under DDP) ----
    test_results = trainer.predict(test_ds)
    logits = test_results.predictions
    preds = np.argmax(logits, axis=1)
    gold = test_results.label_ids

    fold_acc = accuracy_score(gold, preds)
    fold_mf1 = f1_score(gold, preds, average="macro")
    cm = confusion_matrix(gold, preds, labels=np.arange(num_labels))
    report = classification_report(
        gold, preds, labels=np.arange(num_labels),
        target_names=[str(id2label[i]) for i in range(num_labels)],
        output_dict=True, zero_division=0,
    )

    # ---- print + save (results only, never a model), on main process ----
    if accelerator.is_main_process:
        mprint(f"\nFold {fold_idx + 1} accuracy: {fold_acc:.4f} | macro-F1: {fold_mf1:.4f}")
        mprint(classification_report(
            gold, preds, labels=np.arange(num_labels),
            target_names=[str(id2label[i]) for i in range(num_labels)],
            digits=4, zero_division=0,
        ))
        print_confusion_matrix(cm, [id2label[i] for i in range(num_labels)],
                               title=f"[Fold {fold_idx + 1}] confusion matrix")

        np.save(os.path.join(OUTPUT_DIR, f"logits_fold{fold_idx}.npy"), logits)
        np.save(os.path.join(OUTPUT_DIR, f"preds_fold{fold_idx}.npy"), preds)
        np.save(os.path.join(OUTPUT_DIR, f"labels_fold{fold_idx}.npy"), gold)
        np.save(os.path.join(OUTPUT_DIR, f"testidx_fold{fold_idx}.npy"), test_idx)
        np.save(os.path.join(OUTPUT_DIR, f"confmat_fold{fold_idx}.npy"), cm)
        with open(os.path.join(OUTPUT_DIR, f"report_fold{fold_idx}.json"), "w") as f:
            json.dump(report, f, indent=2)

    # ---- free memory on ALL ranks; delete transient checkpoint on main ----
    del trainer, model, test_results
    gc.collect()
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    accelerator.wait_for_everyone()

    return {
        "fold": fold_idx,
        "accuracy": fold_acc,
        "macro_f1": fold_mf1,
        "report": report,
        "confusion_matrix": cm,
        "test_idx": test_idx,
        "preds": preds,
        "labels": gold,
        "logits": logits,
    }


# ------------------------------------------------------------------ Run all folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
fold_splits = list(skf.split(np.zeros(len(y_all)), y_all))

if accelerator.is_main_process:
    with open(os.path.join(OUTPUT_DIR, "fold_indices.json"), "w") as f:
        json.dump(
            {f"fold{k}": {"test_idx": te.tolist()} for k, (_, te) in enumerate(fold_splits)},
            f,
        )

all_runs = [run_one_fold(k, tr, te) for k, (tr, te) in enumerate(fold_splits)]

# ------------------------------------------------------------------ Aggregate (main only)
if accelerator.is_main_process:
    class_names = [id2label[i] for i in range(num_labels)]

    # (1) per-fold metrics: mean +/- SD across folds
    accuracies = np.array([r["accuracy"] for r in all_runs]) * 100
    macro_f1s = np.array([r["macro_f1"] for r in all_runs]) * 100
    mean_acc, std_acc = accuracies.mean(), accuracies.std(ddof=1)
    mean_mf1, std_mf1 = macro_f1s.mean(), macro_f1s.std(ddof=1)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS OVER FOLDS (per-fold mean +/- SD)")
    print("=" * 70)
    for r, a, m in zip(all_runs, accuracies, macro_f1s):
        print(f"  fold {r['fold'] + 1:>2}:  acc {a:.2f} %   macro-F1 {m:.2f} %")
    print(f"\nAccuracy:  {mean_acc:.2f} +/- {std_acc:.2f} %  (mean +/- SD, k={N_FOLDS})")
    print(f"Macro-F1:  {mean_mf1:.2f} +/- {std_mf1:.2f} %")
    print(f"Acc min/max: {accuracies.min():.2f} / {accuracies.max():.2f} %")

    # per-class F1 mean +/- SD across folds
    print("\nPer-class F1 (mean +/- SD across folds):")
    per_class_summary = {}
    for i in range(num_labels):
        key = str(id2label[i])
        f1s = np.array([r["report"][key]["f1-score"] for r in all_runs])
        rec = np.array([r["report"][key]["recall"] for r in all_runs])
        prec = np.array([r["report"][key]["precision"] for r in all_runs])
        per_class_summary[key] = {
            "precision_mean": float(prec.mean()), "precision_std": float(prec.std(ddof=1)),
            "recall_mean": float(rec.mean()), "recall_std": float(rec.std(ddof=1)),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)),
        }
        print(f"  class {key:<6} {f1s.mean():.3f} +/- {f1s.std(ddof=1):.3f}")

    # (2) pooled out-of-fold metrics over the whole training set
    oof_idx = np.concatenate([r["test_idx"] for r in all_runs])
    oof_preds = np.concatenate([r["preds"] for r in all_runs])
    oof_labels = np.concatenate([r["labels"] for r in all_runs])
    oof_logits = np.concatenate([r["logits"] for r in all_runs], axis=0)

    order = np.argsort(oof_idx)   # restore original training-row order (0..N-1)
    oof_idx, oof_preds = oof_idx[order], oof_preds[order]
    oof_labels, oof_logits = oof_labels[order], oof_logits[order]

    pooled_acc = accuracy_score(oof_labels, oof_preds) * 100
    pooled_mf1 = f1_score(oof_labels, oof_preds, average="macro") * 100
    pooled_cm = confusion_matrix(oof_labels, oof_preds, labels=np.arange(num_labels))

    np.save(os.path.join(OUTPUT_DIR, "oof_idx.npy"), oof_idx)
    np.save(os.path.join(OUTPUT_DIR, "oof_preds.npy"), oof_preds)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), oof_labels)
    np.save(os.path.join(OUTPUT_DIR, "oof_logits.npy"), oof_logits)
    np.save(os.path.join(OUTPUT_DIR, "confmat_pooled.npy"), pooled_cm)

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD RESULTS (every training row predicted once)")
    print("=" * 70)
    print(f"Pooled accuracy: {pooled_acc:.2f} %   |   pooled macro-F1: {pooled_mf1:.2f} %")
    print(classification_report(
        oof_labels, oof_preds, labels=np.arange(num_labels),
        target_names=[str(id2label[i]) for i in range(num_labels)],
        digits=4, zero_division=0,
    ))
    print_confusion_matrix(pooled_cm, class_names, title="Pooled out-of-fold confusion matrix")

    # summary + CSV
    summary = {
        "backbone": MODEL_CHECKPOINT,
        "train_csv": TRAIN_CSV,
        "n_folds": N_FOLDS,
        "split_seed": SPLIT_SEED,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "per_fold_accuracy_percent": accuracies.tolist(),
        "per_fold_macro_f1_percent": macro_f1s.tolist(),
        "accuracy_mean": float(mean_acc), "accuracy_std": float(std_acc),
        "macro_f1_mean": float(mean_mf1), "macro_f1_std": float(std_mf1),
        "pooled_accuracy_percent": float(pooled_acc),
        "pooled_macro_f1_percent": float(pooled_mf1),
        "per_class": per_class_summary,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "per_class_summary.csv"), "w") as f:
        f.write("class,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std\n")
        for cls, v in per_class_summary.items():
            f.write(
                f"{cls},{v['precision_mean']:.4f},{v['precision_std']:.4f},"
                f"{v['recall_mean']:.4f},{v['recall_std']:.4f},"
                f"{v['f1_mean']:.4f},{v['f1_std']:.4f}\n"
            )

    print(f"\nResults (NO model weights) saved to {os.path.abspath(OUTPUT_DIR)}/")
    print("  - per-fold: logits/preds/labels/testidx/confmat .npy, report .json")
    print("  - pooled  : oof_idx/oof_preds/oof_labels/oof_logits/confmat_pooled .npy")
    print("  - aggregate: summary.json, per_class_summary.csv, fold_indices.json")

accelerator.wait_for_everyone()
