"""
STRATIFIED 10-FOLD CROSS-VALIDATION, STRONG-recipe ViT fine-tuning for RAF-DB
facial emotion recognition.

This is the cross-validation counterpart of the multi-seed script. Instead of a
reseeded 90/10 hold-out repeated over seeds, it runs proper stratified K-fold CV
*on the training set only*. The independent test/ directory is NOT touched here.

WHAT IT DOES
  - StratifiedKFold(n_splits=10) partitions the TRAINING set so every sample is
    held out exactly once. The held-out fold is that fold's evaluation set.
  - Best-checkpoint selection (load_best_model_at_end on MACRO-F1) is kept, but to
    avoid selecting the checkpoint on the very fold we then report on, a small
    stratified INNER validation slice is carved out of the 9 training folds and used
    for selection only. The held-out test fold is used strictly for final scoring.
        * Set INNER_VAL_FRACTION = 0.0 to fall back to the simpler (mildly optimistic)
          "select the checkpoint on the held-out fold" behaviour.
  - Everything else from the strong recipe is preserved:
        face-domain pretrained backbone, strong augmentation, class-weighted +
        label-smoothed cross-entropy, cosine schedule with warmup.

WHAT IT REPORTS
  (1) Per-fold accuracy / macro-F1, then mean +/- SD across the 10 folds.
  (2) POOLED out-of-fold metrics: because folds are disjoint and cover the whole
      training set, concatenating each fold's predictions gives one prediction per
      training sample -> a single overall accuracy / macro-F1 / confusion matrix
      over the entire training set.

MODEL WEIGHTS ARE NOT KEPT.
  - Per-fold checkpoints go to a SCRATCH directory ($SLURM_TMPDIR / $TMPDIR / /tmp)
    and are deleted after every fold. Nothing under RESULTS_DIR is a model file.
  - What IS written to RESULTS_DIR: per-fold logits/preds/labels/confmat (.npy) and
    reports (.json), pooled out-of-fold arrays, fold indices (for reproducibility),
    and the aggregate summary/CSV.

All per-fold and aggregate results (including confusion matrices) are ALSO printed
to stdout, so they land in your SLURM --output=...out file for quick inspection.

COMPUTE NOTE: this trains a fresh model 10 times (once per fold). Total cost is
~10 x the per-fold cost. With EPOCHS=20 that is 200 epochs of training end to end;
lower EPOCHS if you just want a quick CV estimate.

Intended launch (single GPU, single process):
    accelerate launch --num_processes=1 --mixed_precision=fp16 RAF_DB_vit_cv10fold.py
or simply:
    python RAF_DB_vit_cv10fold.py
Do NOT launch with --num_processes=2: the per-fold np.save() calls are unguarded and
would double-write under DDP.
"""

import os
import gc
import json
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,     # generic head -> works for ViT / Swin / face-ViT
    Trainer,
    TrainingArguments,
    set_seed,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandAugment,
    RandomErasing,
    ToTensor,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------
# 0. Experiment configuration
# ----------------------------------------------------------------------
N_FOLDS = 10                              # stratified K-fold on the TRAIN set
SPLIT_SEED = 42                           # controls the fold partition (reproducible)
INNER_VAL_FRACTION = 0.10                 # carved from the 9 training folds for best-ckpt
                                          # selection. Set 0.0 to select on the test fold.
RESULTS_DIR = "./rafdb_vit_strong_cv10fold_results"

# Where the TRANSIENT training checkpoints live. Deleted after each fold; never in the
# project dir. Prefer node-local scratch (fast + auto-cleaned by the scheduler).
SCRATCH_ROOT = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or tempfile.gettempdir()
)

data_dir = "/project/pakhrin/subash/DATASET"   # RAF-DB imagefolder: train/<class>/*, test/<class>/*
train_dir = os.path.join(data_dir, "train")
# NOTE: test_dir is intentionally NOT used -- CV runs on the training set only.

# ---- Backbone: face-domain pretrained ViT is a much better init than ImageNet ViT ----
model_checkpoint = "trpakov/vit-face-expression"
# One-line alternatives:
#   "google/vit-base-patch16-224-in21k"        (your original baseline)
#   "microsoft/swin-base-patch4-window7-224"   (strong ImageNet backbone)

# ---- Strong-recipe hyper-parameters ----
EPOCHS = 20
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True          # RAF-DB is imbalanced (happy dominates); helps rare classes
TRAIN_BS = 32                     # per-device; drop to 16 on OOM
EVAL_BS = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("RUN CONFIG (stratified K-fold CV on TRAIN only)")
print("=" * 70)
print(f"Backbone             : {model_checkpoint}")
print(f"Folds                : {N_FOLDS}")
print(f"Split seed           : {SPLIT_SEED}")
print(f"Inner val fraction   : {INNER_VAL_FRACTION}  (best-ckpt selection; test fold untouched)")
print(f"Epochs / fold        : {EPOCHS}")
print(f"Results dir          : {os.path.abspath(RESULTS_DIR)}")
print(f"Scratch (checkpoints): {SCRATCH_ROOT}   (deleted after each fold)")
print("Model weights kept   : NO  (best-ckpt selection uses scratch, then removed)")
print("Independent test set : NOT used (CV on training data only)")

# ----------------------------------------------------------------------
# 1. Load the TRAINING data once
# ----------------------------------------------------------------------
print("\nLoading TRAINING dataset from directory...")
dataset = load_dataset(
    "imagefolder",
    data_files={"train": os.path.join(train_dir, "**")},
)
full_train = dataset["train"]

labels = full_train.features["label"].names
num_labels = len(labels)
id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}
print(f"Found {num_labels} classes: {labels}")
print(f"Training samples     : {len(full_train)}")

# Full label array used to build the stratified folds.
all_labels = np.array(full_train["label"])

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
image_mean = image_processor.image_mean
image_std = image_processor.image_std
size = image_processor.size["height"]

# --- STRONG training augmentation (face-safe strengths) ---
train_transforms = Compose(
    [
        RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=7),   # lower to 5 if it hurts subtle classes
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
        RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ]
)


def preprocess_train(example_batch):
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_eval(example_batch):
    images = [image.convert("RGB") for image in example_batch["image"]]
    inputs = image_processor(images, return_tensors="pt")
    example_batch["pixel_values"] = inputs.pixel_values
    return example_batch


def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "macro_f1": f1_score(label_ids, preds, average="macro"),
    }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


def print_confusion_matrix(cm, class_names, title="Confusion matrix (rows=true, cols=pred)"):
    """Pretty-print a confusion matrix to stdout so it shows up in the .out file."""
    print(f"\n{title}")
    col_w = max(6, max(len(c) for c in class_names) + 1)
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        line = f"{class_names[i]:>{col_w}}" + "".join(f"{int(v):>{col_w}}" for v in row)
        print(line)
    # Per-row recall for a quick read of which classes get confused
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = cm.diagonal() / cm.sum(axis=1)
    print("Per-class recall: " + ", ".join(
        f"{class_names[i]}={r:.3f}" for i, r in enumerate(recalls)
    ))


# --- Custom Trainer: class-weighted + label-smoothed cross-entropy ---
class WeightedSmoothTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights
        self._ls = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._cw.to(logits.device) if self._cw is not None else None
        loss = F.cross_entropy(logits, labels, weight=w, label_smoothing=self._ls)
        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------------
# 2. Single-fold run
# ----------------------------------------------------------------------
def run_one_fold(fold_idx, train_idx, test_idx):
    """
    train_idx : indices (into full_train) of the 9 training folds
    test_idx  : indices of the held-out fold -> used ONLY for final scoring
    Returns per-fold results plus the aligned (test_idx, preds, labels, logits) so the
    caller can assemble pooled out-of-fold metrics.
    """
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print("=" * 70)

    fold_seed = SPLIT_SEED + fold_idx      # differ training stochasticity across folds
    set_seed(fold_seed)

    # Transient checkpoint dir on scratch -- NOT in the project/results dir.
    ckpt_dir = os.path.join(SCRATCH_ROOT, f"rafdb_ckpt_fold{fold_idx}")

    # ---- carve an INNER stratified validation slice for best-ckpt selection ----
    # The held-out test fold is never seen during selection.
    if INNER_VAL_FRACTION and INNER_VAL_FRACTION > 0.0:
        tr_idx, val_idx = train_test_split(
            train_idx,
            test_size=INNER_VAL_FRACTION,
            stratify=all_labels[train_idx],
            random_state=fold_seed,
            shuffle=True,
        )
    else:
        # Fall back: select the checkpoint on the held-out fold (mildly optimistic).
        tr_idx, val_idx = train_idx, test_idx

    tr_idx = np.asarray(tr_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    print(f"  train / inner-val / test sizes : "
          f"{len(tr_idx)} / {len(val_idx)} / {len(test_idx)}")

    # Class weights from THIS fold's training slice (before transform is attached)
    train_label_arr = all_labels[tr_idx]
    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_label_arr)
    class_weights_t = torch.tensor(cw, dtype=torch.float) if USE_CLASS_WEIGHTS else None

    train_dataset = full_train.select(tr_idx.tolist()).with_transform(preprocess_train)
    val_dataset = full_train.select(val_idx.tolist()).with_transform(preprocess_eval)
    test_dataset = full_train.select(test_idx.tolist()).with_transform(preprocess_eval)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=ckpt_dir,                # <-- scratch, cleaned up below
        eval_strategy="epoch",
        save_strategy="epoch",              # needed for best-ckpt selection; goes to scratch
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,        # keep: selects best macro-F1 checkpoint
        metric_for_best_model="macro_f1",   # right target under class imbalance
        greater_is_better=True,
        save_total_limit=1,                 # at most one checkpoint on scratch at a time
        logging_steps=20,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=fold_seed,
        data_seed=fold_seed,
        report_to="none",
    )

    trainer = WeightedSmoothTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,          # selection on inner val, NOT the test fold
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights_t,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer.train()

    # ---- final evaluation on THIS fold's held-out slice ----
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_classes = np.argmax(logits, axis=1)
    true_labels = predictions_output.label_ids

    fold_accuracy = accuracy_score(true_labels, predicted_classes)
    fold_macro_f1 = f1_score(true_labels, predicted_classes, average="macro")
    conf_matrix = confusion_matrix(true_labels, predicted_classes, labels=np.arange(num_labels))
    report = classification_report(
        true_labels, predicted_classes,
        labels=np.arange(num_labels), target_names=labels,
        output_dict=True, zero_division=0,
    )

    # ---- print results to stdout (-> lands in the .out file) ----
    print(f"\nFold {fold_idx + 1} accuracy: {fold_accuracy:.4f} | macro-F1: {fold_macro_f1:.4f}")
    print(classification_report(
        true_labels, predicted_classes,
        labels=np.arange(num_labels), target_names=labels, zero_division=0,
    ))
    print_confusion_matrix(conf_matrix, labels, title=f"[Fold {fold_idx + 1}] confusion matrix")

    # ---- save results (NOT models) for later significance testing ----
    np.save(os.path.join(RESULTS_DIR, f"logits_fold{fold_idx}.npy"), logits)
    np.save(os.path.join(RESULTS_DIR, f"preds_fold{fold_idx}.npy"), predicted_classes)
    np.save(os.path.join(RESULTS_DIR, f"labels_fold{fold_idx}.npy"), true_labels)
    np.save(os.path.join(RESULTS_DIR, f"testidx_fold{fold_idx}.npy"), test_idx)
    np.save(os.path.join(RESULTS_DIR, f"confmat_fold{fold_idx}.npy"), conf_matrix)
    with open(os.path.join(RESULTS_DIR, f"report_fold{fold_idx}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- free GPU + delete the scratch checkpoint (no model weights kept) ----
    del trainer, model, predictions_output
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {
        "fold": fold_idx,
        "accuracy": fold_accuracy,
        "macro_f1": fold_macro_f1,
        "report": report,
        "confusion_matrix": conf_matrix,
        "test_idx": test_idx,
        "preds": predicted_classes,
        "labels": true_labels,
        "logits": logits,
    }


# ----------------------------------------------------------------------
# 3. Run all folds and aggregate
# ----------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
fold_splits = list(skf.split(np.zeros(len(all_labels)), all_labels))

# Save the fold partition up front for full reproducibility / auditability.
with open(os.path.join(RESULTS_DIR, "fold_indices.json"), "w") as f:
    json.dump(
        {f"fold{k}": {"test_idx": test_idx.tolist()}
         for k, (_, test_idx) in enumerate(fold_splits)},
        f,
    )

all_runs = [run_one_fold(k, tr, te) for k, (tr, te) in enumerate(fold_splits)]

# ---- (1) per-fold metrics: mean +/- SD across folds ----
accuracies = np.array([r["accuracy"] for r in all_runs]) * 100
macro_f1s = np.array([r["macro_f1"] for r in all_runs]) * 100
mean_acc, std_acc = accuracies.mean(), accuracies.std(ddof=1)   # sample SD -> report in paper
mean_mf1, std_mf1 = macro_f1s.mean(), macro_f1s.std(ddof=1)

print("\n" + "=" * 70)
print("AGGREGATE RESULTS OVER FOLDS (per-fold mean +/- SD)")
print("=" * 70)
for r, a, m in zip(all_runs, accuracies, macro_f1s):
    print(f"  fold {r['fold'] + 1:>2}:  acc {a:.2f} %   macro-F1 {m:.2f} %")
print(f"\nAccuracy:  {mean_acc:.2f} +/- {std_acc:.2f} %  (mean +/- SD, k={N_FOLDS})")
print(f"Macro-F1:  {mean_mf1:.2f} +/- {std_mf1:.2f} %")
print(f"Acc min/max: {accuracies.min():.2f} / {accuracies.max():.2f} %")

# Per-class F1 mean +/- SD across folds -- the table reviewers scrutinize
print("\nPer-class F1 (mean +/- SD across folds):")
per_class_summary = {}
for cls in labels:
    f1s = np.array([r["report"][cls]["f1-score"] for r in all_runs])
    rec = np.array([r["report"][cls]["recall"] for r in all_runs])
    prec = np.array([r["report"][cls]["precision"] for r in all_runs])
    per_class_summary[cls] = {
        "precision_mean": float(prec.mean()), "precision_std": float(prec.std(ddof=1)),
        "recall_mean": float(rec.mean()), "recall_std": float(rec.std(ddof=1)),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)),
    }
    print(f"  {cls:<12} {f1s.mean():.3f} +/- {f1s.std(ddof=1):.3f}")

# ---- (2) POOLED out-of-fold metrics over the WHOLE training set ----
# Folds are disjoint and cover every sample once, so concatenating gives one
# prediction per training sample -> a single overall score + confusion matrix.
oof_idx = np.concatenate([r["test_idx"] for r in all_runs])
oof_preds = np.concatenate([r["preds"] for r in all_runs])
oof_labels = np.concatenate([r["labels"] for r in all_runs])
oof_logits = np.concatenate([r["logits"] for r in all_runs], axis=0)

# Re-order everything back to the original training-set index order (0..N-1).
order = np.argsort(oof_idx)
oof_idx = oof_idx[order]
oof_preds = oof_preds[order]
oof_labels = oof_labels[order]
oof_logits = oof_logits[order]

pooled_acc = accuracy_score(oof_labels, oof_preds) * 100
pooled_mf1 = f1_score(oof_labels, oof_preds, average="macro") * 100
pooled_conf = confusion_matrix(oof_labels, oof_preds, labels=np.arange(num_labels))

np.save(os.path.join(RESULTS_DIR, "oof_idx.npy"), oof_idx)
np.save(os.path.join(RESULTS_DIR, "oof_preds.npy"), oof_preds)
np.save(os.path.join(RESULTS_DIR, "oof_labels.npy"), oof_labels)
np.save(os.path.join(RESULTS_DIR, "oof_logits.npy"), oof_logits)
np.save(os.path.join(RESULTS_DIR, "confmat_pooled.npy"), pooled_conf)

print("\n" + "=" * 70)
print("POOLED OUT-OF-FOLD RESULTS (every training sample predicted once)")
print("=" * 70)
print(f"Pooled accuracy: {pooled_acc:.2f} %   |   pooled macro-F1: {pooled_mf1:.2f} %")
print(classification_report(
    oof_labels, oof_preds,
    labels=np.arange(num_labels), target_names=labels, zero_division=0,
))
print_confusion_matrix(pooled_conf, labels, title="Pooled out-of-fold confusion matrix")

# ---- write summary + CSV ----
summary = {
    "backbone": model_checkpoint,
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
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# CSV ready to paste into the manuscript
with open(os.path.join(RESULTS_DIR, "per_class_summary.csv"), "w") as f:
    f.write("class,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std\n")
    for cls, v in per_class_summary.items():
        f.write(
            f"{cls},{v['precision_mean']:.4f},{v['precision_std']:.4f},"
            f"{v['recall_mean']:.4f},{v['recall_std']:.4f},"
            f"{v['f1_mean']:.4f},{v['f1_std']:.4f}\n"
        )

print(f"\nResults (no model weights) saved to {os.path.abspath(RESULTS_DIR)}/")
print("  - per-fold: logits/preds/labels/testidx/confmat .npy, report .json")
print("  - pooled  : oof_idx/oof_preds/oof_labels/oof_logits/confmat_pooled .npy")
print("  - aggregate: summary.json, per_class_summary.csv, fold_indices.json")
