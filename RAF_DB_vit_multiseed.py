"""
Multi-seed, STRONG-recipe ViT fine-tuning for RAF-DB facial emotion recognition.

This merges two things:
  (A) the multi-seed statistical rigor Reviewer 1 (c4) and Reviewer 3 (c4) asked for
      - N seeds, reseeded 90/10 split each run, mean +/- SD,
      - per-seed logits/preds/labels/confmat saved to .npy for later bootstrap CIs
        and McNemar tests WITHOUT retraining;
  (B) the stronger training recipe aimed at closing the gap to the SOTA table
      (POSTER++ 92.21, DDAMFN 91.35, ...) from the plain-ViT ~83%:
      - face-domain pretrained backbone,
      - strong augmentation (RandAugment + flip + resized-crop + random erasing),
      - class-weighted + label-smoothed cross-entropy,
      - best-checkpoint selection on MACRO-F1 (not final-epoch, not plain accuracy),
      - longer cosine schedule with warmup.

MODEL WEIGHTS ARE NOT KEPT.
  - Best-checkpoint selection (load_best_model_at_end on macro-F1) is preserved, so
    the Trainer still writes a checkpoint each epoch -- but those go to a SCRATCH
    directory ($SLURM_TMPDIR / $TMPDIR / /tmp), NOT the project dir, and are deleted
    after every seed. Nothing under RESULTS_DIR is a model file.
  - What IS written to RESULTS_DIR: per-seed logits/preds/labels/confmat (.npy),
    per-seed reports (.json), and the aggregate summary/CSV. These are results, not
    weights, and are needed for later significance testing without retraining.

All per-seed and aggregate results (including confusion matrices) are ALSO printed to
stdout, so they land in your SLURM --output=...out file for quick inspection.

Intended launch (single GPU, single process):
    accelerate launch --num_processes=1 --mixed_precision=fp16 RAF_DB_vit_multiseed.py
or simply:
    python RAF_DB_vit_multiseed.py
Do NOT launch with --num_processes=2: the per-seed np.save() calls are unguarded and
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
SEEDS = [42, 1337, 2024, 7, 31415]        # 5 independent runs
VAL_FRACTION = 0.10                        # carved out of train, per seed (best-ckpt selection)
RESULTS_DIR = "./rafdb_vit_strong_multiseed_results"

# Where the TRANSIENT training checkpoints live. Deleted after each seed; never in the
# project dir. Prefer node-local scratch (fast + auto-cleaned by the scheduler).
SCRATCH_ROOT = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or tempfile.gettempdir()
)

data_dir = "/project/pakhrin/subash/DATASET"   # RAF-DB imagefolder: train/<class>/*, test/<class>/*
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

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
print("RUN CONFIG")
print("=" * 70)
print(f"Backbone            : {model_checkpoint}")
print(f"Seeds               : {SEEDS}")
print(f"Epochs / seed       : {EPOCHS}")
print(f"Results dir         : {os.path.abspath(RESULTS_DIR)}")
print(f"Scratch (checkpoints): {SCRATCH_ROOT}   (deleted after each seed)")
print("Model weights kept  : NO  (best-ckpt selection uses scratch, then removed)")

# ----------------------------------------------------------------------
# 1. Load data once
# ----------------------------------------------------------------------
print("\nLoading dataset from directories...")
dataset = load_dataset(
    "imagefolder",
    data_files={
        "train": os.path.join(train_dir, "**"),
        "test": os.path.join(test_dir, "**"),
    },
)

labels = dataset["train"].features["label"].names
num_labels = len(labels)
id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}
print(f"Found {num_labels} classes: {labels}")

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
# 2. Single-seed run
# ----------------------------------------------------------------------
def run_one_seed(seed):
    print("\n" + "=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)

    set_seed(seed)

    # Transient checkpoint dir on scratch -- NOT in the project/results dir.
    ckpt_dir = os.path.join(SCRATCH_ROOT, f"rafdb_ckpt_seed{seed}")

    # Stratified 90/10 split reseeded each run -> split variability is part of the SD.
    # The test set is never touched (model selection is on the val slice only).
    split = dataset["train"].train_test_split(
        test_size=VAL_FRACTION,
        stratify_by_column="label",
        seed=seed,
    )

    # Class weights from THIS seed's train slice (before transform is attached)
    train_label_arr = np.array(split["train"]["label"])
    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_label_arr)
    class_weights_t = torch.tensor(cw, dtype=torch.float) if USE_CLASS_WEIGHTS else None

    train_dataset = split["train"].with_transform(preprocess_train)
    val_dataset = split["test"].with_transform(preprocess_eval)
    test_dataset = dataset["test"].with_transform(preprocess_eval)

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
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    trainer = WeightedSmoothTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,          # selection on validation, NOT test
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights_t,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer.train()

    # ---- final evaluation on the untouched independent test set ----
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_classes = np.argmax(logits, axis=1)
    true_labels = predictions_output.label_ids

    test_accuracy = accuracy_score(true_labels, predicted_classes)
    test_macro_f1 = f1_score(true_labels, predicted_classes, average="macro")
    conf_matrix = confusion_matrix(true_labels, predicted_classes)
    report = classification_report(
        true_labels, predicted_classes,
        target_names=labels, output_dict=True, zero_division=0,
    )

    # ---- print results to stdout (-> lands in the .out file) ----
    print(f"\nSeed {seed} test accuracy: {test_accuracy:.4f} | macro-F1: {test_macro_f1:.4f}")
    print(classification_report(
        true_labels, predicted_classes, target_names=labels, zero_division=0
    ))
    print_confusion_matrix(conf_matrix, labels, title=f"[Seed {seed}] confusion matrix")

    # ---- save results (NOT models) for later significance testing ----
    np.save(os.path.join(RESULTS_DIR, f"logits_seed{seed}.npy"), logits)
    np.save(os.path.join(RESULTS_DIR, f"preds_seed{seed}.npy"), predicted_classes)
    np.save(os.path.join(RESULTS_DIR, f"labels_seed{seed}.npy"), true_labels)
    np.save(os.path.join(RESULTS_DIR, f"confmat_seed{seed}.npy"), conf_matrix)
    with open(os.path.join(RESULTS_DIR, f"report_seed{seed}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- free GPU + delete the scratch checkpoint (no model weights kept) ----
    del trainer, model, predictions_output
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {"seed": seed, "accuracy": test_accuracy, "macro_f1": test_macro_f1,
            "report": report, "confusion_matrix": conf_matrix}


# ----------------------------------------------------------------------
# 3. Run all seeds and aggregate
# ----------------------------------------------------------------------
all_runs = [run_one_seed(s) for s in SEEDS]

accuracies = np.array([r["accuracy"] for r in all_runs]) * 100
macro_f1s = np.array([r["macro_f1"] for r in all_runs]) * 100
mean_acc, std_acc = accuracies.mean(), accuracies.std(ddof=1)   # sample SD -> report in paper
mean_mf1, std_mf1 = macro_f1s.mean(), macro_f1s.std(ddof=1)

print("\n" + "=" * 70)
print("AGGREGATE RESULTS OVER SEEDS")
print("=" * 70)
for r, a, m in zip(all_runs, accuracies, macro_f1s):
    print(f"  seed {r['seed']:>6}:  acc {a:.2f} %   macro-F1 {m:.2f} %")
print(f"\nAccuracy:  {mean_acc:.2f} +/- {std_acc:.2f} %  (mean +/- SD, n={len(SEEDS)})")
print(f"Macro-F1:  {mean_mf1:.2f} +/- {std_mf1:.2f} %")
print(f"Acc min/max: {accuracies.min():.2f} / {accuracies.max():.2f} %")

# Per-class F1 mean +/- SD -- the table reviewers scrutinize
print("\nPer-class F1 (mean +/- SD):")
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

# Mean confusion matrix across seeds -- saved AND printed
mean_conf = np.mean([r["confusion_matrix"] for r in all_runs], axis=0)
np.save(os.path.join(RESULTS_DIR, "confmat_mean.npy"), mean_conf)
print_confusion_matrix(
    np.rint(mean_conf).astype(int), labels,
    title="Mean confusion matrix across seeds (rounded)",
)

summary = {
    "backbone": model_checkpoint,
    "seeds": SEEDS,
    "accuracies_percent": accuracies.tolist(),
    "macro_f1_percent": macro_f1s.tolist(),
    "accuracy_mean": float(mean_acc), "accuracy_std": float(std_acc),
    "macro_f1_mean": float(mean_mf1), "macro_f1_std": float(std_mf1),
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
print("  - per-seed: logits/preds/labels/confmat .npy, report .json")
print("  - aggregate: summary.json, per_class_summary.csv, confmat_mean.npy")