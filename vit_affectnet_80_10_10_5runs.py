"""
ViT fine-tuning on AffectNet with a stratified
80 % train / 10 % validation / 10 % independent test split,
repeated over 5 independent training runs (different random seeds).

Replaces the 10-fold stratified cross-validation version.

Design decisions that matter for the manuscript:
  * The split is STRATIFIED at both stages, so every bin holds the same
    class proportions as the full training pool.
  * The split is held FIXED across the 5 runs (SPLIT_SEED), so all runs
    are scored on one identical independent test set. That keeps the
    numbers comparable to the single-split result already in the paper,
    and it is what makes paired significance tests (McNemar) possible
    later. Set FIXED_SPLIT = False to resample the split each run if you
    would rather have split variability folded into the SD.
  * Model selection uses the VALIDATION bin only. The test bin is
    touched exactly once per run, at the end.
  * Per-run test logits / predictions / labels are written to .npy so
    bootstrap CIs and McNemar tests never require retraining.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import torch

from PIL import Image
from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed,
)
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------------------------------------------
# 1. Accelerator
# ----------------------------------------------------------------------
accelerator = Accelerator()
IS_MAIN = accelerator.is_main_process


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


log(f"Detected {accelerator.num_processes} GPU(s).")

# ----------------------------------------------------------------------
# 2. Configuration
# ----------------------------------------------------------------------
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5          # matches the value reported in the manuscript
WEIGHT_DECAY = 0.01

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
RUN_SEEDS = [42, 1337, 2024, 7, 31415]   # 5 independent training runs
SPLIT_SEED = 42                          # used when FIXED_SPLIT is True
FIXED_SPLIT = True

OUTPUT_DIR = "./vit_affectnet_80_10_10_5runs"
KEEP_CHECKPOINTS = False

if IS_MAIN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 3. Load annotations and label maps
# ----------------------------------------------------------------------
log(f"Loading training annotations from {TRAIN_CSV}...")
full_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
full_df[LABEL_COLUMN] = full_df[LABEL_COLUMN].astype(int)

labels = sorted(full_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(l): i for i, l in enumerate(labels)}
id2label = {i: int(l) for i, l in enumerate(labels)}

log(f"Detected {num_labels} emotion classes: {labels}")
log(f"Total pool: {len(full_df)} samples")

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)


def transform(example_batch):
    images = [Image.open(p).convert("RGB") for p in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs


def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": float(np.mean(preds == label_ids))}


# ----------------------------------------------------------------------
# 4. Stratified 80 / 10 / 10 split
# ----------------------------------------------------------------------
def make_stratified_split(df, seed):
    """Two-stage stratified split preserving class ratios in all three bins."""
    y = df[LABEL_COLUMN]

    # stage 1: 80 % train  vs  20 % holdout
    train_df, holdout_df = train_test_split(
        df,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=y,
        random_state=seed,
        shuffle=True,
    )
    # stage 2: split the 20 % holdout evenly into validation and test
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=holdout_df[LABEL_COLUMN],
        random_state=seed,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def report_class_ratios(train_df, val_df, test_df):
    """Print the class distribution of each bin so stratification is verifiable."""
    rows = []
    for name, d in [("full", full_df), ("train", train_df),
                    ("val", val_df), ("test", test_df)]:
        counts = d[LABEL_COLUMN].value_counts().sort_index()
        pct = (counts / len(d) * 100).round(2)
        rows.append(pd.DataFrame({f"{name}_n": counts, f"{name}_%": pct}))
    table = pd.concat(rows, axis=1)
    log("\nClass distribution per bin (percentages should match across bins):")
    log(table.to_string())
    if IS_MAIN:
        table.to_csv(os.path.join(OUTPUT_DIR, "split_class_distribution.csv"))
    return table


# ----------------------------------------------------------------------
# 5. One training run
# ----------------------------------------------------------------------
def run_once(run_idx, seed):
    log("\n" + "=" * 70)
    log(f"RUN {run_idx + 1}/{len(RUN_SEEDS)}  (seed {seed})")
    log("=" * 70)

    set_seed(seed)

    split_seed = SPLIT_SEED if FIXED_SPLIT else seed
    train_df, val_df, test_df = make_stratified_split(full_df, split_seed)
    log(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    if run_idx == 0:
        report_class_ratios(train_df, val_df, test_df)
        if IS_MAIN:
            # release the exact split with the paper for reproducibility
            train_df.to_csv(os.path.join(OUTPUT_DIR, "split_train.csv"), index=False)
            val_df.to_csv(os.path.join(OUTPUT_DIR, "split_val.csv"), index=False)
            test_df.to_csv(os.path.join(OUTPUT_DIR, "split_test.csv"), index=False)

    train_ds = Dataset.from_pandas(train_df).with_transform(transform)
    val_ds = Dataset.from_pandas(val_df).with_transform(transform)
    test_ds = Dataset.from_pandas(test_df).with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    run_dir = os.path.join(OUTPUT_DIR, f"run_seed{seed}")

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,   # must be False, see notes
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,          # selection on validation only
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # predict() must run on EVERY process, otherwise distributed runs deadlock
    log("Evaluating on the held-out independent test bin...")
    test_out = trainer.predict(test_ds)

    logits = test_out.predictions
    preds = np.argmax(logits, axis=1)
    true = test_out.label_ids
    acc = accuracy_score(true, preds)

    result = None
    if IS_MAIN:
        target_names = [str(id2label[i]) for i in range(num_labels)]
        report = classification_report(
            true, preds, target_names=target_names,
            output_dict=True, zero_division=0
        )
        cm = confusion_matrix(true, preds)

        print(f"\nRun seed {seed} — independent test accuracy: {acc * 100:.2f} %")
        print(classification_report(
            true, preds, target_names=target_names, zero_division=0
        ))

        np.save(os.path.join(OUTPUT_DIR, f"logits_seed{seed}.npy"), logits)
        np.save(os.path.join(OUTPUT_DIR, f"preds_seed{seed}.npy"), preds)
        np.save(os.path.join(OUTPUT_DIR, f"labels_seed{seed}.npy"), true)
        np.save(os.path.join(OUTPUT_DIR, f"confmat_seed{seed}.npy"), cm)
        with open(os.path.join(OUTPUT_DIR, f"report_seed{seed}.json"), "w") as f:
            json.dump(report, f, indent=2)

        trainer.save_model(os.path.join(run_dir, "best_model"))
        image_processor.save_pretrained(os.path.join(run_dir, "best_model"))

        result = {"seed": seed, "accuracy": acc, "report": report,
                  "confusion_matrix": cm}

    accelerator.wait_for_everyone()

    del trainer, model, test_out
    gc.collect()
    torch.cuda.empty_cache()

    if IS_MAIN and not KEEP_CHECKPOINTS:
        import shutil
        for entry in os.listdir(run_dir):
            if entry.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(run_dir, entry), ignore_errors=True)

    return result


# ----------------------------------------------------------------------
# 6. Five runs, then aggregate
# ----------------------------------------------------------------------
all_runs = [run_once(i, s) for i, s in enumerate(RUN_SEEDS)]

if IS_MAIN:
    all_runs = [r for r in all_runs if r is not None]
    accs = np.array([r["accuracy"] for r in all_runs]) * 100
    mean_acc, std_acc = accs.mean(), accs.std(ddof=1)

    print("\n" + "=" * 70)
    print(f"SUMMARY OVER {len(all_runs)} RUNS — stratified 80/10/10 split")
    print("=" * 70)
    for r, a in zip(all_runs, accs):
        print(f"  seed {r['seed']:>6}: {a:.2f} %")
    print(f"\nAccuracy: {mean_acc:.2f} +/- {std_acc:.2f} %  (mean +/- SD, n={len(accs)})")
    print(f"Range: {accs.min():.2f} – {accs.max():.2f} %")

    print("\nPer-class metrics (mean +/- SD across runs):")
    per_class = {}
    for i in range(num_labels):
        cls = str(id2label[i])
        p = np.array([r["report"][cls]["precision"] for r in all_runs])
        rc = np.array([r["report"][cls]["recall"] for r in all_runs])
        f1 = np.array([r["report"][cls]["f1-score"] for r in all_runs])
        per_class[cls] = {
            "precision_mean": float(p.mean()), "precision_std": float(p.std(ddof=1)),
            "recall_mean": float(rc.mean()), "recall_std": float(rc.std(ddof=1)),
            "f1_mean": float(f1.mean()), "f1_std": float(f1.std(ddof=1)),
        }
        print(f"  class {cls:<4} P {p.mean():.3f}+/-{p.std(ddof=1):.3f}   "
              f"R {rc.mean():.3f}+/-{rc.std(ddof=1):.3f}   "
              f"F1 {f1.mean():.3f}+/-{f1.std(ddof=1):.3f}")

    mean_cm = np.mean([r["confusion_matrix"] for r in all_runs], axis=0)
    np.save(os.path.join(OUTPUT_DIR, "confmat_mean.npy"), mean_cm)

    # free bonus: logit-averaged ensemble of the 5 runs, same test set
    ens_summary = None
    if FIXED_SPLIT:
        stacked = np.mean(
            [np.load(os.path.join(OUTPUT_DIR, f"logits_seed{s}.npy"))
             for s in [r["seed"] for r in all_runs]], axis=0
        )
        ens_preds = np.argmax(stacked, axis=1)
        true = np.load(os.path.join(OUTPUT_DIR, f"labels_seed{all_runs[0]['seed']}.npy"))
        ens_acc = accuracy_score(true, ens_preds) * 100
        np.save(os.path.join(OUTPUT_DIR, "preds_ensemble.npy"), ens_preds)
        ens_summary = float(ens_acc)
        print(f"\n5-run logit-averaged ensemble accuracy: {ens_acc:.2f} %")

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "protocol": "stratified 80/10/10, repeated training runs",
            "fixed_split": FIXED_SPLIT,
            "split_seed": SPLIT_SEED,
            "run_seeds": [r["seed"] for r in all_runs],
            "accuracies_percent": accs.tolist(),
            "accuracy_mean": float(mean_acc),
            "accuracy_std": float(std_acc),
            "ensemble_accuracy": ens_summary,
            "per_class": per_class,
        }, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "per_class_summary.csv"), "w") as f:
        f.write("class,precision_mean,precision_std,recall_mean,"
                "recall_std,f1_mean,f1_std\n")
        for cls, v in per_class.items():
            f.write(f"{cls},{v['precision_mean']:.4f},{v['precision_std']:.4f},"
                    f"{v['recall_mean']:.4f},{v['recall_std']:.4f},"
                    f"{v['f1_mean']:.4f},{v['f1_std']:.4f}\n")

    print(f"\nAll artifacts written to {OUTPUT_DIR}/")
