import os
import gc
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed,                      # <-- ADDED: seeds torch / numpy / random / CUDA
)
from accelerate import Accelerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# 1. Initialize Accelerator for Automatic Multi-GPU Detection
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Setting up MULTI-SEED training environment...")

# 2. Configuration & Paths
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"
VAL_CSV = "subash_affectnet_validation_filtered_labels.csv"
TEST_CSV = "subash_affectnet_validation_filtered_labels.csv"   # NOTE: test == val in your current setup

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32          # Per-device batch size
EPOCHS = 5
LEARNING_RATE = 2e-4
OUTPUT_DIR = "./vit_affectnet_multiseed_results"

# ---- Multi-seed config: run the WHOLE training >= 5 times ----
# Each seed is a fully independent run (fresh model init, different data shuffling,
# different dropout). Reported at the end as mean +/- std across runs.
SEEDS = [42, 123, 2024, 7, 2025]           # 5 runs; add more for tighter error bars
# Optional override from the shell, e.g.  SEEDS="1,2,3,4,5"  (comma-separated)
if os.environ.get("SEEDS"):
    SEEDS = [int(s) for s in os.environ["SEEDS"].split(",")]

if accelerator.is_main_process:
    print(f"Will run {len(SEEDS)} independent trainings with seeds: {SEEDS}")

# 3. Load CSV Data ONCE (seed-independent) & Cast Labels to Python Int
if accelerator.is_main_process:
    print("Loading CSV annotations...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].astype(int)
val_df[LABEL_COLUMN] = val_df[LABEL_COLUMN].astype(int)
test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].astype(int)

labels = sorted(train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(label): i for i, label in enumerate(labels)}
id2label = {i: int(label) for i, label in enumerate(labels)}

if accelerator.is_main_process:
    print(f"Detected {num_labels} emotion classes: {labels}")

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

# 4. Initialize Fast Image Processor & Transforms (seed-independent)
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def transform(example_batch):
    images = [Image.open(path).convert("RGB") for path in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs

encoded_datasets = dataset_dict.with_transform(transform)

def compute_metrics(eval_pred):
    predictions, labels_ = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(preds == labels_))
    macro_f1 = float(f1_score(labels_, preds, average="macro"))
    return {"accuracy": accuracy, "macro_f1": macro_f1}

# ---- Collectors for the cross-seed summary ----
seed_acc = []
seed_f1 = []

# 5. MULTI-SEED LOOP -------------------------------------------------------
for run_idx, seed in enumerate(SEEDS):
    if accelerator.is_main_process:
        print("\n==========================================")
        print(f"--- Run {run_idx + 1}/{len(SEEDS)}  |  seed = {seed} ---")
        print("==========================================")

    # Seed EVERYTHING (head init, dropout, dataloader shuffling, CUDA) for this run
    set_seed(seed)

    # Fresh model each run -> different random classification head each seed
    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    run_output_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")

    # 6. Training Arguments (per-seed output dir + seed wired into the Trainer)
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,      # keeps the best-accuracy epoch, not the last
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,               # LOWERED 2->1: only the best ckpt, saves disk over 5 runs
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        seed=seed,                        # <-- ADDED: model/trainer seed
        data_seed=seed,                   # <-- ADDED: dataloader shuffling seed
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    # 8. Train
    if accelerator.is_main_process:
        print(f"Starting training (seed {seed}) across {accelerator.num_processes} V100 GPUs...")
    trainer.train()

    # 9. Evaluate on the independent test set.
    #    predict() is COLLECTIVE under DDP -> call it on ALL ranks, guard only printing/saving.
    test_results = trainer.predict(encoded_datasets["test"])

    if accelerator.is_main_process:
        test_preds = np.argmax(test_results.predictions, axis=1)
        test_true = test_results.label_ids

        acc = float(np.mean(test_preds == test_true))
        mf1 = float(f1_score(test_true, test_preds, average="macro"))
        seed_acc.append(acc)
        seed_f1.append(mf1)

        print(f"\n[seed {seed}] Test Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            test_true, test_preds,
            target_names=[str(id2label[i]) for i in range(num_labels)], digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_true, test_preds))

        # Save this run's best model + processor
        best_dir = os.path.join(run_output_dir, "best")
        trainer.save_model(best_dir)
        image_processor.save_pretrained(best_dir)
        print(f"[seed {seed}] Model saved to {best_dir}")

    # 10. Cleanup between runs (prevents GPU/host-RAM creep over 5 trainings)
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()      # keep both ranks in lock-step between runs

# 11. Cross-seed summary --------------------------------------------------
if accelerator.is_main_process and len(seed_acc) > 0:
    acc_arr = np.array(seed_acc)
    f1_arr = np.array(seed_f1)
    print("\n==========================================")
    print(f"--- Multi-seed summary over {len(SEEDS)} runs ---")
    print("==========================================")
    print(f"Seeds:              {SEEDS}")
    print(f"Per-seed accuracy:  {[round(a, 4) for a in seed_acc]}")
    print(f"Per-seed macro-F1:  {[round(f, 4) for f in seed_f1]}")
    # ddof=1 -> SAMPLE std (the conventional choice for reporting variability across runs)
    print(f"Mean Accuracy:  {acc_arr.mean():.4f} +/- {acc_arr.std(ddof=1):.4f}")
    print(f"Mean Macro-F1:  {f1_arr.mean():.4f} +/- {f1_arr.std(ddof=1):.4f}")
