import os
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# 1. Initialize Accelerator for Automatic Multi-GPU Detection
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Setting up training environment...")

# 2. Configuration & Paths
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"        # full train pool -> split 90/10
TEST_CSV = "subash_affectnet_validation_filtered_labels.csv"    # 500/class balanced set -> FINAL TEST ONLY

# ---- FACE-DOMAIN PRETRAINED BACKBONE (the "different method") ----
MODEL_CHECKPOINT = "trpakov/vit-face-expression"

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32          # Per-device batch size (x4 GPUs => effective 128)
EPOCHS = 10
LEARNING_RATE = 5e-5     # LOWERED so it doesn't erase the pretrained face features
WARMUP_RATIO = 0.1
NUM_WORKERS = 4          # <-- SPEED: parallel image loading (4 CPUs per DDP process)
OUTPUT_DIR = "./vit_affectnet_face_results"
VAL_SPLIT = 0.10
SEED = 42

# 3. Load CSV Data & Cast Labels to Python Int
if accelerator.is_main_process:
    print("Loading CSV annotations...")

full_train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

full_train_df[LABEL_COLUMN] = full_train_df[LABEL_COLUMN].astype(int)
test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].astype(int)

# --- Stratified 90/10 split of the TRAIN pool into train + validation ---
train_df, val_df = train_test_split(
    full_train_df,
    test_size=VAL_SPLIT,
    stratify=full_train_df[LABEL_COLUMN],
    random_state=SEED,
    shuffle=True
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

labels = sorted(full_train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(label): i for i, label in enumerate(labels)}
id2label = {i: int(label) for i, label in enumerate(labels)}

if accelerator.is_main_process:
    print(f"Backbone: {MODEL_CHECKPOINT}")
    print(f"Detected {num_labels} emotion classes: {labels}")
    print(f"Train split: {len(train_df)} | Val split (selection): {len(val_df)} | Test (final): {len(test_df)}")
    print("Test class counts:\n", test_df[LABEL_COLUMN].value_counts().sort_index().to_string())

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

# 4. Image Processor & Transforms  (loaded from the SAME checkpoint -> matched normalization)
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def transform(example_batch):
    images = [Image.open(path).convert("RGB") for path in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs

encoded_datasets = dataset_dict.with_transform(transform)

# 5. Initialize Model from the face-pretrained backbone.
model = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=True,
    dataloader_num_workers=NUM_WORKERS,       # <-- SPEED: parallel image decode
    dataloader_pin_memory=True,               # faster host->GPU copies
    dataloader_persistent_workers=True,       # don't respawn workers every epoch
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(preds == labels))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return {"accuracy": accuracy, "macro_f1": macro_f1}

# 7. Trainer
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
    print("Starting training with face-domain pretrained backbone...")
trainer.train()

# 9. Final evaluation on the 500/class AffectNet set.
# IMPORTANT: predict() is a COLLECTIVE op under DDP -- every rank must call it,
# otherwise the job deadlocks. Only the printing/saving is guarded to rank 0.
test_results = trainer.predict(encoded_datasets["test"])

if accelerator.is_main_process:
    print("Evaluating on the 500/class AffectNet test set...")
    test_preds = np.argmax(test_results.predictions, axis=1)
    test_true = test_results.label_ids

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds,
                                target_names=[str(id2label[i]) for i in range(num_labels)],
                                digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true, test_preds))
    print(f"\nOverall accuracy: {np.mean(test_preds == test_true):.4f}")
    print(f"Macro-F1: {f1_score(test_true, test_preds, average='macro'):.4f}")

    trainer.save_model(os.path.join(OUTPUT_DIR, "best_vit_affectnet_face"))
    image_processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_vit_affectnet_face"))
    print(f"Model saved to {OUTPUT_DIR}/best_vit_affectnet_face")

# Make sure all ranks finish together before exit.
accelerator.wait_for_everyone()
