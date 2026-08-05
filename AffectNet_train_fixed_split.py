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
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Initialize Accelerator for Automatic Multi-GPU Detection
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Setting up training environment...")

# 2. Configuration & Paths
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"        # full train pool -> split 90/10
TEST_CSV = "subash_affectnet_validation_filtered_labels.csv"    # 500/class balanced set -> FINAL TEST ONLY

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32  # Per-device batch size
EPOCHS = 5
LEARNING_RATE = 2e-4
OUTPUT_DIR = "./vit_affectnet_results"
VAL_SPLIT = 0.10   # 10% of TRAIN_CSV held out for checkpoint selection
SEED = 42

# 3. Load CSV Data using Pandas & Cast Labels to Python Int (prevents JSON serialization errors)
if accelerator.is_main_process:
    print("Loading CSV annotations...")

full_train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

full_train_df[LABEL_COLUMN] = full_train_df[LABEL_COLUMN].astype(int)
test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].astype(int)

# --- Stratified 90/10 split of the TRAIN pool into train + validation ---
# Validation is used ONLY for early-stopping / best-checkpoint selection.
# The 500/class TEST_CSV is never seen during training or model selection.
train_df, val_df = train_test_split(
    full_train_df,
    test_size=VAL_SPLIT,
    stratify=full_train_df[LABEL_COLUMN],   # preserve class proportions in both slices
    random_state=SEED,
    shuffle=True
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Label mapping is derived from the FULL train pool so every class is covered.
labels = sorted(full_train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(label): i for i, label in enumerate(labels)}
id2label = {i: int(label) for i, label in enumerate(labels)}

if accelerator.is_main_process:
    print(f"Detected {num_labels} emotion classes: {labels}")
    print(f"Train split: {len(train_df)} | Val split (selection): {len(val_df)} | Test (final): {len(test_df)}")
    print("Train class counts:\n", train_df[LABEL_COLUMN].value_counts().sort_index().to_string())
    print("Val class counts:\n", val_df[LABEL_COLUMN].value_counts().sort_index().to_string())
    print("Test class counts:\n", test_df[LABEL_COLUMN].value_counts().sort_index().to_string())

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

# 4. Initialize Fast Image Processor & Transforms
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def transform(example_batch):
    images = [Image.open(path).convert("RGB") for path in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs

encoded_datasets = dataset_dict.with_transform(transform)

# 5. Initialize Model
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
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=True
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

# 7. Initialize Trainer  (validation = the 10% held-out slice, for selection only)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

# 8. Train the Model
if accelerator.is_main_process:
    print("Starting training process across available V100 GPUs...")
trainer.train()

# 9. Evaluate on the held-out AffectNet validation set (used here as the FINAL TEST set)
if accelerator.is_main_process:
    print("Evaluating on the 500/class AffectNet test set (never used for training or selection)...")
    test_results = trainer.predict(encoded_datasets["test"])
    test_preds = np.argmax(test_results.predictions, axis=1)
    test_true = test_results.label_ids

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=[str(id2label[i]) for i in range(num_labels)]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true, test_preds))

    trainer.save_model(os.path.join(OUTPUT_DIR, "best_vit_affectnet"))
    image_processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_vit_affectnet"))
    print(f"Model successfully saved to {OUTPUT_DIR}/best_vit_affectnet")
