import os
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from accelerate import Accelerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Initialize Accelerator for Automatic Multi-GPU Detection
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Setting up 10-Fold Stratified Cross-Validation environment...")

# 2. Configuration & Paths
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-4
OUTPUT_DIR = "./vit_affectnet_10fold_results"
NUM_FOLDS = 10

# 3. Load Training Data and Setup Labels
if accelerator.is_main_process:
    print(f"Loading training annotations from {TRAIN_CSV}...")

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].astype(int)

labels = sorted(train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(label): i for i, label in enumerate(labels)}
id2label = {i: int(label) for i, label in enumerate(labels)}

if accelerator.is_main_process:
    print(f"Detected {num_labels} emotion classes: {labels}")
    print(f"Total dataset size: {len(train_df)} samples. Splitting into {NUM_FOLDS} stratified bins...")

# 4. Initialize Fast Image Processor & Transform Function
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def transform(example_batch):
    images = [Image.open(path).convert("RGB") for path in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs

# 5. Setup Stratified K-Fold (10 Bins)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_metrics = []

# Accumulators for aggregate out-of-fold classification reports/matrices if desired
all_true_labels = []
all_pred_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[LABEL_COLUMN])):
    if accelerator.is_main_process:
        print(f"\n==========================================")
        print(f"--- Training Fold {fold + 1}/{NUM_FOLDS} ---")
        print(f"Training samples: {len(train_idx)} | Test/Validation bin samples: {len(val_idx)}")
        print(f"==========================================")

    # Create train (9 bins) and test (10th bin) dataframes for this iteration
    fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # Convert to Hugging Face Datasets with transform mapping
    fold_train_ds = Dataset.from_pandas(fold_train_df).with_transform(transform)
    fold_val_ds = Dataset.from_pandas(fold_val_df).with_transform(transform)

    # Re-initialize the model for the current fold to ensure a clean slate
    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}")

    # Training Arguments specific to this fold
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
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
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        accuracy = np.mean(preds == labels)
        return {"accuracy": accuracy}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fold_train_ds,
        eval_dataset=fold_val_ds,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    # Train model on the 9 bins
    trainer.train()

    # Evaluate on the 10th bin
    if accelerator.is_main_process:
        print(f"Evaluating Fold {fold + 1} on its held-out bin...")
        eval_results = trainer.predict(fold_val_ds)
        fold_preds = np.argmax(eval_results.predictions, axis=1)
        fold_true = eval_results.label_ids

        acc = np.mean(fold_preds == fold_true)
        fold_metrics.append(acc)
        print(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")

        # Store predictions for aggregate metrics later
        all_true_labels.extend(fold_true)
        all_pred_labels.extend(fold_preds)

        # Save individual fold model if desired
        fold_model_path = os.path.join(fold_output_dir, "best_model")
        trainer.save_model(fold_model_path)
        image_processor.save_pretrained(fold_model_path)

# 6. Final 10-Fold Cross-Validation Summary
if accelerator.is_main_process:
    print("\n==========================================")
    print("--- 10-Fold Cross-Validation Completed ---")
    print(f"==========================================")
    print(f"Accuracies across all 10 bins: {[round(m, 4) for m in fold_metrics]}")
    print(f"Mean Accuracy: {np.mean(fold_metrics):.4f} (+/- {np.std(fold_metrics):.4f})")
    
    print("\nOverall Aggregate Classification Report (Across all 10 held-out bins):")
    print(classification_report(all_true_labels, all_pred_labels, target_names=[str(id2label[i]) for i in range(num_labels)]))

    print("\nOverall Aggregate Confusion Matrix:")
    print(confusion_matrix(all_true_labels, all_pred_labels))