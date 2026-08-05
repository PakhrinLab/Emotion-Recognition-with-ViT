import os
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

# 1. Define paths and model checkpoint
data_dir = "/project/pakhrin/subash/DATASET"
train_dir = os.path.join(data_dir, "train")
model_checkpoint = "google/vit-base-patch16-224-in21k"

# 2. Load only the training dataset using Hugging Face datasets 'imagefolder' format
print("Loading training dataset...")
dataset = load_dataset("imagefolder", data_files={"train": os.path.join(train_dir, "**")})

# Extract label mappings and raw data arrays for StratifiedKFold
train_data = dataset["train"]
labels = train_data.features["label"].names
num_labels = len(labels)
id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}
print(f"Found {num_labels} classes: {labels}")

# Extract images and labels into lists/arrays for stratification splits
all_images = train_data["image"]
all_labels = train_data["label"]

# 3. Initialize Image Processor and setup transforms
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

image_mean = image_processor.image_mean
image_std = image_processor.image_std
size = image_processor.size["height"]

train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ]
)


def preprocess_train(example_batch):
    """Apply train transforms and format batch for ViT"""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Use built-in fast image_processor for validation data"""
    images = [image.convert("RGB") for image in example_batch["image"]]
    inputs = image_processor(images, return_tensors="pt")
    example_batch["pixel_values"] = inputs.pixel_values
    return example_batch


# Define custom collate function to handle batching properly
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


# Define an accuracy metric evaluation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


# 4. Setup 10-Fold Stratified Cross-Validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []

print(f"\nStarting {n_splits}-Fold Stratified Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(
    skf.split(all_images, all_labels)
):
    print(f"\n--- Training Fold {fold + 1}/{n_splits} ---")

    # Create subset datasets for current fold
    fold_train_images = [all_images[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]

    fold_val_images = [all_images[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]

    fold_train_ds = Dataset.from_dict(
        {"image": fold_train_images, "label": fold_train_labels}
    )
    fold_val_ds = Dataset.from_dict(
        {"image": fold_val_images, "label": fold_val_labels}
    )

    # Apply transformations dynamically
    train_dataset_folded = fold_train_ds.with_transform(preprocess_train)
    eval_dataset_folded = fold_val_ds.with_transform(preprocess_val)

    # Re-initialize ViT Model for each fold to ensure clean weights
    model = ViTForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Define Training Arguments unique to the fold output directory
    training_args = TrainingArguments(
        output_dir=f"./vit-custom-output-fold-{fold + 1}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",  # Change to "wandb" or tensorboard if preferred
    )

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_folded,
        eval_dataset=eval_dataset_folded,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate on validation fold
    eval_results = trainer.evaluate(eval_dataset_folded)
    print(f"Fold {fold + 1} Evaluation Results: {eval_results}")
    fold_results.append(eval_results["eval_accuracy"])

    # Generate predictions to compute detailed metrics for this fold
    predictions_output = trainer.predict(eval_dataset_folded)
    predicted_classes = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids

    print(f"\nConfusion Matrix for Fold {fold + 1}:")
    print(confusion_matrix(true_labels, predicted_classes))

    print(f"\nClassification Report for Fold {fold + 1}:")
    print(
        classification_report(
            true_labels, predicted_classes, target_names=labels
        )
    )

# 5. Summary of Cross-Validation
print("\n================================")
print(f"{n_splits}-Fold Cross-Validation Complete!")
print(f"Accuracies per fold: {fold_results}")
print(f"Mean Accuracy: {np.mean(fold_results):.4f}")
print(f"Standard Deviation: {np.std(fold_results):.4f}")
print("================================")