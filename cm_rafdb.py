import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Define paths and model checkpoint
data_dir = "/project/pakhrin/subash/DATASET"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_checkpoint = "google/vit-base-patch16-224-in21k"

# 2. Load dataset using Hugging Face datasets 'imagefolder' format
print("Loading dataset from directories...")
dataset = load_dataset(
    "imagefolder", 
    data_files={
        "train": os.path.join(train_dir, "**"),
        "test": os.path.join(test_dir, "**")
    }
)

# Extract label mappings
labels = dataset["train"].features["label"].names
num_labels = len(labels)
id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}
print(f"Found {num_labels} classes: {labels}")

# 3. Initialize Image Processor and setup transforms
image_processor = AutoImageProcessor.from_pretrained(
    model_checkpoint, use_fast=True
)

# Get normalization stats and image size from processor configuration
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
    """Use the built-in fast image_processor for validation/test data for efficiency"""
    images = [image.convert("RGB") for image in example_batch["image"]]
    inputs = image_processor(images, return_tensors="pt")
    example_batch["pixel_values"] = inputs.pixel_values
    return example_batch

# Apply preprocessing dynamically
print("Preprocessing datasets...")
train_dataset = dataset["train"].with_transform(preprocess_train)
eval_dataset = dataset["test"].with_transform(preprocess_val)

# 4. Initialize ViT Model for Classification
model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Safely replaces the classification head for your number of classes
)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./vit-custom-output",
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
    save_total_limit=2,
    remove_unused_columns=False,
)

# 6. Define an accuracy metric evaluation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Collate function to handle batching properly
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

# 7. Instantiate Trainer and Start Training (using train dataset for training)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# 8. Evaluate and Predict on Independent Test Set with Confusion Matrix
print("Evaluating model on independent test dataset...")
eval_results = trainer.evaluate(eval_dataset)
print(f"Test Evaluation Results: {eval_results}")

# Generate predictions for the independent test dataset
predictions_output = trainer.predict(eval_dataset)
predicted_classes = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

# Compute overall accuracy explicitly on the test set predictions
test_accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Overall Accuracy on Independent Test Dataset: {test_accuracy:.4f}")

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Print a detailed classification report per class
print("\nClassification Report:")
print(classification_report(true_labels, predicted_classes, target_names=labels))
