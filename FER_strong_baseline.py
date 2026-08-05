"""
FER_strong_baseline.py
----------------------
A stronger single-stream Vision-Transformer recipe for AffectNet / RAF-DB,
designed to be competitive with recent transformer FER methods (POSTER++, DDAMFN, ...).

Upgrades over the plain-ViT baseline (all of these move the needle):
  1. Real train-time augmentation (RandAugment + flip + random-resized-crop + random erasing)
     -> biggest single gain, and the fix for the "eval_loss climbs after epoch 2" overfitting.
  2. Class-balanced + label-smoothed cross-entropy -> essential for AffectNet's imbalance.
  3. Best-checkpoint selection on macro-F1 (not final-epoch weights).
  4. Face-domain-pretrained backbone by default (swappable in one line).
  5. STANDARD-PROTOCOL evaluation so the numbers are directly comparable to the literature
     (RAF-DB: official test split; AffectNet: the balanced 500-per-class validation set).

IMPORTANT (read before running):
  * For a PUBLISHABLE AffectNet number, TEST_CSV MUST point at the official AffectNet
    validation set (4000 imgs, 500/class for 8-cls). Do NOT reuse a custom 80/10/10 split
    of the training data -- that inflates accuracy via the majority classes and is not
    comparable to POSTER++/DDAMFN/etc.
"""
import os
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ------------------------------------------------------------------ Accelerator
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Strong FER baseline.")

# ------------------------------------------------------------------ Config
# --- Point these at ONE dataset at a time (RAF-DB or AffectNet) ---
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"
TEST_CSV  = "subash_affectnet_validation_filtered_labels.csv"   # <-- STANDARD val/test split!
VAL_SPLIT = 0.05          # small val slice carved from TRAIN, used only for best-ckpt selection

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

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
SEED = 42
OUTPUT_DIR = "./fer_strong_results"

set_seed(SEED)

# ------------------------------------------------------------------ Data
if accelerator.is_main_process:
    print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].astype(int)
test_df[LABEL_COLUMN]  = test_df[LABEL_COLUMN].astype(int)

labels = sorted(train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(l): i for i, l in enumerate(labels)}
id2label = {i: int(l) for i, l in enumerate(labels)}

# Small stratified val slice from TRAIN (best-checkpoint selection only)
from sklearn.model_selection import train_test_split
tr_df, va_df = train_test_split(
    train_df, test_size=VAL_SPLIT, stratify=train_df[LABEL_COLUMN],
    random_state=SEED, shuffle=True,
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)

# Class weights from the TRAIN slice
mapped = tr_df[LABEL_COLUMN].map(label2id).values
cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=mapped)
class_weights_t = torch.tensor(cw, dtype=torch.float)

if accelerator.is_main_process:
    print(f"Backbone: {MODEL_CHECKPOINT} | classes: {labels}")
    print(f"Train: {len(tr_df)} | Val(for ckpt): {len(va_df)} | Test: {len(test_df)}")
    print(f"Class weights: {np.round(cw, 3)}")

# ------------------------------------------------------------------ Transforms
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
std  = getattr(image_processor, "image_std",  [0.5, 0.5, 0.5])

# Training augmentation. Kept face-safe: mild crop, flip is fine (expressions ~symmetric),
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

train_ds = Dataset.from_pandas(tr_df).with_transform(make_transform(train_ops))
val_ds   = Dataset.from_pandas(va_df).with_transform(make_transform(eval_ops))
test_ds  = Dataset.from_pandas(test_df).with_transform(make_transform(eval_ops))

# ------------------------------------------------------------------ Model
model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

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

# ------------------------------------------------------------------ Training args
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
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,           # keep the best epoch, not the overfit last one
    metric_for_best_model="macro_f1",      # macro-F1: the right target under class imbalance
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    seed=SEED,
    data_seed=SEED,
)

trainer = WeightedSmoothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
    class_weights=class_weights_t if USE_CLASS_WEIGHTS else None,
    label_smoothing=LABEL_SMOOTHING,
)

# ------------------------------------------------------------------ Train
if accelerator.is_main_process:
    print("Training strong FER baseline...")
trainer.train()

# ------------------------------------------------------------------ Standard-protocol test
test_results = trainer.predict(test_ds)      # collective under DDP
if accelerator.is_main_process:
    preds = np.argmax(test_results.predictions, axis=1)
    gold = test_results.label_ids
    acc = float(np.mean(preds == gold))
    mf1 = float(f1_score(gold, preds, average="macro"))
    print("\n================ STANDARD-PROTOCOL TEST ================")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        gold, preds, target_names=[str(id2label[i]) for i in range(num_labels)], digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(gold, preds))

    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    image_processor.save_pretrained(best_dir)
    print(f"\nSaved best model to {best_dir}")
