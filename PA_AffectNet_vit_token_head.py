import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    ViTModel,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback,          # <-- NEW: stop near the macro-F1 peak
)
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 1. Accelerator
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"Detected {accelerator.num_processes} GPU(s). Setting up training environment...")

# 2. Config
TRAIN_CSV = "subash_affectnet_train_filtered_labels.csv"
TEST_CSV = "subash_affectnet_validation_filtered_labels.csv"

# ---- Backbone (config flag): face-pretrained by default; swap to ImageNet ViT for the baseline ----
MODEL_CHECKPOINT = "trpakov/vit-face-expression"
# MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5        # backbone LR; the fresh head gets 10x this (see CustomTrainer)
HEAD_LR_MULT = 10.0
WARMUP_RATIO = 0.1
NUM_WORKERS = 2          # LOWERED 4->2 to cut host-RAM use (was OOM-killed)
OUTPUT_DIR = "./vit_token_head_results"
VAL_SPLIT = 0.10
SEED = 42

# ---- Emotion-transformer head ----
HEAD_LAYERS = 2            # 2 transformer encoder layers on top of the patch tokens
HEAD_NHEADS = 12           # matches ViT-base (768 / 12 = 64)
HEAD_DROPOUT = 0.1
USE_POS_EMBED = True       # <-- NEW: learned positional embedding for the head's own encoder

# ---- Toggles ----
USE_CLASS_WEIGHTS = True   # fixes the Happy-dumping imbalance seen in the confusion matrix
FREEZE_BACKBONE = False    # True = pure feature-extractor ablation; False = end-to-end fine-tune

# ---- Early stopping / checkpoint selection ----
EARLY_STOPPING_PATIENCE = 3   # stop if eval macro-F1 doesn't improve for this many epochs

# 3. Load data + stratified 90/10 split
if accelerator.is_main_process:
    print("Loading CSV annotations...")

full_train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
full_train_df[LABEL_COLUMN] = full_train_df[LABEL_COLUMN].astype(int)
test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].astype(int)

train_df, val_df = train_test_split(
    full_train_df, test_size=VAL_SPLIT,
    stratify=full_train_df[LABEL_COLUMN], random_state=SEED, shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

labels = sorted(full_train_df[LABEL_COLUMN].unique())
num_labels = len(labels)
label2id = {int(l): i for i, l in enumerate(labels)}
id2label = {i: int(l) for i, l in enumerate(labels)}

# Class weights (balanced) computed on the TRAIN split, in model-index order
mapped_train = train_df[LABEL_COLUMN].map(label2id).values
cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=mapped_train)
class_weights_t = torch.tensor(cw, dtype=torch.float)

if accelerator.is_main_process:
    print(f"Backbone: {MODEL_CHECKPOINT} | freeze_backbone={FREEZE_BACKBONE} | class_weights={USE_CLASS_WEIGHTS} | pos_embed={USE_POS_EMBED}")
    print(f"Classes: {labels}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Class weights:", {id2label[i]: round(float(w), 3) for i, w in enumerate(cw)})

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

# 4. Processor + transform
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def transform(example_batch):
    images = [Image.open(p).convert("RGB") for p in example_batch[IMAGE_COLUMN]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = [label2id[l] for l in example_batch[LABEL_COLUMN]]
    return inputs

encoded = dataset_dict.with_transform(transform)

# 5. Model: ViT backbone -> [pool_token + 197 ViT tokens] (+ pos embed) -> 2-layer transformer -> classify
class ViTTokenTransformerHead(nn.Module):
    def __init__(self, checkpoint, num_labels, class_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.vit = ViTModel.from_pretrained(checkpoint, add_pooling_layer=False)
        hidden = self.vit.config.hidden_size  # 768

        if FREEZE_BACKBONE:
            for p in self.vit.parameters():
                p.requires_grad = False

        # fresh learnable pooling token (its own "CLS") prepended to the ViT token sequence
        self.pool_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.pool_token, std=0.02)

        # ---- NEW: learned positional embedding for the head's sequence ----
        # Sequence seen by the head = [pool_token] + [CLS + patch tokens from ViT]
        # = 1 + (num_patches + 1) = num_patches + 2  (e.g. 196 + 2 = 198 for 224/16)
        image_size = self.vit.config.image_size
        patch_size = self.vit.config.patch_size
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        num_patches = (image_size // patch_size) ** 2
        self.head_seq_len = num_patches + 2  # + ViT CLS + our pool token
        self.use_pos_embed = USE_POS_EMBED
        if self.use_pos_embed:
            # index 0 gets its own embedding -> the head can learn "this is the readout slot"
            self.pos_embed = nn.Parameter(torch.zeros(1, self.head_seq_len, hidden))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=HEAD_NHEADS, dim_feedforward=hidden * 4,
            dropout=HEAD_DROPOUT, activation="gelu", batch_first=True, norm_first=True,
        )
        self.head_transformer = nn.TransformerEncoder(layer, num_layers=HEAD_LAYERS)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_labels)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, pixel_values=None, labels=None):
        # ViT patch+CLS tokens: [B, 197, 768]  (CLS kept at index 0)
        tokens = self.vit(pixel_values=pixel_values).last_hidden_state
        B = tokens.size(0)
        pool = self.pool_token.expand(B, -1, -1)          # [B, 1, 768]
        seq = torch.cat([pool, tokens], dim=1)            # [B, 198, 768]

        # ---- NEW: add learned positional embedding before the head encoder ----
        if self.use_pos_embed:
            seq = seq + self.pos_embed[:, : seq.size(1), :]

        seq = self.head_transformer(seq)
        pooled = self.norm(seq[:, 0])                     # learnable pool token position
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            w = self.class_weights if self.class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=w)
        return {"loss": loss, "logits": logits}

model = ViTTokenTransformerHead(
    MODEL_CHECKPOINT, num_labels,
    class_weights=class_weights_t if USE_CLASS_WEIGHTS else None,
)

# 6. Trainer with differential LR (backbone 1x, fresh head 10x)
class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            head_keys = ("pool_token", "pos_embed", "head_transformer", "norm", "classifier")
            backbone, head = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (head if n.startswith(head_keys) or any(k in n for k in head_keys) else backbone).append(p)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone, "lr": self.args.learning_rate},
                    {"params": head, "lr": self.args.learning_rate * HEAD_LR_MULT},
                ],
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",             # <-- CHANGED: was "no"; needed to keep the best epoch
    save_total_limit=2,                # <-- keep disk usage small (best + last)
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,       # <-- CHANGED: was False; test the best model, not epoch 10
    metric_for_best_model="macro_f1",  # <-- select on macro-F1, NOT accuracy (accuracy hid the collapse)
    greater_is_better=True,
    logging_steps=50,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=True,
    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=False,   # OFF: pinned pages were adding host-RAM pressure
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],  # <-- NEW
)

# 7. Train
if accelerator.is_main_process:
    print("Training ViT + token-transformer head...")
trainer.train()

# 8. Final test (predict() is collective under DDP -> all ranks call it)
test_results = trainer.predict(encoded["test"])
if accelerator.is_main_process:
    test_preds = np.argmax(test_results.predictions, axis=1)
    test_true = test_results.label_ids
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds,
                                target_names=[str(id2label[i]) for i in range(num_labels)], digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true, test_preds))
    print(f"\nAccuracy: {np.mean(test_preds == test_true):.4f}")
    print(f"Macro-F1: {f1_score(test_true, test_preds, average='macro'):.4f}")

accelerator.wait_for_everyone()
