"""
RAFDB_train_and_robustness.py
-----------------------------
Trains the STRONG-recipe RAF-DB facial-emotion ViT ONCE (seed 42), keeps the
trained model IN MEMORY, and immediately runs a robustness analysis under four
families of corruptions -- OCCLUSION, ILLUMINATION, HEAD POSE, FACIAL MASKS --
plus a McNemar significance test of each corruption vs. the clean test set.

NO MODEL WEIGHTS ARE SAVED. The best-checkpoint selection still runs (on macro-F1)
using a scratch directory that is deleted at the end; the final in-memory model is
used directly for the robustness pass and then discarded. Only result files are
written: a robustness table, a significance table, and an .npz of predictions.

Launch (single GPU, single process -- do NOT use multi-GPU accelerate launch):
    python RAFDB_train_and_robustness.py

Outputs:
    robustness_results_rafdb.csv        accuracy / macro-F1 / drop per (corruption, severity)
    robustness_significance_rafdb.csv   McNemar clean-vs-corruption p-values
    robustness_predictions_rafdb.npz    per-sample predictions + ground truth
"""

import os
import gc
import shutil
import tempfile
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torchvision.transforms import (
    Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop,
    RandAugment, RandomErasing, ToTensor,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
VAL_FRACTION = 0.10
DATA_DIR = "/project/pakhrin/subash/DATASET"     # RAF-DB imagefolder: train/<cls>/*, test/<cls>/*
MODEL_CHECKPOINT = "trpakov/vit-face-expression"

EPOCHS = 20
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True
TRAIN_BS = 32
EVAL_BS = 32

SCRATCH_ROOT = (os.environ.get("SLURM_TMPDIR")
                or os.environ.get("TMPDIR")
                or tempfile.gettempdir())

_RNG = np.random.default_rng(SEED)   # reproducible random-occlusion positions


# ======================================================================
# 1. Corruption functions (PIL RGB -> PIL RGB, same size)
# ======================================================================
def _occlude_random(img, coverage):
    img = img.copy(); w, h = img.size; draw = ImageDraw.Draw(img)
    n_boxes = 3; area_each = coverage * w * h / n_boxes
    side = max(1, int(area_each ** 0.5))
    for _ in range(n_boxes):
        x0 = int(_RNG.integers(0, max(1, w - side)))
        y0 = int(_RNG.integers(0, max(1, h - side)))
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=(0, 0, 0))
    return img


def _occlude_region(img, region):
    img = img.copy(); w, h = img.size; draw = ImageDraw.Draw(img)
    if region == "eyes":
        draw.rectangle([0.12 * w, 0.26 * h, 0.88 * w, 0.48 * h], fill=(0, 0, 0))
    elif region == "mouth":
        draw.rectangle([0.20 * w, 0.60 * h, 0.80 * w, 0.86 * h], fill=(0, 0, 0))
    return img


def _gamma(img, gamma):
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** gamma, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _directional_light(img, strength):
    w, h = img.size
    grad = np.linspace(-strength, strength, w)[None, :, None]
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr * (1.0 + grad), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _rotate(img, degrees):
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def _find_coeffs(target_corners, source_corners):
    matrix = []
    for t, s in zip(target_corners, source_corners):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(source_corners, dtype=float).reshape(8)
    return np.linalg.lstsq(A, B, rcond=None)[0].tolist()


def _yaw(img, amount):
    w, h = img.size; dy = amount * h
    target = [(0, 0), (w, 0), (w, h), (0, h)]
    source = [(0, 0), (w, dy), (w, h - dy), (0, h)]
    coeffs = _find_coeffs(target, source)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def _mask(img, level):
    img = img.copy(); w, h = img.size; draw = ImageDraw.Draw(img)
    top_y = 0.50 * h if level == "full" else 0.62 * h
    color = (150, 180, 210)
    poly = [(0.16 * w, top_y), (0.84 * w, top_y), (0.90 * w, 0.99 * h), (0.10 * w, 0.99 * h)]
    draw.polygon(poly, fill=color)
    draw.line([(0.16 * w, top_y), (0.02 * w, top_y - 0.04 * h)], fill=color, width=max(2, int(0.02 * w)))
    draw.line([(0.84 * w, top_y), (0.98 * w, top_y - 0.04 * h)], fill=color, width=max(2, int(0.02 * w)))
    return img


def _identity(img):
    return img


def build_suite():
    return {
        "clean": [("none", _identity)],
        "occlusion": [
            ("random_15pct", lambda im: _occlude_random(im, 0.15)),
            ("random_30pct", lambda im: _occlude_random(im, 0.30)),
            ("random_45pct", lambda im: _occlude_random(im, 0.45)),
            ("eyes",  lambda im: _occlude_region(im, "eyes")),
            ("mouth", lambda im: _occlude_region(im, "mouth")),
        ],
        "illumination": [
            ("dark_g1.8",   lambda im: _gamma(im, 1.8)),
            ("dark_g2.6",   lambda im: _gamma(im, 2.6)),
            ("bright_g0.6", lambda im: _gamma(im, 0.6)),
            ("bright_g0.4", lambda im: _gamma(im, 0.4)),
            ("side_light",  lambda im: _directional_light(im, 0.6)),
        ],
        "head_pose": [
            ("roll_10deg", lambda im: _rotate(im, 10)),
            ("roll_20deg", lambda im: _rotate(im, 20)),
            ("roll_30deg", lambda im: _rotate(im, 30)),
            ("yaw_0.20",   lambda im: _yaw(im, 0.20)),
            ("yaw_0.35",   lambda im: _yaw(im, 0.35)),
        ],
        "mask": [
            ("mouth", lambda im: _mask(im, "mouth")),
            ("full",  lambda im: _mask(im, "full")),
        ],
    }


# ======================================================================
# 2. Weighted + label-smoothed Trainer
# ======================================================================
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


# ======================================================================
# 3. Inference under a corruption (stock processor -> matches training eval)
# ======================================================================
@torch.no_grad()
def evaluate(corrupt_fn, images, y_true, model, processor, device, batch_size):
    preds = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        imgs = [corrupt_fn(im.convert("RGB")) for im in batch]
        inputs = processor(imgs, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return accuracy_score(y_true, preds), f1_score(y_true, preds, average="macro"), preds


def mcnemar_vs_clean(clean_preds, corrupt_preds, y_true):
    clean_ok = (clean_preds == y_true); corr_ok = (corrupt_preds == y_true)
    b = int(np.sum(clean_ok & ~corr_ok))   # clean right, corrupt wrong -> corruption hurt
    c = int(np.sum(~clean_ok & corr_ok))   # clean wrong, corrupt right -> corruption helped
    table = [[int(np.sum(clean_ok & corr_ok)), b], [c, int(np.sum(~clean_ok & ~corr_ok))]]
    res = mcnemar(table, exact=(b + c) < 25, correction=True)
    favors = "clean" if b > c else ("corrupt" if c > b else "tie")
    return b, c, float(res.pvalue), favors


# ======================================================================
# 4. Main: train once -> robustness in-memory
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--batch_size", type=int, default=64)      # inference batch
    ap.add_argument("--limit", type=int, default=0)            # limit TEST images (smoke test)
    ap.add_argument("--out_csv", default="robustness_results_rafdb.csv")
    ap.add_argument("--out_sig", default="robustness_significance_rafdb.csv")
    ap.add_argument("--out_preds", default="robustness_predictions_rafdb.npz")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    print(f"Device: {device} | Backbone: {MODEL_CHECKPOINT} | Seed: {SEED}")

    # ---- Data ----
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    print("Loading RAF-DB imagefolder...")
    dataset = load_dataset("imagefolder", data_files={
        "train": os.path.join(train_dir, "**"),
        "test": os.path.join(test_dir, "**"),
    })
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    id2label = {i: c for i, c in enumerate(labels)}
    label2id = {c: i for i, c in enumerate(labels)}
    print(f"Found {num_labels} classes: {labels}")

    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    mean, std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    train_transforms = Compose([
        RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=7),
        ToTensor(),
        Normalize(mean=mean, std=std),
        RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    def preprocess_train(batch):
        batch["pixel_values"] = [train_transforms(im.convert("RGB")) for im in batch["image"]]
        return batch

    def preprocess_eval(batch):
        imgs = [im.convert("RGB") for im in batch["image"]]
        batch["pixel_values"] = processor(imgs, return_tensors="pt").pixel_values
        return batch

    def collate_fn(b):
        return {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
                "labels": torch.tensor([x["label"] for x in b])}

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred[0], axis=1)
        return {"accuracy": accuracy_score(eval_pred[1], preds),
                "macro_f1": f1_score(eval_pred[1], preds, average="macro")}

    # Stratified 90/10 split for best-ckpt selection (test never touched)
    split = dataset["train"].train_test_split(
        test_size=VAL_FRACTION, stratify_by_column="label", seed=SEED)
    cw = compute_class_weight("balanced", classes=np.arange(num_labels),
                              y=np.array(split["train"]["label"]))
    class_weights_t = torch.tensor(cw, dtype=torch.float) if USE_CLASS_WEIGHTS else None

    train_dataset = split["train"].with_transform(preprocess_train)
    val_dataset = split["test"].with_transform(preprocess_eval)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=num_labels,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    ckpt_dir = os.path.join(SCRATCH_ROOT, f"rafdb_ckpt_seed{SEED}")
    training_args = TrainingArguments(
        output_dir=ckpt_dir, eval_strategy="epoch", save_strategy="epoch",
        learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO, lr_scheduler_type="cosine",
        per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=EPOCHS, weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, logging_steps=20, remove_unused_columns=False,
        fp16=torch.cuda.is_available(), seed=SEED, data_seed=SEED, report_to="none")

    trainer = WeightedSmoothTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collate_fn, compute_metrics=compute_metrics,
        class_weights=class_weights_t, label_smoothing=LABEL_SMOOTHING)

    print("\nTraining once (seed 42)...")
    trainer.train()

    # best model is now loaded in memory (load_best_model_at_end=True)
    model = trainer.model.to(device).eval()

    # We no longer need the Trainer / scratch checkpoints
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    # ---- Robustness pass on the untouched TEST set, in memory ----
    test_ds = dataset["test"]
    if args.limit > 0:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
    images = list(test_ds["image"])
    y_true = np.array(test_ds["label"])
    print(f"\nRobustness eval on {len(images)} RAF-DB test images.\n")

    suite = build_suite()
    rows, saved_preds = [], {"y_true": y_true}
    clean_acc = clean_preds = None
    for group, variants in suite.items():
        for sev_label, fn in variants:
            acc, mf1, preds = evaluate(fn, images, y_true, model, processor,
                                       device, args.batch_size)
            if group == "clean":
                clean_acc, clean_preds = acc, preds
            drop = 0.0 if clean_acc is None else (clean_acc - acc) * 100
            rows.append({"corruption": group, "severity": sev_label,
                         "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                         "acc_drop_vs_clean_pct": round(drop, 2)})
            saved_preds[f"{group}__{sev_label}"] = preds
            print(f"{group:>13} | {sev_label:<14} acc={acc:.4f}  macroF1={mf1:.4f}  drop={drop:+.2f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.out_csv, index=False)
    np.savez_compressed(args.out_preds, **saved_preds)

    print("\n================ McNemar: corruption vs. clean ================")
    sig_rows = []
    for group, variants in suite.items():
        if group == "clean":
            continue
        for sev_label, _ in variants:
            b, c, p, favors = mcnemar_vs_clean(clean_preds,
                                               saved_preds[f"{group}__{sev_label}"], y_true)
            sig = "yes" if p < args.alpha else "no"
            sig_rows.append({"corruption": group, "severity": sev_label,
                             "clean_only_correct_b": b, "corrupt_only_correct_c": c,
                             "mcnemar_p": p, f"significant_at_{args.alpha}": sig, "favors": favors})
            print(f"{group:>13} | {sev_label:<14} b={b:<5} c={c:<5} p={p:.3e}  sig={sig}  favors={favors}")
    pd.DataFrame(sig_rows).to_csv(args.out_sig, index=False)

    fam = (results[results.corruption != "clean"].groupby("corruption")
           .agg(mean_accuracy=("accuracy", "mean"),
                mean_drop_pct=("acc_drop_vs_clean_pct", "mean")).round(4).reset_index())

    print("\n================ Robustness summary ================")
    print(f"Clean accuracy: {clean_acc:.4f}\n")
    print(results.to_string(index=False))
    print("\nPer-family mean (over severities):")
    print(fam.to_string(index=False))
    print(f"\nSaved: {args.out_csv}, {args.out_sig}, {args.out_preds}")
    print("No model weights were saved.")


if __name__ == "__main__":
    main()
