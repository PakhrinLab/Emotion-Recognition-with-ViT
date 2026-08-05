"""
Robustness evaluation for AffectNet emotion models.

Loads an ALREADY-TRAINED model and re-evaluates it on the test set under
controlled corruptions -- occlusion, illumination, head pose, and masks --
at several severity levels. No training happens here.

Output: a table (printed + CSV) of accuracy and macro-F1 for every
(corruption, severity), plus the relative accuracy drop vs. the clean set,
and an .npz of per-sample predictions for later significance testing.

Run (single GPU is plenty for inference):
    python AffectNet_robustness_eval.py \
        --model_dir ./vit_affectnet_results/best_vit_affectnet \
        --test_csv  subash_affectnet_validation_filtered_labels.csv

NOTE ON MODEL LOADING
---------------------
This uses AutoModelForImageClassification, which loads the standard HF models
(e.g. the google/vit or swin baselines saved with trainer.save_model).
The custom ViTTokenTransformerHead / ViTTwoStageRegion models are saved as a
raw state_dict (pytorch_model.bin), so for those, replace `load_model()` with:
    from AffectNet_vit_twostage_region import ViTTwoStageRegion   # the class
    model = ViTTwoStageRegion(BACKBONE_CKPT, num_labels)
    model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin"))
and call model(pixel_values=...) instead of model(**inputs).
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------------------------------------------
# Config defaults (override via CLI)
# ----------------------------------------------------------------------------
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "human_label"
SEED = 42

# Deterministic RNG so the "random" occlusion positions are reproducible.
_RNG = np.random.default_rng(SEED)


# ============================================================================
# 1. Corruption functions.  Each takes a PIL RGB image and returns a PIL RGB
#    image of the SAME size.  They run on the raw crop, BEFORE the processor
#    resizes/normalizes, so severities are expressed relative to the face.
# ============================================================================

def _occlude_random(img, coverage):
    """Paste a few black squares covering ~`coverage` fraction of the area."""
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    n_boxes = 3
    area_each = coverage * w * h / n_boxes
    side = max(1, int(area_each ** 0.5))
    for _ in range(n_boxes):
        x0 = int(_RNG.integers(0, max(1, w - side)))
        y0 = int(_RNG.integers(0, max(1, h - side)))
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=(0, 0, 0))
    return img


def _occlude_region(img, region):
    """Black out a semantic band: the eyes or the mouth."""
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if region == "eyes":
        draw.rectangle([0.12 * w, 0.26 * h, 0.88 * w, 0.48 * h], fill=(0, 0, 0))
    elif region == "mouth":
        draw.rectangle([0.20 * w, 0.60 * h, 0.80 * w, 0.86 * h], fill=(0, 0, 0))
    return img


def _gamma(img, gamma):
    """Illumination via gamma. gamma>1 darkens, gamma<1 brightens."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** gamma, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _directional_light(img, strength):
    """Uneven left->right lighting gradient (simulates a side light source)."""
    w, h = img.size
    grad = np.linspace(-strength, strength, w)[None, :, None]  # [1, w, 1]
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr * (1.0 + grad), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _rotate(img, degrees):
    """In-plane rotation = head *roll*. Revealed corners filled black."""
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False,
                      fillcolor=(0, 0, 0))


def _find_coeffs(target_corners, source_corners):
    """Solve the 8 perspective coefficients mapping output->input for PIL."""
    matrix = []
    for t, s in zip(target_corners, source_corners):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(source_corners, dtype=float).reshape(8)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs.tolist()


def _yaw(img, amount):
    """Approximate out-of-plane yaw by compressing one side vertically.

    True yaw/pitch needs a 3D face model; this trapezoidal warp is a 2D proxy
    and should be described as such in the paper.
    """
    w, h = img.size
    dy = amount * h
    target = [(0, 0), (w, 0), (w, h), (0, h)]               # output corners
    source = [(0, 0), (w, dy), (w, h - dy), (0, h)]         # sampled-from corners
    coeffs = _find_coeffs(target, source)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def _mask(img, level):
    """Overlay a synthetic surgical mask over the lower face.

    `level='mouth'` covers mouth/chin; `level='full'` also covers the nose.
    Geometric approximation (no landmarks): a trapezoid + ear straps.
    """
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    top_y = 0.50 * h if level == "full" else 0.62 * h
    mask_color = (150, 180, 210)  # light surgical blue
    poly = [(0.16 * w, top_y), (0.84 * w, top_y),
            (0.90 * w, 0.99 * h), (0.10 * w, 0.99 * h)]
    draw.polygon(poly, fill=mask_color)
    # ear straps
    draw.line([(0.16 * w, top_y), (0.02 * w, top_y - 0.04 * h)],
              fill=mask_color, width=max(2, int(0.02 * w)))
    draw.line([(0.84 * w, top_y), (0.98 * w, top_y - 0.04 * h)],
              fill=mask_color, width=max(2, int(0.02 * w)))
    return img


def _identity(img):
    return img


# ============================================================================
# 2. Corruption suite: name -> list of (severity_label, fn).
#    Add or trim severities here.
# ============================================================================
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


# ============================================================================
# 3. Model loading + batched inference
# ============================================================================
def load_model(model_dir, device):
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device).eval()
    return model, processor


def build_label_mapping(model, test_labels):
    """Map raw human_label ints to the model's class indices via config."""
    id2label = model.config.id2label                     # {idx: labelstr}
    label2idx = {int(v): int(k) for k, v in id2label.items()}
    missing = sorted(set(int(x) for x in test_labels) - set(label2idx))
    if missing:
        raise ValueError(f"Test labels {missing} are not in the model config "
                         f"({sorted(label2idx)}). Wrong checkpoint?")
    return label2idx


@torch.no_grad()
def evaluate(corrupt_fn, paths, y_true_idx, model, processor, device, batch_size):
    preds = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = [corrupt_fn(Image.open(p).convert("RGB")) for p in batch_paths]
        inputs = processor(imgs, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = accuracy_score(y_true_idx, preds)
    mf1 = f1_score(y_true_idx, preds, average="macro")
    return acc, mf1, preds


# ============================================================================
# 4. Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./vit_affectnet_results/best_vit_affectnet",
                    help="Directory with a saved HF image-classification model.")
    ap.add_argument("--test_csv", default="subash_affectnet_validation_filtered_labels.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate only the first N images (0 = all). For a quick smoke test.")
    ap.add_argument("--out_csv", default="robustness_results.csv")
    ap.add_argument("--out_preds", default="robustness_predictions.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from {args.model_dir} ...")
    model, processor = load_model(args.model_dir, device)

    df = pd.read_csv(args.test_csv)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    if args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)

    label2idx = build_label_mapping(model, df[LABEL_COLUMN].values)
    paths = df[IMAGE_COLUMN].tolist()
    y_true_idx = np.array([label2idx[int(l)] for l in df[LABEL_COLUMN].values])
    print(f"Evaluating on {len(paths)} images.\n")

    suite = build_suite()
    rows = []
    saved_preds = {"y_true_idx": y_true_idx}

    clean_acc = None
    for group, variants in suite.items():
        for sev_label, fn in variants:
            acc, mf1, preds = evaluate(fn, paths, y_true_idx, model,
                                       processor, device, args.batch_size)
            if group == "clean":
                clean_acc = acc
            drop = "" if clean_acc is None else f"{(clean_acc - acc) * 100:+.2f}"
            rows.append({
                "corruption": group,
                "severity": sev_label,
                "accuracy": round(acc, 4),
                "macro_f1": round(mf1, 4),
                "acc_drop_vs_clean_pct": drop,
            })
            saved_preds[f"{group}__{sev_label}"] = preds
            print(f"{group:>13} | {sev_label:<14} "
                  f"acc={acc:.4f}  macroF1={mf1:.4f}  drop={drop}")

    results = pd.DataFrame(rows)
    results.to_csv(args.out_csv, index=False)
    np.savez_compressed(args.out_preds, **saved_preds)

    print("\n================ Robustness summary ================")
    print(results.to_string(index=False))
    print(f"\nSaved table       -> {args.out_csv}")
    print(f"Saved predictions -> {args.out_preds}  "
          f"(per-sample preds for McNemar/significance tests later)")


if __name__ == "__main__":
    main()
