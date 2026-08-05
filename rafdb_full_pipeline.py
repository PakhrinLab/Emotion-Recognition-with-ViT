"""
=====================================================================
RAF-DB ViT: full pipeline that runs BOTH recipes and tests significance
=====================================================================

This single file merges your two scripts:

  * BASELINE  -- the recipe from cm_rafdb.py
                 (google/vit-base-patch16-224-in21k, 5 epochs, flip+crop,
                  plain cross-entropy, flat LR)
  * STRONG    -- the recipe from RAF_DB_vit_multiseed.py
                 (trpakov/vit-face-expression, 20 epochs, RandAugment +
                  RandomErasing, class-weighted + label-smoothed CE,
                  cosine LR + warmup, best-ckpt on macro-F1)

...and then, after both have run over the same seeds, it AUTOMATICALLY runs:

  LEVEL 1  paired t-test + Wilcoxon signed-rank on the per-seed test
           accuracies  ("is the recipe reliably better across seeds?")
  LEVEL 2  McNemar's test on the per-image predictions, matched seed by
           seed on the SAME fixed test set  ("do the two models disagree
           on individual images in a way that favours one?")

------------------------------------------------------------------
IMPORTANT METHODOLOGY NOTE (a real improvement over cm_rafdb.py)
------------------------------------------------------------------
Your original baseline selected the best checkpoint on the TEST set
(eval_dataset = test) -- that is a data leak: you must not pick the model
using the same set you report on. This pipeline fixes that: BOTH recipes
carve a fresh stratified 90/10 validation split from train each seed,
select the checkpoint on that VALIDATION slice, and touch the test set
only once for the final numbers. So the two recipes differ ONLY in the
recipe knobs, which is exactly what a fair comparison requires.

No model weights are kept: checkpoints go to node-local scratch and are
deleted after each seed. Only results (.npy preds/labels/logits, .json
reports, aggregate summary) are written to RESULTS_DIR.

Launch (single GPU, single process -- np.save is not DDP-guarded):
    python rafdb_full_pipeline.py
or:
    accelerate launch --num_processes=1 --mixed_precision=fp16 rafdb_full_pipeline.py
"""

import os
import gc
import json
import shutil
import tempfile

import numpy as np
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
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandAugment,
    RandomErasing,
    ToTensor,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats

# =====================================================================
# 0. Global experiment configuration
# =====================================================================
SEEDS = [42, 1337, 2024, 7, 31415]     # same seeds for both recipes -> paired tests
VAL_FRACTION = 0.10                     # carved from train each seed, for ckpt selection
RESULTS_DIR = "./rafdb_full_pipeline_results"

DATA_DIR = "/project/pakhrin/subash/DATASET"   # imagefolder: train/<class>/*, test/<class>/*
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

SCRATCH_ROOT = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or tempfile.gettempdir()
)

# ---- The two recipes as plain config dicts. Add a third here to extend. ----
RECIPES = {
    "baseline": {
        "backbone": "google/vit-base-patch16-224-in21k",
        "epochs": 5,
        "lr": 2e-5,
        "warmup_ratio": 0.0,
        "scheduler": "linear",
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "use_class_weights": False,
        "strong_aug": False,
        "train_bs": 16,
        "eval_bs": 16,
        "metric_for_best_model": "accuracy",
    },
    "strong": {
        "backbone": "trpakov/vit-face-expression",
        "epochs": 20,
        "lr": 3e-5,
        "warmup_ratio": 0.1,
        "scheduler": "cosine",
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "use_class_weights": True,
        "strong_aug": True,
        "train_bs": 32,
        "eval_bs": 32,
        "metric_for_best_model": "macro_f1",
    },
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# 1. Load data once (test set order is FIXED -> McNemar pairs line up)
# =====================================================================
print("Loading dataset from directories...")
dataset = load_dataset(
    "imagefolder",
    data_files={
        "train": os.path.join(TRAIN_DIR, "**"),
        "test": os.path.join(TEST_DIR, "**"),
    },
)
labels = dataset["train"].features["label"].names
num_labels = len(labels)
id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}
print(f"Found {num_labels} classes: {labels}")


# =====================================================================
# 2. Transforms, metrics, custom trainer
# =====================================================================
def build_transforms(image_processor, strong):
    mean, std = image_processor.image_mean, image_processor.image_std
    size = image_processor.size["height"]
    if strong:
        return Compose([
            RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=7),
            ToTensor(),
            Normalize(mean=mean, std=std),
            RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    # baseline: just flip + resized-crop, matching cm_rafdb.py
    return Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "macro_f1": f1_score(label_ids, preds, average="macro"),
    }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


class WeightedSmoothTrainer(Trainer):
    """Class-weighted + label-smoothed CE. With defaults it is plain CE,
    so the SAME trainer class serves both recipes."""
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights
        self._ls = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._cw.to(logits.device) if self._cw is not None else None
        loss = F.cross_entropy(logits, labels_, weight=w, label_smoothing=self._ls)
        return (loss, outputs) if return_outputs else loss


def print_confusion_matrix(cm, class_names, title):
    print(f"\n{title}")
    col_w = max(6, max(len(c) for c in class_names) + 1)
    print(" " * col_w + "".join(f"{c:>{col_w}}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>{col_w}}" + "".join(f"{int(v):>{col_w}}" for v in row))


# =====================================================================
# 3. One (recipe, seed) run
# =====================================================================
def run_one(recipe_name, cfg, seed):
    print("\n" + "=" * 70)
    print(f"RECIPE = {recipe_name.upper()}   |   SEED = {seed}")
    print("=" * 70)

    set_seed(seed)
    ckpt_dir = os.path.join(SCRATCH_ROOT, f"rafdb_{recipe_name}_seed{seed}")

    image_processor = AutoImageProcessor.from_pretrained(cfg["backbone"], use_fast=True)
    train_tf = build_transforms(image_processor, cfg["strong_aug"])

    def preprocess_train(batch):
        batch["pixel_values"] = [train_tf(im.convert("RGB")) for im in batch["image"]]
        return batch

    def preprocess_eval(batch):
        imgs = [im.convert("RGB") for im in batch["image"]]
        batch["pixel_values"] = image_processor(imgs, return_tensors="pt").pixel_values
        return batch

    # Fair protocol for BOTH recipes: stratified 90/10 val split, test untouched.
    split = dataset["train"].train_test_split(
        test_size=VAL_FRACTION, stratify_by_column="label", seed=seed
    )
    train_ds = split["train"].with_transform(preprocess_train)
    val_ds = split["test"].with_transform(preprocess_eval)
    test_ds = dataset["test"].with_transform(preprocess_eval)

    class_weights_t = None
    if cfg["use_class_weights"]:
        y = np.array(split["train"]["label"])
        cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=y)
        class_weights_t = torch.tensor(cw, dtype=torch.float)

    model = AutoModelForImageClassification.from_pretrained(
        cfg["backbone"], num_labels=num_labels,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=ckpt_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["scheduler"],
        per_device_train_batch_size=cfg["train_bs"],
        per_device_eval_batch_size=cfg["eval_bs"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    trainer = WeightedSmoothTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,   # selection on VAL, not test
        data_collator=collate_fn, compute_metrics=compute_metrics,
        class_weights=class_weights_t, label_smoothing=cfg["label_smoothing"],
    )
    trainer.train()

    # ---- final, single evaluation on the untouched test set ----
    out = trainer.predict(test_ds)
    logits = out.predictions
    preds = np.argmax(logits, axis=1)
    true = out.label_ids

    acc = accuracy_score(true, preds)
    mf1 = f1_score(true, preds, average="macro")
    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds, target_names=labels,
                                   output_dict=True, zero_division=0)

    print(f"\n[{recipe_name} seed {seed}] test acc {acc:.4f} | macro-F1 {mf1:.4f}")
    print(classification_report(true, preds, target_names=labels, zero_division=0))
    print_confusion_matrix(cm, labels, f"[{recipe_name} seed {seed}] confusion matrix")

    # ---- save results (NOT weights) -- filenames encode recipe + seed ----
    tag = f"{recipe_name}_seed{seed}"
    np.save(os.path.join(RESULTS_DIR, f"logits_{tag}.npy"), logits)
    np.save(os.path.join(RESULTS_DIR, f"preds_{tag}.npy"), preds)
    np.save(os.path.join(RESULTS_DIR, f"labels_{tag}.npy"), true)
    np.save(os.path.join(RESULTS_DIR, f"confmat_{tag}.npy"), cm)
    with open(os.path.join(RESULTS_DIR, f"report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)

    del trainer, model, out
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {"recipe": recipe_name, "seed": seed, "accuracy": acc,
            "macro_f1": mf1, "report": report, "confusion_matrix": cm,
            "preds": preds, "labels": true}


# =====================================================================
# 4. Run every recipe over every seed
# =====================================================================
results = {name: [] for name in RECIPES}
for name, cfg in RECIPES.items():
    for s in SEEDS:
        results[name].append(run_one(name, cfg, s))


def summarize(runs):
    acc = np.array([r["accuracy"] for r in runs]) * 100
    mf1 = np.array([r["macro_f1"] for r in runs]) * 100
    return acc, mf1


print("\n" + "#" * 70)
print("AGGREGATE RESULTS")
print("#" * 70)
agg = {}
for name in RECIPES:
    acc, mf1 = summarize(results[name])
    agg[name] = {"acc": acc, "mf1": mf1}
    print(f"\n{name.upper()}")
    for r, a, m in zip(results[name], acc, mf1):
        print(f"  seed {r['seed']:>6}:  acc {a:.2f} %   macro-F1 {m:.2f} %")
    print(f"  --> acc {acc.mean():.2f} +/- {acc.std(ddof=1):.2f} % | "
          f"macro-F1 {mf1.mean():.2f} +/- {mf1.std(ddof=1):.2f} %")


# =====================================================================
# 5. LEVEL 1 -- seed-level paired significance tests
# =====================================================================
def paired_report(a_strong, a_base, label):
    diff = a_strong - a_base
    t, p_t = stats.ttest_rel(a_strong, a_base)
    try:
        w, p_w = stats.wilcoxon(a_strong, a_base)
    except ValueError:      # all diffs identical / zero
        w, p_w = float("nan"), float("nan")
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("inf")
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    tcrit = stats.t.ppf(0.975, len(diff) - 1)
    lo, hi = diff.mean() - tcrit * sem, diff.mean() + tcrit * sem
    print(f"\n{label}")
    print(f"  mean diff (strong - baseline): {diff.mean():+.2f} "
          f"(95% CI [{lo:.2f}, {hi:.2f}])")
    print(f"  paired t-test:        t={t:.3f}, p={p_t:.6g}")
    print(f"  Wilcoxon signed-rank: W={w}, p={p_w:.6g}  "
          f"(floor at n={len(diff)} is {0.5**len(diff)*2:.5f})")
    print(f"  Cohen's d (paired):   {d:.2f}")
    return {"mean_diff": float(diff.mean()), "ci": [float(lo), float(hi)],
            "t": float(t), "p_t": float(p_t),
            "W": (None if np.isnan(w) else float(w)),
            "p_wilcoxon": (None if np.isnan(p_w) else float(p_w)),
            "cohens_d": float(d)}

print("\n" + "=" * 70)
print("LEVEL 1: seed-level paired tests  (strong vs baseline)")
print("=" * 70)
lvl1_acc = paired_report(agg["strong"]["acc"], agg["baseline"]["acc"], "ACCURACY")
lvl1_mf1 = paired_report(agg["strong"]["mf1"], agg["baseline"]["mf1"], "MACRO-F1")


# =====================================================================
# 6. LEVEL 2 -- McNemar on per-image predictions (matched by seed)
# =====================================================================
def mcnemar(base_preds, strong_preds, y):
    base_ok = (base_preds == y)
    strong_ok = (strong_preds == y)
    b = int(np.sum(base_ok & ~strong_ok))   # base right, strong wrong
    c = int(np.sum(~base_ok & strong_ok))   # base wrong, strong right
    n = b + c
    if n == 0:
        return b, c, float("nan"), "undefined"
    if n < 25:
        p = stats.binomtest(min(b, c), n, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = stats.chi2.sf(chi2, 1)
    favors = "strong" if c > b else ("baseline" if b > c else "tie")
    return b, c, p, favors

print("\n" + "=" * 70)
print("LEVEL 2: McNemar per matched seed  (same fixed test set)")
print("=" * 70)
lvl2 = []
for rb, rs in zip(results["baseline"], results["strong"]):
    assert rb["seed"] == rs["seed"]
    assert np.array_equal(rb["labels"], rs["labels"]), "test order mismatch!"
    b, c, p, favors = mcnemar(rb["preds"], rs["preds"], rb["labels"])
    print(f"  seed {rb['seed']:>6}: b(base-only right)={b:>4}  "
          f"c(strong-only right)={c:>4}  p={p:.3g}  favors={favors}")
    lvl2.append({"seed": rb["seed"], "b": b, "c": c, "p": float(p), "favors": favors})

# Pooled McNemar across all seeds (5x the test set): maximum power
all_b = np.concatenate([r["preds"] for r in results["baseline"]])
all_s = np.concatenate([r["preds"] for r in results["strong"]])
all_y = np.concatenate([r["labels"] for r in results["baseline"]])
pb, pc, pp, pfav = mcnemar(all_b, all_s, all_y)
print(f"\n  POOLED (all seeds): b={pb}  c={pc}  p={pp:.3g}  favors={pfav}")


# =====================================================================
# 7. Save the significance summary
# =====================================================================
significance = {
    "recipes": {n: {"acc_mean": float(agg[n]["acc"].mean()),
                    "acc_std": float(agg[n]["acc"].std(ddof=1)),
                    "mf1_mean": float(agg[n]["mf1"].mean()),
                    "mf1_std": float(agg[n]["mf1"].std(ddof=1))}
                for n in RECIPES},
    "level1_accuracy": lvl1_acc,
    "level1_macro_f1": lvl1_mf1,
    "level2_mcnemar_per_seed": lvl2,
    "level2_mcnemar_pooled": {"b": pb, "c": pc, "p": float(pp), "favors": pfav},
}
with open(os.path.join(RESULTS_DIR, "significance_summary.json"), "w") as f:
    json.dump(significance, f, indent=2)

print(f"\nAll results + significance_summary.json saved to "
      f"{os.path.abspath(RESULTS_DIR)}/")
