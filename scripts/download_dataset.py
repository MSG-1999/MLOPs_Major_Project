"""
download_dataset.py
===================
Downloads IndianFestivals dataset directly from HuggingFace,
samples 500 images per class, and splits into 80% train / 20% val.

Output structure:
    dataset/
        train/
            Christmas/   *.png
            Diwali/      *.png
            Durga_Puja/  *.png
            Eid/         *.png
            Holi/        *.png
            Navratri/    *.png
            Onam/        *.png
        val/
            Christmas/   *.png
            ...

Run:
    python download_dataset.py
"""

import os
import random
from pathlib import Path
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────
DATASET_ID  = "AIMLOps-C4-G16/IndianFestivals"
OUTPUT_DIR  = Path("dataset")
PER_CLASS   = 500
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.20
SEED        = 42
# ─────────────────────────────────────────────────────────────────────

random.seed(SEED)


def main():
    from datasets import load_dataset

    print("\n" + "="*60)
    print(f"  Dataset   : {DATASET_ID}")
    print(f"  Output    : {OUTPUT_DIR}/")
    print(f"  Per class : {PER_CLASS} images")
    print(f"  Split     : {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val")
    print("="*60 + "\n")

    # ── 1. Download from HuggingFace ──────────────────────────────────
    print("⬇  Downloading dataset from HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train")   # full dataset is in 'train' split
    print(f"   Total rows: {len(ds):,}")

    # Get class names from dataset features
    class_names = ds.features["label"].names
    print(f"   Classes ({len(class_names)}): {class_names}\n")

    # ── 2. Group indices by class ─────────────────────────────────────
    print("🔍  Grouping images by class...")
    class_indices = defaultdict(list)
    for idx, label in enumerate(ds["label"]):
        class_indices[label].append(idx)

    for label_id, cls_name in enumerate(class_names):
        count = len(class_indices[label_id])
        print(f"   {cls_name:<20} {count:>5,} images available")

    # ── 3. Sample & split ─────────────────────────────────────────────
    print(f"\n✂  Sampling {PER_CLASS} per class and splitting {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}...\n")

    summary = []

    for label_id, cls_name in enumerate(class_names):
        indices = class_indices[label_id]
        random.shuffle(indices)

        # Cap at PER_CLASS
        selected = indices[:PER_CLASS]
        n_total  = len(selected)
        n_train  = round(n_total * TRAIN_RATIO)
        n_val    = n_total - n_train

        train_indices = selected[:n_train]
        val_indices   = selected[n_train:]

        # Use underscore for folder names (avoids spaces in paths)
        folder_name = cls_name.replace(" ", "_")

        train_dir = OUTPUT_DIR / "train" / folder_name
        val_dir   = OUTPUT_DIR / "val"   / folder_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Save train images
        for i, idx in enumerate(train_indices):
            img = ds[idx]["image"]
            img.save(train_dir / f"{folder_name}_{i:04d}.png")
            if (i + 1) % 100 == 0:
                print(f"   [{cls_name}] train: {i+1}/{n_train} saved...", flush=True)

        # Save val images
        for i, idx in enumerate(val_indices):
            img = ds[idx]["image"]
            img.save(val_dir / f"{folder_name}_{i:04d}.png")

        summary.append((cls_name, len(indices), n_total, n_train, n_val))
        print(f"   ✅ {cls_name:<20} → train: {n_train}  val: {n_val}")

    # ── 4. Print final summary ────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  {'Class':<20} {'Available':>10} {'Sampled':>8} {'Train':>7} {'Val':>6}")
    print("-"*60)
    total_sampled = total_train = total_val = 0
    for cls_name, avail, sampled, tr, v in summary:
        flag = " ⚠ (< 500)" if avail < PER_CLASS else ""
        print(f"  {cls_name:<20} {avail:>10,} {sampled:>8,} {tr:>7,} {v:>6,}{flag}")
        total_sampled += sampled
        total_train   += tr
        total_val     += v
    print("-"*60)
    print(f"  {'TOTAL':<20} {'':>10} {total_sampled:>8,} {total_train:>7,} {total_val:>6,}")
    print("="*60)

    print(f"\n✅ Done! Dataset saved to: {OUTPUT_DIR.resolve()}")
    print(f"   dataset/train/  → {total_train:,} images")
    print(f"   dataset/val/    → {total_val:,} images\n")


if __name__ == "__main__":
    main()