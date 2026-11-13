"""Turn a folder of collected soil images into a clean, split, preprocessed dataset.

Expected input layout (one folder per class label):
    data/raw/<class_name>/*.jpg|png

What it does
    1. drops corrupt / tiny files and exact duplicates (SHA-256)
    2. stratified train/val/test split per class (split BEFORE augmenting, so augmented
       copies of one photo can never leak into val/test)
    3. runs the OpenCV preprocessing (colour correction, background removal, resize)
    4. optionally writes N augmented copies of each TRAIN image
    5. writes data/processed/manifest.csv and prints class balance

Usage
    python scripts/prepare_dataset.py --raw data/raw --out data/processed --augment 3
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import preprocessing  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(raw: Path, min_side: int):
    """Return {class: [(path, sha)]} after removing corrupt/small/duplicate files."""
    seen, data, dropped = set(), {}, Counter()
    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                dropped["corrupt"] += 1
                continue
            if min(img.shape[:2]) < min_side:
                dropped["too_small"] += 1
                continue
            h = sha256(p)
            if h in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(h)
            data.setdefault(cls_dir.name, []).append((p, h))
    return data, dropped


def split_counts(n, ratios):
    tr, va, _ = ratios
    n_tr, n_va = int(n * tr), round(n * va)
    if n >= 3:  # guarantee at least one val and one test image
        n_va = max(n_va, 1)
        n_tr = min(n_tr, n - n_va - 1)
    return n_tr, n_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--augment", type=int, default=0, help="augmented copies per train image")
    ap.add_argument("--min-side", type=int, default=100)
    ap.add_argument("--no-bg-removal", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw, out = Path(a.raw), Path(a.out)
    if not raw.is_dir():
        sys.exit(f"Raw dataset folder not found: {raw}")
    rng = np.random.default_rng(a.seed)

    data, dropped = collect(raw, a.min_side)
    if not data:
        sys.exit("No valid images found. Put images in data/raw/<class_name>/")

    rows = []
    for cls, items in data.items():
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        n_tr, n_va = split_counts(len(items), a.split)
        parts = {"train": items[:n_tr], "val": items[n_tr:n_tr + n_va], "test": items[n_tr + n_va:]}
        for split, group in parts.items():
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for p, h in group:
                base = preprocessing.preprocess(cv2.imread(str(p)), a.size, not a.no_bg_removal)
                name = f"{h[:12]}.jpg"
                cv2.imwrite(str(dest / name), base)
                rows.append([str((dest / name).as_posix()), cls, split, h, str(p.as_posix()), "original"])
                if split == "train":
                    for k in range(a.augment):
                        aug = preprocessing.augment(base, rng)
                        an = f"{h[:12]}_aug{k}.jpg"
                        cv2.imwrite(str(dest / an), aug)
                        rows.append([str((dest / an).as_posix()), cls, split, h, str(p.as_posix()), "augmented"])

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "split", "source_sha256", "source_file", "kind"])
        w.writerows(rows)

    print(f"Dropped: {dict(dropped) or 'none'}")
    print(f"{'class':<20}{'train':>7}{'val':>6}{'test':>6}   (originals only)")
    for cls in data:
        c = Counter(r[2] for r in rows if r[1] == cls and r[5] == "original")
        flag = "  <-- few images, collect more" if sum(c.values()) < 100 else ""
        print(f"{cls:<20}{c['train']:>7}{c['val']:>6}{c['test']:>6}{flag}")
    print(f"Manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
