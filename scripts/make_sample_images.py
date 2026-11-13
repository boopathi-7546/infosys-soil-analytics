"""Generate SYNTHETIC soil-like images so the dataset pipeline can be smoke-tested.

These are NOT real soil photos and must never be used to train or evaluate the real model.
Usage: python scripts/make_sample_images.py --out data/raw_sample --per-class 12
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# rough BGR base colours per class, for pipeline testing only
CLASSES = {
    "black_soil": (35, 38, 40),
    "red_soil": (60, 80, 150),
    "alluvial_soil": (110, 135, 160),
    "sandy_soil": (150, 190, 215),
}


def make(base_bgr, rng, size=480):
    img = np.full((size, size, 3), base_bgr, np.float32)
    img += rng.normal(0, 18, (size, size, 1))                  # grain noise (same on all channels = brightness only)
    coarse = cv2.resize(rng.normal(0, 25, (12, 12)).astype(np.float32), (size, size))
    img += coarse[:, :, None]                                  # patchy lighting
    for _ in range(rng.integers(20, 60)):                      # pebbles / clods
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        r = int(rng.integers(2, 9))
        shade = float(rng.uniform(0.7, 1.3))
        cv2.circle(img, c, r, tuple(float(v) * shade for v in base_bgr), -1)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_sample")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name, color in CLASSES.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            cv2.imwrite(str(d / f"synthetic_{i:03d}.jpg"), make(color, rng))
    print(f"Wrote {a.per_class * len(CLASSES)} SYNTHETIC images to {a.out}")


if __name__ == "__main__":
    main()
