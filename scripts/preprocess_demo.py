"""Make a before/after montage of the preprocessing steps for one image (for your report/PPT).

Usage: python scripts/preprocess_demo.py path/to/soil.jpg docs/preprocessing_demo.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import preprocessing as pp  # noqa: E402


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(src, dst, size=300):
    img = cv2.imread(src)
    if img is None:
        sys.exit(f"Cannot read {src}")
    r = lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    cc = pp.normalize_illumination(pp.gray_world(img))
    bg = pp.remove_background(cc)
    aug = pp.augment(r(img), np.random.default_rng(1))
    panels = [label(r(img), "1 original"), label(r(cc), "2 colour + light corrected"),
              label(r(bg), "3 background removed"), label(r(pp.preprocess(img, size)), "4 final model input"),
              label(aug, "5 augmentation sample")]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, np.hstack(panels))
    print("Saved", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
