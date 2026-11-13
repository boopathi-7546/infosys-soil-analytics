"""OpenCV preprocessing for the CNN: colour correction, background removal, resize,
noise reduction, and quality assessment.

All functions operate on BGR uint8 numpy arrays (OpenCV convention).
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import config

DATA_PROCESSED_DIR = Path("data/processed")


def decode_bgr(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


def gray_world(img: np.ndarray, strength: float = 0.25, max_gain: float = 0.15) -> np.ndarray:
    """Gentle colour-cast correction (partial gray-world).

    Full gray-world would turn a red soil grey, destroying the colour signal the model needs to
    tell soil types apart. The correction is blended (strength) and each channel gain is clamped.
    """
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    gain = np.clip(1.0 + strength * (gain - 1.0), 1.0 - max_gain, 1.0 + max_gain)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel only: evens out shadows/highlights, leaves hue untouched."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def remove_background(img: np.ndarray) -> np.ndarray:
    """GrabCut with a centre-weighted rectangle. Background pixels become black.
    If segmentation fails or keeps too little of the image, the input is returned unchanged."""
    h, w = img.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        scale = 256 / max(h, w)
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        sm = np.zeros(small.shape[:2], np.uint8)
        sh, sw = small.shape[:2]
        srect = (int(sw * 0.05), int(sh * 0.05), int(sw * 0.90), int(sh * 0.90))
        cv2.grabCut(small, sm, srect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_small = np.where((sm == cv2.GC_FGD) | (sm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        return img
    if (fg > 0).mean() < 0.35:
        return img
    return cv2.bitwise_and(img, img, mask=fg)


def reduce_noise(img: np.ndarray) -> np.ndarray:
    """Gentle Gaussian noise filtering before CNN feature extraction."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def preprocess(
    img: np.ndarray,
    size: int = config.MODEL_INPUT_SIZE,
    bg_removal: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline:
    colour balance -> CLAHE illumination -> noise reduction -> (optional bg removal) -> resize.
    """
    out = gray_world(img)
    out = normalize_illumination(out)
    out = reduce_noise(out)
    if bg_removal:
        out = remove_background(out)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def inspect_and_preprocess(img_bgr: np.ndarray, image_id: int) -> dict[str, Any]:
    """Evaluates image quality, applies preprocessing, saves output artifact,
    and returns normalization statistics.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    quality_ok = True
    rejection_reason = None

    if mean_val < 30.0:
        quality_ok = False
        rejection_reason = f"Image too dark (mean brightness {mean_val:.1f} < 30)"
    elif blur_score < 40.0:
        quality_ok = False
        rejection_reason = f"Image severely blurred (variance {blur_score:.1f} < 40)"

    processed = preprocess(img_bgr, size=config.MODEL_INPUT_SIZE, bg_removal=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_PROCESSED_DIR / f"{image_id}_proc.jpg"
    cv2.imwrite(str(out_file), processed, [cv2.IMWRITE_JPEG_QUALITY, 92])

    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_rgb = [round(float(rgb[:, :, c].mean()), 4) for c in range(3)]

    return {
        "processed_path": str(out_file).replace("\\", "/"),
        "width": config.MODEL_INPUT_SIZE,
        "height": config.MODEL_INPUT_SIZE,
        "mean_rgb": mean_rgb,
        "quality_ok": quality_ok,
        "rejection_reason": rejection_reason,
    }


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Feature-preserving augmentation (training only).
    Geometry and lighting changes only; no hue shifts because soil colour is critical.
    """
    h, w = img.shape[:2]
    out = img

    angle = float(rng.uniform(-25, 25))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, float(rng.uniform(1.0, 1.15)))
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)

    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-20, 20))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.3:
        noise = rng.normal(0, 4, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
