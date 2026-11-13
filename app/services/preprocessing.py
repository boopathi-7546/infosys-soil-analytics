"""OpenCV preprocessing for the CNN: colour correction, background removal, resize,
plus feature-preserving augmentation for the training pipeline.

All functions work on BGR uint8 numpy arrays (OpenCV convention).
"""
import cv2
import numpy as np

from .. import config


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
    tell soil types apart. So the correction is blended (strength) and each channel gain is
    clamped to 1 +/- max_gain: it only removes mild lighting casts (e.g. warm evening light).
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
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def remove_background(img: np.ndarray) -> np.ndarray:
    """GrabCut with a centre-weighted rectangle. Background pixels become black.
    If segmentation fails or keeps too little of the image, the input is returned unchanged
    (soil photos are often fully filled with soil, so this is a safe fallback)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        # GrabCut is slow on big images, so run it on a small copy and upscale the mask
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


def preprocess(img: np.ndarray, size: int = config.MODEL_INPUT_SIZE, bg_removal: bool = True) -> np.ndarray:
    """Full inference/training preprocessing:
    mild colour correction -> illumination normalisation -> (background removal) -> resize."""
    out = normalize_illumination(gray_world(img))
    if bg_removal:
        out = remove_background(out)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Feature-preserving augmentation (training only). Geometry and mild lighting changes only:
    no hue shifts, because soil colour is a key signal for soil type."""
    h, w = img.shape[:2]
    out = img

    angle = float(rng.uniform(-25, 25))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, float(rng.uniform(1.0, 1.15)))
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)

    alpha = float(rng.uniform(0.85, 1.15))   # contrast
    beta = float(rng.uniform(-20, 20))       # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.3:
        noise = rng.normal(0, 4, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
