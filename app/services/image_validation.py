"""Server-side validation + normalisation of uploaded soil photos.

Never trust the filename or the Content-Type header: the file is actually decoded
with Pillow, so renamed text files, truncated files and decompression bombs are rejected.
"""
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .. import config

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class ImageValidationError(ValueError):
    """Raised when an upload must be rejected. The message is safe to show to the user."""


@dataclass
class ValidatedImage:
    data: bytes                 # normalised JPEG bytes (RGB, EXIF stripped, size-capped)
    width: int
    height: int
    original_format: str
    original_size_bytes: int
    blur_score: float
    warnings: list[str] = field(default_factory=list)


def validate_and_normalize(raw: bytes, declared_type: str | None) -> ValidatedImage:
    if not raw:
        raise ImageValidationError("Empty file.")
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageValidationError(f"File too large (max {config.MAX_UPLOAD_MB} MB).")
    if declared_type not in config.ALLOWED_CONTENT_TYPES:
        raise ImageValidationError("Unsupported file type. Upload a JPEG or PNG image.")

    # 1) integrity check, 2) real decode
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Image.DecompressionBombError:
        raise ImageValidationError("Image dimensions are too large.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError("File is not a valid or complete image.")

    original_format = img.format or ""
    if original_format not in config.ALLOWED_FORMATS:
        raise ImageValidationError("Unsupported image format. Use JPEG or PNG.")

    img = ImageOps.exif_transpose(img)  # respect phone-camera orientation

    # flatten transparency onto white, normalise to RGB
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    if min(img.size) < config.MIN_SIDE_PX:
        raise ImageValidationError(
            f"Image too small ({img.width}x{img.height}). Minimum side is {config.MIN_SIDE_PX}px."
        )

    if max(img.size) > config.MAX_SIDE_PX:
        img.thumbnail((config.MAX_SIDE_PX, config.MAX_SIDE_PX), Image.LANCZOS)

    # quality hints (warnings only, the upload is still accepted)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    warnings = []
    if blur_score < config.BLUR_THRESHOLD:
        warnings.append("Image looks blurry. Hold the phone steady and retake for better results.")
    if brightness < 40:
        warnings.append("Image is very dark. Retake in daylight.")
    elif brightness > 220:
        warnings.append("Image is overexposed. Avoid direct harsh sunlight.")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.JPEG_QUALITY, optimize=True)  # EXIF/GPS dropped
    return ValidatedImage(
        data=buf.getvalue(),
        width=img.width,
        height=img.height,
        original_format=original_format,
        original_size_bytes=len(raw),
        blur_score=round(blur_score, 2),
        warnings=warnings,
    )
