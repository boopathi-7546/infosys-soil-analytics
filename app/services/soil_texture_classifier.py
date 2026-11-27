"""Soil Texture Classification Service (Phase C).

Loads models/soil_texture_classifier/best_model.pth (separate from the 6-class
soil-type classifier) and predicts surface soil texture classes with confidence.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXTURE_MODEL_PATH = PROJECT_ROOT / "models" / "soil_texture_classifier" / "best_model.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TextureModelError(Exception):
    """Raised when the soil texture checkpoint is missing or unreadable."""


class InvalidTextureImageError(Exception):
    """Raised when image bytes cannot be decoded for texture prediction."""


class SoilTextureClassifier:
    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_TEXTURE_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.class_names: list[str] = []
        self.model_version = "soil_texture_efficientnet_b0_v1.0"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def load_model(self, force_reload: bool = False) -> None:
        if self.model is not None and not force_reload:
            return
        if not self.model_path.is_file():
            raise TextureModelError(f"Soil texture model checkpoint not found at {self.model_path}")
        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
            classes = ckpt.get("class_names", ["clay", "loam", "sand"])
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(classes))
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            self.class_names = list(classes)
        except Exception as exc:
            if isinstance(exc, TextureModelError):
                raise
            raise TextureModelError(f"Failed to load soil texture model: {exc}") from exc

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self.load_model()
        if not image_bytes:
            raise InvalidTextureImageError("Image bytes are empty.")
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise InvalidTextureImageError(f"Invalid or unreadable image data: {exc}") from exc

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return {
            "predicted_texture": self.class_names[idx],
            "confidence": round(conf, 4),
            "confidence_percentage": round(conf * 100, 2),
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
            "model_version": self.model_version,
            "model_maturity": "demonstration",
            "disclaimer": "Demonstration model: visual surface texture estimation from image only; distinct from 6-class soil type and not a laboratory sieve/hydrometer analysis.",
        }


_instance: Optional[SoilTextureClassifier] = None


def get_soil_texture_classifier() -> SoilTextureClassifier:
    global _instance
    if _instance is None:
        _instance = SoilTextureClassifier()
    return _instance

