"""
SoilSense AI - CNN Soil Image Prediction Service
================================================
Loads the trained EfficientNet-B0 model from models/soil_classifier/best_model.pth.
Performs deterministic evaluation preprocessing (Resize 224x224, ImageNet Normalization)
and computes class probabilities across all 6 soil categories:
  - alluvial_soil
  - black_soil
  - clay_soil
  - laterite_soil
  - red_soil
  - sandy_soil
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

logger = logging.getLogger("soil_classifier_service")

# Class ordering matching models/soil_classifier/class_names.json and checkpoint
DEFAULT_CLASSES: List[str] = [
    "alluvial_soil",
    "black_soil",
    "clay_soil",
    "laterite_soil",
    "red_soil",
    "sandy_soil",
]

# ImageNet normalization statistics matching training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "soil_classifier" / "best_model.pth"


class SoilModelError(Exception):
    """Raised when the model weights file cannot be found or loaded."""
    pass


class InvalidImagePredictionError(Exception):
    """Raised when image bytes cannot be decoded for CNN prediction."""
    pass


class SoilClassifier:
    """Singleton-style model wrapper for fast live inference."""

    def __init__(self, model_path: Path | str | None = None, device: Optional[torch.device] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = DEFAULT_CLASSES
        self.model_version: str = "efficientnet_b0_v1.0"
        self.best_epoch: int = 7

        # Deterministic evaluation transform (strictly matches ml/train_cnn.py eval_transform)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_model(self, force_reload: bool = False) -> None:
        """Load checkpoint and instantiate EfficientNet-B0 architecture."""
        if self.model is not None and not force_reload:
            return

        if not self.model_path.is_file():
            logger.error(f"Soil classification model file not found: {self.model_path}")
            raise SoilModelError(f"Model file not found at {self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # EfficientNet-B0 base
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            num_classes = len(checkpoint.get("class_names", DEFAULT_CLASSES))
            model.classifier[1] = nn.Linear(in_features, num_classes)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.model = model
            self.class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
            self.best_epoch = checkpoint.get("epoch", 7)
            logger.info(
                f"Successfully loaded EfficientNet-B0 soil classifier from {self.model_path} "
                f"(Epoch {self.best_epoch}, Device: {self.device})"
            )
        except Exception as exc:
            if isinstance(exc, SoilModelError):
                raise
            logger.exception(f"Error loading soil classifier weights: {exc}")
            raise SoilModelError(f"Failed to load soil classification model: {exc}") from exc

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run CNN prediction on raw image bytes.
        Returns:
            - predicted_soil_type (str)
            - confidence (float between 0 and 1)
            - confidence_percentage (float, e.g. 96.34)
            - probabilities (Dict[str, float] for all 6 classes)
            - model_version (str)
            - best_epoch (int)
        """
        self.load_model()

        if not image_bytes:
            raise InvalidImagePredictionError("Image bytes are empty.")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGB")
        except Exception as exc:
            logger.warning(f"Failed to decode image bytes: {exc}")
            raise InvalidImagePredictionError(f"Invalid or unreadable image data: {exc}") from exc

        try:
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

            pred_idx = int(np.argmax(probs))
            predicted_class = self.class_names[pred_idx]
            confidence = float(probs[pred_idx])
            confidence_pct = round(confidence * 100, 2)

            all_probs = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self.class_names, probs)
            }

            return {
                "predicted_soil_type": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percentage": confidence_pct,
                "probabilities": all_probs,
                "model_version": self.model_version,
                "best_epoch": self.best_epoch,
            }
        except Exception as exc:
            if isinstance(exc, InvalidImagePredictionError):
                raise
            logger.exception(f"Prediction inference failure: {exc}")
            raise RuntimeError(f"CNN prediction failure: {exc}") from exc


# Global shared classifier instance
_classifier_instance: Optional[SoilClassifier] = None


def get_soil_classifier() -> SoilClassifier:
    """Return singleton SoilClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SoilClassifier()
    return _classifier_instance
