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

import cv2
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

    def generate_gradcam(
        self,
        image_bytes: bytes,
        target_class: str | None = None,
        alpha: float = 0.45,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compute a Grad-CAM attention heatmap for EfficientNet-B0 and overlay it
        onto the original uploaded soil image without modifying model weights.
        """
        self.load_model()

        if not image_bytes:
            raise InvalidImagePredictionError("Image bytes are empty.")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning(f"Failed to decode image bytes for Grad-CAM: {exc}")
            raise InvalidImagePredictionError(f"Invalid or unreadable image data: {exc}") from exc

        if target_class is not None and target_class not in self.class_names:
            raise ValueError(f"Unknown target_class '{target_class}'. Valid classes: {self.class_names}")

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        target_layer = self.model.features[-1]

        def _forward_hook(_module: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            activations.append(out.detach())

        def _backward_hook(
            _module: nn.Module,
            _grad_in: Tuple[Optional[torch.Tensor], ...],
            grad_out: Tuple[Optional[torch.Tensor], ...],
        ) -> None:
            if grad_out and grad_out[0] is not None:
                gradients.append(grad_out[0].detach())

        fwd_handle = target_layer.register_forward_hook(_forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

        try:
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().squeeze(0).numpy()
            pred_idx = int(np.argmax(probs))
            target_idx = self.class_names.index(target_class) if target_class is not None else pred_idx

            score = logits[0, target_idx]
            score.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            act = activations[0].squeeze(0)  # (C, H, W)
            grad = gradients[0].squeeze(0)   # (C, H, W)

            weights = torch.mean(grad, dim=(1, 2))  # (C,)
            cam = torch.sum(weights[:, None, None] * act, dim=0)
            cam = torch.relu(cam).cpu().numpy()

            cam_max = float(np.max(cam))
            if cam_max > 1e-8:
                cam = cam / cam_max
            else:
                cam = np.zeros_like(cam, dtype=np.float32)

            orig_rgb = np.array(pil_image)
            orig_h, orig_w = orig_rgb.shape[:2]
            cam_resized = cv2.resize(cam.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            heatmap_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
            blend_alpha = float(np.clip(alpha, 0.0, 1.0))
            overlay_bgr = cv2.addWeighted(orig_bgr, 1.0 - blend_alpha, heatmap_bgr, blend_alpha, 0.0)

            ok, encoded = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise RuntimeError("Failed to encode Grad-CAM overlay image as JPEG.")

            return encoded.tobytes(), {
                "target_class": self.class_names[target_idx],
                "predicted_soil_type": self.class_names[pred_idx],
                "confidence_percentage": round(float(probs[pred_idx]) * 100, 2),
                "model_version": self.model_version,
            }
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            self.model.zero_grad(set_to_none=True)


# Global shared classifier instance
_classifier_instance: Optional[SoilClassifier] = None


def get_soil_classifier() -> SoilClassifier:
    """Return singleton SoilClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SoilClassifier()
    return _classifier_instance
