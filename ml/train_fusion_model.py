"""Hybrid Image + Structured Soil-Test Fusion Architecture & Training Pipeline (Phase G).

Combines:
  - Image branch: Frozen EfficientNet-B0 1280-D pooled feature vector
  - Tabular branch: 2-layer MLP encoder over standardized soil-test features
  - Fusion head: Concatenated [1280 + 64] -> Linear(1344, 128) -> ReLU -> Dropout -> Linear(128, num_classes)

Training requires real paired (soil_image, soil_test, label) records.
Does NOT fabricate pairings or claim trained accuracy without real paired data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "models" / "hybrid_fusion"

TABULAR_FEATURES = ["nitrogen", "phosphorus", "potassium", "ph", "organic_matter", "moisture"]


class InsufficientPairedDataError(RuntimeError):
    """Raised when real paired (image, soil_test) records are insufficient for training."""


class HybridSoilFusionNet(nn.Module):
    """Multimodal architecture fusing EfficientNet-B0 1280-D embeddings with tabular soil features."""

    def __init__(self, num_tabular_features: int = len(TABULAR_FEATURES), num_classes: int = 4):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.image_features = backbone.features
        self.image_pool = backbone.avgpool
        for p in self.image_features.parameters():
            p.requires_grad = False

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, tabular_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_map = self.image_features(image_tensor)
            img_emb = torch.flatten(self.image_pool(feat_map), 1)
        tab_emb = self.tabular_encoder(tabular_tensor)
        fused = torch.cat([img_emb, tab_emb], dim=1)
        return self.fusion_head(fused)


@dataclass
class PairedSample:
    image_path: Path
    tabular_values: list[float]
    label: int
    field_id: int


def validate_paired_dataset(samples: list[PairedSample], min_required: int = 50) -> None:
    """Validate that enough finite, same-field paired samples exist before training."""
    valid_samples = [
        s for s in samples
        if len(s.tabular_values) == len(TABULAR_FEATURES)
        and all(v is not None and math.isfinite(float(v)) for v in s.tabular_values)
        and s.image_path.is_file()
    ]
    if len(valid_samples) < min_required:
        raise InsufficientPairedDataError(
            f"Hybrid fusion training blocked: found {len(valid_samples)} valid real paired (image, soil_test) "
            f"samples with ground-truth labels, but at least {min_required} are required."
        )


def extract_real_paired_samples(db, upload_dir: Path, label_by_soil_test_id: dict[int, int]) -> list[PairedSample]:
    """Extract only genuine same-field paired (SoilImage, SoilTest) rows with verified labels."""
    from sqlalchemy import select
    from app.models import SoilImage, SoilTest

    rows = db.execute(
        select(SoilImage, SoilTest)
        .join(SoilTest, SoilImage.soil_test_id == SoilTest.id)
        .where(SoilImage.field_id == SoilTest.field_id)
    ).all()

    paired: list[PairedSample] = []
    for img, test in rows:
        if test.id not in label_by_soil_test_id:
            continue
        vals = [getattr(test, feat, None) for feat in TABULAR_FEATURES]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            continue
        img_path = upload_dir / img.file_path
        if not img_path.is_file():
            continue
        paired.append(
            PairedSample(
                image_path=img_path,
                tabular_values=[float(v) for v in vals],
                label=int(label_by_soil_test_id[test.id]),
                field_id=int(test.field_id),
            )
        )
    return paired


def save_architecture_metadata() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "metadata.json"
    meta = {
        "model_name": "hybrid_fusion",
        "model_version": "hybrid_fusion_v1.0",
        "architecture": "EfficientNet-B0 (1280-D frozen backbone) + Tabular MLP Encoder -> Fusion Head",
        "status": "architecture_ready_untrained",
        "tabular_features": TABULAR_FEATURES,
        "dataset": "None (blocked: 0 paired image + soil_test records with ground-truth labels)",
        "metrics": {},
        "limitations": "Training requires real co-located (image, soil_test) pairs with verified ground-truth labels.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


if __name__ == "__main__":
    save_architecture_metadata()
    validate_paired_dataset([])

