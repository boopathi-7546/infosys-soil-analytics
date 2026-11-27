"""Central Model Registry for SoilSense AI.

Tracks all ML models, versions, architectures, datasets, training dates,
measured evaluation metrics, checkpoint paths, input schemas, and limitations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_model_registry() -> dict[str, dict[str, Any]]:
    """Return live metadata and readiness status for all SoilSense AI models."""
    soil_metrics = _read_json(MODELS_DIR / "soil_classifier" / "metrics.json")
    soil_classes = _read_json(MODELS_DIR / "soil_classifier" / "class_names.json")
    texture_meta = _read_json(MODELS_DIR / "soil_texture_classifier" / "metadata.json")
    deficiency_meta = _read_json(MODELS_DIR / "nutrient_deficiency_detector" / "metadata.json")
    nutrient_meta = _read_json(MODELS_DIR / "nutrient_classifier" / "metadata.json")
    fusion_meta = _read_json(MODELS_DIR / "hybrid_fusion" / "metadata.json")
    degradation_meta = _read_json(MODELS_DIR / "degradation_forecaster" / "metadata.json")

    soil_ckpt = MODELS_DIR / "soil_classifier" / "best_model.pth"
    texture_ckpt = MODELS_DIR / "soil_texture_classifier" / "best_model.pth"
    deficiency_ckpt = MODELS_DIR / "nutrient_deficiency_detector" / "best_model.pth"
    nutrient_ckpt = MODELS_DIR / "nutrient_classifier" / "model.joblib"
    fusion_ckpt = MODELS_DIR / "hybrid_fusion" / "best_model.pth"
    degradation_ckpt = MODELS_DIR / "degradation_forecaster" / "model.joblib"

    return {
        "soil_type_classifier": {
            "model_name": "soil_type_classifier",
            "version": "efficientnet_b0_v1.0",
            "architecture": "EfficientNet-B0 (PyTorch)",
            "dataset": "SoilSense 6-Class Soil Type Dataset (Phantom-fs, Kaggle, Wikimedia)",
            "dataset_version": "v1.0",
            "training_date": "2026-11-23",
            "status": "trained_and_validated" if soil_ckpt.is_file() else "missing_checkpoint",
            "checkpoint_path": "models/soil_classifier/best_model.pth",
            "classes": soil_classes.get("classes", [
                "alluvial_soil", "black_soil", "clay_soil",
                "laterite_soil", "red_soil", "sandy_soil",
            ]),
            "metrics": {
                "accuracy": soil_metrics.get("accuracy"),
                "macro_f1": soil_metrics.get("macro_avg", {}).get("f1_score"),
                "total_test_samples": soil_metrics.get("total_test_samples"),
            },
            "limitations": "Classifies surface soil type only; does not measure chemical NPK or particle-size texture.",
        },
        "soil_texture_classifier": {
            "model_name": "soil_texture_classifier",
            "version": texture_meta.get("model_version", "soil_texture_efficientnet_b0_v1.0"),
            "architecture": texture_meta.get("architecture", "EfficientNet-B0 (PyTorch)"),
            "dataset": texture_meta.get("dataset", "92k Soil Texture Image Patch Dataset (Zenodo 10.5281/zenodo.20404766)"),
            "dataset_version": texture_meta.get("dataset_version", "v3"),
            "training_date": texture_meta.get("training_date"),
            "status": "trained_and_validated" if texture_ckpt.is_file() else "not_trained",
            "checkpoint_path": "models/soil_texture_classifier/best_model.pth",
            "classes": texture_meta.get("classes", []),
            "metrics": texture_meta.get("metrics", {}),
            "limitations": texture_meta.get("limitations", "Visual surface texture estimation; not a laboratory hydrometer/sieve test."),
        },
        "nutrient_deficiency_detector": {
            "model_name": "nutrient_deficiency_detector",
            "version": deficiency_meta.get("model_version", "plant_deficiency_efficientnet_b0_v1.0"),
            "architecture": deficiency_meta.get("architecture", "EfficientNet-B0 (PyTorch)"),
            "dataset": deficiency_meta.get("dataset", "Maize Phosphorus Leaf Deficiency (MPLD) (Zenodo 10.5281/zenodo.10041514)"),
            "dataset_version": deficiency_meta.get("dataset_version", "v1.0"),
            "training_date": deficiency_meta.get("training_date"),
            "status": "trained_and_validated" if deficiency_ckpt.is_file() else "not_trained",
            "checkpoint_path": "models/nutrient_deficiency_detector/best_model.pth",
            "classes": deficiency_meta.get("classes", []),
            "metrics": deficiency_meta.get("metrics", {}),
            "limitations": deficiency_meta.get("limitations", "Visual nutrient deficiency indication from plant leaf image only; not a laboratory soil NPK measurement."),
        },
        "nutrient_classifier": {
            "model_name": "nutrient_classifier",
            "version": nutrient_meta.get("model_version", "xgboost_nutrient_v1.0"),
            "architecture": nutrient_meta.get("architecture", "XGBoost Classifier"),
            "dataset": nutrient_meta.get("dataset", "Training Data of Soil NPK Fertility for Rice Cultivation (Zenodo 10.5281/zenodo.15307297)"),
            "dataset_version": nutrient_meta.get("dataset_version", "1.0"),
            "training_date": nutrient_meta.get("training_date"),
            "status": "trained_and_validated" if nutrient_ckpt.is_file() else "not_trained",
            "checkpoint_path": "models/nutrient_classifier/model.joblib",
            "features": nutrient_meta.get("features", []),
            "classes": nutrient_meta.get("classes", []),
            "metrics": nutrient_meta.get("metrics", {}),
            "limitations": nutrient_meta.get("limitations", "Trained on crop/soil fertility dataset; predictions are model estimates, not certified lab assays."),
        },
        "hybrid_fusion": {
            "model_name": "hybrid_fusion",
            "version": fusion_meta.get("model_version", "hybrid_fusion_v1.0"),
            "architecture": "EfficientNet-B0 (1280-D frozen backbone) + Tabular MLP Encoder -> Fusion Head",
            "dataset": fusion_meta.get("dataset", "None (blocked: 0 real paired image + soil-test records)"),
            "dataset_version": fusion_meta.get("dataset_version", "N/A"),
            "training_date": fusion_meta.get("training_date"),
            "status": "trained_and_validated" if fusion_ckpt.is_file() else "architecture_ready_untrained",
            "checkpoint_path": "models/hybrid_fusion/best_model.pth",
            "metrics": fusion_meta.get("metrics", {}),
            "limitations": "Training is pending genuine paired soil-image and soil-test records with shared ground-truth labels.",
        },
        "degradation_forecaster": {
            "model_name": "degradation_forecaster",
            "version": degradation_meta.get("model_version", "degradation_forecaster_v1.0"),
            "architecture": degradation_meta.get("architecture", "Time-Aware Gradient Boosted Regressor/Classifier"),
            "dataset": degradation_meta.get("dataset", "SoilHealthDB V2 (Zenodo 10.5281/zenodo.17565204)"),
            "dataset_version": degradation_meta.get("dataset_version", "V2"),
            "training_date": degradation_meta.get("training_date"),
            "status": "trained_and_validated" if degradation_ckpt.is_file() else "not_trained",
            "checkpoint_path": "models/degradation_forecaster/model.joblib",
            "features": degradation_meta.get("features", []),
            "metrics": degradation_meta.get("metrics", {}),
            "limitations": degradation_meta.get("limitations", "ML Soil Degradation Forecast; distinct from descriptive Historical Soil Trend Analysis."),
        },
    }

