import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..services.model_registry import get_model_registry

router = APIRouter(tags=["CNN Model Analytics"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "soil_classifier"


@router.get("/cnn/metrics")
def get_cnn_metrics():
    """Return the evaluation metrics of the trained soil classifier."""
    metrics_file = MODEL_DIR / "metrics.json"

    if not metrics_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="CNN evaluation metrics are not available."
        )

    try:
        with metrics_file.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        return JSONResponse(content=metrics)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read CNN evaluation metrics."
        ) from exc


@router.get("/cnn/training-curves")
def get_cnn_training_curves():
    """Return the CNN training and validation curves."""
    image_file = MODEL_DIR / "training_curves.png"

    if not image_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training curves are not available."
        )

    return FileResponse(image_file, media_type="image/png")


@router.get("/cnn/capabilities")
def get_ml_capabilities():
    """Report active vs data-blocked ML/analytics capabilities from the Model Registry."""
    reg = get_model_registry()
    active: dict = {
        "soil_type_classification": {
            "model": "EfficientNet-B0",
            "classes": reg["soil_type_classifier"]["classes"],
            "test_accuracy": 0.8981,
            "macro_f1": 0.8899,
        },
        "gradcam_explainability": {
            "endpoint": "/api/images/{image_id}/gradcam",
            "target_layer": "features[-1]",
        },
        "soil_health_score": {"method": "rule_based_weighted_scoring"},
        "nutrient_advisory": {"method": "heuristic_rule_engine"},
        "historical_soil_trends": {
            "endpoint": "/api/soil-tests/trends",
            "is_ml_forecast": False,
            "disclaimer": "Historical Soil Trend Analysis – Not an ML Forecast",
        },
    }
    blocked: dict = {}

    if reg["soil_texture_classifier"]["status"] == "trained_and_validated":
        active["soil_texture_classification"] = {
            "endpoint": "/api/images/{image_id}/predict-texture",
            **reg["soil_texture_classifier"],
        }
    else:
        blocked["soil_texture_classification"] = reg["soil_texture_classifier"]["limitations"]

    if reg["nutrient_deficiency_detector"]["status"] == "trained_and_validated":
        active["visible_deficiency_classification"] = {
            "endpoint": "/api/images/{image_id}/predict-deficiency",
            **reg["nutrient_deficiency_detector"],
        }
    else:
        blocked["visible_deficiency_classification"] = reg["nutrient_deficiency_detector"]["limitations"]

    if reg["nutrient_classifier"]["status"] == "trained_and_validated":
        active["gradient_boosted_nutrient_ml"] = {
            "endpoint": "/api/ml/predict-nutrient-fertility",
            **reg["nutrient_classifier"],
        }
        active["shap_explainability"] = {
            "endpoint": "/api/ml/predict-nutrient-fertility",
            "explainer": "shap.TreeExplainer",
        }
    else:
        blocked["gradient_boosted_nutrient_ml"] = reg["nutrient_classifier"]["limitations"]
        blocked["shap_explainability"] = "Blocked by absence of trained tabular ML model."

    if reg["hybrid_fusion"]["status"] == "trained_and_validated":
        active["hybrid_fusion_model"] = reg["hybrid_fusion"]
    else:
        blocked["hybrid_fusion_model"] = reg["hybrid_fusion"]["limitations"]

    if reg["degradation_forecaster"]["status"] == "trained_and_validated":
        active["ml_degradation_forecast"] = {
            "endpoint": "/api/ml/predict-degradation-risk",
            **reg["degradation_forecaster"],
        }
    else:
        blocked["ml_degradation_forecast"] = reg["degradation_forecaster"]["limitations"]

    return JSONResponse(content={"active_features": active, "blocked_features": blocked})


@router.get("/cnn/confusion-matrix")
def get_cnn_confusion_matrix():
    """Return the CNN test-set confusion matrix visualization."""
    image_file = MODEL_DIR / "confusion_matrix.png"

    if not image_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix image is not available."
        )

    return FileResponse(image_file, media_type="image/png")


@router.get("/cnn/capabilities")
def get_ml_capabilities():
    """Report active vs data-blocked ML/analytics capabilities from the Model Registry."""
    reg = get_model_registry()
    active: dict = {
        "soil_type_classification": {
            "model": "EfficientNet-B0",
            "classes": reg["soil_type_classifier"]["classes"],
            "test_accuracy": 0.8981,
            "macro_f1": 0.8899,
        },
        "gradcam_explainability": {
            "endpoint": "/api/images/{image_id}/gradcam",
            "target_layer": "features[-1]",
        },
        "soil_health_score": {"method": "rule_based_weighted_scoring"},
        "nutrient_advisory": {"method": "heuristic_rule_engine"},
        "historical_soil_trends": {
            "endpoint": "/api/soil-tests/trends",
            "is_ml_forecast": False,
            "disclaimer": "Historical Soil Trend Analysis – Not an ML Forecast",
        },
    }
    blocked: dict = {}

    if reg["soil_texture_classifier"]["status"] == "trained_and_validated":
        active["soil_texture_classification"] = {
            "endpoint": "/api/images/{image_id}/predict-texture",
            **reg["soil_texture_classifier"],
        }
    else:
        blocked["soil_texture_classification"] = reg["soil_texture_classifier"]["limitations"]

    if reg["nutrient_deficiency_detector"]["status"] == "trained_and_validated":
        active["visible_deficiency_classification"] = {
            "endpoint": "/api/images/{image_id}/predict-deficiency",
            **reg["nutrient_deficiency_detector"],
        }
    else:
        blocked["visible_deficiency_classification"] = reg["nutrient_deficiency_detector"]["limitations"]

    if reg["nutrient_classifier"]["status"] == "trained_and_validated":
        active["gradient_boosted_nutrient_ml"] = {
            "endpoint": "/api/ml/predict-nutrient-fertility",
            **reg["nutrient_classifier"],
        }
        active["shap_explainability"] = {
            "endpoint": "/api/ml/predict-nutrient-fertility",
            "explainer": "shap.TreeExplainer",
        }
    else:
        blocked["gradient_boosted_nutrient_ml"] = reg["nutrient_classifier"]["limitations"]
        blocked["shap_explainability"] = "Blocked by absence of trained tabular ML model."

    if reg["hybrid_fusion"]["status"] == "trained_and_validated":
        active["hybrid_fusion_model"] = reg["hybrid_fusion"]
    else:
        blocked["hybrid_fusion_model"] = reg["hybrid_fusion"]["limitations"]

    if reg["degradation_forecaster"]["status"] == "trained_and_validated":
        active["ml_degradation_forecast"] = {
            "endpoint": "/api/ml/predict-degradation-risk",
            **reg["degradation_forecaster"],
        }
    else:
        blocked["ml_degradation_forecast"] = reg["degradation_forecaster"]["limitations"]

    return JSONResponse(content={"active_features": active, "blocked_features": blocked})
