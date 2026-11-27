"""Structured XGBoost Nutrient Fertility Service + SHAP Explainability (Phases E & F)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NUTRIENT_MODEL_PATH = PROJECT_ROOT / "models" / "nutrient_classifier" / "model.joblib"


class NutrientMLModelError(Exception):
    """Raised when the trained XGBoost nutrient model is missing or unreadable."""


class NutrientMLService:
    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_NUTRIENT_MODEL_PATH
        self.bundle: Optional[dict[str, Any]] = None
        self.explainer: Any = None
        self.model_version = "xgboost_nutrient_v1.0"

    def load_model(self, force_reload: bool = False) -> None:
        if self.bundle is not None and not force_reload:
            return
        if not self.model_path.is_file():
            raise NutrientMLModelError(f"Nutrient classifier checkpoint not found at {self.model_path}")
        try:
            self.bundle = joblib.load(self.model_path)
            self.explainer = None
        except Exception as exc:
            raise NutrientMLModelError(f"Failed to load nutrient classifier: {exc}") from exc

    def _get_explainer(self):
        if self.explainer is None:
            import shap
            self.explainer = shap.TreeExplainer(self.bundle["model"])
        return self.explainer

    def predict_and_explain(self, nitrogen: float, phosphorus: float, potassium: float) -> dict[str, Any]:
        self.load_model()
        model = self.bundle["model"]
        features: list[str] = self.bundle["features"]
        class_labels: dict[int, str] = self.bundle["class_labels"]

        X = np.array([[float(nitrogen), float(phosphorus), float(potassium)]], dtype=np.float32)
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_labels[pred_idx]
        conf = float(probs[pred_idx])

        # Compute genuine TreeExplainer SHAP values for the predicted class
        explainer = self._get_explainer()
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            class_shap = np.asarray(shap_vals[pred_idx])[0]
        else:
            arr = np.asarray(shap_vals)
            class_shap = arr[0, :, pred_idx] if arr.ndim == 3 else arr[0]

        contributions = []
        for feat_name, feat_val, sv in zip(features, X[0].tolist(), class_shap.tolist()):
            direction = "increased" if sv >= 0 else "decreased"
            contributions.append({
                "feature": feat_name,
                "value": round(float(feat_val), 2),
                "shap_value": round(float(sv), 4),
                "direction": "positive" if sv >= 0 else "negative",
                "explanation": (
                    f"{feat_name} ({feat_val:.1f}) {direction} the likelihood of "
                    f"{pred_label.replace('_', ' ')} (SHAP={sv:+.4f})."
                ),
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        # Global importance from trained XGBoost feature_importances_
        global_imp = {
            f: round(float(imp), 4)
            for f, imp in zip(features, model.feature_importances_.tolist())
        }

        return {
            "predicted_fertility_class": pred_label,
            "confidence": round(conf, 4),
            "confidence_percentage": round(conf * 100, 2),
            "probabilities": {class_labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "model_version": self.model_version,
            "model_maturity": "experimental",
            "explainer": "shap.TreeExplainer",
            "global_feature_importance": global_imp,
            "feature_contributions": contributions,
            "summary_explanation": contributions[0]["explanation"] if contributions else "",
            "disclaimer": "Experimental model (pipeline demonstration only): not suitable for real fertilizer dosage decisions or laboratory soil assays.",
        }


_instance: Optional[NutrientMLService] = None


def get_nutrient_ml_service() -> NutrientMLService:
    global _instance
    if _instance is None:
        _instance = NutrientMLService()
    return _instance

