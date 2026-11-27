"""ML Soil Degradation Forecasting Service (Phase H).

Uses the chronologically trained XGBClassifier + XGBRegressor from SoilHealthDB V2
to predict Soil Organic Carbon (SOC) decline risk and projected SOC change.
Explicitly separate from descriptive Historical Soil Trend Analysis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEGRADATION_MODEL_PATH = PROJECT_ROOT / "models" / "degradation_forecaster" / "model.joblib"


class DegradationModelError(Exception):
    """Raised when the degradation forecaster checkpoint is missing or unreadable."""


class DegradationForecasterService:
    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_DEGRADATION_MODEL_PATH
        self.bundle: Optional[dict[str, Any]] = None
        self.model_version = "degradation_forecaster_v1.0"

    def load_model(self, force_reload: bool = False) -> None:
        if self.bundle is not None and not force_reload:
            return
        if not self.model_path.is_file():
            raise DegradationModelError(f"Degradation forecaster checkpoint not found at {self.model_path}")
        try:
            self.bundle = joblib.load(self.model_path)
        except Exception as exc:
            raise DegradationModelError(f"Failed to load degradation forecaster: {exc}") from exc

    def predict(
        self,
        baseline_oc: float,
        horizon_years: float = 3.0,
        ph: float | None = None,
        bulk_density: float | None = None,
        sand_pct: float | None = None,
        silt_pct: float | None = None,
        clay_pct: float | None = None,
        mean_annual_temp_c: float | None = None,
        mean_annual_precip_mm: float | None = None,
    ) -> dict[str, Any]:
        self.load_model()
        cls_pipe = self.bundle["classifier"]
        reg_pipe = self.bundle["regressor"]

        def _nan_or(v: float | None) -> float:
            return float(v) if v is not None and math.isfinite(float(v)) else float("nan")

        X = np.array([[
            float(baseline_oc),
            float(horizon_years),
            _nan_or(ph),
            _nan_or(bulk_density),
            _nan_or(sand_pct),
            _nan_or(silt_pct),
            _nan_or(clay_pct),
            _nan_or(mean_annual_temp_c),
            _nan_or(mean_annual_precip_mm),
        ]], dtype=np.float32)

        decline_prob = float(cls_pipe.predict_proba(X)[0, 1])
        projected_oc_change = float(reg_pipe.predict(X)[0])

        if decline_prob >= 0.65:
            risk_level = "high"
        elif decline_prob >= 0.40:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "analysis_type": "ml_soil_degradation_forecast",
            "label": "ML Soil Degradation Forecast",
            "model_version": self.model_version,
            "model_maturity": "experimental",
            "horizon_years": round(float(horizon_years), 2),
            "soc_decline_probability": round(decline_prob, 4),
            "risk_level": risk_level,
            "projected_soc_change": round(projected_oc_change, 4),
            "explanation": (
                f"Over a {horizon_years:.1f}-year horizon from baseline OC={baseline_oc:.2f}, "
                f"the time-aware XGBoost model estimates a {decline_prob*100:.1f}% probability of SOC decline "
                f"({risk_level} risk; projected delta {projected_oc_change:+.3f})."
            ),
            "disclaimer": "Experimental ML Soil Degradation Forecast: not a reliable future prediction or laboratory measurement; use Historical Soil Trend Analysis for observed plot history.",
        }


_instance: Optional[DegradationForecasterService] = None


def get_degradation_forecaster() -> DegradationForecasterService:
    global _instance
    if _instance is None:
        _instance = DegradationForecasterService()
    return _instance

