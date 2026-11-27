"""FastAPI endpoints for Weeks 3-4 ML capabilities:
- Model Registry
- Soil Texture Classification (Phase C)
- Visible Plant Nutrient Deficiency Detection (Phase D)
- XGBoost Structured Nutrient Fertility + SHAP Explainability (Phases E & F)
- Hybrid Fusion Status (Phase G)
- ML Soil Degradation Forecast (Phase H)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Farmer, SoilImage
from ..services import storage
from ..services.auth import get_current_farmer
from ..services.deficiency_detector import (
    DeficiencyModelError,
    InvalidDeficiencyImageError,
    get_deficiency_detector,
)
from ..services.degradation_forecaster import DegradationModelError, get_degradation_forecaster
from ..services.model_registry import get_model_registry
from ..services.nutrient_ml import NutrientMLModelError, get_nutrient_ml_service
from ..services.soil_texture_classifier import (
    InvalidTextureImageError,
    TextureModelError,
    get_soil_texture_classifier,
)
from .images import _get_farmer_image_or_404

router = APIRouter(tags=["ML Models & Explainability"])


class NutrientMLIn(BaseModel):
    nitrogen: float = Field(ge=0, le=1000)
    phosphorus: float = Field(ge=0, le=1000)
    potassium: float = Field(ge=0, le=1000)


class DegradationForecastIn(BaseModel):
    baseline_oc: float = Field(gt=0, le=100, description="Baseline soil organic carbon")
    horizon_years: float = Field(default=3.0, gt=0, le=50, description="Forecast horizon in years")
    ph: float | None = Field(default=None, ge=0, le=14)
    bulk_density: float | None = Field(default=None, gt=0, le=3.5)
    sand_pct: float | None = Field(default=None, ge=0, le=100)
    silt_pct: float | None = Field(default=None, ge=0, le=100)
    clay_pct: float | None = Field(default=None, ge=0, le=100)
    mean_annual_temp_c: float | None = Field(default=None, ge=-40, le=55)
    mean_annual_precip_mm: float | None = Field(default=None, ge=0, le=10000)


@router.get("/ml/registry")
def read_model_registry():
    """Return all registered SoilSense AI models, versions, metrics, and limitations."""
    return {"models": get_model_registry()}


@router.post("/images/{image_id}/predict-texture")
def predict_soil_texture(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Classify visual surface soil texture (distinct from 6-class soil type)."""
    img = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        raw_bytes = storage.read_image(img.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")

    detector = get_soil_texture_classifier()
    try:
        return detector.predict(raw_bytes)
    except TextureModelError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except InvalidTextureImageError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.post("/images/{image_id}/predict-deficiency")
def predict_visible_deficiency(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Detect visible plant nutrient deficiency indications from a leaf image."""
    img = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        raw_bytes = storage.read_image(img.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")

    detector = get_deficiency_detector()
    try:
        return detector.predict(raw_bytes)
    except DeficiencyModelError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except InvalidDeficiencyImageError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.post("/ml/predict-nutrient-fertility")
def predict_nutrient_fertility_with_shap(
    body: NutrientMLIn,
    current_farmer: Farmer = Depends(get_current_farmer),
):
    """Predict structured NPK fertility class using XGBoost and explain with SHAP TreeExplainer."""
    svc = get_nutrient_ml_service()
    try:
        return svc.predict_and_explain(body.nitrogen, body.phosphorus, body.potassium)
    except NutrientMLModelError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.post("/ml/predict-degradation-risk")
def predict_soil_degradation_risk(
    body: DegradationForecastIn,
    current_farmer: Farmer = Depends(get_current_farmer),
):
    """Predict future SOC decline probability and projected SOC change (ML Soil Degradation Forecast)."""
    svc = get_degradation_forecaster()
    try:
        return svc.predict(
            baseline_oc=body.baseline_oc,
            horizon_years=body.horizon_years,
            ph=body.ph,
            bulk_density=body.bulk_density,
            sand_pct=body.sand_pct,
            silt_pct=body.silt_pct,
            clay_pct=body.clay_pct,
            mean_annual_temp_c=body.mean_annual_temp_c,
            mean_annual_precip_mm=body.mean_annual_precip_mm,
        )
    except DegradationModelError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/ml/hybrid-fusion/status")
def get_hybrid_fusion_status(
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Report real paired (image, soil_test) readiness and Hybrid Fusion model status."""
    paired_count = db.scalar(
        select(func.count(SoilImage.id)).where(SoilImage.soil_test_id.is_not(None))
    ) or 0
    registry_entry = get_model_registry()["hybrid_fusion"]
    return {
        **registry_entry,
        "paired_records_in_db": int(paired_count),
        "minimum_required_for_training": 50,
    }

