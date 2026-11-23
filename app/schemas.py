from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------------------
# Farmer Authentication Schemas
# ---------------------------------------------------------------------------
class FarmerRegisterIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120, description="Farmer's full name")
    email: EmailStr = Field(description="Unique email address")
    password: str = Field(min_length=6, max_length=72, description="Password (min 6 characters)")


class FarmerLoginIn(BaseModel):
    email: EmailStr = Field(description="Registered email address")
    password: str = Field(min_length=1, description="Password")


class FarmerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    email: str
    is_active: bool
    created_at: datetime
    google_id: str | None = None
    avatar_url: str | None = None
    auth_provider: str = "local"


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    farmer: FarmerOut


class GoogleAuthUrlOut(BaseModel):
    url: str


class GoogleCallbackIn(BaseModel):
    code: str
    state: str


# ---------------------------------------------------------------------------
# Farm Field Schemas
# ---------------------------------------------------------------------------
class FieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    farmer_name: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=255)


class FieldOut(FieldIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    farmer_id: int | None = None
    created_at: datetime


class FieldStats(BaseModel):
    field_id: int
    total_soil_tests: int
    total_images: int
    latest_soil_test: datetime | None
    avg_nitrogen: float | None
    avg_phosphorus: float | None
    avg_potassium: float | None
    avg_ph: float | None
    avg_organic_matter: float | None
    avg_moisture: float | None


# ---------------------------------------------------------------------------
# Soil Test Schemas
# ---------------------------------------------------------------------------
class SoilTestIn(BaseModel):
    nitrogen: float = Field(ge=0, le=500, description="Available nitrogen, kg/ha")
    phosphorus: float = Field(ge=0, le=150, description="Available phosphorus, kg/ha")
    potassium: float = Field(ge=0, le=800, description="Available potassium, kg/ha")
    ph: float = Field(ge=0, le=14, description="Soil pH")
    organic_matter: float = Field(ge=0, le=100, description="Organic matter, %")
    moisture: float = Field(ge=0, le=100, description="Moisture content, %")
    notes: str | None = Field(default=None, max_length=500)


class SoilTestOut(SoilTestIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    tested_at: datetime
    health_score: float | None = None  # computed field, not stored in DB
    test_number: int | None = None     # account-specific sequential test number (#1, #2, ...)
    display_id: int | None = None      # display number alias for user interface


class AdvisoryOut(BaseModel):
    soil_test_id: int
    health_score: float
    recommendations: list[str]
    suitable_crops: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Image Schemas
# ---------------------------------------------------------------------------
class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    original_filename: str | None
    original_format: str
    size_bytes: int
    width: int
    height: int
    blur_score: float
    uploaded_at: datetime


class ImageUploadOut(BaseModel):
    image: ImageOut
    warnings: list[str]


class ImageUploadResponse(BaseModel):
    image_id: int
    filename: str
    field_id: int
    url: str


class Base64UploadIn(BaseModel):
    field_id: int
    image_data: str = Field(description="data:image/jpeg;base64,... or data:image/png;base64,...")


class PreprocessResult(BaseModel):
    image_id: int
    processed_path: str
    width: int
    height: int
    mean_rgb: list[float]
    quality_ok: bool
    rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# Dataset Schemas
# ---------------------------------------------------------------------------
class DatasetStatusOut(BaseModel):
    soil_types: list[dict[str, Any]]
    total_raw_images: int
    total_processed_images: int
    sources: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# CNN Soil Prediction Schemas
# ---------------------------------------------------------------------------
class SoilPredictionData(BaseModel):
    predicted_soil_type: str
    confidence: float
    confidence_percentage: float
    probabilities: dict[str, float]
    model_version: str
    best_epoch: int | None = None
    type: str = "cnn_prediction"


class SoilPredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    image_id: int | None = None
    soil_test_id: int | None = None
    created_at: datetime
    prediction: SoilPredictionData


class PredictionHistoryItem(BaseModel):
    id: int
    field_id: int
    field_name: str | None = None
    image_id: int | None = None
    created_at: datetime
    predicted_soil_type: str
    confidence: float
    confidence_percentage: float
    probabilities: dict[str, float]
    model_version: str
    best_epoch: int | None = None
