from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


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
ALLOWED_NUTRIENT_UNITS = ("kg/ha", "mg/kg", "ppm")
ALLOWED_BULK_DENSITY_UNITS = ("g/cm3", "kg/m3")
ALLOWED_VERIFICATION_STATUSES = (
    "laboratory_verified",
    "user_entered_unverified",
    "field_sensor_estimate",
)


class SoilTestIn(BaseModel):
    nitrogen: float = Field(ge=0, le=500, description="Available nitrogen")
    phosphorus: float = Field(ge=0, le=150, description="Available phosphorus")
    potassium: float = Field(ge=0, le=800, description="Available potassium")
    nitrogen_unit: Literal["kg/ha", "mg/kg", "ppm"] | None = None
    phosphorus_unit: Literal["kg/ha", "mg/kg", "ppm"] | None = None
    potassium_unit: Literal["kg/ha", "mg/kg", "ppm"] | None = None
    ph: float = Field(ge=0, le=14, description="Soil pH")
    organic_matter: float = Field(ge=0, le=100, description="Organic matter, %")
    moisture: float = Field(ge=0, le=100, description="Moisture content, %")
    testing_method: str | None = Field(default=None, max_length=100)
    sample_depth_cm: float | None = Field(default=None, gt=0, le=300)
    bulk_density: float | None = Field(default=None, gt=0, le=3000)
    bulk_density_unit: Literal["g/cm3", "kg/m3"] | None = None
    sample_collected_at: datetime | None = None
    verification_status: Literal[
        "laboratory_verified",
        "user_entered_unverified",
        "field_sensor_estimate",
    ] | None = None
    image_id: int | None = Field(default=None, ge=1, description="Optional soil image ID to pair with this test")
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_bulk_density_and_units(self):
        if self.bulk_density is None and self.bulk_density_unit is not None:
            raise ValueError("bulk_density is required when bulk_density_unit is specified")
        if self.bulk_density is not None:
            unit = self.bulk_density_unit or "g/cm3"
            self.bulk_density_unit = unit
            if unit == "g/cm3" and not (0.2 <= self.bulk_density <= 3.0):
                raise ValueError("bulk_density in g/cm3 must be between 0.2 and 3.0")
            if unit == "kg/m3" and not (200.0 <= self.bulk_density <= 3000.0):
                raise ValueError("bulk_density in kg/m3 must be between 200 and 3000")
        return self


class SoilTestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    organic_matter: float
    moisture: float
    notes: str | None = None
    tested_at: datetime
    health_score: float | None = None  # computed field, not stored in DB
    test_number: int | None = None     # account-specific sequential test number (#1, #2, ...)
    display_id: int | None = None      # display number alias for user interface
    field_name: str | None = None
    nitrogen_unit: str | None = None
    phosphorus_unit: str | None = None
    potassium_unit: str | None = None
    testing_method: str | None = None
    sample_depth_cm: float | None = None
    bulk_density: float | None = None
    bulk_density_unit: str | None = None
    sample_collected_at: datetime | None = None
    verification_status: str | None = None
    paired_image_ids: list[int] = []
    electrical_conductivity: float | None = None
    organic_carbon: float | None = None
    ph_method: str | None = None
    data_provenance: str | None = None


class AdvisoryOut(BaseModel):
    soil_test_id: int
    health_score: float
    recommendations: list[str]
    suitable_crops: list[str]
    warnings: list[str]


class TrendPointOut(BaseModel):
    test_id: int
    tested_at: datetime
    value: float
    unit: str


class ParameterTrendOut(BaseModel):
    parameter: str
    label: str
    unit: str | None = None
    status: str  # "increasing" | "decreasing" | "stable" | "insufficient_data" | "incompatible_units"
    earliest_value: float | None = None
    latest_value: float | None = None
    change: float | None = None
    percent_change: float | None = None
    comparable_points: int = 0
    note: str | None = None
    points: list[TrendPointOut] = []


class SoilTrendResponseOut(BaseModel):
    analysis_type: str = "historical_trend_analysis"
    is_ml_forecast: bool = False
    disclaimer: str = "Historical Soil Trend Analysis – Not an ML Forecast"
    field_id: int | None = None
    field_name: str | None = None
    total_records: int
    earliest_tested_at: datetime | None = None
    latest_tested_at: datetime | None = None
    status: str  # "available" | "insufficient_data"
    message: str
    parameters: list[ParameterTrendOut]


# ---------------------------------------------------------------------------
# Image Schemas
# ---------------------------------------------------------------------------
class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    soil_test_id: int | None = None
    original_filename: str | None
    original_format: str
    size_bytes: int
    width: int
    height: int
    blur_score: float
    uploaded_at: datetime


class ImagePairSoilTestIn(BaseModel):
    soil_test_id: int = Field(ge=1)


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
    model_version: str | None = None
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
    model_version: str | None = None
    best_epoch: int | None = None
