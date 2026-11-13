from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    farmer_name: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=255)


class FieldOut(FieldIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
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


class AdvisoryOut(BaseModel):
    soil_test_id: int
    health_score: float
    recommendations: list[str]
    suitable_crops: list[str]
    warnings: list[str]


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


# New simplified upload response as required by task spec
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


class DatasetStatusOut(BaseModel):
    soil_types: list[dict[str, Any]]
    total_raw_images: int
    total_processed_images: int
    sources: list[dict[str, Any]]
