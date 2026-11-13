from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    farmer_name: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=255)


class FieldOut(FieldIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class SoilTestIn(BaseModel):
    nitrogen: float = Field(ge=0, le=1000, description="Available nitrogen, kg/ha")
    phosphorus: float = Field(ge=0, le=500, description="Available phosphorus, kg/ha")
    potassium: float = Field(ge=0, le=2000, description="Available potassium, kg/ha")
    ph: float = Field(ge=0, le=14, description="Soil pH")
    organic_matter: float = Field(ge=0, le=100, description="Organic matter, %")
    moisture: float = Field(ge=0, le=100, description="Moisture content, %")
    notes: str | None = Field(default=None, max_length=500)


class SoilTestOut(SoilTestIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    field_id: int
    tested_at: datetime


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
