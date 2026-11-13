"""Database tables. Images are stored on disk; the DB keeps only their path + metadata."""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests)
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Field(Base):
    """A farmer's field/plot. All records hang off a field so trends can be tracked per field."""
    __tablename__ = "fields"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    farmer_name: Mapped[str | None] = mapped_column(String(120))
    location: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    soil_tests: Mapped[list["SoilTest"]] = relationship(back_populates="field", cascade="all, delete-orphan")
    images: Mapped[list["SoilImage"]] = relationship(back_populates="field", cascade="all, delete-orphan")


class SoilTest(Base):
    __tablename__ = "soil_tests"

    id: Mapped[int] = mapped_column(primary_key=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"), index=True)
    nitrogen: Mapped[float] = mapped_column(Float)         # kg/ha
    phosphorus: Mapped[float] = mapped_column(Float)       # kg/ha
    potassium: Mapped[float] = mapped_column(Float)        # kg/ha
    ph: Mapped[float] = mapped_column(Float)
    organic_matter: Mapped[float] = mapped_column(Float)   # %
    moisture: Mapped[float] = mapped_column(Float)         # %
    notes: Mapped[str | None] = mapped_column(String(500))
    tested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    field: Mapped[Field] = relationship(back_populates="soil_tests")


class SoilImage(Base):
    __tablename__ = "soil_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(String(500))    # relative to UPLOAD_DIR
    original_filename: Mapped[str | None] = mapped_column(String(255))
    original_format: Mapped[str] = mapped_column(String(10))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    blur_score: Mapped[float] = mapped_column(Float)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    field: Mapped[Field] = relationship(back_populates="images")


class AnalysisResult(Base):
    """Filled in Milestone 2 (CNN + structured model output). Created now so the schema is stable."""
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[int | None] = mapped_column(ForeignKey("soil_images.id", ondelete="SET NULL"))
    soil_test_id: Mapped[int | None] = mapped_column(ForeignKey("soil_tests.id", ondelete="SET NULL"))
    result: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
