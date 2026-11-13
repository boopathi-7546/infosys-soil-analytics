"""Safe migration script from SQLite (soil.db) to PostgreSQL.

Reads existing records from SQLite and inserts them into PostgreSQL without
modifying or deleting the source SQLite database.
"""
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import Base
from app.models import AnalysisResult, Field, SoilImage, SoilTest


def migrate(sqlite_path: str = "soil.db", pg_url: str | None = None):
    sqlite_file = Path(sqlite_path)
    if not sqlite_file.exists():
        print(f"[INFO] SQLite file '{sqlite_path}' does not exist. Nothing to migrate.")
        return

    target_pg_url = pg_url or os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://soil:soil@localhost:5432/soildb"
    )

    print(f"[INFO] Source SQLite: {sqlite_path}")
    print(f"[INFO] Target PostgreSQL: {target_pg_url.split('@')[-1] if '@' in target_pg_url else target_pg_url}")

    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
    pg_engine = create_engine(target_pg_url, pool_pre_ping=True)

    # Ensure tables exist in target
    Base.metadata.create_all(pg_engine)

    with Session(sqlite_engine) as src_session, Session(pg_engine) as dst_session:
        # 1. Migrate Fields
        src_fields = src_session.scalars(select(Field).order_by(Field.id)).all()
        print(f"[INFO] Found {len(src_fields)} fields in SQLite.")
        for f in src_fields:
            if not dst_session.get(Field, f.id):
                new_f = Field(
                    id=f.id,
                    name=f.name,
                    farmer_name=f.farmer_name,
                    location=f.location,
                    created_at=f.created_at,
                )
                dst_session.add(new_f)
        dst_session.commit()

        # 2. Migrate SoilTests
        src_tests = src_session.scalars(select(SoilTest).order_by(SoilTest.id)).all()
        print(f"[INFO] Found {len(src_tests)} soil tests in SQLite.")
        for t in src_tests:
            if not dst_session.get(SoilTest, t.id):
                new_t = SoilTest(
                    id=t.id,
                    field_id=t.field_id,
                    nitrogen=t.nitrogen,
                    phosphorus=t.phosphorus,
                    potassium=t.potassium,
                    ph=t.ph,
                    organic_matter=t.organic_matter,
                    moisture=t.moisture,
                    notes=t.notes,
                    tested_at=t.tested_at,
                )
                dst_session.add(new_t)
        dst_session.commit()

        # 3. Migrate SoilImages
        src_images = src_session.scalars(select(SoilImage).order_by(SoilImage.id)).all()
        print(f"[INFO] Found {len(src_images)} soil images in SQLite.")
        for img in src_images:
            if not dst_session.get(SoilImage, img.id):
                new_img = SoilImage(
                    id=img.id,
                    field_id=img.field_id,
                    file_path=img.file_path,
                    original_filename=img.original_filename,
                    original_format=img.original_format,
                    size_bytes=img.size_bytes,
                    width=img.width,
                    height=img.height,
                    blur_score=img.blur_score,
                    uploaded_at=img.uploaded_at,
                )
                dst_session.add(new_img)
        dst_session.commit()

        # 4. Migrate AnalysisResults if present
        src_results = src_session.scalars(select(AnalysisResult).order_by(AnalysisResult.id)).all()
        print(f"[INFO] Found {len(src_results)} analysis results in SQLite.")
        for r in src_results:
            if not dst_session.get(AnalysisResult, r.id):
                new_r = AnalysisResult(
                    id=r.id,
                    field_id=r.field_id,
                    image_id=r.image_id,
                    soil_test_id=r.soil_test_id,
                    result=r.result,
                    created_at=r.created_at,
                )
                dst_session.add(new_r)
        dst_session.commit()

    print("[SUCCESS] Safe migration completed successfully. SQLite database was preserved intact.")


if __name__ == "__main__":
    migrate()

