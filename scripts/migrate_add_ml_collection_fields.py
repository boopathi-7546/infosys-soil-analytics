"""Safe, idempotent additive migration for ML-ready soil-test metadata and image pairing.

Adds optional columns to `soil_tests` and `soil_images` without modifying,
overwriting, or deleting any existing records. Works on PostgreSQL and SQLite.
"""

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SOIL_TEST_COLUMNS: list[tuple[str, str]] = [
    ("nitrogen_unit", "VARCHAR(20)"),
    ("phosphorus_unit", "VARCHAR(20)"),
    ("potassium_unit", "VARCHAR(20)"),
    ("testing_method", "VARCHAR(100)"),
    ("sample_depth_cm", "DOUBLE PRECISION"),
    ("bulk_density", "DOUBLE PRECISION"),
    ("bulk_density_unit", "VARCHAR(20)"),
    ("sample_collected_at", "TIMESTAMP WITH TIME ZONE"),
    ("verification_status", "VARCHAR(40)"),
]


def run_migration(engine=None):
    if engine is None:
        from dotenv import load_dotenv

        load_dotenv()
        db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://soil:soil@localhost:5432/soildb")
        _connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        engine = create_engine(db_url, connect_args=_connect_args, pool_pre_ping=True)

    is_sqlite = engine.url.drivername.startswith("sqlite")
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if "soil_tests" in existing_tables:
        existing_cols = {c["name"] for c in inspector.get_columns("soil_tests")}
        with engine.begin() as conn:
            for col_name, col_type in _SOIL_TEST_COLUMNS:
                if col_name not in existing_cols:
                    sqlite_type = "FLOAT" if col_type == "DOUBLE PRECISION" else ("DATETIME" if "TIMESTAMP" in col_type else col_type)
                    ddl_type = sqlite_type if is_sqlite else col_type
                    conn.execute(text(f"ALTER TABLE soil_tests ADD COLUMN {col_name} {ddl_type}"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_soil_tests_verification_status ON soil_tests (verification_status)"
            ))

    if "soil_images" in existing_tables:
        img_cols = {c["name"] for c in inspector.get_columns("soil_images")}
        with engine.begin() as conn:
            if "soil_test_id" not in img_cols:
                conn.execute(text(
                    "ALTER TABLE soil_images ADD COLUMN soil_test_id INTEGER REFERENCES soil_tests(id) ON DELETE SET NULL"
                ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_soil_images_soil_test_id ON soil_images (soil_test_id)"
            ))


if __name__ == "__main__":
    run_migration()

