"""Safe, idempotent database migration to add Farmer authentication and ownership.

Adds:
1. `farmers` table if not present.
2. `farmer_id` nullable foreign key column to `fields` table if not present.
3. Index on `fields.farmer_id` for fast query performance.

Preserves 100% of existing records. Does not drop or delete any table or row.
Works across both PostgreSQL and SQLite.
"""
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import Base
from app.models import Farmer  # ensures Farmer metadata is registered


def run_migration(engine=None):
    if engine is None:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://soil:soil@localhost:5432/soildb")
        _connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        engine = create_engine(db_url, connect_args=_connect_args, pool_pre_ping=True)

    is_sqlite = engine.url.drivername.startswith("sqlite")
    print(f"[MIGRATION] Target database dialect: {engine.url.drivername}")

    # Step 1: Ensure farmers table exists
    Base.metadata.create_all(engine, tables=[Farmer.__table__])
    print("[MIGRATION] Checked/created 'farmers' table.")

    # Step 2: Check and add farmer_id column to fields
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if "fields" in existing_tables:
        existing_cols = [c["name"] for c in inspector.get_columns("fields")]
        if "farmer_id" not in existing_cols:
            print("[MIGRATION] Adding 'farmer_id' column to 'fields' table...")
            with engine.begin() as conn:
                if is_sqlite:
                    conn.execute(text(
                        "ALTER TABLE fields ADD COLUMN farmer_id INTEGER REFERENCES farmers(id) ON DELETE SET NULL"
                    ))
                else:
                    conn.execute(text(
                        "ALTER TABLE fields ADD COLUMN farmer_id INTEGER REFERENCES farmers(id) ON DELETE SET NULL"
                    ))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_fields_farmer_id ON fields (farmer_id)"
                    ))
            print("[MIGRATION] Successfully added 'farmer_id' column to 'fields'.")
        else:
            print("[MIGRATION] 'farmer_id' column already exists on 'fields'.")

    # Step 3: Check and add Google OAuth columns to farmers table if missing
    if "farmers" in existing_tables:
        farmer_cols = [c["name"] for c in inspector.get_columns("farmers")]
        with engine.begin() as conn:
            if "google_id" not in farmer_cols:
                print("[MIGRATION] Adding 'google_id' column to 'farmers' table...")
                conn.execute(text("ALTER TABLE farmers ADD COLUMN google_id VARCHAR(255)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_farmers_google_id ON farmers (google_id)"))
                print("[MIGRATION] Added 'google_id' column and index.")
            if "avatar_url" not in farmer_cols:
                print("[MIGRATION] Adding 'avatar_url' column to 'farmers' table...")
                conn.execute(text("ALTER TABLE farmers ADD COLUMN avatar_url VARCHAR(500)"))
                print("[MIGRATION] Added 'avatar_url' column.")
            if "auth_provider" not in farmer_cols:
                print("[MIGRATION] Adding 'auth_provider' column to 'farmers' table...")
                conn.execute(text("ALTER TABLE farmers ADD COLUMN auth_provider VARCHAR(50) DEFAULT 'local'"))
                print("[MIGRATION] Added 'auth_provider' column.")

    print("[MIGRATION] Migration completed successfully with zero data loss.")


if __name__ == "__main__":
    run_migration()
