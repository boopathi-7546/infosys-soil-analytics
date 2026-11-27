"""Shared pytest configuration ensuring isolated temporary SQLite database and upload dir."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="soilsense_pytest_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test_suite.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"

import pytest  # noqa: E402
from app.database import Base, SessionLocal, engine, get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def init_isolated_test_db():
    """Ensure all tables exist on the isolated temporary SQLite database."""
    Base.metadata.create_all(bind=engine)
    yield


def _override_get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db

