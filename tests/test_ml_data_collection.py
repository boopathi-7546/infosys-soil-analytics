"""Tests for Phase 1 ML-ready soil-test metadata, additive migration, and image pairing."""
import io
import os
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, inspect, text

from app.database import SessionLocal
from app.main import app
from app.models import AnalysisResult, SoilImage, SoilTest
from scripts.migrate_add_ml_collection_fields import run_migration


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _register_farmer(client: TestClient, name: str, email: str, password: str = "password123") -> dict:
    res = client.post("/api/auth/register", json={"full_name": name, "email": email, "password": password})
    if res.status_code == 400 and "Email already registered" in res.text:
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        token = login.json()["access_token"]
    else:
        assert res.status_code == 201, res.text
        token = res.json()["access_token"]
    return {"headers": {"Authorization": f"Bearer {token}"}}


def _make_jpeg_bytes(seed: int = 11) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, size=(224, 224, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_ml_collection_migration_is_idempotent_and_preserves_legacy_rows():
    """Additive migration adds new columns/indexes to legacy SQLite DB without altering existing rows."""
    tmp_dir = tempfile.mkdtemp(prefix="ml_mig_")
    db_path = os.path.join(tmp_dir, "legacy.db")
    legacy_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    with legacy_engine.begin() as conn:
        conn.execute(text("CREATE TABLE fields (id INTEGER PRIMARY KEY, name VARCHAR(120))"))
        conn.execute(text("INSERT INTO fields (id, name) VALUES (1, 'Legacy Field')"))
        conn.execute(text(
            "CREATE TABLE soil_tests ("
            "id INTEGER PRIMARY KEY, field_id INTEGER NOT NULL, "
            "nitrogen FLOAT NOT NULL, phosphorus FLOAT NOT NULL, potassium FLOAT NOT NULL, "
            "ph FLOAT NOT NULL, organic_matter FLOAT NOT NULL, moisture FLOAT NOT NULL, "
            "notes VARCHAR(500), tested_at DATETIME)"
        ))
        conn.execute(text(
            "INSERT INTO soil_tests (id, field_id, nitrogen, phosphorus, potassium, ph, organic_matter, moisture, notes) "
            "VALUES (1, 1, 120, 25, 180, 6.5, 2.1, 35, 'legacy row')"
        ))
        conn.execute(text(
            "CREATE TABLE soil_images ("
            "id INTEGER PRIMARY KEY, field_id INTEGER NOT NULL, file_path VARCHAR(500) NOT NULL, "
            "original_filename VARCHAR(255), original_format VARCHAR(10) NOT NULL, "
            "size_bytes INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, "
            "blur_score FLOAT NOT NULL, uploaded_at DATETIME)"
        ))

    # Run twice to verify idempotency
    run_migration(legacy_engine)
    run_migration(legacy_engine)

    insp = inspect(legacy_engine)
    st_cols = {c["name"] for c in insp.get_columns("soil_tests")}
    si_cols = {c["name"] for c in insp.get_columns("soil_images")}
    for expected in (
        "nitrogen_unit", "phosphorus_unit", "potassium_unit", "testing_method",
        "sample_depth_cm", "bulk_density", "bulk_density_unit",
        "sample_collected_at", "verification_status",
    ):
        assert expected in st_cols
    assert "soil_test_id" in si_cols

    with legacy_engine.connect() as conn:
        row = conn.execute(text("SELECT nitrogen, notes, verification_status FROM soil_tests WHERE id = 1")).fetchone()
        assert row == (120.0, "legacy row", None)

    legacy_engine.dispose()


def test_create_soil_test_with_metadata_and_pair_image(client):
    """Create soil test with ML metadata, unit validation, and image pairing."""
    farmer = _register_farmer(client, "ML Collector", "ml.collector@example.com")
    field_id = client.post("/api/fields", json={"name": "ML Paired Plot"}, headers=farmer["headers"]).json()["id"]

    img_res = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("plot.jpg", _make_jpeg_bytes(42), "image/jpeg")},
        headers=farmer["headers"],
    )
    assert img_res.status_code == 201
    image_id = img_res.json()["image"]["id"]

    # Run prediction first so AnalysisResult exists without soil_test_id
    pred_res = client.post(f"/api/images/{image_id}/predict", headers=farmer["headers"])
    assert pred_res.status_code == 200
    assert pred_res.json()["soil_test_id"] is None

    # Create soil test paired to image_id
    test_res = client.post(
        f"/api/fields/{field_id}/soil-tests",
        json={
            "nitrogen": 210.0,
            "phosphorus": 38.0,
            "potassium": 240.0,
            "nitrogen_unit": "mg/kg",
            "phosphorus_unit": "ppm",
            "potassium_unit": "kg/ha",
            "ph": 6.7,
            "organic_matter": 2.3,
            "moisture": 38.0,
            "testing_method": "Olsen P / KMnO4 / NH4OAc",
            "sample_depth_cm": 15.0,
            "bulk_density": 1.32,
            "bulk_density_unit": "g/cm3",
            "sample_collected_at": "2026-11-27T10:00:00Z",
            "verification_status": "laboratory_verified",
            "image_id": image_id,
            "notes": "Paired sample",
        },
        headers=farmer["headers"],
    )
    assert test_res.status_code == 201, test_res.text
    data = test_res.json()
    assert data["nitrogen_unit"] == "mg/kg"
    assert data["phosphorus_unit"] == "ppm"
    assert data["potassium_unit"] == "kg/ha"
    assert data["testing_method"] == "Olsen P / KMnO4 / NH4OAc"
    assert data["sample_depth_cm"] == 15.0
    assert data["bulk_density"] == 1.32
    assert data["bulk_density_unit"] == "g/cm3"
    assert data["verification_status"] == "laboratory_verified"
    assert data["data_provenance"] == "Laboratory Verified"
    assert data["paired_image_ids"] == [image_id]

    # Verify image metadata & analysis_results now carry soil_test_id
    img_meta = client.get(f"/api/images/{image_id}", headers=farmer["headers"]).json()
    assert img_meta["soil_test_id"] == data["id"]

    pred_after = client.get(f"/api/images/{image_id}/prediction", headers=farmer["headers"]).json()
    assert pred_after["soil_test_id"] == data["id"]


def test_soil_test_unit_and_pairing_validation_and_ownership(client):
    """Validate units, bulk density bounds, field matching, duplicate pairing, and IDOR."""
    farmer_a = _register_farmer(client, "Pair Owner A", "pair.a@example.com")
    farmer_b = _register_farmer(client, "Pair Owner B", "pair.b@example.com")

    f_a1 = client.post("/api/fields", json={"name": "Field A1"}, headers=farmer_a["headers"]).json()["id"]
    f_a2 = client.post("/api/fields", json={"name": "Field A2"}, headers=farmer_a["headers"]).json()["id"]
    f_b1 = client.post("/api/fields", json={"name": "Field B1"}, headers=farmer_b["headers"]).json()["id"]

    img_a1 = client.post(
        f"/api/fields/{f_a1}/images",
        files={"file": ("a1.jpg", _make_jpeg_bytes(1), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]
    img_b1 = client.post(
        f"/api/fields/{f_b1}/images",
        files={"file": ("b1.jpg", _make_jpeg_bytes(2), "image/jpeg")},
        headers=farmer_b["headers"],
    ).json()["image"]["id"]

    # Invalid unit -> 422
    bad_unit = client.post(
        f"/api/fields/{f_a1}/soil-tests",
        json={
            "nitrogen": 100, "phosphorus": 20, "potassium": 150,
            "nitrogen_unit": "ounces/acre",
            "ph": 6.5, "organic_matter": 2.0, "moisture": 30,
        },
        headers=farmer_a["headers"],
    )
    assert bad_unit.status_code == 422

    # Invalid bulk density for g/cm3 -> 422
    bad_bd = client.post(
        f"/api/fields/{f_a1}/soil-tests",
        json={
            "nitrogen": 100, "phosphorus": 20, "potassium": 150,
            "bulk_density": 15.0, "bulk_density_unit": "g/cm3",
            "ph": 6.5, "organic_matter": 2.0, "moisture": 30,
        },
        headers=farmer_a["headers"],
    )
    assert bad_bd.status_code == 422

    # Mismatched field (img_a1 belongs to f_a1, creating test on f_a2) -> 422
    mismatch = client.post(
        f"/api/fields/{f_a2}/soil-tests",
        json={
            "nitrogen": 100, "phosphorus": 20, "potassium": 150,
            "ph": 6.5, "organic_matter": 2.0, "moisture": 30,
            "image_id": img_a1,
        },
        headers=farmer_a["headers"],
    )
    assert mismatch.status_code == 422

    # Farmer A cannot pair Farmer B's image -> 404
    idor_pair = client.post(
        f"/api/fields/{f_a1}/soil-tests",
        json={
            "nitrogen": 100, "phosphorus": 20, "potassium": 150,
            "ph": 6.5, "organic_matter": 2.0, "moisture": 30,
            "image_id": img_b1,
        },
        headers=farmer_a["headers"],
    )
    assert idor_pair.status_code == 404

    # Create valid test on f_a1 without optional fields (defaults to user_entered_unverified)
    t1 = client.post(
        f"/api/fields/{f_a1}/soil-tests",
        json={"nitrogen": 100, "phosphorus": 20, "potassium": 150, "ph": 6.5, "organic_matter": 2.0, "moisture": 30},
        headers=farmer_a["headers"],
    )
    assert t1.status_code == 201
    t1_id = t1.json()["id"]
    assert t1.json()["verification_status"] == "user_entered_unverified"
    assert t1.json()["sample_depth_cm"] is None
    assert t1.json()["bulk_density"] is None

    # Pair via POST /api/images/{image_id}/pair-soil-test
    pair_ep = client.post(
        f"/api/images/{img_a1}/pair-soil-test",
        json={"soil_test_id": t1_id},
        headers=farmer_a["headers"],
    )
    assert pair_ep.status_code == 200
    assert pair_ep.json()["soil_test_id"] == t1_id

    # Cannot pair img_a1 to a second soil test -> 409
    t2_id = client.post(
        f"/api/fields/{f_a1}/soil-tests",
        json={"nitrogen": 110, "phosphorus": 22, "potassium": 160, "ph": 6.6, "organic_matter": 2.1, "moisture": 32},
        headers=farmer_a["headers"],
    ).json()["id"]
    conflict = client.post(
        f"/api/images/{img_a1}/pair-soil-test",
        json={"soil_test_id": t2_id},
        headers=farmer_a["headers"],
    )
    assert conflict.status_code == 409

