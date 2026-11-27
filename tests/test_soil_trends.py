"""Tests for Historical Soil Trend Analysis service and FastAPI endpoint."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.database import SessionLocal
from app.main import app
from app.models import Field, SoilTest
from app.services.soil_trends import compute_historical_soil_trends


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _register_farmer(client: TestClient, name: str, email: str, password: str = "password123") -> dict:
    res = client.post("/api/auth/register", json={
        "full_name": name,
        "email": email,
        "password": password,
    })
    if res.status_code == 400 and "Email already registered" in res.text:
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        token = login.json()["access_token"]
    else:
        assert res.status_code == 201, res.text
        token = res.json()["access_token"]
    return {"headers": {"Authorization": f"Bearer {token}"}}


def test_trends_requires_authentication(client):
    """Unauthenticated trend requests must be rejected with HTTP 401."""
    assert client.get("/api/soil-tests/trends").status_code == 401


def test_trends_empty_and_single_record_history(client):
    """Empty history and 1-record history return insufficient_data safely."""
    farmer = _register_farmer(client, "Trend Empty Farmer", "trend.empty@example.com")

    # 0 records
    res_empty = client.get("/api/soil-tests/trends", headers=farmer["headers"])
    assert res_empty.status_code == 200
    body = res_empty.json()
    assert body["analysis_type"] == "historical_trend_analysis"
    assert body["is_ml_forecast"] is False
    assert body["disclaimer"] == "Historical Soil Trend Analysis – Not an ML Forecast"
    assert body["total_records"] == 0
    assert body["status"] == "insufficient_data"

    # 1 record
    field_id = client.post("/api/fields", json={"name": "Single Record Plot"}, headers=farmer["headers"]).json()["id"]
    post_res = client.post(
        f"/api/fields/{field_id}/soil-tests",
        json={
            "nitrogen": 120.0,
            "phosphorus": 30.0,
            "potassium": 180.0,
            "ph": 6.5,
            "organic_matter": 2.0,
            "moisture": 40.0,
        },
        headers=farmer["headers"],
    )
    assert post_res.status_code == 201

    res_one = client.get(f"/api/soil-tests/trends?field_id={field_id}", headers=farmer["headers"])
    assert res_one.status_code == 200
    one_body = res_one.json()
    assert one_body["total_records"] == 1
    assert one_body["status"] == "insufficient_data"
    n_param = next(p for p in one_body["parameters"] if p["parameter"] == "nitrogen")
    assert n_param["status"] == "insufficient_data"
    assert n_param["comparable_points"] == 1
    assert n_param["change"] is None


def test_trends_chronological_ordering_and_farmer_isolation(client):
    """Trends sort chronologically (including duplicate timestamps) and isolate farmers."""
    farmer_a = _register_farmer(client, "Trend Farmer A", "trend.a@example.com")
    farmer_b = _register_farmer(client, "Trend Farmer B", "trend.b@example.com")

    field_a = client.post("/api/fields", json={"name": "Chrono Plot A"}, headers=farmer_a["headers"]).json()["id"]
    base_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    db = SessionLocal()
    try:
        # Insert out of order + duplicate timestamp to test deterministic (tested_at, id) sorting
        t_latest = SoilTest(
            field_id=field_a,
            nitrogen=200.0,
            phosphorus=40.0,
            potassium=220.0,
            ph=6.8,
            organic_matter=2.5,
            moisture=45.0,
            tested_at=base_time + timedelta(days=30),
        )
        t_earliest = SoilTest(
            field_id=field_a,
            nitrogen=100.0,
            phosphorus=20.0,
            potassium=180.0,
            ph=6.2,
            organic_matter=1.5,
            moisture=35.0,
            tested_at=base_time,
        )
        t_dup = SoilTest(
            field_id=field_a,
            nitrogen=150.0,
            phosphorus=30.0,
            potassium=200.0,
            ph=6.5,
            organic_matter=2.0,
            moisture=40.0,
            tested_at=base_time,
        )
        db.add_all([t_latest, t_earliest, t_dup])
        db.commit()
    finally:
        db.close()

    res_a = client.get(f"/api/soil-tests/trends?field_id={field_a}", headers=farmer_a["headers"])
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["status"] == "available"
    assert data_a["total_records"] == 3

    n_trend = next(p for p in data_a["parameters"] if p["parameter"] == "nitrogen")
    assert n_trend["earliest_value"] == 100.0
    assert n_trend["latest_value"] == 200.0
    assert n_trend["change"] == 100.0
    assert n_trend["percent_change"] == 100.0
    assert n_trend["status"] == "increasing"
    assert [pt["value"] for pt in n_trend["points"]] == [100.0, 150.0, 200.0]

    # Farmer B cannot access Farmer A's field trends -> 404
    res_idor = client.get(f"/api/soil-tests/trends?field_id={field_a}", headers=farmer_b["headers"])
    assert res_idor.status_code == 404


def test_trends_handles_missing_values_and_incompatible_units():
    """Missing values are skipped safely; incompatible nutrient units withhold misleading trends."""
    t1 = SimpleNamespace(
        id=1,
        tested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        nitrogen=100.0,
        nitrogen_unit="kg/ha",
        phosphorus=20.0,
        phosphorus_unit="mg/kg",  # incompatible with kg/ha
        potassium=None,           # missing value
        potassium_unit="kg/ha",
        ph=6.0,
        organic_matter=1.8,
        moisture=40.0,
    )
    t2 = SimpleNamespace(
        id=2,
        tested_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        nitrogen=140.0,
        nitrogen_unit="kg/ha",
        phosphorus=35.0,
        phosphorus_unit="kg/ha",
        potassium=200.0,
        potassium_unit="kg/ha",
        ph=6.4,
        organic_matter=None,      # missing value -> OM has only 1 valid point
        moisture=45.0,
    )

    out = compute_historical_soil_trends([t1, t2], field_id=99, field_name="Unit Safety Plot")
    assert out.is_ml_forecast is False
    assert out.status == "available"

    by_param = {p.parameter: p for p in out.parameters}
    # Nitrogen has 2 comparable kg/ha points -> increasing
    assert by_param["nitrogen"].status == "increasing"
    assert by_param["nitrogen"].change == 40.0

    # Phosphorus has incompatible mg/kg vs kg/ha -> withheld
    assert by_param["phosphorus"].status == "incompatible_units"
    assert by_param["phosphorus"].change is None
    assert by_param["phosphorus"].points == []

    # Potassium & Organic Matter each have only 1 valid value -> insufficient_data
    assert by_param["potassium"].status == "insufficient_data"
    assert by_param["organic_matter"].status == "insufficient_data"

    # Health score requires all 6 parameters valid -> skipped when any is None -> insufficient_data
    assert by_param["health_score"].status == "insufficient_data"

