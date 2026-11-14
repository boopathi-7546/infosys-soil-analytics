"""Comprehensive test suite for Farmer Authentication, JWT verification, and IDOR protection."""
import io
import os
import tempfile
from datetime import timedelta

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test_auth.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads_auth"

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.database import Base, engine
from app.main import app
from app.models import Farmer, Field, SoilImage, SoilTest
from app.services.auth import create_access_token


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def make_test_image():
    arr = np.full((200, 200, 3), 120, dtype="uint8")
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Registration Tests
# ---------------------------------------------------------------------------

def test_farmer_registration_success(client):
    res = client.post("/api/auth/register", json={
        "full_name": "Ravi Kumar",
        "email": "ravi.kumar@example.com",
        "password": "securepassword123",
    })
    assert res.status_code == 201, res.text
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["farmer"]["email"] == "ravi.kumar@example.com"
    assert data["farmer"]["full_name"] == "Ravi Kumar"
    assert "password" not in data["farmer"]


def test_farmer_registration_duplicate_email(client):
    client.post("/api/auth/register", json={
        "full_name": "Ravi Kumar",
        "email": "ravi.dup@example.com",
        "password": "password123",
    })
    res = client.post("/api/auth/register", json={
        "full_name": "Ravi Duplicate",
        "email": "RAVI.DUP@example.com",  # case-insensitive match
        "password": "differentpassword",
    })
    assert res.status_code == 400
    assert "already registered" in res.json()["detail"].lower()


def test_farmer_registration_invalid_email(client):
    res = client.post("/api/auth/register", json={
        "full_name": "Bad Email",
        "email": "not-an-email",
        "password": "password123",
    })
    assert res.status_code == 422


def test_farmer_registration_short_password(client):
    res = client.post("/api/auth/register", json={
        "full_name": "Short Pass",
        "email": "short@example.com",
        "password": "123",
    })
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 2. Login Tests
# ---------------------------------------------------------------------------

def test_farmer_login_success(client):
    client.post("/api/auth/register", json={
        "full_name": "Priya Selvam",
        "email": "priya@example.com",
        "password": "mysecretpassword",
    })
    res = client.post("/api/auth/login", json={
        "email": "priya@example.com",
        "password": "mysecretpassword",
    })
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["farmer"]["email"] == "priya@example.com"


def test_farmer_login_incorrect_password(client):
    client.post("/api/auth/register", json={
        "full_name": "User BadPass",
        "email": "badpass@example.com",
        "password": "correctpassword",
    })
    res = client.post("/api/auth/login", json={
        "email": "badpass@example.com",
        "password": "wrongpassword",
    })
    assert res.status_code == 401
    assert "invalid email or password" in res.json()["detail"].lower()


def test_farmer_login_nonexistent_email(client):
    res = client.post("/api/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "somepassword",
    })
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# 3. JWT Signature and Expiry Tests
# ---------------------------------------------------------------------------

def test_unauthenticated_request_rejected(client):
    res = client.get("/api/fields")
    assert res.status_code == 401
    assert "authentication required" in res.json()["detail"].lower()


def test_invalid_jwt_rejected(client):
    headers = {"Authorization": "Bearer invalid.token.value"}
    res = client.get("/api/fields", headers=headers)
    assert res.status_code == 401
    assert "invalid or malformed" in res.json()["detail"].lower()


def test_expired_jwt_rejected(client):
    # Generate token that expired 1 hour ago
    expired_token = create_access_token(
        data={"sub": "999", "email": "test@example.com"},
        expires_delta=timedelta(minutes=-60),
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    res = client.get("/api/fields", headers=headers)
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4. Multi-Farmer Data Isolation (IDOR Prevention)
# ---------------------------------------------------------------------------

import uuid

@pytest.fixture()
def farmer_a(client):
    u = uuid.uuid4().hex[:8]
    res = client.post("/api/auth/register", json={
        "full_name": f"Farmer A {u}",
        "email": f"farmer.a.{u}@example.com",
        "password": "passwordA123",
    })
    token = res.json()["access_token"]
    return {"token": token, "headers": {"Authorization": f"Bearer {token}"}, "farmer": res.json()["farmer"]}


@pytest.fixture()
def farmer_b(client):
    u = uuid.uuid4().hex[:8]
    res = client.post("/api/auth/register", json={
        "full_name": f"Farmer B {u}",
        "email": f"farmer.b.{u}@example.com",
        "password": "passwordB123",
    })
    token = res.json()["access_token"]
    return {"token": token, "headers": {"Authorization": f"Bearer {token}"}, "farmer": res.json()["farmer"]}


def test_farmer_can_access_own_field_only(client, farmer_a, farmer_b):
    # Farmer A creates a field
    r1 = client.post("/api/fields", json={"name": "A's Cotton Field", "location": "Salem"}, headers=farmer_a["headers"])
    assert r1.status_code == 201
    field_a_id = r1.json()["id"]

    # Farmer B creates a field
    r2 = client.post("/api/fields", json={"name": "B's Paddy Field", "location": "Thanjavur"}, headers=farmer_b["headers"])
    assert r2.status_code == 201
    field_b_id = r2.json()["id"]

    # Farmer A lists fields -> sees only field A
    list_a = client.get("/api/fields", headers=farmer_a["headers"]).json()
    a_ids = [f["id"] for f in list_a]
    assert field_a_id in a_ids
    assert field_b_id not in a_ids

    # Farmer B lists fields -> sees only field B
    list_b = client.get("/api/fields", headers=farmer_b["headers"]).json()
    b_ids = [f["id"] for f in list_b]
    assert field_b_id in b_ids
    assert field_a_id not in b_ids

    # IDOR Test: Farmer B tries to access Farmer A's field directly by ID
    idor_res = client.get(f"/api/fields/{field_a_id}", headers=farmer_b["headers"])
    assert idor_res.status_code == 404

    # IDOR Test: Farmer B tries to delete Farmer A's field
    del_res = client.delete(f"/api/fields/{field_a_id}", headers=farmer_b["headers"])
    assert del_res.status_code == 404

    # IDOR Test: Farmer B tries to get stats for Farmer A's field
    stats_res = client.get(f"/api/fields/{field_a_id}/stats", headers=farmer_b["headers"])
    assert stats_res.status_code == 404


def test_soil_test_isolation_across_farmers(client, farmer_a, farmer_b):
    # Farmer A creates field
    field_a = client.post("/api/fields", json={"name": "A's Groundnut Plot"}, headers=farmer_a["headers"]).json()
    
    # Farmer A submits soil test
    test_data = {"nitrogen": 220, "phosphorus": 30, "potassium": 180, "ph": 6.5, "organic_matter": 1.2, "moisture": 30}
    test_res = client.post(f"/api/fields/{field_a['id']}/soil-tests", json=test_data, headers=farmer_a["headers"])
    assert test_res.status_code == 201
    test_id = test_res.json()["id"]

    # Farmer B tries to submit a test to Farmer A's field -> 404
    bad_submit = client.post(f"/api/fields/{field_a['id']}/soil-tests", json=test_data, headers=farmer_b["headers"])
    assert bad_submit.status_code == 404

    # Farmer B tries to read Farmer A's soil test -> 404
    bad_read = client.get(f"/api/soil-tests/{test_id}", headers=farmer_b["headers"])
    assert bad_read.status_code == 404

    # Farmer B tries to get advisory for Farmer A's soil test -> 404
    bad_advisory = client.get(f"/api/soil-tests/{test_id}/advisory", headers=farmer_b["headers"])
    assert bad_advisory.status_code == 404


def test_image_isolation_across_farmers(client, farmer_a, farmer_b):
    # Farmer A creates field
    field_a = client.post("/api/fields", json={"name": "A's Image Field"}, headers=farmer_a["headers"]).json()
    
    # Farmer A uploads image
    img_bytes = make_test_image()
    up_res = client.post(
        f"/api/fields/{field_a['id']}/images",
        files={"file": ("soil_a.jpg", img_bytes, "image/jpeg")},
        headers=farmer_a["headers"],
    )
    assert up_res.status_code == 201
    image_id = up_res.json()["image"]["id"]

    # Farmer B tries to upload an image to Farmer A's field -> 404
    bad_up = client.post(
        f"/api/fields/{field_a['id']}/images",
        files={"file": ("soil_b.jpg", img_bytes, "image/jpeg")},
        headers=farmer_b["headers"],
    )
    assert bad_up.status_code == 404

    # Farmer B tries to read metadata of Farmer A's image -> 404
    bad_meta = client.get(f"/api/images/{image_id}", headers=farmer_b["headers"])
    assert bad_meta.status_code == 404

    # Farmer B tries to download Farmer A's image file -> 404
    bad_file = client.get(f"/api/images/{image_id}/file", headers=farmer_b["headers"])
    assert bad_file.status_code == 404

    # Farmer B tries to access preprocessed image -> 404
    bad_pre = client.get(f"/api/images/{image_id}/preprocessed", headers=farmer_b["headers"])
    assert bad_pre.status_code == 404

    # Unauthenticated user tries to download image -> 401
    anon_file = client.get(f"/api/images/{image_id}/file")
    assert anon_file.status_code == 401

    # Farmer A downloads image using token query parameter -> 200
    param_file = client.get(f"/api/images/{image_id}/file?token={farmer_a['token']}")
    assert param_file.status_code == 200


def test_claim_legacy_field(client, farmer_a, farmer_b):
    # Directly insert an unassigned legacy field (farmer_id is None)
    with client.app.dependency_overrides.get(app, client) or client:
        pass
    
    from app.database import SessionLocal
    with SessionLocal() as db:
        legacy = Field(name="Unassigned Legacy Plot", farmer_id=None)
        db.add(legacy)
        db.commit()
        db.refresh(legacy)
        legacy_id = legacy.id

    # Farmer A claims the legacy field
    claim_res = client.post(f"/api/fields/{legacy_id}/claim", headers=farmer_a["headers"])
    assert claim_res.status_code == 200
    assert claim_res.json()["farmer_id"] == farmer_a["farmer"]["id"]

    # Farmer B tries to re-claim already claimed field -> 403 Forbidden
    conflict_res = client.post(f"/api/fields/{legacy_id}/claim", headers=farmer_b["headers"])
    assert conflict_res.status_code == 403
