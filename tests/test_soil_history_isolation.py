"""Tests for Account-Specific Soil Test Numbering, Full History Retrieval, and Multi-Tenant Isolation."""
import io
import os
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test_isolation.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads_isolation"

from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import Farmer, Field, SoilImage, SoilTest


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def register_farmer(client, name: str, email: str, password: str = "password123"):
    res = client.post("/api/auth/register", json={
        "full_name": name,
        "email": email,
        "password": password,
    })
    assert res.status_code == 201, res.text
    data = res.json()
    return {
        "farmer": data["farmer"],
        "token": data["access_token"],
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
    }


def make_test_image():
    arr = np.full((150, 150, 3), 110, dtype="uint8")
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test Account-Specific Numbering, Complete History, and IDOR Isolation
# ---------------------------------------------------------------------------

def test_account_specific_test_numbering_and_isolation(client):
    """
    Scenario:
    - Account 1 registers, creates a field, and submits 5 soil tests -> numbered 1 to 5.
    - Account 2 registers, creates a field, and submits 10 soil tests -> numbered 1 to 10.
    - Total 15 records in DB, each with unique primary key.
    - History retrieval displays complete history without truncation.
    - Cross-tenant IDOR checks prevent viewing or querying the other account's data.
    - Account 1 submits a 6th test -> numbered 6, Account 2 is unaffected.
    """
    # 1. Register Account 1 and Account 2
    acc1 = register_farmer(client, "Farmer Alpha", "alpha@soilanalytics.io")
    acc2 = register_farmer(client, "Farmer Beta", "beta@soilanalytics.io")

    # 2. Account 1 creates a field
    field1_res = client.post("/api/fields", json={"name": "Alpha Field 1", "location": "Sector A"}, headers=acc1["headers"])
    assert field1_res.status_code == 201
    field1_id = field1_res.json()["id"]

    # 3. Account 1 submits 5 soil tests
    acc1_tests = []
    for i in range(1, 6):
        test_payload = {
            "nitrogen": 180.0 + i * 5,
            "phosphorus": 30.0 + i,
            "potassium": 150.0 + i * 2,
            "ph": round(6.0 + i * 0.1, 2),
            "organic_matter": round(1.0 + i * 0.1, 2),
            "moisture": 25.0 + i,
        }
        r = client.post(f"/api/fields/{field1_id}/soil-tests", json=test_payload, headers=acc1["headers"])
        assert r.status_code == 201
        data = r.json()
        assert data["test_number"] == i, f"Expected test_number={i}, got {data.get('test_number')}"
        acc1_tests.append(data)

    # 4. Verify Account 1 sees all 5 tests sorted newest first
    res_field1 = client.get(f"/api/fields/{field1_id}/soil-tests", headers=acc1["headers"])
    assert res_field1.status_code == 200
    list_f1 = res_field1.json()
    assert len(list_f1) == 5, f"Expected 5 tests, got {len(list_f1)}"
    test_nums_f1 = [t["test_number"] for t in list_f1]
    assert test_nums_f1 == [5, 4, 3, 2, 1], f"Expected [5, 4, 3, 2, 1], got {test_nums_f1}"

    # Global soil-tests endpoint for Account 1 (without field_id)
    res_all1 = client.get("/api/soil-tests", headers=acc1["headers"])
    assert res_all1.status_code == 200
    list_all1 = res_all1.json()
    assert len(list_all1) == 5
    assert [t["test_number"] for t in list_all1] == [5, 4, 3, 2, 1]

    # 5. Account 2 creates a field
    field2_res = client.post("/api/fields", json={"name": "Beta Field 1", "location": "Sector B"}, headers=acc2["headers"])
    assert field2_res.status_code == 201
    field2_id = field2_res.json()["id"]

    # 6. Account 2 submits 10 soil tests -> must independently start from 1 to 10
    acc2_tests = []
    for j in range(1, 11):
        test_payload = {
            "nitrogen": 200.0 + j * 3,
            "phosphorus": 40.0 + j,
            "potassium": 180.0 + j * 2,
            "ph": round(6.5 + j * 0.05, 2),
            "organic_matter": round(1.5 + j * 0.05, 2),
            "moisture": 30.0 + j,
        }
        r = client.post(f"/api/fields/{field2_id}/soil-tests", json=test_payload, headers=acc2["headers"])
        assert r.status_code == 201
        data = r.json()
        assert data["test_number"] == j, f"Account 2 test {j} got test_number={data.get('test_number')}"
        acc2_tests.append(data)

    # 7. Verify Account 2 sees all 10 tests sorted newest first: 10 down to 1
    res_field2 = client.get(f"/api/fields/{field2_id}/soil-tests", headers=acc2["headers"])
    assert res_field2.status_code == 200
    list_f2 = res_field2.json()
    assert len(list_f2) == 10, f"Expected 10 tests, got {len(list_f2)}"
    test_nums_f2 = [t["test_number"] for t in list_f2]
    assert test_nums_f2 == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # Global soil-tests endpoint for Account 2
    res_all2 = client.get("/api/soil-tests", headers=acc2["headers"])
    assert res_all2.status_code == 200
    list_all2 = res_all2.json()
    assert len(list_all2) == 10
    assert [t["test_number"] for t in list_all2] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # 8. Check Database: All 15 records exist, each with unique database primary key ID
    with SessionLocal() as db:
        acc_tests_db = db.query(SoilTest).filter(SoilTest.field_id.in_([field1_id, field2_id])).all()
        assert len(acc_tests_db) == 15, f"Expected 15 tests for Account 1 and 2, got {len(acc_tests_db)}"
        unique_ids = {t.id for t in acc_tests_db}
        assert len(unique_ids) == 15, "Database IDs must all be unique primary keys"
        # Confirm field ownership in DB
        for t in acc_tests_db:
            if t.field_id == field1_id:
                assert t.field.farmer_id == acc1["farmer"]["id"]
            elif t.field_id == field2_id:
                assert t.field.farmer_id == acc2["farmer"]["id"]

    # 9. Strict Account Isolation & IDOR Protection:
    # Account 1 tries to access Account 2's field soil tests -> 404
    r_bad1 = client.get(f"/api/fields/{field2_id}/soil-tests", headers=acc1["headers"])
    assert r_bad1.status_code == 404

    # Account 2 tries to access Account 1's field soil tests -> 404
    r_bad2 = client.get(f"/api/fields/{field1_id}/soil-tests", headers=acc2["headers"])
    assert r_bad2.status_code == 404

    # Account 1 tries to access any of Account 2's specific tests -> 404
    for t2 in acc2_tests:
        r_idor = client.get(f"/api/soil-tests/{t2['id']}", headers=acc1["headers"])
        assert r_idor.status_code == 404

    # Account 2 tries to access any of Account 1's specific tests -> 404
    for t1 in acc1_tests:
        r_idor = client.get(f"/api/soil-tests/{t1['id']}", headers=acc2["headers"])
        assert r_idor.status_code == 404

    # Account 1 submits a 6th test -> must become test_number 6
    r_sixth = client.post(f"/api/fields/{field1_id}/soil-tests", json={
        "nitrogen": 215.0,
        "phosphorus": 36.0,
        "potassium": 162.0,
        "ph": 6.8,
        "organic_matter": 1.7,
        "moisture": 32.0,
    }, headers=acc1["headers"])
    assert r_sixth.status_code == 201
    assert r_sixth.json()["test_number"] == 6

    # Verify Account 1 now has 6 tests: [6, 5, 4, 3, 2, 1]
    res_all1_updated = client.get("/api/soil-tests", headers=acc1["headers"]).json()
    assert len(res_all1_updated) == 6
    assert [t["test_number"] for t in res_all1_updated] == [6, 5, 4, 3, 2, 1]

    # Verify Account 2 still has exactly 10 tests, completely unaffected: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    res_all2_untouched = client.get("/api/soil-tests", headers=acc2["headers"]).json()
    assert len(res_all2_untouched) == 10
    assert [t["test_number"] for t in res_all2_untouched] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def test_field_soil_photograph_isolation_and_all_images_endpoint(client):
    """Verify GET /api/images returns only the authenticated farmer's images."""
    acc_x = register_farmer(client, "Farmer X", "farmerx@soilanalytics.io")
    acc_y = register_farmer(client, "Farmer Y", "farmery@soilanalytics.io")

    # Fields
    fx = client.post("/api/fields", json={"name": "X Plot"}, headers=acc_x["headers"]).json()["id"]
    fy = client.post("/api/fields", json={"name": "Y Plot"}, headers=acc_y["headers"]).json()["id"]

    img_data = make_test_image()

    # X uploads image
    rx = client.post(
        f"/api/fields/{fx}/images",
        files={"file": ("soil_x.jpg", img_data, "image/jpeg")},
        headers=acc_x["headers"],
    )
    assert rx.status_code == 201
    img_x_id = rx.json()["image"]["id"]

    # Y uploads image
    ry = client.post(
        f"/api/fields/{fy}/images",
        files={"file": ("soil_y.jpg", img_data, "image/jpeg")},
        headers=acc_y["headers"],
    )
    assert ry.status_code == 201
    img_y_id = ry.json()["image"]["id"]

    # X queries /api/images -> should ONLY see img_x_id
    list_x = client.get("/api/images", headers=acc_x["headers"]).json()
    x_ids = [img["id"] for img in list_x]
    assert img_x_id in x_ids
    assert img_y_id not in x_ids

    # Y queries /api/images -> should ONLY see img_y_id
    list_y = client.get("/api/images", headers=acc_y["headers"]).json()
    y_ids = [img["id"] for img in list_y]
    assert img_y_id in y_ids
    assert img_x_id not in y_ids

    # IDOR check on image download
    r_bad_download = client.get(f"/api/images/{img_y_id}/file", headers=acc_x["headers"])
    assert r_bad_download.status_code == 404
