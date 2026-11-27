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
    if res.status_code == 400 and "Email already registered" in res.text:
        login_res = client.post("/api/auth/login", json={
            "email": email,
            "password": password,
        })
        assert login_res.status_code == 200, login_res.text
        data = login_res.json()
        return {
            "farmer": data["farmer"],
            "token": data["access_token"],
            "headers": {"Authorization": f"Bearer {data['access_token']}"},
        }
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


def test_soil_test_detail_view_and_advisory_isolation(client):
    """
    Verify detailed view of a historical soil test:
    - Returns full stored parameters (N, P, K, units, pH, OM, moisture)
    - Returns enriched metadata (field_name, sequential test_number, display_id)
    - Returns optional fields (electrical_conductivity, organic_carbon, etc.) as None if unrecorded
    - Returns advisory strictly calculated for the selected test
    - Enforces IDOR protection on both soil-test details and advisory
    - Returns 404 for non-existent test IDs
    """
    farmer1 = register_farmer(client, "Detail Farmer 1", "detail1@soilanalytics.io")
    farmer2 = register_farmer(client, "Detail Farmer 2", "detail2@soilanalytics.io")

    # Farmer 1 creates field and submits two tests with different values
    f1 = client.post("/api/fields", json={"name": "Thanjavur Delta 1"}, headers=farmer1["headers"]).json()["id"]

    # Test A: Low pH (acidic)
    t1_res = client.post(f"/api/fields/{f1}/soil-tests", json={
        "nitrogen": 80.0,
        "phosphorus": 15.0,
        "potassium": 90.0,
        "ph": 5.2,
        "organic_matter": 0.8,
        "moisture": 18.0,
        "notes": "Sample from north ridge",
    }, headers=farmer1["headers"])
    assert t1_res.status_code == 201
    t1_id = t1_res.json()["id"]

    # Test B: Neutral pH, high fertility
    t2_res = client.post(f"/api/fields/{f1}/soil-tests", json={
        "nitrogen": 220.0,
        "phosphorus": 45.0,
        "potassium": 210.0,
        "ph": 6.8,
        "organic_matter": 2.5,
        "moisture": 45.0,
        "notes": "Sample after organic composting",
    }, headers=farmer1["headers"])
    assert t2_res.status_code == 201
    t2_id = t2_res.json()["id"]

    # 1. Fetch details for Test A
    res_a = client.get(f"/api/soil-tests/{t1_id}", headers=farmer1["headers"])
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["id"] == t1_id
    assert data_a["field_id"] == f1
    assert data_a["field_name"] == "Thanjavur Delta 1"
    assert data_a["nitrogen"] == 80.0
    assert data_a["nitrogen_unit"] == "kg/ha"
    assert data_a["phosphorus"] == 15.0
    assert data_a["phosphorus_unit"] == "kg/ha"
    assert data_a["potassium"] == 90.0
    assert data_a["potassium_unit"] == "kg/ha"
    assert data_a["ph"] == 5.2
    assert data_a["organic_matter"] == 0.8
    assert data_a["moisture"] == 18.0
    assert data_a["notes"] == "Sample from north ridge"
    assert data_a["health_score"] is not None
    assert data_a["electrical_conductivity"] is None
    assert data_a["organic_carbon"] is None

    # 2. Fetch advisory specifically for Test A
    adv_a = client.get(f"/api/soil-tests/{t1_id}/advisory", headers=farmer1["headers"])
    assert adv_a.status_code == 200
    adv_data_a = adv_a.json()
    assert adv_data_a["soil_test_id"] == t1_id
    # Test A had acidic pH 5.2 and low N (< 100) -> warnings or recs must reflect this
    assert any("acidic" in w.lower() for w in adv_data_a["warnings"]) or any("lime" in r.lower() for r in adv_data_a["recommendations"])

    # 3. Fetch advisory specifically for Test B
    adv_b = client.get(f"/api/soil-tests/{t2_id}/advisory", headers=farmer1["headers"])
    assert adv_b.status_code == 200
    adv_data_b = adv_b.json()
    assert adv_data_b["soil_test_id"] == t2_id
    assert adv_data_b["health_score"] > adv_data_a["health_score"]

    # 4. Cross-Farmer IDOR Protection
    assert client.get(f"/api/soil-tests/{t1_id}", headers=farmer2["headers"]).status_code == 404
    assert client.get(f"/api/soil-tests/{t1_id}/advisory", headers=farmer2["headers"]).status_code == 404

    # 5. Non-existent test ID -> 404
    assert client.get("/api/soil-tests/999999", headers=farmer1["headers"]).status_code == 404
    assert client.get("/api/soil-tests/999999/advisory", headers=farmer1["headers"]).status_code == 404

    # 6. Unauthenticated requests -> 401
    assert client.get(f"/api/soil-tests/{t1_id}").status_code == 401
    assert client.get(f"/api/soil-tests/{t1_id}/advisory").status_code == 401


def test_unclaimed_field_and_soil_test_idor_blocked(client):
    """Unclaimed records (farmer_id IS NULL) must not be accessible to arbitrary farmers."""
    farmer = register_farmer(client, "Unclaimed Guard Farmer", "unclaimed.guard@example.com")

    db = SessionLocal()
    try:
        unclaimed_field = Field(name="Legacy Unassigned Plot", location="Unknown", farmer_id=None)
        db.add(unclaimed_field)
        db.commit()
        db.refresh(unclaimed_field)

        unclaimed_test = SoilTest(
            field_id=unclaimed_field.id,
            nitrogen=150.0,
            phosphorus=30.0,
            potassium=150.0,
            ph=6.5,
            organic_matter=2.0,
            moisture=35.0,
        )
        db.add(unclaimed_test)
        db.commit()
        db.refresh(unclaimed_test)
        uf_id = unclaimed_field.id
        ut_id = unclaimed_test.id
    finally:
        db.close()

    assert client.get(f"/api/fields/{uf_id}", headers=farmer["headers"]).status_code == 404
    assert client.delete(f"/api/fields/{uf_id}", headers=farmer["headers"]).status_code == 404
    assert client.get(f"/api/soil-tests/{ut_id}", headers=farmer["headers"]).status_code == 404
    assert client.get(f"/api/soil-tests/{ut_id}/advisory", headers=farmer["headers"]).status_code == 404


def test_frontend_xss_and_scientific_integrity_guards(client):
    """Frontend HTML must include XSS/URL escaping helpers and omit fabricated metadata fallbacks."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "function escHtml(" in html
    assert "function safeUrl(" in html
    assert "function fmtNum(" in html
    assert "const safe = safeUrl(url);" in html
    for cls_name in ("alluvial_soil", "black_soil", "clay_soil", "laterite_soil", "red_soil", "sandy_soil"):
        assert cls_name in html
    for forbidden in (
        "Alkaline permanganate / Lab",
        "Olsen / Bray / Lab",
        "Neutral 1N NH4OAc / Lab",
        "1:2.5 soil-water suspension",
        "Standard Laboratory Measurement",
        "Gravimetric / Field Sensor",
        "ICAR / TNAU Rule Engine",
        "best_epoch || 7",
    ):
        assert forbidden not in html, f"Forbidden fabricated fallback still present: {forbidden}"
