import base64
import io
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # Register default test farmer for API regression testing
        reg = c.post("/api/auth/register", json={
            "full_name": "Test Farmer",
            "email": "test.farmer.api@example.com",
            "password": "testpassword123",
        })
        if reg.status_code == 201:
            token = reg.json()["access_token"]
        else:
            login = c.post("/api/auth/login", json={
                "email": "test.farmer.api@example.com",
                "password": "testpassword123",
            })
            token = login.json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.fixture()
def field_id(client):
    r = client.post("/fields", json={"name": "North plot", "farmer_name": "Ravi", "location": "Erode"})
    assert r.status_code == 201
    return r.json()["id"]


def make_image(fmt="JPEG", size=(400, 300), mode="RGB", textured=True):
    rng = np.random.default_rng(0)
    arr = (
        (rng.integers(60, 200, (size[1], size[0], 3))).astype("uint8")
        if textured
        else np.full((size[1], size[0], 3), 120, "uint8")
    )
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_health(client):
    res = client.get("/health").json()
    assert res["status"] == "ok"
    assert "db" in res


# ---- image upload ----
def test_upload_jpeg(client, field_id):
    r = client.post(f"/fields/{field_id}/images", files={"file": ("a.jpg", make_image(), "image/jpeg")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["image"]["original_format"] == "JPEG"
    assert body["image"]["width"] == 400


def test_upload_png_with_alpha_is_normalised(client, field_id):
    r = client.post(
        f"/fields/{field_id}/images",
        files={"file": ("a.png", make_image("PNG", mode="RGBA"), "image/png")},
    )
    assert r.status_code == 201, r.text
    img = client.get(f"/images/{r.json()['image']['id']}/file")
    assert Image.open(io.BytesIO(img.content)).format == "JPEG"


def test_large_image_is_downscaled(client, field_id):
    r = client.post(
        f"/fields/{field_id}/images",
        files={"file": ("big.jpg", make_image(size=(3000, 2000)), "image/jpeg")},
    )
    assert r.status_code == 201
    assert max(r.json()["image"]["width"], r.json()["image"]["height"]) == 1600


def test_blurry_image_gets_warning(client, field_id):
    r = client.post(
        f"/fields/{field_id}/images",
        files={"file": ("flat.jpg", make_image(textured=False), "image/jpeg")},
    )
    assert r.status_code == 201
    assert any("blurry" in w for w in r.json()["warnings"])


def test_reject_fake_image(client, field_id):
    r = client.post(f"/fields/{field_id}/images", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_reject_wrong_content_type(client, field_id):
    r = client.post(f"/fields/{field_id}/images", files={"file": ("a.gif", make_image("GIF"), "image/gif")})
    assert r.status_code == 422


def test_reject_tiny_image(client, field_id):
    r = client.post(
        f"/fields/{field_id}/images",
        files={"file": ("t.jpg", make_image(size=(50, 50)), "image/jpeg")},
    )
    assert r.status_code == 422


def test_reject_empty_file(client, field_id):
    r = client.post(f"/fields/{field_id}/images", files={"file": ("e.jpg", b"", "image/jpeg")})
    assert r.status_code == 422


def test_reject_too_large(client, field_id, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)  # any non-empty file exceeds the limit
    r = client.post(f"/fields/{field_id}/images", files={"file": ("a.jpg", make_image(), "image/jpeg")})
    assert r.status_code == 413


def test_upload_unknown_field(client):
    r = client.post("/fields/99999/images", files={"file": ("a.jpg", make_image(), "image/jpeg")})
    assert r.status_code == 404


def test_list_and_preprocessed(client, field_id):
    up = client.post(f"/fields/{field_id}/images", files={"file": ("a.jpg", make_image(), "image/jpeg")}).json()
    lst = client.get(f"/fields/{field_id}/images").json()
    assert len(lst) >= 1
    pre = client.get(f"/images/{up['image']['id']}/preprocessed")
    assert pre.status_code == 200
    assert Image.open(io.BytesIO(pre.content)).size == (224, 224)


# ---- Module 1: Live camera upload-base64 tests ----
def test_upload_base64_camera_capture(client, field_id):
    raw_jpg = make_image("JPEG")
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(raw_jpg).decode('utf-8')}"
    r = client.post("/api/images/upload-base64", json={"field_id": field_id, "image_data": b64_str})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["image"]["field_id"] == field_id
    assert body["image"]["original_format"] == "JPEG"


def test_upload_base64_invalid_string(client, field_id):
    r = client.post("/api/images/upload-base64", json={"field_id": field_id, "image_data": "not-base64!!!"})
    assert r.status_code == 422


def test_upload_base64_not_an_image(client, field_id):
    fake_b64 = base64.b64encode(b"this is plain text").decode("utf-8")
    r = client.post("/api/images/upload-base64", json={"field_id": field_id, "image_data": fake_b64})
    assert r.status_code == 422


# ---- Module 4: Preprocessing inspection endpoint ----
def test_image_preprocess_inspection(client, field_id):
    up = client.post(f"/fields/{field_id}/images", files={"file": ("inspect.jpg", make_image(), "image/jpeg")}).json()
    img_id = up["image"]["id"]
    r = client.get(f"/api/images/{img_id}/preprocess")
    assert r.status_code == 200
    data = r.json()
    assert data["image_id"] == img_id
    assert data["width"] == 224
    assert data["height"] == 224
    assert len(data["mean_rgb"]) == 3
    assert data["quality_ok"] is True


# ---- soil tests ----
GOOD = {"nitrogen": 250, "phosphorus": 18, "potassium": 190, "ph": 6.8, "organic_matter": 0.6, "moisture": 22}


def test_soil_test_create_and_list(client, field_id):
    r = client.post(f"/fields/{field_id}/soil-tests", json=GOOD)
    assert r.status_code == 201, r.text
    client.post(f"/fields/{field_id}/soil-tests", json={**GOOD, "ph": 7.2})
    lst = client.get(f"/fields/{field_id}/soil-tests").json()
    assert len(lst) >= 2 and lst[0]["ph"] == 7.2  # newest first
    assert client.get(f"/soil-tests/{r.json()['id']}").json()["nitrogen"] == 250


@pytest.mark.parametrize(
    "bad",
    [
        {"ph": 15},
        {"ph": -1},
        {"nitrogen": -5},
        {"nitrogen": 600},
        {"phosphorus": 200},
        {"potassium": 900},
        {"moisture": 101},
        {"organic_matter": 120},
        {"potassium": "abc"},
    ],
)
def test_soil_test_validation(client, field_id, bad):
    assert client.post(f"/fields/{field_id}/soil-tests", json={**GOOD, **bad}).status_code == 422


def test_soil_test_missing_field_value(client, field_id):
    body = {k: v for k, v in GOOD.items() if k != "ph"}
    assert client.post(f"/fields/{field_id}/soil-tests", json=body).status_code == 422


def test_soil_test_unknown_field(client):
    assert client.post("/fields/99999/soil-tests", json=GOOD).status_code == 404


def test_soil_test_advisory(client, field_id):
    r = client.post(f"/fields/{field_id}/soil-tests", json=GOOD)
    test_id = r.json()["id"]
    adv = client.get(f"/api/soil-tests/{test_id}/advisory")
    assert adv.status_code == 200
    data = adv.json()
    assert "health_score" in data
    assert "recommendations" in data
    assert "suitable_crops" in data
    assert len(data["suitable_crops"]) > 0


# ---- Module 5: Dataset status endpoint ----
def test_dataset_status_endpoint(client):
    r = client.get("/api/dataset/status")
    assert r.status_code == 200
    data = r.json()
    assert "soil_types" in data
    assert "total_raw_images" in data
    assert "sources" in data
    assert len(data["sources"]) > 0


# ---- Field stats & management ----
def test_field_stats(client, field_id):
    client.post(f"/fields/{field_id}/soil-tests", json=GOOD)
    r = client.get(f"/api/fields/{field_id}/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["field_id"] == field_id
    assert data["total_soil_tests"] >= 1
    assert data["avg_nitrogen"] == 250.0
