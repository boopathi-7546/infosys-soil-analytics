"""
Tests for Live CNN Soil Image Prediction with PostgreSQL Storage.
Covers:
  - Successful prediction on uploaded images
  - Prediction accuracy contract (predicted class, confidence, all 6 class probabilities, model version)
  - PostgreSQL database persistence in analysis_results table
  - Prediction history API retrieval and field-level filtering
  - Multi-tenant data isolation & IDOR prevention (Farmer A vs Farmer B)
  - Unauthorized access rejection (HTTP 401)
  - Corrupt or invalid image handling (HTTP 422)
  - Missing model checkpoint handling (HTTP 503)
"""
import io
import os
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import AnalysisResult, Farmer, Field, SoilImage
from app.services.soil_classifier import (
    DEFAULT_CLASSES,
    SoilClassifier,
    SoilModelError,
    get_soil_classifier,
)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def register_farmer(client, name: str, email: str, password: str = "password123"):
    res = client.post(
        "/api/auth/register",
        json={"full_name": name, "email": email, "password": password},
    )
    assert res.status_code == 201, res.text
    data = res.json()
    return {
        "farmer": data["farmer"],
        "token": data["access_token"],
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
    }


def make_test_image_bytes(seed: int = 42) -> bytes:
    """Generate a realistic test image with variance and color."""
    rng = np.random.RandomState(seed)
    # Generate 224x224 RGB image with soil-like brownish/dark tones
    arr = rng.randint(40, 180, (224, 224, 3), dtype="uint8")
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_successful_cnn_prediction_contract(client):
    """Test full prediction flow and verify all required response fields."""
    user = register_farmer(client, "Pred Farmer One", "pred1@example.com")

    # 1. Create a field
    res_field = client.post(
        "/api/fields",
        json={"name": "North Delta Plot", "location": "Thanjavur"},
        headers=user["headers"],
    )
    assert res_field.status_code == 201
    field_id = res_field.json()["id"]

    # 2. Upload an image
    img_bytes = make_test_image_bytes(seed=101)
    res_upload = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("soil_sample.jpg", img_bytes, "image/jpeg")},
        headers=user["headers"],
    )
    assert res_upload.status_code == 201
    image_id = res_upload.json()["image"]["id"]

    # 3. Call CNN prediction endpoint
    res_pred = client.post(
        f"/api/images/{image_id}/predict",
        headers=user["headers"],
    )
    assert res_pred.status_code == 200, res_pred.text
    data = res_pred.json()

    assert data["image_id"] == image_id
    assert data["field_id"] == field_id
    assert "prediction" in data

    pred = data["prediction"]
    assert pred["predicted_soil_type"] in DEFAULT_CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["confidence_percentage"] <= 100.0
    assert pred["model_version"] == "efficientnet_b0_v1.0"
    assert pred["best_epoch"] == 7

    # Check all 6 classes are present in probabilities dictionary
    probs = pred["probabilities"]
    assert len(probs) == 6
    for cls_name in DEFAULT_CLASSES:
        assert cls_name in probs
        assert isinstance(probs[cls_name], (int, float))

    # Sum of probabilities should be approximately 1.0
    prob_sum = sum(probs.values())
    assert 0.98 <= prob_sum <= 1.02


def test_prediction_database_persistence(client):
    """Verify that predictions are safely stored in PostgreSQL/database."""
    user = register_farmer(client, "Persist Farmer", "persist@example.com")

    # Create field & upload image
    res_field = client.post(
        "/api/fields",
        json={"name": "Storage Verification Farm"},
        headers=user["headers"],
    )
    field_id = res_field.json()["id"]
    res_upload = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("soil_persist.jpg", make_test_image_bytes(seed=202), "image/jpeg")},
        headers=user["headers"],
    )
    image_id = res_upload.json()["image"]["id"]

    # Run prediction
    res_pred = client.post(
        f"/api/images/{image_id}/predict",
        headers=user["headers"],
    )
    assert res_pred.status_code == 200
    pred_res = res_pred.json()
    record_id = pred_res["id"]

    # Verify directly via SessionLocal
    with SessionLocal() as db:
        record = db.get(AnalysisResult, record_id)
        assert record is not None
        assert record.field_id == field_id
        assert record.image_id == image_id
        assert isinstance(record.result, dict)
        assert "predicted_soil_type" in record.result
        assert "probabilities" in record.result
        # Ensure no tensor or raw bytes stored in DB
        assert "tensor" not in record.result

    # Also verify via GET endpoint
    res_get = client.get(
        f"/api/images/{image_id}/prediction",
        headers=user["headers"],
    )
    assert res_get.status_code == 200
    assert res_get.json()["id"] == record_id


def test_prediction_history_retrieval_and_filtering(client):
    """Test GET /api/predictions and GET /api/fields/{id}/predictions."""
    user = register_farmer(client, "History Farmer", "history_farmer@example.com")

    # Create two fields
    f1 = client.post("/api/fields", json={"name": "Plot Alpha"}, headers=user["headers"]).json()["id"]
    f2 = client.post("/api/fields", json={"name": "Plot Beta"}, headers=user["headers"]).json()["id"]

    # Upload and predict for f1
    up1 = client.post(
        f"/api/fields/{f1}/images",
        files={"file": ("f1.jpg", make_test_image_bytes(seed=301), "image/jpeg")},
        headers=user["headers"],
    ).json()["image"]["id"]
    client.post(f"/api/images/{up1}/predict", headers=user["headers"])

    # Upload and predict for f2
    up2 = client.post(
        f"/api/fields/{f2}/images",
        files={"file": ("f2.jpg", make_test_image_bytes(seed=302), "image/jpeg")},
        headers=user["headers"],
    ).json()["image"]["id"]
    client.post(f"/api/images/{up2}/predict", headers=user["headers"])

    # Fetch all predictions for farmer
    res_all = client.get("/api/predictions", headers=user["headers"])
    assert res_all.status_code == 200
    all_preds = res_all.json()
    assert len(all_preds) >= 2
    f1_preds = [p for p in all_preds if p["field_id"] == f1]
    f2_preds = [p for p in all_preds if p["field_id"] == f2]
    assert len(f1_preds) >= 1
    assert len(f2_preds) >= 1

    # Filter by field_id query parameter
    res_filtered = client.get(f"/api/predictions?field_id={f1}", headers=user["headers"])
    assert res_filtered.status_code == 200
    assert all(p["field_id"] == f1 for p in res_filtered.json())

    # Filter via field sub-resource endpoint
    res_field_sub = client.get(f"/api/fields/{f2}/predictions", headers=user["headers"])
    assert res_field_sub.status_code == 200
    assert all(p["field_id"] == f2 for p in res_field_sub.json())


def test_cross_farmer_isolation_idor_protection(client):
    """Ensure Farmer A cannot run predictions or view predictions of Farmer B."""
    farmer_a = register_farmer(client, "Farmer Alice", "alice@example.com")
    farmer_b = register_farmer(client, "Farmer Bob", "bob@example.com")

    # Alice creates a field and uploads an image
    f_alice = client.post("/api/fields", json={"name": "Alice Farm"}, headers=farmer_a["headers"]).json()["id"]
    up_alice = client.post(
        f"/api/fields/{f_alice}/images",
        files={"file": ("alice.jpg", make_test_image_bytes(seed=401), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    # Bob attempts to predict Alice's image -> 404 Not Found
    res_bob_predict = client.post(
        f"/api/images/{up_alice}/predict",
        headers=farmer_b["headers"],
    )
    assert res_bob_predict.status_code == 404

    # Alice runs prediction
    res_alice_pred = client.post(
        f"/api/images/{up_alice}/predict",
        headers=farmer_a["headers"],
    )
    assert res_alice_pred.status_code == 200

    # Bob attempts to read Alice's image prediction -> 404 Not Found
    res_bob_read = client.get(
        f"/api/images/{up_alice}/prediction",
        headers=farmer_b["headers"],
    )
    assert res_bob_read.status_code == 404

    # Bob lists his own predictions -> must not include Alice's prediction
    res_bob_list = client.get("/api/predictions", headers=farmer_b["headers"])
    assert res_bob_list.status_code == 200
    bob_img_ids = [p["image_id"] for p in res_bob_list.json()]
    assert up_alice not in bob_img_ids

    # Bob attempts to list predictions for Alice's field -> 404 Not Found
    res_bob_field = client.get(f"/api/fields/{f_alice}/predictions", headers=farmer_b["headers"])
    assert res_bob_field.status_code == 404


def test_unauthorized_access_rejected(client):
    """Ensure endpoints require valid authentication."""
    # Predict without token
    res1 = client.post("/api/images/1/predict")
    assert res1.status_code == 401

    # Get prediction without token
    res2 = client.get("/api/images/1/prediction")
    assert res2.status_code == 401

    # List predictions without token
    res3 = client.get("/api/predictions")
    assert res3.status_code == 401


def test_corrupted_image_handling(client):
    """Ensure invalid or unreadable image bytes result in HTTP 422."""
    user = register_farmer(client, "Corrupt Farmer", "corrupt@example.com")
    f_id = client.post("/api/fields", json={"name": "Corrupt Plot"}, headers=user["headers"]).json()["id"]

    up_id = client.post(
        f"/api/fields/{f_id}/images",
        files={"file": ("valid_init.jpg", make_test_image_bytes(seed=501), "image/jpeg")},
        headers=user["headers"],
    ).json()["image"]["id"]

    # Mock storage.read_image to return corrupted bytes
    with patch("app.routers.images.storage.read_image", return_value=b"NOT_A_VALID_IMAGE_BYTES"):
        res = client.post(f"/api/images/{up_id}/predict", headers=user["headers"])
        assert res.status_code == 422
        assert "Invalid or unreadable image data" in res.json()["detail"]


def test_missing_model_file_handling(client):
    """Ensure missing model checkpoint results in HTTP 503."""
    user = register_farmer(client, "Model Error Farmer", "model_err@example.com")
    f_id = client.post("/api/fields", json={"name": "Model Err Plot"}, headers=user["headers"]).json()["id"]

    up_id = client.post(
        f"/api/fields/{f_id}/images",
        files={"file": ("model_err.jpg", make_test_image_bytes(seed=601), "image/jpeg")},
        headers=user["headers"],
    ).json()["image"]["id"]

    classifier = get_soil_classifier()
    # Mock classifier.predict to simulate missing model file
    with patch.object(classifier, "predict", side_effect=SoilModelError("Model file not found")):
        res = client.post(f"/api/images/{up_id}/predict", headers=user["headers"])
        assert res.status_code == 503
        assert "Soil classification model unavailable" in res.json()["detail"]
