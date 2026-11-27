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
    if res.status_code == 400 and "Email already registered" in res.text:
        login_res = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
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


def test_get_prediction_detail_endpoint(client):
    """Test retrieving individual prediction detail, validating response schema and IDOR isolation."""
    farmer_a = register_farmer(client, "Detail Farmer A", "detail_a@example.com")
    farmer_b = register_farmer(client, "Detail Farmer B", "detail_b@example.com")

    # Farmer A creates field, uploads image, and runs prediction
    field_a = client.post("/api/fields", json={"name": "Delta Detail Plot"}, headers=farmer_a["headers"]).json()["id"]
    img_a = client.post(
        f"/api/fields/{field_a}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=701), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    res_pred = client.post(f"/api/images/{img_a}/predict", headers=farmer_a["headers"])
    assert res_pred.status_code == 200
    pred_record_id = res_pred.json()["id"]

    # 1. Farmer A successfully fetches prediction details
    res_detail = client.get(f"/api/predictions/{pred_record_id}", headers=farmer_a["headers"])
    assert res_detail.status_code == 200
    data = res_detail.json()
    assert data["id"] == pred_record_id
    assert data["field_id"] == field_a
    assert data["field_name"] == "Delta Detail Plot"
    assert data["image_id"] == img_a
    assert "predicted_soil_type" in data
    assert "confidence" in data
    assert "confidence_percentage" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 6
    for cls_name in DEFAULT_CLASSES:
        assert cls_name in data["probabilities"]
    assert data["model_version"] == "efficientnet_b0_v1.0"
    assert data["best_epoch"] == 7

    # 2. Farmer B tries to fetch Farmer A's prediction -> 404 (IDOR protected)
    res_b = client.get(f"/api/predictions/{pred_record_id}", headers=farmer_b["headers"])
    assert res_b.status_code == 404

    # 3. Non-existent prediction ID -> 404
    res_404 = client.get("/api/predictions/999999", headers=farmer_a["headers"])
    assert res_404.status_code == 404

    # 4. Unauthenticated request -> 401
    res_unauth = client.get(f"/api/predictions/{pred_record_id}")
    assert res_unauth.status_code == 401


def test_prediction_detail_handles_null_optional_fields(client):
    """Explicit null values in stored AnalysisResult.result must not trigger ValidationError."""
    farmer = register_farmer(client, "Null Handling Farmer", "null.handler@example.com")
    field_id = client.post("/api/fields", json={"name": "Null Safe Plot"}, headers=farmer["headers"]).json()["id"]
    img_id = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=808), "image/jpeg")},
        headers=farmer["headers"],
    ).json()["image"]["id"]

    db = SessionLocal()
    try:
        null_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": "clay_soil",
                "confidence": None,
                "confidence_percentage": None,
                "probabilities": None,
                "model_version": None,
                "best_epoch": None,
            },
        )
        db.add(null_rec)
        db.commit()
        db.refresh(null_rec)
        rec_id = null_rec.id

        incomplete_rec = AnalysisResult(field_id=field_id, image_id=img_id, result={})
        db.add(incomplete_rec)
        db.commit()
    finally:
        db.close()

    detail_res = client.get(f"/api/predictions/{rec_id}", headers=farmer["headers"])
    assert detail_res.status_code == 200
    body = detail_res.json()
    assert body["predicted_soil_type"] == "clay_soil"
    assert body["confidence"] == 0.0
    assert body["confidence_percentage"] == 0.0
    assert body["probabilities"] == {}
    assert body["model_version"] is None

    list_res = client.get("/api/predictions", headers=farmer["headers"])
    assert list_res.status_code == 200
    assert any(item["id"] == rec_id for item in list_res.json())

    # Latest AnalysisResult on img_id has empty dict {} -> should return 404 rather than 500
    latest_res = client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"])
    assert latest_res.status_code == 404


def test_prediction_endpoints_handle_null_predicted_soil_type(client):
    """Explicit null predicted_soil_type in JSONB must return 404 (not ValidationError 500)."""
    farmer = register_farmer(client, "Null Soil Type Farmer", "null.soiltype@example.com")
    field_id = client.post("/api/fields", json={"name": "Null Soil Type Plot"}, headers=farmer["headers"]).json()["id"]
    img_id = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=909), "image/jpeg")},
        headers=farmer["headers"],
    ).json()["image"]["id"]

    db = SessionLocal()
    try:
        # 1. Valid predicted_soil_type with null optional fields -> 200 on get_image_prediction
        partial_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": "red_soil",
                "confidence": None,
                "confidence_percentage": None,
                "probabilities": None,
                "model_version": None,
            },
        )
        db.add(partial_rec)
        db.commit()
        db.refresh(partial_rec)
        partial_id = partial_rec.id
    finally:
        db.close()

    partial_res = client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"])
    assert partial_res.status_code == 200
    assert partial_res.json()["prediction"]["predicted_soil_type"] == "red_soil"
    assert partial_res.json()["prediction"]["probabilities"] == {}

    # 2. Newer record with predicted_soil_type = None -> 404 on get_image_prediction & get_prediction_detail, skipped in list
    db = SessionLocal()
    try:
        null_type_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": None,
                "confidence": 0.5,
                "confidence_percentage": 50.0,
                "probabilities": {},
                "model_version": "efficientnet_b0_v1.0",
            },
        )
        db.add(null_type_rec)
        db.commit()
        db.refresh(null_type_rec)
        null_type_id = null_type_rec.id
    finally:
        db.close()

    assert client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"]).status_code == 404
    assert client.get(f"/api/predictions/{null_type_id}", headers=farmer["headers"]).status_code == 404
    listed_ids = [item["id"] for item in client.get("/api/predictions", headers=farmer["headers"]).json()]
    assert partial_id in listed_ids
    assert null_type_id not in listed_ids


def test_prediction_endpoints_handle_malformed_jsonb_values(client):
    """Malformed JSONB values (bool, NaN, inf, invalid strings, non-dict) must not raise 500."""
    farmer = register_farmer(client, "Malformed JSONB Farmer", "malformed.jsonb@example.com")
    field_id = client.post("/api/fields", json={"name": "Malformed JSONB Plot"}, headers=farmer["headers"]).json()["id"]
    img_id = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=919), "image/jpeg")},
        headers=farmer["headers"],
    ).json()["image"]["id"]

    db = SessionLocal()
    try:
        malformed_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": "sandy_soil",
                "confidence": True,
                "confidence_percentage": "NaN",
                "probabilities": {
                    "sandy_soil": 0.75,
                    "clay_soil": "invalid",
                    "red_soil": False,
                    "black_soil": "inf",
                },
                "model_version": 12345,
                "best_epoch": True,
            },
        )
        db.add(malformed_rec)
        db.commit()
        db.refresh(malformed_rec)
        rec_id = malformed_rec.id
    finally:
        db.close()

    res_img = client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"])
    assert res_img.status_code == 200
    pred = res_img.json()["prediction"]
    assert pred["predicted_soil_type"] == "sandy_soil"
    assert pred["confidence"] == 0.0
    assert pred["confidence_percentage"] == 0.0
    assert pred["model_version"] is None
    assert pred["best_epoch"] is None
    assert pred["probabilities"] == {
        "sandy_soil": 0.75,
        "clay_soil": 0.0,
        "red_soil": 0.0,
        "black_soil": 0.0,
    }

    res_detail = client.get(f"/api/predictions/{rec_id}", headers=farmer["headers"])
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["confidence"] == 0.0
    assert detail["confidence_percentage"] == 0.0
    assert detail["model_version"] is None
    assert detail["best_epoch"] is None


def test_gradcam_endpoint_returns_overlay_and_enforces_ownership(client):
    """Grad-CAM endpoint returns valid JPEG heatmap overlay and enforces farmer isolation."""
    farmer_a = register_farmer(client, "GradCAM Farmer A", "gradcam.a@example.com")
    farmer_b = register_farmer(client, "GradCAM Farmer B", "gradcam.b@example.com")

    field_a = client.post("/api/fields", json={"name": "GradCAM Plot A"}, headers=farmer_a["headers"]).json()["id"]
    img_a = client.post(
        f"/api/fields/{field_a}/images",
        files={"file": ("soil_sample.jpg", make_test_image_bytes(seed=321), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    # 1. Authenticated owner retrieves Grad-CAM overlay
    res = client.get(f"/api/images/{img_a}/gradcam", headers=farmer_a["headers"])
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/jpeg")
    assert res.headers["x-gradcam-predicted-class"] in DEFAULT_CLASSES
    overlay_img = Image.open(io.BytesIO(res.content))
    assert overlay_img.size == (224, 224)

    # 2. Explicit target_class query parameter works
    res_target = client.get(
        f"/api/images/{img_a}/gradcam?target_class=red_soil",
        headers=farmer_a["headers"],
    )
    assert res_target.status_code == 200
    assert res_target.headers["x-gradcam-target-class"] == "red_soil"

    # 3. Invalid target_class returns 422
    res_bad_cls = client.get(
        f"/api/images/{img_a}/gradcam?target_class=invalid_soil",
        headers=farmer_a["headers"],
    )
    assert res_bad_cls.status_code == 422

    # 4. IDOR protection: Farmer B receives 404, unauthenticated receives 401
    assert client.get(f"/api/images/{img_a}/gradcam", headers=farmer_b["headers"]).status_code == 404
    assert client.get(f"/api/images/{img_a}/gradcam").status_code == 401

    # 5. Missing checkpoint returns 503
    with patch.object(SoilClassifier, "load_model", side_effect=SoilModelError("Checkpoint missing")):
        res_503 = client.get(f"/api/images/{img_a}/gradcam", headers=farmer_a["headers"])
        assert res_503.status_code == 503

def test_prediction_endpoints_handle_null_predicted_soil_type(client):
    """Explicit null predicted_soil_type in JSONB must return 404 (not ValidationError 500)."""
    farmer = register_farmer(client, "Null Soil Type Farmer", "null.soiltype@example.com")
    field_id = client.post("/api/fields", json={"name": "Null Soil Type Plot"}, headers=farmer["headers"]).json()["id"]
    img_id = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=909), "image/jpeg")},
        headers=farmer["headers"],
    ).json()["image"]["id"]

    db = SessionLocal()
    try:
        # 1. Valid predicted_soil_type with null optional fields -> 200 on get_image_prediction
        partial_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": "red_soil",
                "confidence": None,
                "confidence_percentage": None,
                "probabilities": None,
                "model_version": None,
            },
        )
        db.add(partial_rec)
        db.commit()
        db.refresh(partial_rec)
        partial_id = partial_rec.id
    finally:
        db.close()

    partial_res = client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"])
    assert partial_res.status_code == 200
    assert partial_res.json()["prediction"]["predicted_soil_type"] == "red_soil"
    assert partial_res.json()["prediction"]["probabilities"] == {}

    # 2. Newer record with predicted_soil_type = None -> 404 on get_image_prediction & get_prediction_detail, skipped in list
    db = SessionLocal()
    try:
        null_type_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": None,
                "confidence": 0.5,
                "confidence_percentage": 50.0,
                "probabilities": {},
                "model_version": "efficientnet_b0_v1.0",
            },
        )
        db.add(null_type_rec)
        db.commit()
        db.refresh(null_type_rec)
        null_type_id = null_type_rec.id
    finally:
        db.close()

    assert client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"]).status_code == 404
    assert client.get(f"/api/predictions/{null_type_id}", headers=farmer["headers"]).status_code == 404
    listed_ids = [item["id"] for item in client.get("/api/predictions", headers=farmer["headers"]).json()]
    assert partial_id in listed_ids
    assert null_type_id not in listed_ids


def test_prediction_endpoints_handle_malformed_jsonb_values(client):
    """Malformed JSONB values (bool, NaN, inf, invalid strings, non-dict) must not raise 500."""
    farmer = register_farmer(client, "Malformed JSONB Farmer", "malformed.jsonb@example.com")
    field_id = client.post("/api/fields", json={"name": "Malformed JSONB Plot"}, headers=farmer["headers"]).json()["id"]
    img_id = client.post(
        f"/api/fields/{field_id}/images",
        files={"file": ("sample.jpg", make_test_image_bytes(seed=919), "image/jpeg")},
        headers=farmer["headers"],
    ).json()["image"]["id"]

    db = SessionLocal()
    try:
        malformed_rec = AnalysisResult(
            field_id=field_id,
            image_id=img_id,
            result={
                "predicted_soil_type": "sandy_soil",
                "confidence": True,
                "confidence_percentage": "NaN",
                "probabilities": {
                    "sandy_soil": 0.75,
                    "clay_soil": "invalid",
                    "red_soil": False,
                    "black_soil": "inf",
                },
                "model_version": 12345,
                "best_epoch": True,
            },
        )
        db.add(malformed_rec)
        db.commit()
        db.refresh(malformed_rec)
        rec_id = malformed_rec.id
    finally:
        db.close()

    res_img = client.get(f"/api/images/{img_id}/prediction", headers=farmer["headers"])
    assert res_img.status_code == 200
    pred = res_img.json()["prediction"]
    assert pred["predicted_soil_type"] == "sandy_soil"
    assert pred["confidence"] == 0.0
    assert pred["confidence_percentage"] == 0.0
    assert pred["model_version"] is None
    assert pred["best_epoch"] is None
    assert pred["probabilities"] == {
        "sandy_soil": 0.75,
        "clay_soil": 0.0,
        "red_soil": 0.0,
        "black_soil": 0.0,
    }

    res_detail = client.get(f"/api/predictions/{rec_id}", headers=farmer["headers"])
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["confidence"] == 0.0
    assert detail["confidence_percentage"] == 0.0
    assert detail["model_version"] is None
    assert detail["best_epoch"] is None


def test_gradcam_endpoint_returns_overlay_and_enforces_ownership(client):
    """Grad-CAM endpoint returns valid JPEG heatmap overlay and enforces farmer isolation."""
    farmer_a = register_farmer(client, "GradCAM Farmer A", "gradcam.a@example.com")
    farmer_b = register_farmer(client, "GradCAM Farmer B", "gradcam.b@example.com")

    field_a = client.post("/api/fields", json={"name": "GradCAM Plot A"}, headers=farmer_a["headers"]).json()["id"]
    img_a = client.post(
        f"/api/fields/{field_a}/images",
        files={"file": ("soil_sample.jpg", make_test_image_bytes(seed=321), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    # 1. Authenticated owner retrieves Grad-CAM overlay
    res = client.get(f"/api/images/{img_a}/gradcam", headers=farmer_a["headers"])
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/jpeg")
    assert res.headers["x-gradcam-predicted-class"] in DEFAULT_CLASSES
    overlay_img = Image.open(io.BytesIO(res.content))
    assert overlay_img.size == (224, 224)

    # 2. Explicit target_class query parameter works
    res_target = client.get(
        f"/api/images/{img_a}/gradcam?target_class=red_soil",
        headers=farmer_a["headers"],
    )
    assert res_target.status_code == 200
    assert res_target.headers["x-gradcam-target-class"] == "red_soil"

    # 3. Invalid target_class returns 422
    res_bad_cls = client.get(
        f"/api/images/{img_a}/gradcam?target_class=invalid_soil",
        headers=farmer_a["headers"],
    )
    assert res_bad_cls.status_code == 422

    # 4. IDOR protection: Farmer B receives 404, unauthenticated receives 401
    assert client.get(f"/api/images/{img_a}/gradcam", headers=farmer_b["headers"]).status_code == 404
    assert client.get(f"/api/images/{img_a}/gradcam").status_code == 401

    # 5. Missing checkpoint returns 503
    with patch.object(SoilClassifier, "load_model", side_effect=SoilModelError("Checkpoint missing")):
        res_503 = client.get(f"/api/images/{img_a}/gradcam", headers=farmer_a["headers"])
        assert res_503.status_code == 503
