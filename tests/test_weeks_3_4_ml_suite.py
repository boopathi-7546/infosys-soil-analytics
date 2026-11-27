"""Comprehensive tests for Weeks 3-4 ML models, SHAP, Hybrid Fusion, Degradation Forecast, and Model Registry."""
import csv
import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.deficiency_detector import DeficiencyModelError, NutrientDeficiencyDetector
from app.services.degradation_forecaster import DegradationForecasterService, DegradationModelError
from app.services.nutrient_ml import NutrientMLModelError, NutrientMLService
from app.services.soil_texture_classifier import SoilTextureClassifier, TextureModelError
from ml.train_fusion_model import HybridSoilFusionNet, InsufficientPairedDataError, validate_paired_dataset


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


def _make_jpeg_bytes(seed: int = 55) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, size=(224, 224, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_model_registry_and_manifest_exist(client):
    """Model registry lists all 6 models and DATASET_MANIFEST.csv records acquired datasets."""
    res = client.get("/api/ml/registry")
    assert res.status_code == 200
    models = res.json()["models"]
    for key in (
        "soil_type_classifier",
        "soil_texture_classifier",
        "nutrient_deficiency_detector",
        "nutrient_classifier",
        "hybrid_fusion",
        "degradation_forecaster",
    ):
        assert key in models

    manifest = Path("data/DATASET_MANIFEST.csv")
    assert manifest.is_file()
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3
    for row in rows:
        assert row["license"] == "CC BY 4.0"
        assert len(row["sha256"]) == 64


def test_existing_cnn_checkpoint_sha256_preserved():
    """Original 6-class EfficientNet-B0 checkpoint must remain untouched."""
    ckpt = Path("models/soil_classifier/best_model.pth")
    assert ckpt.is_file()
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert digest == "1b11f7e08c386e85f863c307c8256d8c4998a8f63b39b1a62ae4e03d6e2ac1e5"


def test_visible_deficiency_endpoint_and_isolation(client):
    """Plant nutrient deficiency detector predicts cleanly, enforces ownership, and handles missing model."""
    farmer_a = _register_farmer(client, "Deficiency A", "def.a@example.com")
    farmer_b = _register_farmer(client, "Deficiency B", "def.b@example.com")

    field_a = client.post("/api/fields", json={"name": "Deficiency Plot"}, headers=farmer_a["headers"]).json()["id"]
    img_a = client.post(
        f"/api/fields/{field_a}/images",
        files={"file": ("leaf.jpg", _make_jpeg_bytes(101), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    res = client.post(f"/api/images/{img_a}/predict-deficiency", headers=farmer_a["headers"])
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["predicted_class"] in {"-p", "-p50", "_c"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_maturity"] == "demonstration"
    assert "demonstration model" in body["disclaimer"].lower()
    assert "not a laboratory soil nutrient measurement" in body["disclaimer"].lower()

    assert client.post(f"/api/images/{img_a}/predict-deficiency", headers=farmer_b["headers"]).status_code == 404
    assert client.post(f"/api/images/{img_a}/predict-deficiency").status_code == 401

    with patch.object(NutrientDeficiencyDetector, "load_model", side_effect=DeficiencyModelError("Missing")):
        assert client.post(f"/api/images/{img_a}/predict-deficiency", headers=farmer_a["headers"]).status_code == 503


def test_soil_texture_endpoint_and_isolation(client):
    """Soil texture classifier endpoint predicts when checkpoint exists, enforces ownership, and returns 503 if missing."""
    farmer_a = _register_farmer(client, "Texture A", "tex.a@example.com")
    farmer_b = _register_farmer(client, "Texture B", "tex.b@example.com")

    field_a = client.post("/api/fields", json={"name": "Texture Plot"}, headers=farmer_a["headers"]).json()["id"]
    img_a = client.post(
        f"/api/fields/{field_a}/images",
        files={"file": ("tex.jpg", _make_jpeg_bytes(202), "image/jpeg")},
        headers=farmer_a["headers"],
    ).json()["image"]["id"]

    assert client.post(f"/api/images/{img_a}/predict-texture", headers=farmer_b["headers"]).status_code == 404
    assert client.post(f"/api/images/{img_a}/predict-texture").status_code == 401

    if Path("models/soil_texture_classifier/best_model.pth").is_file():
        res = client.post(f"/api/images/{img_a}/predict-texture", headers=farmer_a["headers"])
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["predicted_texture"] in data["probabilities"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_maturity"] == "demonstration"
        assert "demonstration model" in data["disclaimer"].lower()
        assert "soil type" in data["disclaimer"].lower()

    with patch.object(SoilTextureClassifier, "load_model", side_effect=TextureModelError("Missing")):
        assert client.post(f"/api/images/{img_a}/predict-texture", headers=farmer_a["headers"]).status_code == 503


def test_structured_nutrient_ml_and_shap_endpoint(client):
    """XGBoost nutrient fertility endpoint returns TreeExplainer SHAP contributions and enforces auth."""
    farmer = _register_farmer(client, "Nutrient ML Farmer", "nutml@example.com")
    payload = {"nitrogen": 45.0, "phosphorus": 28.0, "potassium": 22.0}

    assert client.post("/api/ml/predict-nutrient-fertility", json=payload).status_code == 401
    assert client.post("/api/ml/predict-nutrient-fertility", json={"nitrogen": -5, "phosphorus": 10, "potassium": 10}, headers=farmer["headers"]).status_code == 422

    res = client.post("/api/ml/predict-nutrient-fertility", json=payload, headers=farmer["headers"])
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["explainer"] == "shap.TreeExplainer"
    assert data["model_maturity"] == "experimental"
    assert "not suitable for real fertilizer dosage decisions" in data["disclaimer"].lower()
    assert len(data["feature_contributions"]) == 3
    assert set(data["global_feature_importance"].keys()) == {"Nitrogen", "Phosphorus", "Potassium"}
    assert 0.0 <= data["confidence"] <= 1.0

    with patch.object(NutrientMLService, "load_model", side_effect=NutrientMLModelError("Missing")):
        assert client.post("/api/ml/predict-nutrient-fertility", json=payload, headers=farmer["headers"]).status_code == 503


def test_hybrid_fusion_architecture_and_status_endpoint(client):
    """Hybrid fusion net produces expected logits and refuses training without real paired records."""
    net = HybridSoilFusionNet(num_tabular_features=6, num_classes=4)
    out = net(torch.randn(2, 3, 224, 224), torch.randn(2, 6))
    assert out.shape == (2, 4)

    with pytest.raises(InsufficientPairedDataError):
        validate_paired_dataset([])

    farmer = _register_farmer(client, "Fusion Farmer", "fusion@example.com")
    assert client.get("/api/ml/hybrid-fusion/status").status_code == 401
    res = client.get("/api/ml/hybrid-fusion/status", headers=farmer["headers"])
    assert res.status_code == 200
    assert res.json()["status"] == "architecture_ready_untrained"
    assert "pending genuine paired" in res.json()["limitations"].lower()


def test_degradation_forecast_endpoint(client):
    """Time-aware ML degradation forecaster returns SOC decline risk and enforces auth/validation."""
    farmer = _register_farmer(client, "Degrad Farmer", "degrad@example.com")
    payload = {"baseline_oc": 1.4, "horizon_years": 3.0, "ph": 6.4}

    assert client.post("/api/ml/predict-degradation-risk", json=payload).status_code == 401
    assert client.post("/api/ml/predict-degradation-risk", json={"baseline_oc": -1.0, "horizon_years": 3.0}, headers=farmer["headers"]).status_code == 422

    res = client.post("/api/ml/predict-degradation-risk", json=payload, headers=farmer["headers"])
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["label"] == "ML Soil Degradation Forecast"
    assert data["model_maturity"] == "experimental"
    assert "not a reliable future prediction" in data["disclaimer"].lower()
    assert 0.0 <= data["soc_decline_probability"] <= 1.0
    assert data["risk_level"] in {"low", "moderate", "high"}

    with patch.object(DegradationForecasterService, "load_model", side_effect=DegradationModelError("Missing")):
        assert client.post("/api/ml/predict-degradation-risk", json=payload, headers=farmer["headers"]).status_code == 503


def test_dashboard_displays_honest_model_maturity_badges(client):
    """Dashboard HTML clearly labels demo vs experimental vs historical trend features."""
    html = client.get("/").text
    assert "Demonstration Model" in html
    assert "Experimental • Not for Dosage" in html
    assert "Experimental ML Forecast" in html
    assert "Historical Soil Trend Analysis – Not an ML Forecast" in html

