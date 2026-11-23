"""Unit and integration tests for the CNN Model Analytics API endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_cnn_metrics_endpoint(client):
    res = client.get("/api/cnn/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "accuracy" in data
    assert "macro_avg" in data
    assert "classification_report" in data
    assert "confusion_matrix" in data
    assert data["total_test_samples"] == 206
    assert data["best_epoch"] == 7

    # Verify all 6 soil classes are present
    expected_classes = [
        "alluvial_soil",
        "black_soil",
        "clay_soil",
        "laterite_soil",
        "red_soil",
        "sandy_soil",
    ]
    report = data["classification_report"]
    for cls_name in expected_classes:
        assert cls_name in report
        assert "precision" in report[cls_name]
        assert "recall" in report[cls_name]
        assert "f1-score" in report[cls_name]
        assert "support" in report[cls_name]


def test_cnn_training_curves_endpoint(client):
    res = client.get("/api/cnn/training-curves")
    assert res.status_code == 200
    assert "image/png" in res.headers.get("content-type", "")
    assert len(res.content) > 1000


def test_cnn_confusion_matrix_endpoint(client):
    res = client.get("/api/cnn/confusion-matrix")
    assert res.status_code == 200
    assert "image/png" in res.headers.get("content-type", "")
    assert len(res.content) > 1000

