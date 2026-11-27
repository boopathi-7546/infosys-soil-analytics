"""Tests covering Weeks 1-2 OpenCV preprocessing, augmentation, camera UI, and dataset provenance."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import preprocessing


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _make_red_soil_bgr(size: int = 256) -> np.ndarray:
    """Create a synthetic BGR image with strong red channel dominance (like red/laterite soil)."""
    rng = np.random.default_rng(7)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(35, 65, size=(size, size), dtype=np.uint8)    # B
    img[:, :, 1] = rng.integers(60, 95, size=(size, size), dtype=np.uint8)    # G
    img[:, :, 2] = rng.integers(140, 190, size=(size, size), dtype=np.uint8)  # R
    return img


def test_opencv_color_correction_and_preprocessing_preserve_soil_signal():
    """Partial gray-world, CLAHE, noise reduction, background removal, and resize work cleanly."""
    img = _make_red_soil_bgr(256)

    balanced = preprocessing.gray_world(img)
    assert balanced.shape == img.shape
    assert balanced.dtype == np.uint8
    # Partial gray-world must preserve red-soil dominance (R > B)
    assert float(balanced[:, :, 2].mean()) > float(balanced[:, :, 0].mean())

    clahe = preprocessing.normalize_illumination(balanced)
    assert clahe.shape == img.shape
    assert clahe.dtype == np.uint8

    denoised = preprocessing.reduce_noise(clahe)
    assert denoised.shape == img.shape

    fg = preprocessing.remove_background(denoised)
    assert fg.shape == img.shape
    assert fg.dtype == np.uint8

    full = preprocessing.preprocess(img, size=224, bg_removal=True)
    assert full.shape == (224, 224, 3)
    assert full.dtype == np.uint8


def test_feature_preserving_augmentation():
    """Training augmentation preserves image dimensions and uint8 BGR format."""
    img = _make_red_soil_bgr(224)
    rng = np.random.default_rng(123)
    aug = preprocessing.augment(img, rng)
    assert aug.shape == (224, 224, 3)
    assert aug.dtype == np.uint8
    assert not np.array_equal(aug, img)


def test_frontend_supports_both_getusermedia_and_capture_environment(client):
    """Android camera capture supports both getUserMedia and input[capture=environment]."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "getUserMedia" in html
    assert "facingMode" in html
    assert 'capture="environment"' in html


def test_dataset_provenance_includes_usda_icar_and_open_sources(client):
    """Dataset provenance endpoint exposes USDA, ICAR, and open-source licenses."""
    res = client.get("/api/dataset/status")
    assert res.status_code == 200
    data = res.json()
    origins = " ".join(s.get("source_name", "") for s in data.get("sources", []))
    assert "USDA-NRCS" in origins
    assert "ICAR-NBSS&LUP" in origins
    assert "Kaggle" in origins

