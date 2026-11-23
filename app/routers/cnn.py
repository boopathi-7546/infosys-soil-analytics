import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(tags=["CNN Model Analytics"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "soil_classifier"


@router.get("/cnn/metrics")
def get_cnn_metrics():
    """Return the evaluation metrics of the trained soil classifier."""
    metrics_file = MODEL_DIR / "metrics.json"

    if not metrics_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="CNN evaluation metrics are not available."
        )

    try:
        with metrics_file.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        return JSONResponse(content=metrics)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read CNN evaluation metrics."
        ) from exc


@router.get("/cnn/training-curves")
def get_cnn_training_curves():
    """Return the CNN training and validation curves."""
    image_file = MODEL_DIR / "training_curves.png"

    if not image_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training curves are not available."
        )

    return FileResponse(image_file, media_type="image/png")


@router.get("/cnn/confusion-matrix")
def get_cnn_confusion_matrix():
    """Return the CNN test-set confusion matrix visualization."""
    image_file = MODEL_DIR / "confusion_matrix.png"

    if not image_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix image is not available."
        )

    return FileResponse(image_file, media_type="image/png")
