import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas import DatasetStatusOut

router = APIRouter(tags=["dataset"])

DATA_RAW_DIR = Path("data/raw")
DATA_PROC_DIR = Path("data/processed")
SOURCES_CSV = Path("data/sources.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.get("/dataset/status", response_model=DatasetStatusOut)
def get_dataset_status():
    """Returns the current status of collected real-world soil datasets across all soil categories."""
    soil_types: list[dict[str, Any]] = []
    total_raw = 0

    if DATA_RAW_DIR.exists():
        for folder in sorted(DATA_RAW_DIR.iterdir()):
            if folder.is_dir() and not folder.name.startswith("."):
                img_count = sum(
                    1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
                )
                soil_types.append({
                    "name": folder.name,
                    "display_name": folder.name.replace("_", " ").title(),
                    "image_count": img_count,
                })
                total_raw += img_count

    total_proc = 0
    if DATA_PROC_DIR.exists():
        total_proc = sum(
            1 for f in DATA_PROC_DIR.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        )

    sources: list[dict[str, Any]] = []
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sources.append(dict(row))

    return DatasetStatusOut(
        soil_types=soil_types,
        total_raw_images=total_raw,
        total_processed_images=total_proc,
        sources=sources,
    )

