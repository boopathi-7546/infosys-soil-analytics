"""Reusable legal public dataset acquisition script for SoilSense AI.

Downloads open-licensed datasets into data/external/<key>/, preserves original
archives, prevents zip-slip during extraction, computes SHA-256 checksums, and
generates data/DATASET_MANIFEST.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
MANIFEST_PATH = PROJECT_ROOT / "data" / "DATASET_MANIFEST.csv"

MANIFEST_FIELDS = [
    "dataset_name",
    "source",
    "doi",
    "license",
    "download_date",
    "version",
    "file_count",
    "sample_count",
    "classes",
    "sha256",
    "notes",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    url: str
    filename: str
    source: str
    doi: str
    license: str
    version: str
    is_zip: bool
    notes: str


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "rice_npk_fertility": DatasetSpec(
        key="rice_npk_fertility",
        dataset_name="Training Data of Soil NPK Fertility for Rice Cultivation",
        url="https://zenodo.org/api/records/15307297/files/Machine%20Learning-Training%20Data%20(Rice).csv/content",
        filename="Machine_Learning_Training_Data_Rice.csv",
        source="Zenodo",
        doi="10.5281/zenodo.15307297",
        license="CC BY 4.0",
        version="1.0",
        is_zip=False,
        notes="Structured NPK soil fertility CSV for rice cultivation",
    ),
    "soilhealthdb_v2": DatasetSpec(
        key="soilhealthdb_v2",
        dataset_name="SoilHealthDB V2 Global Longitudinal Soil Health Database",
        url="https://raw.githubusercontent.com/jinshijian/SoilHealthDB/master/data/SoilHealthDB_V2.csv",
        filename="SoilHealthDB_V2.csv",
        source="GitHub / Zenodo (jinshijian/SoilHealthDB)",
        doi="10.5281/zenodo.17565204",
        license="CC BY 4.0",
        version="V2",
        is_zip=False,
        notes="Multi-year global soil health observations with SOC/TN/management covariates",
    ),
    "mpld_nutrient_deficiency": DatasetSpec(
        key="mpld_nutrient_deficiency",
        dataset_name="Maize Phosphorus Leaf Deficiency (MPLD) Database 224x224",
        url="https://zenodo.org/api/records/10041514/files/MPLD_databse.zip/content",
        filename="MPLD_databse.zip",
        source="Zenodo",
        doi="10.5281/zenodo.10041514",
        license="CC BY 4.0",
        version="1.0",
        is_zip=True,
        notes="224x224 RGB leaf images for visible phosphorus deficiency vs healthy maize",
    ),
    "soil_texture_92k": DatasetSpec(
        key="soil_texture_92k",
        dataset_name="A 92k Soil Texture Image Patch Dataset (Loam, Sand, Clay)",
        url="https://zenodo.org/api/records/20404766/files/dataset_v3_patches.zip/content",
        filename="dataset_v3_patches.zip",
        source="Zenodo",
        doi="10.5281/zenodo.20404766",
        license="CC BY 4.0",
        version="v3",
        is_zip=True,
        notes="224x224 RGB soil texture patches categorized into Loam, Sand, and Clay",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_zip(archive_path: Path, target_dir: Path) -> int:
    """Extract zip safely while preventing zip-slip path traversal."""
    target_resolved = target_dir.resolve()
    extracted = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(target_resolved)):
                raise RuntimeError(f"Unsafe zip path detected: {member.filename}")
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, member_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
    return extracted


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "SoilSenseAI-DatasetDownloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp_dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    os.replace(tmp_dest, dest)


def _count_samples_and_classes(extract_dir: Path, is_zip: bool, archive_path: Path) -> tuple[int, int, str]:
    if not is_zip:
        with archive_path.open("r", encoding="utf-8", errors="replace") as f:
            row_count = max(0, sum(1 for _ in f) - 1)
        return 1, row_count, "tabular_csv"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [p for p in extract_dir.rglob("*") if p.is_file()]
    img_files = [p for p in all_files if p.suffix.lower() in img_exts]
    classes = sorted({p.parent.name for p in img_files})
    return len(all_files), len(img_files), ";".join(classes[:20])


def acquire_dataset(spec: DatasetSpec, force: bool = False) -> dict[str, str]:
    ds_dir = EXTERNAL_DIR / spec.key
    archive_path = ds_dir / spec.filename
    extract_dir = ds_dir / "extracted"

    if force or not archive_path.is_file():
        print(f"[DOWNLOAD] {spec.key} <- {spec.url}")
        download_file(spec.url, archive_path)
    else:
        print(f"[CACHE] {spec.key} archive already present at {archive_path}")

    checksum = sha256_file(archive_path)
    if spec.is_zip and (force or not extract_dir.exists()):
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[EXTRACT] {archive_path} -> {extract_dir}")
        safe_extract_zip(archive_path, extract_dir)

    file_count, sample_count, classes = _count_samples_and_classes(extract_dir, spec.is_zip, archive_path)
    return {
        "dataset_name": spec.dataset_name,
        "source": f"{spec.source} ({spec.url})",
        "doi": spec.doi,
        "license": spec.license,
        "download_date": dt.date.today().isoformat(),
        "version": spec.version,
        "file_count": str(file_count),
        "sample_count": str(sample_count),
        "classes": classes,
        "sha256": checksum,
        "notes": spec.notes,
    }


def write_manifest(rows: list[dict[str, str]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing[row["dataset_name"]] = row
    for r in rows:
        existing[r["dataset_name"]] = r

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)
    print(f"[MANIFEST] Updated {MANIFEST_PATH} ({len(existing)} datasets)")


def main():
    parser = argparse.ArgumentParser(description="Download verified open-licensed datasets for SoilSense AI.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset keys to download.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if archive exists.")
    args = parser.parse_args()

    rows = [acquire_dataset(DATASET_REGISTRY[k], force=args.force) for k in args.datasets]
    write_manifest(rows)


if __name__ == "__main__":
    main()

