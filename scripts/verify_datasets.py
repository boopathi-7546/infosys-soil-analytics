"""Dataset verification, deduplication, and data-quality reporting for SoilSense AI.

Verifies SHA-256 checksums against data/DATASET_MANIFEST.csv, checks image
readability via Pillow, detects exact SHA-256 and perceptual dHash duplicates,
analyzes tabular missing/duplicate values, and writes docs/DATA_QUALITY_REPORT.md.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
MANIFEST_PATH = PROJECT_ROOT / "data" / "DATASET_MANIFEST.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "DATA_QUALITY_REPORT.md"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_hex(img: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash (dHash) for near-duplicate detection."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return f"{bits:016x}"


def verify_image_dataset(ds_dir: Path, max_dhash: int = 10000) -> dict:
    extract_dir = ds_dir / "extracted"
    if not extract_dir.exists():
        return {"status": "missing_extracted"}

    img_files = sorted(p for p in extract_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    corrupt: list[str] = []
    sha_map: dict[str, list[str]] = defaultdict(list)
    dhash_map: dict[str, list[str]] = defaultdict(list)
    class_counts: Counter[str] = Counter()

    for idx, p in enumerate(img_files):
        class_counts[p.parent.name] += 1
        h = sha256_file(p)
        sha_map[h].append(str(p.relative_to(extract_dir)))
        try:
            with Image.open(p) as im:
                im.verify()
            if idx < max_dhash:
                with Image.open(p) as im2:
                    dh = dhash_hex(im2)
                    dhash_map[dh].append(str(p.relative_to(extract_dir)))
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            corrupt.append(str(p.relative_to(extract_dir)))

    exact_dups = sum(len(v) - 1 for v in sha_map.values() if len(v) > 1)
    near_dups = sum(len(v) - 1 for v in dhash_map.values() if len(v) > 1)
    return {
        "type": "image",
        "total_images": len(img_files),
        "unique_sha256": len(sha_map),
        "exact_duplicates": exact_dups,
        "dhash_checked": min(len(img_files), max_dhash),
        "near_duplicates_dhash": near_dups,
        "corrupt_count": len(corrupt),
        "class_counts": dict(sorted(class_counts.items())),
    }


def verify_csv_dataset(csv_path: Path) -> dict:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing_counts = {col: 0 for col in fieldnames}
    row_hashes: set[str] = set()
    dup_rows = 0
    for row in rows:
        key = "|".join(str(row.get(c, "")) for c in fieldnames)
        rh = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if rh in row_hashes:
            dup_rows += 1
        else:
            row_hashes.add(rh)
        for c in fieldnames:
            val = (row.get(c) or "").strip()
            if val == "" or val.lower() in {"na", "nan", "null", "none"}:
                missing_counts[c] += 1

    non_zero_missing = {k: v for k, v in missing_counts.items() if v > 0}
    return {
        "type": "tabular",
        "total_rows": len(rows),
        "columns": fieldnames,
        "duplicate_rows": dup_rows,
        "missing_by_column": non_zero_missing,
    }


def verify_all(write_report: bool = True) -> dict[str, dict]:
    manifest_by_sha: dict[str, dict] = {}
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                manifest_by_sha[row["sha256"]] = row

    results: dict[str, dict] = {}
    if not EXTERNAL_DIR.exists():
        return results

    for ds_dir in sorted(EXTERNAL_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        archives = [p for p in ds_dir.iterdir() if p.is_file() and not p.name.endswith(".part")]
        archive_sha = sha256_file(archives[0]) if archives else None
        manifest_match = archive_sha in manifest_by_sha if archive_sha else False

        if (ds_dir / "extracted").exists():
            info = verify_image_dataset(ds_dir)
        elif archives and archives[0].suffix.lower() == ".csv":
            info = verify_csv_dataset(archives[0])
        else:
            info = {"type": "unknown"}

        info["archive_sha256"] = archive_sha
        info["manifest_verified"] = manifest_match
        results[ds_dir.name] = info

    if write_report:
        _write_quality_report(results)
    return results


def _write_quality_report(results: dict[str, dict]) -> None:
    lines = [
        "# SoilSense AI – Acquired Dataset Quality & Leakage Audit",
        "",
        "Generated automatically by `scripts/verify_datasets.py`.",
        "",
    ]
    for key, info in results.items():
        lines.append(f"## Dataset: `{key}`")
        lines.append(f"- **Manifest SHA-256 Verified:** `{info.get('manifest_verified')}`")
        lines.append(f"- **Archive SHA-256:** `{info.get('archive_sha256')}`")
        if info.get("type") == "image":
            lines.append(f"- **Total Images:** {info['total_images']}")
            lines.append(f"- **Unique SHA-256:** {info['unique_sha256']} (exact duplicates: {info['exact_duplicates']})")
            lines.append(f"- **Perceptual dHash Near-Duplicates (first {info['dhash_checked']}):** {info['near_duplicates_dhash']}")
            lines.append(f"- **Corrupt Images:** {info['corrupt_count']}")
            lines.append(f"- **Class Distribution:** `{info['class_counts']}`")
        elif info.get("type") == "tabular":
            lines.append(f"- **Total Rows:** {info['total_rows']}")
            lines.append(f"- **Duplicate Rows:** {info['duplicate_rows']}")
            lines.append(f"- **Columns ({len(info['columns'])}):** `{', '.join(info['columns'][:20])}`" + (" ..." if len(info["columns"]) > 20 else ""))
            lines.append(f"- **Columns with Missing Values:** `{len(info['missing_by_column'])}`")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Wrote {REPORT_PATH}")


if __name__ == "__main__":
    res = verify_all(write_report=True)
    for k, v in res.items():
        print(k, "->", {kk: vv for kk, vv in v.items() if kk != "columns"})

