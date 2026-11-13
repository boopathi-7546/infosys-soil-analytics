"""Local file storage for images. Swap this module for S3/GCS in Milestone 3."""
import uuid
from pathlib import Path

from .. import config


def save_image(field_id: int, data: bytes) -> str:
    """Write bytes under UPLOAD_DIR/<field_id>/<uuid>.jpg and return the relative path."""
    rel = Path(str(field_id)) / f"{uuid.uuid4().hex}.jpg"
    dest = config.UPLOAD_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return rel.as_posix()


def read_image(rel_path: str) -> bytes:
    base = config.UPLOAD_DIR.resolve()
    path = (base / rel_path).resolve()
    if base not in path.parents:  # path-traversal guard
        raise FileNotFoundError(rel_path)
    return path.read_bytes()


def delete_image(rel_path: str) -> None:
    try:
        (config.UPLOAD_DIR / rel_path).unlink(missing_ok=True)
    except OSError:
        pass
