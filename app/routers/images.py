import base64
import re
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import Field, SoilImage
from ..schemas import (
    Base64UploadIn,
    ImageOut,
    ImageUploadOut,
    ImageUploadResponse,
    PreprocessResult,
)
from ..services import preprocessing, storage
from ..services.image_validation import ImageValidationError, validate_and_normalize

router = APIRouter(tags=["images"])


def _get_image_or_404(image_id: int, db: Session) -> SoilImage:
    obj = db.get(SoilImage, image_id)
    if not obj:
        raise HTTPException(404, "Image not found")
    return obj


@router.post("/fields/{field_id}/images", response_model=ImageUploadOut, status_code=201)
async def upload_image_for_field(
    field_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Multipart file upload (Gallery/file picker or direct file capture)."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(413, f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw, file.content_type)
    except ImageValidationError as e:
        raise HTTPException(status_code=422 if "large" not in str(e) else 413, detail=str(e))

    rel_path = storage.save_image(field_id, result.data)
    obj = SoilImage(
        field_id=field_id,
        file_path=rel_path,
        original_filename=(file.filename or "")[:255] or None,
        original_format=result.original_format,
        size_bytes=len(result.data),
        width=result.width,
        height=result.height,
        blur_score=result.blur_score,
    )
    try:
        db.add(obj)
        db.commit()
    except Exception:
        db.rollback()
        storage.delete_image(rel_path)
        raise
    db.refresh(obj)
    return ImageUploadOut(image=ImageOut.model_validate(obj), warnings=result.warnings)


@router.post("/images/upload", response_model=ImageUploadOut, status_code=201)
async def upload_image_flat(
    field_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Alternative flat upload endpoint matching frontend form-data conventions."""
    return await upload_image_for_field(field_id=field_id, file=file, db=db)


@router.post("/images/upload-base64", response_model=ImageUploadOut, status_code=201)
def upload_image_base64(payload: Base64UploadIn, db: Session = Depends(get_db)):
    """Receives base64-encoded image data from the browser Live Camera capture.

    Validates format and size, normalises to RGB JPEG, and persists through
    the exact same pipeline as standard multipart uploads.
    """
    field = db.get(Field, payload.field_id)
    if not field:
        raise HTTPException(404, "Field not found")

    raw_str = payload.image_data.strip()
    content_type = "image/jpeg"

    # Match data URL pattern e.g. data:image/png;base64,...
    match = re.match(r"^data:(image/[a-zA-Z0-9\+\-\.]+);base64,(.*)$", raw_str, re.DOTALL)
    if match:
        content_type = match.group(1)
        b64_content = match.group(2)
    else:
        b64_content = raw_str

    try:
        raw_bytes = base64.b64decode(b64_content, validate=True)
    except Exception:
        raise HTTPException(422, "Invalid base64 encoding")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(413, f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw_bytes, content_type)
    except ImageValidationError as e:
        raise HTTPException(status_code=422 if "large" not in str(e) else 413, detail=str(e))

    rel_path = storage.save_image(payload.field_id, result.data)
    obj = SoilImage(
        field_id=payload.field_id,
        file_path=rel_path,
        original_filename="camera_capture.jpg",
        original_format=result.original_format,
        size_bytes=len(result.data),
        width=result.width,
        height=result.height,
        blur_score=result.blur_score,
    )
    try:
        db.add(obj)
        db.commit()
    except Exception:
        db.rollback()
        storage.delete_image(rel_path)
        raise
    db.refresh(obj)
    return ImageUploadOut(image=ImageOut.model_validate(obj), warnings=result.warnings)


@router.get("/fields/{field_id}/images", response_model=list[ImageOut])
def list_images(field_id: int, db: Session = Depends(get_db)):
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    return db.scalars(
        select(SoilImage).where(SoilImage.field_id == field_id).order_by(SoilImage.id.desc())
    ).all()


@router.get("/images/{image_id}", response_model=ImageOut)
def get_image_metadata(image_id: int, db: Session = Depends(get_db)):
    return _get_image_or_404(image_id, db)


@router.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    obj = _get_image_or_404(image_id, db)
    try:
        return Response(storage.read_image(obj.file_path), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(404, "Image file missing on disk")


@router.get("/images/{image_id}/preprocessed")
def get_preprocessed_image(
    image_id: int,
    size: int = 224,
    background_removal: bool = True,
    db: Session = Depends(get_db),
):
    """Returns the processed image stream (JPEG 224x224) ready for CNN ingestion."""
    if not 32 <= size <= 1024:
        raise HTTPException(422, "size must be between 32 and 1024")
    obj = _get_image_or_404(image_id, db)
    try:
        img = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(404, "Image file missing on disk")
    out = preprocessing.preprocess(img, size=size, bg_removal=background_removal)
    return Response(preprocessing.encode_jpeg(out), media_type="image/jpeg")


@router.get("/images/{image_id}/preprocess", response_model=PreprocessResult)
def preprocess_and_inspect(
    image_id: int,
    db: Session = Depends(get_db),
):
    """Detailed OpenCV inspection metadata (original vs preprocessed metrics)."""
    obj = _get_image_or_404(image_id, db)
    try:
        img_bgr = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(404, "Image file missing on disk")

    res = preprocessing.inspect_and_preprocess(img_bgr, obj.id)
    return PreprocessResult(
        image_id=obj.id,
        processed_path=res["processed_path"],
        width=res["width"],
        height=res["height"],
        mean_rgb=res["mean_rgb"],
        quality_ok=res["quality_ok"],
        rejection_reason=res["rejection_reason"],
    )
