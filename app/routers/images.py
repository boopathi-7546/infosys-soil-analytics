from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import Field, SoilImage
from ..schemas import ImageOut, ImageUploadOut
from ..services import preprocessing, storage
from ..services.image_validation import ImageValidationError, validate_and_normalize

router = APIRouter(tags=["images"])


@router.post("/fields/{field_id}/images", response_model=ImageUploadOut, status_code=201)
async def upload_image(field_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Accepts a JPEG/PNG (file picker or phone camera capture), validates it server-side,
    normalises it to an RGB JPEG, stores it on disk and records metadata in PostgreSQL."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    raw = await file.read(max_bytes + 1)  # never read more than the limit
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
        storage.delete_image(rel_path)  # don't leave orphan files if the DB insert fails
        raise
    db.refresh(obj)
    return ImageUploadOut(image=ImageOut.model_validate(obj), warnings=result.warnings)


@router.get("/fields/{field_id}/images", response_model=list[ImageOut])
def list_images(field_id: int, db: Session = Depends(get_db)):
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    return db.scalars(
        select(SoilImage).where(SoilImage.field_id == field_id).order_by(SoilImage.id)
    ).all()


def _get_image_or_404(image_id: int, db: Session) -> SoilImage:
    obj = db.get(SoilImage, image_id)
    if not obj:
        raise HTTPException(404, "Image not found")
    return obj


@router.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    obj = _get_image_or_404(image_id, db)
    try:
        return Response(storage.read_image(obj.file_path), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(404, "Image file missing on disk")


@router.get("/images/{image_id}/preprocessed")
def get_preprocessed(image_id: int, size: int = 224, background_removal: bool = True,
                     db: Session = Depends(get_db)):
    """Returns the image after the OpenCV pipeline (what the CNN will see). Handy for demos."""
    if not 32 <= size <= 1024:
        raise HTTPException(422, "size must be between 32 and 1024")
    obj = _get_image_or_404(image_id, db)
    try:
        img = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(404, "Image file missing on disk")
    out = preprocessing.preprocess(img, size=size, bg_removal=background_removal)
    return Response(preprocessing.encode_jpeg(out), media_type="image/jpeg")
