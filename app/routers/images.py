import base64
import re
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import AnalysisResult, Farmer, Field, SoilImage
from ..schemas import (
    Base64UploadIn,
    ImageOut,
    ImageUploadOut,
    PreprocessResult,
    PredictionHistoryItem,
    SoilPredictionData,
    SoilPredictionOut,
)
from ..services import preprocessing, storage
from ..services.auth import get_current_farmer
from ..services.image_validation import ImageValidationError, validate_and_normalize
from ..services.soil_classifier import (
    InvalidImagePredictionError,
    SoilModelError,
    get_soil_classifier,
)
from .fields import get_farmer_field_or_404

router = APIRouter(tags=["images"])


def _get_farmer_image_or_404(image_id: int, current_farmer: Farmer, db: Session) -> SoilImage:
    """Retrieve image verifying that its parent field belongs to the current farmer (prevents IDOR)."""
    obj = db.get(SoilImage, image_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if obj.field and obj.field.farmer_id is not None and obj.field.farmer_id != current_farmer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return obj


@router.post("/fields/{field_id}/images", response_model=ImageUploadOut, status_code=status.HTTP_201_CREATED)
async def upload_image_for_field(
    field_id: int,
    file: UploadFile = File(...),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Multipart file upload (strictly owned by authenticated farmer)."""
    field = get_farmer_field_or_404(field_id, current_farmer, db)

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw, file.content_type)
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if "large" not in str(e) else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )

    rel_path = storage.save_image(field.id, result.data)
    obj = SoilImage(
        field_id=field.id,
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


@router.post("/images/upload", response_model=ImageUploadOut, status_code=status.HTTP_201_CREATED)
async def upload_image_flat(
    field_id: int = Form(...),
    file: UploadFile = File(...),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Alternative flat upload endpoint matching frontend form-data conventions."""
    return await upload_image_for_field(field_id=field_id, file=file, current_farmer=current_farmer, db=db)


@router.post("/images/upload-base64", response_model=ImageUploadOut, status_code=status.HTTP_201_CREATED)
def upload_image_base64(
    payload: Base64UploadIn,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Receives base64-encoded image data from live camera capture (strictly checks farm ownership)."""
    field = get_farmer_field_or_404(payload.field_id, current_farmer, db)

    raw_str = payload.image_data.strip()
    content_type = "image/jpeg"

    match = re.match(r"^data:(image/[a-zA-Z0-9\+\-\.]+);base64,(.*)$", raw_str, re.DOTALL)
    if match:
        content_type = match.group(1)
        b64_content = match.group(2)
    else:
        b64_content = raw_str

    try:
        raw_bytes = base64.b64decode(b64_content, validate=True)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid base64 encoding")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw_bytes, content_type)
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if "large" not in str(e) else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )

    rel_path = storage.save_image(field.id, result.data)
    obj = SoilImage(
        field_id=field.id,
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
def list_images(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """List images for a specific field owned by the authenticated farmer."""
    field = get_farmer_field_or_404(field_id, current_farmer, db)
    return db.scalars(
        select(SoilImage).where(SoilImage.field_id == field.id).order_by(SoilImage.uploaded_at.desc(), SoilImage.id.desc())
    ).all()


@router.get("/images", response_model=list[ImageOut])
def list_all_images(
    field_id: int | None = Query(default=None, description="Optional field ID filter. If omitted, returns all images for the authenticated farmer across all fields."),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """List soil photographs for the authenticated farmer (across all fields or filtered by field)."""
    stmt = (
        select(SoilImage)
        .join(Field, SoilImage.field_id == Field.id)
        .where(Field.farmer_id == current_farmer.id)
    )
    if field_id is not None:
        get_farmer_field_or_404(field_id, current_farmer, db)
        stmt = stmt.where(SoilImage.field_id == field_id)

    return db.scalars(stmt.order_by(SoilImage.uploaded_at.desc(), SoilImage.id.desc())).all()


@router.get("/images/{image_id}", response_model=ImageOut)
def get_image_metadata(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Retrieve metadata of an image (strictly owned by authenticated farmer)."""
    return _get_farmer_image_or_404(image_id, current_farmer, db)


@router.get("/images/{image_id}/file")
def get_image_file(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Securely stream image bytes with farmer ownership check (prevents public unrestricted access)."""
    obj = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        return Response(storage.read_image(obj.file_path), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")


@router.get("/images/{image_id}/preprocessed")
def get_preprocessed_image(
    image_id: int,
    size: int = 224,
    background_removal: bool = True,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Stream preprocessed image bytes (224x224) with ownership validation."""
    if not 32 <= size <= 1024:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="size must be between 32 and 1024")
    obj = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        img = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")
    out = preprocessing.preprocess(img, size=size, bg_removal=background_removal)
    return Response(preprocessing.encode_jpeg(out), media_type="image/jpeg")


@router.get("/images/{image_id}/preprocess", response_model=PreprocessResult)
def preprocess_and_inspect(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Detailed OpenCV inspection metadata (strictly owned by authenticated farmer)."""
    obj = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        img_bgr = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")

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


# ---------------------------------------------------------------------------
# CNN Soil Prediction Endpoints
# ---------------------------------------------------------------------------
@router.post("/images/{image_id}/predict", response_model=SoilPredictionOut, status_code=status.HTTP_200_OK)
def predict_soil_image(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """
    Run CNN soil prediction on an uploaded image and store results in PostgreSQL.
    Strictly verifies ownership: Farmer A cannot run predictions on Farmer B's images.
    """
    image = _get_farmer_image_or_404(image_id, current_farmer, db)

    try:
        raw_bytes = storage.read_image(image.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")

    classifier = get_soil_classifier()
    try:
        pred_data = classifier.predict(raw_bytes)
    except SoilModelError as sme:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Soil classification model unavailable: {sme}",
        )
    except InvalidImagePredictionError as ipe:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or unreadable image data: {ipe}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CNN inference failure: {exc}",
        )

    # Save prediction details in PostgreSQL linked to image, field and farmer
    analysis_record = AnalysisResult(
        field_id=image.field_id,
        image_id=image.id,
        result=pred_data,
    )
    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    return SoilPredictionOut(
        id=analysis_record.id,
        field_id=analysis_record.field_id,
        image_id=analysis_record.image_id,
        soil_test_id=analysis_record.soil_test_id,
        created_at=analysis_record.created_at,
        prediction=SoilPredictionData(**pred_data),
    )


@router.get("/images/{image_id}/prediction", response_model=SoilPredictionOut)
def get_image_prediction(
    image_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Get the latest recorded CNN prediction for a specific image."""
    image = _get_farmer_image_or_404(image_id, current_farmer, db)
    analysis = db.scalars(
        select(AnalysisResult)
        .where(AnalysisResult.image_id == image.id)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    ).first()

    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction found for this image")

    return SoilPredictionOut(
        id=analysis.id,
        field_id=analysis.field_id,
        image_id=analysis.image_id,
        soil_test_id=analysis.soil_test_id,
        created_at=analysis.created_at,
        prediction=SoilPredictionData(**analysis.result),
    )


@router.get("/predictions", response_model=list[PredictionHistoryItem])
def list_prediction_history(
    field_id: int | None = Query(default=None, description="Filter predictions by field ID"),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """
    Retrieve all saved AI predictions for the authenticated farmer's plots.
    Strictly isolated: only returns predictions belonging to the authenticated farmer.
    """
    query = (
        select(AnalysisResult, Field.name.label("field_name"))
        .join(Field, AnalysisResult.field_id == Field.id)
        .where(Field.farmer_id == current_farmer.id)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )

    if field_id is not None:
        get_farmer_field_or_404(field_id, current_farmer, db)
        query = query.where(AnalysisResult.field_id == field_id)

    rows = db.execute(query).all()
    results: list[PredictionHistoryItem] = []
    for analysis, field_name in rows:
        res = analysis.result or {}
        if "predicted_soil_type" not in res:
            continue
        results.append(
            PredictionHistoryItem(
                id=analysis.id,
                field_id=analysis.field_id,
                field_name=field_name,
                image_id=analysis.image_id,
                created_at=analysis.created_at,
                predicted_soil_type=res.get("predicted_soil_type", "unknown"),
                confidence=res.get("confidence", 0.0),
                confidence_percentage=res.get("confidence_percentage", 0.0),
                probabilities=res.get("probabilities", {}),
                model_version=res.get("model_version", "efficientnet_b0_v1.0"),
                best_epoch=res.get("best_epoch"),
            )
        )
    return results


@router.get("/fields/{field_id}/predictions", response_model=list[PredictionHistoryItem])
def list_field_predictions(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Retrieve saved predictions for a specific field owned by the authenticated farmer."""
    return list_prediction_history(field_id=field_id, current_farmer=current_farmer, db=db)
