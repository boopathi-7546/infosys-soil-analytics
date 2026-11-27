import base64
import math
import re
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import AnalysisResult, Farmer, Field, SoilImage, SoilTest
from ..schemas import (
    Base64UploadIn,
    ImageOut,
    ImagePairSoilTestIn,
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
    # Strictly require positive ownership match — treat unclaimed (farmer_id IS NULL) as inaccessible
    if not obj.field or obj.field.farmer_id != current_farmer.id:
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
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw, file.content_type)
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if "large" not in str(e) else status.HTTP_413_CONTENT_TOO_LARGE,
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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid base64 encoding")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=f"Image exceeds maximum upload size of {config.MAX_UPLOAD_MB}MB")

    try:
        result = validate_and_normalize(raw_bytes, content_type)
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if "large" not in str(e) else status.HTTP_413_CONTENT_TOO_LARGE,
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


@router.post("/images/{image_id}/pair-soil-test", response_model=ImageOut)
def pair_image_with_soil_test(
    image_id: int,
    body: ImagePairSoilTestIn,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Pair an existing soil image with a soil-test record from the same field."""
    img = _get_farmer_image_or_404(image_id, current_farmer, db)
    test = db.get(SoilTest, body.soil_test_id)
    if not test or not test.field or test.field.farmer_id != current_farmer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soil test not found")
    if img.field_id != test.field_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Soil image and soil test must belong to the same field",
        )
    if img.soil_test_id is not None and img.soil_test_id != test.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Soil image is already paired with another soil test",
        )

    img.soil_test_id = test.id
    analyses = db.scalars(
        select(AnalysisResult).where(
            AnalysisResult.image_id == img.id,
            AnalysisResult.soil_test_id.is_(None),
        )
    ).all()
    for ar in analyses:
        ar.soil_test_id = test.id

    db.commit()
    db.refresh(img)
    return img


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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="size must be between 32 and 1024")
    obj = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        img = preprocessing.decode_bgr(storage.read_image(obj.file_path))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")
    out = preprocessing.preprocess(img, size=size, bg_removal=background_removal)
    return Response(preprocessing.encode_jpeg(out), media_type="image/jpeg")


@router.get("/images/{image_id}/gradcam")
def get_image_gradcam(
    image_id: int,
    target_class: str | None = Query(default=None, description="Optional soil class name to explain"),
    alpha: float = Query(default=0.45, ge=0.0, le=1.0, description="Heatmap overlay opacity"),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Generate a Grad-CAM attention heatmap overlay for an uploaded soil image."""
    obj = _get_farmer_image_or_404(image_id, current_farmer, db)
    try:
        raw_bytes = storage.read_image(obj.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on disk")

    classifier = get_soil_classifier()
    try:
        overlay_bytes, meta = classifier.generate_gradcam(
            raw_bytes,
            target_class=target_class,
            alpha=alpha,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(ve))
    except SoilModelError as sme:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Soil classification model unavailable: {sme}",
        )
    except InvalidImagePredictionError as ipe:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid or unreadable image data: {ipe}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failure: {exc}",
        )

    return Response(
        overlay_bytes,
        media_type="image/jpeg",
        headers={
            "X-GradCAM-Target-Class": meta["target_class"],
            "X-GradCAM-Predicted-Class": meta["predicted_soil_type"],
        },
    )


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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


def _safe_float(val: object, default: float = 0.0) -> float:
    """Safely convert JSON value to a finite float, rejecting booleans and invalid types."""
    if val is None or isinstance(val, bool):
        return default
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _normalize_prediction_result(raw_result: object) -> dict | None:
    """Validate and normalize stored JSONB prediction data; return None if missing or null."""
    if not isinstance(raw_result, dict):
        return None
    soil_type = raw_result.get("predicted_soil_type")
    if not isinstance(soil_type, str) or not soil_type.strip():
        return None
    probs = raw_result.get("probabilities")
    safe_probs: dict[str, float] = {}
    if isinstance(probs, dict):
        for k, v in probs.items():
            if isinstance(k, str):
                safe_probs[k] = _safe_float(v, 0.0)
    model_ver = raw_result.get("model_version")
    normalized_model_ver = model_ver.strip() if isinstance(model_ver, str) and model_ver.strip() else None
    raw_epoch = raw_result.get("best_epoch")
    safe_epoch = raw_epoch if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) else None
    pred_type = raw_result.get("type")
    return {
        "predicted_soil_type": soil_type,
        "confidence": _safe_float(raw_result.get("confidence"), 0.0),
        "confidence_percentage": _safe_float(raw_result.get("confidence_percentage"), 0.0),
        "probabilities": safe_probs,
        "model_version": normalized_model_ver,
        "best_epoch": safe_epoch,
        "type": pred_type if isinstance(pred_type, str) and pred_type else "cnn_prediction",
    }


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

    normalized = _normalize_prediction_result(analysis.result)
    if not normalized:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction data not found on analysis result")

    return SoilPredictionOut(
        id=analysis.id,
        field_id=analysis.field_id,
        image_id=analysis.image_id,
        soil_test_id=analysis.soil_test_id,
        created_at=analysis.created_at,
        prediction=SoilPredictionData(**normalized),
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
        normalized = _normalize_prediction_result(analysis.result)
        if not normalized:
            continue
        results.append(
            PredictionHistoryItem(
                id=analysis.id,
                field_id=analysis.field_id,
                field_name=field_name,
                image_id=analysis.image_id,
                created_at=analysis.created_at,
                predicted_soil_type=normalized["predicted_soil_type"],
                confidence=normalized["confidence"],
                confidence_percentage=normalized["confidence_percentage"],
                probabilities=normalized["probabilities"],
                model_version=normalized["model_version"],
                best_epoch=normalized["best_epoch"],
            )
        )
    return results


@router.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem)
def get_prediction_detail(
    prediction_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """
    Retrieve full details for a specific AI soil prediction record.
    Strictly isolated: returns HTTP 404 if the record does not exist or belongs to another farmer.
    """
    row = db.execute(
        select(AnalysisResult, Field.name.label("field_name"))
        .join(Field, AnalysisResult.field_id == Field.id)
        .where(
            AnalysisResult.id == prediction_id,
            Field.farmer_id == current_farmer.id,
        )
    ).first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction record not found",
        )

    analysis, field_name = row
    normalized = _normalize_prediction_result(analysis.result)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction data not found on analysis result",
        )

    return PredictionHistoryItem(
        id=analysis.id,
        field_id=analysis.field_id,
        field_name=field_name,
        image_id=analysis.image_id,
        created_at=analysis.created_at,
        predicted_soil_type=normalized["predicted_soil_type"],
        confidence=normalized["confidence"],
        confidence_percentage=normalized["confidence_percentage"],
        probabilities=normalized["probabilities"],
        model_version=normalized["model_version"],
        best_epoch=normalized["best_epoch"],
    )


@router.get("/fields/{field_id}/predictions", response_model=list[PredictionHistoryItem])
def list_field_predictions(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Retrieve saved predictions for a specific field owned by the authenticated farmer."""
    return list_prediction_history(field_id=field_id, current_farmer=current_farmer, db=db)
