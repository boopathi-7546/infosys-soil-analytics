from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Farmer, Field, SoilImage, SoilTest
from ..schemas import FieldIn, FieldOut, FieldStats
from ..services.auth import get_current_farmer

router = APIRouter(prefix="/fields", tags=["fields"])


def get_farmer_field_or_404(field_id: int, current_farmer: Farmer, db: Session) -> Field:
    """Retrieve field ensuring it belongs to the authenticated farmer (prevents IDOR)."""
    field = db.get(Field, field_id)
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    # If field has an owner and it's not the current farmer, return 404 to avoid leaking field existence
    if field.farmer_id is not None and field.farmer_id != current_farmer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    return field


@router.post("", response_model=FieldOut, status_code=status.HTTP_201_CREATED)
def create_field(
    body: FieldIn,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Create a new farm field owned by the currently authenticated farmer."""
    data = body.model_dump()
    if not data.get("farmer_name"):
        data["farmer_name"] = current_farmer.full_name

    obj = Field(farmer_id=current_farmer.id, **data)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("", response_model=list[FieldOut])
def list_fields(
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """List all farm fields owned by the authenticated farmer."""
    return db.scalars(
        select(Field)
        .where(Field.farmer_id == current_farmer.id)
        .order_by(Field.id)
    ).all()


@router.get("/{field_id}", response_model=FieldOut)
def get_field(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Get details of a specific field (strictly owned by the authenticated farmer)."""
    return get_farmer_field_or_404(field_id, current_farmer, db)


@router.get("/{field_id}/stats", response_model=FieldStats)
def get_field_stats(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Summary statistics for a given farm field (strictly owned by the authenticated farmer)."""
    field = get_farmer_field_or_404(field_id, current_farmer, db)

    total_tests = db.scalar(
        select(func.count(SoilTest.id)).where(SoilTest.field_id == field.id)
    ) or 0
    total_images = db.scalar(
        select(func.count(SoilImage.id)).where(SoilImage.field_id == field.id)
    ) or 0

    latest_test = db.scalar(
        select(SoilTest.tested_at)
        .where(SoilTest.field_id == field.id)
        .order_by(SoilTest.tested_at.desc())
        .limit(1)
    )

    avg_n = db.scalar(select(func.avg(SoilTest.nitrogen)).where(SoilTest.field_id == field.id))
    avg_p = db.scalar(select(func.avg(SoilTest.phosphorus)).where(SoilTest.field_id == field.id))
    avg_k = db.scalar(select(func.avg(SoilTest.potassium)).where(SoilTest.field_id == field.id))
    avg_ph = db.scalar(select(func.avg(SoilTest.ph)).where(SoilTest.field_id == field.id))
    avg_om = db.scalar(select(func.avg(SoilTest.organic_matter)).where(SoilTest.field_id == field.id))
    avg_moist = db.scalar(select(func.avg(SoilTest.moisture)).where(SoilTest.field_id == field.id))

    return FieldStats(
        field_id=field.id,
        total_soil_tests=total_tests,
        total_images=total_images,
        latest_soil_test=latest_test,
        avg_nitrogen=round(avg_n, 2) if avg_n is not None else None,
        avg_phosphorus=round(avg_p, 2) if avg_p is not None else None,
        avg_potassium=round(avg_k, 2) if avg_k is not None else None,
        avg_ph=round(avg_ph, 2) if avg_ph is not None else None,
        avg_organic_matter=round(avg_om, 2) if avg_om is not None else None,
        avg_moisture=round(avg_moist, 2) if avg_moist is not None else None,
    )


@router.delete("/{field_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_field(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Delete a farm field and its cascaded tests/images (strictly owned by the authenticated farmer)."""
    obj = get_farmer_field_or_404(field_id, current_farmer, db)
    db.delete(obj)
    db.commit()
    return None


@router.post("/{field_id}/claim", response_model=FieldOut)
def claim_legacy_field(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Safe legacy migration endpoint: allows a registered farmer to claim an unassigned legacy field."""
    field = db.get(Field, field_id)
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    if field.farmer_id is not None and field.farmer_id != current_farmer.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Field is already assigned to another farmer")
    
    field.farmer_id = current_farmer.id
    db.commit()
    db.refresh(field)
    return field
