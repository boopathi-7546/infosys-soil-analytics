from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Field, SoilImage, SoilTest
from ..schemas import FieldIn, FieldOut, FieldStats

router = APIRouter(prefix="/fields", tags=["fields"])


@router.post("", response_model=FieldOut, status_code=201)
def create_field(body: FieldIn, db: Session = Depends(get_db)):
    obj = Field(**body.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("", response_model=list[FieldOut])
def list_fields(db: Session = Depends(get_db)):
    return db.scalars(select(Field).order_by(Field.id)).all()


@router.get("/{field_id}", response_model=FieldOut)
def get_field(field_id: int, db: Session = Depends(get_db)):
    obj = db.get(Field, field_id)
    if not obj:
        raise HTTPException(404, "Field not found")
    return obj


@router.get("/{field_id}/stats", response_model=FieldStats)
def get_field_stats(field_id: int, db: Session = Depends(get_db)):
    """Summary statistics for a given farm field."""
    field = db.get(Field, field_id)
    if not field:
        raise HTTPException(404, "Field not found")

    total_tests = db.scalar(
        select(func.count(SoilTest.id)).where(SoilTest.field_id == field_id)
    ) or 0
    total_images = db.scalar(
        select(func.count(SoilImage.id)).where(SoilImage.field_id == field_id)
    ) or 0

    latest_test = db.scalar(
        select(SoilTest.tested_at)
        .where(SoilTest.field_id == field_id)
        .order_by(SoilTest.tested_at.desc())
        .limit(1)
    )

    avg_n = db.scalar(select(func.avg(SoilTest.nitrogen)).where(SoilTest.field_id == field_id))
    avg_p = db.scalar(select(func.avg(SoilTest.phosphorus)).where(SoilTest.field_id == field_id))
    avg_k = db.scalar(select(func.avg(SoilTest.potassium)).where(SoilTest.field_id == field_id))
    avg_ph = db.scalar(select(func.avg(SoilTest.ph)).where(SoilTest.field_id == field_id))
    avg_om = db.scalar(select(func.avg(SoilTest.organic_matter)).where(SoilTest.field_id == field_id))
    avg_moist = db.scalar(select(func.avg(SoilTest.moisture)).where(SoilTest.field_id == field_id))

    return FieldStats(
        field_id=field_id,
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


@router.delete("/{field_id}", status_code=204)
def delete_field(field_id: int, db: Session = Depends(get_db)):
    obj = db.get(Field, field_id)
    if not obj:
        raise HTTPException(404, "Field not found")
    db.delete(obj)
    db.commit()
    return None
