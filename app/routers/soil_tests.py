from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Field, SoilTest
from ..schemas import SoilTestIn, SoilTestOut

router = APIRouter(tags=["soil-tests"])


@router.post("/fields/{field_id}/soil-tests", response_model=SoilTestOut, status_code=201)
def create_soil_test(field_id: int, body: SoilTestIn, db: Session = Depends(get_db)):
    """Stores laboratory soil parameters (N, P, K, pH, organic matter, moisture) for a field."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    obj = SoilTest(field_id=field_id, **body.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("/fields/{field_id}/soil-tests", response_model=list[SoilTestOut])
def list_soil_tests(field_id: int, db: Session = Depends(get_db)):
    """History for one field, newest first (used later for degradation trend analysis)."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    return db.scalars(
        select(SoilTest).where(SoilTest.field_id == field_id).order_by(SoilTest.tested_at.desc(), SoilTest.id.desc())
    ).all()


@router.get("/soil-tests/{test_id}", response_model=SoilTestOut)
def get_soil_test(test_id: int, db: Session = Depends(get_db)):
    obj = db.get(SoilTest, test_id)
    if not obj:
        raise HTTPException(404, "Soil test not found")
    return obj
