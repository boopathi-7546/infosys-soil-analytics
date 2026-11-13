from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Field
from ..schemas import FieldIn, FieldOut

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
