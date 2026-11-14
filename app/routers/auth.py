"""Authentication router: registration, login, and profile endpoints."""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import Farmer
from ..schemas import FarmerLoginIn, FarmerOut, FarmerRegisterIn, TokenOut
from ..services.auth import create_access_token, get_current_farmer, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(payload: FarmerRegisterIn, db: Session = Depends(get_db)):
    """Register a new farmer account and return an access token."""
    normalized_email = payload.email.lower().strip()
    
    # Check if email is already registered
    existing = db.scalar(
        select(Farmer).where(func.lower(Farmer.email) == normalized_email)
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered. Please sign in or use another email.",
        )

    # Hash password securely
    hashed_pwd = hash_password(payload.password)
    farmer = Farmer(
        full_name=payload.full_name.strip(),
        email=normalized_email,
        hashed_password=hashed_pwd,
        is_active=True,
    )
    db.add(farmer)
    db.commit()
    db.refresh(farmer)

    # Generate initial access token
    expires_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES
    access_token = create_access_token(
        data={"sub": str(farmer.id), "email": farmer.email},
        expires_delta=timedelta(minutes=expires_minutes),
    )

    return TokenOut(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_minutes * 60,
        farmer=FarmerOut.model_validate(farmer),
    )


@router.post("/login", response_model=TokenOut)
def login(payload: FarmerLoginIn, db: Session = Depends(get_db)):
    """Authenticate a farmer and return a signed JWT access token."""
    normalized_email = payload.email.lower().strip()
    farmer = db.scalar(
        select(Farmer).where(func.lower(Farmer.email) == normalized_email)
    )

    if not farmer or not verify_password(payload.password, farmer.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not farmer.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Farmer account is deactivated. Please contact support.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES
    access_token = create_access_token(
        data={"sub": str(farmer.id), "email": farmer.email},
        expires_delta=timedelta(minutes=expires_minutes),
    )

    return TokenOut(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_minutes * 60,
        farmer=FarmerOut.model_validate(farmer),
    )


@router.get("/me", response_model=FarmerOut)
def get_profile(current_farmer: Farmer = Depends(get_current_farmer)):
    """Return the profile of the currently authenticated farmer."""
    return current_farmer
