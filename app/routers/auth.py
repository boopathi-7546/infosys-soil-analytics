"""Authentication router: registration, login, profile, and Google OAuth endpoints."""
from datetime import timedelta
import urllib.parse

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import Farmer
from ..schemas import (
    FarmerLoginIn,
    FarmerOut,
    FarmerRegisterIn,
    GoogleAuthUrlOut,
    GoogleCallbackIn,
    TokenOut,
)
from ..services.auth import (
    create_access_token,
    get_current_farmer,
    hash_password,
    verify_password,
)
from ..services.google_oauth import (
    authenticate_or_link_google_user,
    exchange_code_for_tokens,
    get_google_auth_url,
    get_google_user_info,
    verify_oauth_state,
)

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


# ---------------------------------------------------------------------------
# Google OAuth 2.0 / OpenID Connect Endpoints
# ---------------------------------------------------------------------------

@router.get("/google/url", response_model=GoogleAuthUrlOut)
def google_auth_url(redirect_uri: str | None = None):
    """Generate and return Google OAuth authorization URL with signed state."""
    auth_url, _ = get_google_auth_url(redirect_uri=redirect_uri)
    return GoogleAuthUrlOut(url=auth_url)


@router.get("/google/login", include_in_schema=False)
def google_login_redirect(redirect_uri: str | None = None):
    """Direct browser redirect to Google OAuth consent screen."""
    auth_url, _ = get_google_auth_url(redirect_uri=redirect_uri)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


def _build_auth_bridge_html(access_token: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SoilSense AI — Signing In</title>
</head>
<body style="font-family: system-ui, -apple-system, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #F8F6F0;">
  <div style="text-align: center; background: white; padding: 32px 40px; border-radius: 14px; box-shadow: 0 4px 20px rgba(27,67,50,0.08); max-width: 400px; width: 90%;">
    <div style="font-size: 36px; margin-bottom: 12px;">🌱</div>
    <h2 style="color: #1B4332; margin-bottom: 8px; font-size: 18px;">Signing you in to SoilSense AI...</h2>
    <p style="color: #5C675E; font-size: 13px;">Redirecting to your farm dashboard...</p>
  </div>
  <script>
    try {{
      localStorage.setItem('soilsense_token', '{access_token}');
    }} catch (_) {{}}
    window.location.replace('/?token=' + encodeURIComponent('{access_token}'));
  </script>
</body>
</html>"""


@router.get("/google/callback")
def google_callback_get(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
    format: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Handle Google OAuth 2.0 callback for browser redirect or API client."""
    accept_header = request.headers.get("accept", "")
    wants_json = format == "json" or "application/json" in accept_header

    if error:
        err_msg = error_description or error or "Google authentication was cancelled."
        if wants_json:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
        return RedirectResponse(
            url=f"/?auth_error={urllib.parse.quote(err_msg)}",
            status_code=status.HTTP_302_FOUND,
        )

    if not code or not state:
        err_msg = "Missing authorization code or state from Google callback."
        if wants_json:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
        return RedirectResponse(
            url=f"/?auth_error={urllib.parse.quote(err_msg)}",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        # Step 1: CSRF state validation
        verify_oauth_state(state)

        # Step 2: Code exchange
        tokens = exchange_code_for_tokens(code)
        google_access_token = tokens.get("access_token")
        if not google_access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google token response did not contain an access_token.",
            )

        # Step 3: Retrieve verified user info
        user_info = get_google_user_info(google_access_token)

        # Step 4: Account creation or secure linking
        farmer = authenticate_or_link_google_user(user_info, db)

        # Step 5: Issue SoilSense JWT
        expires_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES
        access_token = create_access_token(
            data={"sub": str(farmer.id), "email": farmer.email},
            expires_delta=timedelta(minutes=expires_minutes),
        )

        if wants_json:
            return TokenOut(
                access_token=access_token,
                token_type="bearer",
                expires_in=expires_minutes * 60,
                farmer=FarmerOut.model_validate(farmer),
            )

        # Browser response: return HTML bridge with immediate token storage & redirect
        return HTMLResponse(content=_build_auth_bridge_html(access_token))

    except HTTPException as exc:
        if wants_json:
            raise exc
        return RedirectResponse(
            url=f"/?auth_error={urllib.parse.quote(str(exc.detail))}",
            status_code=status.HTTP_302_FOUND,
        )
    except Exception as exc:
        if wants_json:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication failed: {exc}",
            )
        return RedirectResponse(
            url=f"/?auth_error={urllib.parse.quote(f'Authentication error: {exc}')}",
            status_code=status.HTTP_302_FOUND,
        )


@router.post("/google/callback", response_model=TokenOut)
def google_callback_post(payload: GoogleCallbackIn, db: Session = Depends(get_db)):
    """API-friendly POST endpoint for exchanging Google OAuth code and state for a JWT."""
    # Step 1: CSRF state verification
    verify_oauth_state(payload.state)

    # Step 2: Code exchange
    tokens = exchange_code_for_tokens(payload.code)
    google_access_token = tokens.get("access_token")
    if not google_access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token response did not contain an access_token.",
        )

    # Step 3: User info
    user_info = get_google_user_info(google_access_token)

    # Step 4: Link or create farmer
    farmer = authenticate_or_link_google_user(user_info, db)

    # Step 5: JWT token
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
