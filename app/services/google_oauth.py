"""Google OAuth 2.0 / OpenID Connect service for SoilSense AI.

Handles:
1. Google OAuth authorization URL generation with cryptographically signed state (CSRF protection).
2. Authorization code exchange for Google tokens (access token, id token).
3. User profile retrieval from Google OpenID Connect userinfo endpoint.
4. Secure farmer account creation and account linking for verified emails.
"""
from datetime import datetime, timedelta, timezone
import secrets
from typing import Any
import urllib.parse

from fastapi import HTTPException, status
import httpx
import jwt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..models import Farmer
from .auth import create_access_token, hash_password

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"


def is_google_auth_configured() -> bool:
    """Return True if Google OAuth client ID and secret are configured."""
    client_id = config.GOOGLE_CLIENT_ID
    client_secret = config.GOOGLE_CLIENT_SECRET
    return bool(
        client_id
        and client_secret
        and not client_id.startswith("your-")
        and not client_secret.startswith("your-")
    )


def create_oauth_state() -> str:
    """Create a signed, time-limited JWT state token for CSRF defense."""
    now = datetime.now(timezone.utc)
    payload = {
        "csrf": secrets.token_urlsafe(32),
        "type": "google_oauth_state",
        "iat": now,
        "exp": now + timedelta(minutes=15),
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def verify_oauth_state(state: str) -> dict[str, Any]:
    """Verify that the OAuth state parameter is valid, unexpired, and untampered."""
    if not state or not state.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing OAuth state parameter (CSRF verification failed).",
        )
    try:
        payload = jwt.decode(
            state, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
        if payload.get("type") != "google_oauth_state":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OAuth state type.",
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth session expired. Please try signing in again.",
        )
    except (jwt.PyJWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or tampered OAuth state parameter (CSRF verification failed).",
        )


def get_google_auth_url(redirect_uri: str | None = None, state: str | None = None) -> tuple[str, str]:
    """Generate the Google OAuth 2.0 authorization URL and state token.
    
    Returns:
        (authorization_url, state_token)
    """
    client_id = config.GOOGLE_CLIENT_ID
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Sign-In is not configured on this server. Please set GOOGLE_CLIENT_ID in .env.",
        )

    target_redirect_uri = (redirect_uri or config.GOOGLE_REDIRECT_URI).strip()
    if not target_redirect_uri:
        target_redirect_uri = "http://localhost:8000/api/auth/google/callback"

    state_token = state or create_oauth_state()

    params = {
        "client_id": client_id,
        "redirect_uri": target_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state_token,
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = f"{GOOGLE_AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"
    return url, state_token


def exchange_code_for_tokens(
    code: str, redirect_uri: str | None = None, client: httpx.Client | None = None
) -> dict[str, Any]:
    """Exchange authorization code with Google token endpoint for access and ID tokens."""
    if not config.GOOGLE_CLIENT_ID or not config.GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth credentials are not fully configured in environment variables.",
        )

    target_redirect_uri = (redirect_uri or config.GOOGLE_REDIRECT_URI).strip()
    data = {
        "code": code,
        "client_id": config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": target_redirect_uri,
        "grant_type": "authorization_code",
    }

    headers = {"Accept": "application/json"}
    try:
        if client:
            resp = client.post(GOOGLE_TOKEN_ENDPOINT, data=data, headers=headers, timeout=15.0)
        else:
            with httpx.Client(timeout=15.0) as http_client:
                resp = http_client.post(GOOGLE_TOKEN_ENDPOINT, data=data, headers=headers)
        
        if resp.status_code != 200:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_desc = err_data.get("error_description") or err_data.get("error") or resp.text
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to exchange authorization code with Google: {err_desc}",
            )
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Network error connecting to Google OAuth servers: {exc}",
        )


def get_google_user_info(access_token: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Retrieve verified profile information from Google UserInfo endpoint."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    try:
        if client:
            resp = client.get(GOOGLE_USERINFO_ENDPOINT, headers=headers, timeout=15.0)
        else:
            with httpx.Client(timeout=15.0) as http_client:
                resp = http_client.get(GOOGLE_USERINFO_ENDPOINT, headers=headers)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to retrieve profile information from Google. Token may be expired or invalid.",
            )
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Network error contacting Google UserInfo endpoint: {exc}",
        )


def authenticate_or_link_google_user(user_info: dict[str, Any], db: Session) -> Farmer:
    """Validate Google user profile, link existing farmer account, or create a new farmer."""
    raw_email = user_info.get("email")
    if not raw_email or not str(raw_email).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google profile did not contain an email address.",
        )

    email = str(raw_email).lower().strip()
    email_verified = user_info.get("email_verified", False)
    # OpenID Connect standard: boolean true or string "true"
    if email_verified not in (True, "true", "True", 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your Google email address is not verified. Please verify your email with Google before signing in.",
        )

    google_id = str(user_info.get("sub", "")).strip() or None
    full_name = str(user_info.get("name", "")).strip() or email.split("@")[0]
    picture = user_info.get("picture")

    # Step A: Check if a farmer account exists with the exact verified email (case-insensitive)
    farmer = db.scalar(select(Farmer).where(func.lower(Farmer.email) == email))

    # Step B: If not found by email, check by google_id
    if not farmer and google_id:
        farmer = db.scalar(select(Farmer).where(Farmer.google_id == google_id))

    if farmer:
        # Secure Account Linking:
        # Link Google ID if not already bound
        if google_id and (not farmer.google_id or farmer.google_id != google_id):
            farmer.google_id = google_id
        if picture and not farmer.avatar_url:
            farmer.avatar_url = picture
        if farmer.auth_provider == "local":
            farmer.auth_provider = "google_linked"

        if not farmer.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Farmer account has been deactivated. Please contact support.",
            )
        db.commit()
        db.refresh(farmer)
        return farmer

    # Step C: Create brand-new farmer account with random hashed password
    random_password_seed = secrets.token_urlsafe(32)
    new_farmer = Farmer(
        full_name=full_name,
        email=email,
        hashed_password=hash_password(random_password_seed),
        is_active=True,
        google_id=google_id,
        avatar_url=picture,
        auth_provider="google",
    )
    db.add(new_farmer)
    db.commit()
    db.refresh(new_farmer)
    return new_farmer

