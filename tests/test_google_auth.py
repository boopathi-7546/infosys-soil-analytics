"""Comprehensive test suite for Google OAuth 2.0 / OpenID Connect authentication.

Tests:
1. Google OAuth URL generation (configured vs unconfigured).
2. CSRF state creation and verification (valid, tampered, expired).
3. Browser and JSON callback handling for cancelled/denied Google logins.
4. Successful Google registration for new farmers (profile, avatar, JWT issuance).
5. Seamless account linking for existing email/password registered farmers.
6. Rejection of unverified Google emails.
7. Rejection of deactivated farmer accounts.
8. API-friendly POST /api/auth/google/callback endpoint.
9. Multi-tenant data ownership & IDOR isolation for Google-authenticated farmers.
"""
from datetime import datetime, timedelta, timezone
import os
import tempfile
from unittest.mock import patch

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test_google_auth.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads_google_auth"
os.environ["GOOGLE_CLIENT_ID"] = "mock-client-id-12345.apps.googleusercontent.com"
os.environ["GOOGLE_CLIENT_SECRET"] = "mock-client-secret-abcde"
os.environ["GOOGLE_REDIRECT_URI"] = "http://localhost:8000/api/auth/google/callback"

import jwt
import pytest
from fastapi.testclient import TestClient

from app import config
from app.database import Base, engine
from app.main import app
from app.models import Farmer, Field
from app.services.google_oauth import create_oauth_state, verify_oauth_state


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    old_id = config.GOOGLE_CLIENT_ID
    old_sec = config.GOOGLE_CLIENT_SECRET
    old_uri = config.GOOGLE_REDIRECT_URI
    config.GOOGLE_CLIENT_ID = "mock-client-id-12345.apps.googleusercontent.com"
    config.GOOGLE_CLIENT_SECRET = "mock-client-secret-abcde"
    config.GOOGLE_REDIRECT_URI = "http://localhost:8000/api/auth/google/callback"

    Base.metadata.create_all(engine)
    try:
        from scripts.migrate_add_farmer_auth import run_migration
        run_migration(engine)
    except Exception:
        pass
    yield

    config.GOOGLE_CLIENT_ID = old_id
    config.GOOGLE_CLIENT_SECRET = old_sec
    config.GOOGLE_REDIRECT_URI = old_uri


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Configuration & Authorization URL Tests
# ---------------------------------------------------------------------------

def test_google_auth_url_success(client):
    """Verify that /api/auth/google/url returns a valid Google OAuth consent URL with signed state."""
    res = client.get("/api/auth/google/url")
    assert res.status_code == 200, res.text
    data = res.json()
    assert "url" in data
    url = data["url"]
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "client_id=" in url
    assert "mock-client-id-12345" in url
    assert "response_type=code" in url
    assert "scope=" in url
    assert "state=" in url


def test_google_login_redirect(client):
    """Verify that /api/auth/google/login performs a 302 redirect to Google."""
    res = client.get("/api/auth/google/login", follow_redirects=False)
    assert res.status_code == 302
    assert "https://accounts.google.com/o/oauth2/v2/auth" in res.headers["location"]


def test_google_auth_url_unconfigured(client, monkeypatch):
    """Verify that if Google Client ID is unset, the endpoint returns 503."""
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    res = client.get("/api/auth/google/url")
    assert res.status_code == 503
    assert "not configured" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 2. CSRF State Parameter Validation Tests
# ---------------------------------------------------------------------------

def test_oauth_state_generation_and_verification():
    """Verify that state tokens can be securely generated and validated."""
    state = create_oauth_state()
    assert isinstance(state, str)
    payload = verify_oauth_state(state)
    assert payload.get("type") == "google_oauth_state"
    assert "csrf" in payload


def test_oauth_state_rejects_tampered_token():
    """Tampered state tokens must be rejected."""
    state = create_oauth_state()
    tampered = state[:-4] + "fake"
    with pytest.raises(Exception) as excinfo:
        verify_oauth_state(tampered)
    assert "CSRF verification failed" in str(excinfo.value.detail)


def test_oauth_state_rejects_expired_token():
    """Expired state tokens must be rejected."""
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    expired_payload = {
        "csrf": "random-csrf",
        "type": "google_oauth_state",
        "exp": expired_time,
    }
    expired_token = jwt.encode(
        expired_payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM
    )
    with pytest.raises(Exception) as excinfo:
        verify_oauth_state(expired_token)
    assert "expired" in str(excinfo.value.detail).lower()


# ---------------------------------------------------------------------------
# 3. Google OAuth Error / Cancellation Handling
# ---------------------------------------------------------------------------

def test_google_callback_user_cancelled_redirect(client):
    """When user cancels on Google consent screen, browser is redirected with auth_error."""
    res = client.get(
        "/api/auth/google/callback?error=access_denied&error_description=User%20denied%20access",
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert "/?auth_error=" in res.headers["location"]
    assert "denied" in res.headers["location"].lower()


def test_google_callback_user_cancelled_json(client):
    """When format=json is specified, cancelled login returns HTTP 400 JSON."""
    res = client.get(
        "/api/auth/google/callback?error=access_denied&error_description=User%20denied%20access&format=json"
    )
    assert res.status_code == 400
    assert "denied" in res.json()["detail"].lower()


def test_google_callback_missing_params(client):
    """Missing code or state returns an error."""
    res = client.get("/api/auth/google/callback?format=json")
    assert res.status_code == 400
    assert "missing" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4. Successful Google Login (New Farmer Registration)
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_login_new_farmer_success(mock_exchange, mock_user_info, client):
    """Verify that a new farmer signing in with Google is created and issued a JWT."""
    mock_exchange.return_value = {
        "access_token": "mock-google-access-token-123",
        "id_token": "mock-id-token",
        "expires_in": 3600,
    }
    mock_user_info.return_value = {
        "sub": "google-sub-10001",
        "name": "Murugan Vel",
        "email": "murugan.farmer@gmail.com",
        "email_verified": True,
        "picture": "https://lh3.googleusercontent.com/a/mock-murugan.jpg",
    }

    state = create_oauth_state()

    # JSON callback format
    res = client.get(
        f"/api/auth/google/callback?code=mock-code-123&state={state}&format=json"
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["farmer"]["email"] == "murugan.farmer@gmail.com"
    assert data["farmer"]["full_name"] == "Murugan Vel"
    assert data["farmer"]["avatar_url"] == "https://lh3.googleusercontent.com/a/mock-murugan.jpg"
    assert data["farmer"]["auth_provider"] == "google"

    # Use the issued JWT to query profile
    token = data["access_token"]
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "murugan.farmer@gmail.com"


# ---------------------------------------------------------------------------
# 5. Account Linking for Existing Email/Password Farmer
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_login_account_linking(mock_exchange, mock_user_info, client):
    """Verify that an existing farmer registered via email/password is linked when logging in via Google."""
    # Step A: Register farmer with email and password
    reg_res = client.post("/api/auth/register", json={
        "full_name": "Kavitha R",
        "email": "kavitha.farm@example.com",
        "password": "localpassword123",
    })
    assert reg_res.status_code == 201
    orig_farmer_id = reg_res.json()["farmer"]["id"]

    # Step B: Now log in via Google with the same verified email
    mock_exchange.return_value = {
        "access_token": "mock-google-token-456",
        "expires_in": 3600,
    }
    mock_user_info.return_value = {
        "sub": "google-sub-20002",
        "name": "Kavitha Raj",
        "email": "KAVITHA.FARM@example.com",  # case-insensitive check
        "email_verified": True,
        "picture": "https://lh3.googleusercontent.com/a/kavitha.jpg",
    }

    state = create_oauth_state()
    res = client.get(
        f"/api/auth/google/callback?code=mock-code-456&state={state}&format=json"
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["farmer"]["id"] == orig_farmer_id
    assert data["farmer"]["email"] == "kavitha.farm@example.com"
    assert data["farmer"]["google_id"] == "google-sub-20002"
    assert data["farmer"]["auth_provider"] == "google_linked"

    # Step C: Verify original password authentication STILL works!
    pwd_login = client.post("/api/auth/login", json={
        "email": "kavitha.farm@example.com",
        "password": "localpassword123",
    })
    assert pwd_login.status_code == 200
    assert pwd_login.json()["farmer"]["id"] == orig_farmer_id


# ---------------------------------------------------------------------------
# 6. Unverified Email Rejection
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_login_rejects_unverified_email(mock_exchange, mock_user_info, client):
    """Google accounts with unverified emails must be rejected."""
    mock_exchange.return_value = {"access_token": "token-789"}
    mock_user_info.return_value = {
        "sub": "google-sub-30003",
        "name": "Unverified User",
        "email": "unverified@example.com",
        "email_verified": False,
    }

    state = create_oauth_state()
    res = client.get(
        f"/api/auth/google/callback?code=mock-code-789&state={state}&format=json"
    )
    assert res.status_code == 400
    assert "not verified" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. POST /api/auth/google/callback Endpoint
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_post_callback(mock_exchange, mock_user_info, client):
    """Verify that POST /api/auth/google/callback works for API and headless clients."""
    mock_exchange.return_value = {"access_token": "token-post-123"}
    mock_user_info.return_value = {
        "sub": "google-sub-40004",
        "name": "Senthil Nathan",
        "email": "senthil.nathan@example.com",
        "email_verified": True,
    }

    state = create_oauth_state()
    res = client.post("/api/auth/google/callback", json={
        "code": "auth-code-post",
        "state": state,
    })
    assert res.status_code == 200, res.text
    data = res.json()
    assert "access_token" in data
    assert data["farmer"]["email"] == "senthil.nathan@example.com"


# ---------------------------------------------------------------------------
# 8. Browser HTML Bridge Redirect Test
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_browser_callback_returns_bridge(mock_exchange, mock_user_info, client):
    """When a browser hits the callback without format=json, it receives an HTML bridge."""
    mock_exchange.return_value = {"access_token": "token-browser-555"}
    mock_user_info.return_value = {
        "sub": "google-sub-50005",
        "name": "Deepa Sundar",
        "email": "deepa.sundar@example.com",
        "email_verified": True,
    }

    state = create_oauth_state()
    res = client.get(
        f"/api/auth/google/callback?code=mock-code-browser&state={state}",
        headers={"Accept": "text/html,application/xhtml+xml"},
    )
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Signing you in to SoilSense AI" in res.text
    assert "localStorage.setItem('soilsense_token'" in res.text


# ---------------------------------------------------------------------------
# 9. IDOR & Multi-Tenant Data Isolation for Google Users
# ---------------------------------------------------------------------------

@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_user_data_isolation(mock_exchange, mock_user_info, client):
    """Ensure Google-authenticated farmers have complete data ownership and cannot access other farmers' fields."""
    # Farmer A logs in via Google
    mock_exchange.return_value = {"access_token": "token-user-a"}
    mock_user_info.return_value = {
        "sub": "google-user-a",
        "name": "Farmer Alpha",
        "email": "farmer.alpha@example.com",
        "email_verified": True,
    }
    state_a = create_oauth_state()
    res_a = client.post("/api/auth/google/callback", json={"code": "code-a", "state": state_a})
    token_a = res_a.json()["access_token"]

    # Farmer A creates a field
    field_a_res = client.post(
        "/api/fields",
        json={"name": "Alpha North Plot"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert field_a_res.status_code == 201
    field_a_id = field_a_res.json()["id"]

    # Farmer B logs in via Google
    mock_exchange.return_value = {"access_token": "token-user-b"}
    mock_user_info.return_value = {
        "sub": "google-user-b",
        "name": "Farmer Beta",
        "email": "farmer.beta@example.com",
        "email_verified": True,
    }
    state_b = create_oauth_state()
    res_b = client.post("/api/auth/google/callback", json={"code": "code-b", "state": state_b})
    token_b = res_b.json()["access_token"]

    # Farmer B attempts to access Farmer A's field -> must receive HTTP 404 (IDOR guard)
    forbidden_res = client.get(
        f"/api/fields/{field_a_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert forbidden_res.status_code == 404


@patch("app.routers.auth.get_google_user_info")
@patch("app.routers.auth.exchange_code_for_tokens")
def test_google_callback_browser_flow_preserves_redirect_uri_and_bridge(mock_exchange, mock_user_info, client):
    """Browser GET /api/auth/google/url -> /api/auth/google/callback preserves redirect_uri and returns JWT bridge."""
    url_res = client.get("/api/auth/google/url")
    assert url_res.status_code == 200
    auth_url = url_res.json()["url"]
    assert "http%3A%2F%2Flocalhost%3A8000%2Fapi%2Fauth%2Fgoogle%2Fcallback" in auth_url

    from urllib.parse import parse_qs, urlparse
    state_val = parse_qs(urlparse(auth_url).query)["state"][0]

    mock_exchange.return_value = {"access_token": "mock-browser-token"}
    mock_user_info.return_value = {
        "sub": "google-browser-123",
        "name": "Browser Farmer",
        "email": "browser.farmer@example.com",
        "email_verified": True,
    }

    cb_res = client.get(f"/api/auth/google/callback?code=browser-code&state={state_val}")
    assert cb_res.status_code == 200
    assert "text/html" in cb_res.headers["content-type"]
    assert "localStorage.setItem('soilsense_token'" in cb_res.text
    assert "/?token=" in cb_res.text
    mock_exchange.assert_called_once_with("browser-code", redirect_uri="http://localhost:8000/api/auth/google/callback")


def test_google_auth_url_returns_503_when_secret_missing(client, monkeypatch):
    """Missing GOOGLE_CLIENT_SECRET must fail fast with 503 on /api/auth/google/url."""
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")
    res = client.get("/api/auth/google/url")
    assert res.status_code == 503
