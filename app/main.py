from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from . import config
from .database import Base, engine
from .routers import auth, cnn, dataset, fields, images, soil_tests

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Safe schema creation & migration
    Base.metadata.create_all(engine)
    try:
        from scripts.migrate_add_farmer_auth import run_migration
        run_migration(engine)
    except Exception as e:
        print(f"[STARTUP] Warning during schema migration check: {e}")
    yield


app = FastAPI(
    title="SoilSense AI — Soil Analytics & Crop Advisory System",
    description="Secure Farmer Authentication, Soil Nutrient Assessment, and Intelligent Crop Advisory.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS configuration: avoids unrestricted credentialed access with wildcards
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: Unrestricted public static mount of /uploads has been removed for privacy & security.
# Images are securely streamed via the authenticated /api/images/{id}/file endpoint.

# Include routers without prefix (backward compatibility)
app.include_router(auth.router)
app.include_router(fields.router)
app.include_router(images.router)
app.include_router(soil_tests.router)
app.include_router(dataset.router)
app.include_router(cnn.router)

# Include routers with /api prefix (standard REST convention)
app.include_router(auth.router, prefix="/api")
app.include_router(fields.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(soil_tests.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(cnn.router, prefix="/api")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    """Liveness check and database connectivity verification."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"
    return {"status": "ok", "db": db_status}
