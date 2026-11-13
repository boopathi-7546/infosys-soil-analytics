from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from . import config
from .database import Base, engine
from .routers import dataset, fields, images, soil_tests

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="AI-Powered Soil Analytics System",
    description="Nutrient Assessment, OpenCV Preprocessing, and Intelligent Crop Advisory.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for mobile and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
if config.UPLOAD_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR), html=False), name="uploads")

# Include routers without prefix (for backwards compatibility with existing clients/tests)
app.include_router(fields.router)
app.include_router(images.router)
app.include_router(soil_tests.router)
app.include_router(dataset.router)

# Include routers with /api prefix (standard REST endpoints)
app.include_router(fields.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(soil_tests.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")


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
