from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import text

from . import config
from .database import Base, engine
from .routers import fields, images, soil_tests

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)  # swap for Alembic migrations later if needed
    yield


app = FastAPI(
    title="AI-Powered Soil Analytics System",
    description="Milestone 1: image ingestion, soil test data, preprocessing and PostgreSQL storage.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(fields.router)
app.include_router(images.router)
app.include_router(soil_tests.router)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}
