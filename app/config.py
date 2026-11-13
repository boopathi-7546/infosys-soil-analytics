import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL is the default; set DATABASE_URL in .env to override
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://soil:soil@localhost:5432/soildb"
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}   # checked against the real file content, not the filename
MIN_SIDE_PX = 100                   # reject images smaller than this
MAX_SIDE_PX = 1600                  # larger images are downscaled
MAX_PIXELS = 50_000_000             # decompression-bomb guard
BLUR_THRESHOLD = 60.0               # Laplacian variance below this -> "blurry" warning
JPEG_QUALITY = 90
MODEL_INPUT_SIZE = 224              # CNN input size used by preprocessing
