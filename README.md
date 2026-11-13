# AI-Powered Soil Analytics System: Milestone 1 backend

FastAPI + PostgreSQL backend for soil photo and soil test ingestion, with an OpenCV preprocessing
pipeline and dataset tooling.

| Task | Where |
|---|---|
| Soil image upload API + server-side validation | `app/routers/images.py`, `app/services/image_validation.py` |
| Soil test data API (N, P, K, pH, organic matter, moisture) | `app/routers/soil_tests.py`, `app/schemas.py` |
| OpenCV preprocessing + augmentation | `app/services/preprocessing.py` |
| Dataset collection and preparation | `DATASET.md`, `scripts/prepare_dataset.py` |
| PostgreSQL integration (soil records and images) | `app/models.py`, `app/database.py` |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

docker compose up -d db            # PostgreSQL 16 (or use your own server)
cp .env.example .env               # Windows: copy .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for interactive Swagger docs. Tables are created on startup.

No Docker? Set `DATABASE_URL=sqlite:///./soil.db` to try it on SQLite, or install PostgreSQL and
create a `soil` user and database to match `.env.example`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/fields` | create a field (name, farmer_name, location) |
| GET | `/fields`, `/fields/{id}` | list / read fields |
| POST | `/fields/{id}/images` | upload a JPEG/PNG (multipart field `file`) |
| GET | `/fields/{id}/images` | image metadata for a field |
| GET | `/images/{id}/file` | the stored (normalised) image |
| GET | `/images/{id}/preprocessed` | the image after the OpenCV pipeline (`?size=224&background_removal=true`) |
| POST | `/fields/{id}/soil-tests` | store N, P, K (kg/ha), pH, organic matter (%), moisture (%) |
| GET | `/fields/{id}/soil-tests` | history, newest first (used for trend analysis later) |
| GET | `/soil-tests/{id}` | one soil test |
| GET | `/health` | liveness + DB connectivity |

```bash
curl -X POST localhost:8000/fields -H "Content-Type: application/json" \
     -d '{"name":"North plot","farmer_name":"Ravi","location":"Erode"}'
curl -X POST localhost:8000/fields/1/images -F "file=@soil.jpg;type=image/jpeg"
curl -X POST localhost:8000/fields/1/soil-tests -H "Content-Type: application/json" \
     -d '{"nitrogen":250,"phosphorus":18,"potassium":190,"ph":6.8,"organic_matter":0.6,"moisture":22}'
```

## Design decisions (good to know for your review)

- **Validation never trusts the filename or Content-Type.** The file is decoded with Pillow, so renamed
  text files, truncated files and decompression bombs are rejected (422/413). Limits: 10 MB, minimum 100 px
  per side, JPEG/PNG only.
- **Normalisation:** every upload is saved as an RGB JPEG, EXIF-rotated, downscaled to at most 1600 px, with
  EXIF/GPS metadata stripped (farmer privacy). Blurry, dark or overexposed photos are accepted but
  return `warnings`, so the PWA can ask the farmer to retake.
- **Images live on disk, metadata in PostgreSQL.** `app/services/storage.py` is the only file to
  change when you move to S3/GCS in Milestone 3. A failed DB insert deletes the saved file.
- **Colour correction is deliberately gentle.** Full gray-world balancing turned red soil grey in testing,
  which would remove the main signal for soil type. The pipeline uses a clamped partial correction and
  CLAHE on the lightness channel only.
- **Augmentation uses geometry and mild lighting only (no hue shifts)**, and the split happens before
  augmentation so copies of one photo never end up in both train and test.
- **`analysis_results` is created but unused.** The Milestone 2 model output will go there (JSONB).

## Tests

```bash
pytest -q
```
21 tests cover upload (JPEG/PNG/alpha, downscaling, fake/empty/tiny/oversized/wrong-type files),
soil-test validation ranges, 404 handling and the preprocessed endpoint. They run on SQLite by default; I also ran the
same suite against PostgreSQL 16 and all passed.

## Not done yet / yours to do

- **A real dataset.** See `DATASET.md`. The included scripts only produce synthetic test images.
- Authentication and farmer accounts (not in the Milestone 1 scope).
- The model training, recommendation engine and PWA (Milestones 2 and 3).

## Make the before/after figure for your report
```bash
python scripts/preprocess_demo.py path/to/real_soil_photo.jpg docs/preprocessing_demo.jpg
```
