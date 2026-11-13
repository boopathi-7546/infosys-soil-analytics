# AI-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-31%20Passed-2EA44F.svg?style=flat&logo=pytest&logoColor=white)](tests/)

An end-to-end enterprise agricultural analytics system combining **OpenCV image preprocessing (224×224 CNN input)**, **laboratory soil test ingestion (N, P, K, pH, Organic Matter, Moisture)**, **PostgreSQL 16 persistence**, and an **agronomic rule-based crop advisory engine** presented through a farmer-friendly, multilingual dashboard.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Farmer Client Interface"]
        A1["📱 Android Chrome Live Camera\n(getUserMedia rear lens)"]
        A2["📁 Gallery / Desktop Upload\n(JPEG / JPG / PNG)"]
        A3["🧪 Lab Soil Test Form\n(N, P, K, pH, OM, Moisture)"]
    end

    subgraph Backend["FastAPI Backend (app/)"]
        B1["/api/images/upload-base64\n& /fields/{id}/images"]
        B2["Image Validation Engine\nPillow verify & EXIF sanitization"]
        B3["OpenCV Preprocessing\nGray-World + CLAHE + 224x224 Resize"]
        B4["/api/soil-tests\nValidation & Health Scoring"]
        B5["Agronomic Advisory Engine\nCrop Compatibility & Soil Remedies"]
        B6["/api/dataset/status\nBenchmark Distribution"]
    end

    subgraph Storage["Storage & Persistence"]
        DB[("🐘 PostgreSQL 16\n(soildb)")]
        Disk[("💾 uploads/ & data/processed/\n(Preprocessed artifacts)")]
    end

    A1 -->|Base64 JSON| B1
    A2 -->|Multipart/Form-Data| B1
    A3 -->|JSON Payload| B4

    B1 --> B2 --> B3 --> Disk
    B2 -->|Metadata & Blur Score| DB
    B4 -->|Nutrient Records| DB
    B4 --> B5
    B6 --> DB
```

---

## 📦 Key Upgraded Modules

| Module | Feature | Implementation Details |
|---|---|---|
| **Module 1** | Separate Upload & Camera | Option A: Gallery picker (file preview, remove/replace). Option B: Live camera stream via WebRTC `getUserMedia({ facingMode: { ideal: "environment" } })` with Capture, Retake, and Use Photo actions. |
| **Module 2** | Soil Test Data API | Validated ranges for Nitrogen (0–500 kg/ha), Phosphorus (0–150 kg/ha), Potassium (0–800 kg/ha), pH (0–14), Organic Matter (0–100%), and Moisture (0–100%). Rule-based advisory engine. |
| **Module 3** | PostgreSQL Migration | Primary production database migrated to PostgreSQL 16 on Docker Compose with SQLAlchemy 2.0 and `psycopg2-binary`. Safe migration script `scripts/migrate_sqlite_to_pg.py` preserves SQLite records. |
| **Module 4** | OpenCV Preprocessing | Gentle partial Gray-World color-cast correction, LAB lightness channel CLAHE, Gaussian noise reduction, GrabCut background extraction, and standardized 224×224 resize for CNN ingestion. |
| **Module 5** | Real Soil Dataset | Organized raw directory schema (`black_soil`, `red_soil`, `alluvial_soil`, `laterite_soil`, `sandy_soil`, `clay_soil`), `data/sources.csv` attribution catalog, and `/api/dataset/status` endpoint. |
| **Module 6** | Redesigned Dashboard | Modern agricultural tech aesthetic (warm cream `#F8F6F0`, forest green `#1B4332`, leaf green `#40916C`), 7 modular sections, KPI cards, interactive SVG charts, and printable PDF reports. |
| **Module 7** | Multilingual Architecture | Full interface translations for **English**, **தமிழ் (Tamil)**, **हिंदी (Hindi)**, and **ಕನ್ನಡ (Kannada)** with accurate preservation of scientific units (`kg/ha`, `pH`, `%`). |
| **Module 8** | Mobile & Android Testing | Mobile-first bottom navigation, touch targets ≥ 44px, rear camera preference, and complete `docs/android_testing_checklist.md`. |
| **Module 9** | Test Suite & Quality | 31 comprehensive pytest test cases covering all image formats, oversized payloads, malformed data, advisory formulas, and API endpoints with 100% pass rate. |

---

## 🚀 Quick Start Guide

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/boopathi-7546/infosys-soil-analytics.git
cd infosys-soil-analytics

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows PowerShell
# source venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch PostgreSQL with Docker Compose

```bash
# Start PostgreSQL 16 container
docker compose up -d db

# Verify container is healthy
docker compose ps
```

### 3. Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env             # Linux / macOS
# copy .env.example .env         # Windows CMD
```

Ensure `.env` contains:
```ini
DATABASE_URL=postgresql+psycopg2://soil:soil@localhost:5432/soildb
UPLOAD_DIR=uploads
MAX_UPLOAD_MB=10
```

### 4. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Dashboard**: http://localhost:8000/
- **Interactive Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🗄️ Database Setup & Migration

### Schema (4 Core Entities):
1. **`fields`**: Farm plot name, farmer name, location, and creation timestamp.
2. **`soil_tests`**: Laboratory N, P, K, pH, organic matter, moisture, notes, and test date.
3. **`soil_images`**: File path, original format, byte size, width, height, blur score (Laplacian variance), and upload date.
4. **`analysis_results`**: Relational junction table linking fields, images, and laboratory tests with model prediction payloads (JSONB).

### Safe SQLite to PostgreSQL Migration:
If you have existing historical data in `soil.db`, migrate it safely without data loss:
```bash
python scripts/migrate_sqlite_to_pg.py
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application liveness and database connectivity check |
| `POST` | `/api/fields` | Register a new farm field |
| `GET` | `/api/fields` | List all registered farm fields |
| `GET` | `/api/fields/{id}/stats` | Retrieve aggregate statistics for a farm field |
| `DELETE`| `/api/fields/{id}` | Delete a farm field and its associated records |
| `POST` | `/api/fields/{id}/images` | Multipart upload for soil photographs |
| `POST` | `/api/images/upload-base64`| Base64 image upload from live camera capture |
| `GET` | `/api/images/{id}/file` | Serve normalized RGB JPEG image file |
| `GET` | `/api/images/{id}/preprocessed`| Stream OpenCV preprocessed 224×224 image bytes |
| `GET` | `/api/images/{id}/preprocess` | Return preprocessing metrics (mean RGB, quality ok, blur variance) |
| `POST` | `/api/fields/{id}/soil-tests` | Save laboratory soil test measurements |
| `GET` | `/api/fields/{id}/soil-tests` | Retrieve historical soil tests for a field |
| `GET` | `/api/soil-tests/{id}/advisory` | Generate rule-based crop recommendations and soil remedies |
| `GET` | `/api/dataset/status` | Current count of real-world dataset samples per soil class |

---

## 🧪 Testing Suite

Execute the automated test suite with verbose output:

```bash
pytest -v
```

**Results:**
```text
======================== 31 passed in 2.04s ========================
```

---

## 📱 Android Device Verification

For complete mobile camera testing procedures on a physical Android smartphone, refer to [docs/android_testing_checklist.md](docs/android_testing_checklist.md).
