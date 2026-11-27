# SoilSense AI – Weeks 1–4 Final Audit & Completion Deliverable

**Audit Date:** 2026-11-27  
**Project Directory:** `D:\Infosys 7.0 Al-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory\soil-analytics`  
**Original CNN Checkpoint (`models/soil_classifier/best_model.pth`) SHA-256:** `1b11f7e08c386e85f863c307c8256d8c4998a8f63b39b1a62ae4e03d6e2ac1e5` (Unchanged)  
**PostgreSQL `soildb` Verified Counts:** `farmers=11`, `fields=19`, `soil_tests=12`, `soil_images=29`, `analysis_results=18`

---

## 1. Completed Features (Weeks 1–4)

### Weeks 1–2
- **Image Upload & Validation:** JPEG/PNG multipart + base64 uploads (`app/routers/images.py`), server-side Pillow verification, decompression-bomb guard, EXIF orientation transpose, RGB normalization, `224x224` resizing, blur/brightness warnings (`app/services/image_validation.py`).
- **Android Camera Capture:** Live `getUserMedia` (`facingMode: { ideal: 'environment' }`) and `<input type="file" capture="environment">` fallback (`app/static/index.html`).
- **OpenCV Preprocessing & Augmentation:** Partial gray-world color correction, CLAHE illumination normalization, Gaussian noise reduction, GrabCut background removal, and feature-preserving training augmentation (`app/services/preprocessing.py`, `ml/train_cnn.py`).
- **Structured Soil Input & ML-Ready Pairing:** NPK + units (`kg/ha`, `mg/kg`, `ppm`), pH, OM, moisture, testing method, sample depth, bulk density + unit, sample collection date, verification status, and `SoilImage.soil_test_id` / `AnalysisResult.soil_test_id` pairing (`app/models.py`, `app/schemas.py`, `scripts/migrate_add_ml_collection_fields.py`).
- **Dataset Provenance & PostgreSQL Storage:** `data/sources.csv`, `data/DATASET_MANIFEST.csv`, `GET /api/dataset/status`, farmer-owned relational schema.

### Weeks 3–4
- **6-Class Soil Type Classification (`EfficientNet-B0`):** `POST /api/images/{id}/predict`, `GET /api/predictions`, `GET /api/predictions/{id}` (preserved checkpoint).
- **Grad-CAM Explainability:** `SoilClassifier.generate_gradcam()`, `GET /api/images/{id}/gradcam`, side-by-side overlay in Prediction Detail Modal & Live Prediction UI.
- **Soil Texture Classification (`EfficientNet-B0`, Demonstration):** `models/soil_texture_classifier/best_model.pth`, `POST /api/images/{id}/predict-texture`.
- **Visible Plant Nutrient Deficiency Detection (`EfficientNet-B0`, Demonstration):** `models/nutrient_deficiency_detector/best_model.pth`, `POST /api/images/{id}/predict-deficiency`.
- **Structured Gradient-Boosted Nutrient Classifier + SHAP (`XGBClassifier` + `shap.TreeExplainer`, Experimental):** `models/nutrient_classifier/model.joblib`, `POST /api/ml/predict-nutrient-fertility`.
- **Soil Health Score & Heuristic Crop Advisory:** `_compute_health_score()`, `_make_advisory()`, `GET /api/soil-tests/{id}/advisory`.
- **Historical Soil Trend Analysis (Non-ML):** `app/services/soil_trends.py`, `GET /api/soil-tests/trends`, inline SVG sparklines in `#history` (`"Historical Soil Trend Analysis – Not an ML Forecast"`).
- **ML Soil Degradation Forecast (Experimental):** `models/degradation_forecaster/model.joblib`, `POST /api/ml/predict-degradation-risk`.
- **Hybrid Image + Tabular Fusion Architecture (Ready, Untrained):** `HybridSoilFusionNet` in `ml/train_fusion_model.py`, `GET /api/ml/hybrid-fusion/status`.
- **Central Model Registry & Capabilities:** `app/services/model_registry.py`, `GET /api/ml/registry`, `GET /api/cnn/capabilities`.

---

## 2. Fixed Bugs & Security Hardening
- **Frontend XSS & URL Safety (`app/static/index.html`):** Added `escHtml()`, `safeUrl()`, and `fmtNum()` across all `innerHTML` templates and image/avatar attributes.
- **Scientific Integrity & Honest Maturity Badges (`app/static/index.html`, `app/services/*`):** Removed fabricated lab method/provenance fallbacks; labelled Texture & Deficiency models `"Demonstration Model"`, Nutrient Fertility `"Experimental • Not for Dosage"`, Degradation Forecast `"Experimental ML Forecast • Not Reliable"`, and kept Historical Soil Trends separate.
- **IDOR Ownership Enforcement (`app/routers/fields.py`, `images.py`, `soil_tests.py`):** Closed fail-open `farmer_id IS NULL` bypasses in single-resource lookups; enforced same-field, same-farmer pairing validation.
- **JSONB Null/Malformed Coercion (`app/routers/images.py`):** Added `_normalize_prediction_result()` and `_safe_float()` to prevent `ValidationError` HTTP 500s on `None`, `bool`, `NaN`, or `inf` in stored `AnalysisResult.result`.
- **Google OAuth Root Cause & Hardening (`app/services/google_oauth.py`, `app/routers/auth.py`):** Identified port `8000` conflict (`FactoryFlow API`, PID `19824` occupying `127.0.0.1:8000` while SoilSense AI ran on `8001`), embedded `redirect_uri` in signed JWT `state`, and enforced `is_google_auth_configured()` on `/api/auth/google/url`.
- **Test DB Isolation (`tests/conftest.py`):** Isolated temporary SQLite DB and `UPLOAD_DIR` before app imports so tests never touch production PostgreSQL.

---

## 3. End-to-End & Automated Test Results
- **Full Pytest Suite (`.\venv\Scripts\python.exe -m pytest -q`):** Executed after final edits (see latest run output).
- **E2E Integration Verification:** Verified email/password login, Google OAuth callback + bridge, image upload, 6-class CNN, Grad-CAM, texture prediction, deficiency prediction, soil-test submission, `soil_test_id` pairing, advisory, historical trends, XGBoost + SHAP fertility, degradation forecast, CSV/print report hooks, `/health`, `/api/cnn/capabilities`, `/api/ml/registry`, and cross-farmer IDOR `404` / `401` checks (`E2E_ALL_FLOWS_PASSED`).

---

## 4. Model Performance Summary (Exact Measured Values)
- **6-Class Soil Type (`EfficientNet-B0`, preserved):** Accuracy `89.81%`, Macro F1 `88.99%` (`206` test samples).
- **Soil Texture (`EfficientNet-B0`, `Clay`/`Loam`/`Sand`):** Random patch-split `100.00%` (leaked); **Parent-grouped `StratifiedGroupKFold` (`0` group overlap, `1,000` test patches / `50` parent images): Accuracy `90.80%`, Macro F1 `0.9169`** (Demonstration).
- **Plant Nutrient Deficiency (`MPLD`, `-p`/`-p50`/`_c`):** Accuracy `87.22%`, Macro F1 `0.8723` (`360` test images; Demonstration).
- **Nutrient Fertility (`Rice NPK`, 4 classes):** Accuracy `26.25%`, Macro F1 `0.2082`, ROC-AUC `0.4831` (`DummyMostFreq`: `34.75%`; Experimental — uninformative labels, not for dosage).
- **Degradation Forecaster (`SoilHealthDB V2`, chronological `>2009`):** Classification Accuracy `67.88%`, Macro F1 `0.4532`, ROC-AUC `0.4662`; Regression MAE `0.1461`, R² `-0.0008` (`DummyCls`: `87.96%`; Experimental — not reliable for field forecasting).

---

## 5. Remaining Blockers & Tasks Requiring Real-World Data / Human Approval
1. **Hybrid Fusion Training:** Blocked by `0` genuine paired `(image, soil_test)` records with shared ground-truth labels in `soildb`. Requires collecting `>= 50–300+` real co-located paired records via the UI/API pairing workflow.
2. **Production Nutrient & Degradation ML:** Requires acquiring multi-feature Soil Health Card datasets (`N, P, K, pH, EC, OC, crop`) and multi-season single-plot longitudinal records.
3. **Checkpoint Replacement Approval:** Replacing `models/soil_texture_classifier/best_model.pth` with the parent-grouped `90.80%` checkpoint requires explicit approval (and backup of the existing checkpoint).
4. **Port 8000 Conflict:** Stop the external `FactoryFlow API` process (PID `19824`) on `127.0.0.1:8000` before starting SoilSense AI on port `8000` for Google OAuth callback (`http://localhost:8000/api/auth/google/callback`).
5. **Git Commit / Push:** All changes remain uncommitted pending your approval.

---

## 6. Exact Commands to Run the Project & Tests
```powershell
# 1. Start PostgreSQL (if not running)
docker compose up -d db

# 2. Start SoilSense AI FastAPI server on port 8000 (ensure port 8000 is free)
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 3. Run the complete test suite
.\venv\Scripts\python.exe -m pytest -q
```

