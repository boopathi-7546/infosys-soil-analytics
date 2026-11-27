# SoilSense AI – Weeks 1–4 Master Project Audit

**Audit Date:** 2026-11-27  
**Baseline:** `97 passed, 1 warning` (`.\venv\Scripts\python.exe -m pytest -q`)  
**Existing CNN Checkpoint:** `models/soil_classifier/best_model.pth`  
**Existing CNN SHA-256:** `1b11f7e08c386e85f863c307c8256d8c4998a8f63b39b1a62ae4e03d6e2ac1e5`  
**PostgreSQL `soildb` Baseline:** `farmers=11`, `fields=19`, `soil_tests=11`, `soil_images=28`, `analysis_results=17`

---

## 1. Repository & Architecture Inventory

| Directory | Contents & Role |
| :--- | :--- |
| `app/` | FastAPI application (`main.py`, `config.py`, `database.py`, `models.py`, `schemas.py`), routers (`auth.py`, `fields.py`, `images.py`, `soil_tests.py`, `dataset.py`, `cnn.py`), services (`auth.py`, `google_oauth.py`, `image_validation.py`, `preprocessing.py`, `storage.py`, `soil_classifier.py`, `soil_trends.py`), and SPA frontend (`static/index.html`). |
| `ml/` | CNN training pipeline (`train_cnn.py`) for 6-class EfficientNet-B0 soil-type classification. |
| `data/` | `raw/` (`1,361` images across 6 soil-type classes), `processed/` (`3,784` train, `203` val, `206` test), `sources.csv`. |
| `models/soil_classifier/` | `best_model.pth`, `class_names.json`, `metrics.json`, `training_history.json`, `confusion_matrix.png`, `training_curves.png`. |
| `scripts/` | `migrate_add_farmer_auth.py`, `migrate_add_ml_collection_fields.py`, `fetch_sandy_soil.py`. |
| `tests/` | `test_api.py`, `test_auth.py`, `test_google_auth.py`, `test_cnn_router.py`, `test_cnn_predictions.py`, `test_ml_cnn.py`, `test_soil_history_isolation.py`, `test_soil_trends.py`, `test_ml_data_collection.py`, `test_weeks_1_2_audit.py`, `conftest.py`. |
| `docs/` | `android_testing_checklist.md`, `BLOCKED_ML_REQUIREMENTS.md`, `WEEKS_3_4_DATA_READINESS_AUDIT.md`. |

### Installed ML Packages (`venv`)
- `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, `cv2 5.0.0`, `numpy 2.4.6`, `PIL 13.0.0`, `scikit-learn 1.9.1`.
- **Missing (needed for Phases E–F):** `xgboost`, `shap` (will be installed only when training structured ML + SHAP).

---

## 2. Weeks 1–2 Status (Complete)

| Requirement | Status | Exact Files / Functions |
| :--- | :--- | :--- |
| JPEG & PNG upload | **Complete** | `app/routers/images.py`: `upload_image_for_field`, `upload_image_flat`; `app/static/index.html`: `#fileInput` |
| Android camera capture | **Complete** | `app/static/index.html`: `getUserMedia` (`facingMode: 'environment'`) & `#mobileCameraInput` (`capture="environment"`); `app/routers/images.py`: `upload_image_base64` |
| Server-side image validation | **Complete** | `app/services/image_validation.py`: `validate_and_normalize` |
| Image resizing & format normalization | **Complete** | `app/services/image_validation.py`: `validate_and_normalize`; `app/services/preprocessing.py`: `preprocess` |
| Structured NPK, pH, OM, moisture + ML metadata | **Complete** | `app/models.py`: `SoilTest`; `app/schemas.py`: `SoilTestIn`, `SoilTestOut`; `app/routers/soil_tests.py`: `create_soil_test` |
| OpenCV color correction & background removal | **Complete** | `app/services/preprocessing.py`: `gray_world`, `normalize_illumination`, `reduce_noise`, `remove_background` |
| Feature-preserving augmentation | **Complete** | `app/services/preprocessing.py`: `augment`; `ml/train_cnn.py`: `get_transforms` |
| Dataset provenance | **Complete** | `data/sources.csv`; `app/routers/dataset.py`: `get_dataset_status` |
| PostgreSQL storage & pairing | **Complete** | `app/models.py`: `SoilTest`, `SoilImage.soil_test_id`, `AnalysisResult.soil_test_id`; `scripts/migrate_add_ml_collection_fields.py` |

---

## 3. Weeks 3–4 Status

### Completed Weeks 3–4 Requirements
- **EfficientNet-B0 6-Class Soil Type Classifier:** `app/services/soil_classifier.py` (`SoilClassifier.predict`), `89.81%` test accuracy, `88.99%` macro F1 (`models/soil_classifier/metrics.json`).
- **Grad-CAM Explainability:** `SoilClassifier.generate_gradcam()`, `GET /api/images/{image_id}/gradcam`, UI overlays in `app/static/index.html`.
- **Soil Health Score & Heuristic Advisory:** `app/routers/soil_tests.py` (`_compute_health_score`, `_make_advisory`).
- **Historical Soil Trend Analysis:** `app/services/soil_trends.py` (`compute_historical_soil_trends`), `GET /api/soil-tests/trends`, UI chart in `app/static/index.html`.
- **Capabilities Endpoint:** `app/routers/cnn.py` (`get_ml_capabilities`, `GET /api/cnn/capabilities`).

### Missing / Blocked Weeks 3–4 Requirements
| Requirement | Current Status | Data / Implementation Blocker |
| :--- | :--- | :--- |
| **1. Dataset Acquisition Engine** | **Missing** | No `scripts/download_datasets.py`, `scripts/verify_datasets.py`, or `data/DATASET_MANIFEST.csv`. |
| **2. Soil Texture Classification** | **Missing / Data Blocked** | Local `data/` has 6 soil-type classes only; zero USDA/FAO texture labels. |
| **3. Visible Nutrient Deficiency Detection** | **Missing / Data Blocked** | Local `data/` has zero plant/leaf nutrient deficiency images/labels. |
| **4. Gradient-Boosted Nutrient ML (XGBoost)** | **Missing / Data Blocked** | `soildb.soil_tests` has only 11 unlabelled rows; `xgboost` not installed. |
| **5. SHAP Explainability** | **Missing / Data Blocked** | Depends on trained structured ML model; `shap` not installed. |
| **6. Hybrid Image + Structured Fusion** | **Missing / Data Blocked** | `0` paired `(image, soil_test)` records in `soildb`; no fusion model/pipeline. |
| **7. ML Soil Degradation Forecasting** | **Missing / Data Blocked** | `soildb.soil_tests` spans only 13 days across 3 fields; no longitudinal multi-year dataset. |
| **8. Model Registry** | **Missing** | Model paths/versions are currently hardcoded in individual services/routers. |

