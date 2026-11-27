# SoilSense AI – Weeks 1–4 Final Completion Report

**Completion Date:** 2026-11-27  
**Project Root:** `D:\Infosys 7.0 Al-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory\soil-analytics`  
**Original CNN Checkpoint Preserved:** `models/soil_classifier/best_model.pth` (`SHA-256: 1b11f7e08c386e85f863c307c8256d8c4998a8f63b39b1a62ae4e03d6e2ac1e5`)  
**PostgreSQL `soildb` Preserved:** `farmers=11`, `fields=19`, `soil_tests=11`, `soil_images=28`, `analysis_results=17`

---

## 1. Weeks 1–2 Requirements (100% Complete & Verified)
1. **JPEG & PNG upload:** `app/routers/images.py` (`upload_image_for_field`, `upload_image_flat`), `app/static/index.html` (`#fileInput`).
2. **Android camera capture:** `app/static/index.html` (`getUserMedia` with `facingMode: { ideal: 'environment' }` and `#mobileCameraInput` with `capture="environment"`), `app/routers/images.py` (`upload_image_base64`).
3. **Server-side image validation:** `app/services/image_validation.py` (`validate_and_normalize`: Pillow `verify()` + `load()`, decompression bomb check, minimum side & Laplacian blur/brightness warnings).
4. **Image resizing & format normalization:** EXIF orientation transpose, RGBA-to-RGB white composite, thumbnail cap, JPEG re-encoding stripping EXIF, `224x224` resize.
5. **Structured NPK, pH, OM, moisture + ML metadata:** `app/models.py` (`SoilTest`), `app/schemas.py` (`SoilTestIn`, `SoilTestOut`), `app/routers/soil_tests.py` (`create_soil_test`), `scripts/migrate_add_ml_collection_fields.py`.
6. **OpenCV color correction:** `app/services/preprocessing.py` (`gray_world`, `normalize_illumination`, `reduce_noise`).
7. **Background removal:** `app/services/preprocessing.py` (`remove_background` via GrabCut with `0.35` foreground fallback).
8. **Feature-preserving augmentation:** `app/services/preprocessing.py` (`augment`), `ml/train_cnn.py` (`get_transforms`).
9. **USDA, ICAR & open-source dataset provenance:** `data/sources.csv`, `data/DATASET_MANIFEST.csv`, `app/routers/dataset.py` (`GET /api/dataset/status`).
10. **PostgreSQL storage per farmer & field + `soil_test_id` pairing:** `app/models.py` (`Farmer`, `Field`, `SoilTest`, `SoilImage.soil_test_id`, `AnalysisResult.soil_test_id`).

---

## 2. Weeks 3–4 Requirements & Implementation Status

| Feature | Status | Model / Endpoint | Dataset & License |
| :--- | :--- | :--- | :--- |
| **Soil Type Classification (6 classes)** | **Complete** | `models/soil_classifier/best_model.pth` / `POST /api/images/{id}/predict` | `data/processed` (Phantom-fs `CC BY-NC 4.0`, Kaggle `CC0`, Wikimedia `CC BY-SA 4.0`) |
| **Grad-CAM Explainability** | **Complete** | `SoilClassifier.generate_gradcam` / `GET /api/images/{id}/gradcam` | Uses existing `EfficientNet-B0` (`features[-1]`) |
| **Soil Texture Classification** | **Complete** | `models/soil_texture_classifier/best_model.pth` / `POST /api/images/{id}/predict-texture` | `soil_texture_92k` (`10.5281/zenodo.20404766`, `CC BY 4.0`, 92,050 patches; 3,600 balanced subset trained) |
| **Visible Plant Nutrient Deficiency** | **Complete** | `models/nutrient_deficiency_detector/best_model.pth` / `POST /api/images/{id}/predict-deficiency` | `mpld_nutrient_deficiency` (`10.5281/zenodo.10041514`, `CC BY 4.0`, 20,892 images; 2,400 balanced subset trained) |
| **Gradient-Boosted Nutrient ML** | **Complete** | `models/nutrient_classifier/model.joblib` / `POST /api/ml/predict-nutrient-fertility` | `rice_npk_fertility` (`10.5281/zenodo.15307297`, `CC BY 4.0`, 2,000 rows; 1,999 deduplicated) |
| **SHAP Explainability** | **Complete** | `shap.TreeExplainer` in `app/services/nutrient_ml.py` / `POST /api/ml/predict-nutrient-fertility` | Explains XGBoost `nutrient_classifier` predictions |
| **Hybrid Image + Tabular Fusion** | **Architecture Ready (Blocked by Data)** | `HybridSoilFusionNet` (`ml/train_fusion_model.py`) / `GET /api/ml/hybrid-fusion/status` | **Blocked:** `0` real paired `(image, soil_test)` records with shared ground-truth labels in `soildb` or open repositories |
| **Historical Soil Trend Analysis** | **Complete (Non-ML)** | `app/services/soil_trends.py` / `GET /api/soil-tests/trends` | Farmer-owned `soildb.soil_tests` |
| **ML Soil Degradation Forecast** | **Complete** | `models/degradation_forecaster/model.joblib` / `POST /api/ml/predict-degradation-risk` | `soilhealthdb_v2` (`10.5281/zenodo.17565204`, `CC BY 4.0`, 5,907 rows; chronological split) |
| **Model Registry** | **Complete** | `app/services/model_registry.py` / `GET /api/ml/registry`, `GET /api/cnn/capabilities` | Dynamic metadata & metrics |

---

## 3. Exact Measured Evaluation Metrics

1. **`soil_type_classifier` (`EfficientNet-B0`, preserved):**
   - Test accuracy: **89.81%** (`0.8981`), Macro F1: **88.99%** (`0.8899`), `206` test samples.
2. **`soil_texture_classifier` (`EfficientNet-B0`, `Clay`, `Loam`, `Sand`, 540 test patches):**
   - Test accuracy: **100.00%** (`1.0000`), Macro Precision: **1.0000**, Macro Recall: **1.0000**, Macro F1: **1.0000**, Confusion Matrix: `[[180, 0, 0], [0, 180, 0], [0, 0, 180]]`.
   - *Limitation:* `soil_texture_92k` consists of 224x224 patches without parent-image IDs; patch-level random splitting yields 100% accuracy and should be re-evaluated with source-image grouping if parent IDs become available.
3. **`nutrient_deficiency_detector` (`EfficientNet-B0`, `-p`, `-p50`, `_c`, 360 test images):**
   - Test accuracy: **87.22%** (`0.8722`), Macro Precision: **0.8739**, Macro Recall: **0.8722**, Macro F1: **0.8723**, Confusion Matrix: `[[111, 4, 5], [7, 105, 8], [2, 20, 98]]`.
4. **`nutrient_classifier` (`XGBClassifier`, 4 fertility classes, 400 deduplicated test rows):**
   - Test accuracy: **26.25%** (`0.2625`), Macro Precision: **0.2321**, Macro Recall: **0.2138**, Macro F1: **0.2082**, Weighted F1: **0.2438**, Macro ROC-AUC (OvR): **0.4831**, Log-loss: **1.4625**.
   - *Limitation:* `Machine_Learning_Training_Data_Rice.csv` contains only N, P, K without pH, EC, OC, or crop management covariates, resulting in near-chance separability across the 4 fertility labels.
5. **`degradation_forecaster` (`XGBClassifier` + `XGBRegressor`, `SoilHealthDB V2` chronological split, 274 later-year test rows):**
   - Classification (SOC decline `OC_T < OC_C`): Accuracy: **67.88%** (`0.6788`), Macro F1: **0.4532**, ROC-AUC: **0.4662**, PR-AUC: **0.1172**, Confusion Matrix: `[[181, 60], [28, 5]]`.
   - Regression (`OC_T - OC_C`): MAE: **0.1461**, RMSE: **0.2044**, R²: **-0.0008**.
   - *Limitation:* High missingness in covariates (`SoilBD` missing in 4,222/5,907 rows) and unobserved treatment details limit cross-site temporal generalization.

---

## 4. Reproducibility Instructions
```powershell
# 1. Download & verify open-licensed datasets
.\venv\Scripts\python.exe scripts\download_datasets.py
.\venv\Scripts\python.exe scripts\verify_datasets.py

# 2. Train new models (does NOT overwrite models/soil_classifier/best_model.pth)
.\venv\Scripts\python.exe ml\train_texture_model.py
.\venv\Scripts\python.exe ml\train_deficiency_model.py
.\venv\Scripts\python.exe ml\train_nutrient_model.py
.\venv\Scripts\python.exe ml\train_degradation_model.py

# 3. Run full test suite
.\venv\Scripts\python.exe -m pytest -q
```

