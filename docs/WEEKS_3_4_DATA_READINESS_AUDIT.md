# SoilSense AI – Weeks 3–4 Data Readiness Audit

**Audit Date:** 2026-11-27  
**Project Root:** `D:\Infosys 7.0 Al-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory\soil-analytics`  
**Audit Scope:** Read-only inspection of `data/`, `models/soil_classifier/`, `ml/`, `app/`, and PostgreSQL `soildb`. No source code, database records, or model checkpoints were modified.

---

## Executive Summary

| Feature | Data Status | Verified Local Data Available | Blocking Gap | Realistic Now? |
| :--- | :--- | :--- | :--- | :--- |
| **1. Soil texture classification** | **Blocked** | `data/raw` (1,361 imgs) & `data/processed` (4,193 imgs) labelled by 6 soil types only | Zero USDA/FAO particle-size texture labels | **No** (requires labelled texture dataset) |
| **2. Visible plant nutrient deficiency detection** | **Blocked** | Zero plant/leaf/soil deficiency images or labels | Zero deficiency images & verified labels | **No** (requires external labelled dataset) |
| **3. Gradient-boosted nutrient deficiency prediction** | **Blocked** | `11` `soil_tests` rows in PostgreSQL `soildb` (unlabelled) | Insufficient rows (`11`), 0 verified deficiency labels, missing unit/method columns | **No** (rule-based advisory only) |
| **4. Hybrid image + structured soil-test fusion** | **Blocked** | `17` `analysis_results` (`image_id` only), `11` `soil_tests`, **`0` paired records** | Zero paired `(image, soil_test)` rows + zero joint labels | **No** (requires paired data collection) |
| **5. SHAP explainability** | **Blocked** | Depends on Feature 3 / Feature 4 (`shap` not installed) | No trained tabular/fusion ML model exists | **No** (blocked by Feature 3) |
| **6. ML-based soil degradation forecasting** | **Blocked** (Trend Analysis Active) | `11` `soil_tests` over `2026-11-13` to `2026-11-26` (13 days, 3 fields) | Insufficient multi-season history & zero degradation outcome labels | **Partial** (Historical Soil Trend Analysis only; ML forecast blocked) |

---

## Verified Project Baseline (Observed Facts)

### A. Image Dataset (`data/raw` and `data/processed`)
- **Label classes (6 soil types only):** `alluvial_soil`, `black_soil`, `clay_soil`, `laterite_soil`, `red_soil`, `sandy_soil`.
- **Exact file counts by split and class:**

| Class | `data/raw` | `data/processed/train` | `data/processed/val` | `data/processed/test` |
| :--- | ---: | ---: | ---: | ---: |
| `alluvial_soil` | 343 | 956 | 51 | 52 |
| `black_soil` | 356 | 992 | 53 | 54 |
| `clay_soil` | 120 | 332 | 18 | 18 |
| `laterite_soil` | 215 | 596 | 32 | 33 |
| `red_soil` | 266 | 740 | 40 | 40 |
| `sandy_soil` | 61 | 168 | 9 | 9 |
| **Total** | **1,361** | **3,784** | **203** | **206** |

- **Deduplication & Leakage Check (SHA-256 across `data/processed`):**
  - `processed/train`: 3,784 files, 3,782 unique SHA-256 hashes (**2** intra-split exact duplicates).
  - `processed/val`: 203 files, 203 unique SHA-256 hashes (**0** intra-split exact duplicates).
  - `processed/test`: 206 files, 206 unique SHA-256 hashes (**0** intra-split exact duplicates).
  - Cross-split exact SHA-256 overlap (`train<->val`, `train<->test`, `val<->test`): **0** exact matches.
  - Note: `processed/train` (3,784) is ~2.78x `data/raw` (1,361) due to offline augmentation on the training split; near-duplicate perceptual hashing (pHash) was not run, so perceptual similarity from web-scraped sources remains an unverified risk.
- **Class Imbalance:** `black_soil` (356 raw / 54 test) outnumbers `sandy_soil` (61 raw / 9 test) by **5.84:1**.
- **Existing Trained CNN (`models/soil_classifier/best_model.pth`):** EfficientNet-B0 (`metrics.json`: test accuracy `89.81%`, macro F1 `88.99%`, `206` test samples).

### B. PostgreSQL Database (`soildb`)
- **`soil_tests`:** **11 rows** across 3 fields (`field_id=1`: 3 rows, `2026-11-13` to `2026-11-14`; `field_id=2`: 7 rows, `2026-11-14` to `2026-11-26`; `field_id=3`: 1 row, `2026-11-18`).
- **`soil_tests` columns:** `id`, `field_id`, `nitrogen`, `phosphorus`, `potassium`, `ph`, `organic_matter`, `moisture`, `notes`, `tested_at`.
- **Missing columns in `soil_tests`:** No `nitrogen_unit`, `phosphorus_unit`, `potassium_unit`, `electrical_conductivity`, `organic_carbon`, `ph_method`, `data_provenance`, sampling depth, bulk density, or ground-truth deficiency/degradation labels.
- **`soil_images`:** **28 rows** (`id`, `field_id`, `file_path`, `original_filename`, `original_format`, `size_bytes`, `width`, `height`, `blur_score`, `uploaded_at`).
- **`analysis_results`:** **17 rows** (`count(image_id) = 17`, **`count(soil_test_id) = 0`**).

---

## Feature-by-Feature Data Readiness Audit

### 1. Soil Texture Classification
- **Existing Dataset & Location:** `data/raw/` (1,361 images) and `data/processed/{train,val,test}/` (4,193 images total).
- **Available Labels & Classes:** Only 6 pedological soil-type classes (`alluvial_soil`, `black_soil`, `clay_soil`, `laterite_soil`, `red_soil`, `sandy_soil`). **Zero** USDA/FAO particle-size texture labels (e.g., *Sand*, *Loamy Sand*, *Sandy Loam*, *Loam*, *Silt Loam*, *Clay Loam*, *Silty Clay*, *Clay*).
- **Pairing with Soil Tests:** None (0 paired records).
- **Missing Fields:** Lab-measured sand/silt/clay percentages (hydrometer/pipette sieve analysis) and standardized macro/close-up soil photography with scale/lighting calibration.
- **Imbalance, Duplicates & Leakage Risks:** Existing soil-type dataset has 5.84:1 class imbalance (`sandy_soil` has only 9 test images) and 2 exact duplicates in `train`. More fundamentally, **soil type cannot be assumed to equal soil texture** (e.g., alluvial soils range from sandy loam to silty clay).
- **Recommended Architecture & Metrics:**
  - *Architecture:* EfficientNet-B0 or ConvNeXt-Tiny multi-class classifier (or multi-task regression predicting `% sand, % silt, % clay` constrained to sum to 100%).
  - *Metrics:* Macro F1-score, per-class precision/recall, confusion matrix (for classification), or MAE/RMSE + texture-class accuracy (for regression).
- **Suitable Public Datasets & Licensing:**
  - **LUCAS Topsoil Survey (EU Joint Research Centre):** Contains laboratory-measured clay/silt/sand % and soil photographs; **open access / European Commission reuse notice** (requires attribution; verify commercial terms per release).
  - **Soil Classification / Soil Texture Datasets on Kaggle/Roboflow:** Often lack lab-verified sieve percentages and carry **dataset-specific licenses (CC BY 4.0 / CC0 / Apache-2.0)**—must verify lab provenance and license before use.
- **Verified vs Assumed:** Verified that 0 texture labels exist locally; assumed that surface-only photos have limited accuracy for fine texture separation without macro scale/moisture control.

---

### 2. Visible Plant Nutrient Deficiency Detection
- **Existing Dataset & Location:** **None.** `data/` contains soil surface images only; `soil_images` in PostgreSQL (28 rows) are soil captures.
- **Available Images & Labels:** `0` plant/leaf images and `0` visible deficiency labels (e.g., N chlorosis, P purpling, K marginal scorch, Fe/Mg interveinal chlorosis).
- **Pairing with Soil Tests:** `0` paired records.
- **Missing Fields:** Crop species, growth stage, leaf/canopy RGB images, and tissue/soil-confirmed deficiency ground truth.
- **Imbalance, Duplicates & Leakage Risks:** Not applicable locally (0 samples). Public leaf-disease/deficiency datasets frequently suffer from same-plant/same-field photo bursts, requiring **grouped splits by plant/plot** to prevent leakage.
- **Recommended Architecture & Metrics:**
  - *Architecture:* EfficientNet-B0 / MobileNetV3-Small multi-label or multi-class classifier (crop-conditioned if multiple crops are supported).
  - *Metrics:* Macro F1-score, per-deficiency precision/recall, AUROC, and confusion matrix.
- **Suitable Public Datasets & Licensing:**
  - **Mendeley Data – Nutrient Deficiency in Rice / Banana / Maize / Apple Leaf Datasets:** Typically **CC BY 4.0** (verify each dataset record individually).
  - **PlantVillage (for disease vs abiotic stress separation):** **CC0 / Public Domain** (note: primarily infectious diseases, not pure nutrient deficiencies).
- **Verified vs Assumed:** Verified 0 deficiency images/labels in repo and DB; assumed crop-specific deficiency models cannot generalize across unrelated crops without crop-species metadata.

---

### 3. Gradient-Boosted Nutrient Deficiency Prediction
- **Existing Dataset & Location:** PostgreSQL `soildb` table `soil_tests` (**11 rows**).
- **Available Fields:** `nitrogen`, `phosphorus`, `potassium`, `ph`, `organic_matter`, `moisture`, `notes`, `tested_at` (all non-null in the 11 rows).
- **Available Labels:** **0** agronomist-verified nutrient deficiency labels. (Current `AdvisoryOut` uses hardcoded heuristic rules in `app/routers/soil_tests.py:_make_advisory`).
- **Pairing:** `0` paired `analysis_results`.
- **Missing Fields:**
  - Ground-truth deficiency target labels (verified independently of `_make_advisory()`).
  - Unit columns (`nitrogen_unit`, `phosphorus_unit`, `potassium_unit`), extraction methods (`Olsen P`, `Bray P`, `KMnO4 N`, `NH4OAc K`), sampling depth (`cm`), bulk density (`g/cm3`), EC, and crop target.
- **Imbalance, Duplicates & Leakage Risks:** 11 rows across 3 fields (7 rows in `field_id=2` within 12 days) is far too small for supervised ML and would severely overfit or leak same-field readings across train/test splits. `xgboost` is also not installed in `venv`.
- **Recommended Architecture & Metrics:**
  - *Architecture:* `XGBoost` (`XGBClassifier`) or `LightGBM` / `HistGradientBoostingClassifier` with **GroupKFold** grouped by `field_id`.
  - *Metrics:* Stratified/Grouped Macro F1-score, per-class precision, recall, PR-AUC, and confusion matrix.
- **Suitable Public Datasets & Licensing:**
  - **Kaggle Crop Recommendation / Soil Fertility Datasets:** Often **CC0 / DbCL / MIT** (caution: many Kaggle soil-fertility CSVs are synthetic or lack extraction method metadata—must audit provenance before training).
  - **ISRIC WoSIS (World Soil Information Service) / SoilGrids:** **CC BY 4.0** (provides global lab-measured soil profiles with pH, OC, N, P, K, texture, and bulk density).
  - **ICAR / Indian Soil Health Card Public Portals:** Government open data (verify portal terms).
- **Verified vs Assumed:** Verified 11 unlabelled `soil_tests` rows in PostgreSQL and `xgboost` absent from `venv`; assumed >= 300–500 independently labelled samples across >= 30+ fields are needed.

---

### 4. Hybrid Image + Structured Soil-Test Fusion
- **Existing Dataset & Location:** `analysis_results` (17 CNN-only rows) and `soil_tests` (11 lab-only rows) in PostgreSQL `soildb`.
- **Pairing Status:** `SELECT count(*), count(soil_test_id), count(image_id) FROM analysis_results` returned **`(17, 0, 17)`** — **0 paired records**.
- **Missing Fields:** Co-located, same-date `(soil_image, soil_test)` pairs linked via `analysis_results.soil_test_id`, plus a shared ground-truth target label (e.g., lab-verified soil class or fertility/deficiency status).
- **Imbalance, Duplicates & Leakage Risks:** With 0 paired rows, no fusion training or baseline comparison (CNN-only vs Tabular-only vs Fusion) is possible. Future paired collection must split by `field_id` so the same field does not appear in both train and test.
- **Recommended Architecture & Metrics:**
  - *Architecture:* Late or intermediate fusion combining `EfficientNet-B0` `1280-D` pooled embeddings (or 6-class logits) with normalized tabular features (`N, P, K, pH, OM, moisture`) via a shallow MLP fusion head or GBDT meta-learner.
  - *Metrics:* Identical held-out `GroupKFold` / field-disjoint test split comparing **CNN-only**, **Structured-only**, and **Fusion** using Accuracy, Macro F1-score, Precision, Recall, and Confusion Matrix.
- **Suitable Public Datasets & Licensing:** No single standard public dataset pairs RGB soil photos with full NPK+pH+OM lab assays under an open commercial license; **LUCAS Topsoil** provides soil photos + selected lab properties (pH, OC, N, P, K, texture) under European Commission reuse terms (requires registration/verification).
- **Verified vs Assumed:** Verified 0 paired records in `soildb`.

---

### 5. SHAP Explainability
- **Existing Dataset & Dependencies:** Blocked by Feature 3 / Feature 4 (no trained tabular or fusion ML model exists) and `shap` is not installed in `venv`.
- **Missing Requirements:** A trained, validated `XGBoost` / `LightGBM` / `scikit-learn` tree or tabular model and a representative background dataset (`>= 100` real samples).
- **Recommended Architecture & Metrics:**
  - *Architecture:* `shap.TreeExplainer` for gradient-boosted models returning per-prediction Shapley values mapped to plain-language farmer explanations.
  - *Metrics:* Explanation stability across CV folds and additivity check (`sum(shap_values) + base_value == model_output`).
- **Verified vs Assumed:** Verified `shap` is not installed and no supervised tabular model exists; rule-based health score weights are **not** presented as SHAP.

---

### 6. ML-Based Soil Degradation Forecasting
- **Existing Dataset & Location:** PostgreSQL `soil_tests` (**11 rows** spanning **13 days**: `2026-11-13` to `2026-11-26`), currently served by the non-ML **Historical Soil Trend Analysis** (`GET /api/soil-tests/trends`).
- **Records per Field:** `field_id=1`: 3 rows (spanning 8.5 hours); `field_id=2`: 7 rows (spanning 12 days); `field_id=3`: 1 row.
- **Missing Fields:**
  - Multi-season / multi-year longitudinal observations (e.g., >= 4–6 seasonal readings per plot across >= 30–50+ plots).
  - Verified degradation outcome labels (e.g., measured SOC loss, acidification rate, salinization/EC rise, or crop yield decline).
  - Management & environmental covariates (fertilizer applied, tillage, irrigation, rainfall, crop rotation).
- **Imbalance, Duplicates & Leakage Risks:** Training a forecasting model on 13 days of data across 3 plots would fit short-term noise. Time-series validation requires **forward-chaining / temporal splits** (never random shuffling across dates).
- **Recommended Architecture & Metrics:**
  - *Current Safe Implementation:* Chronological **Historical Soil Trend Analysis** (`is_ml_forecast=False`, `"Historical Soil Trend Analysis – Not an ML Forecast"`).
  - *Future ML Architecture:* Panel regression / Gradient-Boosted survival or risk classifier (`XGBoost`) or temporal sequence model once multi-season data exists.
  - *Metrics:* Temporal-split AUROC / PR-AUC / Brier score (risk classification) or MAE / RMSE (parameter trajectory).
- **Suitable Public Datasets & Licensing:** **ISRIC WoSIS** (**CC BY 4.0**) or long-term agronomic trials (e.g., Rothamsted e-RA / LTER datasets, subject to individual dataset data-use agreements).
- **Verified vs Assumed:** Verified 11 rows over 13 days in `soildb`; confirmed insufficient for ML forecasting.

---

## Prioritized Implementation Plan

1. **Immediate (Already Realistic & Active Today — No New ML Training Required):**
   - **EfficientNet-B0 6-Class Soil Type Classification + Grad-CAM** (active, `89.81%` test accuracy, `88.99%` macro F1).
   - **Historical Soil Trend Analysis (`GET /api/soil-tests/trends`)** (active, explicitly non-ML).
   - **Rule-Based Soil Health Score & Heuristic Advisory** (active).
   - **Schema & Data-Capture Upgrade (Next Engineering Step):** Add non-destructive additive columns (`nitrogen_unit`, `phosphorus_unit`, `potassium_unit`, extraction method, sampling depth, bulk density, and `soil_test_id` pairing in UI) so future user submissions build clean ML-ready paired records.
2. **Short-Term (After External Dataset Acquisition & License Review):**
   - **Feature 1 (Soil Texture)** or **Feature 2 (Visible Deficiency):** Acquire a verified public dataset (e.g., LUCAS Topsoil or CC-BY leaf deficiency dataset), deduplicate with pHash + SHA-256, and train a separate model head with grouped splits.
3. **Medium-Term (After Collecting >= 300–500 Labelled Lab Records):**
   - **Feature 3 (Gradient-Boosted Nutrient Classifier):** Install `xgboost`, train with `GroupKFold` by `field_id`, and evaluate precision/recall/F1/confusion matrix.
   - **Feature 4 (SHAP Explainability):** Install `shap` and integrate `TreeExplainer` with the validated Feature 3 model.
4. **Long-Term (After Collecting Paired & Multi-Season Field Data):**
   - **Feature 5 (Hybrid Fusion):** Requires `>= 300` co-located, same-date `(image, soil_test)` pairs with ground-truth labels.
   - **Feature 6 (ML Degradation Forecasting):** Requires multi-season longitudinal monitoring across dozens of fields with verified degradation outcomes.

