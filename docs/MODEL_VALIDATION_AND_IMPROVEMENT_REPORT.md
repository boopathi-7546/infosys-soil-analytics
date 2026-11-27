# SoilSense AI – Model Validation and Improvement Report

**Audit Date:** 2026-11-27  
**Preserved Checkpoints:** No existing model checkpoints (`models/soil_classifier/best_model.pth`, `models/soil_texture_classifier/best_model.pth`, `models/nutrient_deficiency_detector/best_model.pth`, `models/nutrient_classifier/model.joblib`, `models/degradation_forecaster/model.joblib`) were overwritten during this validation audit.

---

## Executive Summary & Deployment Readiness

| Model | Previous Metrics | Validated / Baseline Metrics | Leakage / Data Findings | Safe for Demo? | Safe for Real-World Use? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Soil Type Classifier** (`EfficientNet-B0`, preserved) | Acc `89.81%`, Macro F1 `88.99%` | Unchanged (`SHA-256: 1b11f7e0...`) | Clean `train`/`val`/`test` split (0 exact SHA-256 overlap) | **Yes** | **Indicative only** (visual surface type, not lab NPK) |
| **2. Soil Texture Classifier** (`soil_texture_92k`) | Acc `100.00%`, Macro F1 `1.0000` (random patch split) | **Grouped (`StratifiedGroupKFold`, 0 parent overlap):** Acc **`90.80%`**, Macro F1 **`0.9169`** | **Patch-level leakage found:** 92,050 patches come from only **246** parent images (up to 600 patches/image). Random patch split leaked same-image patches! | **Yes** (retrain with grouped split upon approval) | **No** (patch-trained on 3 classes; not a lab sieve/hydrometer test) |
| **3. Nutrient Deficiency Detector** (`MPLD`) | Acc `87.22%`, Macro F1 `0.8723` | Verified class-wise: `-P` F1 `0.9250`, `-P50` F1 `0.8434`, `_C` F1 `0.8485` | Controlled maize P-omission dataset (`-P` = 0% P, `-P50` = 50% P, `_C` = Control) | **Yes** (with non-lab disclaimer) | **No** (maize-specific, single-nutrient P only) |
| **4. Nutrient Fertility Classifier** (`rice_npk_fertility`) | Acc `26.25%`, Macro F1 `0.2082` | `DummyMostFreq` Acc `34.75%`, `RandomForest(bal)` Acc `27.50%` / F1 `0.2555`, `Tuned XGB(bal)` Acc `24.00%` / F1 `0.2374` | **Uninformative labels:** All 4 classes have identical NPK means (`~57.8, ~35.2, ~29.8`) + `15` conflicting `(N,P,K)` triples | **Pipeline demo only** | **No — Unsafe** (near-chance predictive signal) |
| **5. Degradation Forecaster** (`SoilHealthDB V2`) | Cls Acc `67.88%`, AUC `0.4662`, Reg R² `-0.0008` | `DummyCls` Acc `87.96%` / F1 `0.4680`, `DummyReg(median)` MAE `0.1446` / R² `-0.1891` | **Target & temporal shift:** `OC_T < OC_C` rate shifts from `23.76%` (`<=2009`) to `12.04%` (`>2009`); `SoilBD` 71% missing; measures treatment effect, not single-plot decay | **Pipeline demo only** | **No — Unsafe** (use Historical Soil Trend Analysis for user plots) |
| **6. Hybrid Fusion** (`HybridSoilFusionNet`) | Untrained (`architecture_ready_untrained`) | `0` genuine paired `(image, soil_test)` records in `soildb` | No fake pairing performed | **Architecture only** | **No — Blocked by Data** |

---

## 1. Soil Texture Classifier (`soil_texture_92k`)

### Data Leakage Investigation
- Inspection of `data/external/soil_texture_92k/extracted/dataset_v3/` revealed that filenames encode a **parent image prefix** (e.g., `01aaa7eb8147_0000.jpg` … `01aaa7eb8147_0599.jpg`, up to **600 patches per parent image**).
- Across all `92,050` patches, there are only **246** unique `(class, parent_prefix)` groups (`train`: 183, `val`: 31, `test`: 32, with `0` parent overlap in the official `dataset_v3` folders).
- The previous `ml/train_texture_model.py` run subsampled the first `1,200` patches per class and applied a random `StratifiedShuffleSplit`, leaking patches from the same parent soil photograph across train, validation, and test sets and inflating accuracy to `100.0%`.

### Previous vs Parent-Grouped Evaluation (`StratifiedGroupKFold`, 0 Group Overlap)
- **Split:** `3,920` train patches (`196` parent groups) vs `1,000` held-out test patches (`50` disjoint parent groups).
- **Overall Grouped Metrics:** Accuracy = **`0.9080`**, Macro Precision = **`0.9231`**, Macro Recall = **`0.9255`**, Macro F1 = **`0.9169`**.
- **Per-Class Metrics & Confusion Matrix (`[[Clay], [Loam], [Sand]]`):**
  - `Clay`: Precision `0.7841`, Recall `0.9875`, F1 `0.8741` (support `320`)
  - `Loam`: Precision `0.9852`, Recall `0.7929`, F1 `0.8786` (support `420`)
  - `Sand`: Precision `1.0000`, Recall `0.9962`, F1 `0.9981` (support `260`)
  - Confusion Matrix: `[[316, 4, 0], [87, 333, 0], [0, 1, 259]]` (`87` Loam patches predicted as Clay).
- **Recommendation:** Safe for **demonstration**; if approved, replace `models/soil_texture_classifier/best_model.pth` (after backing up the existing checkpoint) using a grouped `StratifiedGroupKFold` / `dataset_v3` parent-disjoint split.

---

## 2. Nutrient Fertility Classifier (`rice_npk_fertility`)

### Root-Cause Diagnosis of `26.25%` Accuracy
- **Train/Test Leakage Check:** `load_deduplicated_data()` deduplicates exact `(N, P, K, Fertility)` rows prior to `StratifiedShuffleSplit` (no row leakage).
- **Feature & Label Quality:** Across `2,000` rows (`1,984` unique `(N, P, K)` triples), **`15` identical `(N, P, K)` triples carry conflicting `Fertility` labels**. Furthermore, all 4 classes have nearly identical distributions:
  - Class 0 (`n=331`): mean NPK = `[57.52, 36.22, 30.21]`, std = `[23.20, 14.42, 8.64]`
  - Class 1 (`n=694`): mean NPK = `[57.85, 35.24, 29.86]`, std = `[24.01, 14.48, 8.82]`
  - Class 2 (`n=654`): mean NPK = `[57.82, 34.93, 29.83]`, std = `[24.30, 14.44, 8.68]`
  - Class 3 (`n=321`): mean NPK = `[57.72, 34.79, 29.49]`, std = `[23.03, 14.53, 8.40]`

### Baseline & Tuning Comparison
- `DummyClassifier(most_frequent)`: Acc `0.3475`, Macro F1 `0.1289`
- `LogisticRegression(class_weight="balanced")`: Acc `0.1850`, Macro F1 `0.1691`
- `XGBoost` (original): Acc `0.2625`, Macro F1 `0.2082`
- `XGBoost` (tuned `max_depth=3`, `lr=0.05`, balanced sample weights): Acc `0.2400`, Macro F1 `0.2374`
- `RandomForest(class_weight="balanced")`: Acc `0.2750`, Macro F1 `0.2555`
- **Recommendation:** **Not safe for real-world agronomic decisions.** N, P, K alone in this dataset carry negligible mutual information with `Fertility`. Keep only as an end-to-end XGBoost + SHAP pipeline demonstration until a verified multi-feature Soil Health Card dataset (`N, P, K, pH, EC, OC, crop`) is acquired.

---

## 3. Degradation Forecaster (`SoilHealthDB V2`)

### Chronological Audit & Baseline Comparison
- **Leakage Check:** `train_degradation_model.py` splits strictly by `SamplingYear <= 2009` (`n=1,292`) vs `SamplingYear > 2009` (`n=274`) and fits `SimpleImputer(strategy="median")` inside the `Pipeline` on training data only. Zero future leakage.
- **Target & Distribution Shift:** `OC_T < OC_C` rate drops from `23.76%` in `<= 2009` to `12.04%` in `> 2009`. Moreover, `OC_T - OC_C` compares a conservation *treatment* against a *control* rather than tracking single-plot degradation over time.
- **Baseline Comparison:**
  - `DummyClassifier(most_frequent)`: Acc `0.8796`, Macro F1 `0.4680` vs `XGBClassifier`: Acc `0.6788`, Macro F1 `0.4532`, ROC-AUC `0.4662`.
  - `DummyRegressor(median)`: MAE `0.1446`, R² `-0.1891` vs `XGBRegressor`: MAE `0.1461`, R² `-0.0008`.
- **Recommendation:** **Not safe for real-world degradation forecasting.** Retain the clearly labelled **Historical Soil Trend Analysis** (`is_ml_forecast=False`) for user plots until multi-season single-plot longitudinal records are collected.

---

## 4. Nutrient Deficiency Detector (`MPLD`)

- **Class Meaning Verification:** In *Maize Phosphorus Leaf Deficiency (MPLD)* (`10.5281/zenodo.10041514`, `CC BY 4.0`):
  - `-P` (`-p`): **0% Phosphorus** (complete P omission / severe deficiency) — Precision `0.9250`, Recall `0.9250`, F1 `0.9250` (`support=120`)
  - `-P50` (`-p50`): **50% Phosphorus reduction** (moderate deficiency) — Precision `0.8140`, Recall `0.8750`, F1 `0.8434` (`support=120`)
  - `_C` (`_c`): **Control** (100% P / healthy maize leaf) — Precision `0.8829`, Recall `0.8167`, F1 `0.8485` (`support=120`)
  - Confusion Matrix: `[[111, 4, 5], [7, 105, 8], [2, 20, 98]]` (Overall Acc `87.22%`, Macro F1 `0.8723`).
- **Recommendation:** Safe for **demonstration** on maize leaf images with the existing non-lab disclaimer (`"Visual nutrient deficiency indication from plant image — not a laboratory soil nutrient measurement."`).

---

## 5. Hybrid Fusion (`HybridSoilFusionNet`)

- **Paired Data Count:** `0` genuine paired `(soil_image, soil_test)` records with shared ground-truth labels exist in `soildb`.
- **Requirement to Activate:** Collect `>= 50–300+` co-located, same-date paired records via `SoilImage.soil_test_id` and `AnalysisResult.soil_test_id` with verified ground-truth labels, split by `field_id` (`GroupKFold`).

