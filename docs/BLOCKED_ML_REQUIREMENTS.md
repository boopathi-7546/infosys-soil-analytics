# SoilSense AI – Blocked ML Features & Data Requirements

**Status:** Documented (Implementation Blocked by Insufficient Labelled & Paired Data)  
**Image Dataset Audited:** `data/raw` (1,361 images) & `data/processed/{train,val,test}` (6 soil-type folders only)  
**Database Audited:** PostgreSQL `soildb` (`11` `soil_tests` records, `17` `analysis_results` records, `0` paired `(soil_image, soil_test)` records)

---

## 0. Why Soil Texture & Visible Deficiency Image Models Are Blocked (Phase 1)

- **Dataset Inspection:** Both `data/raw/` and `data/processed/{train,val,test}/` are organized strictly into **six soil-type categories** (`alluvial_soil`, `black_soil`, `clay_soil`, `laterite_soil`, `red_soil`, `sandy_soil`).
- **No Texture or Deficiency Labels:** There are **zero** separate USDA/FAO soil texture annotations (e.g., Loam, Sandy Loam, Silty Clay Loam) and **zero** visible plant/soil nutrient deficiency labels (e.g., Nitrogen chlorosis, Phosphorus purpling, Potassium marginal scorch).
- **Scientific Boundary:** Broad pedological soil-type classes cannot be assumed to equal particle-size texture classes or visual nutrient deficiencies.

### Required Dataset Additions
1. **Soil Texture Dataset:** Lab-sieved/hydrometer-verified particle-size labels (sand/silt/clay % or USDA texture triangle classes) paired with standardized soil photographs (>= 100 images per texture class).
2. **Visible Deficiency Dataset:** Agronomist-annotated leaf/soil deficiency photographs with verified tissue/soil chemical confirmation (>= 150 images per deficiency class).

---

## 1. Why Supervised Nutrient Deficiency Classification Cannot Be Trained Today

- **Insufficient Sample Count:** The production database currently contains only **11** `soil_tests` records across 3 fields (7 on field #2, 3 on field #1, 1 on field #3). Training a gradient-boosted classifier (e.g., XGBoost / LightGBM / `HistGradientBoostingClassifier`) on 11 rows causes severe overfitting and cannot generalize across soil types, seasons, or agro-climatic zones.
- **No Ground-Truth Deficiency Labels:** Existing `soil_tests` rows store raw numeric inputs (`nitrogen`, `phosphorus`, `potassium`, `ph`, `organic_matter`, `moisture`, `notes`, `tested_at`) with **zero** agronomist-verified nutrient deficiency labels. Generating labels from existing rule-based thresholds (`_make_advisory()`) and training an ML model on those synthetic labels would only approximate the existing `if/elif` rules while falsely presenting them as learned ML accuracy.
- **Missing ML Dependencies:** Neither `xgboost` nor `shap` is installed in the project virtual environment (`venv`), and no synthetic training data may be fabricated.

### Required Labelled Soil-Test Data
To train and evaluate a defensible nutrient deficiency classifier (precision, recall, F1-score, confusion matrix):
1. **Minimum Dataset Size:** At least **300–500+ independent, non-duplicate soil samples** spanning diverse fields, crop seasons, and deficiency severities (with >= 50 samples per target class for stratified train/validation/test splits).
2. **Verified Ground-Truth Labels:** Agronomist-verified or laboratory-certified deficiency categories (e.g., `N_status`, `P_status`, `K_status`, or overall deficiency class) derived independently from crop response trials or Soil Health Card reference standards—not reverse-engineered from `_make_advisory()`.
3. **Standardized Units & Lab Metadata:** Recorded measurement units (`kg/ha` vs `mg/kg` / `ppm`), extraction methods (e.g., Alkaline KMnO4, Olsen/Bray P, 1N NH4OAc K), sampling depth (`cm`), and bulk density (`g/cm3`) when converting concentrations.

---

## 2. Why Hybrid Fusion (CNN + Structured Soil Data) Is Blocked

- **Zero Paired Records:** In `soildb`, `analysis_results` has 17 CNN prediction rows and `soil_tests` has 11 lab rows, with **`0` records where `analysis_results.soil_test_id IS NOT NULL`**.
- **Modality Alignment:** Effective late or intermediate fusion (combining EfficientNet-B0 1280-D visual embeddings or 6-class softmax probabilities with structured chemical features) requires **co-located, same-date** soil photographs and laboratory chemical analyses from the exact same soil sample, accompanied by a shared ground-truth target label.
- **Experimental Evaluation Requirement:** Claiming that fusion outperforms standalone EfficientNet-B0 (`89.81%` test accuracy, `88.99%` macro F1) or standalone tabular ML requires a held-out paired test set comparing all three models under identical splits.

### Required Paired Data Collection
1. Capture standardized surface/profile soil images alongside laboratory soil sampling at the same GPS plot and date, linking `analysis_results.soil_test_id` to `soil_tests.id`.
2. Collect >= 300 paired `(image, soil_test, ground_truth_label)` samples across all target classes before training and benchmarking a fusion head.

### Proposed Hybrid Fusion Architecture (For Future Implementation)
- **Visual Branch:** Frozen or fine-tuned `EfficientNet-B0` (`1280-D` pooled embedding before the 6-class linear head).
- **Tabular Branch:** Standardized structured vector (`N`, `P`, `K`, `pH`, `organic_matter`, `moisture`) normalized via `StandardScaler` or projected through a 2-layer MLP (`64-D`).
- **Fusion Head:** Concatenated `[1280 + 64]` feature vector -> `Dropout(0.3)` -> `Linear(1344, num_classes)` (or late-fusion gradient-boosted meta-learner over CNN class probabilities + tabular features), evaluated strictly against CNN-only and tabular-only baselines on identical held-out test splits.

---

## 3. Why Genuine SHAP Explanations Require a Trained ML Model

- **Model Dependency:** SHAP (`TreeExplainer`, `KernelExplainer`, or `LinearExplainer`) computes Shapley additive feature attributions with respect to a **trained predictive model's** learned decision function and background training distribution.
- **No Fabricated Attributions:** Without a trained, validated tabular or fusion model from Phases 2–3, computing or displaying "SHAP values" from hardcoded rule weights would misrepresent heuristic thresholds as Shapley game-theoretic attributions.
- **Prerequisite:** Once a validated gradient-boosted model is trained on real labelled data (Phase 2), `shap.TreeExplainer` can be integrated to return per-sample feature contributions in plain farmer-friendly language.

---

## 4. Why ML Degradation-Risk Forecasting Is Blocked (and What Is Supported Now)

- **Insufficient Longitudinal Depth:** Predicting future soil degradation risk via ML/time-series models requires multi-season, multi-year longitudinal observations (e.g., >= 4–6 seasonal readings per plot across dozens of plots) plus observed degradation outcome labels (organic carbon decline, salinization/EC rise, acidification, or yield drop).
- **What Is Supported Today:** Fields with >= 2 historical `soil_tests` records (such as field #2 with 7 tests and field #1 with 3 tests) support **Historical Soil Trend Analysis**—chronological tracking and earliest-to-latest deltas of comparable parameters, explicitly labelled as *"Historical Soil Trend Analysis – Not an ML Forecast"*.

