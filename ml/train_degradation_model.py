"""Train and evaluate Time-Aware ML Soil Degradation Forecaster (Phase H).

Dataset: SoilHealthDB V2 (Zenodo DOI: 10.5281/zenodo.17565204, CC BY 4.0)
Target:  Soil Organic Carbon (SOC) decline risk (OC_T < OC_C) and SOC change (OC_T - OC_C)
Validation: Strict chronological split (earlier SamplingYear -> train, later SamplingYear -> test)
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "external" / "soilhealthdb_v2" / "SoilHealthDB_V2.csv"
OUT_DIR = PROJECT_ROOT / "models" / "degradation_forecaster"

FEATURES = ["OC_C", "Duration", "SoilpH", "SoilBD", "SandPerc", "SiltPerc", "ClayPerc", "MAT", "MAP"]


def _parse_float(val: object) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none"}:
        return float("nan")
    try:
        v = float(s)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")


def _parse_year(row: dict[str, str]) -> int | None:
    for col in ("SamplingYear", "YearPublication"):
        v = _parse_float(row.get(col))
        if not math.isnan(v) and 1900 <= v <= 2100:
            return int(v)
    return None


def load_chronological_records(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[list[float]] = []
    y_cls_list: list[int] = []
    y_reg_list: list[float] = []
    years: list[int] = []

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            yr = _parse_year(row)
            oc_c = _parse_float(row.get("OC_C"))
            oc_t = _parse_float(row.get("OC_T"))
            dur = _parse_float(row.get("Duration"))
            if yr is None or math.isnan(oc_c) or math.isnan(oc_t) or math.isnan(dur):
                continue

            feat = [
                oc_c,
                dur,
                _parse_float(row.get("SoilpH")),
                _parse_float(row.get("SoilBD")),
                _parse_float(row.get("SandPerc")),
                _parse_float(row.get("SiltPerc")),
                _parse_float(row.get("ClayPerc")),
                _parse_float(row.get("MAT")) if not math.isnan(_parse_float(row.get("MAT"))) else _parse_float(row.get("Tannual")),
                _parse_float(row.get("MAP")) if not math.isnan(_parse_float(row.get("MAP"))) else _parse_float(row.get("Pannual")),
            ]
            X_list.append(feat)
            y_cls_list.append(1 if oc_t < oc_c else 0)
            y_reg_list.append(oc_t - oc_c)
            years.append(yr)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_cls_list, dtype=np.int64),
        np.array(y_reg_list, dtype=np.float32),
        np.array(years, dtype=np.int64),
    )


def train_and_evaluate(seed: int = 42) -> dict:
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f"Dataset not found at {DATA_CSV}. Run scripts/download_datasets.py first.")

    X, y_cls, y_reg, years = load_chronological_records(DATA_CSV)
    # Chronological 80/20 split by SamplingYear quantile (no future observations in training)
    cutoff_year = int(np.quantile(years, 0.80))
    train_mask = years <= cutoff_year
    test_mask = years > cutoff_year
    if test_mask.sum() < 50:
        cutoff_year = int(np.quantile(years, 0.75))
        train_mask = years <= cutoff_year
        test_mask = years > cutoff_year

    X_train, X_test = X[train_mask], X[test_mask]
    y_cls_train, y_cls_test = y_cls[train_mask], y_cls[test_mask]
    y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]

    cls_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            eval_metric="logloss",
            random_state=seed,
        )),
    ])
    cls_pipe.fit(X_train, y_cls_train)

    reg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            random_state=seed,
        )),
    ])
    reg_pipe.fit(X_train, y_reg_train)

    prob_test = cls_pipe.predict_proba(X_test)[:, 1]
    pred_cls = (prob_test >= 0.5).astype(int)
    pred_reg = reg_pipe.predict(X_test)

    acc = float(accuracy_score(y_cls_test, pred_cls))
    f1 = float(f1_score(y_cls_test, pred_cls, average="macro", zero_division=0))
    roc_auc = float(roc_auc_score(y_cls_test, prob_test)) if len(np.unique(y_cls_test)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_cls_test, prob_test)) if len(np.unique(y_cls_test)) > 1 else 0.0
    cm = confusion_matrix(y_cls_test, pred_cls).tolist()

    mae = float(mean_absolute_error(y_reg_test, pred_reg))
    rmse = float(math.sqrt(mean_squared_error(y_reg_test, pred_reg)))
    r2 = float(r2_score(y_reg_test, pred_reg))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "model.joblib"
    joblib.dump(
        {
            "classifier": cls_pipe,
            "regressor": reg_pipe,
            "features": FEATURES,
            "cutoff_year": cutoff_year,
        },
        model_path,
    )

    metadata = {
        "model_name": "degradation_forecaster",
        "model_version": "degradation_forecaster_v1.0",
        "architecture": "Time-Aware XGBClassifier + XGBRegressor (Median Imputer)",
        "dataset": "SoilHealthDB V2 (Zenodo 10.5281/zenodo.17565204)",
        "dataset_version": "V2",
        "license": "CC BY 4.0",
        "training_date": dt.date.today().isoformat(),
        "random_seed": seed,
        "validation_strategy": f"Chronological split: train SamplingYear <= {cutoff_year}, test SamplingYear > {cutoff_year}",
        "train_years_range": [int(years[train_mask].min()), int(years[train_mask].max())],
        "test_years_range": [int(years[test_mask].min()), int(years[test_mask].max())],
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "features": FEATURES,
        "targets": {
            "classification": "soc_decline_risk (1 if OC_T < OC_C else 0)",
            "regression": "soc_change (OC_T - OC_C)",
        },
        "metrics": {
            "classification": {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "confusion_matrix": cm,
            },
            "regression": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
            },
        },
        "limitations": "ML Soil Degradation Forecast trained on SoilHealthDB V2 multi-site experiments; distinct from descriptive Historical Soil Trend Analysis.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DEGRADATION_ML] Saved to {model_path} | Cls Acc={acc:.4f} MacroF1={f1:.4f} AUC={roc_auc:.4f} | Reg MAE={mae:.4f} R2={r2:.4f}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

