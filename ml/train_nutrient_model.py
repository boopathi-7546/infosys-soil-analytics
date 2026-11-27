"""Train and evaluate XGBoost Soil Nutrient Fertility Classifier (Phase E).

Dataset: Training Data of Soil NPK Fertility for Rice Cultivation
         (Zenodo DOI: 10.5281/zenodo.15307297, CC BY 4.0)
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "external" / "rice_npk_fertility" / "Machine_Learning_Training_Data_Rice.csv"
OUT_DIR = PROJECT_ROOT / "models" / "nutrient_classifier"

FEATURES = ["Nitrogen", "Phosphorus", "Potassium"]
CLASS_LABELS = {
    0: "fertility_class_0_low",
    1: "fertility_class_1_moderate",
    2: "fertility_class_2_high",
    3: "fertility_class_3_very_high",
}


def load_deduplicated_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    seen: set[tuple[float, float, float, int]] = set()
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    total_raw = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total_raw += 1
            n = float(row["Nitrogen"])
            p = float(row["Phosphorus"])
            k = float(row["Potassium"])
            label = int(row["Fertility"])
            key = (n, p, k, label)
            if key in seen:
                continue
            seen.add(key)
            X_rows.append([n, p, k])
            y_rows.append(label)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64), total_raw


def train_and_evaluate(seed: int = 42) -> dict:
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f"Dataset not found at {DATA_CSV}. Run scripts/download_datasets.py first.")

    X, y, total_raw = load_deduplicated_data(DATA_CSV)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=1.0,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_p = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
    nll = float(log_loss(y_test, y_prob))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[CLASS_LABELS[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "class_labels": CLASS_LABELS,
            "background_X": X_train[: min(200, len(X_train))],
        },
        model_path,
    )

    metadata = {
        "model_name": "nutrient_classifier",
        "model_version": "xgboost_nutrient_v1.0",
        "architecture": "XGBClassifier (n_estimators=150, max_depth=5)",
        "dataset": "Training Data of Soil NPK Fertility for Rice Cultivation (Zenodo 10.5281/zenodo.15307297)",
        "dataset_version": "1.0",
        "license": "CC BY 4.0",
        "training_date": dt.date.today().isoformat(),
        "random_seed": seed,
        "features": FEATURES,
        "classes": [CLASS_LABELS[i] for i in range(4)],
        "total_raw_rows": total_raw,
        "deduplicated_rows": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc_ovr_macro": round(roc_auc, 4),
            "log_loss": round(nll, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "limitations": "Trained on Rice NPK fertility dataset (N, P, K); requires unit compatibility with training distribution.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[NUTRIENT_ML] Saved model to {model_path} | Acc={acc:.4f} | MacroF1={macro_f1:.4f} | ROC-AUC={roc_auc:.4f}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

