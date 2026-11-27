"""Train and evaluate EfficientNet-B0 Visible Plant Nutrient Deficiency Detector (Phase D).

Dataset: Maize Phosphorus Leaf Deficiency (MPLD) Database 224x224
         (Zenodo DOI: 10.5281/zenodo.10041514, CC BY 4.0)
Saves checkpoint separately to models/nutrient_deficiency_detector/best_model.pth.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = PROJECT_ROOT / "data" / "external" / "mpld_nutrient_deficiency" / "extracted"
OUT_DIR = PROJECT_ROOT / "models" / "nutrient_deficiency_detector"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeafImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return self.transform(rgb), label


def discover_deduplicated_samples(root_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Extracted dataset directory not found: {root_dir}")

    # Collect immediate parent folders that contain images, ignoring __MACOSX
    by_class: dict[str, list[Path]] = {}
    seen_sha: set[str] = set()
    for p in sorted(root_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if "__MACOSX" in p.parts:
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        cls_name = p.parent.name.strip().lower().replace(" ", "_")
        by_class.setdefault(cls_name, []).append(p)

    # Keep classes with >= 10 images
    valid_classes = sorted(c for c, paths in by_class.items() if len(paths) >= 10)
    if len(valid_classes) < 2:
        raise RuntimeError(f"Need at least 2 valid classes; found {list(by_class.keys())}")

    class_to_idx = {c: i for i, c in enumerate(valid_classes)}
    samples: list[tuple[Path, int]] = []
    for c in valid_classes:
        for p in by_class[c]:
            samples.append((p, class_to_idx[c]))
    return samples, valid_classes


def train_and_evaluate(epochs: int = 4, batch_size: int = 32, seed: int = 42, max_per_class: int | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples, classes = discover_deduplicated_samples(EXTRACT_DIR)
    if max_per_class is not None:
        capped: list[tuple[Path, int]] = []
        counts = Counter()
        for p, y in samples:
            if counts[y] < max_per_class:
                capped.append((p, y))
                counts[y] += 1
        samples = capped

    labels = np.array([y for _, y in samples])
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(s1.split(np.zeros(len(labels)), labels))

    temp_labels = labels[temp_idx]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(s2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(LeafImageDataset(train_samples, train_tf), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LeafImageDataset(val_samples, eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(LeafImageDataset(test_samples, eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    model.to(device)

    train_counts = Counter(labels[train_idx])
    total_train = len(train_idx)
    weights = torch.tensor([total_train / (len(classes) * train_counts[i]) for i in range(len(classes))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_epoch = 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "best_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(yb.tolist())
        val_f1 = float(f1_score(val_true, val_preds, average="macro", zero_division=0))
        print(f"[DEFICIENCY_CNN] Epoch {epoch}/{epochs} | val_macro_f1={val_f1:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "epoch": best_epoch,
                "val_macro_f1": best_val_f1,
            }, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            test_preds.extend(logits.argmax(dim=1).cpu().tolist())
            test_true.extend(yb.tolist())

    acc = float(accuracy_score(test_true, test_preds))
    macro_p = float(precision_score(test_true, test_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(test_true, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(test_true, test_preds, average="macro", zero_division=0))
    cm = confusion_matrix(test_true, test_preds).tolist()
    report = classification_report(test_true, test_preds, target_names=classes, output_dict=True, zero_division=0)

    (OUT_DIR / "class_names.json").write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    metadata = {
        "model_name": "nutrient_deficiency_detector",
        "model_version": "plant_deficiency_efficientnet_b0_v1.0",
        "architecture": "EfficientNet-B0 (PyTorch)",
        "dataset": "Maize Phosphorus Leaf Deficiency (MPLD) Database (Zenodo 10.5281/zenodo.10041514)",
        "dataset_version": "1.0",
        "license": "CC BY 4.0",
        "training_date": dt.date.today().isoformat(),
        "random_seed": seed,
        "best_epoch": best_epoch,
        "classes": classes,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "limitations": "Visual nutrient deficiency indication from plant leaf image only; not a laboratory soil NPK measurement.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DEFICIENCY_CNN] Test Acc={acc:.4f} | MacroF1={macro_f1:.4f} | Saved to {ckpt_path}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

