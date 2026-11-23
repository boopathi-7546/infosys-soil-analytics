"""
SoilSense AI - EfficientNet-B0 Soil Classification CNN Training Module
====================================================================
Uses PyTorch and Torchvision EfficientNet-B0 with ImageNet pretrained weights.
Classifies soil photographs into 6 agricultural categories:
  - alluvial_soil
  - black_soil
  - clay_soil
  - laterite_soil
  - red_soil
  - sandy_soil

Features:
- Robust image loading with error recovery for corrupted files.
- Bilinear resize to 224x224 with ImageNet normalization.
- Data augmentation applied strictly to the training split only.
- Weighted CrossEntropyLoss to resolve class distribution imbalance.
- Automatic CUDA GPU acceleration with CPU fallback.
- Validation loss monitoring with early stopping patience.
- Checkpoint saving to models/soil_classifier/ (best_model.pth, class_names.json,
  training_history.json, metrics.json, training_curves.png, confusion_matrix.png).
- Comprehensive test evaluation with precision, recall, F1-scores, classification
  report, and confusion matrix heatmap.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("soil_cnn")

# Six target soil classes in alphabetical order
DEFAULT_CLASSES: List[str] = [
    "alluvial_soil",
    "black_soil",
    "clay_soil",
    "laterite_soil",
    "red_soil",
    "sandy_soil",
]

# ImageNet normalization standard statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    """Ensure fully reproducible training by fixing seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Reproducible random seed set to: {seed}")


class RobustSoilDataset(Dataset):
    """
    Soil image dataset with robust error handling for damaged or corrupted files.
    """

    def __init__(
        self,
        root_dir: str | Path,
        classes: List[str] = DEFAULT_CLASSES,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for cls_name in self.classes:
            cls_folder = self.root_dir / cls_name
            if not cls_folder.is_dir():
                logger.warning(f"Class folder does not exist: {cls_folder}")
                continue
            cls_idx = self.class_to_idx[cls_name]
            for file_path in sorted(cls_folder.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                    self.samples.append((file_path, cls_idx))

        if not self.samples:
            raise ValueError(f"No valid images found in directory: {self.root_dir}")

        logger.info(
            f"Indexed {len(self.samples)} images from {self.root_dir} across {len(self.classes)} classes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[index]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    tensor_img = self.transform(img)
                else:
                    tensor_img = transforms.ToTensor()(img)
                return tensor_img, target
        except Exception as exc:
            logger.warning(
                f"Corrupt or unreadable image skipped at {img_path}: {exc}. Recovering gracefully."
            )
            # Gracefully recover by falling back to the next valid sample
            fallback_idx = (index + 1) % len(self.samples)
            return self.__getitem__(fallback_idx)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return training (with data augmentation) and validation/test (deterministic) transforms.
    Augmentation is strictly isolated to the training split.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def compute_class_weights(
    train_dataset: RobustSoilDataset,
    num_classes: int = len(DEFAULT_CLASSES),
) -> torch.Tensor:
    """
    Calculate balanced class weights to compensate for class frequency disparities:
        w_c = total_samples / (num_classes * count_c)
    """
    class_counts = [0] * num_classes
    for _, target in train_dataset.samples:
        class_counts[target] += 1

    total_samples = sum(class_counts)
    weights = []
    for count in class_counts:
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Computed training class distribution & balanced CrossEntropy weights:")
    for cls_name, count, w in zip(train_dataset.classes, class_counts, weights):
        logger.info(f"  [{cls_name:15s}] count = {count:4d} | loss weight = {w:.4f}")

    return weights_tensor


def build_model(
    num_classes: int = len(DEFAULT_CLASSES),
    pretrained: bool = True,
) -> nn.Module:
    """
    Construct EfficientNet-B0 with ImageNet pretrained weights and adapted classification head.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # In EfficientNet-B0, model.classifier is:
    # Sequential(
    #   (0): Dropout(p=0.2, inplace=True),
    #   (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    logger.info(
        f"Built EfficientNet-B0 (pretrained={pretrained}). Replaced head: Linear({in_features} -> {num_classes})."
    )
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch across all batches in the dataloader."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate loss, accuracy, and macro F1 on the validation split."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = float((np.array(all_preds) == np.array(all_targets)).mean()) if total > 0 else 0.0
    _, _, val_macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return val_loss, val_acc, float(val_macro_f1)


def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Evaluate the model on the test split.
    Calculates accuracy, precision, recall, F1 (macro and weighted),
    classification report, and confusion matrix.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    test_loss = running_loss / total if (criterion is not None and total > 0) else None
    test_acc = float((np.array(all_preds) == np.array(all_targets)).mean()) if total > 0 else 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))

    results: Dict[str, Any] = {
        "test_loss": test_loss,
        "accuracy": test_acc,
        "macro_avg": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(p_weight),
            "recall": float(r_weight),
            "f1_score": float(f1_weight),
        },
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": total,
    }

    return results


def plot_and_save_training_curves(
    history: Dict[str, List[Any]],
    output_path: str | Path,
    best_epoch: Optional[int] = None,
) -> None:
    """Plot and save loss and accuracy curves for training and validation."""
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1F77B4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#FF7F0E", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax1.axvline(x=best_epoch, color="#2CA02C", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_accuracy"]], label="Train Accuracy", color="#1F77B4", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_accuracy"]], label="Val Accuracy", color="#FF7F0E", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax2.axvline(x=best_epoch, color="#2CA02C", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved training curves figure to: {output_path}")


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
) -> None:
    """Plot and save confusion matrix as a high-resolution heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Soil Classification Confusion Matrix (Test Split)",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {output_path}")


def run_smoke_test(data_dir: str | Path = "data/processed") -> bool:
    """
    Perform a rapid smoke test verifying data loading, model instantiation,
    loss calculation, and gradient backward propagation without running full training.
    """
    logger.info("=" * 60)
    logger.info("RUNNING SOIL CLASSIFIER SMOKE TEST")
    logger.info("=" * 60)

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for d in (train_dir, val_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required dataset directory: {d}")

    train_tf, eval_tf = get_transforms()

    logger.info("Validating dataset splits and classes...")
    train_ds = RobustSoilDataset(train_dir, DEFAULT_CLASSES, transform=train_tf)
    val_ds = RobustSoilDataset(val_dir, DEFAULT_CLASSES, transform=eval_tf)
    test_ds = RobustSoilDataset(test_dir, DEFAULT_CLASSES, transform=eval_tf)

    assert len(train_ds) > 0, "Train dataset is empty!"
    assert len(val_ds) > 0, "Validation dataset is empty!"
    assert len(test_ds) > 0, "Test dataset is empty!"

    # Test batch with batch size 4
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    images, labels = next(iter(train_loader))

    logger.info(f"[Check] Train batch tensor shape: {images.shape} (Expected: [4, 3, 224, 224])")
    assert images.shape == (4, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected label shape: {labels.shape}"

    # Verify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Check] Active computation device: {device} (CUDA available: {torch.cuda.is_available()})")

    # Build model and verify architecture
    model = build_model(num_classes=len(DEFAULT_CLASSES), pretrained=True)
    model.to(device)

    # Class weights and criterion
    weights = compute_class_weights(train_ds, len(DEFAULT_CLASSES))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Forward & Backward Pass
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    logger.info(f"[Check] Model prediction logits shape: {outputs.shape} (Expected: [4, 6])")
    assert outputs.shape == (4, 6), f"Unexpected output shape: {outputs.shape}"

    loss = criterion(outputs, labels)
    loss_val = float(loss.item())
    logger.info(f"[Check] CrossEntropyLoss value: {loss_val:.4f}")
    assert not torch.isnan(loss), "Loss evaluated to NaN!"
    assert loss_val > 0.0, "Loss must be positive!"

    loss.backward()
    optimizer.step()
    logger.info("[Check] Backward pass & optimizer step succeeded.")

    logger.info("=" * 60)
    logger.info("SMOKE TEST COMPLETE: Pipeline, transforms, and model are fully operational!")
    logger.info("=" * 60)
    return True


def train(
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = "models/soil_classifier",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.0001,
    patience: int = 4,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Main training execution function.
    Trains EfficientNet-B0 with early stopping, saves best checkpoint and evaluation artifacts.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = get_transforms()

    # Load splits (test split is strictly isolated from training/validation)
    train_ds = RobustSoilDataset(data_dir / "train", DEFAULT_CLASSES, transform=train_tf)
    val_ds = RobustSoilDataset(data_dir / "val", DEFAULT_CLASSES, transform=eval_tf)
    test_ds = RobustSoilDataset(data_dir / "test", DEFAULT_CLASSES, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class-weighted loss
    class_weights = compute_class_weights(train_ds, len(DEFAULT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # EfficientNet-B0 Model
    model = build_model(num_classes=len(DEFAULT_CLASSES), pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = output_dir / "best_model.pth"

    logger.info(f"Beginning training for up to {epochs} epochs (Early Stopping Patience = {patience})...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
        epoch_dur = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_f1:.4f}"
        )

        # Validation monitoring & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "class_names": DEFAULT_CLASSES,
                "class_to_idx": train_ds.class_to_idx,
                "arch": "efficientnet_b0",
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  --> Validation loss did not improve ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time/60:.2f} minutes. Best Epoch: {best_epoch}")

    # 1. Save Class Names metadata
    class_meta_path = output_dir / "class_names.json"
    with open(class_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": DEFAULT_CLASSES,
                "class_to_idx": train_ds.class_to_idx,
                "num_classes": len(DEFAULT_CLASSES),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved class metadata to {class_meta_path}")

    # 2. Save Training History
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # 3. Generate & Save Loss/Accuracy Curves
    curves_path = output_dir / "training_curves.png"
    plot_and_save_training_curves(history, curves_path, best_epoch=best_epoch)

    # 4. Strict Final Evaluation on Untouched Test Split using Best Model Checkpoint
    logger.info("=" * 60)
    logger.info("EVALUATING BEST MODEL ON UNTOUCHED TEST SET")
    logger.info("=" * 60)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_results = evaluate_test(
        model=model,
        test_loader=test_loader,
        class_names=DEFAULT_CLASSES,
        device=device,
        criterion=criterion,
    )
    test_results["best_epoch"] = best_epoch
    test_results["total_training_time_seconds"] = total_training_time

    # 5. Save Test Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # 6. Generate & Save Confusion Matrix Heatmap
    cm_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(np.array(test_results["confusion_matrix"]), DEFAULT_CLASSES, cm_path)

    # Log summary
    logger.info("Final Test Performance Summary:")
    logger.info(f"  Test Accuracy      : {test_results['accuracy']*100:.2f}%")
    logger.info(f"  Macro Precision    : {test_results['macro_avg']['precision']:.4f}")
    logger.info(f"  Macro Recall       : {test_results['macro_avg']['recall']:.4f}")
    logger.info(f"  Macro F1-Score     : {test_results['macro_avg']['f1_score']:.4f}")
    logger.info(f"  Weighted F1-Score  : {test_results['weighted_avg']['f1_score']:.4f}")
    logger.info("\nClassification Report:\n" + test_results["classification_report_text"])

    return test_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoilSense AI - Train EfficientNet-B0 Soil Classification CNN"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed dataset directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/soil_classifier",
        help="Directory where model checkpoints, metrics, and plots will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and evaluation (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for Adam optimizer (default: 0.0001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience in epochs (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test to verify data loading, model initialization, and backward pass without full training.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate an existing best_model.pth checkpoint on the test split without retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        run_smoke_test(data_dir=args.data_dir)
        sys.exit(0)

    if args.evaluate_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(args.output_dir) / "best_model.pth"
        if not ckpt_path.is_file():
            logger.error(f"Checkpoint not found at: {ckpt_path}")
            sys.exit(1)

        _, eval_tf = get_transforms()
        test_ds = RobustSoilDataset(Path(args.data_dir) / "test", DEFAULT_CLASSES, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = build_model(num_classes=len(DEFAULT_CLASSES), pretrained=False).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        results = evaluate_test(model, test_loader, DEFAULT_CLASSES, device)
        print("\nTest Evaluation Results:")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(results["classification_report_text"])
        sys.exit(0)

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

