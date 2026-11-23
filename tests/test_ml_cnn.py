"""Unit tests for the SoilSense AI CNN training module."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.train_cnn import (
    DEFAULT_CLASSES,
    build_model,
    get_transforms,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)


def test_classes_definition():
    assert len(DEFAULT_CLASSES) == 6
    expected = [
        "alluvial_soil",
        "black_soil",
        "clay_soil",
        "laterite_soil",
        "red_soil",
        "sandy_soil",
    ]
    assert DEFAULT_CLASSES == expected


def test_seed_reproducibility():
    set_seed(42)
    t1 = torch.randn(5)
    set_seed(42)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)


def test_transforms_configuration():
    train_tf, eval_tf = get_transforms()
    assert train_tf is not None
    assert eval_tf is not None
    # Training transform must have more steps (data augmentation) than eval transform
    assert len(train_tf.transforms) > len(eval_tf.transforms)


def test_model_architecture():
    model = build_model(num_classes=6, pretrained=False)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == 6

    # Test forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 6)


def test_training_and_validation_epoch():
    device = torch.device("cpu")
    model = build_model(num_classes=6, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dummy_inputs = torch.randn(8, 3, 224, 224)
    dummy_targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_inputs, dummy_targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

    val_loss, val_acc, val_f1 = validate_one_epoch(model, loader, criterion, device)
    assert val_loss > 0.0
    assert 0.0 <= val_acc <= 1.0
    assert 0.0 <= val_f1 <= 1.0
