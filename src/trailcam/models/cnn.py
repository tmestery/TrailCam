"""Convolutional neural network for species classification from raw pixels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from trailcam.eval.metrics import compute_metrics


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int = 224, train: bool = False):
    """Resize/normalize transforms; light flip augmentation on train."""
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_cnn(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet-18 classifier for ``num_classes`` species."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    *,
    device: torch.device | None = None,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train with Adam; early-stop on validation macro-F1; save best weights."""
    device = device or get_device()
    model = model.to(device)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_tensor = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = float(w)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        y_true, y_pred = predict_loader(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred)
        train_loss = running_loss / max(n_batches, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch}: loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1 + 1e-6:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val macro-F1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_macro_f1": best_val_f1,
        "epochs_ran": len(history),
        "history": history,
        "device": str(device),
    }
