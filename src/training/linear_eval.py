"""Linear evaluation protocol for self-supervised representations.

Standard protocol to measure how good SSL-learned features are:
    1. Freeze the pretrained encoder (no gradient updates to it).
    2. Train ONLY a linear classifier on top of frozen features.
    3. Report test accuracy.

If the encoder learned meaningful representations during SSL pretraining,
a simple linear classifier should perform well — no need for a deep MLP.

This is a much stricter test than end-to-end fine-tuning: it directly
measures the linear separability of classes in the feature space.
"""

import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import wandb

from src.models.simclr import SimCLRModel


class LinearClassifier(nn.Module):
    """A single linear layer used as classifier on top of frozen features."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


@torch.no_grad()
def extract_features(
    encoder: SimCLRModel,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from a frozen encoder for all samples in a loader.

    Returns:
        features: (N, feature_dim) tensor.
        labels: (N,) tensor.
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = encoder.encode(images)  # (B, feature_dim)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def train_one_epoch_linear(
    classifier: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: str,
) -> dict:
    """One epoch of linear classifier training on pre-extracted features."""
    classifier.train()
    n_samples = features.shape[0]
    perm = torch.randperm(n_samples)

    running_loss = 0.0
    running_correct = 0
    num_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_idx = perm[i : i + batch_size]
        batch_features = features[batch_idx].to(device, non_blocking=True)
        batch_labels = labels[batch_idx].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        num_batches += 1

    return {
        "loss": running_loss / num_batches,
        "acc": running_correct / n_samples,
    }


@torch.no_grad()
def evaluate(
    classifier: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict:
    """Evaluate the linear classifier on a set of pre-extracted features."""
    classifier.eval()
    n_samples = features.shape[0]
    running_correct = 0

    for i in range(0, n_samples, batch_size):
        batch_features = features[i : i + batch_size].to(device, non_blocking=True)
        batch_labels = labels[i : i + batch_size].to(device, non_blocking=True)
        logits = classifier(batch_features)
        running_correct += (logits.argmax(dim=1) == batch_labels).sum().item()

    return {"acc": running_correct / n_samples}


def linear_evaluation(
    config: dict,
    encoder: SimCLRModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
) -> dict:
    """Full linear evaluation protocol.

    Steps:
        1. Extract features for train and test sets ONCE (encoder frozen).
        2. Train a linear classifier on train features for N epochs.
        3. Report best test accuracy.
    """
    device = config["experiment"]["device"]
    epochs = config["linear_eval"]["epochs"]
    batch_size = config["linear_eval"]["batch_size"]
    lr = config["linear_eval"]["learning_rate"]
    weight_decay = config["linear_eval"]["weight_decay"]

    encoder = encoder.to(device)

    print(f"\n{'=' * 60}")
    print("Extracting features from frozen encoder...")
    print(f"{'=' * 60}")

    t0 = time.time()
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    print(
        f"Extracted in {time.time() - t0:.1f}s | "
        f"train: {train_features.shape}, test: {test_features.shape}"
    )

    # Build linear classifier
    feature_dim = train_features.shape[1]
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'=' * 60}")
    print(f"Training linear classifier for {epochs} epochs")
    print(f"{'=' * 60}\n")

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_linear(
            classifier=classifier,
            features=train_features,
            labels=train_labels,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            device=device,
        )
        test_metrics = evaluate(
            classifier=classifier,
            features=test_features,
            labels=test_labels,
            batch_size=batch_size,
            device=device,
        )
        scheduler.step()

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]

        wandb.log({
            "linear_eval/train_loss": train_metrics["loss"],
            "linear_eval/train_acc": train_metrics["acc"],
            "linear_eval/test_acc": test_metrics["acc"],
            "linear_eval/best_test_acc": best_test_acc,
            "linear_eval/epoch": epoch,
        })

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss: {train_metrics['loss']:.4f} | "
                f"train_acc: {train_metrics['acc']:.4f} | "
                f"test_acc: {test_metrics['acc']:.4f} | "
                f"best: {best_test_acc:.4f}"
            )

    print(f"\nLinear evaluation complete. Best test accuracy: {best_test_acc:.4f}")

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": test_metrics["acc"],
    }