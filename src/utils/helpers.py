"""Utility functions: seeding, config loading, checkpointing."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file into a dictionary."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    state: dict,
    save_dir: str | Path,
    filename: str,
) -> Path:
    """Save a training checkpoint.

    Args:
        state: dict containing at least 'model_state_dict' and 'epoch'.
        save_dir: directory where the checkpoint will be saved.
        filename: name of the checkpoint file (e.g. 'epoch_50.pth').

    Returns:
        Path to the saved checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> dict:
    """Load a checkpoint from disk."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)