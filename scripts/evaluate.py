"""Standalone linear evaluation for a pretrained SimCLR checkpoint.

Usage:
    python scripts/evaluate.py \
        --config configs/simclr_cifar10.yaml \
        --checkpoint checkpoints/simclr_cifar10/epoch_200.pth

Useful when you want to:
    - Evaluate an intermediate checkpoint (e.g. epoch 100 vs epoch 200).
    - Re-run linear eval with different hyperparameters (LR, epochs).
    - Compare multiple pretrained models without re-training them.
"""

import argparse
import sys
from pathlib import Path

import torch

import wandb

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10_ssl import get_eval_loaders
from src.models.simclr import build_simclr_model
from src.training.linear_eval import linear_evaluation
from src.utils.helpers import load_checkpoint, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear evaluation of a pretrained SimCLR checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained SimCLR checkpoint (.pth file)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional wandb run name (defaults to 'eval_' + checkpoint filename)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (useful for quick tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Reproducibility
    set_seed(config["experiment"]["seed"])

    # Device check
    device = config["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n>>> Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}")

    # Rebuild model architecture and load pretrained weights
    print("\n>>> Rebuilding model and loading weights...")
    model = build_simclr_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Weights loaded successfully.")

    # Init wandb (optional)
    if not args.no_wandb:
        run_name = args.run_name or f"eval_{checkpoint_path.stem}"
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            name=run_name,
            config={**config, "checkpoint": str(checkpoint_path)},
            job_type="linear_eval",
        )
    else:
        # Dummy wandb context so linear_evaluation() logs don't crash
        wandb.init(mode="disabled")

    # Build eval loaders
    print("\n>>> Building evaluation dataloaders...")
    eval_train_loader, eval_test_loader = get_eval_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["linear_eval"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
    )

    # Run linear eval
    results = linear_evaluation(
        config=config,
        encoder=model,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        num_classes=10,
    )

    print(f"\n{'=' * 60}")
    print("Final results:")
    print(f"  Best test accuracy:  {results['best_test_acc']:.4f}")
    print(f"  Final test accuracy: {results['final_test_acc']:.4f}")
    print(f"{'=' * 60}")

    if not args.no_wandb:
        wandb.summary["best_linear_acc"] = results["best_test_acc"]
        wandb.summary["final_linear_acc"] = results["final_test_acc"]
        wandb.summary["checkpoint_epoch"] = checkpoint.get("epoch", -1)
        wandb.finish()


if __name__ == "__main__":
    main()