"""End-to-end SimCLR pipeline: pretraining + linear evaluation.

Usage:
    python scripts/train_simclr.py --config configs/simclr_cifar10.yaml

This script:
    1. Loads config and sets random seeds.
    2. Initializes Weights & Biases logging.
    3. Builds SimCLR model and pretraining dataloader.
    4. Runs SSL pretraining for the configured number of epochs.
    5. Runs linear evaluation on the frozen encoder.
    6. Logs final metrics and finishes the wandb run.
"""

import argparse
import sys
from pathlib import Path

import torch

import wandb

# Add project root to Python path so `from src.` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10_ssl import get_eval_loaders, get_pretrain_loader
from src.models.simclr import build_simclr_model
from src.training.linear_eval import linear_evaluation
from src.training.pretrain import pretrain_simclr
from src.utils.helpers import count_parameters, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simclr_cifar10.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional wandb run name (defaults to config experiment.name)",
    )
    parser.add_argument(
        "--skip-linear-eval",
        action="store_true",
        help="Skip linear evaluation after pretraining",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Reproducibility
    set_seed(config["experiment"]["seed"])

    # Sanity check: is CUDA actually available?
    device = config["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Init wandb
    run_name = args.run_name or config["experiment"]["name"]
    wandb.init(
        project=config["logging"]["wandb_project"],
        entity=config["logging"]["wandb_entity"],
        name=run_name,
        config=config,
    )

    # -------------------- Pretraining --------------------
    print("\n>>> Building pretraining dataloader...")
    pretrain_loader = get_pretrain_loader(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
    )
    print(f"Pretraining dataset size: {len(pretrain_loader.dataset)}")
    print(f"Batches per epoch: {len(pretrain_loader)}")

    print("\n>>> Building model...")
    model = build_simclr_model(config)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    model = pretrain_simclr(config=config, model=model, loader=pretrain_loader)

    # -------------------- Linear evaluation --------------------
    if not args.skip_linear_eval:
        print("\n>>> Building evaluation dataloaders...")
        eval_train_loader, eval_test_loader = get_eval_loaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["linear_eval"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
        )

        results = linear_evaluation(
            config=config,
            encoder=model,
            train_loader=eval_train_loader,
            test_loader=eval_test_loader,
            num_classes=10,
        )

        # Save final result summary
        wandb.summary["best_linear_acc"] = results["best_test_acc"]
        wandb.summary["final_linear_acc"] = results["final_test_acc"]

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()