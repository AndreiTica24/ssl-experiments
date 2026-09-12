"""SimCLR pretraining loop.

Trains a SimCLR model on unlabeled images using NT-Xent contrastive loss.
Uses mixed precision (AMP) to fit larger batches in limited VRAM.
Logs metrics to Weights & Biases and saves periodic checkpoints.
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from src.models.simclr import SimCLRModel
from src.training.losses import NTXentLoss
from src.utils.helpers import save_checkpoint


def build_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> LambdaLR:
    """Cosine schedule with linear warmup, applied per-step (not per-epoch).

    LR grows linearly from 0 to base_lr over the first `warmup_epochs`,
    then decays following a cosine curve to 0 by `total_epochs`.
    """
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: SimCLRModel,
    loader: DataLoader,
    loss_fn: NTXentLoss,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: str,
    epoch: int,
    log_every: int,
    use_amp: bool,
) -> dict:
    """Train one epoch of SimCLR pretraining.

    Returns:
        dict with 'loss' (mean over epoch), 'lr' (final LR), 'time' (seconds).
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for step, (view1, view2) in enumerate(loader):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass: computes in FP16 where safe, FP32 where needed
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = loss_fn(z1, z2)

        # Scaled backward pass (prevents FP16 gradient underflow)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        running_loss += loss_val
        num_batches += 1

        # Log to wandb every `log_every` steps
        if step % log_every == 0:
            wandb.log({
                "train/loss_step": loss_val,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": epoch * len(loader) + step,
            })

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / num_batches

    return {
        "loss": epoch_loss,
        "lr": scheduler.get_last_lr()[0],
        "time": epoch_time,
    }


def pretrain_simclr(
    config: dict,
    model: SimCLRModel,
    loader: DataLoader,
) -> SimCLRModel:
    """Full SimCLR pretraining routine.

    Args:
        config: parsed YAML config dictionary.
        model: SimCLR model (encoder + projection head).
        loader: pretraining DataLoader (returns pairs of augmented views).

    Returns:
        Trained model (already moved to device).
    """
    device = config["experiment"]["device"]
    epochs = config["training"]["epochs"]
    use_amp = config["training"]["use_amp"]
    save_dir = Path(config["checkpointing"]["save_dir"])
    save_every = config["checkpointing"]["save_every"]

    model = model.to(device)

    # Optimizer: Adam is fine for CIFAR-10 scale; SimCLR paper used LARS for ImageNet
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # LR schedule
    scheduler = build_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=epochs,
        warmup_epochs=config["scheduler"]["warmup_epochs"],
        steps_per_epoch=len(loader),
    )

    # Loss
    loss_fn = NTXentLoss(temperature=config["training"]["temperature"])

    # Mixed precision scaler (no-op if use_amp is False)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    print(f"\n{'=' * 60}")
    print(f"Starting SimCLR pretraining on {device}")
    print(f"Epochs: {epochs}, Batch size: {loader.batch_size}, AMP: {use_amp}")
    print(f"Steps per epoch: {len(loader)}, Total steps: {epochs * len(loader)}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            log_every=config["logging"]["log_every"],
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss: {metrics['loss']:.4f} | "
            f"lr: {metrics['lr']:.2e} | "
            f"time: {metrics['time']:.1f}s"
        )

        wandb.log({
            "train/loss_epoch": metrics["loss"],
            "train/lr_epoch": metrics["lr"],
            "epoch": epoch,
        })

        # Periodic checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": metrics["loss"],
                    "config": config,
                },
                save_dir=save_dir,
                filename=f"epoch_{epoch:03d}.pth",
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    print("\nPretraining complete.")
    return model