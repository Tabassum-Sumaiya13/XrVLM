"""
Two-stage training pipeline for EfficientNet-B4 on NIH CXR-14.

Stage 1: backbone frozen, head-only warm-up
Stage 2: full network fine-tuned at lower LR with ReduceLROnPlateau

After training, temperature scaling is tuned on the validation set.

Usage:
    python train.py
    python train.py --epochs-stage1 3 --epochs-stage2 12 --batch 16
    python train.py --sample 500         # quick test with 500 images
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CKPT_DIR, OUTPUT_DIR, FIGURES_DIR,
    STAGE1_LR_WARMUP, STAGE1_LR_FINETUNE,
    STAGE1_EPOCHS_WARMUP, STAGE1_EPOCHS_FINETUNE,
    STAGE1_DROPOUT, TEMPERATURE_INIT, BATCH_SIZE,
    get_device,
)
from dataset import build_dataloaders
from stage1_model import (
    build_stage1_model, freeze_backbone, unfreeze_backbone,
    tune_temperature,
)

LOSS_LOG_PATH = OUTPUT_DIR / "loss_log.csv"


# ── device ────────────────────────────────────────────────────────────────────
def _get_device() -> torch.device:
    device = get_device()
    print(f"Training on device: {device}")
    return device


# ── one epoch of training ─────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    running_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            avg = running_loss / (batch_idx + 1)
            print(
                f"  Epoch {epoch}  [{batch_idx+1:>4}/{n_batches}]  "
                f"train loss: {avg:.4f}",
                end="\r",
            )

    print()
    return running_loss / max(n_batches, 1)


# ── validation pass ───────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return average validation loss."""
    model.eval()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


# ── checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, path, device):
    """Save model + optimiser state."""
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "temperature": model.temp_scaler.temperature.item(),
        },
        path,
    )


def load_checkpoint(model, path, device):
    """Load a checkpoint; return the metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ── loss logger ───────────────────────────────────────────────────────────────
class LossLogger:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "stage", "train_loss", "val_loss", "elapsed_sec", "lr"]
            )

    def log(self, epoch, stage, train_loss, val_loss, elapsed, lr):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, stage, f"{train_loss:.6f}",
                f"{val_loss:.6f}", f"{elapsed:.1f}", f"{lr:.2e}",
            ])


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════
def train(
    epochs_stage1: int = STAGE1_EPOCHS_WARMUP,
    epochs_stage2: int = STAGE1_EPOCHS_FINETUNE,
    batch_size: int = BATCH_SIZE,
    lr_stage1: float = STAGE1_LR_WARMUP,
    lr_stage2: float = STAGE1_LR_FINETUNE,
    num_workers: int = 0,
    dropout: float = STAGE1_DROPOUT,
    seed: int = 42,
    sample_n: int = None,
) -> Path:
    """
    Run the full two-stage training loop and save checkpoints.
    Returns path to best checkpoint.
    """
    torch.manual_seed(seed)
    device = _get_device()

    # ── data ──────────────────────────────────────────────────────────────
    print("\nBuilding DataLoaders …")
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        sample_n=sample_n,
    )

    if len(train_loader) == 0:
        print("ERROR: No training data found. Check your data directory.")
        return None

    # ── model ─────────────────────────────────────────────────────────────
    print("Building EfficientNet-B4 model …")
    model = build_stage1_model(dropout=dropout).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(device)
    )

    logger = LossLogger(LOSS_LOG_PATH)
    best_val_loss = float("inf")
    epoch_global = 0

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — head-only warm-up
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("STAGE 1 — Head warm-up (backbone frozen)")
    print("═" * 60)

    freeze_backbone(model)

    optimizer_s1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1, weight_decay=1e-5,
    )

    for epoch in range(1, epochs_stage1 + 1):
        epoch_global += 1
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, device, epoch_global
        )
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr = optimizer_s1.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch_global:>3} [stage1]  "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )
        logger.log(epoch_global, "stage1", train_loss, val_loss, elapsed, lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer_s1, epoch_global, val_loss,
                CKPT_DIR / "best_model.pth", device,
            )
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — full fine-tune
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("STAGE 2 — Full fine-tune (backbone unfrozen)")
    print("═" * 60)

    unfreeze_backbone(model)

    optimizer_s2 = torch.optim.Adam(
        model.parameters(), lr=lr_stage2, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2, mode="min", factor=0.5, patience=2,
    )

    for epoch in range(1, epochs_stage2 + 1):
        epoch_global += 1
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, device, epoch_global
        )
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        lr = optimizer_s2.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch_global:>3} [stage2]  "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"lr: {lr:.2e}  ({elapsed:.0f}s)"
        )
        logger.log(epoch_global, "stage2", train_loss, val_loss, elapsed, lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer_s2, epoch_global, val_loss,
                CKPT_DIR / "best_model.pth", device,
            )
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")

        save_checkpoint(
            model, optimizer_s2, epoch_global, val_loss,
            CKPT_DIR / "last_model.pth", device,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Temperature calibration
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("TEMPERATURE CALIBRATION")
    print("═" * 60)

    # Reload best weights
    best_path = CKPT_DIR / "best_model.pth"
    load_checkpoint(model, best_path, device)
    model.to(device)

    # Tune temperature on validation set
    final_T = tune_temperature(model, val_loader, device)
    print(f"Calibrated temperature: {final_T:.4f}")

    # Re-save with calibrated temperature
    save_checkpoint(
        model, optimizer_s2, epoch_global, best_val_loss,
        CKPT_DIR / "best_model_calibrated.pth", device,
    )

    print(f"\n{'═' * 60}")
    print(f"Training complete!")
    print(f"Best val loss     : {best_val_loss:.4f}")
    print(f"Temperature       : {final_T:.4f}")
    print(f"Best checkpoint   : {best_path}")
    print(f"Calibrated ckpt   : {CKPT_DIR / 'best_model_calibrated.pth'}")
    print(f"Loss log          : {LOSS_LOG_PATH}")
    print(f"{'═' * 60}")

    return best_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B4 on NIH CXR-14")
    p.add_argument("--epochs-stage1", type=int, default=STAGE1_EPOCHS_WARMUP)
    p.add_argument("--epochs-stage2", type=int, default=STAGE1_EPOCHS_FINETUNE)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr-stage1", type=float, default=STAGE1_LR_WARMUP)
    p.add_argument("--lr-stage2", type=float, default=STAGE1_LR_FINETUNE)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=STAGE1_DROPOUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=None,
                   help="Use only N images for quick testing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        num_workers=args.workers,
        dropout=args.dropout,
        seed=args.seed,
        sample_n=args.sample,
    )
