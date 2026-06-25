"""
Training pipeline for WildfireConvLSTM.

Usage:
    python src/train.py

Key options (edit CONFIG dict below or pass via CLI):
    --epochs        number of epochs          (default 30)
    --batch_size    samples per batch         (default 8)
    --lr            initial learning rate     (default 3e-4)
    --window        sliding window length     (default 20)
    --checkpoint    directory for checkpoints (default models/)
    --focal         use Focal loss instead of BCE
    --device        cuda / cpu / mps          (default: auto-detect)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset  import build_loaders
from src.convlstm import WildfireConvLSTM


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Cross-Entropy on probability outputs (already sigmoided)."""
    return F.binary_cross_entropy(pred, target)


class FocalLoss(nn.Module):
    """
    Focal loss to handle extreme class imbalance (most cells unburned).
    alpha : weight for positive class
    gamma : focusing exponent
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pred  = pred.clamp(eps, 1.0 - eps)
        p_t   = torch.where(target == 1, pred, 1.0 - pred)
        alpha = torch.where(target == 1,
                            torch.full_like(pred, self.alpha),
                            torch.full_like(pred, 1.0 - self.alpha))
        loss  = -alpha * (1.0 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred: torch.Tensor,      # (B, H, W) probabilities
    target: torch.Tensor,    # (B, H, W) binary
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_bin = (pred >= threshold).float()
    TP = (pred_bin * target).sum().item()
    FP = (pred_bin * (1 - target)).sum().item()
    FN = ((1 - pred_bin) * target).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = TP / (TP + FP + FN + 1e-6)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_list[0].keys()
    return {k: sum(m[k] for m in metric_list) / len(metric_list) for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# Training / Validation loops
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:      WildfireConvLSTM,
    loader,
    criterion,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> tuple[float, dict[str, float]]:
    """Run one epoch. Returns (avg_loss, avg_metrics)."""
    model.train(is_train)
    total_loss   = 0.0
    metric_list  = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (x, target) in enumerate(loader):
            x      = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(x)                     # (B, H, W)
            loss = criterion(pred, target)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            metric_list.append(compute_metrics(pred.detach(), target))

            if is_train and (batch_idx % 20 == 0):
                print(
                    f"  batch {batch_idx:4d}/{len(loader)} | "
                    f"loss={loss.item():.4f} | "
                    f"iou={metric_list[-1]['iou']:.3f}"
                )

    avg_loss    = total_loss / len(loader)
    avg_metrics = aggregate_metrics(metric_list)
    return avg_loss, avg_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Building data loaders...")
    train_loader, val_loader, _ = build_loaders(
        dataset_root=args.dataset_root,
        window_len=args.window,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        stride=args.stride,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = WildfireConvLSTM(
        in_channels  = 10,
        hidden_dims  = [64, 64],
        kernel_size  = 3,
        proj_channels= 32,
        dropout      = 0.1,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion: nn.Module | any
    if args.focal:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        print("Loss: Focal Loss (alpha=0.75, gamma=2.0)")
    else:
        criterion = bce_loss
        print("Loss: Binary Cross-Entropy")

    # ── optimiser & scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── checkpointing setup ───────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = math.inf
    history: list[dict] = []
    start_epoch = 1

    # ── resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", [])
        best_val_loss = min((r["val_loss"] for r in history), default=math.inf)
        print(f"Resumed from epoch {ckpt['epoch']} | best val_loss so far: {best_val_loss:.4f}")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_iou={val_metrics['iou']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | "
            f"time={elapsed:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(record)

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":   val_loss,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

        # save latest checkpoint (for resume)
        torch.save(
            {
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":   val_loss,
                "history":    history,
            },
            ckpt_dir / "latest_model.pt",
        )

    # ── save history ──────────────────────────────────────────────────────────
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved → {history_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WildfireConvLSTM")
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--window",       type=int, default=20)
    p.add_argument("--checkpoint",   type=str, default="models/convlstm")
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--stride",       type=int, default=1,
                   help="Window stride (default 1). Use 10 on CPU to reduce I/O by ~10x.")
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--focal",        action="store_true",
                   help="Use Focal loss instead of BCE")
    p.add_argument("--resume",       type=str, default=None,
                   help="Path to latest_model.pt to resume training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── resume support ────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # checkpoint is loaded inside main after model is built
        # (handled via args.resume being passed through)

    main(args)
