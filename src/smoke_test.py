"""
Smoke-test for WildfireDataset + WildfireConvLSTM.

Run from project root:
    python src/smoke_test.py

Checks:
    1. Dataset loads without error
    2. __len__ and __getitem__ work correctly
    3. Shapes of x and target are correct
    4. Model forward pass produces expected output shape
    5. Loss is computable
    6. Backward pass + optimizer step runs without error
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset  import WildfireDataset
from src.convlstm import WildfireConvLSTM

SEP = "=" * 60


def check(label: str) -> None:
    print(f"    [OK] {label}")


def main() -> None:
    print(SEP)
    print("  WildfireDataset + WildfireConvLSTM Smoke Test")
    print(SEP)

    DATASET_ROOT = Path("dataset")
    WINDOW_LEN   = 20
    BATCH_SIZE   = 2

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1] Loading WildfireDataset (train split)...")
    ds = WildfireDataset(
        split_json = DATASET_ROOT / "metadata" / "train.json",
        norm_json  = DATASET_ROOT / "normalization.json",
        window_len = WINDOW_LEN,
        sim_dir    = DATASET_ROOT / "simulations",
    )
    n = len(ds)
    print(f"    Dataset size (total windows): {n:,}")
    assert n > 0, "Dataset is empty!"
    check(f"Dataset loaded: {n:,} windows")

    # ── 2. Single sample ───────────────────────────────────────────────────────
    print("\n[2] Fetching single sample (index 0)...")
    x0, y0 = ds[0]
    print(f"    x.shape  = {tuple(x0.shape)}  (expected [{WINDOW_LEN}, 10, H, W])")
    print(f"    y.shape  = {tuple(y0.shape)}  (expected [H, W])")
    assert x0.ndim == 4,               f"Expected x to be 4D, got {x0.ndim}D"
    assert y0.ndim == 2,               f"Expected y to be 2D, got {y0.ndim}D"
    assert x0.shape[0] == WINDOW_LEN,  f"Window length mismatch: {x0.shape[0]} vs {WINDOW_LEN}"
    assert x0.shape[1] == 10,          f"Expected 10 channels, got {x0.shape[1]}"
    check("Sample shapes correct")

    # ── 3. Target range ────────────────────────────────────────────────────────
    print("\n[3] Checking target value range...")
    assert y0.min() >= 0.0 and y0.max() <= 1.0, \
        f"Target out of [0,1]: min={y0.min()}, max={y0.max()}"
    fire_frac = y0.mean().item()
    print(f"    Fire fraction in sample: {fire_frac:.2%}")
    check("Target values in [0, 1]")

    # ── 4. Mini-batch ──────────────────────────────────────────────────────────
    print("\n[4] Building mini-batch...")
    xs = torch.stack([ds[i][0] for i in range(BATCH_SIZE)])  # (B, T, C, H, W)
    ys = torch.stack([ds[i][1] for i in range(BATCH_SIZE)])  # (B, H, W)
    print(f"    x batch shape: {tuple(xs.shape)}")
    print(f"    y batch shape: {tuple(ys.shape)}")
    check("Mini-batch stacked successfully")

    # ── 5. Model init ──────────────────────────────────────────────────────────
    print("\n[5] Initializing WildfireConvLSTM...")
    _, T, C, H, W = xs.shape
    model = WildfireConvLSTM(
        in_channels   = C,
        hidden_dims   = [64, 64],
        kernel_size   = 3,
        proj_channels = 32,
        dropout       = 0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {n_params:,}")
    check("Model initialised")

    # ── 6. Forward pass ────────────────────────────────────────────────────────
    print("\n[6] Running model forward pass (eval)...")
    model.eval()
    with torch.no_grad():
        pred = model(xs)
    print(f"    Output shape: {tuple(pred.shape)}  (expected [{BATCH_SIZE}, {H}, {W}])")
    assert pred.shape == (BATCH_SIZE, H, W), f"Shape mismatch: {pred.shape}"
    assert pred.min() >= 0.0 and pred.max() <= 1.0, \
        f"Output out of [0,1]: min={pred.min():.4f}, max={pred.max():.4f}"
    print(f"    Output range: [{pred.min():.4f}, {pred.max():.4f}]")
    check("Forward pass shape and range correct")

    # ── 7. Loss computation ───────────────────────────────────────────────────
    print("\n[7] Computing BCE loss...")
    model.train()
    pred_train = model(xs)
    loss = F.binary_cross_entropy(pred_train, ys)
    print(f"    Loss value: {loss.item():.4f}")
    assert loss.item() > 0.0, "Loss is zero — something is wrong"
    check("Loss computed successfully")

    # ── 8. Backward + optimizer step ──────────────────────────────────────────
    print("\n[8] Testing backward pass + optimizer step...")
    optimizer = AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad()
    loss.backward()
    # check gradients exist
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    if no_grad:
        print(f"    WARNING: {len(no_grad)} params have no grad: {no_grad[:3]}")
    else:
        print(f"    All parameter gradients computed")
    optimizer.step()
    check("Backward + optimizer step succeeded")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SMOKE TEST PASSED")
    print(f"  Dataset windows : {n:,}")
    print(f"  Input shape     : {tuple(xs.shape)}")
    print(f"  Output shape    : {tuple(pred.shape)}")
    print(f"  Model params    : {n_params:,}")
    print(f"  Loss (one step) : {loss.item():.4f}")
    print(SEP)


if __name__ == "__main__":
    main()
