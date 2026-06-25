"""
Single-batch CPU profiler.
Uses a synthetic in-memory tensor — no disk, no DataLoader.
Safe to run on constrained hardware.

Usage:
    python src/profile_batch.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.convlstm import WildfireConvLSTM, ConvLSTMCell


def time_it(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<45} {elapsed*1000:>8.1f} ms")
    return result


def profile_cell(B, C_in, hidden, H, W, T):
    """Time a single ConvLSTMCell for T steps."""
    cell = ConvLSTMCell(C_in, hidden, kernel_size=3)
    cell.eval()
    x = torch.randn(B, C_in, H, W)
    state = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(T):
            state = cell(x, state)
    return (time.perf_counter() - t0) * 1000


print("=" * 60)
print("  ConvLSTM Single-Batch CPU Profile")
print("=" * 60)

B   = 4     # batch size
T   = 20    # sequence length
C   = 10    # input channels
H   = 100   # grid height
W   = 100   # grid width

x      = torch.randn(B, T, C, H, W)
target = torch.randint(0, 2, (B, H, W)).float()

print(f"\nConfig: B={B}, T={T}, C={C}, H={H}, W={W}")
print(f"Input tensor RAM: {x.numel()*4/1e6:.1f} MB\n")

# ── Model architecture breakdown ────────────────────────────────────────────
print("── Component timing (no_grad, eval mode) ──────────────────────────")

model = WildfireConvLSTM(
    in_channels   = C,
    hidden_dims   = [64, 64],
    kernel_size   = 3,
    proj_channels = 32,
    dropout       = 0.0,
)
model.eval()

# 1. Input projection only
x_flat = x.view(B * T, C, H, W)
time_it("Input projection (Conv2d 10→32, BxT frames)",
        lambda: model.input_proj(x_flat))

# 2. ConvLSTM layer 1: input=32 channels, hidden=64
ms_l1 = profile_cell(B, 32, 64, H, W, T)
print(f"  {'ConvLSTM layer 1 (32→64, T=20 steps)':<45} {ms_l1:>8.1f} ms")

# 3. ConvLSTM layer 2: input=64 channels, hidden=64
ms_l2 = profile_cell(B, 64, 64, H, W, T)
print(f"  {'ConvLSTM layer 2 (64→64, T=20 steps)':<45} {ms_l2:>8.1f} ms")

# 4. Prediction head
h_dummy = torch.randn(B, 64, H, W)
time_it("Prediction head (Conv2d 64→32→1 + sigmoid)",
        lambda: model.head(h_dummy))

# ── Full forward pass ────────────────────────────────────────────────────────
print()
print("── Full forward pass (eval, no_grad) ──────────────────────────────")
with torch.no_grad():
    pred_warmup = model(x)   # warmup

fwd_ms = time_it("Forward pass (full model)",
                  lambda: model(x))

# ── Forward + backward ───────────────────────────────────────────────────────
print()
print("── Forward + backward (train mode) ────────────────────────────────")
model.train()
criterion = nn.BCELoss()

t0 = time.perf_counter()
pred  = model(x)
loss  = criterion(pred, target)
fwd_time = time.perf_counter() - t0

t1 = time.perf_counter()
loss.backward()
bwd_time = time.perf_counter() - t1

total_ms = (fwd_time + bwd_time) * 1000
print(f"  {'Forward':<45} {fwd_time*1000:>8.1f} ms")
print(f"  {'Backward':<45} {bwd_time*1000:>8.1f} ms")
print(f"  {'Total (fwd + bwd)':<45} {total_ms:>8.1f} ms")

# ── Projections ──────────────────────────────────────────────────────────────
print()
print("── Epoch time projections ──────────────────────────────────────────")
batches_stride10 = 3308
est_epoch_min    = total_ms * batches_stride10 / 1000 / 60
print(f"  Seconds per batch (fwd+bwd):        {total_ms/1000:.2f}s")
print(f"  Batches per epoch (stride=10, B=4): {batches_stride10}")
print(f"  Estimated epoch time:               {est_epoch_min:.1f} minutes")

# ── Reduced-model suggestion ─────────────────────────────────────────────────
print()
print("── Reduced model (hidden=[32,32], proj=16) ─────────────────────────")
small = WildfireConvLSTM(
    in_channels   = C,
    hidden_dims   = [32, 32],
    kernel_size   = 3,
    proj_channels = 16,
    dropout       = 0.0,
)
small.train()
x2 = torch.randn(B, T, C, H, W)
t2 = torch.randint(0, 2, (B, H, W)).float()
t0 = time.perf_counter()
p2 = small(x2)
l2 = criterion(p2, t2)
l2.backward()
small_ms = (time.perf_counter() - t0) * 1000
params_small = sum(p.numel() for p in small.parameters() if p.requires_grad)
est_small = small_ms * batches_stride10 / 1000 / 60
print(f"  Params:                 {params_small:,}")
print(f"  Time per batch:         {small_ms:.1f} ms")
print(f"  Estimated epoch time:   {est_small:.1f} minutes")

print()
print("=" * 60)
print("  Profile complete — no training was run")
print("=" * 60)
