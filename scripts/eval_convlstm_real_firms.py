#!/usr/bin/env python3
"""
Real-world generalization check: does the ConvLSTM checkpoint trained on the
synthetic cellular-automata simulator produce anything useful when fed a
REAL observed wildfire, with NO retraining/fine-tuning?

Data: NASA FIRMS MODIS Collection 6.1 global 24h active-fire CSV (public,
no auth), downloaded to /tmp/firms_test.csv. A real, currently-burning fire
complex in West Siberia (~65.5N, 76E) had two genuinely separate satellite
overpasses ~6h apart on the same day -- real observed "before" and "after"
fire footprints, not synthetic and not fabricated.

Honesty / limitation, stated up front and in the report:
  - Only channel 0 (fire state) is real observed data here.
  - Channels 1 (potential ROS) and 2 (fireline intensity) are quantities
    INTERNAL to this project's cellular-automata simulator -- they are not
    observable in any real dataset, full stop. They are filled with the
    training-set mean (normalizes to 0, a "climatological average" neutral
    filler), clearly logged as synthetic.
  - Channels 3/4/7 (vegetation density, slope, wind speed) would need a real
    land-cover / DEM / weather fetch for this exact region and are also
    filled with training-set means for the same reason (out of scope for
    this lightweight check).
  - Channels 5/6/8/9 (aspect cos/sin, wind dir cos/sin) are set to a neutral
    direction (1, 0).
This is therefore a PARTIAL generalization test (real fire geometry/history,
synthetic terrain/weather context) -- not a full real-world validation.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.convlstm import WildfireConvLSTM
from src.train import compute_metrics
from src.vision.convlstm_bridge import resolve_device
from src.vision.paths import DEFAULT_CONVLSTM_CHECKPOINT, DEFAULT_NORM_JSON

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "convlstm_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIRMS_CSV = Path("/tmp/firms_test.csv")

GRID = 64
LAT_MIN, LAT_MAX = 64.4, 67.1
LON_MIN, LON_MAX = 73.4, 78.9
EARLY_HOURS = set(range(0, 11))   # ~03:00-10:00 UTC overpass
LATE_HOURS = set(range(13, 16))   # ~13:00-15:00 UTC overpass, same day
WINDOW = 20

NORM_CHANNELS = {1: "potential_ros", 2: "fireline_intensity", 3: "vegetation_density", 4: "slope", 7: "wind_speed"}


def rasterize(points: list[tuple[float, float]]) -> np.ndarray:
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    for lat, lon in points:
        row = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID)
        col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID)
        if 0 <= row < GRID and 0 <= col < GRID:
            grid[GRID - 1 - row, col] = 1.0  # flip so north is up
    return grid


def load_real_fire_pair() -> tuple[np.ndarray, np.ndarray, int, int]:
    rows = list(csv.DictReader(open(FIRMS_CSV)))
    early_pts, late_pts = [], []
    for r in rows:
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue
        hour = int(r["acq_time"][:2])
        if hour in EARLY_HOURS:
            early_pts.append((lat, lon))
        elif hour in LATE_HOURS:
            late_pts.append((lat, lon))
    early_mask = rasterize(early_pts)
    late_mask = rasterize(late_pts)
    return early_mask, late_mask, len(early_pts), len(late_pts)


def build_input_sequence(fire_mask: np.ndarray, norm: dict) -> torch.Tensor:
    """(WINDOW, 10, GRID, GRID): real fire history (replicated, see limitation
    above) + training-mean-filled synthetic terrain/weather channels."""
    frame = np.zeros((10, GRID, GRID), dtype=np.float32)
    frame[0] = fire_mask  # real observed fire state
    for ch, key in NORM_CHANNELS.items():
        frame[ch] = norm[key]["mean"]  # -> normalizes to 0 below
    frame[5] = 1.0  # aspect cos (neutral, aspect=0)
    frame[6] = 0.0  # aspect sin
    frame[8] = 1.0  # wind dir cos (neutral)
    frame[9] = 0.0  # wind dir sin

    seq = np.repeat(frame[None, ...], WINDOW, axis=0)
    seq_t = torch.from_numpy(seq)
    for ch, key in NORM_CHANNELS.items():
        mean, std = norm[key]["mean"], norm[key]["std"] or 1.0
        seq_t[:, ch] = (seq_t[:, ch] - mean) / std
    return seq_t


def save_comparison(early, target, pred, n_early, n_late):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_bin = (pred >= 0.5).float().numpy()
    error = pred_bin - target

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(early, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title(f"REAL input (t)\nFIRMS morning overpass\n({int(early.sum())} fire cells)")
    axes[1].imshow(target, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title(f"REAL ground truth (t+~6h)\nFIRMS afternoon overpass\n({int(target.sum())} fire cells, raw pts: early={n_early} late={n_late})")
    axes[2].imshow(pred.numpy(), cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("ConvLSTM prediction\n(synthetic-trained, 0 fine-tuning)")
    im = axes[3].imshow(error, cmap="coolwarm", vmin=-1, vmax=1)
    axes[3].set_title("Error\n(blue=false neg, red=false pos)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.suptitle("West Siberia wildfire complex (~65.5N, 76E) -- REAL NASA FIRMS MODIS active-fire data, same UTC day", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "real_data_firms_siberia_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    if not FIRMS_CSV.exists():
        print("FIRMS CSV not found; download it first."); return

    early_mask, late_mask, n_early, n_late = load_real_fire_pair()
    print(f"Real FIRMS points in region: early={n_early}, late={n_late}")
    print(f"Rasterized ({GRID}x{GRID}): early fire cells={int(early_mask.sum())}, late fire cells={int(late_mask.sum())}")
    if early_mask.sum() < 3 or late_mask.sum() < 3:
        print("Too few points rasterized to a meaningful mask; aborting."); return

    norm = json.load(open(DEFAULT_NORM_JSON))
    device = resolve_device("auto")
    model = WildfireConvLSTM(in_channels=10, hidden_dims=[64, 64], kernel_size=3, proj_channels=32, dropout=0.1)
    ckpt = torch.load(DEFAULT_CONVLSTM_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    x = build_input_sequence(early_mask, norm).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu()

    # target: "ever active" footprint, matching the training target's
    # (state >= 1) semantics -- union of both real overpasses.
    target = np.clip(early_mask + late_mask, 0, 1)
    target_t = torch.from_numpy(target)

    model_metrics = compute_metrics(pred, target_t)
    eps = 1e-7
    p = pred.clamp(eps, 1 - eps)
    bce = -(target_t * torch.log(p) + (1 - target_t) * torch.log(1 - p)).mean().item()

    # Persistence baseline: "nothing changes" (predict = input).
    persistence_metrics = compute_metrics(torch.from_numpy(early_mask), target_t)

    print("\n=== ConvLSTM prediction vs REAL FIRMS ground truth ===")
    for k, v in model_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  bce_loss: {bce:.4f}")
    print("\n=== Persistence baseline (predict = input, no model) ===")
    for k, v in persistence_metrics.items():
        print(f"  {k}: {v:.4f}")

    save_comparison(early_mask, target, pred, n_early, n_late)

    with open(OUT_DIR / "real_data_firms_metrics.json", "w") as f:
        json.dump({
            "region": "West Siberia ~65.5N 76E",
            "source": "NASA FIRMS MODIS C6.1 global 24h active fire CSV",
            "n_early_points": n_early, "n_late_points": n_late,
            "model_metrics": model_metrics, "bce_loss": bce,
            "persistence_baseline_metrics": persistence_metrics,
            "limitation": "channel 0 is real; channels 1-9 are training-mean/neutral placeholders (see docstring)",
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'real_data_firms_metrics.json'}")


if __name__ == "__main__":
    main()
