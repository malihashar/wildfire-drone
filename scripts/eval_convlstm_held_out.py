#!/usr/bin/env python3
"""
Diagnostic re-evaluation of models/convlstm/best_model.pt on whatever
val/test-split simulations are actually present on this machine.

Reuses existing code only: src.convlstm.WildfireConvLSTM, the checkpoint
loader / normalization logic in src.vision.convlstm_bridge, and the exact
compute_metrics() from src.train (same precision/recall/F1/IoU definitions
used during training). No new training, no new dataset generation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.convlstm import WildfireConvLSTM
from src.train import compute_metrics, aggregate_metrics
from src.vision.convlstm_bridge import resolve_device
from src.vision.paths import DEFAULT_CONVLSTM_CHECKPOINT, DEFAULT_NORM_JSON

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "dataset" / "simulations"
META_DIR = ROOT / "dataset" / "metadata"
WINDOW = 20
STRIDE = 25  # keep runtime small; still covers early/mid/late fire behaviour
OUT_DIR = ROOT / "results" / "convlstm_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_CHANNELS = {1: "potential_ros", 2: "fireline_intensity", 3: "vegetation_density", 4: "slope", 7: "wind_speed"}


def local_overlap(split: str) -> list[str]:
    ids = {x["id"] for x in json.load(open(META_DIR / f"{split}.json"))}
    local = {f.stem for f in SIM_DIR.glob("*.pt")}
    return sorted(ids & local)


def normalize(frames: torch.Tensor, norm: dict) -> torch.Tensor:
    out = frames.clone()
    for ch, key in NORM_CHANNELS.items():
        mean, std = norm[key]["mean"], norm[key]["std"] or 1.0
        out[:, ch] = (out[:, ch] - mean) / std
    return out


def main() -> None:
    device = resolve_device("auto")
    model = WildfireConvLSTM(in_channels=10, hidden_dims=[64, 64], kernel_size=3, proj_channels=32, dropout=0.1)
    ckpt = torch.load(DEFAULT_CONVLSTM_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    norm = json.load(open(DEFAULT_NORM_JSON))

    val_ids = local_overlap("val")
    test_ids = local_overlap("test")
    print(f"Local val-split sims:  {val_ids}")
    print(f"Local test-split sims: {test_ids}")

    all_metrics = []
    plot_saved = 0
    bce_losses = []

    for split_name, ids in [("val", val_ids), ("test", test_ids)]:
        for sim_id in ids:
            tensor = torch.load(SIM_DIR / f"{sim_id}.pt", map_location="cpu", weights_only=True).float()
            T = tensor.shape[0]
            if T < WINDOW + 1:
                continue
            for start in range(0, T - WINDOW - 1, STRIDE):
                window = tensor[start:start + WINDOW].clone()
                target_frame = tensor[start + WINDOW, 0]
                target = (target_frame >= 1).float()

                window_n = normalize(window, norm)
                x = window_n.unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(x).squeeze(0).cpu()

                eps = 1e-7
                p = pred.clamp(eps, 1 - eps)
                bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean().item()
                bce_losses.append(bce)

                m = compute_metrics(pred, target)
                m["split"] = split_name
                m["sim"] = sim_id
                m["start"] = start
                all_metrics.append(m)

                if plot_saved < 3 and target.sum() > 5:  # skip near-empty frames for visual usefulness
                    save_comparison_plot(window[-1, 0], pred, target, sim_id, start, plot_saved)
                    plot_saved += 1

    print(f"\nEvaluated {len(all_metrics)} held-out windows "
          f"({len(val_ids)} val sims + {len(test_ids)} test sims present locally)")
    numeric_only = [{"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "iou": m["iou"]}
                    for m in all_metrics]
    agg = aggregate_metrics(numeric_only)
    print("Aggregate metrics (fire-class, threshold=0.5):")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")
    print(f"  bce_loss: {float(np.mean(bce_losses)):.4f}  (std={float(np.std(bce_losses)):.4f})")

    # per-simulation breakdown
    print("\nPer-simulation IoU:")
    by_sim: dict[str, list[float]] = {}
    for m in all_metrics:
        by_sim.setdefault(m["sim"], []).append(m["iou"])
    for sim, ious in sorted(by_sim.items()):
        print(f"  {sim}: n={len(ious):2d}  mean_iou={np.mean(ious):.4f}  min_iou={np.min(ious):.4f}")

    with open(OUT_DIR / "held_out_metrics.json", "w") as f:
        json.dump({"aggregate": agg, "bce_loss_mean": float(np.mean(bce_losses)),
                   "per_window": all_metrics}, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'held_out_metrics.json'}")


def save_comparison_plot(last_fire_frame, pred, target, sim_id, start, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_bin = (pred >= 0.5).float()
    error = (pred_bin - target)  # +1 false positive, -1 false negative, 0 correct

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(last_fire_frame, cmap="hot", vmin=0, vmax=2)
    axes[0].set_title(f"Input: last fire-state frame\n({sim_id}, t={start + 19})")
    axes[1].imshow(target, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Ground truth (t+1)\nburning OR burned")
    axes[2].imshow(pred, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Predicted P(fire) (t+1)")
    im = axes[3].imshow(error, cmap="coolwarm", vmin=-1, vmax=1)
    axes[3].set_title("Error\n(blue=false neg, red=false pos)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.tight_layout()
    out = OUT_DIR / f"comparison_{idx}_{sim_id}_t{start}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved plot: {out}")


if __name__ == "__main__":
    main()
