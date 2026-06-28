#!/usr/bin/env python3
"""Run the YOLO -> ConvLSTM wildfire prediction pipeline on one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.pipeline import predict_wildfire_from_image
from src.vision.yolo_fire_adapter import plot_fire_grid_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Drone or satellite RGB image")
    parser.add_argument(
        "--sim",
        type=Path,
        required=True,
        help="Simulator .pt file for terrain/weather channels 1-9",
    )
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--convlstm-checkpoint", type=Path, default=None)
    parser.add_argument("--norm-json", type=Path, default=None)
    parser.add_argument("--start-t", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory for saved PNG diagnostics",
    )
    args = parser.parse_args()

    result = predict_wildfire_from_image(
        args.image,
        args.sim,
        yolo_model_path=args.yolo_model,
        convlstm_checkpoint=args.convlstm_checkpoint,
        norm_json=args.norm_json,
        start_t=args.start_t,
        timesteps=args.timesteps,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    plot_fire_grid_diagnostics(
        args.image,
        result.grid_result,
        title=f"YOLO fire grid — {stem}",
    )
    plt.savefig(args.output_dir / f"{stem}_yolo_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.imshow(result.next_fire_probability.numpy(), cmap="hot", vmin=0, vmax=1)
    plt.title(f"ConvLSTM next-fire probability — {stem}")
    plt.colorbar()
    plt.savefig(args.output_dir / f"{stem}_convlstm_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.save(args.output_dir / f"{stem}_fire_grid.npy", result.grid_result.fire_state_grid)
    np.save(
        args.output_dir / f"{stem}_next_fire.npy",
        result.next_fire_probability.numpy(),
    )

    burning = float(result.grid_result.fire_state_grid.sum())
    peak = float(result.next_fire_probability.max())
    print(f"Observed burning cells: {burning:.0f}")
    print(f"Peak next-fire probability: {peak:.3f}")
    print(f"Saved outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
