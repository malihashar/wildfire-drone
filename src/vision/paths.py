"""Default model and data paths for the vision + ConvLSTM product flow."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YOLO_SEG_MODEL = PROJECT_ROOT / "models" / "yolo" / "fire_smoke_seg_best.pt"
DEFAULT_CONVLSTM_CHECKPOINT = PROJECT_ROOT / "models" / "convlstm" / "best_model.pt"
DEFAULT_NORM_JSON = PROJECT_ROOT / "dataset" / "normalization.json"


def resolve_path(path: str | Path | None, default: Path, label: str) -> Path:
    """Resolve an optional user path, falling back to a project default."""
    candidate = default if path is None else Path(path)
    if not candidate.exists():
        raise FileNotFoundError(
            f"{label} not found: {candidate}. "
            "Place trained weights under models/ or pass an explicit path."
        )
    return candidate.resolve()


def resolve_convlstm_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve a ConvLSTM checkpoint file or checkpoint directory."""
    if path is None:
        return resolve_path(None, DEFAULT_CONVLSTM_CHECKPOINT, "ConvLSTM checkpoint")

    candidate = Path(path)
    if candidate.is_dir():
        for name in ("best_model.pt", "latest_model.pt"):
            ckpt = candidate / name
            if ckpt.exists():
                return ckpt.resolve()
        raise FileNotFoundError(
            f"ConvLSTM checkpoint directory has no best_model.pt or latest_model.pt: {candidate}"
        )

    if not candidate.exists():
        raise FileNotFoundError(f"ConvLSTM checkpoint not found: {candidate}")
    return candidate.resolve()
