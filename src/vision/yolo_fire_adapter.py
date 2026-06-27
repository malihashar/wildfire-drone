"""
YOLO fire detection adapter.

This module is the boundary between computer vision and the existing wildfire
ConvLSTM. It converts YOLO fire/smoke segmentation masks or detection boxes
into the fire-state channel expected by the simulator-trained model:

    0 = unburned / no observed active fire
    1 = burning / observed active fire
    2 = burned   / reserved for future burned-area detection

The ConvLSTM and dataset format remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


DEFAULT_YOLO_MODEL = "yolo11s-seg.pt"
FIRE_STATE_UNBURNED = 0
FIRE_STATE_BURNING = 1


@dataclass(frozen=True)
class FireGridConfig:
    """Configuration for projecting YOLO outputs into the ConvLSTM grid."""

    grid_shape: tuple[int, int] = (100, 100)
    fire_classes: frozenset[str] = frozenset({"fire", "flame", "wildfire"})
    smoke_classes: frozenset[str] = frozenset({"smoke"})
    fire_conf_threshold: float = 0.25
    smoke_conf_threshold: float = 0.40
    grid_threshold: float = 0.25
    fire_weight: float = 1.0
    smoke_weight: float = 0.35
    min_component_cells: int = 4
    close_kernel_size: int = 0
    close_iterations: int = 1


@dataclass(frozen=True)
class FireMaskDetection:
    """
    One fire/smoke segmentation output in original image coordinates.

    Parameters
    ----------
    mask:
        Binary or probability mask shaped (image_h, image_w).
    class_name:
        Class label such as "fire" or "smoke".
    confidence:
        YOLO confidence score in [0, 1].
    box_xyxy:
        Optional bounding box in original image coordinates.
    """

    mask: np.ndarray
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FireBoxDetection:
    """
    One fire/smoke object detection output in original image coordinates.

    Box-based grids are less precise than mask-based grids, but they let the
    current YOLO detection model feed the ConvLSTM fire channel immediately.
    """

    box_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float


FireDetection = FireMaskDetection | FireBoxDetection


@dataclass(frozen=True)
class FireGridResult:
    """Outputs produced by projecting detections into a 100x100 grid."""

    fire_state_grid: np.ndarray
    evidence_grid: np.ndarray
    image_evidence: np.ndarray
    used_detections: list[FireDetection] = field(default_factory=list)


def _class_weight(class_name: str, config: FireGridConfig) -> float:
    name = class_name.lower()
    if name in config.fire_classes:
        return config.fire_weight
    if name in config.smoke_classes:
        return config.smoke_weight
    return 0.0


def _passes_threshold(det: FireMaskDetection, config: FireGridConfig) -> bool:
    name = det.class_name.lower()
    if name in config.fire_classes:
        return det.confidence >= config.fire_conf_threshold
    if name in config.smoke_classes:
        return det.confidence >= config.smoke_conf_threshold
    return False


def _resize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_shape
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.shape == (image_h, image_w):
        return np.clip(mask_arr, 0.0, 1.0)
    resized = cv2.resize(mask_arr, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def _box_to_mask(
    box_xyxy: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize an `xyxy` detection box into a binary image-space mask."""
    image_h, image_w = image_shape
    x0, y0, x1, y1 = box_xyxy
    x0_i = max(0, min(image_w, int(np.floor(x0))))
    y0_i = max(0, min(image_h, int(np.floor(y0))))
    x1_i = max(0, min(image_w, int(np.ceil(x1))))
    y1_i = max(0, min(image_h, int(np.ceil(y1))))

    mask = np.zeros((image_h, image_w), dtype=np.float32)
    if x1_i > x0_i and y1_i > y0_i:
        mask[y0_i:y1_i, x0_i:x1_i] = 1.0
    return mask


def _area_pool_to_grid(
    evidence: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Downsample an image-space evidence map into grid cells by area average."""
    grid_h, grid_w = grid_shape
    evidence = np.asarray(evidence, dtype=np.float32)
    pooled = cv2.resize(evidence, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    return np.clip(pooled, 0.0, 1.0)


def _max_pool_to_grid(
    evidence: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """
    Downsample image-space evidence using max occupancy per output cell.

    This is intentionally different from average pooling: thin fire fronts can
    cover only a small fraction of a 100x100 cell, but they should survive the
    image-to-grid projection if YOLO confidently segmented them.
    """
    image_h, image_w = evidence.shape
    grid_h, grid_w = grid_shape
    pooled = np.zeros((grid_h, grid_w), dtype=np.float32)

    for row in range(grid_h):
        y0 = int(np.floor(row * image_h / grid_h))
        y1 = int(np.ceil((row + 1) * image_h / grid_h))
        for col in range(grid_w):
            x0 = int(np.floor(col * image_w / grid_w))
            x1 = int(np.ceil((col + 1) * image_w / grid_w))
            cell = evidence[y0:y1, x0:x1]
            if cell.size:
                pooled[row, col] = float(cell.max())

    return np.clip(pooled, 0.0, 1.0)


def _filter_fire_grid(grid: np.ndarray, config: FireGridConfig) -> np.ndarray:
    """
    Remove tiny isolated detections while preserving thin connected fronts.

    We avoid morphological opening because erosion would delete narrow fire
    fronts. Connected-component area filtering removes small specks directly.
    """
    binary = (grid > 0).astype(np.uint8)

    if config.close_kernel_size > 1:
        kernel_size = int(config.close_kernel_size)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(config.close_iterations)),
        )

    min_area = max(0, int(config.min_component_cells))
    if min_area <= 1:
        return binary.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = FIRE_STATE_BURNING
    return filtered.astype(np.uint8)


def detections_to_fire_grid(
    detections: Iterable[FireDetection],
    image_shape: tuple[int, int],
    config: FireGridConfig | None = None,
) -> FireGridResult:
    """
    Convert fire/smoke masks or boxes into a ConvLSTM-compatible fire-state grid.

    The returned `fire_state_grid` is shaped (100, 100) by default and contains
    only values 0 and 1. Burned-state value 2 is intentionally not inferred from
    a single RGB image.
    """
    cfg = config or FireGridConfig()
    image_h, image_w = image_shape
    image_evidence = np.zeros((image_h, image_w), dtype=np.float32)
    used: list[FireDetection] = []

    for det in detections:
        if not _passes_threshold(det, cfg):
            continue

        weight = _class_weight(det.class_name, cfg)
        if weight <= 0.0:
            continue

        if isinstance(det, FireMaskDetection):
            mask = _resize_mask(det.mask, image_shape)
        else:
            mask = _box_to_mask(det.box_xyxy, image_shape)
        contribution = np.clip(mask * det.confidence * weight, 0.0, 1.0)
        image_evidence = np.maximum(image_evidence, contribution)
        used.append(det)

    evidence_grid = _max_pool_to_grid(image_evidence, cfg.grid_shape)
    fire_state_grid = np.where(
        evidence_grid >= cfg.grid_threshold,
        FIRE_STATE_BURNING,
        FIRE_STATE_UNBURNED,
    ).astype(np.uint8)
    fire_state_grid = _filter_fire_grid(fire_state_grid, cfg)

    return FireGridResult(
        fire_state_grid=fire_state_grid,
        evidence_grid=evidence_grid,
        image_evidence=image_evidence,
        used_detections=used,
    )


def cell_to_image_rect(
    row: int,
    col: int,
    image_shape: tuple[int, int],
    grid_shape: tuple[int, int] = (100, 100),
) -> tuple[int, int, int, int]:
    """
    Map a grid cell to an image rectangle `(x0, y0, x1, y1)`.

    This is the image-plane mapping used for the first integration stage. A
    later geospatial projection can replace it without changing ConvLSTM input.
    """
    image_h, image_w = image_shape
    grid_h, grid_w = grid_shape

    x0 = int(round(col * image_w / grid_w))
    x1 = int(round((col + 1) * image_w / grid_w))
    y0 = int(round(row * image_h / grid_h))
    y1 = int(round((row + 1) * image_h / grid_h))
    return x0, y0, x1, y1


def overlay_grid_on_image(
    image: np.ndarray,
    grid: np.ndarray,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (255, 80, 0),
) -> np.ndarray:
    """
    Overlay a 100x100 fire/probability grid onto the original RGB image.

    `grid` can be binary fire state or a probability heatmap in [0, 1].
    """
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    image_h, image_w = image_arr.shape[:2]
    grid_prob = np.asarray(grid, dtype=np.float32)
    grid_prob = np.clip(grid_prob, 0.0, 1.0)
    heat = cv2.resize(grid_prob, (image_w, image_h), interpolation=cv2.INTER_LINEAR)

    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    base = image_arr.astype(np.float32)
    blended = base * (1.0 - alpha * heat[..., None]) + color_arr * (alpha * heat[..., None])
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_fire_grid_diagnostics(
    image: np.ndarray | str | Path,
    result: FireGridResult,
    title: str | None = None,
):
    """
    Create a side-by-side diagnostic figure for YOLO-to-grid quality review.

    Panels are: original image, YOLO mask/evidence, binary 100x100 grid, and
    grid overlay on the original image. Matplotlib is imported lazily so this
    module remains usable in non-notebook environments.
    """
    import matplotlib.pyplot as plt

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        image_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        image_arr = np.asarray(image)

    overlay = overlay_grid_on_image(image_arr, result.fire_state_grid)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if title:
        fig.suptitle(title)

    axes[0].imshow(image_arr)
    axes[0].set_title("Original Image")
    axes[1].imshow(result.image_evidence, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("YOLO Mask / Evidence")
    axes[2].imshow(result.fire_state_grid, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("100x100 Fire Grid")
    axes[3].imshow(overlay)
    axes[3].set_title("Grid Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def build_convlstm_sequence(
    fire_state_grid: np.ndarray,
    terrain_weather: np.ndarray | torch.Tensor,
    timesteps: int = 20,
) -> torch.Tensor:
    """
    Build a `(T, 10, 100, 100)` tensor for the existing ConvLSTM.

    `terrain_weather` may be either `(9, H, W)` for static channels or
    `(T, 9, H, W)` for time-varying channels.
    """
    fire = torch.as_tensor(fire_state_grid, dtype=torch.float32)
    if fire.ndim != 2:
        raise ValueError("fire_state_grid must have shape (H, W)")

    terrain = torch.as_tensor(terrain_weather, dtype=torch.float32)
    if terrain.ndim == 3:
        terrain = terrain.unsqueeze(0).repeat(timesteps, 1, 1, 1)
    elif terrain.ndim == 4:
        if terrain.shape[0] != timesteps:
            raise ValueError(f"terrain_weather has {terrain.shape[0]} timesteps, expected {timesteps}")
    else:
        raise ValueError("terrain_weather must have shape (9, H, W) or (T, 9, H, W)")

    if terrain.shape[1] != 9:
        raise ValueError("terrain_weather must contain exactly 9 channels")
    if tuple(terrain.shape[-2:]) != tuple(fire.shape):
        raise ValueError("terrain_weather spatial shape must match fire_state_grid")

    fire_seq = fire.unsqueeze(0).unsqueeze(1).repeat(timesteps, 1, 1, 1)
    return torch.cat([fire_seq, terrain], dim=1)


class YoloFireSegmenter:
    """
    Thin wrapper around Ultralytics YOLO11 for fire-grid inference.

    Ultralytics is imported lazily so the rest of the repository remains usable
    in environments where YOLO is not installed. Segmentation models use masks;
    detection models fall back to bounding boxes.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_MODEL,
        config: FireGridConfig | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Install ultralytics to use YoloFireSegmenter: pip install ultralytics"
            ) from exc

        self.model = YOLO(str(model_path))
        self.config = config or FireGridConfig()
        self.device = device

    def predict_grid(self, image: np.ndarray | str | Path) -> FireGridResult:
        """Run YOLO inference and return the 100x100 fire grid result."""
        results = self.model.predict(
            image,
            conf=min(self.config.fire_conf_threshold, self.config.smoke_conf_threshold),
            device=self.device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no result objects")

        result = results[0]
        image_shape = tuple(int(v) for v in result.orig_shape)  # (H, W)
        detections = self._result_to_detections(result, image_shape)
        return detections_to_fire_grid(detections, image_shape, self.config)

    def _result_to_detections(
        self,
        result: Any,
        image_shape: tuple[int, int],
    ) -> list[FireDetection]:
        if result.boxes is None:
            return []

        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confs = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        names = result.names

        detections: list[FireDetection] = []
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            for mask, cls_id, conf, box in zip(masks, cls_ids, confs, boxes):
                class_name = str(names.get(int(cls_id), cls_id)).lower()
                detections.append(
                    FireMaskDetection(
                        mask=_resize_mask(mask, image_shape),
                        class_name=class_name,
                        confidence=float(conf),
                        box_xyxy=tuple(float(v) for v in box),
                    )
                )
            return detections

        for cls_id, conf, box in zip(cls_ids, confs, boxes):
            class_name = str(names.get(int(cls_id), cls_id)).lower()
            detections.append(
                FireBoxDetection(
                    box_xyxy=tuple(float(v) for v in box),
                    class_name=class_name,
                    confidence=float(conf),
                )
            )
        return detections
