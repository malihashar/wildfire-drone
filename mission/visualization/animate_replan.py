"""
Save online-replanning animations as PNG frames, GIF, and MP4.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np

from mission.replanning.online_replanner import OnlineReplanResult
from mission.visualization.plot_replan import build_animation_frames


def save_replan_outputs(
    result: OnlineReplanResult,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Render and export PNG frames + GIF + MP4 for an online-replan run.

    ``animation_fps`` and ``hold_frames_per_event`` in the config control speed.
    """
    cfg = result.config
    out = Path(output_dir) if output_dir is not None else cfg.output_dir
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    figures = build_animation_frames(result)
    png_paths: list[Path] = []
    for i, fig in enumerate(figures):
        path = frames_dir / f"frame_{i:03d}.png"
        # Fixed canvas size (no tight bbox) keeps GIF/MP4 frame shapes identical.
        fig.savefig(path, dpi=cfg.dpi)
        png_paths.append(path)
        plt.close(fig)

    # Expand frames for slower playback (configurable hold).
    hold = max(1, int(cfg.hold_frames_per_event))
    expanded = []
    for p in png_paths:
        expanded.extend([p] * hold)

    gif_path = out / "online_replan.gif"
    mp4_path = out / "online_replan.mp4"
    _write_gif(expanded, gif_path, fps=cfg.animation_fps)
    _write_mp4(expanded, mp4_path, fps=cfg.animation_fps)

    return {
        "frames_dir": frames_dir,
        "gif": gif_path,
        "mp4": mp4_path,
        "first_frame": png_paths[0],
        "last_frame": png_paths[-1],
    }


def _write_gif(frame_paths: list[Path], path: Path, fps: float) -> None:
    images = [_crop_to_common(imageio.imread(p)) for p in frame_paths]
    # Ensure identical shapes after crop.
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    images = [im[:h, :w] for im in images]
    duration = 1.0 / max(fps, 1e-6)
    imageio.mimsave(path, images, format="GIF", duration=duration, loop=0)


def _write_mp4(frame_paths: list[Path], path: Path, fps: float) -> None:
    """Encode MP4 using the imageio-ffmpeg binary (no system ffmpeg required)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe

    images = [_crop_to_common(imageio.imread(p)) for p in frame_paths]
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    h -= h % 2
    w -= w % 2
    # Pad to macro_block_size=16 for broad player compatibility.
    h16 = (h + 15) // 16 * 16
    w16 = (w + 15) // 16 * 16
    clipped = []
    for im in images:
        rgb = im[:h, :w, :3]
        if h16 != h or w16 != w:
            canvas = np.zeros((h16, w16, 3), dtype=rgb.dtype)
            canvas[:h, :w] = rgb
            clipped.append(canvas)
        else:
            clipped.append(rgb)

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for im in clipped:
            writer.append_data(im)
    finally:
        writer.close()


def _crop_to_common(im):
    """Convert RGBA/RGB frames to RGB ndarray."""
    if im.ndim == 2:
        return im
    if im.shape[2] == 4:
        return im[:, :, :3]
    return im
