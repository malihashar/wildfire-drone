# YOLO Fire Detection Integration

This stage converts a drone RGB image into the existing ConvLSTM fire-state
channel. It does not change the simulator, dataset format, or ConvLSTM model.

## Model Choice

Use Ultralytics YOLO11 segmentation as the default computer vision model.

Recommended starting model:

```text
yolo11s-seg.pt
```

`yolo11s-seg` is a practical first choice because it returns instance masks,
class labels, confidences, and boxes while remaining small enough for drone
or Kaggle GPU inference. If speed is the bottleneck, use `yolo11n-seg.pt`. If
small or distant fires are missed, try `yolo11m-seg.pt`.

## Output Contract

The adapter output is a 100x100 grid that becomes channel `0` of the existing
10-channel ConvLSTM tensor.

Fire-state values:

```text
0 = unburned / no observed active fire
1 = burning / observed active fire
2 = burned / reserved for later burned-area detection
```

The first implementation only emits `0` and `1`. A single RGB image does not
contain enough temporal evidence to infer burned cells safely.

## Image-To-Grid Flow

1. Run YOLO11 segmentation on the drone RGB image.
2. Keep `fire` and `smoke` detections above class-specific confidence thresholds.
3. Convert each mask back into original image coordinates.
4. Build an image-space evidence map.
5. Fuse multiple detections with max confidence.
6. Downsample evidence into a 100x100 grid using area pooling.
7. Threshold the evidence grid into the ConvLSTM fire-state grid.

Default thresholds:

```text
fire detection confidence  = 0.25
smoke detection confidence = 0.40
grid occupancy threshold   = 0.25
```

Smoke is treated as weaker evidence than visible fire. Do not map the entire
smoke plume as burning ground unless a later smoke-base estimator is added.

## ConvLSTM Integration

The helper `build_convlstm_sequence()` creates a `(T, 10, 100, 100)` tensor by
combining:

- channel `0`: YOLO fire-state grid
- channels `1-9`: existing terrain/weather channels

For a single still image, repeat the detected fire-state grid across the 20
timesteps. For real drone video, maintain a rolling buffer of the last 20
observed fire grids.

## Public Fine-Tuning Datasets

Recommended sources:

- Boreal Forest Fire dataset: UAV wildfire imagery with bounding boxes and
  segmentation masks.
- Pyro-SDIS: large YOLO-format wildfire smoke dataset.
- HPWREN FIgLib: wildfire ignition image sequences from fixed cameras.
- AI For Mankind wildfire smoke dataset: HPWREN-derived bounding-box labels.
- Fire VN YOLO11-Seg v1: fire/smoke segmentation with hard negatives.
- DBA-Fire / DBA-YOLO dataset: fire and smoke images in YOLO-compatible format.

Use these as transfer-learning data, not as a reason to train from scratch.

## Training Recommendation

Start from pretrained YOLO11 segmentation weights:

```text
yolo11s-seg.pt
```

Suggested fine-tuning order:

1. Broad fire/smoke segmentation data.
2. UAV/aerial wildfire imagery.
3. Hard negatives: cloud, fog, dust, sunset, glare, steam, orange objects.

Training from scratch is not recommended because the available wildfire data is
too small and domain-specific compared with the visual knowledge already present
in YOLO pretrained backbones.

## Failure Cases

Expected risks:

- Smoke confused with cloud or fog.
- Small distant fires missed.
- Flame hidden by smoke or tree canopy.
- Image-plane grid does not match ground geometry for oblique camera angles.
- Sunset, glare, or orange objects create false positives.
- Noisy YOLO grids differ from clean simulator fire-state grids.

Mitigations:

- Add hard negatives.
- Use temporal smoothing over drone video.
- Tune thresholds on a validation set.
- Add camera calibration and ground-plane projection for deployment.
- Keep smoke as weak evidence unless smoke-base localization is added.
