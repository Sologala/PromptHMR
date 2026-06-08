# HMR Pipeline Documentation

## Project Overview

This repository contains two human mesh recovery projects under `/home/wen/ws/HMR`:

| Project | Paper | Purpose |
|---------|-------|---------|
| **CameraHMR/** | 3DV 2025 | Single-image HMR with explicit perspective camera estimation |
| **PromptHMR/** | CVPR 2025 | Multi-person video pipeline: detection → tracking → HMR → export |

The primary pipeline is `PromptHMR/pipeline/pipeline.py` (class `Pipeline`).

---

## Pipeline Architecture

### Processing Flow

```
Input Video
  │
  ├─► YOLO + BotSort tracking (all frames)
  │     └─► clip_video_by_detect_and_tracking()
  │           Groups consecutive tracked frames into clips (min 30 frames)
  │           Preserves: original_frames (global video frame nums)
  │                      frames (clip-local, 0-based)
  │
  └─► Per-Clip Processing (process_on_clip):
        1. Naive Camera Calibration
           focal = max(h, w), cx = w/2, cy = h/2
           (or from config file if provided)
        2. SPEC Camera Calibration (gravity direction)
           Estimates pitch/roll of camera relative to gravity
        3. DeepLabV3 Person Segmentation (class 15, threshold > 0.1)
        4. Static Camera Motion (NO DROID-SLAM)
           cam_R = I, cam_T = 0 for all frames
        5. ViTPose 2D Keypoint Estimation (ViTPose-H, COCO-25)
        6. PromptHMR Image Model → per-frame SMPL-X params
        7. PRHMR-Vid Video Model → temporal refinement
        8. Export:
           - results.pkl (full pipeline results)
           - subject-{id}.smpl (per-person SMPL animation)
           - world4d.mcs / world4d.glb (scene with camera)
           - vis.mp4 (visualization video)
```

### Key Design Decisions (2026-06)

1. **No DROID-SLAM**: Camera motion estimation removed. All outputs are in per-frame camera coordinates. `has_slam` is always `False`.
2. **Frame Index Mapping**: Done at clip creation time. Each track has both `frames` (clip-local, 0-based) and `original_frames` (global video frame numbers).
3. **No World Transform**: `world_hps_estimation()` is disabled. `smplx_world` is not computed. Export uses `smplx_cam` data directly.
4. **SPEC calibration** is kept for potential gravity-alignment use, but is not applied to outputs.

---

## Coordinate Systems

### Primary: Camera Coordinates (smplx_cam)

All pipeline outputs are in this coordinate system.

| Property | Convention |
|----------|-----------|
| Convention | OpenCV |
| +X | Right (in image) |
| +Y | Down (in image) |
| +Z | Forward (into scene, depth) |
| Origin | Camera optical center |
| Units | Meters |

**SMPL-X params in camera space:**
- `smplx_cam.rotmat`: Joint rotation matrices (global_orient is relative to camera axes)
- `smplx_cam.trans`: Pelvis translation from camera origin, in meters
  - `trans[0]`: meters right of camera
  - `trans[1]`: meters below camera
  - `trans[2]`: meters in front of camera (depth)

### Internal: SMPL-X Model Space (OpenGL)

Used internally by the `smplx` library.

| Property | Convention |
|----------|-----------|
| Convention | OpenGL |
| +Y | Up |
| -Z | Forward |
| Origin | Pelvis joint |

The conversion from OpenGL to Camera CV is handled internally by PromptHMR when it predicts camera-space parameters.

### Image/Pixel Coordinates

| Property | Convention |
|----------|-----------|
| Convention | OpenCV |
| +X | Right (columns) |
| +Y | Down (rows) |
| Origin | Top-left pixel corner |

Used for: 2D keypoints (`keypoints_2d`, `vitpose`), bounding boxes (`bboxes`).

### Camera Intrinsics

Stored in `results['camera']`:
- `img_focal`: float, focal length in pixels (same for fx and fy)
- `img_center`: [cx, cy], principal point in pixels

Intrinsic matrix K:
```
K = [[focal, 0,      cx],
     [0,     focal,  cy],
     [0,     0,      1 ]]
```

### Output Coordinate Systems Summary

```
┌─────────────────────────────────────────────────────────┐
│  Image/Pixel (2D)                                       │
│  • keypoints_2d: pixel coords (x=col, y=row)           │
│  • bboxes: xyxy in pixels                              │
│  • vitpose: pixel coords                               │
├─────────────────────────────────────────────────────────┤
│  Camera Coordinates (3D) — THIS IS THE MAIN OUTPUT      │
│  • smplx_cam.trans: [X_right, Y_down, Z_forward] (m)  │
│  • smplx_cam.rotmat: rotation matrices                 │
│  • smplx_cam.pose: axis-angle body pose                │
│  • smplx_cam.shape: betas (body shape)                 │
│  • camera.pred_cam_R: I (identity)                     │
│  • camera.pred_cam_T: 0 (zero)                         │
├─────────────────────────────────────────────────────────┤
│  Camera (for MCS/glTF export)                           │
│  • camera_world.Rcw: I (identity)                      │
│  • camera_world.Tcw: 0 (zero)                          │
│  Converted to glTF convention in mcs_export_cam.py:     │
│    cv_to_gltf = [[1,0,0],[0,-1,0],[0,0,-1]]            │
└─────────────────────────────────────────────────────────┘
```

---

## Export Fields Reference

### results.pkl

Dumped by `joblib.dump()` at `{output_dir}/results.pkl`.

```
{
    'camera': {
        'pred_cam_R': np.ndarray [T, 3, 3]  — Camera rotation, always identity (static cam)
        'pred_cam_T': np.ndarray [T, 3]     — Camera translation, always zero
        'img_focal': float                   — Focal length in pixels
        'img_center': np.ndarray [2]         — Principal point (cx, cy)
    },

    'spec_calib': {
        'first_frame': {
            'pitch': float  — Camera pitch (radians, rotation around X axis)
            'roll': float   — Camera roll (radians, rotation around Y axis)
            'vfov': float   — Vertical field of view
        },
        'median_focal_length': float  — Median focal length from SPEC estimation
    },

    'people': {
        <track_id: int>: {
            'frames': np.ndarray [N]         — Clip-local frame indices (0-based)
            'original_frames': np.ndarray [N]— Original video frame numbers (global)
            'bboxes': np.ndarray [N, 4]      — Bounding boxes xyxy in pixel coords
            'track_id': int                  — Unique tracking ID
            'detected': int                  — Total frames where person was detected

            # 2D Keypoints
            'keypoints_2d': np.ndarray [N, 25, 3]
                — OpenPose 25-joint format: [x_pixel, y_pixel, confidence]
                — Joint order: nose, neck, R-shoulder, R-elbow, R-wrist,
                  L-shoulder, L-elbow, L-wrist, mid-hip, R-hip, R-knee, R-ankle,
                  L-hip, L-knee, L-ankle, R-eye, L-eye, R-ear, L-ear,
                  L-bigtoe, L-smalltoe, L-heel, R-bigtoe, R-smalltoe, R-heel

            'vitpose': np.ndarray [N, 133, 3]
                — COCO WholeBody 133-joint format: [x_pixel, y_pixel, confidence]

            'prhmr_img_feats': np.ndarray [N, D]
                — Per-frame image features from PromptHMR (used by video model)

            'smplx_cam': {  ★ PRIMARY OUTPUT — Camera coordinates
                'rotmat': np.ndarray [N, 55, 3, 3]
                    — SMPL-X joint rotation matrices (camera space)
                    — 55 joints: 22 body + 30 hands + 1 jaw + 2 eyes
                'pose': np.ndarray [N, 165]
                    — SMPL-X pose in axis-angle (camera space)
                    — 55 joints × 3 = 165 dimensions
                'shape': np.ndarray [N, 10]
                    — SMPL-X shape parameters (betas)
                'trans': np.ndarray [N, 3]
                    — Root (pelvis) translation in camera space, meters
                    — [X_right, Y_down, Z_forward]
                'contact': np.ndarray [N]
                    — Binary foot contact labels (1=contact with ground)
                'static_conf_logits': np.ndarray [N]
                    — Raw contact confidence logits (before sigmoid)
            },

            # smplx_world: NOT COMPUTED (world transform disabled)
        }
    },

    'camera_world': {
        'pred_cam_R': np.ndarray [T, 3, 3]  — Same as camera (identity)
        'pred_cam_T': np.ndarray [T, 3]     — Same as camera (zero)
        'Rwc': np.ndarray [T, 3, 3]         — World-to-camera rotation → identity
        'Twc': np.ndarray [T, 3]            — World-to-camera translation → zero
        'Rcw': np.ndarray [T, 3, 3]         — Camera-to-world rotation → identity
        'Tcw': np.ndarray [T, 3]            — Camera-to-world translation → zero
        'img_focal': float                   — Focal length in pixels
        'img_center': np.ndarray [2]         — Principal point (cx, cy)
        'viz_scale': float                   — Visualization scale
        'viz_center': list [cx, 0, cz]       — Visualization center
    },

    'contact_joint_ids': [7, 10, 8, 11, 20, 21]
        — SMPL-X joint indices used for contact detection (feet joints)

    'masks': np.ndarray | None               — Person segmentation masks [T, H, W]

    'timings': dict                           — Timing information per stage

    # Status flags
    'has_tracks': bool     — Tracking completed
    'has_hps_cam': bool    — Camera-space HMR completed
    'has_hps_world': bool  — Always true (identity pass-through)
    'has_slam': bool       — Always false (no SLAM used)
    'has_hands': bool      — Hand pose estimation
    'has_2d_kpts': bool    — 2D keypoint estimation completed
    'has_post_opt': bool   — Post-optimization (disabled)
}
```

### subject-{id}.smpl (per person)

Written via `SMPLCodec`. Binary SMPL animation file:
- `shape_parameters`: Mean betas across all frames [10]
- `body_pose`: Axis-angle body pose per frame [N, 22, 3] — in camera coordinates
- `body_translation`: Per-frame translation [N, 3] — in camera coordinates, meters
- `frame_count`: Number of frames this person appears
- `frame_rate`: FPS from config (default 30)

### world4d.mcs / world4d.glb

MCS (Meshcapade Scene) and glTF binary formats:
- SMPL body buffers per person
- Frame presence ranges
- Camera animation: identity R/T (static camera at origin)
- Camera intrinsics (converted to yfov + aspect ratio)

Open in:
- https://me.meshcapade.com/editor (drag-and-drop .mcs)
- Blender (import .glb)

### vis.mp4

Visualization video with:
- Bounding boxes (colored by track ID)
- 2D keypoint overlays (confidence > 0.3)
- Track ID labels

---

## Key Thresholds & Configurations

### Detection & Tracking (config.yaml)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `det_thresh` | 0.1 | YOLO detection score threshold |
| `det_score_thresh` | 0.1 | Detection score threshold |
| `det_height_thresh` | 0.3 | Min height ratio vs largest person |
| `tracker` | `bytetrack` | Tracker type (`bytetrack` or `sam2`) |
| `bbox_interp` | `true` | Interpolate bboxes for missing frames |
| `fps` | 30 | Output frame rate |
| `max_fps` | 60 | Max input FPS (downsample if higher) |

### ViTPose Confidence (hardcoded)

| Threshold | Location | Purpose |
|-----------|----------|---------|
| 0.3 | `pipeline.py:199` | Visualization — keypoints with confidence < 0.3 not drawn |
| 0.9 | `postprocessing.py:159` | Reprojection loss (only used in `post_optimization()`, which is disabled) |
| 0.1 | `tools.py:212` | SAM2 point prompts (only used when `tracker=sam2`) |

### Clip Segmentation

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_continues_frames` | 30 | Minimum consecutive tracked frames to form a clip |
| `min_frames` | 10 | Not actively used |

---

## PromptHMR-Video Model Load Analysis

### Two-Stage Architecture

**Stage 1: Image Model (PromptHMR)**
- Model: PromptHMR SMPL-X head (~50M params)
- Input: Single image + mask + optional keypoints
- Output: per-frame rotmat, transl, betas, features
- Cost: Runs on EVERY frame with detected people (batched by 16)
- This dominates pipeline time

**Stage 2: Video Model (PRHMR-Vid / GVHMR DemoPL)**
- Model: Transformer-based temporal refinement
- Input: Full track sequence of image features + ViTPose keypoints + smoothed bbox trajectory
- Output: refined SMPL-X params + contact labels
- Cost: Runs once PER TRACK (not per frame). Two forward passes (with/without keypoints)
- The transformer attention is O(n²) in sequence length — for very long tracks (>1000 frames) this can become memory-intensive

**Typical load for a 300-frame clip with 5 people:**
- Image model: ~19 batches (300 frames / 16 batch size)
- Video model: 5 track sequences × 300 time steps each

**Conclusion:** The video model is NOT the primary bottleneck — it runs once per track. The per-frame image model dominates. However, the video model is essential for temporal consistency (preventing frame-to-frame pose jitter). Keep it.

---

## SAM2 / Segmentation Notes

### Default: DeepLabV3 (Dedicated Human Segmentation)

The default pipeline uses **DeepLabV3-ResNet50** with class 15 (person), threshold > 0.1. This IS a dedicated human segmentation model. It runs frame-by-frame on all frames in the clip.

### Alternative: SAM2 (Video Object Segmentation)

When `cfg.tracker == 'sam2'`, the pipeline uses:
1. ViTDet (detectron2) to detect people on a start frame
2. SAM2 video predictor to propagate masks across the entire video

SAM2 mode is experimental. It does TRUE video tracking (not frame-by-frame segmentation). It uses keypoint prompts and negative points from other people for disambiguation.

**Recommendation:** Stick with the default DeepLabV3. Only use SAM2 if you need mask propagation across occlusions.

---

## Environment Setup

### Prerequisites

- CUDA-capable GPU
- Conda (Miniconda or Anaconda)

### CameraHMR Environment

```bash
conda create -n camerahmr python=3.10
conda activate camerahmr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
cd CameraHMR
pip install -r requirements.txt
```

### PromptHMR Environment (choose one)

```bash
# PyTorch 2.4 + CUDA 12.1 (recommended)
bash PromptHMR/scripts/install.sh --pt_version=2.4

# PyTorch 2.6 + CUDA 12.6
bash PromptHMR/scripts/install.sh --pt_version=2.6
```

This creates conda environments named `phmr_pt2.4` or `phmr_pt2.6`.

### Verify Installation

```bash
# Check environments exist
conda env list | grep -E "camerahmr|phmr_pt"

# If conda activate has issues (permission denied on __conda_exe), 
# initialize conda with a fresh hook first:
eval "$(/home/wen/miniconda3/bin/conda shell.bash hook)"
conda activate camerahmr  # or phmr_pt2.4

# Verify CameraHMR
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Verify PromptHMR
conda activate phmr_pt2.4
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

### Commands Reference

```bash
# If conda shell function is broken, refresh it first:
eval "$(/home/wen/miniconda3/bin/conda shell.bash hook)"

# ===== CameraHMR =====
conda activate camerahmr

# Run demo on a single image
cd CameraHMR
python demo.py --img_path /path/to/image.jpg

# Run CamSMPLify optimization
cd CameraHMR/CamSMPLify
python optimize.py --img_path /path/to/image.jpg

# ===== PromptHMR Pipeline =====
conda activate phmr_pt2.4

# Run full pipeline on a video
cd PromptHMR
python -c "
from pipeline import Pipeline
p = Pipeline(static_cam=True)
p('input_video.mp4', 'output_dir/')
"

# Run pipeline with streaming (low memory)
python -c "
from pipeline import Pipeline
p = Pipeline(use_streaming=True)
p('input_video.mp4', 'output_dir/')
"

# Run pipeline with chunked processing (long videos)
python -c "
from pipeline import Pipeline
p = Pipeline(use_chunked_processing=True, chunk_size=300, chunk_overlap=30)
p('input_video.mp4', 'output_dir/')
"

# Read results
python -c "
import joblib, numpy as np
res = joblib.load('output_dir/clip_00/results.pkl')
print('Camera:', res['camera'])
print('People:', list(res['people'].keys()))
for pid, data in res['people'].items():
    print(f'  Person {pid}: {data[\"frames\"].shape[0]} frames, trans range: [{data[\"smplx_cam\"][\"trans\"][:,2].min():.2f}, {data[\"smplx_cam\"][\"trans\"][:,2].max():.2f}]m depth')
"
```

### UV (boxmot submodule only)

UV is only used for the BoxMOT submodule at `PromptHMR/submodule/boxmot/`. It is NOT used for the main pipeline.

```bash
cd PromptHMR/submodule/boxmot
uv sync --all-extras --all-groups
uv run <command>
```

---

## Data Dependencies

Required model files (download to `PromptHMR/data/pretrain/`):

| File | Purpose |
|------|---------|
| `phmr/checkpoint.ckpt` | PromptHMR image model |
| `phmr_vid/prhmr_release_002.ckpt` | PRHMR-Vid video model |
| `vitpose-h-coco_25.pth` | ViTPose 2D keypoint detector |
| `yolo11x.pt` | YOLO person detector |
| `droid.pth` | DROID-SLAM (NO LONGER NEEDED) |
| `sam_vit_h_4b8939.pth` | SAM (only for SAM2 mode) |
| `sam2_ckpts/` | SAM2 checkpoints (only for SAM2 mode) |

SMPL-X body models (download to `PromptHMR/data/body_models/`):
- `smplx/SMPLX_NEUTRAL.npz`
- `smplx/SMPLX_neutral_array_f32_slim.npz`
