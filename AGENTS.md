# LingBot-MAP — Agent Notes

## Environment

- **Conda env**: `lingbot-map` (Python 3.10.20)
- **GPU**: RTX 4070 Laptop (8 GB VRAM)
- **CUDA**: Driver 13.2 / PyTorch 2.8.0+cu128
- **Model**: `models/lingbot-map-long.pt` (4.63 GB, 1,158M params)

Run with: `conda activate lingbot-map && python -u <script> ...`

## Repo Structure

```
lingbot_map_cuda/
├── lingbot_map/           # Core library (models, inference, vis, utils)
│   └── utils/
│       └── alignment.py   # Deferred window alignment (apply at load time)
├── apps/
│   ├── cli/
│   │   ├── demo.py        # Unified pipeline CLI — all modes
│   │   └── gct_profile.py # FPS profiling
│   ├── rgbd_render/       # Offline render library + cli.py
│   ├── viewer/            # WebSocket 3D viewer (separate app)
│   └── cuda_ext/          # CUDA kernels (frustum cull, voxelize)
├── config/                # YAML presets for render pipeline
├── example/               # Courthouse (286), university, loop, oxford
├── scripts/
│   ├── process_videos.sh  # Batch inference + render
│   └── compare_npz.py     # Frame-by-frame NPZ diff tool
├── demo.py / gct_profile.py  # Root shims → apps/cli/
└── pyproject.toml
```

## demo.py — Unified Pipeline CLI

`apps/cli/demo.py` auto-detects mode from flags:

- `--image_folder` / `--video_path` alone → **Interactive viewer** (viser 3D)
- `+ --headless` → log summary only
- `+ --save_predictions DIR` → **per-window NPZ** (unaligned) + alignment metadata + legacy flat copy
- `+ --save_images` → also embed uint8 images in meta.npz
- `+ --render OUT.mp4` → full pipeline: inference → NPZ → video
- `--load_predictions DIR` → **Viewer from NPZ** (reloads images from `--image_folder`)
- `--lazy_images` → use memmap-backed images (O(window) RAM instead of O(frames))
- `--input_folder DIR` → batch mode: discover scenes → process all

### Common Commands

```bash
# Quick interactive test (5 frames, SDPA)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --first_k 5

# Full courthouse interactive (286 frames, windowed, 8 GB safe)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 16 --overlap_size 4 --num_scale_frames 4 --offload_to_cpu

# Export NPZ (per-window format + aligned legacy copy)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 16 --num_scale_frames 4 --headless \
    --save_predictions outputs/courthouse/

# Re-open saved NPZ interactively (no model needed)
python apps/cli/demo.py --load_predictions outputs/courthouse/ \
    --image_folder example/courthouse

# Viewer with sky masking
python apps/cli/demo.py --load_predictions outputs/courthouse/ \
    --image_folder example/courthouse --mask_sky

# Lazy image loading (O(window) RAM for long sequences)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 16 --num_scale_frames 4 --headless \
    --lazy_images --save_predictions outputs/courthouse/
```

## NPZ Format

`--save_predictions` produces a directory with two complementary layouts:

### Per-window (primary — enables deferred alignment)

```
output_dir/
├── window_000/           # Frame NPZs with LOCAL indices (unaligned)
├── window_001/
├── ...
├── alignment.npz         # chunk_scales [W], chunk_transforms [W,4,4]
├── windows.json          # window boundaries, overlap, frame count
├── meta.npz              # frame_type, is_keyframe, images (optional)
└── aligned/              # Legacy flat copy (aligned, bit-identical to backup)
    ├── frame_000000.npz
    └── ...
```

Alignment transforms are computed during inference but **applied at load time**
by `lingbot_map.utils.alignment`.  Loading code auto-detects the per-window
format via the presence of `window_000/`.  The `aligned/` subdirectory is a
backward-compatible flat copy suitable for direct comparison with older exports.

### Legacy flat (per-frame, aligned)

| Key | Dtype | Raw/frame (518×294) | Notes |
|---|---|---|---|
| `depth` | float16 | 298 KB | Upcast to float32 on load |
| `depth_conf` | **uint8** | 149 KB | ±0.5 max quantisation error |
| `extrinsic` | float32 | 48 B | Camera-to-world 3×4 |
| `intrinsic` | float32 | 36 B | 3×3 intrinsics |
| `images` | uint8 | 446 KB | Only with `--save_images` |

286-frame courthouse: **~50 MB** (aligned flat), **~246 MB** total (per-window
with float32 depth for bit-accurate deferred alignment).

## Comparing NPZ Exports

`scripts/compare_npz.py` does frame-by-frame numerical diff:

```bash
# Compare aligned flat copies
python scripts/compare_npz.py outputs/baseline/aligned/ outputs/experiment/aligned/
python scripts/compare_npz.py outputs/baseline/aligned/ outputs/experiment/aligned/ --metric depth
python scripts/compare_npz.py outputs/baseline/aligned/ outputs/experiment/aligned/ --frames 0:50
```

The `aligned/` subdirectory is the stable comparison target (bit-identical
depth, extrinsic rtol=1e-5, depth_conf ±0.5, intrinsic rtol=1e-5).

## Architecture

**Model**: 1,158M params. Aggregator (DINOv2-L + 24 GCT blocks): 1.82 GB (bf16). Camera head: 0.86 GB (fp32). Depth head: 0.13 GB (fp32).

**KV cache**: FlashInfer (paged, ~8.6 GB, needs ≥12 GB GPU) or SDPA (dict-based, ~5.5 GB, flag: `--use_sdpa`).

**Pipeline**: `images → preprocess (518×W, crop) → model.forward() → pose_enc [S,9] → extrinsic (3×4) + intrinsic (3×3) → depth [S,H,W,1] → NPZ export`

**Windowed mode**: Overlapping windows processed with fresh KV cache each. `_align_and_stitch_windows()` computes pairwise similarity transforms to bring all windows into the first window's coordinate frame. Per-window callback saves raw predictions incrementally.

**Deferred alignment**: Alignment transforms are saved to `alignment.npz` and applied at load time by `lingbot_map/utils/alignment.py`. This decouples alignment from inference, enabling future overlap recycling and progressive visualization.

**Keyframe interval**: ≤320 frames → interval=1. >320 → auto `ceil(S/320)`. `--flow_threshold >0` → adaptive.

**Memory**: `validate_frame_count()` caps streaming ≤1100, windowed ≤50000. Peak for 286 frames windowed: ~5.1 GB (SDPA).

## Key Library Files

| File | Purpose |
|---|---|
| `lingbot_map/io_protocol.py` | I/O abstractions (FrameSource, PredictionSink, NPZDirectorySink) |
| `lingbot_map/inference.py` | `load_model`, `postprocess`, `prepare_for_visualization` |
| `lingbot_map/models/gct_stream.py` | `GCTStream` + `inference_streaming()` |
| `lingbot_map/models/gct_stream_window.py` | `GCTStream` + `inference_windowed()` + `_align_and_stitch_windows()` |
| `lingbot_map/utils/pose_enc.py` | 9-dim pose encoding ↔ extrinsics + intrinsics |
| `lingbot_map/utils/alignment.py` | Deferred alignment: apply per-window transforms at load time |
| `lingbot_map/vis/point_cloud_viewer.py` | Interactive viser 3D viewer |
| `lingbot_map/vis/sky_segmentation.py` | ONNX sky segmentation |
| `apps/rgbd_render/cli.py` | Standalone render CLI (NPZ → MP4) |
| `apps/rgbd_render/data/loader.py` | NPZ data loader (handles both legacy flat and per-window formats) |
| `apps/cli/demo.py` | Unified pipeline CLI — NPZ save/load, viewer, render |
