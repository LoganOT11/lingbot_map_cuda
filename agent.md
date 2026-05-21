# LingBot-MAP — Agent Notes

## Environment

- **Conda environment**: `lingbot-map` (Python 3.10.20)
- **GPU**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
- **CUDA**: Driver 13.2 / PyTorch 2.8.0+cu128
- **Model**: `models/lingbot-map-long.pt` (4.63 GB, 1,158M params)

```bash
conda activate lingbot-map && python -u <script> ...
```
(Use `conda activate` + `python -u` — `conda run` buffers stderr.)

## Repo Structure

```
lingbot_map_cuda/
├── lingbot_map/           # Core library (models, inference, vis, utils)
├── apps/
│   ├── cli/
│   │   ├── demo.py        # Unified pipeline CLI — all modes
│   │   └── gct_profile.py # FPS profiling
│   ├── rgbd_render/       # Offline point-cloud → video render library
│   │   └── cli.py         # Standalone render CLI (NPZ → MP4, no model)
│   ├── viewer/            # WebSocket 3D viewer server (separate app)
│   └── cuda_ext/          # CUDA kernels (frustum cull, voxelize)
├── config/                # YAML presets for render pipeline
├── example/               # Courthouse (286 frames), university, loop, oxford
├── scripts/
│   └── process_videos.sh  # Batch inference + render pipeline
├── demo.py                # Root shim → apps/cli/demo.py
├── gct_profile.py         # Root shim → apps/cli/gct_profile.py
└── pyproject.toml
```

## demo.py — Unified Pipeline CLI

`apps/cli/demo.py` is the single entry point. Mode is auto-detected from flags.

| Flags | Mode |
|---|---|
| `--image_folder` / `--video_path` (no export) | **Interactive**: inference → viser 3D viewer |
| `+ --headless` | **Headless**: inference → log summary |
| `+ --save_predictions DIR` | **Export**: inference → minimal NPZ (depth+extrinsic+intrinsic) |
| `+ --save_predictions DIR --save_images` | **Export + images**: NPZ with embedded uint8 images |
| `+ --render OUT.mp4` | **Full pipeline**: inference → NPZ → video render |
| `--load_predictions DIR` | **Viewer from NPZ**: load NPZ → interactive viewer (reloads images from `--image_folder`) |
| `--load_predictions DIR --render OUT.mp4` | **Render from NPZ**: NPZ → video (needs `--save_images` on export) |
| `--input_folder DIR --output_folder DIR` | **Batch**: discover scenes → process all |

### Common commands

```bash
# Interactive viewer (streaming inference)
python demo.py --model_path models/lingbot-map-long.pt --image_folder example/courthouse

# Windowed inference for long sequences (8 GB GPU safe)
python demo.py --model_path models/lingbot-map-long.pt --image_folder example/courthouse \
    --use_sdpa --mode windowed --window_size 16 --overlap_size 4 \
    --num_scale_frames 4 --offload_to_cpu --headless

# Export minimal NPZ (~84 MB for 286 frames)
python demo.py --model_path models/lingbot-map-long.pt --image_folder example/courthouse \
    --use_sdpa --mode windowed --window_size 16 --num_scale_frames 4 \
    --headless --save_predictions outputs/

# Re-open saved NPZ in interactive viewer
python demo.py --load_predictions outputs/ --image_folder example/courthouse
```

### NPZ format

Saved as per-frame files (`frame_000000.npz`, ...) + `meta.npz` for non-sequence data. Parallel I/O via `ThreadPoolExecutor`.

| Key | Dtype | Size/frame (518×294) | Notes |
|---|---|---|---|
| `depth` | **float16** | 298 KB | Half precision; upcast to float32 on load |
| `extrinsic` | float32 | 48 B | Camera-to-world 3×4 |
| `intrinsic` | float32 | 36 B | 3×3 intrinsics |
| `images` | uint8 | 446 KB | **Only with `--save_images`**; regenerated from `--image_folder` otherwise |

Dropped keys: `depth_conf` (noisy), `pose_enc` (redundant with extrinsic).

### Logging

`logging` to stderr (unbuffered). Key messages:
- `GPU: ... | X GB free / Y GB total` — startup
- `WARNING  Less than 10 GB...` — recommends safe flags for 8 GB GPUs
- `Inference dtype: torch.bfloat16` — dtype selection
- `Inference done in X s (Y FPS)` — throughput
- `GPU peak: alloc=X GB` — peak memory

## Architecture

### Model: 1,158M params
- Aggregator (DINOv2-L + 24 GCT blocks): 1.82 GB (bf16)
- Camera head (iterative refinement, fp32): 0.86 GB
- Depth head (DPT, fp32): 0.13 GB

### KV cache backends
- **FlashInfer** — paged, ~8.6 GB. Requires ≥12 GB GPU.
- **SDPA** — dict-based, ~5.5 GB at 72 frames. Works on 8 GB (required flag: `--use_sdpa`).

### Pipeline
```
images → preprocess (518×W, crop) → model.forward()
  → pose_enc [S, 9] → extrinsic (3×4) + intrinsic (3×3)
  → depth [S, H, W, 1]
  → export: NPZ (per-frame, parallel I/O)
```

### Keyframe interval
- `≤320` frames → interval = 1 (every frame cached)
- `>320` frames → auto `ceil(S/320)` (stays within RoPE training range)
- `--flow_threshold > 0` → adaptive flow-based keyframe selection

### Memory budget
`estimate_gpu_memory(resolution, window_frames, backend)` in `lingbot_map/inference.py`.
`validate_frame_count()` enforces caps: streaming ≤1100, windowed ≤50000.

## Key Files

| File | Purpose |
|---|---|
| `apps/cli/demo.py` | Unified CLI — all pipeline stages |
| `lingbot_map/inference.py` | `load_model`, `postprocess`, `prepare_for_visualization` |
| `lingbot_map/models/gct_stream.py` | `GCTStream` + `inference_streaming()` |
| `lingbot_map/models/gct_stream_window.py` | `GCTStream` + `inference_windowed()` |
| `lingbot_map/utils/pose_enc.py` | `pose_encoding_to_extri_intri()` — 9-dim → 3×4 + 3×3 |
| `lingbot_map/vis/point_cloud_viewer.py` | Interactive viser 3D viewer |
| `lingbot_map/vis/sky_segmentation.py` | ONNX sky segmentation masks |
| `apps/rgbd_render/cli.py` | Standalone render CLI (NPZ → MP4) |
