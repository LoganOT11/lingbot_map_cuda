# LingBot-MAP — Agent Notes

## Environment

- **Conda environment**: `lingbot-map` (Python 3.10.20)
- **GPU**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
- **CUDA**: Driver 13.2 / PyTorch 2.8.0+cu128
- **Model file**: `models/lingbot-map-long.pt` (4.63 GB)

Run commands via:
```bash
conda activate lingbot-map && python -u <script> ...
```
(Use `conda activate` + `python -u` rather than `conda run` when capturing
logs — `conda run` buffers stderr.)

## Repo Structure

```
lingbot_map_cuda/
├── lingbot_map/           # Core library — pip install -e .
│   ├── models/            # GCTStream, GCTBase, windowed variant
│   ├── aggregator/        # AggregatorStream (KV cache: FlashInfer or SDPA)
│   ├── layers/            # Attention, blocks, RoPE, FlashInfer cache, ViT
│   ├── heads/             # CameraCausalHead, DPTHead
│   ├── utils/             # Pose encoding, geometry, image loading
│   ├── vis/               # GLB export, sky segmentation, Viser wrapper
│   └── inference.py       # Shared: load_model, postprocess, prepare_for_visualization
│
├── apps/
│   ├── cli/
│   │   ├── demo.py        # THE unified pipeline CLI — inference, export, render, viewer
│   │   └── gct_profile.py # FPS profiling tool
│   ├── rgbd_render/       # Offline point-cloud → video render pipeline library
│   │   └── cli.py         # Standalone render CLI (NPZ → video, no inference)
│   ├── viewer/            # Interactive WebSocket 3D viewer server
│   └── cuda_ext/          # CUDA kernels (frustum cull, voxelize)
│
├── config/                # YAML presets for render pipeline
├── example/               # Courthouse (286 frames), university, loop, oxford
│
├── demo.py                # Root shim → apps/cli/demo.py
├── gct_profile.py         # Root shim → apps/cli/gct_profile.py
└── pyproject.toml
```

**One file to rule them all**: `apps/cli/demo.py` is the single CLI entry point for
all pipeline stages — inference, NPZ export, video rendering, GLB export, batch
processing, and interactive 3D visualization.  Root shims (`demo.py`, `gct_profile.py`)
delegate to `apps/cli/` via `runpy.run_path()`.

## Key Files

| File | Purpose |
|---|---|
| `apps/cli/demo.py` | **Unified CLI** — all modes: interactive, headless, render, batch |
| `apps/cli/gct_profile.py` | FPS profiling tool |
| `apps/rgbd_render/cli.py` | Standalone render CLI (NPZ → video, no model needed) |
| `lingbot_map/inference.py` | Shared `load_model`, `postprocess`, `prepare_for_visualization` |
| `lingbot_map/models/gct_stream.py` | `GCTStream` + `inference_streaming()` |
| `lingbot_map/models/gct_stream_window.py` | `GCTStream` + `inference_windowed()` |
| `lingbot_map/aggregator/stream.py` | AggregatorStream (FlashInfer/SDPA KV cache) |
| `lingbot_map/utils/pose_enc.py` | `pose_encoding_to_extri_intri()` — 9-dim pose → 3×4 + 3×3 |
| `lingbot_map/utils/geometry.py` | Depth unprojection, SE3 inverse, camera math |
| `lingbot_map/vis/glb_export.py` | `predictions_to_glb()` — point cloud + cameras → .glb |
| `lingbot_map/vis/sky_segmentation.py` | ONNX sky segmentation masks |

## demo.py Modes

`apps/cli/demo.py` auto-detects its mode based on which flags are provided:

| Flags | Mode | Description |
|---|---|---|
| `--image_folder` / `--video_path` (no export flags) | **Interactive** | Inference → viser 3D viewer |
| `--image_folder` / `--video_path` + `--headless` | **Headless** | Inference → print summary |
| `--image_folder` / `--video_path` + `--save_predictions` | **Export** | Inference → per-frame NPZ files |
| `--image_folder` / `--video_path` + `--render` | **Full pipeline** | Inference → NPZ → video render |
| `--load_predictions` + `--render` | **Render only** | NPZ → video (no inference) |
| `--input_folder` + `--output_folder` | **Batch** | Discover scenes → process all |
| `--load_predictions` + `--visualize_sky_mask_only` | **Sky masks** | Generate sky seg masks |

All render/camera/overlay args from the old `batch_demo.py` are available in the
unified parser.  A `--config` YAML preset can seed defaults; CLI flags override.

### Common commands

```bash
# Interactive viewer (streaming inference + viser)
python demo.py --model_path models/lingbot-map-long.pt --image_folder example/courthouse

# Headless export to NPZ
python demo.py --model_path models/lingbot-map-long.pt --image_folder example/courthouse \
    --use_sdpa --first_k 10 --headless --save_predictions outputs/

# Full pipeline: inference + NPZ + video render
python demo.py --model_path models/lingbot-map-long.pt --video_path video.mp4 \
    --render outputs/scene.mp4 --camera_mode follow --mask_sky

# Render saved NPZ to video (tweak camera without re-running inference)
python demo.py --load_predictions outputs/scene/ --render outputs/scene_v2.mp4 \
    --camera_mode birdeye --config config/indoor.yaml

# Render saved NPZ with render-only CLI (lighter, no torch import)
python apps/rgbd_render/cli.py --input_npz outputs/scene/ --output_video out.mp4 \
    --mask_sky --camera_vis default

# Batch process all scenes
python demo.py --model_path models/lingbot-map-long.pt \
    --input_folder /data/scenes --output_folder /data/outputs --render
```

## Architecture

### Model: 1,158M params, 2.81 GB mixed-precision on GPU
- Aggregator (DINOv2-L + 24 GCT blocks): 1.82 GB (bf16)
- Camera head (iterative refinement, fp32): 0.86 GB
- Depth head (DPT, fp32): 0.13 GB

### KV cache backends
- **FlashInfer** — paged, pre-allocated ~8.6 GB. Requires ≥12 GB GPU.
- **SDPA** — dict-based, incremental ~5.5 GB at 72 frames. Works on 8 GB.

### Pipeline
```
images → preprocess (518×W, crop) → model.forward()
  → pose_enc [S, 9]
  → depth [S, H, W, 1] + depth_conf
  → postprocess: pose_enc → extrinsics (3×4) + intrinsics (3×3)
  → unproject: depth + extrinsics → world_points [S, H, W, 3]
  → export: NPZ (per-frame parallel I/O), GLB
  → render: rgbd_render offline pipeline → MP4
```

### Prediction shapes (16:9, 518×294, S frames)
| Key | Shape | Size/frame |
|---|---|---|
| pose_enc | [S, 9] | 36 B |
| extrinsic | [S, 3, 4] | 48 B |
| intrinsic | [S, 3, 3] | 36 B |
| depth | [S, 294, 518, 1] | 609 KB |
| world_points | [S, 294, 518, 3] | 1.8 MB |

## Keyframe Interval

- `num_frames ≤ 320` → interval = 1 (every frame cached)
- `num_frames > 320` → auto `interval = ceil(num_frames / 320)` (RoPE training range)
- `--flow_threshold > 0` → flow-based keyframe selection (adaptive, takes precedence)

### Memory budget estimation
`estimate_gpu_memory(resolution, window_frames, backend)` returns predicted
GPU VRAM breakdown (model, KV cache, special pages, activations).
At 518×294 with 72-frame window: ~13 GB FlashInfer, ~9 GB SDPA.
`validate_frame_count()` enforces streaming/windowed caps before inference.

## Logging

`demo.py` uses `logging` with `stream=sys.stderr` for unbuffered output.
Key messages:
- `GPU: ... | X GB free / Y GB total` — memory at startup
- `WARNING  Less than 10 GB ...` — recommends safe flags
- `Inference dtype: torch.bfloat16` — dtype selection
- `Inference done in X s (Y FPS)` — throughput
- `GPU peak during inference: alloc=X GB` — peak memory

## Test Suite

```bash
# Run all 164 tests (most run without GPU or model checkpoint)
conda run -n lingbot-map python -m pytest tests/ -v

# Just the fast unit tests (no model instantiation)
pytest tests/test_safety_fixes.py tests/test_io_protocol.py tests/test_processor.py -v
```
