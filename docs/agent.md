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
│   ├── cli/               # demo.py, gct_profile.py
│   ├── batch/             # main.py (batch processing), rgbd_scan_render.py
│   ├── rgbd_render/       # Point-cloud → video offline render pipeline
│   ├── viewer/            # Interactive WebSocket 3D viewer server
│   └── cuda_ext/          # CUDA kernels (frustum cull, voxelize)
│
├── models/                # .pt, .onnx files (gitignored, download separately)
├── config/                # YAML presets for render pipeline
├── example/               # courthouse (286), university, loop, oxford
├── docs/                  # agent.md, webapp_analysis.md, repo_structure_review.md
│
├── demo.py                # Shim → apps/cli/demo.py
├── gct_profile.py         # Shim → apps/cli/gct_profile.py
├── demo_render/           # Shims → apps/batch/
└── pyproject.toml
```

Backward-compat shims at root delegate to `apps/` via `runpy.run_path()`.
The core library (`lingbot_map/`) is unchanged — all internal imports remain
`from lingbot_map.xxx import yyy`.

## Key Files

| File | Purpose |
|---|---|
| `apps/cli/demo.py` | Main CLI — streaming + windowed inference, viser viewer |
| `apps/cli/gct_profile.py` | FPS profiling tool |
| `apps/batch/main.py` | Batch processing + offline MP4 rendering |
| `lingbot_map/inference.py` | Shared `load_model`, `postprocess`, `prepare_for_visualization` |
| `lingbot_map/models/gct_stream.py` | `GCTStream` + `inference_streaming()` |
| `lingbot_map/models/gct_stream_window.py` | `GCTStream` + `inference_windowed()` |
| `lingbot_map/aggregator/stream.py` | AggregatorStream (FlashInfer/SDPA KV cache) |
| `lingbot_map/utils/pose_enc.py` | `pose_encoding_to_extri_intri()` — 9-dim pose → 3×4 + 3×3 |
| `lingbot_map/utils/geometry.py` | Depth unprojection, SE3 inverse, camera math |
| `lingbot_map/vis/glb_export.py` | `predictions_to_glb()` — point cloud + cameras → .glb |
| `lingbot_map/vis/sky_segmentation.py` | ONNX sky segmentation masks |

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
  → export: NPZ, GLB
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

## Working Test Commands

```bash
# Quick test (5 frames, SDPA, headless)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --first_k 5 --headless

# Full windowed (286 frames, ~1 min, ~5.6 GB peak)
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 16 --overlap_size 4 --num_scale_frames 4 --offload_to_cpu --headless

# With NPZ export
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --first_k 10 --headless \
    --save_predictions outputs/
```

## Logging

`demo.py` uses `logging` with `stream=sys.stderr` for unbuffered output.
Key messages:
- `GPU: ... | X GB free / Y GB total` — memory at startup
- `WARNING  Less than 10 GB ...` — recommends safe flags
- `Inference dtype: torch.bfloat16` — dtype selection
- `Inference done in X s (Y FPS)` — throughput
- `GPU peak during inference: alloc=X GB` — peak memory
