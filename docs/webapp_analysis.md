# LingBot-MAP → Web App: Skeptical Engineering Analysis

> **Question:** Can this repo serve as the core of a real-time web app that takes
> videos, runs `demo.py`, and exports point clouds + camera `.npz`?

---

## 1. What the repo actually provides (the good)

### Core inference engine (`lingbot_map/`)

The library package is well-structured. The public API surface that matters:

| Entry point | What it does |
|---|---|
| `GCTStream(model).inference_streaming(images, ...)` | Frame-by-frame with KV cache. Returns `{pose_enc, depth, depth_conf, world_points, world_points_conf}` |
| `GCTStream(model).inference_windowed(images, ...)` | Overlapping windows. Same output shape + `{frame_type, is_keyframe, alignment_mode, chunk_scales, chunk_transforms}` |
| `GCTStream(model).clean_kv_cache()` | Reset KV between sequences |
| `pose_encoding_to_extri_intri(pose_enc, ...)` | 9-dim pose encoding → 3×4 extrinsics + 3×3 intrinsics |
| `depth_to_world_coords_points(depth, ext, intr)` | Per-frame depth → world-space XYZ |
| `predictions_to_glb(predictions, ...)` | Point cloud + cameras → `.glb` 3D file (needs `trimesh`) |
| `load_and_preprocess_images(paths, ...)` | Disk images → `[N, 3, H, W]` tensor in [0,1] |

The pipeline is: **images → preprocess → model.forward() → pose + depth → unproject → world points → export**.

### Prediction shapes for a 16:9 scene (518×294, `S` frames)

| Key | Shape | Dtype | Storage |
|---|---|---|---|
| `pose_enc` | `[1, S, 9]` | float32 | 36 bytes/frame |
| `extrinsic` | `[S, 3, 4]` | float32 | 48 bytes/frame |
| `intrinsic` | `[S, 3, 3]` | float32 | 36 bytes/frame |
| `depth` | `[S, 294, 518, 1]` | float32 | 609 KB/frame |
| `depth_conf` | `[S, 294, 518]` | float32 | 609 KB/frame |
| `world_points` | `[S, 294, 518, 3]` | float32 | 1.8 MB/frame |

For 300 frames: ~900 MB of raw predictions, ~180 MB NPZ compressed.

---

## 2. Current limits — these are the real constraints

### 2.1 Hardware floor

| Constraint | Value | Why |
|---|---|---|
| **Minimum GPU VRAM** | 8 GB (12 GB comfortable) | Model alone: 2.8 GB mixed-precision. KV cache for FlashInfer: ~8.6 GB at 518×294. SDPA: ~5.5 GB at 72 cached frames. |
| **CUDA required** | Yes, practically | Model is 1.1B params. CPU inference exists but is infeasible (100× slower). |
| **GPU architecture** | Turing+ (SM 7.5, RTX 20-series+) for bfloat16. Fallback to float16 on older cards. | bfloat16 aggregator + fp32 heads = 2.81 GB. float16 would be similar but less numerically stable for depth. |
| **FlashInfer requires ≤12 GB** | FlashInfer pre-allocates ~8.6 GB paged cache | Use `--use_sdpa` for ≤12 GB GPUs. SDPA is slower (~2.8 FPS vs ~5.7 FPS for FlashInfer on 5 frames) but uses less memory. |

### 2.2 Sequence constraints

| Constraint | Limit | Mitigation |
|---|---|---|
| **Streaming mode max frames** | ~320 frames (RoPE training range) | Auto keyframe interval kicks in. KV cache holds at most 320 keyframes. |
| **Windowed mode** | Arbitrary length, but each window ≤ `window_size` keyframes | 24 windows × 58s for 286 frames. Temporal drift between windows is corrected via chunk alignment. |
| **Minimum frames** | 4-8 scale frames required for bootstrap | Model needs bidirectional attention on initial frames to establish scale. 1-frame input won't work. |
| **Fixed input resolution** | Width = 518px (default). Height depends on aspect ratio — typically 294px for 16:9. | Set via `--image_size 518 --patch_size 14`. Must be divisible by 14. Changing drastically would need model retraining. |
| **No incremental streaming output** | Even `inference_streaming()` accumulates all frames and concatenates at the end | The per-frame loop exists internally but results aren't yielded. Memory grows with sequence length. |

### 2.3 Model constraints

| Constraint | Detail |
|---|---|
| **Single model instance per GPU** | 2.8 GB model + KV cache leaves no room for a second instance on 8 GB GPUs. Batching (multiple sequences on one GPU) is not supported — the KV cache is per-sequence. |
| **Checkpoint loading is slow** | 4.63 GB file, ~5s to load from disk to RAM, then weights copied to GPU. One-time startup cost. |
| **No fine-tuning API** | Inference-only code — no training loop, no dataset loader, no optimizer hooks. |
| **Camera head is iterative** | `camera_num_iterations` (default 4) re-runs the camera head for refinement. 4 → accurate but slower; 1 → fast but noisier. |
| **Sky segmentation is optional and slow** | ONNX model (`skyseg.onnx`, 176 MB) runs per-frame. Adds ~0.3s/frame on CPU. Irrelevant for web app if user supplies their own mask or doesn't need sky filtering. |

### 2.4 I/O constraints

| Constraint | Detail |
|---|---|
| **Image loading is file-based** | `load_and_preprocess_images()` reads from disk paths. A web app would receive bytes in memory. |
| **Video extraction writes to disk** | `demo.py` extracts video frames to JPEG files, then reads them back. Wasteful for a web app. |
| **No native video stream decoding** | Only OpenCV `VideoCapture` (file-based). Could use `ffmpeg` pipe or PyAV for streaming. |

---

## 3. What a web app would need (the gap)

### 3.1 Per-request lifecycle

A web request wants: "upload video → get point cloud + camera `.npz`"

Current `demo.py` flow:
```
parse_args() → load_images() → load_model() → prepare_model() → run_inference() → postprocess() → save/visualize
```

In a web app this becomes:
```
┌─ Startup ──────────────────────────────────────────┐
│ load model ONCE, keep warm on GPU                  │
└────────────────────────────────────────────────────┘
                         │
┌─ Per-request ──────────────────────────────────────┐
│ 1. Receive uploaded video / image zip             │
│ 2. Extract/decode frames → tensor (in memory)     │
│ 3. Preprocess → [S, 3, H, W] tensor              │
│ 4. Clean KV cache                                 │
│ 5. Run inference_streaming / inference_windowed   │
│ 6. Postprocess → extrinsics + depth               │
│ 7. Unproject depth → world points (can do         │
│    client-side or server-side)                    │
│ 8. Serialize → NPZ / GLB / JSON                   │
│ 9. Return to client                               │
│ 10. Free GPU tensors, clean KV cache              │
└────────────────────────────────────────────────────┘
```

### 3.2 What's missing

| Missing piece | Priority | Effort |
|---|---|---|
| **Model singleton / session manager** — Model loads once, handles multiple requests. Clear KV cache between requests. Queue requests if GPU is busy. | Critical | Medium |
| **In-memory image decode** — Accept bytes (uploaded video frames, image buffers) without writing to disk. | Critical | Medium |
| **Incremental / streaming results** — For live camera feed: yield per-frame results as they're computed rather than waiting for all frames. `inference_streaming()` already loops frame-by-frame internally but concatenates at the end. | High | Medium |
| **Progress reporting** — Long sequences (300+ frames) take ~60s. Need to stream progress back to client (WebSocket / SSE). | High | Low |
| **Error handling at API boundaries** — Invalid inputs, OOM, CUDA errors, timeout. Graceful degradation. | Critical | Medium |
| **Result serialization format** — NPZ works but is large. For web: GLB for point cloud viewer, JSON for camera path, or a compressed binary format like Draco/glTF for point clouds. | Medium | Medium |
| **Multi-GPU / multi-worker** — One GPU serves one request at a time. Multiple GPUs → multiple workers. GPU scheduling queue. | High (for production) | High |
| **Authentication / rate limiting / storage** — Standard web app concerns. Users expect their processed scenes to be retrievable later. | Medium | Medium |
| **Client-side viewer** — The Viser viewer runs as a separate Python process. A web app would need a client-side 3D viewer (Three.js + glTF/GLB, or potree for large point clouds). | Medium | High (but mostly frontend work) |

### 3.3 The frame streaming problem

For a **real-time** use case (camera feed → live point cloud):

`inference_streaming()` currently:
1. Processes all scale frames at once
2. Loops through remaining frames one-by-one
3. Concatenates all results at the end
4. Returns the complete dict

To make this web-friendly:
- The loop already processes frame-by-frame with KV cache. Results just need to be **yielded** instead of accumulated.
- Each yielded frame would contain: `{pose_enc, extrinsic, intrinsic, depth, depth_conf, world_points}` for that single frame.
- Scale frames (first 4-8) are processed as a batch — these could be sent as one chunk.
- The `output_device='cpu'` path already offloads per-frame, so memory would stay bounded.

---

## 4. Should it be an API? Yes — a REST + WebSocket hybrid

### 4.1 API surface

```
POST /api/scenes                    # Upload video/images, start processing
  → 201 {scene_id, status: "processing"}

GET /api/scenes/{scene_id}          # Check status, get results when done
  → 200 {status, progress_pct, results: {extrinsic_url, depth_url, ...}}

GET /api/scenes/{scene_id}/frame/{n}  # Get per-frame predictions
  → 200 {extrinsic, intrinsic, depth, ...}

WS /api/scenes/{scene_id}/stream    # Real-time frame-by-frame results
  → per-frame: {frame_idx, extrinsic, depth, world_points, ...}
  → completion: {status: "done", summary}

DELETE /api/scenes/{scene_id}       # Clean up

GET /api/health                     # GPU status, queue depth, model ready
```

### 4.2 Why not just a library call?

A library API (`lingbot_map.process_video(video_bytes) → predictions_dict`) would also work. But:

- A library still needs the **model loaded on GPU** — calling it from a web framework like FastAPI is effectively the same as an API.
- The library approach is better for **batch/offline** use (e.g., `demo_render/batch_demo.py` already does this).
- The API approach is better for **interactive web** use (multiple clients, progress, streaming).

Recommendation: **wrap the model in a class with a clean API, then expose it via FastAPI**. The class is the reusable unit; the HTTP layer is thin.

---

## 5. Recommended architecture

```
┌──────────────────────────────────────────────────────┐
│ FastAPI application                                  │
│                                                      │
│  POST /upload  ──┐                                   │
│  GET  /result   ─┤                                   │
│  WS   /stream   ─┤                                   │
│                  │                                   │
│  ┌───────────────▼────────────────────────────┐      │
│  │ SceneProcessor (singleton)                │      │
│  │                                            │      │
│  │  model: GCTStream  (loaded once, warm)    │      │
│  │  queue: asyncio.Queue  (serialize GPU)    │      │
│  │                                            │      │
│  │  async process(video_bytes, options):     │      │
│  │    1. decode_video_to_tensor(bytes)       │      │
│  │    2. self.model.clean_kv_cache()         │      │
│  │    3. for frame in streaming_loop():      │      │
│  │         yield per-frame predictions       │      │
│  │    4. postprocess → extrinsics, depth     │      │
│  │    5. unproject → world_points            │      │
│  │    6. serialize → npz/json/glb            │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  ┌──────────────▼──────────────────────────────┐     │
│  │ SceneStore (disk / S3)                      │     │
│  │  - Raw uploads                              │     │
│  │  - Processed NPZ files                      │     │
│  │  - GLB exports                              │     │
│  │  - Metadata (scene_id, status, timestamps)  │     │
│  └──────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

### 5.1 Concurrency model

Only **one request processes at a time** on a single GPU. The model's KV cache is fundamentally per-sequence — two concurrent `inference_streaming()` calls would corrupt each other's cache. Options:

1. **Queue (simple)**: FIFO. Requests queue up. Works for low traffic.
2. **Multi-GPU (production)**: One model per GPU, load-balanced. Each GPU handles one request.
3. **GPU time-slicing (not recommended)**: Would need to save/restore KV cache per request. Complex, fragile, no upside.

For an MVP: **single GPU, FIFO queue, with WebSocket progress**. This handles dozens of requests per hour easily (most processing is I/O-bound anyway: loading/saving).

### 5.2 Where to start

The single highest-value starting point is **refactoring `demo.py`'s inference pipeline into a reusable `SceneProcessor` class**:

```python
# lingbot_map/processor.py  (new file)

class SceneProcessor:
    """Processes image sequences through the GCT model.

    Designed to be instantiated once (model stays on GPU) and called
    repeatedly for different sequences.
    """

    def __init__(self, model_path: str, device: str = "cuda", ...):
        self.model = self._load_model(model_path, device)
        self.model.eval()

    def process(
        self,
        images: torch.Tensor,          # [S, 3, H, W] in [0,1]
        mode: str = "streaming",
    ) -> dict:
        """Process all frames → complete predictions dict."""
        ...

    def process_streaming(
        self,
        images: torch.Tensor,
    ) -> Iterator[dict]:
        """Process frame-by-frame, yielding per-frame predictions."""
        ...

    def postprocess(self, predictions: dict) -> dict:
        """Convert pose_enc → extrinsics/intrinsics, move to CPU."""
        ...

    def unproject_to_world(
        self, depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
    ) -> np.ndarray:
        """Depth map → world-space point cloud (client-side optional)."""
        ...
```

Then the FastAPI layer is thin glue:

```python
# app.py  (new file)
from fastapi import FastAPI, UploadFile, WebSocket
from lingbot_map.processor import SceneProcessor

app = FastAPI()
processor = SceneProcessor(model_path="lingbot-map-long.pt")

@app.post("/api/scenes")
async def upload_scene(video: UploadFile):
    scene_id = create_scene()
    frames = decode_upload_to_tensor(video)
    asyncio.create_task(process_in_background(scene_id, frames))
    return {"scene_id": scene_id, "status": "processing"}

@router.websocket("/api/scenes/{scene_id}/stream")
async def stream_results(ws: WebSocket, scene_id: str):
    for frame_pred in processor.process_streaming(frames):
        await ws.send_json(frame_pred)
```

---

## 6. Quick wins (what to do now, before building the web app)

These are changes to the existing `demo.py` / `lingbot_map/` codebase that make the web app integration cleaner:

### 6.1 Decouple video decode from image preprocessing

Current: `load_images()` in `demo.py` decodes video to disk, then `load_and_preprocess_images()` reads from disk.

Needed: A function that takes video bytes → preprocessed tensor, all in memory. Use PyAV or `cv2.VideoCapture` with a memory buffer.

### 6.2 Make `inference_streaming()` a generator

Add a `stream=True` parameter (or new method `stream_inference()`) that yields `(frame_idx, frame_prediction)` tuples instead of accumulating everything:

```python
# In GCTStream
def inference_streaming(self, images, ..., stream: bool = False):
    ...
    for i in range(scale_frames, S):
        frame_output = self.forward(...)
        if stream:
            yield i, frame_output   # NEW: incremental yield
        else:
            all_pose_enc.append(...)  # existing behavior
```

### 6.3 Separate postprocessing from main()

`postprocess()` and `prepare_for_visualization()` are already functions. Move them to `lingbot_map/utils/postprocess.py` so they're importable without demo.py.

### 6.4 Add `SceneProcessor` class

See section 5.2. This is the single biggest enabler. `demo.py`'s `main()` becomes:

```python
def main():
    args = parse_args()
    processor = SceneProcessor(args.model_path, device="cuda")
    images = load_images(args.image_folder, ...)
    predictions = processor.process(images, mode=args.mode, ...)
    if args.save_predictions:
        save_predictions_npz(predictions, args.save_predictions)
    if not args.headless:
        launch_viewer(predictions, ...)
```

### 6.5 GPU health endpoint

Add a `/api/health` equivalent that reports:
- Model loaded / ready
- Current GPU memory (free, allocated, reserved)
- Queue depth
- Whether FlashInfer or SDPA is active

---

## 7. What NOT to do

| Anti-pattern | Why |
|---|---|
| **Don't spawn a new Python subprocess per request** | Model load time is 12s. Subprocesses compete for GPU memory. |
| **Don't try to serve the Viser viewer directly** | Viser runs its own HTTP server on a separate port. For a web app, export GLB and use a client-side viewer (Three.js, Cesium, potree). |
| **Don't keep KV cache across requests** | Always `clean_kv_cache()` between sequences. KV cache is sequence-specific and large (5-9 GB). |
| **Don't force all processing server-side** | `unproject_to_world()` (depth → XYZ) is a simple matrix multiplication. It can run client-side (WebAssembly/WebGL) or server-side. Client-side saves bandwidth (depth is 1 channel vs XYZ is 3 channels). |
| **Don't use FlashInfer without testing on target GPU** | `expandable_segments` compatibility varies. SDPA is safer for heterogeneous deployments. |
| **Don't hardcode the model path** | Production deployments need configurable model paths, not CLI args. |

---

## 8. Deployment sizing

| Traffic level | GPU | Architecture |
|---|---|---|
| **Dev / single user** | 1× 8 GB (RTX 4070 Laptop, RTX 4060) | Single FastAPI process, FIFO queue |
| **Small team (5-10 users/day)** | 1× 12 GB (RTX 4070 Ti, A2000) | Same, with S3 storage for results |
| **Production (50+ users/day)** | 2× 24 GB (A10G, L40S) | Load-balanced, Redis queue, GPU health monitoring |
| **Scale (100+ concurrent)** | 4-8× A10G / L40S | Kubernetes, autoscaling, persistent storage |

Per-request cost: ~1 minute of GPU time for 286 frames. An 8 GB GPU can handle ~30-50 scenes/hour.

---

## 9. Bottom line

**Yes, this repo can serve as the core of a web app.** The inference engine is solid, the model is production-quality, and the output (camera poses + depth maps + point clouds) is exactly what a user would want to export.

The three blockers to production readiness are:
1. **No model server class** — model load + inference + cleanup is scripted in `demo.py`'s `main()`, not packaged for reuse.
2. **No incremental streaming** — results aren't yielded frame-by-frame, so the web app can't show live progress.
3. **No memory-bounded processing** — everything accumulates in GPU memory, which works for 300-frame courthouse but not for a 10,000-frame video uploaded by a user.

The `SceneProcessor` class (section 5.2) is the right place to start. It addresses all three.
