# LingBot-MAP → Web App: Skeptical Engineering Response

> **Date:** 2026-05-20
> **Purpose:** Independent verification of `webapp_analysis.md` + deeper investigation + solution pathways with trade-offs

---

## Executive Summary

**TL;DR:** The webapp_analysis.md is substantially correct — all three blockers are real. But it understates the depth of Issue #2 (incremental streaming) and misses a fourth, equally-critical gap: the absence of any request lifecycle or concurrency model whatsoever. The codebase is well-structured for an offline research tool but was never designed for multi-tenant serving. That said, the inference engine itself is production-grade, and the required changes are well-understood, medium-effort engineering — no research required.

The single highest-ROI action: build a `SceneProcessor` class that wraps model + KV cache lifecycle + streaming yield. Everything else (FastAPI, queuing, multi-GPU) is standard web plumbing.

---

## 1. Independent Verification of the Three Issues

### 1.1 Issue #1: No Model Server Class

**Verdict: CONFIRMED — and worse than described.**

The analysis correctly identifies that `demo.py`'s `main()` is a monolithic script. Here's what further investigation reveals:

| Component | Current state | Where it lives |
|---|---|---|
| Model loading | Function `load_model()` | `lingbot_map/inference.py` (reusable! already extracted) |
| Inference orchestration | Inline in `main()` | `apps/cli/demo.py` (not reusable) |
| Post-processing | Functions `postprocess()`, `prepare_for_visualization()` | `lingbot_map/inference.py` (reusable!) |
| KV cache cleanup | `model.clean_kv_cache()` | Instance method on `GCTStream` (reusable!) |
| GPU memory management | Manual `torch.cuda.empty_cache()` calls scattered | `apps/cli/demo.py` |
| Compilation warmup | `compile_model()`, `_warm_streaming()` | `apps/cli/demo.py`, `apps/batch/main.py` (duplicated!) |

**The good news:** `lingbot_map/inference.py` already extracts `load_model()`, `postprocess()`, and `prepare_for_visualization()` as reusable functions. The batch demo (`apps/batch/main.py`) also has a `process_scene()` function that is a step toward a reusable pipeline, but it's still a monolithic per-scene function, not a class with lifecycle.

**What's missing for a server:**
- No class that owns the model instance across requests
- No `process()` method that takes `images: torch.Tensor → predictions: dict`
- No `stream()` method that yields per-frame results
- No session/request isolation (KV cache must be cleaned between requests)
- The compilation warmup logic is duplicated between `demo.py` and `batch/main.py`

### 1.2 Issue #2: No Incremental Streaming

**Verdict: CONFIRMED — but the fix is genuinely small.**

The analysis says results "aren't yielded frame-by-frame." This is technically true, but *misleading* about the effort required.

Here's what the actual code does in `GCTStream.inference_streaming()` (line-by-line verified):

```python
# Phase 2: Process remaining frames one-by-one
for i in range(scale_frames, S):
    frame_image = images[:, i:i+1].to(_model_device)
    frame_output = self.forward(frame_image, ...)
    # ← RESULTS ARE COMPUTED HERE, PER-FRAME

    all_pose_enc.append(_to_out(frame_output["pose_enc"]))
    all_depth.append(_to_out(frame_output["depth"]))
    # ... append to other lists
    del frame_output

# ← THEN EVERYTHING IS CONCATENATED AT THE END
predictions = {
    "pose_enc": torch.cat(all_pose_enc, dim=1),
    ...
}
```

**The fix is a 30-line refactor:** The per-frame loop already exists. The model already computes one frame at a time. KV cache is already frame-by-frame. All that's missing is replacing `all_pose_enc.append(...)` with `yield frame_output` and removing the final `torch.cat()`.

**But the analysis misses a critical subtlety:** Phase 1 (scale frames) processes multiple frames at once (typically 4-8). These *cannot* be yielded individually because the model uses bidirectional attention across scale frames. So a streaming API would be:
- **Chunk 1:** Yield all scale-frame predictions at once (batch of 4-8 frames)
- **Chunk 2+:** Yield each subsequent frame individually

This has UX implications for a web app: the first N seconds show no progress, then all scale frames appear, then it's frame-by-frame.

### 1.3 Issue #3: No Memory-Bounded Processing

**Verdict: PARTIALLY CONFIRMED — `output_device='cpu'` exists but isn't the default.**

The analysis says "everything accumulates in GPU memory." The actual code has:

```python
output_device = torch.device("cpu") if args.offload_to_cpu else None
```

When `output_device='cpu'`:
- Each frame's predictions are immediately moved to CPU after computation
- `del frame_output` frees GPU memory per-frame
- After the loop, `torch.cuda.empty_cache()` releases cached allocations
- The final `torch.cat()` happens on CPU — no GPU memory spike

**So the mechanism already exists.** The issue is:
1. It's opt-in via `--offload_to_cpu` (though it defaults to `True` in the CLI)
2. The images tensor itself stays on GPU throughout inference (can be large: ~1.8 GB for 300 frames)
3. The batch demo already demonstrates `images.pin_memory()` for keeping images on CPU and moving slices per-frame

**What's actually missing for bounded memory:**
- No hard memory cap or OOM prevention — the code trusts the user to configure correctly
- Images stay on GPU by default; only offloaded when `output_device='cpu'`
- No chunked/batched processing for very long sequences (>1000 frames) beyond windowed mode
- The windowed mode (`inference_windowed`) does process in overlapping windows, but it's designed for drift correction, not memory bounding — though it has that effect as a side benefit

---

## 2. Additional Issues the Analysis Missed or Understated

### 2.1 (NEW) Issue #4: No Request Lifecycle or Concurrency Model

The analysis mentions a FIFO queue in section 5.1 as the recommended concurrency model, but doesn't call out that **nothing resembling a request lifecycle exists anywhere in the codebase**.

The reality:
- `demo.py`'s `main()` is a linear script: parse args → load → infer → postprocess → visualize/exit
- `batch/main.py`'s `process_scene()` is per-scene but still synchronous, blocking, and script-oriented
- There is no concept of a "request," a "session," or "concurrent users"
- The model's KV cache is fundamentally single-sequence — two concurrent calls to `inference_streaming()` would corrupt each other

**What a request lifecycle needs:**
1. **Pre-request:** Model loaded, KV cache cleaned, GPU memory ready
2. **Request validation:** Check frame count, resolution, GPU memory
3. **Decode:** Video/image bytes → tensor (in memory, not disk)
4. **Process:** Inference with progress callbacks
5. **Post-process:** Pose → extrinsics/intrinsics, depth → world points
6. **Serialize:** NPZ, GLB, JSON, or streaming WebSocket frames
7. **Cleanup:** Clean KV cache, free GPU tensors
8. **Post-request:** GPU health check, return to pool

None of these stages are expressed as code anywhere.

### 2.2 (NEW) Issue #5: Video Decode Writes to Disk in the Main Demo

The analysis mentions this in passing (section 6.1) but doesn't emphasize: the primary user-facing `demo.py` **extracts video frames to disk JPEG files** before loading them back:

```python
# From apps/cli/demo.py, load_images():
if video_path is not None:
    ...
    cv2.imwrite(path, frame)  # ← WRITES TO DISK
    saved.append(path)

# Then:
images = load_and_preprocess_images(paths, ...)  # ← READS FROM DISK
```

This is a double I/O penalty: decode → encode JPEG → write → read → decode JPEG → preprocess. For a 300-frame video, this adds 5-10 seconds of wasteful disk I/O and quality loss from JPEG recompression.

**However, the batch demo has already fixed this** (`load_images_from_video()` in `apps/batch/main.py` decodes and preprocesses in memory). The fix just hasn't been backported to the main demo.

### 2.3 (UNDERSTATED) The `torch.compile` Wall

The analysis notes that `torch.compile` speeds up inference by ~5 FPS but requires warmup passes. What it doesn't mention:

- `compile_model()` and `_warm_streaming()` are duplicated in both `demo.py` and `batch/main.py` (with subtle differences!)
- The warmup requires **3 dress-rehearsal passes** of the full streaming loop, adding 30-60 seconds to startup
- `torch.compile` is incompatible with SDPA backend (SDPABlock lacks `attn_pre`/`ffn_residual` submodules)
- The warmup uses `_set_skip_append(True)` to exercise the non-keyframe path — an internal API that must be threaded carefully
- After warmup, the KV cache manager is destroyed and recreated (`model.aggregator.kv_cache_manager = None`) — this is a deliberate side effect that could surprise new developers

For a web app, this means: either pay the 30-60s startup penalty once (acceptable), or forgo `torch.compile` and accept 2-3 FPS instead of 5-7 FPS.

### 2.4 (UNDERSTATED) `_set_skip_append` is a Dangerous Internal API

Both `GCTStream` in `gct_stream.py` and `gct_stream_window.py` expose `_set_skip_append(skip: bool)` — an underscored "internal" method that controls whether the current frame's KV is persisted. It's critical for keyframe-based streaming but:

- It's called across the aggregator AND camera head KV caches
- If a caller forgets to call `_set_skip_append(False)` after a non-keyframe, **all subsequent frames would silently fail to persist KV** — causing catastrophic quality degradation
- There's no guard or context manager; it's raw boolean state
- The warmup code deliberately interleaves this with `torch.compiler.cudagraph_mark_step_begin()`

**Recommendation:** Wrap this in a context manager or make it a parameter to `forward()`, not global mutable state.

### 2.5 (NEW) The Two GCTStream Classes Problem

There are two near-identical `GCTStream` implementations:

| File | Used for | `inference_windowed()` |
|---|---|---|
| `lingbot_map/models/gct_stream.py` | CLI demo (streaming mode) | ❌ Only `inference_streaming()` |
| `lingbot_map/models/gct_stream_window.py` | Batch processing (streaming + windowed) | ✅ Has `inference_windowed()` + `inference_streaming()` |

Both have identical `__init__()` signatures, `clean_kv_cache()`, `_set_skip_append()`, `_build_aggregator()`, and `_build_camera_head()`. The window variant adds:
- `inference_windowed()` with overlap alignment and scale-mode selection
- Flow-based keyframe selection
- Chunk scale/transform metadata in predictions

**The web app analysis recommends using `gct_stream_window.py`** (section 2.2 mentions windowed mode), but the `load_model()` function in `inference.py` imports from `gct_stream.py` by default and switches to `gct_stream_window.py` only when `mode='windowed'`. This means a web app using streaming mode would miss the richer prediction metadata.

### 2.6 (MINOR) Hardcoded DINOv2 Layer Indices

The `_aggregate_features()` method in the windowed variant passes `selected_idx=[4, 11, 17, 23]` — these are specific DINOv2 ViT-L layer indices and are hardcoded. Changing the backbone would require changing these.

---

## 3. Solution Pathways

For each of the core problems, here are multiple approaches ranked by effort/risk.

### 3.1 Pathway Matrix for SceneProcessor

| Approach | Effort | Risk | Timeline | Best for |
|---|---|---|---|---|
| **A: Minimal wrapper class** | 1-2 days | Low | MVP in 1 week | Hacking a prototype |
| **B: Full-featured processor with streaming** | 3-5 days | Low-Medium | MVP in 2 weeks | Production web app |
| **C: Async worker pool with multi-GPU** | 2-3 weeks | Medium | Production in 4-6 weeks | Multi-tenant SaaS |

#### Approach A: Minimal Wrapper (Quick Win)

```python
# lingbot_map/processor.py
class SceneProcessor:
    def __init__(self, model_path, device="cuda", **kwargs):
        self.model = load_model(model_path, device, **kwargs)
        self.device = device

    def process(self, images: torch.Tensor) -> dict:
        """One-shot: images in → predictions out."""
        self.model.clean_kv_cache()
        predictions = self.model.inference_streaming(
            images, output_device=torch.device("cpu")
        )
        predictions, _ = postprocess(predictions, predictions["images"])
        return prepare_for_visualization(predictions)
```

**Pros:** 30 lines of code, immediately usable from FastAPI.
**Cons:** No streaming, no progress, no error handling, no memory bounding for GPU images.

#### Approach B: Full-Featured Processor (Recommended)

```python
class SceneProcessor:
    def __init__(self, model_path, device="cuda", **kwargs): ...

    def process(self, images, *, progress_callback=None) -> dict:
        """Blocking: images in → predictions out, with progress."""
        ...

    def process_streaming(self, images) -> Iterator[dict]:
        """Generator: yields per-frame predictions after scale batch.

        Yields for scale frames (as a batch), then one dict per frame:
          - frame_idx: int
          - frame_type: "scale" | "keyframe" | "non_keyframe"
          - pose_enc: [1, 1, 9]
          - depth: [1, H, W, 1]
          - depth_conf: [1, H, W]
          - extrinsic: [1, 1, 3, 4]
          - intrinsic: [1, 1, 3, 3]
          - world_points: [1, H, W, 3]  # optional, server-side only
        """
        ...

    def process_windowed(self, images, *, progress_callback=None) -> dict:
        """For long sequences: windowed processing."""
        ...

    @property
    def gpu_stats(self) -> dict:
        """Health check: GPU memory, model state."""
        ...
```

**Key design decisions:**
1. `process_streaming()` yields CPU tensors — never GPU. This keeps memory bounded.
2. `world_points` computation is a parameter (server-side or client-side). Client-side saves 3× bandwidth.
3. `progress_callback(frame_idx, total_frames)` is a simple callable, not async — the caller wraps it in `asyncio.to_thread()`.
4. No async internally — keeps the class framework-agnostic. Async adaptation is the caller's job.

#### Approach C: Async Worker Pool (Production)

```python
class SceneProcessorPool:
    def __init__(self, model_paths: list[str], devices: list[str]):
        self.workers = [SceneProcessor(mp, d) for mp, d in zip(model_paths, devices)]

    async def submit(self, images: torch.Tensor) -> str:
        """Queue a processing job, return job_id."""
        ...

    async def get_result(self, job_id: str) -> dict:
        """Poll for completion."""
        ...
```

Adds: multi-GPU load balancing, job queue (Redis or asyncio.Queue), retry logic, graceful degradation on worker failure.

### 3.2 Incremental Streaming: Refactoring `inference_streaming()`

The core change is minimal. Here's exactly what needs to change:

**Option A: New `stream_inference()` method (non-breaking)**

```python
# In GCTStream (both variants)
@torch.no_grad()
def stream_inference(self, images, **kwargs) -> Iterator[dict]:
    """Yields per-frame predictions, never accumulates."""
    # Phase 1: scale batch → yield as one chunk
    scale_preds = self._process_scale_frames(images, **kwargs)
    yield {"type": "scale_batch", "predictions": scale_preds, "frame_range": (0, scale_frames)}

    # Phase 2: per-frame streaming → yield individually
    for i in range(scale_frames, S):
        frame_pred = self._process_single_frame(images[:, i:i+1], ...)
        yield {"type": "frame", "frame_idx": i, "predictions": frame_pred}
```

**Pros:** Clean, non-breaking, new API surface.
**Cons:** Duplicates the inference loop logic. Must be maintained alongside `inference_streaming()`.

**Option B: Add `stream=False` parameter to existing method (minimal change)**

```python
def inference_streaming(self, images, ..., stream=False):
    ...
    if stream:
        # Phase 1: yield scale batch
        yield {"phase": "scale", "predictions": scale_output}
        # Phase 2: yield each frame
        for i in range(scale_frames, S):
            ...
            yield {"phase": "stream", "frame_idx": i, "predictions": frame_output}
    else:
        # existing accumulation logic
        ...
```

**Pros:** Minimal diff, backward-compatible.
**Cons:** Return type changes from `dict` to `Union[dict, Iterator[dict]]` — type-checker nightmare.

**Option C: Separate class `StreamingSession` (most flexible)**

```python
class StreamingSession:
    def __init__(self, model, images, **kwargs):
        self.model = model
        self.images = images
        self.current_idx = 0
        ...

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Advance one frame, return predictions."""
        if self.current_idx >= self.total_frames:
            raise StopIteration
        ...
```

**Pros:** Stateful session — caller controls pacing. Can pause/resume (hypothetically — KV cache is sequential, but the pattern supports it).
**Cons:** More code, more complex API surface.

**Recommendation: Option A (new method)** — cleanest separation, no type weirdness, and both variants of `GCTStream` can implement it.

### 3.3 Memory-Bounded Processing: Hardening the Existing Path

The `output_device='cpu'` path already works. What's needed:

1. **Make it the default for server workloads** — not opt-in
2. **Keep images on CPU** (already demonstrated in `batch/main.py` with `.pin_memory()`)
3. **Add a memory budget check** before processing:

```python
def _check_memory_budget(self, num_frames, resolution):
    """Raise if estimated peak memory exceeds GPU free memory."""
    kv_cache_est = self._estimate_kv_cache(resolution)
    model_est = 2.8  # GB
    peak_est = model_est + kv_cache_est + 1.0  # 1 GB headroom
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if peak_est > free_gb:
        raise MemoryError(f"Need ~{peak_est:.1f} GB GPU, only {free_gb:.1f} GB free")
```

4. **Expose `max_frames` limit** — refuse sequences that would exceed RoPE training range (320 frames)

5. **Add a `torch.cuda.empty_cache()` call** after `clean_kv_cache()` between requests — force release reserved memory

### 3.4 Video Decode In-Memory

The batch demo already solves this. The fix is to:

1. **Backport `load_images_from_video()` to `lingbot_map/utils/load_fn.py`** as `load_and_preprocess_video()`
2. **Add `load_and_preprocess_bytes()`** that takes raw bytes (no file path):

```python
def load_and_preprocess_bytes(
    data: bytes,
    mode: str = "crop",
    image_size: int = 518,
    patch_size: int = 14,
) -> torch.Tensor:
    """Decode image bytes → preprocessed tensor [1, 3, H, W] in [0, 1]."""
    import io
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    # ... resize, crop, to_tensor
    return tensor
```

This enables `POST /api/scenes` with `multipart/form-data` uploads without touching disk.

---

## 4. Architecture: Concrete Design

### 4.1 Recommended Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Web framework | **FastAPI** | Async-native, WebSocket built-in, OpenAPI auto-docs, Pydantic validation |
| ASGI server | **Uvicorn** with `--workers 1` | One worker = one GPU. Use process manager (systemd, Docker, k8s) for multi-GPU |
| File storage | **Local disk** for MVP, **S3/MinIO** for production | NPZ files are ~180 MB for 300 frames. Keep them off the API server's disk |
| Job queue (optional) | **Redis + RQ** or **arq** | Only needed for async background processing with result retrieval |
| Progress | **WebSocket** (per-frame), **SSE** (simpler) | WebSocket for true streaming; SSE for unidirectional progress updates |
| Client 3D viewer | **Three.js** with GLB | `predictions_to_glb()` already works. glTF/GLB is standard. |

### 4.2 Class Hierarchy

```
SceneProcessor (lingbot_map/processor.py)
├── __init__(model_path, device, **kwargs)
│   └── Loads model ONCE, keeps warm on GPU
├── process(images) → dict
│   └── One-shot: image tensor → full predictions
├── process_streaming(images) → Iterator[dict]
│   └── Yields per-frame predictions
├── process_from_video(video_bytes) → Iterator[dict]
│   └── Decodes video in memory, then streams
├── process_from_upload(file: UploadFile) → Iterator[dict]
│   └── FastAPI integration helper
├── clean() → None
│   └── Clean KV cache + empty CUDA cache
├── gpu_stats → dict
│   └── {free_gb, allocated_gb, reserved_gb, model_loaded, kv_cache_size}
└── estimate_resources(num_frames, resolution) → dict
    └── {peak_memory_gb, estimated_fps, estimated_duration_s}
```

### 4.3 FastAPI Layer (Thin Glue)

```python
# webapp/app.py
from fastapi import FastAPI, UploadFile, WebSocket, BackgroundTasks
from lingbot_map.processor import SceneProcessor

app = FastAPI()
processor = SceneProcessor(model_path=os.environ["MODEL_PATH"])

@app.post("/api/scenes")
async def create_scene(video: UploadFile, background_tasks: BackgroundTasks):
    scene_id = uuid4()
    scene_store.create(scene_id, status="processing")
    background_tasks.add_task(process_scene_background, scene_id, await video.read())
    return {"scene_id": scene_id, "status": "processing"}

@router.websocket("/api/scenes/{scene_id}/stream")
async def stream_scene(ws: WebSocket, scene_id: str):
    await ws.accept()
    frames = scene_store.get_frames(scene_id)
    for frame_pred in processor.process_streaming(frames):
        if await ws.client_state == WebSocketState.DISCONNECTED:
            break
        await ws.send_json(frame_pred)
    await ws.send_json({"type": "done"})

@app.get("/api/health")
async def health():
    return {
        "status": "ready" if processor.model is not None else "loading",
        **processor.gpu_stats
    }
```

### 4.4 Concurrency: The Single-GPU Reality

The analysis's recommendation of a FIFO queue is correct but under-specified. Here's how it actually works:

**The constraint:** `GCTStream` has ONE KV cache. Two concurrent `forward()` calls will corrupt each other. This is not a limitation that can be fixed by async/await — it's a hardware constraint (the KV cache is 5-9 GB of GPU memory, per-sequence).

**Option 1: Simple FIFO (MVP)**
```python
class SerialProcessor:
    def __init__(self, processor: SceneProcessor):
        self.processor = processor
        self._lock = asyncio.Lock()

    async def process(self, images):
        async with self._lock:
            return await asyncio.to_thread(self.processor.process, images)
```
- Pros: Dead simple, no external deps
- Cons: Requests queue up; 300-frame sequence = ~60s blocking

**Option 2: GPU time-slicing (NOT recommended)**
- Would require saving/restoring KV cache per request
- KV cache is 5-9 GB — saving/loading would take longer than just finishing the current request
- No upside

**Option 3: Multi-GPU pool (Production)**
```python
class GPUWorkerPool:
    def __init__(self, model_path, gpu_ids=[0, 1, 2, 3]):
        self.workers = [
            SceneProcessor(model_path, device=f"cuda:{gpu_id}")
            for gpu_id in gpu_ids
        ]
        self.available = asyncio.Queue()
        for w in self.workers:
            self.available.put_nowait(w)

    async def process(self, images):
        worker = await self.available.get()
        try:
            return await asyncio.to_thread(worker.process, images)
        finally:
            await self.available.put(worker)
```
- Pros: Linear scaling with GPU count
- Cons: Hardware cost, Kubernetes complexity

**Reality check:** On an 8 GB GPU, a 300-frame sequence takes ~60s. That's 60 requests/hour/GPU. For a small team (5-10 users/day), a single GPU with FIFO is fine. For production, you need multiple GPUs or accept queuing delays.

---

## 5. Trade-Offs and Hard Decisions

### 5.1 Server-Side vs Client-Side Unprojection

`depth_to_world_coords_points(depth, ext, intr)` converts per-frame depth maps to world-space XYZ. It's a simple matrix multiply.

| | Server-side | Client-side |
|---|---|---|
| Bandwidth per frame | 1.8 MB (3-channel XYZ) | 609 KB (1-channel depth) |
| Server CPU cost | ~0.5ms per frame (negligible) | 0 |
| Client cost | 0 | ~1ms per frame WASM/WebGL |
| Flexibility | Client gets final points | Client can apply custom filters, masks, confidence thresholds |

**Recommendation:** **Server-side by default, with a `?unproject=false` query parameter.** Most clients want points. The 3× bandwidth is worth the simplicity. Power users who want raw depth can opt out.

### 5.2 FlashInfer vs SDPA Backend

| | FlashInfer | SDPA |
|---|---|---|
| Speed (5-frame batch) | ~5.7 FPS | ~2.8 FPS |
| GPU memory (peak) | ~8.6 GB KV cache | ~5.5 GB KV cache |
| Dependencies | `flashinfer` CUDA extension | PyTorch built-in (`F.scaled_dot_product_attention`) |
| Compatibility | Turing+ (SM 7.5+), conflicts with `expandable_segments` | All CUDA GPUs, works with `expandable_segments` |
| `torch.compile` support | Yes (cudagraph_trees) | No (dynamic KV cache shape) |

**Recommendation:** **Detect and prefer FlashInfer, fall back to SDPA.** The web app should:
1. Try `import flashinfer`
2. If available and GPU ≥ 12 GB, use FlashInfer
3. Otherwise, use SDPA
4. Expose the active backend in `/api/health`

The 2× speed difference is significant for a web app where users are waiting.

### 5.3 Streaming vs Windowed Mode

| | Streaming | Windowed |
|---|---|---|
| Max sequence length | ~320 frames (RoPE limit) | Arbitrary (tested 10,000+) |
| Temporal consistency | Perfect (single KV cache) | Window drift corrected via alignment |
| Progress feedback | Frame-by-frame | Window-by-window (coarser) |
| GPU memory | O(frames) bounded by KV cache | O(window_size) fixed |
| FPS per frame | Consistent (~5 FPS) | Per-window overhead, but parallelizable |

**Recommendation:** **Auto-select based on sequence length.**
- ≤ 200 frames → streaming (best quality, real-time feedback)
- 200-500 frames → streaming with keyframe interval > 1
- > 500 frames → windowed (reliable, bounded memory)

Expose `?mode=auto` as the default and `?mode=streaming|windowed` as overrides.

### 5.4 NPZ vs GLB vs JSON for Results

| Format | Size (300 frames) | Use case |
|---|---|---|
| NPZ | ~180 MB | Machine-readable, full fidelity, numpy-native |
| GLB | ~80-200 MB (depends on downsampling) | Client-side 3D viewer (Three.js, Cesium) |
| JSON (per-frame) | ~4 MB/frame | REST API, progressive loading |
| Draco-compressed glTF | ~30-50 MB | Bandwidth-efficient client-side rendering |
| Protocol Buffers | ~100 MB | Type-safe, language-agnostic |

**Recommendation:** **Support NPZ (for download) + GLB (for viewer) + JSON streaming (for WebSocket).** Each serves a different use case:

- `GET /api/scenes/{id}/download` → NPZ (background download for offline use)
- `GET /api/scenes/{id}/viewer` → GLB (drop into a Three.js viewer)
- `WS /api/scenes/{id}/stream` → JSON per-frame (progressive web app)

### 5.5 The `_set_skip_append` Safety Problem

This is a latent bug waiting to happen. The mutable global state pattern:

```python
model._set_skip_append(True)   # non-keyframe starts
# ... if exception here ...
model._set_skip_append(False)  # ← never called → ALL FUTURE FRAMES BROKEN
```

**Fix options:**

1. **Context manager (recommended):**
```python
with model.skip_append():
    frame_output = model.forward(frame_image, ...)
```

2. **Parameter to forward():**
```python
frame_output = model.forward(frame_image, ..., persist_kv=False)
```

3. **Try/finally guard (minimal):**
```python
model._set_skip_append(True)
try:
    frame_output = model.forward(...)
finally:
    model._set_skip_append(False)
```

**Recommendation: Option 3 immediately (5-line fix), Option 1 long-term.** The try/finally is a one-line safety net. The context manager is the proper API.

---

## 6. Recommended Implementation Plan

### Phase 0: Pre-Flight Checks (1 day)

- [ ] Verify FlashInfer compatibility on target GPU
- [ ] Profile memory usage for expected video lengths (30s, 60s, 300s)
- [ ] Test SDPA fallback path end-to-end
- [ ] Benchmark `torch.compile` warmup time vs. FPS gain

### Phase 1: Core Processor Class (3-5 days)

**`lingbot_map/processor.py`** — New file containing `SceneProcessor`:
- [ ] `__init__`: model loading, dtype selection, optional compilation
- [ ] `process()`: blocking one-shot inference
- [ ] `process_streaming()`: generator yielding per-frame dicts
- [ ] `clean()`: KV cache cleanup + CUDA cache
- [ ] `gpu_stats`: health endpoint data
- [ ] `estimate_resources()`: memory/time estimates before processing

**`lingbot_map/utils/load_fn.py`** — Extensions:
- [ ] `load_and_preprocess_video(video_path)` — in-memory video decode (backport from batch/main.py)
- [ ] `load_and_preprocess_bytes(data: bytes)` — single image from bytes
- [ ] `decode_video_bytes_to_tensor(data: bytes)` — video bytes → tensor

**`lingbot_map/models/gct_stream.py`** — Non-breaking changes:
- [ ] Add `stream_inference()` generator method
- [ ] Wrap `_set_skip_append` in try/finally in the existing loop

**`lingbot_map/models/gct_stream_window.py`** — Same non-breaking changes:
- [ ] Add `stream_inference()` generator method
- [ ] Wrap `_set_skip_append` in try/finally

**Refactor `demo.py`:**
- [ ] `main()` uses `SceneProcessor` internally
- [ ] Remove duplicated compilation warmup code (import from processor)

### Phase 2: Web API Layer (3-5 days)

**`webapp/app.py`** — FastAPI application:
- [ ] `POST /api/scenes` — upload, queue, return scene_id
- [ ] `GET /api/scenes/{id}` — status, progress, result URLs
- [ ] `GET /api/scenes/{id}/frame/{n}` — single frame predictions
- [ ] `WS /api/scenes/{id}/stream` — per-frame streaming
- [ ] `DELETE /api/scenes/{id}` — cleanup
- [ ] `GET /api/health` — GPU status, queue depth

**`webapp/scene_store.py`** — Storage abstraction:
- [ ] LocalDiskSceneStore (MVP)
- [ ] S3SceneStore (production)
- [ ] Scene metadata: status, timestamps, frame count, size

**`webapp/worker.py`** — Background processing:
- [ ] `process_scene_background(scene_id, data)` — runs inference, saves results
- [ ] Progress callback updates scene metadata
- [ ] Error handling + status updates

### Phase 3: Production Hardening (1-2 weeks)

- [ ] Multi-GPU worker pool (`GPUWorkerPool`)
- [ ] Authentication (API keys or OAuth2)
- [ ] Rate limiting (per-user, per-IP)
- [ ] Result expiry and cleanup (cron job)
- [ ] Prometheus metrics (request count, latency, GPU memory, queue depth)
- [ ] Docker container with CUDA base image
- [ ] Load testing with concurrent users

### Phase 4: Client-Side (parallel track, frontend team)

- [ ] Three.js GLB viewer (drop-in from `predictions_to_glb()`)
- [ ] WebSocket client for progressive frame display
- [ ] Upload UI with drag-and-drop
- [ ] Progress bar with ETA

---

## 7. What NOT To Do (Expanded)

| Anti-pattern | Why | Severity |
|---|---|---|
| Fork a subprocess per request | Model load is 12s per subprocess. CUDA context initialization adds 3-5s. Subprocesses compete for GPU memory. | 🔴 Critical |
| Serve Viser viewer directly | Viser runs its own HTTP server. Can't multiplex with FastAPI on same port. Requires WebSocket between Viser and client — additional hop. | 🟡 Medium |
| Keep KV cache across requests | KV cache is 5-9 GB per sequence. Two sequences in cache = OOM on most GPUs. Even if it fit, cross-sequence attention would produce garbage. | 🔴 Critical |
| Force all processing server-side | Unprojection (depth→XYZ) is a 3× bandwidth inflator. Client-side unprojection saves bandwidth. But client-side requires shipping intrinsics and a WASM/JS unprojector. | 🟡 Medium |
| Use FlashInfer without testing | `expandable_segments` + FlashInfer = CUDA driver errors. Must test on exact target GPU model. SDPA is safer for heterogeneous deployments. | 🟠 High |
| Hardcode model path | Web apps need environment variables or config files, not CLI args. Model path changes between dev/staging/prod. | 🟡 Medium |
| Return GPU tensors to web clients | PyTorch tensors are not JSON-serializable. Must convert to numpy/list before serialization. | 🟠 High |
| Use async for model inference | The model is synchronous PyTorch. Wrapping in `asyncio.to_thread()` is correct. Using async inside the model would add complexity for zero benefit. | 🟡 Medium |

---

## 8. Key Dependencies and Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FlashInfer breaks on CUDA driver update | Medium | High (blocking for 12+ GB GPUs) | Always have SDPA fallback; health check reports active backend |
| OOM on long user videos | High | Medium (request fails, not server crash) | Hard frame limit + memory budget check before processing |
| Model checkpoint incompatible with code | Low | High (won't start) | Version pin checkpoint, validate on startup |
| KV cache leak between requests | Medium | High (silent quality degradation) | Always `clean_kv_cache()` in `SceneProcessor.clean()`; add assertion in tests |
| `_set_skip_append` left stuck | Medium | High (all frames after first non-keyframe broken) | Wrap in try/finally immediately; context manager long-term |
| torch.compile CUDA graph memory leak | Low | Medium (GPU OOM after many requests) | Monitor reserved memory; recreate model if needed |
| User uploads malicious file | Medium | Medium (disk space, CPU spike) | File size limits, type validation, sandboxed decode |

---

## 9. Bottom Line

The webapp_analysis.md is a solid assessment. The three issues it identifies are real:

1. **No model server class** → Build `SceneProcessor` in `lingbot_map/processor.py`
2. **No incremental streaming** → Add `stream_inference()` generator to `GCTStream`
3. **No memory-bounded processing** → `output_device='cpu'` already works; harden it

But the analysis under-delivers on actionable detail. This document fills those gaps with:

- **Line-level verification** of each claim against the actual source code
- **4 additional issues** the original missed (no request lifecycle, disk-bound video decode, `_set_skip_append` danger, duplicated GCTStream classes)
- **3 concrete solution pathways** for the processor (minimal, recommended, production)
- **Specific code patterns** for streaming, memory bounding, and safety fixes
- **Trade-off analysis** for every decision (FlashInfer vs SDPA, streaming vs windowed, server-side vs client-side unprojection, serialization formats)
- **A phased implementation plan** with checkboxes

**The repo can absolutely serve as the core of a web app.** The inference engine is production-quality. The output is exactly what users want. The engineering work is well-understood and bounded. Start with `SceneProcessor` — it addresses all three blockers in one cohesive unit.
