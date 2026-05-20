# LingBot-MAP → Web App: Implementation Blueprint

> **Audience:** Senior engineers tasked with building this
> **Principle:** Every module independently testable. Every interface explicitly contracted. Every side effect documented.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: Client (frontend/)                  │
│  upload UI  │  GS viewer (gsplat.js)  │  Point cloud (Three.js)    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────────────┐
│                     LAYER 2: Web API (webapp/)                      │
│  app.py (FastAPI)  │  worker.py (background)  │  scene_store.py    │
│  models.py (Pydantic contracts)                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  LAYER 1: Core Library (lingbot_map/)               │
│                                                                     │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ processor.py │  │ io_protocol.py│  │ segmentation.py          │ │
│  │ SceneProcessor│  │ FrameSource   │  │ SegmentationPipeline     │ │
│  │              │  │ PredictionSink│  │ lift_to_3d()             │ │
│  │              │  │ ProgressReporter│ │ export_semantic_glb()    │ │
│  └──────┬───────┘  └───────┬───────┘  └────────────┬─────────────┘ │
│         │                  │                        │               │
│  ┌──────▼──────────────────▼────────────────────────▼──────────┐   │
│  │                 Existing model code (UNCHANGED)              │   │
│  │  inference.py  │  models/gct_stream.py  │  models/gct_*.py  │   │
│  │  utils/pose_enc.py  │  utils/geometry.py  │  utils/load_fn.py│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  gs/ (NEW — Gaussian Splatting)                               │   │
│  │  initializer.py  │  trainer.py  │  export.py                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│              LAYER 0: Safety Fixes (existing files only)            │
│  gct_stream.py  │  gct_stream_window.py  │  inference.py            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key rule:** No new code in existing model files except for the safety fixes. Everything else goes in new files that import the existing modules. This keeps the model code untouched (minimizes regression risk) while enabling reuse.

---

## Layer 0: Pre-Flight Safety Fixes

### F-1: Guard `_set_skip_append` with try/finally

**Files:** `lingbot_map/models/gct_stream.py`, `lingbot_map/models/gct_stream_window.py`

**Current code (the bug):**
```python
# gct_stream.py line ~453, gct_stream_window.py line ~1242
if not is_keyframe:
    self._set_skip_append(True)

torch.compiler.cudagraph_mark_step_begin()
frame_output = self.forward(frame_image, ...)  # ← CUDA OOM or any exception HERE

if not is_keyframe:
    self._set_skip_append(False)  # ← NEVER REACHED → KV cache permanently poisoned
```

**Fix (surgical, 3 lines added per file):**
```python
if not is_keyframe:
    self._set_skip_append(True)
try:
    torch.compiler.cudagraph_mark_step_begin()
    frame_output = self.forward(frame_image, ...)
finally:
    if not is_keyframe:
        self._set_skip_append(False)
```

**Test F-1:**
- Unit test: Call `_set_skip_append(True)`, raise inside try block, verify `_set_skip_append(False)` is called (check `self.aggregator.kv_cache["_skip_append"]` after)
- Integration test: Inject a fake frame that triggers OOM; verify next valid frame's KV is still persisted correctly

### F-2: Add Hard Frame Limit

**File:** `lingbot_map/inference.py` (add to `load_model()` or as a separate validation function)

```python
# NEW function in inference.py
def validate_frame_count(
    num_frames: int,
    mode: str,
    max_frames_streaming: int = 1100,  # below 1124 special-page cap
    max_frames_windowed: int = 50000,  # arbitrary safety limit
) -> None:
    """Raise ValueError if frame count would crash or degrade."""
    if mode == "streaming" and num_frames > max_frames_streaming:
        raise ValueError(
            f"Streaming mode limited to {max_frames_streaming} frames "
            f"(got {num_frames}). Use mode='windowed' for longer sequences."
        )
    if num_frames > max_frames_windowed:
        raise ValueError(
            f"Maximum {max_frames_windowed} frames supported (got {num_frames})."
        )
```

**Test F-2:**
- Feed 2000 frames to streaming mode → expect ValueError
- Feed 100 frames to streaming mode → expect no error
- Feed 60000 frames to windowed mode → expect ValueError

### F-3: Memory Budget Check

**File:** `lingbot_map/inference.py` (new function)

```python
def estimate_gpu_memory(
    resolution: tuple[int, int],   # (H, W)
    num_frames_per_window: int,
    backend: str,                   # "flashinfer" | "sdpa"
    dtype: torch.dtype,
) -> dict:
    """Return estimated peak GPU memory in GB for a processing window."""
    h, w = resolution
    patches = (w // 14) * (h // 14)
    tokens_per_frame = patches + 6  # 6 special tokens
    
    # Model weights: fixed ~2.8 GB
    model_gb = 2.8
    
    # KV cache
    if backend == "flashinfer":
        page_size = tokens_per_frame - 6
        elements_per_page = page_size * 16 * 64  # 16 heads, 64 head_dim
        bytes_per_page = elements_per_page * (2 if dtype == torch.bfloat16 else 4)
        pages = 8 + num_frames_per_window + 16  # scale + window + headroom
        kv_cache_gb = (pages * bytes_per_page * 24) / 1e9  # 24 blocks
        special_gb = (25 * bytes_per_page * 24) / 1e9  # pre-alloc special pages
    else:  # SDPA
        bytes_per_frame = 2 * 16 * tokens_per_frame * 64 * (2 if dtype == torch.bfloat16 else 4)
        kv_cache_gb = (num_frames_per_window * bytes_per_frame * 24) / 1e9
        special_gb = 0
    
    activations_gb = 1.0  # conservative estimate
    total_gb = model_gb + kv_cache_gb + special_gb + activations_gb
    return {
        "model_gb": round(model_gb, 1),
        "kv_cache_gb": round(kv_cache_gb, 1),
        "special_pages_gb": round(special_gb, 1),
        "activations_gb": activations_gb,
        "total_gb": round(total_gb, 1),
    }
```

**Test F-3:**
- Compute estimate for 518×294, window=64, FlashInfer → expect ~12.4 GB
- Compute estimate for 518×518, window=64, FlashInfer → expect higher (more patches)
- Verify against actual `torch.cuda.max_memory_allocated()` on a real GPU

---

## Layer 1: Core Library

### Module 1A: I/O Protocols (`lingbot_map/io_protocol.py` — NEW)

This module defines the abstract interfaces that decouple the processing core from I/O. **Every I/O operation goes through these protocols — no direct file I/O or network I/O in the processor.**

```python
from abc import ABC, abstractmethod
from typing import Iterator, Protocol
import torch
import numpy as np

# ── Frame Source ──────────────────────────────────────────────────────────

class FrameSource(ABC):
    """Produces preprocessed image tensors one at a time.
    
    Designed for streaming: frames are yielded on demand, not loaded
    all at once. Caller controls pacing.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames. Must be known before iteration."""
        ...
    
    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield frames as [1, 3, H, W] tensors in [0, 1], on CPU."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(H, W) of preprocessed frames."""
        ...
    
    @property
    @abstractmethod
    def original_paths(self) -> list[str] | None:
        """Original file paths if available (for sky seg, debug)."""
        ...


class VideoFileSource(FrameSource):
    """Decode video file frame-by-frame using cv2.VideoCapture."""
    def __init__(self, path: str, fps: int = 5,
                 image_size: int = 518, patch_size: int = 14): ...
    # Implementation: backport load_images_from_video() from batch/main.py


class ImageFolderSource(FrameSource):
    """Load pre-extracted image files from a folder."""
    def __init__(self, folder: str, image_size: int = 518,
                 patch_size: int = 14, stride: int = 1): ...


class BytesUploadSource(FrameSource):
    """Decode video from in-memory bytes (FastAPI UploadFile)."""
    def __init__(self, data: bytes, fps: int = 5,
                 image_size: int = 518, patch_size: int = 14): ...
    # Implementation: write bytes to tempfile, use cv2.VideoCapture,
    # or use PyAV for true in-memory decode


# ── Prediction Sink ───────────────────────────────────────────────────────

class PredictionSink(ABC):
    """Receives per-frame predictions. Caller pushes; sink persists.
    
    This is the write side of the incremental-save architecture.
    Each frame's predictions are pushed immediately after computation,
    so the processor never accumulates >1 frame in memory.
    """
    
    @abstractmethod
    def write_metadata(self, metadata: dict) -> None:
        """Called once before any frames. Config, resolution, frame count."""
        ...
    
    @abstractmethod
    def write_frame(self, frame_idx: int, predictions: dict) -> None:
        """Called once per frame, in order.
        
        predictions keys: pose_enc, depth, depth_conf, extrinsic, intrinsic.
        All values are numpy arrays on CPU.
        """
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Called after all frames. Finalize, flush, close files."""
        ...


class NPZDirectorySink(PredictionSink):
    """Write each frame as frame_NNNNNN.npz in a directory."""
    def __init__(self, output_dir: str): ...


class WebSocketSink(PredictionSink):
    """Stream frames over a WebSocket connection."""
    def __init__(self, websocket): ...
    # Note: this is async. Use asyncio.Queue or run in thread with
    # loop.call_soon_threadsafe for the websocket sends.


class NullSink(PredictionSink):
    """Discard all predictions. For benchmarking throughput."""
    ...


# ── Progress Reporter ─────────────────────────────────────────────────────

class ProgressReporter(ABC):
    """Called by the processor to report progress.
    
    The processor never knows or cares how progress is consumed
    (log, WebSocket, progress bar, Prometheus).
    """
    
    @abstractmethod
    def on_start(self, total_frames: int, config: dict) -> None: ...
    
    @abstractmethod
    def on_frame(self, frame_idx: int, stage: str, fps: float | None = None) -> None: ...
    
    @abstractmethod
    def on_complete(self, summary: dict) -> None: ...
    
    @abstractmethod
    def on_error(self, error: Exception, frame_idx: int | None) -> None: ...


class CallbackProgress(ProgressReporter):
    """Simple callable-based reporter. Pass functions."""
    def __init__(self, on_frame_fn=None, on_complete_fn=None, on_error_fn=None): ...


class TqdmProgress(ProgressReporter):
    """Renders a tqdm progress bar (for CLI use)."""
    ...


class NullProgress(ProgressReporter):
    """Silent. For tests."""
    ...
```

**Tests for Module 1A:**

| Test | What it verifies | Needs GPU? |
|---|---|---|
| `test_video_file_source_frame_count` | `len(source)` matches expected for known video | No (tiny test video) |
| `test_video_file_source_resolution` | Preprocessed frames have correct H, W | No |
| `test_video_file_source_iteration` | Iterating yields correct number of tensors in [0,1] | No |
| `test_bytes_upload_source` | Bytes → frames round-trips correctly | No |
| `test_npz_directory_sink_roundtrip` | Write frames → read back → compare | No |
| `test_npz_directory_sink_metadata` | Metadata written and readable | No |
| `test_null_sink_accepts_all` | NullSink never raises | No |
| `test_callback_progress_called` | Callbacks fire in correct order with correct args | No |
| `test_progress_reporter_ordering` | on_start → N× on_frame → on_complete (no missing, no extra) | No |

### Module 1B: SceneProcessor (`lingbot_map/processor.py` — NEW)

This is the central orchestration class. It depends on the existing model code and the I/O protocols above. It does NOT depend on FastAPI, asyncio, or any web framework.

```python
import torch
from dataclasses import dataclass
from typing import Iterator, Optional
from lingbot_map.inference import load_model, postprocess, validate_frame_count, estimate_gpu_memory
from lingbot_map.io_protocol import FrameSource, PredictionSink, ProgressReporter

@dataclass
class ProcessorConfig:
    """All configuration for a processing run. No argparse coupling."""
    mode: str = "streaming"            # "streaming" | "windowed"
    image_size: int = 518
    patch_size: int = 14
    num_scale_frames: int = 8
    keyframe_interval: int = 1
    kv_cache_sliding_window: int = 64
    window_size: int = 64
    overlap_size: int | None = None
    overlap_keyframes: int | None = None
    camera_num_iterations: int = 2     # 2 for speed, 4 for accuracy
    use_sdpa: bool = False
    compile: bool = False
    output_device: str = "cpu"
    include_world_points: bool = False  # Compute server-side?
    include_images: bool = False         # Echo input images?


class SceneProcessor:
    """Owns the model, manages KV cache lifecycle, runs inference.
    
    USAGE:
        processor = SceneProcessor(model_path="/data/model.pt")
        
        # Blocking: all frames at once
        predictions = processor.process(source, config, sink, progress)
        
        # Streaming: frame-by-frame via sink
        processor.process_streaming(source, config, sink, progress)
        
        # Clean between sequences (MANDATORY)
        processor.clean()
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        compile: bool = False,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = self._resolve_dtype(dtype)
        self._model = None           # Lazy-loaded on first use
        self._model_loaded = False
        self._should_compile = compile
    
    # ── Public API ─────────────────────────────────────────────────────
    
    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def process(
        self,
        source: FrameSource,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Process all frames, saving results via sink. Non-streaming.
        
        Uses process_streaming internally but collects nothing in memory.
        """
        self.process_streaming(source, config, sink, progress)
    
    def process_streaming(
        self,
        source: FrameSource,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Process frames one at a time, pushing each to sink immediately.
        
        Peak CPU RAM = O(1 frame) after this change (was O(total_frames)).
        """
        progress = progress or NullProgress()
        
        # ── Validate ───────────────────────────────────────────────
        num_frames = len(source)
        validate_frame_count(num_frames, config.mode)
        
        # ── Estimate & check memory ─────────────────────────────────
        resources = estimate_gpu_memory(
            source.resolution,
            config.kv_cache_sliding_window + config.num_scale_frames,
            "sdpa" if config.use_sdpa else "flashinfer",
            self.dtype,
        )
        self._check_memory_budget(resources)
        
        # ── Prepare ─────────────────────────────────────────────────
        self.clean()
        model = self.model
        images = self._load_all_frames_to_cpu(source)  # ← FIXME: see optimization below
        progress.on_start(num_frames, {"mode": config.mode, "resources": resources})
        
        # ── Infer with per-frame sink ───────────────────────────────
        self._run_inference_with_sink(model, images, config, sink, progress, num_frames)
        
        progress.on_complete({"frames_processed": num_frames})
    
    def clean(self) -> None:
        """Clean KV cache + free CUDA memory. Call between every request."""
        if self._model is not None:
            self._model.clean_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def gpu_stats(self) -> dict:
        """Snapshot of GPU state for health checks."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "model_loaded": self._model_loaded,
            "backend": "flashinfer" if not getattr(self._model, 'use_sdpa', False) else "sdpa",
        }
    
    # ── Private ──────────────────────────────────────────────────────
    
    def _resolve_dtype(self, dtype) -> torch.dtype: ...
    def _load_model(self) -> torch.nn.Module: ...
    def _check_memory_budget(self, estimated: dict) -> None: ...
    def _load_all_frames_to_cpu(self, source: FrameSource) -> torch.Tensor: ...
    
    def _run_inference_with_sink(
        self, model, images, config, sink, progress, num_frames
    ) -> None:
        """Core inference loop — per-frame output → sink immediately."""
        # ... implementation that calls sink.write_frame(i, frame_dict)
        # for each frame, never accumulating more than one frame's data
```

**Key design decisions for SceneProcessor:**

1. **Lazy model loading**: Model isn't loaded on `__init__` but on first `process()` call. This allows instant import and health checks without GPU.

2. **`config` is a dataclass, not argparse**: Framework-agnostic. Both CLI and web layer construct the same config object.

3. **All output via `sink`**: The processor never writes to disk or network. The sink does. This makes the processor testable with a `NullSink` and testable with an `InMemorySink` that captures predictions for assertions.

4. **`clean()` is explicit**: Not a context manager (avoids async complexity). Web layer must call `clean()` between requests.

5. **No async in the processor**: All async adaptation happens in the web layer via `asyncio.to_thread()`.

**⚠️ Optimization deferred:** The `_load_all_frames_to_cpu()` call above still loads ALL frames to CPU RAM. This is the Phase 1.2 incremental improvement. The initial implementation loads all frames (acceptable for MVP with a frame limit), then a follow-up PR makes `process_streaming()` accept a `FrameSource` that truly streams frame-by-frame. The interface already supports this because `FrameSource.__iter__` yields one frame at a time — we just need to change the loop to pull from the iterator rather than indexing into a tensor.

**Tests for Module 1B:**

| Test | What it verifies | Needs GPU? | Mock strategy |
|---|---|---|---|
| `test_processor_init_no_gpu` | Constructor works without CUDA | No | CPU device |
| `test_processor_lazy_model_load` | Model not loaded until first process() | No | Mock `load_model` |
| `test_processor_config_dataclass` | All config fields have defaults, no argparse needed | No | Pure Python |
| `test_processor_clean_between_requests` | `clean()` zeros KV cache state | Yes | Real model, 2-frame test |
| `test_process_with_null_sink` | Processing completes without error, sink receives all frames | Yes | NullSink, 10-frame tensor |
| `test_process_with_inmemory_sink` | Sink captures correct predictions | Yes | InMemorySink, verify keys/shapes |
| `test_process_streaming_memory` | Peak CPU RAM is O(1 frame) not O(N) | Yes | `tracemalloc` or `psutil` |
| `test_frame_limit_streaming_rejected` | 2000 frames → ValueError in streaming mode | No | Mock source |
| `test_frame_limit_windowed_accepted` | 2000 frames → OK in windowed mode | Yes | Real model |
| `test_gpu_stats_returns_dict` | `gpu_stats` returns expected keys | No (if CUDA) | CPU returns `cuda_available: False` |
| `test_progress_callbacks_fire` | All progress stages called in order | Yes | CallbackProgress, 5-frame test |
| `test_error_during_inference_cleans_up` | Exception → KV cache cleaned, sink.close() called | Yes | Inject error mid-sequence |

### Module 1C: In-Memory Video Decode (`lingbot_map/utils/load_fn.py` — MODIFY)

**Add one new function.** Everything else in this file is untouched.

```python
def load_and_preprocess_video_stream(
    video_source: str | bytes,
    *,
    fps: int = 5,
    image_size: int = 518,
    patch_size: int = 14,
    max_frames: int | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Stream-decode video, yielding (frame_idx, tensor[1, 3, H, W]) tuples.
    
    Never holds more than one frame in memory at a time.
    
    Args:
        video_source: File path (str) or in-memory bytes.
        fps: Target extraction frame rate.
        image_size: Width for resize. Height derived from aspect ratio.
        patch_size: Height rounded to multiple of this.
        max_frames: Stop after this many frames (None = all).
    
    Yields:
        (global_frame_index, preprocessed_tensor) where tensor is [1, 3, H, W]
        in range [0, 1], on CPU.
    """
    # Implementation: cv2.VideoCapture if str, BytesIO→tempfile→cv2 if bytes
    # Or use PyAV for true in-memory decode of bytes
    ...
```

**Test 1C:**
- `test_stream_yields_correct_count`: Known 30-frame video → 30 yields
- `test_stream_max_frames`: max_frames=10 → 10 yields
- `test_stream_tensor_shape`: Each tensor is [1, 3, H, W] with H%14==0
- `test_stream_values_in_range`: All values in [0, 1]
- `test_stream_bytes_input`: bytes input produces same frames as file input
- `test_stream_memory`: `tracemalloc` confirms O(1 frame) peak

### Module 1D: Gaussian Splatting Pipeline (`lingbot_map/gs/` — NEW)

Three modules, each independently testable:

```
lingbot_map/gs/
├── __init__.py
├── initializer.py    # depth + poses → Gaussian point cloud
├── trainer.py        # Gaussian point cloud → trained .ply
└── export.py         # .ply → .ply.gz / .splat
```

**`initializer.py` contract:**
```python
def initialize_gaussians_from_predictions(
    predictions_dir: str,          # path to per-frame NPZ directory
    *,
    confidence_threshold: float = 0.5,
    voxel_size: float = 0.02,      # meters (deduplication)
    max_gaussians: int = 5_000_000,
    subsample: int = 4,            # stride for spatial subsampling
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz [N,3], rgb [N,3]) for Gaussian initialization.
    
    Reads depth + extrinsic from per-frame NPZ files.
    Unprojects each frame, deduplicates via voxel grid, 
    returns the merged point cloud.
    
    Does NOT require GPU. Pure NumPy.
    """
```

**`trainer.py` contract:**
```python
def train_gaussians(
    xyz: np.ndarray,               # [N, 3] initial positions
    rgb: np.ndarray,               # [N, 3] initial colors
    images_dir: str,               # original frames (for training views)
    poses: np.ndarray,             # [S, 4, 4] camera-to-world matrices
    intrinsics: np.ndarray,        # [S, 3, 3]
    *,
    output_path: str,              # where to save .ply
    iterations: int = 15_000,
    densify_until: int = 7_000,
    device: str = "cuda",
    fastgs_pruning: bool = True,   # apply FastGS pruning strategy
    gs_scale_offload: bool = False, # CPU offload for >2M Gaussians
) -> str:
    """Train 3DGS and return path to the saved .ply file.
    
    Wraps gsplat (or diff-gaussian-rasterization) training loop.
    Uses depth-initialized positions as starting point (no random init).
    """
```

**`export.py` contract:**
```python
def compress_ply(
    ply_path: str,
    output_path: str | None = None,
    *,
    format: str = "ply.gz",        # "ply.gz" | "splat"
    quantize_positions: bool = True,
    remove_low_opacity: float = 0.005,
) -> str:
    """Compress .ply file for web delivery. Returns output path."""
```

**Tests for Module 1D:**

| Test | What it verifies | GPU? |
|---|---|---|
| `test_initialize_from_two_frames` | Two overlapping frames → deduplicated points | No |
| `test_initialize_voxel_dedup` | Same 3D point from two frames → one Gaussian | No |
| `test_initialize_confidence_filter` | Low-conf points excluded | No |
| `test_initialize_max_gaussians` | Hard cap on Gaussians respected | No |
| `test_initialize_no_gpu_required` | Runs on CPU without CUDA import | No |
| `test_train_small_scene` | 100 Gaussians, 5 views → converges | Yes (small) |
| `test_train_output_is_valid_ply` | Output .ply loads with `trimesh.load()` | Yes |
| `test_compress_reduces_size` | .ply.gz < .ply by at least 30% | No |
| `test_compress_roundtrip` | Compress → decompress → same point count | No |
| `test_export_splat_format` | .splat output valid for gsplat.js | No |

### Module 1E: Segmentation Pipeline (`lingbot_map/segmentation.py` — NEW)

```python
class SegmentationPipeline:
    """Frame-by-frame segmentation with optional 3D lifting.
    
    USAGE:
        seg = SegmentationPipeline(model="sam2.1_hiera_tiny")
        for frame_idx, frame in enumerate(frames):
            mask = seg.segment_frame(frame)          # 2D mask
            seg.save_mask(mask, frame_idx, out_dir)  # persist
    
        # After all frames + depth available:
        labeled_3d = seg.lift_to_3d(mask_dir, predictions_dir)
        seg.export_semantic_glb(labeled_3d, "scene_semantic.glb")
    """
    
    def __init__(self, model: str = "sam2.1_hiera_tiny", device: str = "cuda"):
        self.model = self._load_seg_model(model, device)
    
    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Segment a single BGR frame → uint8 mask [H, W]."""
        ...
    
    def segment_batch(self, frames: np.ndarray) -> np.ndarray:
        """Segment multiple frames at once → [N, H, W] uint8."""
        ...
    
    @staticmethod
    def lift_to_3d(
        mask_dir: str,              # per-frame .png masks
        predictions_dir: str,       # per-frame .npz with depth+extrinsic
        target_classes: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """2D masks → 3D labeled point cloud.
        
        Returns: {class_id: [N, 3] world XYZ array}
        Pure NumPy. No GPU needed.
        """
        ...
    
    @staticmethod
    def export_semantic_glb(
        labeled_points: dict[int, np.ndarray],
        output_path: str,
        class_names: dict[int, str] | None = None,
        downsample: int = 4,
    ) -> str:
        """Export labeled 3D points as GLB with per-class colors."""
        ...
```

**Tests for Module 1E:**

| Test | What it verifies | GPU? |
|---|---|---|
| `test_segment_frame_shape` | Output mask matches input frame H×W | Yes |
| `test_segment_frame_values` | Mask values are uint8 class IDs | Yes |
| `test_segment_batch_consistency` | Batch output = concatenated individual outputs | Yes |
| `test_lift_to_3d_two_frames` | Two frames → merged labeled point cloud | No |
| `test_lift_to_3d_class_separation` | Different classes → different dict keys | No |
| `test_export_semantic_glb_valid` | Output loads in trimesh with per-geometry colors | No |
| `test_export_roundtrip` | Export → load → same point count per class | No |
| `test_no_gpu_for_lift_and_export` | lift_to_3d + export run on CPU-only machine | No |

---

## Layer 2: Web API

### Module 2A: Pydantic Models (`webapp/models.py` — NEW)

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class SceneStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class SceneMetadata(BaseModel):
    scene_id: str
    status: SceneStatus
    created_at: datetime
    total_frames: int | None = None
    frames_processed: int = 0
    estimated_duration_s: float | None = None
    config: dict = {}
    result_urls: dict[str, str] = {}   # e.g., {"preview_glb": "...", "full_glb": "...", "gs_ply": "..."}
    error_message: str | None = None

class ProcessRequest(BaseModel):
    """Request body for POST /api/scenes"""
    fps: int = Field(default=5, ge=1, le=30)
    mode: str = Field(default="auto", pattern="^(auto|streaming|windowed)$")
    max_frames: int = Field(default=10000, le=50000)
    quality: str = Field(default="balanced", pattern="^(fast|balanced|accurate)$")

class HealthResponse(BaseModel):
    status: str
    gpu: dict
    queue_depth: int
    active_scene: str | None
```

### Module 2B: Scene Store (`webapp/scene_store.py` — NEW)

```python
from abc import ABC, abstractmethod

class SceneStore(ABC):
    """Storage abstraction for scene artifacts and metadata.
    
    Two implementations: LocalDiskSceneStore (MVP) and S3SceneStore (prod).
    The web layer only depends on this ABC.
    """
    
    @abstractmethod
    def create(self, scene_id: str, metadata: SceneMetadata) -> None: ...
    
    @abstractmethod
    def get(self, scene_id: str) -> SceneMetadata: ...
    
    @abstractmethod
    def update(self, scene_id: str, **kwargs) -> None: ...
    
    @abstractmethod
    def get_predictions_dir(self, scene_id: str) -> str: ...
    
    @abstractmethod
    def get_artifact_path(self, scene_id: str, artifact: str) -> str: ...
    
    @abstractmethod
    def delete(self, scene_id: str) -> None: ...
    
    @abstractmethod
    def list_expired(self, max_age_days: int = 7) -> list[str]: ...


class LocalDiskSceneStore(SceneStore):
    """Stores everything under /data/scenes/{scene_id}/.
    
    Layout:
        {base_dir}/{scene_id}/
        ├── meta.json
        ├── predictions/
        │   ├── frame_000000.npz
        │   ├── frame_000001.npz
        │   └── ...
        ├── masks/
        │   ├── frame_000000.png
        │   └── ...
        ├── preview.glb
        ├── full.glb
        └── scene.ply.gz
    """
    def __init__(self, base_dir: str = "/data/scenes"): ...
```

**Tests for Module 2B:**

| Test | What it verifies |
|---|---|
| `test_create_and_get` | Round-trip: create → get returns same metadata |
| `test_update_increments_progress` | update(frames_processed=5) → get shows 5 |
| `test_predictions_dir_path` | Path follows expected layout |
| `test_delete_removes_all` | After delete, get raises KeyError |
| `test_list_expired` | Old scenes returned, new ones not |

### Module 2C: Background Worker (`webapp/worker.py` — NEW)

```python
import asyncio
from lingbot_map.processor import SceneProcessor, ProcessorConfig
from lingbot_map.io_protocol import VideoFileSource, NPZDirectorySink, CallbackProgress
from webapp.scene_store import SceneStore

class InferenceWorker:
    """Serializes GPU access: one inference at a time, FIFO queue."""
    
    def __init__(self, processor: SceneProcessor, store: SceneStore):
        self.processor = processor
        self.store = store
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active: str | None = None
    
    async def enqueue(self, scene_id: str, video_path: str, config: ProcessorConfig):
        """Add a scene to the processing queue. Non-blocking."""
        await self._queue.put((scene_id, video_path, config))
    
    async def run(self):
        """Main loop: dequeue → process → update store. Run as background task."""
        while True:
            scene_id, video_path, config = await self._queue.get()
            self._active = scene_id
            try:
                await self._process_one(scene_id, video_path, config)
            except Exception as e:
                self.store.update(scene_id, status="error", error_message=str(e))
            finally:
                self.processor.clean()
                self._active = None
                self._queue.task_done()
    
    async def _process_one(self, scene_id, video_path, config):
        self.store.update(scene_id, status="processing")
        
        source = VideoFileSource(video_path, fps=config.get("fps", 5))
        sink = NPZDirectorySink(self.store.get_predictions_dir(scene_id))
        
        def on_frame(frame_idx, stage, fps):
            self.store.update(scene_id, frames_processed=frame_idx)
        
        progress = CallbackProgress(on_frame_fn=on_frame)
        
        # Run blocking inference in thread pool
        await asyncio.to_thread(
            self.processor.process_streaming,
            source, config, sink, progress,
        )
        
        self.store.update(scene_id, status="ready")
        
        # Fire-and-forget post-processing tasks
        asyncio.create_task(self._generate_glb(scene_id))
        asyncio.create_task(self._generate_gs(scene_id))
```

**Tests for Module 2C:**

| Test | What it verifies | Mock strategy |
|---|---|---|
| `test_enqueue_updates_status` | Scene status = "queued" after enqueue | Mock processor, real store |
| `test_sequential_processing` | Two scenes processed in order, not concurrent | Mock processor with sleep |
| `test_clean_called_between` | processor.clean() called between scenes | Spy on processor.clean |
| `test_error_updates_metadata` | Exception → store shows status="error" with message | Mock processor that raises |
| `test_fifo_ordering` | Queue processed in FIFO order | Three scenes, timestamps |

### Module 2D: FastAPI Application (`webapp/app.py` — NEW)

```python
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, BackgroundTasks
from webapp.models import ProcessRequest, SceneMetadata, HealthResponse
from webapp.worker import InferenceWorker
from webapp.scene_store import SceneStore

def create_app(processor, store) -> FastAPI:
    app = FastAPI(title="LingBot-MAP API")
    worker = InferenceWorker(processor, store)
    
    @app.on_event("startup")
    async def startup():
        asyncio.create_task(worker.run())
    
    @app.post("/api/scenes", status_code=201)
    async def create_scene(
        video: UploadFile = File(...),
        fps: int = Form(5),
        mode: str = Form("auto"),
    ) -> SceneMetadata:
        ...
    
    @app.get("/api/scenes/{scene_id}")
    async def get_scene(scene_id: str) -> SceneMetadata:
        ...
    
    @app.get("/api/scenes/{scene_id}/preview.glb")
    async def get_preview(scene_id: str):
        ...
    
    @app.get("/api/scenes/{scene_id}/gaussian_splat")
    async def get_gs(scene_id: str):
        ...
    
    @app.websocket("/api/scenes/{scene_id}/stream")
    async def stream_progress(websocket: WebSocket, scene_id: str):
        ...
    
    @app.get("/api/health")
    async def health() -> HealthResponse:
        ...
    
    @app.delete("/api/scenes/{scene_id}")
    async def delete_scene(scene_id: str):
        ...
    
    return app
```

**Tests for Module 2D:**

| Test | What it verifies | Test client |
|---|---|---|
| `test_create_scene_returns_201` | Upload → 201 with scene_id | `TestClient` |
| `test_get_scene_returns_status` | GET → SceneMetadata with status | `TestClient` |
| `test_health_returns_gpu_info` | /api/health → HealthResponse schema | `TestClient` |
| `test_delete_cleans_up` | DELETE → subsequent GET returns 404 | `TestClient` |
| `test_websocket_streams_progress` | WS connects → receives progress messages | `TestClient.websocket_connect` |
| `test_invalid_file_rejected` | Upload non-video → 400 | `TestClient` |
| `test_missing_file_rejected` | POST without file → 422 | `TestClient` |

---

## Integration Tests (Cross-Module)

These tests verify that modules compose correctly. They use real implementations but small test data.

### IT-1: End-to-End with 30-Frame Test Video

```
Setup:    30-frame test video (synthetic or courthouse subset)
          Processor with real model on GPU

Test:     Create VideoFileSource → NPZDirectorySink → process_streaming()
          → read back NPZ files → verify:
            - All 30 frames written
            - Each frame has depth, pose_enc, extrinsic, intrinsic
            - Shapes correct
            - No NaNs in depth
            - Extrinsics are valid (rotation part has det ≈ 1)

Requires: GPU with model checkpoint
```

### IT-2: Streaming Memory Test

```
Setup:    300-frame test video
          tracemalloc or psutil monitoring

Test:     process_streaming() with NPZDirectorySink
          → peak CPU RAM < 2 GB (currently ~5 GB without fix, ~200 MB target)
          → no monotonic growth in memory over time

Requires: GPU with model checkpoint, psutil
```

### IT-3: WebSocket Streaming Test

```
Setup:    FastAPI TestClient + WebSocket
          NullProcessor (mock) that yields fake frames

Test:     Connect WebSocket → POST scene → worker processes
          → WebSocket receives frame messages in order
          → WebSocket receives "done" message
          → Connection closes cleanly

Requires: No GPU (mock processor)
```

### IT-4: GS Pipeline Test (Small)

```
Setup:    5-frame scene, known camera poses
          Train GS with 100 iterations

Test:     initialize → train → export → compress
          → .ply.gz loads in trimesh
          → Point count matches initialization (within 10%)
          → File size < 5 MB

Requires: GPU (small, any CUDA GPU)
```

### IT-5: Segmentation + GS Combination

```
Setup:    10-frame scene
          SegmentationPipeline with mock segmenter (returns random masks)
          GS pipeline with real training

Test:     Segment all frames → lift to 3D → train GS with per-class initialization
          → GS .ply contains Gaussian count proportional to class frequency
          → Semantic GLB has per-class colors

Requires: GPU (small)
```

---

## Dependency Graph

```
F-1 (_set_skip_append) ─────────────────────────────────────────────┐
F-2 (frame limit)      ─────────────────────────────────────────────┤
F-3 (memory budget)    ─────────────────────────────────────────────┤
                                                                     │
1A (io_protocol.py) ─── no dependencies ─────────────────────────────┤
1C (load_fn stream)  ─── depends on existing load_fn ────────────────┤
                                                                     │
1B (processor.py) ───── depends on 1A, F-1, F-2, F-3, inference.py ─┤
                        does NOT depend on 1C (FrameSource is injected)
                                                                     │
1D (gs/) ────────────── depends on predictions format (NPZ) ─────────┤
                        does NOT depend on 1A or 1B                   │
                                                                     │
1E (segmentation.py) ── depends on predictions format ───────────────┤
                        does NOT depend on 1A or 1B                   │
                                                                     │
2A (models.py) ──────── no dependencies ─────────────────────────────┤
2B (scene_store.py) ── no dependencies ──────────────────────────────┤
2C (worker.py) ──────── depends on 1B, 1A, 2B, 1D, 1E ──────────────┤
2D (app.py) ─────────── depends on 2A, 2B, 2C ───────────────────────┤
```

**Key insight:** 1D (GS) and 1E (segmentation) are fully independent of 1B (processor). They only need the NPZ file format that the processor produces. This means:
- GS training can be developed and tested with pre-saved NPZ files, no processor needed
- Segmentation can be developed with any depth + pose source
- The web worker orchestrates them but they don't import each other

---

## File Manifest: Every File Created or Modified

### Modified files (Layer 0 — surgical changes)

| File | Change | Lines |
|---|---|---|
| `lingbot_map/models/gct_stream.py` | try/finally around `_set_skip_append` in `inference_streaming()` loop | +2 |
| `lingbot_map/models/gct_stream_window.py` | Same try/finally in both `inference_streaming()` and `inference_windowed()` | +4 |
| `lingbot_map/inference.py` | Add `validate_frame_count()`, `estimate_gpu_memory()` | +60 |
| `lingbot_map/utils/load_fn.py` | Add `load_and_preprocess_video_stream()` | +50 |

### New files (Layer 1 — core library)

| File | Purpose | Approx lines |
|---|---|---|
| `lingbot_map/io_protocol.py` | FrameSource, PredictionSink, ProgressReporter ABCs + implementations | ~350 |
| `lingbot_map/processor.py` | SceneProcessor class | ~400 |
| `lingbot_map/gs/__init__.py` | Empty init | 0 |
| `lingbot_map/gs/initializer.py` | Depth → Gaussian init | ~200 |
| `lingbot_map/gs/trainer.py` | Wrap gsplat training | ~250 |
| `lingbot_map/gs/export.py` | .ply compression/format conversion | ~100 |
| `lingbot_map/segmentation.py` | SegmentationPipeline + lift/export | ~300 |

### New files (Layer 2 — web API)

| File | Purpose | Approx lines |
|---|---|---|
| `webapp/__init__.py` | Empty init | 0 |
| `webapp/models.py` | Pydantic schemas | ~80 |
| `webapp/scene_store.py` | SceneStore ABC + LocalDisk impl | ~150 |
| `webapp/worker.py` | InferenceWorker with FIFO queue | ~150 |
| `webapp/app.py` | FastAPI application factory | ~200 |

### New files (Layer 3 — not covered in detail here)

| File | Purpose |
|---|---|
| `frontend/index.html` | Upload UI + GS viewer + point cloud viewer |
| `frontend/gs-viewer.js` | gsplat.js integration |
| `Dockerfile` | CUDA 12.x base, all deps |
| `docker-compose.yml` | App + optional Redis for multi-worker |

### Test files

| File | What it tests |
|---|---|
| `tests/test_safety_fixes.py` | F-1, F-2, F-3 |
| `tests/test_io_protocol.py` | 1A (all sources and sinks) |
| `tests/test_processor.py` | 1B (unit + integration) |
| `tests/test_load_fn_stream.py` | 1C |
| `tests/test_gs_initializer.py` | 1D initializer (CPU tests) |
| `tests/test_gs_trainer.py` | 1D trainer (requires GPU) |
| `tests/test_gs_export.py` | 1D export |
| `tests/test_segmentation.py` | 1E |
| `tests/test_scene_store.py` | 2B |
| `tests/test_worker.py` | 2C |
| `tests/test_app.py` | 2D |
| `tests/test_integration.py` | IT-1 through IT-5 |

---

## What NOT to Do (Anti-Patterns to Avoid)

| Anti-pattern | Why | Do this instead |
|---|---|---|
| **Import FastAPI in processor.py** | Couples core logic to web framework. Makes testing require HTTP. | Processor takes `FrameSource`/`PredictionSink` ABCs. Web layer implements them. |
| **Add async to processor** | PyTorch inference is synchronous. async adds complexity for zero benefit. | Wrap in `asyncio.to_thread()` in the web layer. |
| **Store state on the processor between requests** | KV cache poisoning if clean() not called. | Always `processor.clean()` in worker's `finally` block. |
| **Load model in processor.__init__** | Blocks import, prevents health checks without GPU. | Lazy-load on first `.process()` call. |
| **Hardcode paths in core library** | Can't test without filesystem setup. | Inject paths via config or use `Path` objects. |
| **Catch-all except in processor** | Swallows CUDA errors, hides bugs. | Let exceptions propagate to worker; worker updates scene status to "error". |
| **Create abstract bases for everything prematurely** | Over-engineering. Only abstract what has >1 implementation. | FrameSource/PredictionSink need ABCs (multiple impls). Processor does not (only one). |
| **Skip the safety fixes to save time** | `_set_skip_append` bug causes silent corruption. Frame limit prevents OOM. | These are 1-2 days total. Do them first. |
