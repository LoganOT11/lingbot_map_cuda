"""SceneProcessor — central orchestration for LingBot-MAP inference.

Wraps the GCTStream model lifecycle (load, warm, infer, clean) behind a
framework-agnostic interface.  All I/O flows through :class:`FrameSource`,
:class:`PredictionSink`, and :class:`ProgressReporter` — the processor
never touches the filesystem or network directly.

Usage::

    from lingbot_map.processor import SceneProcessor, ProcessorConfig
    from lingbot_map.io_protocol import TensorFrameSource, NPZDirectorySink, NullProgress

    processor = SceneProcessor(model_path="/data/model.pt")
    config = ProcessorConfig()

    source = TensorFrameSource(my_tensor)       # [S, 3, H, W]
    sink   = NPZDirectorySink("/data/scenes/001")

    processor.process(source, config, sink, NullProgress())

    processor.clean()  # mandatory between sequences
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

from lingbot_map.inference import (
    estimate_gpu_memory,
    load_model,
    postprocess,
    validate_frame_count,
)
from lingbot_map.io_protocol import (
    FrameSource,
    NullProgress,
    PredictionSink,
    ProgressReporter,
)

_log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Thresholds for auto mode selection (see ProcessorConfig._resolve_mode)
_AUTO_STREAMING_MAX_FRAMES = 200
_AUTO_WINDOWED_MIN_FRAMES = 500

# GPU memory warning threshold (GB)
_GPU_MEM_WARN_GB = 10.0

# bf16 minimum compute capability (SM 7.5 = Turing)
_BF16_MIN_CAPABILITY = 8


# ────────────────────────────────────────────────────────────────────────────
# ProcessorConfig
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ProcessorConfig:
    """All tunables for a single processing run.  Framework-agnostic — no
    argparse, no env vars, no FastAPI coupling.  Build this however you
    want (CLI flags, JSON request body, YAML) and pass it in."""

    # ── Mode ────────────────────────────────────────────────────────────
    mode: str = "auto"  # "auto" | "streaming" | "windowed"

    # ── Input ───────────────────────────────────────────────────────────
    image_size: int = 518
    patch_size: int = 14

    # ── Streaming / KV cache ────────────────────────────────────────────
    num_scale_frames: int = 8
    keyframe_interval: int = 1  # 1 = every frame; >1 = throttled
    kv_cache_sliding_window: int = 64
    max_frame_num: int = 1024

    # ── Windowed mode ───────────────────────────────────────────────────
    window_size: int = 64  # keyframes per window
    overlap_size: int | None = None  # actual-frame overlap (default = scale_frames)
    overlap_keyframes: int | None = None  # keyframe-based overlap (preferred)
    scale_mode: str = "median"

    # ── Quality / speed ─────────────────────────────────────────────────
    camera_num_iterations: int = 2  # 2 = balanced; 4 = most accurate
    enable_3d_rope: bool = True
    compile: bool = False  # torch.compile warmup (adds ~30s startup)

    # ── Backend ─────────────────────────────────────────────────────────
    use_sdpa: bool = False  # True = SDPA; False = FlashInfer (faster)

    # ── Output control ──────────────────────────────────────────────────
    output_device: str = "cpu"
    include_world_points: bool = False  # compute world_points server-side
    include_images: bool = False  # echo input images in output

    # ── Safety ──────────────────────────────────────────────────────────
    max_frames_streaming: int = 1100  # below FlashInfer special-page cap
    max_frames_windowed: int = 50000

    def _resolve_mode(self, num_frames: int) -> str:
        """Resolve ``"auto"`` to a concrete mode based on frame count."""
        if self.mode != "auto":
            return self.mode
        if num_frames <= _AUTO_STREAMING_MAX_FRAMES:
            return "streaming"
        if num_frames >= _AUTO_WINDOWED_MIN_FRAMES:
            return "windowed"
        # 201–499 frames: still streaming, but auto-increase keyframe
        # interval to keep KV cache bounded
        return "streaming"

    def _auto_keyframe_interval(self, num_frames: int) -> int:
        """For sequences > 200 frames, automatically throttle keyframes."""
        if self.keyframe_interval > 1:
            return self.keyframe_interval  # user set explicitly
        if num_frames <= _AUTO_STREAMING_MAX_FRAMES:
            return 1
        # One keyframe per ~80 actual frames, keeping total cached ≤ ~200
        return max(1, num_frames // 200)


# ────────────────────────────────────────────────────────────────────────────
# SceneProcessor
# ────────────────────────────────────────────────────────────────────────────


class SceneProcessor:
    """Owns a GCTStream model and orchestrates inference across sequences.

    The model is loaded lazily on first use.  Between sequences, call
    :meth:`clean` to reset the KV cache and free GPU memory.

    All inference methods are **synchronous** (PyTorch is synchronous).
    Wrap them in ``asyncio.to_thread()`` at the web layer.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        compile: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self._dtype_str = dtype
        self._should_compile = compile

        self._model: torch.nn.Module | None = None
        self._model_loaded = False
        self._dtype: torch.dtype | None = None
        self._active_backend: str = "unknown"

    # ── Public API ─────────────────────────────────────────────────────────

    def process(
        self,
        source: FrameSource,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Process all frames, pushing results to *sink*.

        This is a **blocking** call that returns after every frame has been
        processed and written to the sink.  Peak CPU RAM is O(1 frame)."""
        progress = progress or NullProgress()
        num_frames = len(source)
        mode = self._validate_and_resolve(source, config, progress, num_frames)
        model = self._ensure_model(config)

        images = self._load_all_frames(source, progress, num_frames)
        try:
            sink.write_metadata(self._build_metadata(num_frames, config, mode))
            self._run_inference(model, images, config, sink, progress, num_frames, mode)
        finally:
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_streaming(
        self,
        source: FrameSource,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Same as :meth:`process` — alias for clarity.

        The name emphasises that results are streamed to the sink
        frame-by-frame rather than accumulated in memory."""
        self.process(source, config, sink, progress)

    def clean(self) -> None:
        """Reset KV cache and free GPU memory.  **Must** be called between
        every processing sequence."""
        if self._model is not None:
            self._model.clean_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def gpu_stats(self) -> dict:
        """Snapshot of GPU state.  Safe to call any time."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free, total = (x / 1e9 for x in torch.cuda.mem_get_info())
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "free_gb": round(free, 2),
            "total_gb": round(total, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "model_loaded": self._model_loaded,
            "backend": self._active_backend,
            "dtype": str(self._dtype) if self._dtype else "unknown",
        }

    @staticmethod
    def estimate_resources(
        resolution: tuple[int, int],
        num_frames_in_window: int,
        backend: str = "flashinfer",
        dtype: torch.dtype | None = None,
    ) -> dict:
        """Static memory estimate (delegates to inference.py)."""
        return estimate_gpu_memory(
            resolution, num_frames_in_window, backend=backend, dtype=dtype
        )

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # ── Private: validation & preparation ───────────────────────────────────

    def _validate_and_resolve(
        self,
        source: FrameSource,
        config: ProcessorConfig,
        progress: ProgressReporter,
        num_frames: int,
    ) -> str:
        """Validate inputs, resolve mode, check memory budget."""
        mode = config._resolve_mode(num_frames)
        validate_frame_count(
            num_frames,
            mode,
            max_frames_streaming=config.max_frames_streaming,
            max_frames_windowed=config.max_frames_windowed,
        )

        # Estimate memory and warn if tight
        window_frames = config.kv_cache_sliding_window + config.num_scale_frames
        backend = "sdpa" if config.use_sdpa else "flashinfer"
        est = self.estimate_resources(
            source.resolution, window_frames, backend=backend, dtype=self._dtype
        )
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb < est["total_gb"]:
                _log.warning(
                    "GPU may OOM: need ~%.1f GB, only %.1f GB free. "
                    "Consider: --use_sdpa, --keyframe_interval 4, "
                    "or --mode windowed --window_size 32",
                    est["total_gb"],
                    free_gb,
                )

        progress.on_start(num_frames, {"mode": mode, "resources": est, "config": _config_summary(config)})
        return mode

    def _ensure_model(self, config: ProcessorConfig) -> torch.nn.Module:
        """Load model on first call; validate dtype."""
        if self._model is None:
            self._load_model(config)
        return self._model

    def _load_model(self, config: ProcessorConfig) -> None:
        """Load model from disk, move to device, select dtype."""
        _log.info("Loading model from %s ...", self.model_path)
        t0 = time.time()

        # Resolve dtype
        if self._dtype_str == "auto":
            if self.device.type == "cuda":
                cap = torch.cuda.get_device_capability()[0]
                self._dtype = torch.bfloat16 if cap >= _BF16_MIN_CAPABILITY else torch.float16
            else:
                self._dtype = torch.float32
        elif isinstance(self._dtype_str, torch.dtype):
            self._dtype = self._dtype_str
        else:
            self._dtype = getattr(torch, self._dtype_str)

        self._active_backend = "sdpa" if config.use_sdpa else "flashinfer"

        self._model = load_model(
            self.model_path,
            self.device,
            mode=config._resolve_mode(100),  # dummy — only matters for windowed import
            image_size=config.image_size,
            patch_size=config.patch_size,
            enable_3d_rope=config.enable_3d_rope,
            max_frame_num=config.max_frame_num,
            kv_cache_sliding_window=config.kv_cache_sliding_window,
            num_scale_frames=config.num_scale_frames,
            use_sdpa=config.use_sdpa,
            camera_num_iterations=config.camera_num_iterations,
        )

        # Cast aggregator to inference dtype (heads stay fp32)
        if self._dtype != torch.float32 and self._model.aggregator is not None:
            _log.info("Casting aggregator to %s", self._dtype)
            self._model.aggregator = self._model.aggregator.to(dtype=self._dtype)

        self._model_loaded = True
        _log.info("Model loaded in %.1f s (dtype=%s, backend=%s)",
                   time.time() - t0, self._dtype, self._active_backend)

    def _load_all_frames(
        self, source: FrameSource, progress: ProgressReporter, num_frames: int
    ) -> torch.Tensor:
        """Load all frames into a CPU tensor [S, 3, H, W].

        Note: For very long videos, this still loads all frames into CPU RAM.
        A follow-up optimisation (Phase 1.2) will stream from the source
        directly into the inference loop, keeping peak RAM at O(1 frame).
        The FrameSource iterator protocol already supports this — the
        inference loop just needs to pull from the iterator instead of
        indexing into a tensor.
        """
        _log.info("Loading %d frames into CPU memory...", num_frames)
        frames = list(source)
        tensor = torch.cat(frames, dim=0)  # [S, 3, H, W]
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor

    # ── Private: inference dispatch ─────────────────────────────────────────

    def _run_inference(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter,
        num_frames: int,
        mode: str,
    ) -> None:
        """Dispatch to the correct inference mode and stream results to sink."""
        self.clean()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        output_device = torch.device(config.output_device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._dtype):
            if mode == "streaming":
                self._infer_streaming(model, images, config, sink, progress,
                                      num_frames, output_device)
            else:
                self._infer_windowed(model, images, config, sink, progress,
                                     num_frames, output_device)

        elapsed = time.time() - t0
        fps = num_frames / elapsed if elapsed > 0 else float("inf")
        _log.info("Inference done: %d frames in %.1f s (%.1f FPS)",
                   num_frames, elapsed, fps)

        progress.on_complete({
            "frames_processed": num_frames,
            "duration_s": round(elapsed, 1),
            "fps": round(fps, 1),
        })

    # ── Streaming inference ─────────────────────────────────────────────────

    def _infer_streaming(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter,
        num_frames: int,
        output_device: torch.device,
    ) -> None:
        """Frame-by-frame inference with KV cache, pushing each frame to sink."""
        scale_frames = min(config.num_scale_frames, num_frames)
        if scale_frames >= num_frames:
            scale_frames = max(1, num_frames - 1)

        kf_interval = config._auto_keyframe_interval(num_frames)
        model_device = next(model.parameters()).device

        # ── Phase 1: scale frames (bidirectional attention) ─────────────
        scale_images = images[:scale_frames].to(model_device, non_blocking=True)
        torch.compiler.cudagraph_mark_step_begin()
        scale_out = model.forward(
            scale_images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
        self._emit_scale_frames(scale_out, scale_frames, config, sink, progress)
        del scale_out, scale_images

        # ── Phase 2: streaming frames ───────────────────────────────────
        for i in range(scale_frames, num_frames):
            frame_img = images[i:i + 1].to(model_device, non_blocking=True)
            is_keyframe = (kf_interval <= 1) or ((i - scale_frames) % kf_interval == 0)

            if not is_keyframe:
                model._set_skip_append(True)
            try:
                torch.compiler.cudagraph_mark_step_begin()
                frame_out = model.forward(
                    frame_img,
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            finally:
                if not is_keyframe:
                    model._set_skip_append(False)

            # Convert pose_enc → extrinsic/intrinsic, move to CPU
            frame_dict = self._postprocess_frame(frame_out, images[i:i + 1], config)
            sink.write_frame(i, frame_dict)
            progress.on_frame(i, "stream")

            del frame_out, frame_dict

    # ── Windowed inference ──────────────────────────────────────────────────

    def _infer_windowed(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter,
        num_frames: int,
        output_device: torch.device,
    ) -> None:
        """Process frames in overlapping windows, pushing each to sink."""
        ws = min(config.num_scale_frames, num_frames)
        kf_int = max(config._auto_keyframe_interval(num_frames), 1)

        # Compute window geometry
        if config.overlap_keyframes is not None:
            eff_overlap = max(ws, config.overlap_keyframes * kf_int)
        elif config.overlap_size is not None:
            eff_overlap = config.overlap_size
        else:
            eff_overlap = ws
        eff_overlap = min(eff_overlap, num_frames - 1) if num_frames > 1 else 0

        phase2_kf = max(config.window_size - ws, 0)
        phase2_frames = phase2_kf * kf_int
        actual_window = ws + phase2_frames
        eff_window = min(actual_window, num_frames)
        step = max(eff_window - eff_overlap, 1)

        model_device = next(model.parameters()).device
        global_frame_idx = 0  # tracks absolute frame index across windows

        for start in range(0, num_frames, step):
            end = min(start + eff_window, num_frames)
            window_images = images[start:end].to(model_device, non_blocking=True)
            window_len = end - start
            window_scale = min(ws, window_len)

            model.clean_kv_cache()

            # ── Phase 1: scale ──────────────────────────────────────────
            scale_out = model.forward(
                window_images[:window_scale],
                num_frame_for_scale=window_scale,
                num_frame_per_block=window_scale,
                causal_inference=True,
            )
            for j in range(window_scale):
                frame_dict = self._postprocess_frame(
                    scale_out, window_images[j:j + 1], config, frame_idx=j
                )
                sink.write_frame(global_frame_idx, frame_dict)
                progress.on_frame(global_frame_idx, "window")
                global_frame_idx += 1
            del scale_out

            # ── Phase 2: stream ─────────────────────────────────────────
            for j in range(window_scale, window_len):
                is_kf = (kf_int <= 1) or ((j - window_scale) % kf_int == 0)

                if not is_kf:
                    model._set_skip_append(True)
                try:
                    frame_out = model.forward(
                        window_images[j:j + 1],
                        num_frame_for_scale=window_scale,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                finally:
                    if not is_kf:
                        model._set_skip_append(False)

                frame_dict = self._postprocess_frame(
                    frame_out, window_images[j:j + 1], config, frame_idx=j
                )
                sink.write_frame(global_frame_idx, frame_dict)
                progress.on_frame(global_frame_idx, "window")
                global_frame_idx += 1
                del frame_out, frame_dict

            del window_images
            if end >= num_frames:
                break

    # ── Post-processing helpers ─────────────────────────────────────────────

    def _emit_scale_frames(
        self,
        scale_out: dict,
        count: int,
        config: ProcessorConfig,
        sink: PredictionSink,
        progress: ProgressReporter,
    ) -> None:
        """Extract per-frame predictions from the scale batch and push to sink.

        The scale batch is processed with bidirectional attention, producing
        a single output dict with leading batch dim = count.  We slice each
        key along the sequence dimension and emit individually.
        """
        # The scale output has shape [1, count, ...] for batched keys.
        # Slice the count dim to get per-frame dicts.
        for i in range(count):
            frame = {}
            for key in ("pose_enc", "depth", "depth_conf"):
                if key in scale_out:
                    val = scale_out[key]
                    # val shape: [1, count, ...] → slice to [1, 1, ...]
                    if isinstance(val, torch.Tensor) and val.dim() >= 2:
                        frame[key] = val[:, i:i + 1]
                    else:
                        frame[key] = val

            # Postprocess: pose_enc → extrinsic + intrinsic
            # We don't have per-frame images here (they're batched),
            # so we do a lightweight postprocess that doesn't need images.
            frame = self._postprocess_frame(frame, None, config, frame_idx=i)
            sink.write_frame(i, frame)
            progress.on_frame(i, "scale")

    def _postprocess_frame(
        self,
        frame_out: dict,
        images: torch.Tensor | None,
        config: ProcessorConfig,
        frame_idx: int = 0,
    ) -> dict:
        """Convert a single-frame model output to a clean CPU dict.

        - Converts pose_enc → extrinsic + intrinsic
        - Moves all tensors to CPU
        - Strips keys per config (include_world_points, include_images)
        - Returns plain dict of numpy arrays
        """
        result: dict = {}

        # ── pose_enc → extrinsic + intrinsic ────────────────────────────
        if "pose_enc" in frame_out:
            pose = frame_out["pose_enc"]
            if isinstance(pose, torch.Tensor):
                if images is not None and images.dim() >= 3:
                    h, w = images.shape[-2], images.shape[-1]
                else:
                    # Fallback: use depth map dimensions
                    depth = frame_out.get("depth")
                    if depth is not None and isinstance(depth, torch.Tensor):
                        h, w = depth.shape[2], depth.shape[3]
                    else:
                        h, w = (294, 518)  # last resort default

                from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
                from lingbot_map.utils.geometry import closed_form_inverse_se3_general

                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, (h, w))

                # Convert world→camera to camera→world
                ext_4x4 = torch.zeros(
                    (*extrinsic.shape[:-2], 4, 4),
                    device=extrinsic.device,
                    dtype=extrinsic.dtype,
                )
                ext_4x4[..., :3, :4] = extrinsic
                ext_4x4[..., 3, 3] = 1.0
                ext_4x4 = closed_form_inverse_se3_general(ext_4x4)
                extrinsic = ext_4x4[..., :3, :4]

                result["extrinsic"] = self._to_numpy(extrinsic)
                result["intrinsic"] = self._to_numpy(intrinsic)

        # ── Depth ───────────────────────────────────────────────────────
        for key in ("depth", "depth_conf"):
            if key in frame_out:
                result[key] = self._to_numpy(frame_out[key])

        # ── Optional: world_points ──────────────────────────────────────
        if config.include_world_points and "world_points" in frame_out:
            result["world_points"] = self._to_numpy(frame_out["world_points"])
        if config.include_world_points and "world_points_conf" in frame_out:
            result["world_points_conf"] = self._to_numpy(frame_out["world_points_conf"])

        # ── Optional: echoed images ─────────────────────────────────────
        if config.include_images and images is not None:
            result["images"] = self._to_numpy(images)

        return result

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Move tensor to CPU and convert to numpy, squeezing batch dims."""
        import numpy as np

        arr = tensor.detach().cpu().numpy()
        # Squeeze leading singleton batch dims (common pattern: [1,1,...] → [...])
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return np.asarray(arr)

    def _build_metadata(
        self, num_frames: int, config: ProcessorConfig, mode: str
    ) -> dict:
        import datetime

        return {
            "num_frames": num_frames,
            "resolution": list(config._last_resolution) if hasattr(config, "_last_resolution") else [],
            "mode": mode,
            "config": _config_summary(config),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _config_summary(config: ProcessorConfig) -> dict:
    """Return a JSON-serialisable subset of config for metadata."""
    return {
        "mode": config.mode,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "num_scale_frames": config.num_scale_frames,
        "keyframe_interval": config.keyframe_interval,
        "kv_cache_sliding_window": config.kv_cache_sliding_window,
        "window_size": config.window_size,
        "camera_num_iterations": config.camera_num_iterations,
        "use_sdpa": config.use_sdpa,
        "compile": config.compile,
    }
