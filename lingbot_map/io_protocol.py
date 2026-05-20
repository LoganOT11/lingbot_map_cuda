"""I/O abstractions for LingBot-MAP processing pipelines.

Every I/O operation in the processor flows through these protocols.
The processor never touches the filesystem, network, or GPU device
directly — it only talks to FrameSource, PredictionSink, and
ProgressReporter instances.  This decoupling means:

- The processor is testable with InMemorySink (no disk I/O)
- The processor is benchmarkable with NullSink (no overhead)
- WebSocket streaming is just another PredictionSink implementation
- Frame sources can be swapped (video file, image folder, bytes upload)
  without changing a single line of processor code

All classes in this module are CPU-only.  No CUDA required.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch

# ────────────────────────────────────────────────────────────────────────────
# Frame Source  —  produces preprocessed image tensors one at a time
# ────────────────────────────────────────────────────────────────────────────


class FrameSource(ABC):
    """Produces preprocessed image tensors on demand.

    Designed for streaming: frames are yielded one at a time, never all
    loaded into memory at once.  The caller controls pacing by iterating.

    Every concrete implementation must guarantee that yielded tensors:
    - Have shape ``[1, 3, H, W]``
    - Are in range ``[0.0, 1.0]``
    - Live on CPU
    - Are yielded in temporal order
    """

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames.  Must return the same value before and
        during iteration."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield preprocessed frames one at a time.

        Each yielded tensor has shape ``[1, 3, H, W]``, dtype float32,
        values in [0, 1], on CPU.
        """
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """``(height, width)`` of every yielded frame."""
        ...

    @property
    def original_paths(self) -> list[str] | None:
        """Original file paths, if the source was constructed from files.
        May be ``None`` for in-memory or synthetic sources."""
        return None


# ── Concrete frame sources ──────────────────────────────────────────────────


class ImageFolderSource(FrameSource):
    """Load pre-extracted image files from a folder on disk.

    All images are preprocessed on construction (resize + crop to canonical
    format) and stored as a single tensor.  For very large folders (>1000
    images), consider :class:`StreamingImageFolderSource` instead to avoid
    holding all frames in RAM.
    """

    def __init__(
        self,
        folder: str | Path,
        *,
        image_ext: str = ".jpg,.jpeg,.png,.JPG,.PNG",
        image_size: int = 518,
        patch_size: int = 14,
        first_k: int | None = None,
        stride: int = 1,
    ) -> None:
        import glob as _glob

        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {folder}")

        exts = [e.strip() for e in image_ext.split(",") if e.strip()]
        paths: list[str] = []
        for ext in exts:
            paths.extend(str(p) for p in folder.glob(f"*{ext}"))
            if not ext.startswith("."):
                paths.extend(str(p) for p in folder.glob(f"*{ext.upper()}"))
        paths = sorted(set(paths))

        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")

        if first_k is not None and first_k > 0:
            paths = paths[:first_k]
        if stride > 1:
            paths = paths[::stride]

        self._paths = paths
        self._tensor = self._load_and_preprocess(paths, image_size, patch_size)
        self._resolution: tuple[int, int] = tuple(self._tensor.shape[-2:])

    @staticmethod
    def _load_and_preprocess(
        paths: list[str], image_size: int, patch_size: int
    ) -> torch.Tensor:
        """Thin wrapper around the existing loader in load_fn.py."""
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(
            paths, mode="crop", image_size=image_size, patch_size=patch_size
        )

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self._tensor[i : i + 1]  # [1, 3, H, W]

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def original_paths(self) -> list[str]:
        return list(self._paths)


class TensorFrameSource(FrameSource):
    """Wrap an existing preprocessed tensor as a FrameSource.

    The tensor must have shape ``[S, 3, H, W]`` with values in [0, 1].
    This is the simplest source — ideal for tests and benchmarks.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # [3, H, W] → [1, 3, H, W]
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected [S, 3, H, W] tensor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels, got {tensor.shape[1]}"
            )
        self._tensor = tensor.float().cpu()
        # Clamp to [0, 1] if needed
        if self._tensor.min() < 0 or self._tensor.max() > 1:
            import warnings

            warnings.warn(
                "TensorFrameSource: values outside [0, 1] — clamping."
            )
            self._tensor = self._tensor.clamp(0, 1)

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self._tensor[i : i + 1].clone()

    @property
    def resolution(self) -> tuple[int, int]:
        h, w = self._tensor.shape[2], self._tensor.shape[3]
        return (h, w)


class VideoFileSource(FrameSource):
    """Decode video frames on-the-fly using OpenCV.

    Frames are decoded, preprocessed, and yielded one at a time.
    Peak memory is O(1 frame) no matter how long the video.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        fps: int = 5,
        image_size: int = 518,
        patch_size: int = 14,
        max_frames: int | None = None,
    ) -> None:
        import cv2

        self._image_size = image_size
        self._patch_size = patch_size

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._interval = max(1, round(src_fps / fps)) if fps > 0 else 1
        # Count how many frames we'll actually extract
        count = 0
        idx = 0
        while True:
            if idx % self._interval == 0:
                ret = cap.grab()
                if not ret:
                    break
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            else:
                if not cap.grab():
                    break
            idx += 1
        cap.release()

        self._total_frames = count
        self._video_path = str(video_path)
        self._max_frames = max_frames

        # Determine resolution from first frame
        cap2 = cv2.VideoCapture(str(video_path))
        ret, frame = cap2.read()
        cap2.release()
        if not ret:
            raise ValueError("Video has no readable frames")

        h, w = frame.shape[:2]
        new_width = image_size
        new_height = round(h * (new_width / w) / patch_size) * patch_size
        self._resolution: tuple[int, int] = (new_height, new_width)

    def __len__(self) -> int:
        return self._total_frames

    def __iter__(self) -> Iterator[torch.Tensor]:
        import cv2

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video became unreadable: {self._video_path}")

        yielded = 0
        idx = 0
        try:
            while True:
                if idx % self._interval == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tensor = self._preprocess_frame(frame)
                    yielded += 1
                    yield tensor
                    if self._max_frames is not None and yielded >= self._max_frames:
                        break
                else:
                    if not cap.grab():
                        break
                idx += 1
        finally:
            cap.release()

    def _preprocess_frame(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """Resize + crop a single BGR frame → tensor [1, 3, H, W]."""
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        new_h, new_w = self._resolution

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # Center-crop height if it exceeds image_size
        if new_h > self._image_size:
            start_y = (new_h - self._image_size) // 2
            tensor = tensor[:, start_y : start_y + self._image_size, :]

        return tensor.unsqueeze(0)  # [1, 3, H, W]

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution


class BytesUploadSource(FrameSource):
    """Decode video from in-memory bytes (e.g. FastAPI UploadFile).

    Writes bytes to a temporary file, then delegates to VideoFileSource.
    For true zero-disk decode, use PyAV; this implementation prioritises
    compatibility (OpenCV is already a project dependency).
    """

    def __init__(
        self,
        data: bytes,
        *,
        fps: int = 5,
        image_size: int = 518,
        patch_size: int = 14,
        max_frames: int | None = None,
    ) -> None:
        self._data = data
        self._tmpdir = tempfile.TemporaryDirectory(prefix="lingbot_upload_")
        self._tmpfile = os.path.join(self._tmpdir.name, "upload.mp4")
        with open(self._tmpfile, "wb") as f:
            f.write(data)

        self._delegate = VideoFileSource(
            self._tmpfile,
            fps=fps,
            image_size=image_size,
            patch_size=patch_size,
            max_frames=max_frames,
        )

    def __len__(self) -> int:
        return len(self._delegate)

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self._delegate

    @property
    def resolution(self) -> tuple[int, int]:
        return self._delegate.resolution

    def close(self) -> None:
        """Clean up the temporary file.  Called automatically on GC."""
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Prediction Sink  —  receives per-frame predictions
# ────────────────────────────────────────────────────────────────────────────


class PredictionSink(ABC):
    """Receives per-frame prediction dictionaries.

    The processor calls :meth:`write_metadata` once at the start, then
    :meth:`write_frame` once per frame in order, and finally :meth:`close`.
    The sink is responsible for persistence, streaming, or discarding.

    All tensors passed to the sink are numpy arrays on CPU.
    """

    @abstractmethod
    def write_metadata(self, metadata: dict) -> None:
        """Called once before any frames.

        *metadata* should include at least::

            {
                "num_frames": int,
                "resolution": [H, W],
                "config": {...},
                "timestamp": "iso8601",
            }
        """
        ...

    @abstractmethod
    def write_frame(self, frame_idx: int, predictions: dict) -> None:
        """Called once per frame, in strictly increasing *frame_idx* order.

        *predictions* is a dict of numpy arrays.  Guaranteed keys:
        ``pose_enc``, ``depth``, ``depth_conf``.  Optional keys (depending
        on config): ``extrinsic``, ``intrinsic``, ``world_points``,
        ``world_points_conf``, ``images``.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Called after the last frame (or on error).  Flush, finalise."""
        ...

    def __enter__(self) -> PredictionSink:
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ── Concrete sinks ──────────────────────────────────────────────────────────


class NPZDirectorySink(PredictionSink):
    """Write each frame as ``frame_NNNNNN.npz`` in a directory.

    Uses a thread pool for parallel I/O — ``np.savez_compressed`` releases
    the GIL during compression, so multiple frames can be written
    concurrently.

    Directory layout::

        {output_dir}/
        ├── meta.json
        ├── frame_000000.npz
        ├── frame_000001.npz
        └── ...
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        num_workers: int = 8,
        clean_existing: bool = True,
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        if clean_existing:
            for old in self._dir.glob("frame_*.npz"):
                old.unlink()
            meta = self._dir / "meta.json"
            if meta.exists():
                meta.unlink()

        self._num_workers = min(num_workers, 32)
        self._pool: ThreadPoolExecutor | None = None
        self._metadata: dict = {}
        self._frame_count = 0

    def write_metadata(self, metadata: dict) -> None:
        self._metadata = dict(metadata)

    def write_frame(self, frame_idx: int, predictions: dict) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._num_workers)

        # Convert any torch tensors to numpy (belt-and-suspenders)
        clean: dict[str, np.ndarray] = {}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            clean[k] = np.asarray(v)

        path = self._dir / f"frame_{frame_idx:06d}.npz"
        self._pool.submit(self._write_one, path, clean)
        self._frame_count += 1

    @staticmethod
    def _write_one(path: Path, data: dict) -> None:
        np.savez_compressed(str(path), **data)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

        # Write metadata
        self._metadata.setdefault("frame_count", self._frame_count)
        meta_path = self._dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    @property
    def output_dir(self) -> Path:
        return self._dir


class NullSink(PredictionSink):
    """Discard all predictions.  For benchmarking throughput without I/O."""

    def write_metadata(self, metadata: dict) -> None:
        pass

    def write_frame(self, frame_idx: int, predictions: dict) -> None:
        pass

    def close(self) -> None:
        pass


class InMemorySink(PredictionSink):
    """Capture all predictions in memory.  For testing and debugging.

    After :meth:`close`, the full prediction dict is available as
    :attr:`predictions`, with all frames concatenated along axis 0.
    """

    def __init__(self) -> None:
        self._frames: list[dict] = []
        self._metadata: dict = {}
        self._closed = False

    def write_metadata(self, metadata: dict) -> None:
        self._metadata = dict(metadata)

    def write_frame(self, frame_idx: int, predictions: dict) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to closed InMemorySink")
        # Deep-copy the dict so caller can reuse arrays
        frame = {}
        for k, v in predictions.items():
            if isinstance(v, np.ndarray):
                frame[k] = v.copy()
            elif isinstance(v, torch.Tensor):
                frame[k] = v.cpu().numpy().copy()
            else:
                frame[k] = v
        self._frames.append(frame)

    def close(self) -> None:
        self._closed = True
        if not self._frames:
            self._predictions: dict = {}
            return

        keys = list(self._frames[0].keys())
        stacked: dict = {}
        for key in keys:
            arrays = [f[key] for f in self._frames]
            # All arrays for a given key must have the same shape except
            # possibly the leading (frame) dimension
            try:
                stacked[key] = np.stack(arrays, axis=0)
            except ValueError:
                # Heterogeneous shapes — store as list
                stacked[key] = arrays
        self._predictions = stacked

    @property
    def predictions(self) -> dict:
        """Full prediction dict (all frames stacked).  Available after close."""
        if not self._closed:
            raise RuntimeError("InMemorySink not yet closed")
        return self._predictions

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def frame_count(self) -> int:
        return len(self._frames)


# ────────────────────────────────────────────────────────────────────────────
# Progress Reporter  —  callbacks during processing
# ────────────────────────────────────────────────────────────────────────────


class ProgressReporter(ABC):
    """Receives progress updates from the processor.

    The processor calls these methods at well-defined points during
    inference.  The reporter decides how to surface the information
    (tqdm bar, log messages, WebSocket events, Prometheus gauge).
    """

    @abstractmethod
    def on_start(self, total_frames: int, config: dict) -> None:
        """Called before processing the first frame."""
        ...

    @abstractmethod
    def on_frame(
        self, frame_idx: int, stage: str, fps: float | None = None
    ) -> None:
        """Called after each frame is processed.

        Args:
            frame_idx: 0-based index of the frame just completed.
            stage: ``"scale"``, ``"stream"``, or ``"window"``.
            fps: Rolling average frames-per-second, if available.
        """
        ...

    @abstractmethod
    def on_complete(self, summary: dict) -> None:
        """Called after all frames are processed successfully."""
        ...

    @abstractmethod
    def on_error(self, error: Exception, frame_idx: int | None) -> None:
        """Called if processing fails.

        Args:
            error: The exception that terminated processing.
            frame_idx: The frame being processed when the error occurred,
                or ``None`` if the error happened before or after frames.
        """
        ...


# ── Concrete reporters ──────────────────────────────────────────────────────


class CallbackProgress(ProgressReporter):
    """A reporter backed by plain callables.

    Pass ``None`` for any callback you don't need::

        CallbackProgress(
            on_frame_fn=lambda i, stage, fps: print(f"Frame {i}"),
        )
    """

    def __init__(
        self,
        on_start_fn: Callable[[int, dict], None] | None = None,
        on_frame_fn: Callable[[int, str, float | None], None] | None = None,
        on_complete_fn: Callable[[dict], None] | None = None,
        on_error_fn: Callable[[Exception, int | None], None] | None = None,
    ) -> None:
        self._on_start = on_start_fn or (lambda *a: None)
        self._on_frame = on_frame_fn or (lambda *a: None)
        self._on_complete = on_complete_fn or (lambda *a: None)
        self._on_error = on_error_fn or (lambda *a: None)

    def on_start(self, total_frames: int, config: dict) -> None:
        self._on_start(total_frames, config)

    def on_frame(
        self, frame_idx: int, stage: str, fps: float | None = None
    ) -> None:
        self._on_frame(frame_idx, stage, fps)

    def on_complete(self, summary: dict) -> None:
        self._on_complete(summary)

    def on_error(self, error: Exception, frame_idx: int | None) -> None:
        self._on_error(error, frame_idx)


class NullProgress(ProgressReporter):
    """Silent reporter.  For tests and benchmarks."""

    def on_start(self, total_frames: int, config: dict) -> None:
        pass

    def on_frame(self, frame_idx: int, stage: str, fps: float | None = None) -> None:
        pass

    def on_complete(self, summary: dict) -> None:
        pass

    def on_error(self, error: Exception, frame_idx: int | None) -> None:
        pass
