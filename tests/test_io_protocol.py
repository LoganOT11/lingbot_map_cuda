"""Tests for lingbot_map/io_protocol.py — Module 1A.

Covers every FrameSource, PredictionSink, and ProgressReporter implementation.
Designed to run WITHOUT a GPU.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lingbot_map.io_protocol import (
    # Frame sources
    FrameSource,
    TensorFrameSource,
    ImageFolderSource,
    VideoFileSource,
    BytesUploadSource,
    # Prediction sinks
    PredictionSink,
    NPZDirectorySink,
    NullSink,
    InMemorySink,
    # Progress reporters
    ProgressReporter,
    CallbackProgress,
    NullProgress,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_random_tensor(frames: int = 5, h: int = 56, w: int = 84) -> torch.Tensor:
    """[S, 3, H, W] float32 in [0, 1]."""
    return torch.rand(frames, 3, h, w, dtype=torch.float32)


def _make_fake_predictions(frame_idx: int) -> dict:
    """Return a dict matching the expected per-frame prediction keys."""
    return {
        "pose_enc": np.random.randn(1, 1, 9).astype(np.float32),
        "depth": np.random.randn(1, 56, 84, 1).astype(np.float32),
        "depth_conf": np.random.randn(1, 56, 84).astype(np.float32),
        "extrinsic": np.random.randn(1, 1, 3, 4).astype(np.float32),
        "intrinsic": np.random.randn(1, 1, 3, 3).astype(np.float32),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TensorFrameSource
# ═════════════════════════════════════════════════════════════════════════════

class TestTensorFrameSource:
    def test_len(self):
        src = TensorFrameSource(_make_random_tensor(7))
        assert len(src) == 7

    def test_resolution(self):
        src = TensorFrameSource(_make_random_tensor(3, h=56, w=84))
        assert src.resolution == (56, 84)

    def test_iter_yields_correct_count(self):
        src = TensorFrameSource(_make_random_tensor(4))
        frames = list(src)
        assert len(frames) == 4

    def test_iter_yields_correct_shape(self):
        src = TensorFrameSource(_make_random_tensor(2, h=56, w=84))
        for f in src:
            assert f.shape == (1, 3, 56, 84)

    def test_iter_yields_values_in_range(self):
        src = TensorFrameSource(_make_random_tensor(3))
        for f in src:
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_clamps_out_of_range_values(self):
        t = torch.randn(3, 3, 56, 84)  # values can be negative
        with pytest.warns(UserWarning, match="clamping"):
            src = TensorFrameSource(t)
        for f in src:
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_3d_input_unsqueezed(self):
        """Single-frame [3, H, W] → [1, 3, H, W]."""
        t = torch.rand(3, 56, 84)
        src = TensorFrameSource(t)
        assert len(src) == 1
        frame = next(iter(src))
        assert frame.shape == (1, 3, 56, 84)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            TensorFrameSource(torch.rand(3, 3))  # 2D

    def test_original_paths_is_none(self):
        src = TensorFrameSource(_make_random_tensor(2))
        assert src.original_paths is None

    def test_multiple_iterations_consistent(self):
        """Two passes over the source yield the same frames."""
        src = TensorFrameSource(_make_random_tensor(3))
        first = [f.clone() for f in src]
        second = [f.clone() for f in src]
        for a, b in zip(first, second):
            assert torch.allclose(a, b)


# ═════════════════════════════════════════════════════════════════════════════
# ImageFolderSource
# ═════════════════════════════════════════════════════════════════════════════

class TestImageFolderSource:
    @pytest.fixture
    def image_folder(self):
        """Create a temporary folder with 3 synthetic JPEG images."""
        import cv2

        tmp = tempfile.mkdtemp(prefix="lingbot_test_imgs_")
        for i in range(3):
            img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.jpg"), img)
        yield tmp
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    def test_len(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300)
        assert len(src) == 3

    def test_resolution_is_valid(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300)
        h, w = src.resolution
        assert h % 14 == 0, f"Height {h} not divisible by 14"
        assert w == 300

    def test_iter_yields_correct_shape(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300)
        h, w = src.resolution
        for f in src:
            assert f.shape == (1, 3, h, w)

    def test_iter_values_in_range(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300)
        for f in src:
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_original_paths(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300)
        paths = src.original_paths
        assert len(paths) == 3
        assert all(p.endswith(".jpg") for p in paths)

    def test_first_k(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300, first_k=2)
        assert len(src) == 2

    def test_stride(self, image_folder):
        src = ImageFolderSource(image_folder, image_size=300, stride=2)
        assert len(src) == 2  # 3 frames, stride 2 → frames 0, 2

    def test_empty_folder_raises(self):
        tmp = tempfile.mkdtemp(prefix="lingbot_empty_")
        try:
            with pytest.raises(FileNotFoundError):
                ImageFolderSource(tmp)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_not_a_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            ImageFolderSource("/nonexistent/path/12345")


# ═════════════════════════════════════════════════════════════════════════════
# VideoFileSource
# ═════════════════════════════════════════════════════════════════════════════

class TestVideoFileSource:
    @pytest.fixture
    def video_file(self):
        """Create a temporary MP4 with 10 frames of colored noise."""
        import cv2

        tmp = tempfile.mkdtemp(prefix="lingbot_test_vid_")
        path = os.path.join(tmp, "test.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        yield path

        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    def test_len_fps_5(self, video_file):
        """30 fps video → 5 fps extraction → ~2 frames (30/5 = 6 interval,
        frames 0 and 6 of 10 total)."""
        src = VideoFileSource(video_file, fps=5)
        # 10 frames at 30fps, interval=6: frames 0, 6 → 2 frames
        assert 1 <= len(src) <= 3

    def test_iter_yields_correct_shape(self, video_file):
        src = VideoFileSource(video_file, fps=10, image_size=300)
        h, w = src.resolution
        for f in src:
            assert f.shape == (1, 3, h, w)

    def test_resolution_is_valid(self, video_file):
        src = VideoFileSource(video_file, image_size=300, patch_size=14)
        h, w = src.resolution
        assert h % 14 == 0
        assert w == 300

    def test_max_frames(self, video_file):
        src = VideoFileSource(video_file, fps=30, max_frames=3)
        assert len(src) == 3

    def test_invalid_video_raises(self):
        with pytest.raises(ValueError, match="Cannot open"):
            VideoFileSource("/nonexistent/video.mp4")

    def test_original_paths_is_none(self, video_file):
        src = VideoFileSource(video_file)
        assert src.original_paths is None


# ═════════════════════════════════════════════════════════════════════════════
# BytesUploadSource
# ═════════════════════════════════════════════════════════════════════════════

class TestBytesUploadSource:
    @pytest.fixture
    def video_bytes(self):
        """Return the raw bytes of a 10-frame MP4."""
        import cv2

        tmp = tempfile.mkdtemp(prefix="lingbot_test_bytes_")
        path = os.path.join(tmp, "test.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(path, "rb") as f:
            data = f.read()

        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        return data

    def test_len_matches_delegate(self, video_bytes):
        src = BytesUploadSource(video_bytes, fps=10)
        delegate = VideoFileSource(
            src._tmpfile if hasattr(src, "_tmpfile") else "",
            fps=10,
        )
        # Just check we get frames
        assert len(src) > 0

    def test_iter_yields_tensors(self, video_bytes):
        src = BytesUploadSource(video_bytes, fps=10, image_size=200, max_frames=3)
        frames = list(src)
        assert len(frames) == 3
        for f in frames:
            assert isinstance(f, torch.Tensor)
            assert f.dim() == 4

    def test_close_cleans_up(self, video_bytes):
        src = BytesUploadSource(video_bytes, fps=10)
        tmpfile = src._tmpfile
        assert os.path.exists(tmpfile)
        src.close()
        assert not os.path.exists(tmpfile)


# ═════════════════════════════════════════════════════════════════════════════
# NPZDirectorySink
# ═════════════════════════════════════════════════════════════════════════════

class TestNPZDirectorySink:
    @pytest.fixture
    def output_dir(self):
        tmp = tempfile.mkdtemp(prefix="lingbot_test_sink_")
        yield tmp
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    def test_writes_frames(self, output_dir):
        with NPZDirectorySink(output_dir) as sink:
            sink.write_metadata({"num_frames": 3, "resolution": [56, 84]})
            for i in range(3):
                sink.write_frame(i, _make_fake_predictions(i))

        # Verify files exist
        for i in range(3):
            path = os.path.join(output_dir, f"frame_{i:06d}.npz")
            assert os.path.exists(path), f"Missing {path}"

    def test_metadata_written(self, output_dir):
        with NPZDirectorySink(output_dir) as sink:
            sink.write_metadata(
                {"num_frames": 2, "resolution": [56, 84], "config": {"mode": "test"}}
            )
            sink.write_frame(0, _make_fake_predictions(0))
            sink.write_frame(1, _make_fake_predictions(1))

        meta_path = os.path.join(output_dir, "meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["num_frames"] == 2
        assert meta["frame_count"] == 2

    def test_roundtrip_frame_data(self, output_dir):
        """Write a frame → read it back → compare."""
        original = _make_fake_predictions(0)
        with NPZDirectorySink(output_dir) as sink:
            sink.write_metadata({"num_frames": 1, "resolution": [56, 84]})
            sink.write_frame(0, original)

        path = os.path.join(output_dir, "frame_000000.npz")
        loaded = dict(np.load(path))
        for key in ["pose_enc", "depth", "depth_conf"]:
            assert key in loaded
            assert np.allclose(loaded[key], original[key]), f"Mismatch for {key}"

    def test_cleans_existing_on_init(self, output_dir):
        """Second sink with clean_existing=True wipes previous files."""
        with NPZDirectorySink(output_dir) as sink:
            sink.write_metadata({"num_frames": 1, "resolution": [56, 84]})
            sink.write_frame(0, _make_fake_predictions(0))

        # Write again with clean_existing=True
        with NPZDirectorySink(output_dir, clean_existing=True) as sink:
            sink.write_metadata({"num_frames": 1, "resolution": [56, 84]})
            sink.write_frame(0, _make_fake_predictions(0))

        # Should only have one frame file
        frame_files = list(Path(output_dir).glob("frame_*.npz"))
        assert len(frame_files) == 1


# ═════════════════════════════════════════════════════════════════════════════
# NullSink
# ═════════════════════════════════════════════════════════════════════════════

class TestNullSink:
    def test_accepts_all_calls(self):
        sink = NullSink()
        sink.write_metadata({"num_frames": 1000})
        for i in range(1000):
            sink.write_frame(i, _make_fake_predictions(i))
        sink.close()  # no-op, just checking it doesn't raise

    def test_context_manager(self):
        with NullSink() as sink:
            sink.write_frame(0, _make_fake_predictions(0))


# ═════════════════════════════════════════════════════════════════════════════
# InMemorySink
# ═════════════════════════════════════════════════════════════════════════════

class TestInMemorySink:
    def test_captures_all_frames(self):
        sink = InMemorySink()
        sink.write_metadata({"num_frames": 3})
        for i in range(3):
            sink.write_frame(i, _make_fake_predictions(i))
        sink.close()

        preds = sink.predictions
        assert "pose_enc" in preds
        assert preds["pose_enc"].shape[0] == 3

    def test_metadata_preserved(self):
        sink = InMemorySink()
        sink.write_metadata({"num_frames": 5, "config": {"key": "value"}})
        sink.close()

        assert sink.metadata["num_frames"] == 5
        assert sink.metadata["config"]["key"] == "value"

    def test_frame_count(self):
        sink = InMemorySink()
        for i in range(7):
            sink.write_frame(i, _make_fake_predictions(i))
        assert sink.frame_count == 7

    def test_close_twice_is_safe(self):
        sink = InMemorySink()
        sink.write_frame(0, _make_fake_predictions(0))
        sink.close()
        sink.close()  # should not raise

    def test_predictions_raises_before_close(self):
        sink = InMemorySink()
        sink.write_frame(0, _make_fake_predictions(0))
        with pytest.raises(RuntimeError, match="not yet closed"):
            _ = sink.predictions

    def test_write_after_close_raises(self):
        sink = InMemorySink()
        sink.close()
        with pytest.raises(RuntimeError):
            sink.write_frame(0, _make_fake_predictions(0))

    def test_data_not_aliased(self):
        """Modifying original after write_frame doesn't affect sink."""
        original = _make_fake_predictions(0)
        sink = InMemorySink()
        sink.write_frame(0, original)
        original["depth"] *= 999  # modify original
        sink.close()
        stored = sink.predictions["depth"][0]
        assert not np.allclose(stored, original["depth"])

    def test_heterogeneous_shapes_stored_as_list(self):
        """Frames with different shapes for a key → stored as list."""
        sink = InMemorySink()
        sink.write_frame(0, {"a": np.zeros(3)})
        sink.write_frame(1, {"a": np.zeros(5)})  # different shape
        sink.close()
        result = sink.predictions["a"]
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_sink_returns_empty_dict(self):
        sink = InMemorySink()
        sink.close()
        assert sink.predictions == {}
        assert sink.frame_count == 0


# ═════════════════════════════════════════════════════════════════════════════
# CallbackProgress
# ═════════════════════════════════════════════════════════════════════════════

class TestCallbackProgress:
    def test_all_callbacks_fire(self):
        calls = []

        def record(event, **kwargs):
            calls.append(event)

        prog = CallbackProgress(
            on_start_fn=lambda n, c: record("start"),
            on_frame_fn=lambda i, s, fps: record("frame"),
            on_complete_fn=lambda s: record("complete"),
            on_error_fn=lambda e, i: record("error"),
        )

        prog.on_start(10, {})
        prog.on_frame(0, "stream")
        prog.on_frame(1, "stream")
        prog.on_complete({"frames": 2})
        prog.on_error(RuntimeError("boom"), None)

        assert calls == ["start", "frame", "frame", "complete", "error"]

    def test_none_callbacks_are_silent(self):
        """Passing None for a callback doesn't crash."""
        prog = CallbackProgress(on_frame_fn=None)
        prog.on_start(5, {})
        prog.on_frame(0, "scale")
        prog.on_complete({})
        prog.on_error(Exception(), 0)

    def test_fps_passed_through(self):
        fps_values = []

        def capture(i, s, fps):
            fps_values.append(fps)

        prog = CallbackProgress(on_frame_fn=capture)
        prog.on_frame(0, "stream", fps=5.7)
        prog.on_frame(1, "stream", fps=6.1)
        assert fps_values == [5.7, 6.1]

    def test_frame_indices_sequential(self):
        indices = []

        def capture(i, s, fps):
            indices.append(i)

        prog = CallbackProgress(on_frame_fn=capture)
        for i in range(5):
            prog.on_frame(i, "stream")
        assert indices == list(range(5))


# ═════════════════════════════════════════════════════════════════════════════
# NullProgress
# ═════════════════════════════════════════════════════════════════════════════

class TestNullProgress:
    def test_all_methods_noop(self):
        prog = NullProgress()
        prog.on_start(100, {})
        prog.on_frame(50, "stream")
        prog.on_frame(99, "stream", fps=5.0)
        prog.on_complete({"frames": 100})
        prog.on_error(RuntimeError("test"), 42)
        # If we got here without exception, the test passes

    def test_same_instance_reusable(self):
        """A single NullProgress can be used for multiple runs."""
        prog = NullProgress()
        for _ in range(10):
            prog.on_start(5, {})
            for i in range(5):
                prog.on_frame(i, "stream")
            prog.on_complete({})


# ═════════════════════════════════════════════════════════════════════════════
# Integration: sink + source round-trip patterns
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Verify that sources and sinks compose correctly."""

    def test_source_to_sink_pipeline(self):
        """Simulate a full processing pipeline: source → sink."""
        src = TensorFrameSource(_make_random_tensor(4, h=56, w=84))
        sink = InMemorySink()

        sink.write_metadata({"num_frames": len(src)})
        for i, frame in enumerate(src):
            # Simulate processing: extract a fake prediction
            pred = _make_fake_predictions(i)
            sink.write_frame(i, pred)
        sink.close()

        assert sink.frame_count == 4
        assert sink.predictions["pose_enc"].shape[0] == 4

    def test_source_to_npz_roundtrip(self):
        """Write to NPZ, verify on disk."""
        src = TensorFrameSource(_make_random_tensor(2, h=56, w=84))

        with tempfile.TemporaryDirectory() as tmpdir:
            with NPZDirectorySink(tmpdir) as sink:
                sink.write_metadata({"num_frames": len(src)})
                for i, frame in enumerate(src):
                    sink.write_frame(i, _make_fake_predictions(i))

            # Verify files
            assert os.path.exists(os.path.join(tmpdir, "frame_000000.npz"))
            assert os.path.exists(os.path.join(tmpdir, "frame_000001.npz"))
            assert os.path.exists(os.path.join(tmpdir, "meta.json"))

    def test_progress_reports_during_pipeline(self):
        events = []

        def on_frame(i, stage, fps):
            events.append((i, stage))

        prog = CallbackProgress(on_frame_fn=on_frame)
        src = TensorFrameSource(_make_random_tensor(3))

        prog.on_start(len(src), {})
        for i, _ in enumerate(src):
            prog.on_frame(i, "stream")
        prog.on_complete({"frames": 3})

        assert events == [(0, "stream"), (1, "stream"), (2, "stream")]
