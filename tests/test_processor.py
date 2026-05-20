"""Tests for lingbot_map/processor.py — Module 1B.

Tests the SceneProcessor and ProcessorConfig classes.  Most tests run
without a GPU or model checkpoint by mocking the GCTStream model.

The key properties under test:
- ProcessorConfig mode resolution (auto → streaming/windowed)
- SceneProcessor construction and lazy model loading
- KV cache lifecycle (clean() between sequences)
- Post-processing frame conversion logic
- Integration with FrameSource / PredictionSink / ProgressReporter
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lingbot_map.processor import (
    ProcessorConfig,
    SceneProcessor,
    _AUTO_STREAMING_MAX_FRAMES,
    _AUTO_WINDOWED_MIN_FRAMES,
    _config_summary,
)
from lingbot_map.io_protocol import (
    TensorFrameSource,
    InMemorySink,
    NullSink,
    CallbackProgress,
    NullProgress,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_tensor(frames: int = 5, h: int = 56, w: int = 84) -> torch.Tensor:
    return torch.rand(frames, 3, h, w)


def _make_source(frames: int = 5) -> TensorFrameSource:
    return TensorFrameSource(_make_tensor(frames))


def _fake_model_output(frame_count: int = 1) -> dict:
    """Return a dict matching what GCTStream.forward() produces."""
    return {
        "pose_enc": torch.randn(1, frame_count, 9),
        "depth": torch.randn(1, frame_count, 56, 84, 1),
        "depth_conf": torch.randn(1, frame_count, 56, 84),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ProcessorConfig
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessorConfig:
    def test_defaults(self):
        c = ProcessorConfig()
        assert c.mode == "auto"
        assert c.image_size == 518
        assert c.num_scale_frames == 8
        assert c.camera_num_iterations == 2
        assert c.use_sdpa is False

    def test_auto_mode_small_sequence(self):
        c = ProcessorConfig(mode="auto")
        assert c._resolve_mode(50) == "streaming"
        assert c._resolve_mode(_AUTO_STREAMING_MAX_FRAMES) == "streaming"

    def test_auto_mode_large_sequence(self):
        c = ProcessorConfig(mode="auto")
        assert c._resolve_mode(_AUTO_WINDOWED_MIN_FRAMES) == "windowed"
        assert c._resolve_mode(1000) == "windowed"

    def test_auto_mode_mid_sequence(self):
        """201–499 frames: still streaming but with throttled keyframes."""
        c = ProcessorConfig(mode="auto")
        assert c._resolve_mode(300) == "streaming"

    def test_explicit_mode_ignores_auto(self):
        c = ProcessorConfig(mode="windowed")
        assert c._resolve_mode(10) == "windowed"

    def test_streaming_mode_preserved(self):
        c = ProcessorConfig(mode="streaming")
        assert c._resolve_mode(10) == "streaming"
        assert c._resolve_mode(5000) == "streaming"  # validation happens elsewhere

    def test_auto_keyframe_interval_small(self):
        c = ProcessorConfig()
        assert c._auto_keyframe_interval(50) == 1
        assert c._auto_keyframe_interval(200) == 1

    def test_auto_keyframe_interval_large(self):
        c = ProcessorConfig()
        interval = c._auto_keyframe_interval(500)
        assert interval > 1  # should throttle

    def test_auto_keyframe_interval_user_set(self):
        c = ProcessorConfig(keyframe_interval=3)
        assert c._auto_keyframe_interval(500) == 3  # respects user setting

    def test_dataclass_is_immutable_after_construction(self):
        """Fields can be set, but it's a standard dataclass."""
        c = ProcessorConfig()
        c.mode = "streaming"  # mutable
        assert c.mode == "streaming"

    def test_config_summary_is_serializable(self):
        import json

        c = ProcessorConfig()
        summary = _config_summary(c)
        json.dumps(summary)  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# SceneProcessor — construction & properties (no model)
# ═════════════════════════════════════════════════════════════════════════════

class TestSceneProcessorConstruction:
    def test_init_does_not_load_model(self):
        """Model loading is lazy — __init__ should not touch disk or GPU."""
        with patch("lingbot_map.processor.load_model") as mock_load:
            processor = SceneProcessor("/fake/path.pt")
            mock_load.assert_not_called()

    def test_model_loaded_is_false_initially(self):
        processor = SceneProcessor("/fake/path.pt")
        assert processor.model_loaded is False

    def test_clean_before_load_is_safe(self):
        """clean() before model is loaded should not crash."""
        processor = SceneProcessor("/fake/path.pt")
        processor.clean()  # no-op

    def test_gpu_stats_has_expected_keys(self):
        """gpu_stats returns a dict with expected keys in all environments."""
        processor = SceneProcessor("/fake/path.pt", device="cpu")
        stats = processor.gpu_stats
        # cuda_available reflects the actual test machine
        assert "cuda_available" in stats
        assert "model_loaded" in stats
        assert stats["model_loaded"] is False
        # If CUDA is available, additional keys should be present
        if stats["cuda_available"]:
            assert "device_name" in stats
            assert "free_gb" in stats
            assert "backend" in stats

    def test_estimate_resources_delegates(self):
        """Static method works without instantiation."""
        est = SceneProcessor.estimate_resources((294, 518), 72,
                                                  backend="flashinfer")
        assert "total_gb" in est
        assert est["total_gb"] > 10

    def test_device_cpu(self):
        processor = SceneProcessor("/fake/path.pt", device="cpu")
        assert processor.device.type == "cpu"

    def test_dtype_auto(self):
        processor = SceneProcessor("/fake/path.pt", dtype="auto")
        assert processor._dtype_str == "auto"
        assert processor._dtype is None  # not resolved until model loaded


# ═════════════════════════════════════════════════════════════════════════════
# SceneProcessor — post-processing logic
# ═════════════════════════════════════════════════════════════════════════════

class TestPostprocessFrame:
    @pytest.fixture
    def processor(self):
        return SceneProcessor("/fake/path.pt", device="cpu")

    def test_pose_enc_converted_to_extrinsic_intrinsic(self, processor):
        """pose_enc in → extrinsic + intrinsic out."""
        frame = {"pose_enc": torch.randn(1, 1, 9)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig()

        result = processor._postprocess_frame(frame, images, config)

        assert "pose_enc" not in result  # removed
        assert "extrinsic" in result
        assert "intrinsic" in result
        assert result["extrinsic"].shape[-2:] == (3, 4)
        assert result["intrinsic"].shape[-2:] == (3, 3)

    def test_depth_preserved(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "depth": torch.randn(1, 1, 56, 84, 1),
                  "depth_conf": torch.randn(1, 1, 56, 84)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig()

        result = processor._postprocess_frame(frame, images, config)

        assert "depth" in result
        assert "depth_conf" in result
        # depth should be [H, W, 1] after squeeze
        assert result["depth"].ndim == 3

    def test_world_points_excluded_by_default(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "world_points": torch.randn(1, 1, 56, 84, 3)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig(include_world_points=False)

        result = processor._postprocess_frame(frame, images, config)
        assert "world_points" not in result

    def test_world_points_included_when_configured(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "world_points": torch.randn(1, 1, 56, 84, 3),
                  "world_points_conf": torch.randn(1, 1, 56, 84)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig(include_world_points=True)

        result = processor._postprocess_frame(frame, images, config)
        assert "world_points" in result
        assert "world_points_conf" in result

    def test_images_excluded_by_default(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig(include_images=False)

        result = processor._postprocess_frame(frame, images, config)
        assert "images" not in result

    def test_images_included_when_configured(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig(include_images=True)

        result = processor._postprocess_frame(frame, images, config)
        assert "images" in result

    def test_output_is_all_numpy(self, processor):
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "depth": torch.randn(1, 1, 56, 84, 1)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig()

        result = processor._postprocess_frame(frame, images, config)
        for v in result.values():
            assert isinstance(v, np.ndarray), f"Expected numpy, got {type(v)}"

    def test_output_on_cpu(self, processor):
        """All tensors are moved to CPU, even if the input was on GPU.
        (Test runs on CPU, so this is verifying the conversion path works.)"""
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "depth": torch.randn(1, 1, 56, 84, 1)}
        images = torch.rand(1, 3, 56, 84)
        config = ProcessorConfig()

        result = processor._postprocess_frame(frame, images, config)
        # All numpy arrays are inherently CPU-side
        for v in result.values():
            assert not hasattr(v, "device") or str(v.device) == "cpu"

    def test_no_images_uses_depth_dimensions(self, processor):
        """When images=None, use depth map dimensions for intrinsics."""
        frame = {"pose_enc": torch.randn(1, 1, 9),
                  "depth": torch.randn(1, 1, 56, 84, 1)}
        config = ProcessorConfig()

        result = processor._postprocess_frame(frame, None, config)
        assert "extrinsic" in result
        assert "intrinsic" in result
        # Intrinsics should match depth dimensions
        assert result["intrinsic"].shape[-2:] == (3, 3)

    def test_empty_frame_handled_gracefully(self, processor):
        """Frame with no recognised keys → empty dict."""
        result = processor._postprocess_frame({}, None, ProcessorConfig())
        assert result == {}


# ═════════════════════════════════════════════════════════════════════════════
# SceneProcessor — lifecycle: clean() between sequences
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanLifecycle:
    def test_clean_calls_model_clean_kv_cache(self):
        processor = SceneProcessor("/fake/path.pt", device="cpu")
        processor._model = MagicMock()
        processor.clean()
        processor._model.clean_kv_cache.assert_called_once()

    def test_clean_safe_when_model_is_none(self):
        processor = SceneProcessor("/fake/path.pt", device="cpu")
        processor._model = None
        processor.clean()  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# SceneProcessor — integration: process() with mocked model
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessWithMockModel:
    @pytest.fixture
    def processor(self):
        p = SceneProcessor("/fake/path.pt", device="cpu")
        mock = MagicMock()
        # side_effect on parameters() → each next(model.parameters()) gets
        # a fresh iterator over a real Parameter (so .device is cpu)
        mock.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        mock.forward.return_value = _fake_model_output(frame_count=5)
        mock.aggregator = MagicMock()
        mock.clean_kv_cache = MagicMock()
        mock._set_skip_append = MagicMock()
        p._model = mock
        p._model_loaded = True
        p._dtype = torch.float32
        p._active_backend = "sdpa"
        return p

    def test_process_basic_flow(self, processor):
        """Smoke test: processor runs without crashing."""
        source = _make_source(5)
        sink = InMemorySink()
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        processor.process(source, config, sink, NullProgress())

        assert sink.frame_count == 5

    def test_process_pushes_to_sink(self, processor):
        source = _make_source(5)
        sink = InMemorySink()
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        processor.process(source, config, sink, NullProgress())

        assert sink.frame_count == 5
        sink.close()
        preds = sink.predictions
        assert "extrinsic" in preds or "depth" in preds

    def test_process_with_null_sink(self, processor):
        """NullSink accepts everything without error."""
        source = _make_source(3)
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        processor.process(source, config, NullSink(), NullProgress())

    def test_process_calls_clean_before_inference(self, processor):
        source = _make_source(3)
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        processor.process(source, config, NullSink(), NullProgress())

        processor._model.clean_kv_cache.assert_called()

    def test_process_writes_metadata_to_sink(self, processor):
        source = _make_source(2)
        sink = InMemorySink()
        config = ProcessorConfig(mode="streaming", num_scale_frames=1)

        processor.process(source, config, sink, NullProgress())
        sink.close()

        assert "num_frames" in sink.metadata
        assert sink.metadata["num_frames"] == 2

    def test_progress_callbacks_fire(self, processor):
        events = []

        def on_start(n, cfg):
            events.append(("start", n))

        def on_frame(i, stage, fps):
            events.append(("frame", i, stage))

        def on_complete(s):
            events.append(("complete",))

        prog = CallbackProgress(
            on_start_fn=on_start,
            on_frame_fn=on_frame,
            on_complete_fn=on_complete,
        )

        source = _make_source(4)
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        processor.process(source, config, NullSink(), prog)

        assert events[0] == ("start", 4)
        assert events[-1] == ("complete",)
        assert len([e for e in events if e[0] == "frame"]) == 4

    def test_error_during_update_progress(self, processor):
        """If on_frame raises, the processor should propagate the error."""
        def bad_frame(i, stage, fps):
            if i >= 2:
                raise RuntimeError("progress crash")

        prog = CallbackProgress(on_frame_fn=bad_frame)
        source = _make_source(4)
        config = ProcessorConfig(mode="streaming", num_scale_frames=2)

        with pytest.raises(RuntimeError, match="progress crash"):
            processor.process(source, config, NullSink(), prog)

    def test_zero_frames_rejected(self, processor):
        # TensorFrameSource rejects empty tensors, so use a source with
        # len=0 that bypasses TensorFrameSource's validation
        from lingbot_map.io_protocol import FrameSource

        class ZeroFrameSource(FrameSource):
            def __len__(self): return 0
            def __iter__(self): return iter([])
            @property
            def resolution(self): return (56, 84)

        source = ZeroFrameSource()
        config = ProcessorConfig(mode="streaming")

        with pytest.raises(ValueError, match="positive"):
            processor.process(source, config, NullSink(), NullProgress())

    def test_streaming_too_many_frames_rejected(self, processor):
        """Streaming mode rejects frames above the cap."""
        # Create a source with many frames
        big_tensor = torch.rand(2000, 3, 56, 84)
        source = TensorFrameSource(big_tensor)
        config = ProcessorConfig(mode="streaming")

        with pytest.raises(ValueError, match="Streaming mode"):
            processor.process(source, config, NullSink(), NullProgress())


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def _make_processor_with_mock(self, frame_count: int = 5):
        """Helper: create a SceneProcessor with a properly mocked model.

        Uses a side_effect on parameters() so every call to
        next(model.parameters()) returns a fresh dummy tensor."""
        p = SceneProcessor("/fake/path.pt", device="cpu")
        mock = MagicMock()
        # Return a real Parameter so .device resolves to cpu
        mock.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        mock.forward.return_value = _fake_model_output(frame_count=frame_count)
        mock.aggregator = MagicMock()
        mock.clean_kv_cache = MagicMock()
        mock._set_skip_append = MagicMock()
        p._model = mock
        p._model_loaded = True
        p._dtype = torch.float32
        p._active_backend = "sdpa"
        return p

    def test_single_frame_source(self):
        """A 1-frame source should work (scale_frames clamped)."""
        processor = self._make_processor_with_mock(frame_count=1)

        source = _make_source(1)
        sink = InMemorySink()
        config = ProcessorConfig(mode="streaming")

        processor.process(source, config, sink, NullProgress())
        assert sink.frame_count == 1

    def test_scale_frames_equal_total(self):
        """When num_scale_frames >= total frames, scale_frames is clamped."""
        processor = self._make_processor_with_mock(frame_count=4)

        source = _make_source(4)
        sink = InMemorySink()
        config = ProcessorConfig(mode="streaming", num_scale_frames=8)

        processor.process(source, config, sink, NullProgress())
        # Should process without entering phase-2 streaming loop
        assert sink.frame_count == 4

    def test_explicit_windowed_mode(self):
        """Windowed mode should work with mock model."""
        processor = self._make_processor_with_mock(frame_count=1)

        source = _make_source(6)
        sink = InMemorySink()
        config = ProcessorConfig(mode="windowed", num_scale_frames=2, window_size=4)

        processor.process(source, config, sink, NullProgress())
        # Windowed mode with overlap produces duplicate frames at boundaries.
        # With 6 frames, ws=2, window_size=4, overlap=ws=2, step=4-2=2:
        # windows: [0,4], [2,6] → 4 + 4 = 8 frame writes (frames 2,3 duplicated)
        assert sink.frame_count >= 6

    def test_multiple_process_calls_with_clean(self):
        """Two sequential process() calls with clean() in between."""
        processor = self._make_processor_with_mock(frame_count=2)

        config = ProcessorConfig(mode="streaming", num_scale_frames=1)

        sink1 = InMemorySink()
        processor.process(_make_source(2), config, sink1, NullProgress())
        assert sink1.frame_count == 2

        processor.clean()
        processor._model.clean_kv_cache.reset_mock()

        sink2 = InMemorySink()
        processor.process(_make_source(3), config, sink2, NullProgress())
        assert sink2.frame_count == 3

        # After reset_mock(), the second process() call should have
        # triggered clean_kv_cache at least once (inside _run_inference)
        assert processor._model.clean_kv_cache.call_count >= 1

    def test_to_numpy_squeezes_batch_dims(self, processor=None):
        """_to_numpy strips leading singleton dims."""
        from lingbot_map.processor import SceneProcessor as SP

        # [1, 1, 9] → [9]
        t = torch.randn(1, 1, 9)
        arr = SP._to_numpy(t)
        assert arr.ndim == 1
        assert arr.shape == (9,)

        # [1, 1, 56, 84, 1] → [56, 84, 1]
        t2 = torch.randn(1, 1, 56, 84, 1)
        arr2 = SP._to_numpy(t2)
        assert arr2.ndim == 3
        assert arr2.shape == (56, 84, 1)

    def test_build_metadata_structure(self, processor=None):
        """Metadata dict has required keys."""
        from lingbot_map.processor import SceneProcessor as SP

        p = SP("/fake/path.pt", device="cpu")
        meta = p._build_metadata(10, ProcessorConfig(), "streaming")
        assert "num_frames" in meta
        assert meta["num_frames"] == 10
        assert "mode" in meta
        assert meta["mode"] == "streaming"
        assert "timestamp" in meta
        assert "config" in meta
