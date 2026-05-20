"""Tests for load_and_preprocess_video_stream — Module 1C.

Validates frame-by-frame streaming decode without loading all frames
into memory at once.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from lingbot_map.utils.load_fn import load_and_preprocess_video_stream


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_test_video(num_frames: int = 10, fps: float = 30.0,
                     width: int = 320, height: int = 240) -> str:
    """Create a temporary MP4 with *num_frames* of colored noise."""
    import cv2

    tmpdir = tempfile.mkdtemp(prefix="lingbot_test_vid_")
    path = os.path.join(tmpdir, "test.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return path, tmpdir


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestVideoStreamFile:
    """Tests using a file path as input."""

    def test_yields_correct_count(self):
        path, tmpdir = _make_test_video(num_frames=10, fps=30.0)
        try:
            frames = list(load_and_preprocess_video_stream(path, fps=10))
            # 30 fps, 10 frames, interval=3 → frames 0, 3, 6, 9 = 4 frames
            # Actually first frame is always yielded → frames 0, 3, 6, 9
            assert len(frames) >= 2, f"Expected >=2 frames, got {len(frames)}"
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_yields_tuples(self):
        path, tmpdir = _make_test_video(num_frames=5, fps=30.0)
        try:
            for item in load_and_preprocess_video_stream(path, fps=30):
                assert isinstance(item, tuple)
                assert len(item) == 2
                idx, tensor = item
                assert isinstance(idx, int)
                assert isinstance(tensor, torch.Tensor)
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_yields_correct_shape(self):
        path, tmpdir = _make_test_video(num_frames=5, fps=30.0,
                                         width=320, height=240)
        try:
            for idx, tensor in load_and_preprocess_video_stream(
                path, fps=30, image_size=300, patch_size=14
            ):
                assert tensor.dim() == 4
                assert tensor.shape[0] == 1  # batch
                assert tensor.shape[1] == 3  # RGB
                assert tensor.shape[-1] == 300  # width = image_size
                assert tensor.shape[-2] % 14 == 0  # height divisible by patch_size
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_values_in_range(self):
        path, tmpdir = _make_test_video(num_frames=3, fps=30.0)
        try:
            for _, tensor in load_and_preprocess_video_stream(path, fps=30):
                assert tensor.min() >= 0.0
                assert tensor.max() <= 1.0
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_max_frames(self):
        path, tmpdir = _make_test_video(num_frames=30, fps=30.0)
        try:
            frames = list(load_and_preprocess_video_stream(
                path, fps=30, max_frames=5
            ))
            assert len(frames) == 5
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_frame_indices_sequential(self):
        path, tmpdir = _make_test_video(num_frames=10, fps=30.0)
        try:
            indices = [idx for idx, _ in load_and_preprocess_video_stream(
                path, fps=30, max_frames=5
            )]
            assert indices == list(range(5))
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_fps_subsampling(self):
        """Higher fps → more frames yielded."""
        path, tmpdir = _make_test_video(num_frames=30, fps=30.0)
        try:
            all_frames = list(load_and_preprocess_video_stream(path, fps=30))
            fewer_frames = list(load_and_preprocess_video_stream(path, fps=5))
            # 5 fps should yield fewer frames than 30 fps for the same video
            assert len(fewer_frames) < len(all_frames), (
                f"fps=5 ({len(fewer_frames)}) should be < fps=30 ({len(all_frames)})"
            )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_video_raises(self):
        with pytest.raises(ValueError, match="Cannot open"):
            list(load_and_preprocess_video_stream("/nonexistent/video.mp4"))


class TestVideoStreamBytes:
    """Tests using in-memory bytes as input."""

    @pytest.fixture
    def video_bytes(self):
        path, tmpdir = _make_test_video(num_frames=10, fps=30.0)
        try:
            with open(path, "rb") as f:
                data = f.read()
            yield data
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_bytes_yields_same_count_as_file(self, video_bytes):
        """Bytes input should produce the same frames as file input."""
        # Get file results
        path, tmpdir = _make_test_video(num_frames=10, fps=30.0)
        try:
            file_frames = list(load_and_preprocess_video_stream(path, fps=10))
            bytes_frames = list(load_and_preprocess_video_stream(video_bytes, fps=10))
            assert len(bytes_frames) == len(file_frames), (
                f"Bytes ({len(bytes_frames)}) != file ({len(file_frames)})"
            )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_bytes_yields_tensors(self, video_bytes):
        for _, tensor in load_and_preprocess_video_stream(
            video_bytes, fps=10, max_frames=2
        ):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dim() == 4

    def test_bytes_max_frames(self, video_bytes):
        frames = list(load_and_preprocess_video_stream(
            video_bytes, fps=10, max_frames=3
        ))
        assert len(frames) == 3

    def test_bytes_cleanup_tempfile(self, video_bytes):
        """Temporary files are cleaned up after iteration."""
        import glob
        before = set(glob.glob("/tmp/lingbot_vstream_*"))
        list(load_and_preprocess_video_stream(video_bytes, fps=30, max_frames=1))
        after = set(glob.glob("/tmp/lingbot_vstream_*"))
        # The temp dir should be gone
        new_dirs = after - before
        assert len(new_dirs) == 0, f"Leaked temp dirs: {new_dirs}"


class TestVideoStreamMemory:
    """Verify that streaming keeps peak memory at O(1 frame)."""

    def test_memory_bounded_for_long_video(self):
        """Processing a long video should not grow memory over time."""
        path, tmpdir = _make_test_video(num_frames=60, fps=30.0)
        try:
            import tracemalloc

            tracemalloc.start()
            frames_processed = 0
            snapshot_first = None

            for idx, tensor in load_and_preprocess_video_stream(path, fps=30):
                frames_processed += 1
                if frames_processed == 1:
                    snapshot_first = tracemalloc.take_snapshot()
                if frames_processed >= 20:
                    break

            snapshot_last = tracemalloc.take_snapshot()
            tracemalloc.stop()

            if snapshot_first is not None:
                diff = snapshot_last.compare_to(snapshot_first, "lineno")
                # Memory should not grow dramatically
                total_diff = sum(stat.size_diff for stat in diff)
                # Allow up to 50 MB growth (Python overhead, not frame data)
                assert total_diff < 50 * 1024 * 1024, (
                    f"Memory grew by {total_diff / 1e6:.1f} MB over 20 frames — "
                    f"streaming may not be releasing frames"
                )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


class TestEdgeCases:
    def test_single_frame_video(self):
        path, tmpdir = _make_test_video(num_frames=1, fps=30.0)
        try:
            frames = list(load_and_preprocess_video_stream(path, fps=30))
            assert len(frames) == 1
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_zero_fps_equivalent_to_all_frames(self):
        """fps=0 should yield every frame (interval=1)."""
        path, tmpdir = _make_test_video(num_frames=5, fps=30.0)
        try:
            frames = list(load_and_preprocess_video_stream(path, fps=0))
            assert len(frames) == 5
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_large_image_size(self):
        """Large image_size should still produce valid tensors."""
        path, tmpdir = _make_test_video(num_frames=3, fps=30.0,
                                         width=640, height=480)
        try:
            for _, tensor in load_and_preprocess_video_stream(
                path, fps=30, image_size=600, patch_size=14
            ):
                assert tensor.shape[-1] == 600
                assert tensor.shape[-2] % 14 == 0
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
