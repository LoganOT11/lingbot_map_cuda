"""NPZ data loading for RGBD scans.

Supports three input formats:
  1. Single NPZ file:  ``np.load('scene.npz')``
  2. Per-frame NPZ directory (produced by ``batch_demo.py --save_predictions``):
     ``scene_dir/frame_000000.npz, frame_000001.npz, ..., meta.npz``
     Per-frame files are loaded in parallel via ThreadPoolExecutor.
  3. Per-window NPZ directory (deferred alignment format):
     ``scene_dir/window_000/frame_000000.npz, ..., alignment.npz, windows.json``
     Alignment is applied at load time.
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np


# Keys SceneBuilder actually consumes.  ``world_points`` / ``world_points_conf``
# / ``pose_enc`` are deliberately omitted: the renderer re-derives points from
# (depth, intrinsic, extrinsic), and the saved world_points alone can dominate
# RAM (e.g. 25k frames × 1.8 MB = ~46 GB).
_NEEDED_KEYS = frozenset({
    "images", "depth", "depth_conf", "confidence",
    "intrinsic", "extrinsic",
    "is_keyframe", "frame_type",
})


def _load_single_frame(path: str) -> Dict[str, np.ndarray]:
    """Load one per-frame npz, keeping only keys the renderer needs."""
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files if key in _NEEDED_KEYS}


def _load_perframe_dir(dir_path: str, num_workers: int = 16) -> Dict[str, np.ndarray]:
    """Load a per-frame NPZ directory with parallel I/O.

    Detects the per-window format (``window_000/`` subdirectory) and applies
    deferred alignment automatically.

    Returns a single dict where each key has arrays stacked along a new
    leading dimension (S, ...).  Only keys in ``_NEEDED_KEYS`` are kept to
    bound memory on long sequences.
    """
    # Detect per-window format
    if os.path.isdir(os.path.join(dir_path, 'window_000')):
        return _load_per_window_format(dir_path, num_workers=num_workers)

    # Legacy flat format
    frame_files = sorted(glob.glob(os.path.join(dir_path, 'frame_*.npz')))
    if not frame_files:
        raise ValueError(f"No frame_*.npz files found in {dir_path}")

    # Parallel load
    workers = min(num_workers, len(frame_files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frame_dicts = list(pool.map(_load_single_frame, frame_files))

    # Stack all keys along dim 0
    all_keys = list(frame_dicts[0].keys())
    data = {}
    for key in all_keys:
        data[key] = np.stack([fd[key] for fd in frame_dicts], axis=0)

    # Merge metadata if present (filtered the same way)
    meta_path = os.path.join(dir_path, 'meta.npz')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        for key in meta.files:
            if key in _NEEDED_KEYS:
                data[key] = meta[key]

    return data


def _load_per_window_format(dir_path: str, num_workers: int = 16) -> Dict[str, np.ndarray]:
    """Load the per-window NPZ format and apply deferred alignment.

    Returns the same flat dict structure as :func:`_load_perframe_dir`.
    """
    from lingbot_map.utils.alignment import apply_alignment_to_frame

    # Load metadata
    with open(os.path.join(dir_path, 'windows.json'), 'r') as f:
        win_meta = json.load(f)
    windows = win_meta['windows']  # [[start, end], ...]
    overlap = win_meta.get('overlap', 0)
    num_windows = win_meta['num_windows']

    # Load alignment (transforms are already cumulative).
    align_path = os.path.join(dir_path, 'alignment.npz')
    if os.path.exists(align_path):
        align = np.load(align_path, allow_pickle=False)
        cum_scales = np.asarray(align['chunk_scales'], dtype=np.float32)
        cum_R = np.asarray(align['chunk_transforms'][:, :3, :3], dtype=np.float32)
        cum_t = np.asarray(align['chunk_transforms'][:, :3, 3], dtype=np.float32)
    else:
        cum_scales = np.ones(num_windows, dtype=np.float32)
        cum_R = np.tile(np.eye(3, dtype=np.float32), (num_windows, 1, 1))
        cum_t = np.zeros((num_windows, 3), dtype=np.float32)

    # Load all per-window frames
    def _load_window(wi):
        wdir = os.path.join(dir_path, f"window_{wi:03d}")
        frame_files = sorted(glob.glob(os.path.join(wdir, 'frame_*.npz')))
        frames = []
        for ff in frame_files:
            data = np.load(ff, allow_pickle=False)
            frames.append({key: data[key] for key in data.files if key in _NEEDED_KEYS})
        return frames

    workers = min(num_workers, num_windows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        all_window_frames = list(pool.map(_load_window, range(num_windows)))

    # Apply alignment per window, build global flat list
    aligned_frames = []
    for wi in range(num_windows):
        start, end = windows[wi]
        window_frames = all_window_frames[wi]
        s = float(cum_scales[wi])
        R = cum_R[wi]
        t_vec = cum_t[wi]
        is_last = wi == num_windows - 1

        for local_idx, fd in enumerate(window_frames):
            global_idx = start + local_idx
            if not is_last and global_idx >= end - overlap:
                continue

            result = apply_alignment_to_frame(
                fd.get('extrinsic', np.eye(3, dtype=np.float32)[:, :4]),
                fd.get('depth', np.zeros((1, 1), dtype=np.float16)),
                R, t_vec, s,
                depth_conf=fd.get('depth_conf'),
                intrinsic=fd.get('intrinsic'),
            )
            aligned_frames.append(result)

    # Stack into flat dict
    all_keys = list(aligned_frames[0].keys())
    data = {}
    for key in all_keys:
        arr = np.stack([fd[key] for fd in aligned_frames], axis=0)
        if key in ('depth', 'depth_conf') and arr.dtype in (np.float16, np.uint8):
            arr = arr.astype(np.float32)
        data[key] = arr

    # Load meta.npz if present
    meta_path = os.path.join(dir_path, 'meta.npz')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        for key in meta.files:
            if key in _NEEDED_KEYS:
                data[key] = meta[key]

    return data


def _parse_raw_data(data: Dict[str, np.ndarray]) -> Dict:
    """Parse raw NPZ arrays into the standardized format.

    Handles both single-file and per-frame-dir data (same key structure).

    Returns dict: images (S,H,W,3) uint8, depth (S,H,W) float32,
                  c2w (S,4,4) float32, K (S,3,3) float32,
                  confidence (S,H,W) float32 or None.
    """
    images = data['images']
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.ascontiguousarray(images.transpose(0, 2, 3, 1))
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype(np.uint8) if images.max() <= 1.0 else images.astype(np.uint8)

    depth = data['depth'].astype(np.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]

    K_raw = data['intrinsic'].astype(np.float32)
    if K_raw.ndim == 2:
        K_raw = np.tile(K_raw[None], (len(images), 1, 1))

    if 'extrinsic' not in data:
        raise ValueError("NPZ must contain 'extrinsic' (W2C poses).")
    ext = data['extrinsic'].astype(np.float32)
    nf = ext.shape[0]
    w2c = np.zeros((nf, 4, 4), dtype=np.float32)
    w2c[:, :3, :] = ext[:, :3, :]
    w2c[:, 3, 3] = 1.0
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3:4]
    Rt = R.transpose(0, 2, 1)
    c2w = np.zeros((nf, 4, 4), dtype=np.float32)
    c2w[:, :3, :3] = Rt
    c2w[:, :3, 3:4] = -Rt @ t
    c2w[:, 3, 3] = 1.0

    confidence = None
    for conf_key in ('depth_conf', 'confidence'):
        if conf_key in data:
            confidence = data[conf_key].astype(np.float32)
            if confidence.ndim == 4:
                confidence = confidence[..., 0]
            break

    # Optional keyframe mask from meta.npz (produced by batch_demo.py).
    # ``is_keyframe`` is the preferred key (bool per frame); ``frame_type``
    # (uint8, 0=scale, 1=keyframe, 2=non-keyframe) is accepted as a fallback.
    is_keyframe = None
    if 'is_keyframe' in data:
        is_keyframe = np.asarray(data['is_keyframe']).astype(bool)
    elif 'frame_type' in data:
        is_keyframe = np.asarray(data['frame_type']) != 2
    if is_keyframe is not None:
        is_keyframe = np.squeeze(is_keyframe)
        if is_keyframe.ndim == 0:
            is_keyframe = is_keyframe.reshape(1)

    return {
        'images': images, 'depth': depth, 'c2w': c2w, 'K': K_raw,
        'confidence': confidence, 'is_keyframe': is_keyframe,
    }


def load_npz_data(npz_path: str, num_workers: int = 16) -> Dict:
    """Load NPZ produced by batch_demo.py.

    Accepts either:
      - A single ``.npz`` file path
      - A directory of per-frame ``frame_*.npz`` files (parallel loading)

    Returns dict: images (S,H,W,3) uint8, depth (S,H,W) float32,
                  c2w (S,4,4) float32, K (S,3,3) float32,
                  confidence (S,H,W) float32 or None.
    """
    if os.path.isdir(npz_path):
        raw = _load_perframe_dir(npz_path, num_workers=num_workers)
    else:
        # np.load on .npz is lazy — iterating .files lists keys without loading;
        # only materialize the ones we need.  Avoids OOM on huge sequences
        # where world_points* would otherwise dominate memory.
        loaded = np.load(npz_path, allow_pickle=False)
        raw = {key: loaded[key] for key in loaded.files if key in _NEEDED_KEYS}

    return _parse_raw_data(raw)
