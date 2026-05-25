"""Alignment utilities for deferred window alignment.

During :meth:`~lingbot_map.models.gct_stream_window.GCTStream.inference_windowed`,
the model processes overlapping windows and computes pairwise similarity
transforms ``(s, R, t)`` via :meth:`~GCTStream._align_and_stitch_windows`.
These transforms are **already cumulative** — each maps its window directly
into window 0's coordinate frame — because ``_pairwise_alignment`` compares
against the *previously warped* window, which is already in the anchor frame.

With deferred alignment, raw (unaligned) per-window frames are saved to disk
alongside the transforms in ``alignment.npz``.  At load time this module
applies the transforms to produce the aligned predictions.

Math (pose_enc → c2w extrinsic)
--------------------------------
The warp operates on *pose_enc* (w2c translation ``T`` + quaternion ``R_quat``)::

    T'      = s · R · T + t
    R_quat' = R · R_quat

The stored NPZ uses *c2w extrinsic* where ``c2w_rot = R_quat^T`` and
``c2w_ctr = −R_quat^T · T``.  Substituting yields::

    new_c2w_rot  =  c2w_rot · R^T
    new_center   =  s · center − c2w_rot · R^T · t
    new_depth    =  s · depth

These are implemented in :func:`apply_alignment_to_frame`.
"""

from __future__ import annotations

import numpy as np


def compute_cumulative_transforms(
    chunk_scales: np.ndarray,
    chunk_transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-window transforms (already cumulative, no composition needed).

    The transforms stored in ``chunk_scales`` / ``chunk_transforms`` are
    already cumulative — each maps its window directly into window 0's
    coordinate frame (because ``_pairwise_alignment`` compares against the
    *previously warped* window, which is already in window 0's frame).

    Args:
        chunk_scales:   [W]          per-window scales.
        chunk_transforms: [W, 4, 4]  per-window (R | t) as 4×4 matrices.

    Returns:
        (scales [W], rotations [W,3,3], translations [W,3]).
    """
    W = len(chunk_scales)
    scales = np.asarray(chunk_scales, dtype=np.float32)
    rotations = np.asarray(chunk_transforms[:, :3, :3], dtype=np.float32)
    translations = np.asarray(chunk_transforms[:, :3, 3], dtype=np.float32)
    return scales, rotations, translations


def apply_alignment_to_frame(
    extrinsic: np.ndarray,
    depth: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
    depth_conf: np.ndarray | None = None,
    intrinsic: np.ndarray | None = None,
) -> dict:
    """Apply a cumulative alignment transform to a single frame.

    The alignment transform ``(R, t, s)`` is the same as used by
    :meth:`GCTStream._warp_predictions`, which operates on *pose_enc*
    (w2c translation + quaternion).  The stored NPZ format uses *c2w
    extrinsic* where::

        c2w_rot   =  R_quat^T
        c2w_ctr   =  −R_quat^T · T          (T = w2c translation)

    Warping pose_enc gives ``T' = s·R·T + t`` and ``R_quat' = R·R_quat``.
    Substituting into the c2w formulas yields::

        new_c2w_rot  =  raw_c2w_rot  ·  R^T
        new_center   =  s · raw_center  −  raw_c2w_rot · R^T · t
        new_depth    =  s · raw_depth

    Args:
        extrinsic:  [3,4]  camera-to-world matrix (R_c2w | center).
        depth:      [H,W] or [H,W,1]  depth map.
        R:          [3,3]  cumulative rotation (maps raw → aligned for
                           pose_enc quaternion matrices).
        t:          [3]    cumulative translation.
        s:          float  cumulative scale.
        depth_conf: [H,W]  optional confidence map (passed through unchanged).
        intrinsic:  [3,3]  optional intrinsics (passed through unchanged).

    Returns:
        dict with keys ``extrinsic``, ``depth``, and optionally ``depth_conf``,
        ``intrinsic``.
    """
    out: dict = {}

    R_c2w = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    center = np.asarray(extrinsic[:3, 3], dtype=np.float32)

    # c2w rotation is the *transpose* of the pose_enc quaternion matrix,
    # so left-multiplication in _warp_predictions (R @ local_rot) becomes
    # right-multiplication by R^T for c2w.
    new_R_c2w = R_c2w @ R.T
    # c2w centre = -R_quat^T @ T (where T = w2c translation).
    # After warp: T' = s·R·T + t, R_quat' = R·R_quat.
    # → centre' = -(R·R_quat)^T @ (s·R·T + t)
    #            = -R_quat^T·R^T @ (s·R·T + t)
    #            = s·(-R_quat^T @ T)  −  R_quat^T·R^T @ t
    #            = s·centre  −  c2w_rot @ R^T @ t
    new_center = s * center - new_R_c2w @ t

    new_ext = np.zeros((3, 4), dtype=np.float32)
    new_ext[:3, :3] = new_R_c2w
    new_ext[:3, 3] = new_center
    out["extrinsic"] = new_ext

    out["depth"] = (depth.astype(np.float32) * s).astype(np.float16).astype(np.float32)

    if depth_conf is not None:
        out["depth_conf"] = depth_conf
    if intrinsic is not None:
        out["intrinsic"] = intrinsic

    return out


def apply_alignment_to_predictions(
    predictions: dict,
    chunk_scales: np.ndarray,
    chunk_transforms: np.ndarray,
    frame_to_window: np.ndarray,
) -> dict:
    """Apply deferred alignment to an already-loaded flat predictions dict.

    Modifies *predictions* in-place for memory efficiency.

    Args:
        predictions: dict with ``extrinsic`` [S,3,4], ``depth`` [S,H,W,1],
                     and optionally ``depth_conf`` [S,H,W].
        chunk_scales:   [W]     pairwise scales.
        chunk_transforms: [W,4,4] pairwise transforms.
        frame_to_window: [S]    int array mapping global frame → window index.

    Returns:
        *predictions* (mutated in-place).
    """
    cum_scales, cum_R, cum_t = compute_cumulative_transforms(
        chunk_scales, chunk_transforms,
    )

    S = len(frame_to_window)
    for i in range(S):
        w = int(frame_to_window[i])
        depth = predictions["depth"][i]
        conf = None
        if "depth_conf" in predictions and predictions["depth_conf"] is not None:
            conf = predictions["depth_conf"][i]

        result = apply_alignment_to_frame(
            predictions["extrinsic"][i],
            depth,
            cum_R[w],
            cum_t[w],
            float(cum_scales[w]),
            depth_conf=conf,
        )
        predictions["extrinsic"][i] = result["extrinsic"]
        predictions["depth"][i] = result["depth"]
        if conf is not None and "depth_conf" in result:
            predictions["depth_conf"][i] = result["depth_conf"]

    return predictions


def build_frame_to_window_mapping(
    window_boundaries: list[tuple[int, int]],
    overlap: int,
    num_frames: int,
) -> np.ndarray:
    """Build a mapping from global frame index to the window that owns it.

    Follows the same overlap-resolution rule as ``_stitch_windows``:
    non-final windows contribute frames ``[0, window_len - overlap)`` and
    the final window contributes all its frames.

    Args:
        window_boundaries: list of (start, end) global indices per window.
        overlap: number of overlapping frames between consecutive windows.
        num_frames: total number of global frames (for validation).

    Returns:
        [num_frames] int32 array mapping frame → window index.
    """
    mapping = np.full(num_frames, -1, dtype=np.int32)
    n_win = len(window_boundaries)

    for wi, (start, end) in enumerate(window_boundaries):
        is_last = wi == n_win - 1
        keep_start = start
        keep_end = end if is_last else end - overlap
        if keep_end > keep_start:
            mapping[keep_start:keep_end] = wi

    # Sanity check — every frame should be mapped
    if (mapping < 0).any():
        missing = int((mapping < 0).sum())
        raise ValueError(
            f"{missing} frames are not covered by any window; "
            f"window_boundaries={window_boundaries}, overlap={overlap}"
        )

    return mapping
