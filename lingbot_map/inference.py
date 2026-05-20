"""Shared inference utilities used by both the CLI demo and batch processing.

Extracted from ``demo.py`` so that entry-point scripts (CLI, batch, API) can
import model loading, post-processing, and visualisation helpers from the core
library rather than from each other.
"""

import logging
from typing import Optional

import numpy as np
import torch

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constant (kept here because postprocess depends on it)
# ---------------------------------------------------------------------------

_BATCHED_NDIMS = {
    "pose_enc": 3, "depth": 5, "depth_conf": 4,
    "world_points": 5, "world_points_conf": 4,
    "extrinsic": 4, "intrinsic": 4,
    "chunk_scales": 2, "chunk_transforms": 4, "images": 5,
}


def _squeeze_single_batch(key: str, value: torch.Tensor | np.ndarray):
    """Remove leading singleton batch dim for known keys."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    *,
    mode: str = "streaming",
    image_size: int = 518,
    patch_size: int = 14,
    enable_3d_rope: bool = True,
    max_frame_num: int = 1024,
    kv_cache_sliding_window: int = 64,
    num_scale_frames: int = 8,
    use_sdpa: bool = False,
    camera_num_iterations: int = 4,
):
    """Build and return a GCTStream model, load checkpoint, move to *device*.

    Parameters match the CLI flags but are passed explicitly (no argparse
    dependency) so the function can be called from any context.
    """
    if mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    _log.info("Building model...")
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        enable_3d_rope=enable_3d_rope,
        max_frame_num=max_frame_num,
        kv_cache_sliding_window=kv_cache_sliding_window,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=camera_num_iterations,
    )

    if model_path:
        _log.info("Loading checkpoint: %s", model_path)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _log.info("  Missing keys: %d", len(missing))
        if unexpected:
            _log.info("  Unexpected keys: %d", len(unexpected))
        del ckpt
        _log.info("  Checkpoint loaded.")

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess(predictions: dict, images: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Convert pose encoding to extrinsics (c2w) and move tensors to CPU.

    Returns ``(predictions_dict, images_cpu)``.
    """
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:],
    )

    # Convert 3×4 extrinsics to 4×4, invert (w2c → c2w), strip back to 3×4
    extrinsic_4x4 = torch.zeros(
        (*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype,
    )
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    _log.info("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True),
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, images_cpu


def prepare_for_visualization(
    predictions: dict, images: Optional[torch.Tensor | np.ndarray] = None,
) -> dict:
    """Convert predictions to unbatched NumPy dict for the viewer."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images.detach().cpu()).numpy()
    elif isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)

    if images is not None:
        vis_predictions["images"] = images
    return vis_predictions
