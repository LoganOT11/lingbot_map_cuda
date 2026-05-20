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
# Safety: frame-count validation
# ---------------------------------------------------------------------------

# Hard cap for streaming mode: the FlashInfer special-page pool is pre-allocated
# for max_frame_num + 100 frames.  Beyond that, append_frame() asserts.
_MAX_FRAMES_STREAMING = 1100

# Arbitrary safety limit for windowed mode (each window is independent, but
# we need a ceiling to prevent unbounded disk/RAM usage).
_MAX_FRAMES_WINDOWED = 50000


def validate_frame_count(
    num_frames: int,
    mode: str,
    *,
    max_frames_streaming: int = _MAX_FRAMES_STREAMING,
    max_frames_windowed: int = _MAX_FRAMES_WINDOWED,
) -> None:
    """Raise ``ValueError`` if *num_frames* would crash or degrade.

    Streaming mode has a hard cap because the FlashInfer special-token page
    pool is pre-allocated for ``max_frame_num + 100`` frames (default 1124).
    Windowed mode is unbounded per-window but we apply a safety ceiling.

    Args:
        num_frames: Number of frames in the input sequence.
        mode: ``"streaming"`` or ``"windowed"``.
        max_frames_streaming: Override for the streaming cap.
        max_frames_windowed: Override for the windowed ceiling.

    Raises:
        ValueError: If *num_frames* exceeds the limit for the chosen *mode*.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    if mode == "streaming" and num_frames > max_frames_streaming:
        raise ValueError(
            f"Streaming mode supports at most {max_frames_streaming} frames "
            f"(got {num_frames}).  Switch to mode='windowed' for longer "
            f"sequences, or increase --max_frame_num to grow the special-page "
            f"pool (at the cost of GPU memory)."
        )

    if num_frames > max_frames_windowed:
        raise ValueError(
            f"Maximum {max_frames_windowed} frames supported "
            f"(got {num_frames})."
        )


# ---------------------------------------------------------------------------
# Safety: GPU memory budget estimation
# ---------------------------------------------------------------------------

def estimate_gpu_memory(
    resolution: tuple[int, int],
    num_frames_in_window: int,
    backend: str = "flashinfer",
    dtype: torch.dtype | None = None,
    *,
    patch_size: int = 14,
    num_blocks: int = 24,
    num_heads: int = 16,
    head_dim: int = 64,
    num_special_tokens: int = 6,
    model_weight_gb: float = 2.8,
    activations_gb: float = 1.0,
) -> dict:
    """Estimate peak GPU memory for one processing window.

    This is a static approximation — it does **not** query the GPU.  Use it
    for pre-flight checks before launching inference.

    Args:
        resolution: ``(height, width)`` of preprocessed frames.
        num_frames_in_window: Frames held in the KV cache at once
            (typically ``num_scale_frames + kv_cache_sliding_window``).
        backend: ``"flashinfer"`` (paged) or ``"sdpa"`` (dict-based).
        dtype: ``torch.bfloat16`` or ``torch.float16``.  If ``None``,
            ``torch.bfloat16`` is assumed (2 bytes per element).
        patch_size: Patch size for ViT embedding (default 14).
        num_blocks: Transformer blocks (default 24 for ViT-L).
        num_heads: Attention heads (default 16).
        head_dim: Dimension per head (default 64).
        num_special_tokens: Special tokens per frame (camera + reg + scale).
        model_weight_gb: Approximate model weight size in GB (bf16 mixed).
        activations_gb: Conservative activation memory estimate.

    Returns:
        dict with keys ``model_gb``, ``kv_cache_gb``, ``special_pages_gb``,
        ``activations_gb``, ``total_gb`` — all rounded to one decimal.
    """
    if dtype is None:
        bytes_per_element = 2  # bfloat16
    elif dtype == torch.float32:
        bytes_per_element = 4
    else:
        bytes_per_element = 2  # float16 / bfloat16

    h, w = resolution
    patches_h = h // patch_size
    patches_w = w // patch_size
    patches_per_frame = patches_h * patches_w
    tokens_per_frame = patches_per_frame + num_special_tokens

    if backend not in ("flashinfer", "sdpa"):
        raise ValueError(
            f"Unknown backend '{backend}'.  Choose 'flashinfer' or 'sdpa'."
        )

    if backend == "flashinfer":
        # Page size = patches_per_frame (exact fit for FA2)
        page_size = patches_per_frame
        elements_per_page = page_size * num_heads * head_dim
        bytes_per_page_block = elements_per_page * bytes_per_element * 2  # K + V

        # Patch pages: scale (8) + window (variable) + headroom (16)
        patch_pages = 8 + num_frames_in_window + 16
        kv_cache_gb = (patch_pages * bytes_per_page_block * num_blocks) / 1e9

        # Special pages: pre-allocated for ~1124 frames worth of specials
        # ceil(max_frames * specials_per_frame / page_size) + 16 headroom
        max_frames_special = 1124
        special_pages = (
            (max_frames_special * num_special_tokens + page_size - 1) // page_size
            + 16
        )
        special_gb = (special_pages * bytes_per_page_block * num_blocks) / 1e9
    else:
        # SDPA: per-frame K+V tensors, no paging overhead
        elements_per_frame = (
            2  # K + V
            * num_heads
            * tokens_per_frame
            * head_dim
        )
        bytes_per_frame = elements_per_frame * bytes_per_element
        kv_cache_gb = (num_frames_in_window * bytes_per_frame * num_blocks) / 1e9
        special_gb = 0.0

    total_gb = model_weight_gb + kv_cache_gb + special_gb + activations_gb

    return {
        "model_gb": round(model_weight_gb, 1),
        "kv_cache_gb": round(kv_cache_gb, 1),
        "special_pages_gb": round(special_gb, 1),
        "activations_gb": round(activations_gb, 1),
        "total_gb": round(total_gb, 1),
    }


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
