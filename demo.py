"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python demo.py --model_path lingbot-map-long.pt --image_folder example/courthouse

    # Windowed inference (for long sequences)
    python demo.py --model_path lingbot-map-long.pt --image_folder example/courthouse \
        --mode windowed --window_size 16 --num_scale_frames 4

    # Headless inference with prediction export
    python demo.py --model_path lingbot-map-long.pt --image_folder example/courthouse \
        --headless --save_predictions outputs/

    # From video
    python demo.py --model_path lingbot-map-long.pt --video_path video.mp4 --fps 10
"""

import argparse
import glob
import logging
import os
import sys
import tempfile
import time

# ── Logging ─────────────────────────────────────────────────────────────────
# stderr is unbuffered → messages appear immediately even when the viser viewer
# runs its infinite loop.  Use `conda activate && python -u demo.py ...` to see
# logs in real time (conda run buffers stderr).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
_log = logging.getLogger("demo")

# ── Named constants ─────────────────────────────────────────────────────────
_KEYFRAME_AUTO_THRESHOLD = 320   # max frames before keyframe auto-selection (RoPE training range)
_WARM_STREAM_N_DEFAULT   = 10    # streaming frames in torch.compile warmup
_COMPILED_WARMUP_PASSES  = 3     # compile dress-rehearsal passes
_BF16_MIN_CAPABILITY     = 8     # Turing+ for bfloat16
_GPU_MEM_WARN_GB         = 10.0  # warn if free GPU memory is below this


# ═════════════════════════════════════════════════════════════════════════════
# Pre-parse --compile before importing torch, so we can set the CUDA allocator
# configuration before the first CUDA init.
# ═════════════════════════════════════════════════════════════════════════════
def _parse_args_pre() -> argparse.Namespace:
    """Minimal parse for flags that must be known before torch import.

    ``expandable_segments:True`` is a CUDA allocator optimisation that
    reduces the reserved-vs-allocated gap, but it conflicts with:

    * FlashInfer  (runtime: ``CUDA driver error: device not ready``)
    * torch.compile cudagraph_trees  (``RuntimeError: Expected
      curr_block->next == nullptr``, PyTorch ≤2.8)

    So we only enable it when **both** ``--use_sdpa`` (SDPA KV cache
    instead of FlashInfer) is requested **and** ``--compile`` is **not**.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--use_sdpa", action="store_true", default=False)
    args, _ = p.parse_known_args()
    return args


_pre_args = _parse_args_pre()
if _pre_args.use_sdpa and not _pre_args.compile:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Now safe to import torch / CUDA
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Build and return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming",
                        choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame; windowed: overlapping windows")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=None,
                        help="Every N-th frame after scale frames is kept as a keyframe")
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head refinement steps (1=faster, 4=accurate)")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (required on ≤12 GB GPUs)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (~5 FPS faster, adds warmup)")
    parser.add_argument("--offload_to_cpu", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Offload per-frame predictions to CPU (on by default)")

    # Windowed options
    parser.add_argument("--window_size", type=int, default=64,
                        help="Keyframes per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16,
                        help="Overlap between windows in actual frames")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in keyframes")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true",
                        help="Apply sky segmentation to filter sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Cache directory for sky masks")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export preprocessed images to this folder")

    # Headless / export options
    parser.add_argument("--headless", action="store_true",
                        help="Skip the 3D viewer (print summary and exit)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Save predictions as NPZ files to this directory")

    args = parser.parse_args()
    if not args.image_folder and not args.video_path:
        parser.error("Provide --image_folder or --video_path")
    return args


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png,.JPG",
                first_k=None, stride=1, image_size=518, patch_size=14,
                rotate_clockwise_90=False):
    """Load images from folder or video and preprocess into a tensor."""
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        _log.info("Extracted %d frames (total %d, interval=%d)",
                  len(paths), total_frames, interval)
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    if rotate_clockwise_90:
        rotated_dir = tempfile.mkdtemp(prefix="lingbot_rot_cw90_")
        rotated_paths = []
        for p in tqdm(paths, desc="Rotating images 90° CW"):
            out_path = os.path.join(rotated_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out_path)
            rotated_paths.append(out_path)
        paths = rotated_paths
        resolved_folder = rotated_dir
        _log.info("Rotated %d images → %s", len(paths), rotated_dir)

    _log.info("Loading %d images...", len(paths))
    images = load_and_preprocess_images(paths, mode="crop",
                                        image_size=image_size, patch_size=patch_size)
    h, w = images.shape[-2:]
    _log.info("Preprocessed to %dx%d (canonical crop mode)", w, h)
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Build the GCTStream model and load the checkpoint onto *device*."""
    if args.mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    _log.info("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        _log.info("Loading checkpoint: %s", args.model_path)
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _log.info("  Missing keys: %d", len(missing))
        if unexpected:
            _log.info("  Unexpected keys: %d", len(unexpected))
        del ckpt
        _log.info("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# torch.compile helpers  (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """torch.compile the hot, fixed-shape modules with mode='reduce-overhead'."""
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive scale → streaming forward ``passes`` times to capture CUDA graphs.

    NOTE: this uses ``model.forward()`` directly rather than the public
    ``model.inference_streaming()`` because the warmup must interleave
    ``torch.compiler.cudagraph_mark_step_begin()`` calls between every forward
    pass — something the public streaming API intentionally hides.  The manual
    loop also alternates keyframe / non-keyframe patterns so the ``skip_append``
    path is captured during warmup (without it, the first non-keyframe in the
    real run hits cold orchestration code and can confuse cudagraph_trees).
    """
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(warm_scale, num_frame_for_scale=scale_frames,
                          num_frame_per_block=scale_frames, causal_inference=True)
        for i in range(warm_stream_n):
            is_keyframe = (kf_int <= 1) or (i % kf_int == 0)
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(warm_stream[:, i:i + 1],
                              num_frame_for_scale=scale_frames,
                              num_frame_per_block=1, causal_inference=True)
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3, "depth": 5, "depth_conf": 4,
    "world_points": 5, "world_points_conf": 4,
    "extrinsic": 4, "intrinsic": 4,
    "chunk_scales": 2, "chunk_transforms": 4, "images": 5,
}


def _squeeze_single_batch(key, value):
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move tensors to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])

    extrinsic_4x4 = torch.zeros(
        (*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
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
                k, predictions[k].to("cpu", non_blocking=True))
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
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


# =============================================================================
# Pipeline stages  (called by main)
# =============================================================================

def _check_gpu_memory(device: torch.device) -> None:
    """Log GPU info and warn if free memory is low."""
    if device.type != "cuda":
        return
    free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
    _log.info("GPU: %s | %.1f GB free / %.1f GB total",
              torch.cuda.get_device_name(0), free_gb, total_gb)
    if free_gb < _GPU_MEM_WARN_GB:
        _log.warning(
            "Less than %.0f GB GPU memory free (%.1f GB). "
            "Consider: --use_sdpa --offload_to_cpu --num_scale_frames 4 "
            "--mode windowed --window_size 16",
            _GPU_MEM_WARN_GB, free_gb)


def _select_dtype() -> torch.dtype:
    """Pick the best inference dtype for the available GPU."""
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= _BF16_MIN_CAPABILITY:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
    _log.info("Inference dtype: %s", dtype)
    return dtype


def _prepare_model(model, dtype: torch.dtype) -> None:
    """Cast the aggregator trunk to *dtype* (heads stay in fp32)."""
    if dtype != torch.float32 and model.aggregator is not None:
        _log.info("Casting aggregator to %s (heads kept in fp32)", dtype)
        model.aggregator = model.aggregator.to(dtype=dtype)


def _export_preprocessed(export_dir: str, images: torch.Tensor) -> None:
    """Write preprocessed images to *export_dir* as PNG files."""
    if not export_dir:
        return
    os.makedirs(export_dir, exist_ok=True)
    _log.info("Exporting %d preprocessed images to %s ...", images.shape[0], export_dir)
    for i in range(images.shape[0]):
        img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(export_dir, f"{i:06d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    _log.info("Exported to %s", export_dir)


def _log_memory_after_load(num_frames: int, shape: tuple, mode: str) -> None:
    """Report GPU memory state after model + images are on device."""
    _log.info("Input: %d frames, shape %s", num_frames, shape)
    _log.info("Mode: %s", mode)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log.info("GPU mem after load: alloc=%.2f GB, reserved=%.2f GB",
                  torch.cuda.memory_allocated() / 1e9,
                  torch.cuda.memory_reserved() / 1e9)


def _auto_keyframe_interval(args, num_frames: int) -> None:
    """Auto-select keyframe_interval if the user did not set one."""
    if args.keyframe_interval is not None:
        return
    if args.mode == "streaming" and num_frames > _KEYFRAME_AUTO_THRESHOLD:
        args.keyframe_interval = (
            (num_frames + _KEYFRAME_AUTO_THRESHOLD - 1) // _KEYFRAME_AUTO_THRESHOLD)
        _log.info("Auto-selected --keyframe_interval=%d (num_frames=%d > %d)",
                  args.keyframe_interval, num_frames, _KEYFRAME_AUTO_THRESHOLD)
    else:
        args.keyframe_interval = 1

    if args.keyframe_interval <= 1:
        return
    if args.mode == "streaming":
        _log.info("Keyframe streaming: interval=%d (after %d scale frames)",
                  args.keyframe_interval, args.num_scale_frames)
    else:
        actual = (args.num_scale_frames
                  + max(0, args.window_size - args.num_scale_frames) * args.keyframe_interval)
        _log.info("Keyframe windowed: interval=%d, each window ≤%d actual frames "
                  "(window_size=%d keyframes, scale=%d)",
                  args.keyframe_interval, actual, args.window_size, args.num_scale_frames)


def _warmup_compile(model, images: torch.Tensor, args, dtype: torch.dtype,
                    num_frames: int) -> None:
    """Run eager warmup + torch.compile + compiled warmup if --compile is set."""
    if not args.compile:
        return
    if args.mode != "streaming":
        _log.info("--compile only applies to --mode streaming (got %r); skipping",
                  args.mode)
        return

    scale_w = min(args.num_scale_frames, num_frames)
    if scale_w >= num_frames:
        scale_w = max(1, num_frames - 1)
    stream_n = min(_WARM_STREAM_N_DEFAULT, max(1, num_frames - scale_w))
    h, w = int(images.shape[-2]), int(images.shape[-1])
    _log.info("Warmup eager (scale=%d + %d streaming, shape=%dx%d, kf_int=%d)...",
              scale_w, stream_n, h, w, args.keyframe_interval)

    t_w = time.time()
    _warm_streaming(model, images, scale_w, stream_n, dtype,
                    passes=1, keyframe_interval=args.keyframe_interval)
    _log.info("  eager warmup: %.1f s", time.time() - t_w)

    _log.info("Compiling hot modules...")
    compile_model(model)

    _log.info("Warmup compiled (%dx dress rehearsal)...", _COMPILED_WARMUP_PASSES)
    t_w = time.time()
    _warm_streaming(model, images, scale_w, stream_n, dtype,
                    passes=_COMPILED_WARMUP_PASSES, keyframe_interval=args.keyframe_interval)
    _log.info("  compiled warmup: %.1f s", time.time() - t_w)


def _run_inference(model, images: torch.Tensor, args, dtype: torch.dtype):
    """Dispatch to streaming or windowed inference and return predictions."""
    _log.info("Running %s inference (dtype=%s)...", args.mode, dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images, num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device)
        else:
            predictions = model.inference_windowed(
                images, window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device)

    num_frames = images.shape[0]
    elapsed = time.time() - t0
    _log.info("Inference done in %.1f s (%.1f FPS)",
              elapsed, num_frames / elapsed if elapsed > 0 else float("inf"))
    if torch.cuda.is_available():
        _log.info("GPU peak during inference: alloc=%.2f GB, reserved=%.2f GB",
                  torch.cuda.max_memory_allocated() / 1e9,
                  torch.cuda.max_memory_reserved() / 1e9)
    return predictions


def _save_predictions_npz(predictions: dict, output_dir: str) -> None:
    """Save each prediction tensor as a compressed .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    _log.info("Saving predictions to %s ...", output_dir)
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            path = os.path.join(output_dir, f"{key}.npz")
            np.savez_compressed(path, **{key: value})
        elif isinstance(value, torch.Tensor):
            path = os.path.join(output_dir, f"{key}.npz")
            np.savez_compressed(path, **{key: value.cpu().numpy()})
    _log.info("Predictions saved to %s", output_dir)


def _launch_viewer(predictions: dict, images_cpu, args, image_folder: str) -> None:
    """Build and run the interactive Viser 3D viewer."""
    try:
        from lingbot_map.vis import PointCloudViewer
    except ImportError:
        _log.warning("viser not installed.  Install with: pip install lingbot-map[vis]")
        _log.info("Predictions keys: %s", sorted(predictions.keys()))
        return

    _log.info("Building 3D viewer...")
    viewer = PointCloudViewer(
        pred_dict=prepare_for_visualization(predictions, images_cpu),
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        mask_sky=args.mask_sky,
        image_folder=image_folder,
        sky_mask_dir=args.sky_mask_dir,
        sky_mask_visualization_dir=args.sky_mask_visualization_dir,
    )
    _log.info("=== Pipeline complete ===")
    _log.info("3D viewer at http://localhost:%d  (Ctrl+C to stop)", args.port)
    viewer.run()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _check_gpu_memory(device)

    # ── Load ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    images, _paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
    )
    _export_preprocessed(args.export_preprocessed, images)

    model = load_model(args, device)
    _log.info("Total load time: %.1f s", time.time() - t0)

    # ── Prepare ──────────────────────────────────────────────────────────────
    dtype = _select_dtype()
    _prepare_model(model, dtype)

    images = images.to(device)
    num_frames = images.shape[0]
    _log_memory_after_load(num_frames, tuple(images.shape), args.mode)
    _auto_keyframe_interval(args, num_frames)

    # ── Optional compile warmup ─────────────────────────────────────────────
    _warmup_compile(model, images, args, dtype, num_frames)

    # ── Inference ────────────────────────────────────────────────────────────
    predictions = _run_inference(model, images, args, dtype)

    # ── Post-process ─────────────────────────────────────────────────────────
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    images_for_post = predictions["images"]

    _log.info("Post-processing predictions...")
    predictions, images_cpu = postprocess(predictions, images_for_post)
    _log.info("Post-processing complete.  Predictions: %s",
              sorted(predictions.keys()))

    # ── Output ───────────────────────────────────────────────────────────────
    if args.save_predictions:
        _save_predictions_npz(predictions, args.save_predictions)

    if args.headless:
        _log.info("Headless mode — skipping viewer.")
    else:
        _launch_viewer(predictions, images_cpu, args, resolved_image_folder)


if __name__ == "__main__":
    main()
