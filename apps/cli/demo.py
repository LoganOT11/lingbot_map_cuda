#!/usr/bin/env python3
"""LingBot-MAP unified pipeline: inference → export → render → visualize.

Modes (auto-selected based on flags):
    # Interactive 3D viewer (inference + viser)
    python demo.py --model_path models/model.pt --image_folder example/courthouse

    # Headless export (inference → NPZ)
    python demo.py --model_path models/model.pt --image_folder example/courthouse \\
        --headless --save_predictions outputs/

    # Render from saved NPZ (no inference)
    python demo.py --load_predictions outputs/scene/ --render outputs/scene.mp4

    # Full pipeline (inference → NPZ → video in one shot)
    python demo.py --model_path models/model.pt --video_path video.mp4 \\
        --render outputs/scene.mp4

    # Batch process all scenes under a folder
    python demo.py --model_path models/model.pt --input_folder /data/scenes \\
        --output_folder /data/outputs --render

    # Render with YAML config preset
    python demo.py --load_predictions outputs/scene/ --render out.mp4 \\
        --config config/indoor.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# Pre-parse --compile / --use_sdpa BEFORE importing torch, so we can set the
# CUDA allocator config before the first CUDA init.
# ═════════════════════════════════════════════════════════════════════════════
def _parse_args_pre() -> argparse.Namespace:
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

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.inference import (
    load_model as _load_model_lib,
    postprocess,
    prepare_for_visualization,
    validate_frame_count,
    estimate_gpu_memory,
)

# ── sys.path: make apps/ packages importable ───────────────────────────────
# demo.py lives in apps/cli/; we need apps/ and apps/cuda_ext/ on the path so
# that rgbd_render and render_cuda_ext resolve as top-level packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → repo root
_APPS_DIR = _PROJECT_ROOT / "apps"
if str(_APPS_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_DIR))
if str(_APPS_DIR / "cuda_ext") not in sys.path:
    sys.path.insert(0, str(_APPS_DIR / "cuda_ext"))

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
_log = logging.getLogger("demo")

# ── Named constants ─────────────────────────────────────────────────────────
_KEYFRAME_AUTO_THRESHOLD = 320
_WARM_STREAM_N_DEFAULT   = 10
_COMPILED_WARMUP_PASSES  = 3
_BF16_MIN_CAPABILITY     = 8
_GPU_MEM_WARN_GB         = 10.0


# =============================================================================
# Argument parsing (unified — all modes)
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the unified argument parser covering all pipeline stages."""
    p = argparse.ArgumentParser(
        description="LingBot-MAP: streaming 3D reconstruction — inference, export, render, visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input sources ──
    inp = p.add_argument_group("Input")
    inp.add_argument("--image_folder", type=str, default=None)
    inp.add_argument("--video_path", type=str, default=None)
    inp.add_argument("--input_folder", type=str, default=None,
                     help="Root folder containing scene sub-folders (batch mode)")
    inp.add_argument("--load_predictions", type=str, nargs="+", default=None,
                     help="Render saved NPZ predictions (skips inference)")
    inp.add_argument("--fps", type=int, default=10)
    inp.add_argument("--target_frames", type=int, default=None,
                     help="Auto-compute fps to hit target frame count from video")
    inp.add_argument("--save_frames_dir", type=str, default=None,
                     help="Cache extracted video frames as PNG for reuse")
    inp.add_argument("--image_extension", type=str, default=".jpg,.jpeg,.png,.JPG")
    inp.add_argument("--first_k", type=int, default=None)
    inp.add_argument("--last_k", type=int, default=None)
    inp.add_argument("--stride", type=int, default=1)
    inp.add_argument("--image_range", type=str, default=None,
                     help="Frame range: start:end[:stride] (e.g. 0:100:2)")
    inp.add_argument("--min_images", type=int, default=2,
                     help="Minimum images per scene in batch mode")
    inp.add_argument("--rotate_clockwise_90", action="store_true")

    # ── Model ──
    mod = p.add_argument_group("Model")
    mod.add_argument("--model_path", type=str, default=None,
                     help="Path to checkpoint (required for inference)")
    mod.add_argument("--image_size", type=int, default=518)
    mod.add_argument("--patch_size", type=int, default=14)

    # ── Inference mode ──
    inf = p.add_argument_group("Inference")
    inf.add_argument("--mode", type=str, default="streaming",
                     choices=["streaming", "windowed"])
    inf.add_argument("--enable_3d_rope", action="store_true", default=True)
    inf.add_argument("--max_frame_num", type=int, default=1024)
    inf.add_argument("--num_scale_frames", type=int, default=8)
    inf.add_argument("--keyframe_interval", type=int, default=None,
                     help="Every N-th frame is a keyframe (auto if unset)")
    inf.add_argument("--flow_threshold", type=float, default=0.0,
                     help="Flow-based keyframe threshold in px (>0 enables flow mode)")
    inf.add_argument("--max_non_keyframe_gap", type=int, default=100,
                     help="Max consecutive non-keyframes before forcing one (flow mode)")
    inf.add_argument("--kv_cache_sliding_window", type=int, default=64)
    inf.add_argument("--camera_num_iterations", type=int, default=4)
    inf.add_argument("--use_sdpa", action="store_true", default=False)
    inf.add_argument("--compile", action="store_true", default=False)
    inf.add_argument("--offload_to_cpu", action=argparse.BooleanOptionalAction, default=True)
    inf.add_argument("--window_size", type=int, default=64)
    inf.add_argument("--overlap_size", type=int, default=None)
    inf.add_argument("--overlap_keyframes", type=int, default=None)
    inf.add_argument("--scale_mode", type=str, default="median",
                     choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all"])

    # ── Output ──
    out = p.add_argument_group("Output")
    out.add_argument("--output_folder", type=str, default=None,
                     help="Output directory (batch/render modes)")
    out.add_argument("--render", type=str, default=None, const=".", nargs="?",
                     help="Render predictions to video (optional output path or directory)")
    out.add_argument("--save_predictions", type=str, default=None, const=".", nargs="?",
                     help="Save predictions as per-frame NPZ files (minimal: depth+extrinsic+intrinsic)")
    out.add_argument("--save_images", action="store_true",
                     help="Also embed preprocessed images in NPZ (needed for offline rendering)")
    out.add_argument("--save_glb", action="store_true",
                     help="Export GLB 3D model alongside other outputs")
    out.add_argument("--headless", action="store_true",
                     help="Skip 3D viewer (implied by --render / --save_predictions)")
    out.add_argument("--export_preprocessed", type=str, default=None,
                     help="Export preprocessed images as PNG to this folder")
    out.add_argument("--video_suffix", type=str, default="_pointcloud")

    # ── Visualization (viser viewer) ──
    vis = p.add_argument_group("Viewer")
    vis.add_argument("--port", type=int, default=8080)
    vis.add_argument("--conf_threshold", type=float, default=1.5,
                     help="Confidence threshold for viser viewer point filtering")
    vis.add_argument("--downsample_factor", type=int, default=5,
                     help="Point downsampling factor for viewer and render")
    vis.add_argument("--point_size", type=float, default=0.00001)

    # ── Render pipeline ──
    rnd = p.add_argument_group("Render")
    rnd.add_argument("--config", type=str, default=None,
                     help="YAML preset for render/scene/camera/overlay defaults")
    rnd.add_argument("--video_fps", type=int, default=30)
    rnd.add_argument("--video_width", type=int, default=None)
    rnd.add_argument("--video_height", type=int, default=None)
    rnd.add_argument("--render_stride", type=int, default=1)

    # ── Camera ──
    cam = p.add_argument_group("Camera")
    cam.add_argument("--camera_mode", type=str, default="follow",
                     choices=["follow", "birdeye", "static", "pivot"])
    cam.add_argument("--fov", type=float, default=None)
    cam.add_argument("--smooth_window", type=int, default=None)
    cam.add_argument("--back_offset", type=float, default=0.3)
    cam.add_argument("--up_offset", type=float, default=0.1)
    cam.add_argument("--look_offset", type=float, default=0.5)
    cam.add_argument("--follow_scale_frames", type=int, default=0)
    cam.add_argument("--birdeye_start", type=str, default=None,
                     help="Comma-separated frame indices for birdeye inserts")
    cam.add_argument("--birdeye_duration", type=str, default=None)
    cam.add_argument("--birdeye_transition", type=int, default=None)
    cam.add_argument("--reveal_height_mult", type=float, default=None)

    # ── Camera overlay ──
    ovr = p.add_argument_group("Overlay")
    ovr.add_argument("--camera_vis", type=str, default="",
                     choices=["", "default", "frustum", "textured", "trail"])
    ovr.add_argument("--trail_color_ramp", type=str, default=None,
                     choices=["cyan_blue", "white", "rainbow", "red", "green", "yellow", "magenta"])
    ovr.add_argument("--trail_line_width", type=float, default=None)
    ovr.add_argument("--trail_tail_len", type=int, default=None)
    ovr.add_argument("--head_num_frames", type=int, default=None)
    ovr.add_argument("--head_point_size", type=float, default=None)
    ovr.add_argument("--head_frustum_scale", type=float, default=None)
    ovr.add_argument("--head_frustum_line_width", type=float, default=None)
    ovr.add_argument("--head_frustum_color", type=str, default=None)
    ovr.add_argument("--head_texture_alpha", type=float, default=None)
    ovr.add_argument("--frame_tag", action="store_true")
    ovr.add_argument("--frame_tag_position", type=str, default=None,
                     choices=["top_left", "top_right", "bottom_left", "bottom_right"])

    # ── Sky masking ──
    sky = p.add_argument_group("Sky")
    sky.add_argument("--mask_sky", action="store_true")
    sky.add_argument("--skyseg_model_path", type=str, default="skyseg.onnx")
    sky.add_argument("--sky_mask_dir", type=str, default=None)
    sky.add_argument("--sky_mask_visualization_dir", type=str, default=None)
    sky.add_argument("--visualize_sky_mask_only", action="store_true",
                     help="Generate sky masks and exit without inference")

    # ── Scene (render pipeline) ──
    scn = p.add_argument_group("Scene")
    scn.add_argument("--voxel_size", type=float, default=None)

    scn.add_argument("--keyframes_only_points", action="store_true")
    scn.add_argument("--max_render_points", type=int, default=1_000_000_000)
    scn.add_argument("--vis_threshold", type=float, default=1.5,
                     help="Confidence threshold for render point filtering")

    # ── Batch control ──
    bat = p.add_argument_group("Batch")
    bat.add_argument("--scenes", type=str, nargs="+", default=None,
                     help="Process only these scene names")
    bat.add_argument("--exclude_scenes", type=str, nargs="+", default=None)
    bat.add_argument("--skip_existing", action="store_true")
    bat.add_argument("--dry_run", action="store_true")
    bat.add_argument("--num_workers", type=int, default=8)
    bat.add_argument("--lazy_images", action="store_true",
                     help="Load images via memory-mapped file (O(window) RAM, not O(frames))")

    return p


# =============================================================================
# Stage 1: Image / video loading
# =============================================================================

def _normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def _list_image_paths(folder: str, image_ext: str) -> list[str]:
    paths = []
    for ext in image_ext.split(","):
        ext_norm = _normalize_ext(ext)
        if not ext_norm:
            continue
        paths.extend(glob.glob(os.path.join(folder, f"*{ext_norm}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext_norm.upper()}")))
    return sorted(set(paths))


def _apply_image_filters(
    paths: list[str],
    first_k: int | None = None,
    last_k: int | None = None,
    stride: int = 1,
    image_range: str | None = None,
) -> list[str]:
    if image_range:
        parts = image_range.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(f"Invalid --image_range '{image_range}', expected start:end[:stride]")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else len(paths)
        range_stride = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        paths = paths[start:end:range_stride]
    else:
        if first_k is not None and first_k > 0:
            paths = paths[:first_k]
        if last_k is not None and last_k > 0:
            paths = paths[-last_k:]
    if stride > 1:
        paths = paths[::stride]
    return paths


def _load_images_from_folder(
    folder: str, args,
) -> tuple[torch.Tensor, str]:
    """Load images from a folder, return (tensor, resolved_folder)."""
    paths = _list_image_paths(folder, args.image_extension)
    paths = _apply_image_filters(
        paths, args.first_k, args.last_k, args.stride, getattr(args, 'image_range', None),
    )
    if not paths:
        raise ValueError(f"No images found in {folder}")

    if args.rotate_clockwise_90:
        rotated_dir = tempfile.mkdtemp(prefix="lingbot_rot_cw90_")
        rotated_paths = []
        for p in tqdm(paths, desc="Rotating images 90° CW"):
            out_path = os.path.join(rotated_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out_path)
            rotated_paths.append(out_path)
        paths = rotated_paths
        folder = rotated_dir
        _log.info("Rotated %d images → %s", len(paths), rotated_dir)

    _log.info("Loading %d images from %s...", len(paths), folder)
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=args.image_size, patch_size=args.patch_size,
    )
    _log.info("Preprocessed to %dx%d", images.shape[-1], images.shape[-2])
    return images, folder


def _load_images_lazy(
    folder: str, args,
) -> tuple[torch.Tensor, str, str]:
    """Load images via memory-mapped numpy array → torch tensor.

    Images are preprocessed in parallel and written to a temporary memmap
    file.  The returned tensor is a view of the memmap — the OS pages frames
    in/out as ``inference_windowed`` accesses each window.  Peak RAM is
    O(window_size) instead of O(num_frames).

    Returns (tensor, folder, mmap_path) — caller must delete mmap_path after use.
    """
    import tempfile

    paths = _list_image_paths(folder, args.image_extension)
    paths = _apply_image_filters(
        paths, args.first_k, args.last_k, args.stride,
        getattr(args, 'image_range', None),
    )
    if not paths:
        raise ValueError(f"No images found in {folder}")

    S = len(paths)

    # Determine resolution from first image (same as load_and_preprocess_images logic)
    from PIL import Image, ImageOps
    img0 = Image.open(paths[0])
    img0 = ImageOps.exif_transpose(img0)
    w0, h0 = img0.size
    new_w = args.image_size
    new_h = round(h0 * (new_w / w0) / args.patch_size) * args.patch_size
    if new_h > args.image_size:
        new_h = args.image_size
    resolution = (new_h, new_w)

    # Create temp memmap file
    tmp = tempfile.NamedTemporaryFile(suffix='.dat', delete=False, prefix='lingbot_lazy_')
    tmp.close()
    mmap = np.memmap(tmp.name, dtype='float32', mode='w+', shape=(S, 3, resolution[0], resolution[1]))

    _log.info("Loading %d images → memmap %s (lazy, O(window) RAM)", S, tmp.name)
    t0 = time.time()

    # Preprocess in parallel, write directly to memmap
    from concurrent.futures import ThreadPoolExecutor

    # Load in batches to keep peak RAM bounded (~2× batch_size frames)
    batch_size = min(64, S)
    for batch_start in range(0, S, batch_size):
        batch_end = min(batch_start + batch_size, S)
        batch_paths = paths[batch_start:batch_end]
        batch_tensor = load_and_preprocess_images(
            batch_paths, mode="crop",
            image_size=args.image_size, patch_size=args.patch_size,
        )
        mmap[batch_start:batch_end] = batch_tensor.numpy()
        del batch_tensor

    mmap.flush()
    elapsed = time.time() - t0
    _log.info("Memmap ready: %d frames @ %dx%d in %.1f s (%.1f MB on disk)",
              S, resolution[1], resolution[0], elapsed,
              os.path.getsize(tmp.name) / 1024 / 1024)

    tensor = torch.from_numpy(mmap)
    return tensor, folder, tmp.name


def _load_images_from_video(
    video_path: str, args,
) -> tuple[torch.Tensor, str]:
    """Load frames from video with optional caching, return (tensor, frame_dir)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = args.save_frames_dir or os.path.join(
        args.output_folder or os.path.dirname(video_path), f"{video_name}_frames",
    )

    # Try cached frames first
    if os.path.isdir(save_dir):
        cached = sorted(glob.glob(os.path.join(save_dir, 'frame_*.png')))
        if cached:
            cached = _apply_image_filters(
                cached, args.first_k, args.last_k, args.stride,
                getattr(args, 'image_range', None),
            )
            _log.info("Loading %d cached frames from %s", len(cached), save_dir)
            images = load_and_preprocess_images(
                cached, mode="crop", image_size=args.image_size, patch_size=args.patch_size,
            )
            _log.info("Preprocessed to %dx%d", images.shape[-1], images.shape[-2])
            return images, save_dir

    # Decode video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = args.fps
    if fps is None and args.target_frames is not None and args.target_frames > 0:
        fps = max(1, round(src_fps * args.target_frames / total_frames))
        _log.info("Auto fps: %d (target ~%d frames)", fps, args.target_frames)

    interval = max(1, round(src_fps / fps)) if fps is not None and fps > 0 else 1
    max_collect = args.first_k if (args.first_k is not None and args.first_k > 0) else float('inf')

    os.makedirs(save_dir, exist_ok=True)
    _log.info("Extracting frames from %s (interval=%d, target_fps=%s)...",
              video_path, interval, fps)

    images = []
    idx, collected = 0, 0
    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
    while True:
        if idx % interval == 0:
            ret, frame = cap.read()
            if not ret:
                break
            # Save raw frame for reuse
            cv2.imwrite(os.path.join(save_dir, f"frame_{collected:06d}.png"), frame)

            # Preprocess inline
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            new_width = args.image_size
            new_height = round(h * (new_width / w) / args.patch_size) * args.patch_size
            resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            img = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            if new_height > args.image_size:
                start_y = (new_height - args.image_size) // 2
                img = img[:, start_y:start_y + args.image_size, :]
            images.append(img)

            collected += 1
            if collected >= max_collect:
                pbar.update(1)
                break
        else:
            if not cap.grab():
                break
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    images = torch.stack(images)
    _log.info("Extracted %d frames (saved to %s)", collected, save_dir)

    if args.stride > 1:
        images = images[::args.stride]
        _log.info("Applied stride=%d: %d → %d frames", args.stride, collected, len(images))

    _log.info("Preprocessed to %dx%d", images.shape[-1], images.shape[-2])
    return images, save_dir


def load_images(args) -> tuple[torch.Tensor, str]:
    """Load images from folder or video.  Returns (tensor, source_dir)."""
    if args.video_path:
        return _load_images_from_video(args.video_path, args)
    else:
        return _load_images_from_folder(args.image_folder, args)


# =============================================================================
# Stage 2: Scene discovery (batch mode)
# =============================================================================

def _find_image_folder(scene_path: str, image_ext: str) -> str | None:
    """Locate a folder containing images within a scene directory."""
    exts = [_normalize_ext(x).lower() for x in image_ext.split(",") if x.strip()]
    common = ["images", "imgs", "rgb", "color", "frames", "input", "raw"]

    def _has_images(folder: str) -> bool:
        for e in exts:
            if glob.glob(os.path.join(folder, f"*{e}")):
                return True
            if glob.glob(os.path.join(folder, f"*{e.upper()}")):
                return True
        return False

    if _has_images(scene_path):
        return scene_path
    for name in common:
        cand = os.path.join(scene_path, name)
        if os.path.isdir(cand) and _has_images(cand):
            return cand
    for root, _, _ in os.walk(scene_path):
        if root[len(scene_path):].count(os.sep) > 2:
            continue
        if _has_images(root):
            return root
    return None


def discover_scenes(args) -> list[tuple[str, str, int]]:
    """Find scenes under --input_folder.  Returns [(name, image_folder, image_count), ...]."""
    scenes = []
    root = _find_image_folder(args.input_folder, args.image_extension)
    if root:
        count = len(_list_image_paths(root, args.image_extension))
        if count >= args.min_images:
            scenes.append((os.path.basename(os.path.abspath(args.input_folder)), root, count))
            if root == args.input_folder:
                # input_folder IS the image folder → done
                pass
            else:
                # also scan subdirs
                pass

    # Always scan subdirs for multi-scene batch
    for item in sorted(os.listdir(args.input_folder)):
        item_path = os.path.join(args.input_folder, item)
        if not os.path.isdir(item_path):
            continue
        img_folder = _find_image_folder(item_path, args.image_extension)
        if img_folder is None:
            continue
        count = len(_list_image_paths(img_folder, args.image_extension))
        if count >= args.min_images:
            scenes.append((item, img_folder, count))

    # Deduplicate
    unique, seen = [], set()
    for s in scenes:
        key = (s[0], s[1])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# =============================================================================
# Stage 3: Model loading
# =============================================================================

def _load_model(args, device: torch.device):
    return _load_model_lib(
        args.model_path, device,
        mode=args.mode,
        image_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        num_scale_frames=args.num_scale_frames,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )


# =============================================================================
# Stage 4: torch.compile (opt-in)
# =============================================================================

def _compile_model(model):
    """torch.compile hot modules with mode='reduce-overhead'."""
    agg = model.aggregator
    # frame_blocks
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    # DINOv2 blocks
    if hasattr(agg.patch_embed, "blocks"):
        for i, b in enumerate(agg.patch_embed.blocks):
            agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    # global_blocks
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive scale → streaming forward passes to capture CUDA graphs."""
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


def _warmup_compile(model, images: torch.Tensor, args, dtype: torch.dtype,
                    num_frames: int) -> None:
    """Eager warmup + compile + compiled warmup if --compile is set."""
    if not args.compile:
        return
    if args.mode != "streaming":
        _log.info("--compile only applies to --mode streaming (got %r); skipping", args.mode)
        return
    if getattr(model.aggregator, "use_sdpa", False):
        _log.info("--compile skipped: SDPA backend not compatible with CUDA graphs")
        return

    scale_w = min(args.num_scale_frames, num_frames)
    if scale_w >= num_frames:
        scale_w = max(1, num_frames - 1)
    stream_n = min(_WARM_STREAM_N_DEFAULT, max(1, num_frames - scale_w))

    _log.info("Warmup eager (scale=%d + %d streaming, kf_int=%d)...",
              scale_w, stream_n, args.keyframe_interval)
    t_w = time.time()
    _warm_streaming(model, images, scale_w, stream_n, dtype,
                    passes=1, keyframe_interval=args.keyframe_interval)
    _log.info("  eager warmup: %.1f s", time.time() - t_w)

    _log.info("Compiling hot modules...")
    _compile_model(model)

    _log.info("Warmup compiled (%dx dress rehearsal)...", _COMPILED_WARMUP_PASSES)
    t_w = time.time()
    _warm_streaming(model, images, scale_w, stream_n, dtype,
                    passes=_COMPILED_WARMUP_PASSES, keyframe_interval=args.keyframe_interval)
    _log.info("  compiled warmup: %.1f s", time.time() - t_w)

    # Destroy warmup KV cache so real images get the correct tokens_per_frame
    model.aggregator.kv_cache_manager = None


# =============================================================================
# Stage 5: Inference
# =============================================================================

def _run_inference(model, images: torch.Tensor, args, dtype: torch.dtype,
                  per_window_callback=None) -> dict:
    """Run streaming or windowed inference. Returns raw predictions dict.

    Args:
        per_window_callback: Optional callable(w_pred, start, end) fired
            after each window's raw predictions are assembled (before
            alignment).  Used for incremental NPZ saving.
    """
    _log.info("Running %s inference (dtype=%s)...", args.mode, dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    flow_threshold = getattr(args, "flow_threshold", 0.0)
    max_non_keyframe_gap = getattr(args, "max_non_keyframe_gap", 30)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images, num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:
            if flow_threshold > 0:
                _log.info("Flow-based keyframe: threshold=%.1f px, max_gap=%d",
                          flow_threshold, max_non_keyframe_gap)
            predictions = model.inference_windowed(
                images, window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                scale_mode=args.scale_mode,
                output_device=output_device,
                keyframe_interval=args.keyframe_interval,
                flow_threshold=flow_threshold,
                max_non_keyframe_gap=max_non_keyframe_gap,
                per_window_callback=per_window_callback,
            )

    num_frames = images.shape[0]
    elapsed = time.time() - t0
    _log.info("Inference done: %.1f s (%.1f FPS)", elapsed, num_frames / max(elapsed, 1e-6))
    if torch.cuda.is_available():
        _log.info("GPU peak: alloc=%.2f GB, reserved=%.2f GB",
                  torch.cuda.max_memory_allocated() / 1e9,
                  torch.cuda.max_memory_reserved() / 1e9)
    return predictions


# =============================================================================
# Stage 6: Post-processing
# =============================================================================

def _postprocess(predictions: dict, images: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Convert pose encoding → extrinsics/intrinsics, move to CPU."""
    _log.info("Post-processing predictions...")
    predictions, images_cpu = postprocess(predictions, images)
    _log.info("Post-processed. Keys: %s", sorted(predictions.keys()))
    return predictions, images_cpu


# =============================================================================
# Stage 7: NPZ I/O (per-frame parallel)
# =============================================================================

def _save_predictions_npz(predictions: dict, output_path: str) -> str:
    """Save predictions as per-frame .npz files (parallel I/O).

    Creates a directory: frame_000000.npz, frame_000001.npz, ..., meta.npz.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Convert torch tensors to numpy (postprocess returns CPU tensors, not ndarrays)
    _clean: dict = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            _clean[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            _clean[k] = v
        else:
            _clean[k] = v
    predictions = _clean

    dir_path = output_path
    if dir_path.endswith('.npz'):
        dir_path = dir_path[:-4]

    # Clean stale files
    if os.path.isdir(dir_path):
        for f in glob.glob(os.path.join(dir_path, 'frame_*.npz')):
            os.remove(f)
        meta_file = os.path.join(dir_path, 'meta.npz')
        if os.path.exists(meta_file):
            os.remove(meta_file)
    os.makedirs(dir_path, exist_ok=True)

    # Separate sequence arrays from metadata
    # Skip truly redundant keys only.  depth_conf is NOT redundant — it is the
    # model's per-pixel uncertainty and essential for clean point clouds.
    seq_keys = []
    meta_dict = {}
    S = None
    _SKIP_KEYS = {'pose_enc'}  # redundant with extrinsic
    for key, value in predictions.items():
        if not isinstance(value, np.ndarray):
            continue
        if key in _SKIP_KEYS:
            continue
        if value.ndim >= 2 and S is None:
            S = value.shape[0]
        if value.ndim >= 2 and value.shape[0] == S:
            seq_keys.append(key)
        else:
            meta_dict[key] = value

    if S is None:
        save_dict = {k: v for k, v in predictions.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(os.path.join(dir_path, "frame_000000.npz"), **save_dict)
        _log.info("Saved predictions to %s/ (1 file, %d keys)", dir_path, len(save_dict))
        return dir_path

    def _save_frame(frame_idx):
        frame_dict = {}
        for key in seq_keys:
            val = predictions[key][frame_idx]
            # Store depth and depth_conf as float16 (2× smaller, sufficient precision)
            if key in ('depth', 'depth_conf') and val.dtype == np.float32:
                val = val.astype(np.float16)
            # Store depth_conf as uint8 (4× smaller than float16, negligible loss:
            # values range 1–24, used only for binary threshold filtering)
            if key == 'depth_conf' and val.dtype == np.float16:
                val = np.round(val.astype(np.float32)).clip(0, 255).astype(np.uint8)
            # Store images as uint8 (4× smaller than float32)
            if key == 'images' and val.dtype == np.float32:
                val = (val * 255).clip(0, 255).astype(np.uint8)
            frame_dict[key] = val
        np.savez_compressed(os.path.join(dir_path, f"frame_{frame_idx:06d}.npz"), **frame_dict)

    n_workers = min(32, S)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_save_frame, range(S)))

    if meta_dict:
        np.savez_compressed(os.path.join(dir_path, "meta.npz"), **meta_dict)

    _log.info("Saved predictions to %s/ (%d frames, %d keys/frame)", dir_path, S, len(seq_keys))
    return dir_path


# ── Incremental NPZ (per-window streaming save) ─────────────────────────

def _setup_incremental_npz(output_path: str, num_frames: int, args) -> dict:
    """Prepare a directory for incremental per-window NPZ saving.

    Creates a per-window layout::

        output_dir/
        ├── window_000/  (local frame_000000.npz …)
        ├── window_001/
        ├── …
        ├── alignment.npz   (chunk_scales, chunk_transforms — saved at finalize)
        └── windows.json    (window boundaries — saved at finalize)

    Returns a state dict passed to _on_window_complete and
    _finalize_incremental_npz.
    """
    dir_path = output_path
    if dir_path.endswith('.npz'):
        dir_path = dir_path[:-4]

    # Clean stale files (both old flat and new per-window)
    if os.path.isdir(dir_path):
        for f in glob.glob(os.path.join(dir_path, 'frame_*.npz')):
            os.remove(f)
        for f in glob.glob(os.path.join(dir_path, 'meta.npz')):
            os.remove(f)
        for f in glob.glob(os.path.join(dir_path, 'alignment.npz')):
            os.remove(f)
        for f in glob.glob(os.path.join(dir_path, 'windows.json')):
            os.remove(f)
        # Remove old window subdirectories
        for d in glob.glob(os.path.join(dir_path, 'window_*')):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)

    return {
        'dir': dir_path,
        'num_frames': num_frames,
        'save_images': getattr(args, 'save_images', False),
        'window_count': 0,
        'window_boundaries': [],  # list of (start, end) per window
        'image_size': getattr(args, 'image_size', 518),
    }


def _on_window_complete(w_pred: dict, start: int, end: int,
                        state: dict, args) -> None:
    """Per-window callback: save raw (UNALIGNED) window predictions.

    Each window gets its own subdirectory with locally-indexed frames.
    Alignment transforms are saved later by :func:`_finalize_incremental_npz`
    and applied at load time.

    *w_pred* keys are batched as [1, window_len, ...] from the model.
    """
    from concurrent.futures import ThreadPoolExecutor
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    dir_path = state['dir']
    window_len = end - start
    window_idx = state['window_count']
    window_dir = os.path.join(dir_path, f"window_{window_idx:03d}")
    os.makedirs(window_dir, exist_ok=True)

    # Compute extrinsic / intrinsic from pose_enc on-the-fly so the
    # per-window NPZ files are self-contained (needed for deferred alignment).
    # pose_encoding_to_extri_intri returns w2c (OpenCV: cam-from-world);
    # we invert to c2w to match the legacy NPZ format.
    if 'pose_enc' in w_pred and 'extrinsic' not in w_pred:
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general
        pe = w_pred['pose_enc']
        # depth shape: [1, window_len, H, W, 1]
        H_img, W_img = w_pred['depth'].shape[2], w_pred['depth'].shape[3]
        ext_w2c, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H_img, W_img))
        # Invert w2c → c2w
        ext_4x4 = torch.zeros(
            ext_w2c.shape[0], ext_w2c.shape[1], 4, 4,
            device=ext_w2c.device, dtype=ext_w2c.dtype,
        )
        ext_4x4[..., :3, :4] = ext_w2c
        ext_4x4[..., 3, 3] = 1.0
        ext_c2w_4x4 = closed_form_inverse_se3_general(ext_4x4)
        ext_c2w = ext_c2w_4x4[..., :3, :4]
        w_pred = dict(w_pred)  # shallow copy so we don't mutate original
        w_pred['extrinsic'] = ext_c2w
        w_pred['intrinsic'] = intr

    # Extract per-frame arrays from the batched window dict
    frame_dicts = []
    for j in range(window_len):
        fd = {}
        for key in ('depth', 'depth_conf', 'extrinsic', 'intrinsic'):
            if key not in w_pred:
                continue
            val = w_pred[key]
            # val shape: [1, window_len, ...] → index j
            if isinstance(val, torch.Tensor):
                val = val[0, j].detach().cpu().numpy()
            elif isinstance(val, np.ndarray) and val.ndim >= 2:
                val = val[0, j]
            else:
                val = np.asarray(val)

            # Dtype optimisations.
            # depth_conf → uint8 (lossy but sufficient for threshold filtering).
            # depth is kept as float32: alignment at load time multiplies by
            # a scale factor, and float16 quantisation before scaling would
            # introduce ~0.005 error that breaks bit-identical comparison.
            if key == 'depth_conf' and val.dtype == np.float32:
                val = np.round(val).clip(0, 255).astype(np.uint8)
            elif key == 'depth_conf' and val.dtype == np.float16:
                val = np.round(val.astype(np.float32)).clip(0, 255).astype(np.uint8)

            fd[key] = val
        frame_dicts.append(fd)

    # Skip if no keys found (flow-callback for alignment-only windows)
    if not frame_dicts or not any(fd for fd in frame_dicts):
        return

    # Save per-frame NPZ files in parallel (LOCAL indices within the window)
    def _save_one(idx):
        path = os.path.join(window_dir, f"frame_{idx:06d}.npz")
        np.savez_compressed(path, **frame_dicts[idx])

    n_workers = min(8, window_len)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_save_one, range(window_len)))

    # Track window boundary for later finalisation
    state['window_boundaries'].append((start, end))
    state['window_count'] += 1
    _log.debug("Incremental save: window %d → %s/ (frames [%d, %d))",
               window_idx, window_dir, start, end)


def _finalize_incremental_npz(predictions: dict, state: dict, args) -> str:
    """After inference completes, finalise the NPZ directory.

    Saves two formats side-by-side::

        *Per-window* (primary — enables deferred alignment)::
            window_000/  window_001/  …  alignment.npz  windows.json

        *Legacy flat* (for backward-compatible comparison)::
            aligned/frame_000000.npz  frame_000001.npz  …  meta.npz

    The legacy flat subdirectory contains **aligned** predictions and is
    bit-identical to the backup format.  Loading code
    (:func:`_load_predictions_from_npz`) auto-detects the per-window format
    and applies alignment at load time; the aligned/ subdirectory is
    available for direct comparison with older exports.

    Returns the NPZ directory path.
    """
    import json as _json

    dir_path = state['dir']
    window_boundaries = state.get('window_boundaries', [])

    # Calculate overlap from consecutive window boundaries
    overlap = 0
    if len(window_boundaries) >= 2:
        _, prev_end = window_boundaries[0]
        next_start, _ = window_boundaries[1]
        overlap = prev_end - next_start

    # ── Per-window metadata ──
    chunk_scales = predictions.get('chunk_scales')
    chunk_transforms = predictions.get('chunk_transforms')
    if chunk_scales is not None and chunk_transforms is not None:
        if isinstance(chunk_scales, torch.Tensor):
            chunk_scales = chunk_scales.detach().cpu().numpy()
        if isinstance(chunk_transforms, torch.Tensor):
            chunk_transforms = chunk_transforms.detach().cpu().numpy()
        np.savez_compressed(
            os.path.join(dir_path, 'alignment.npz'),
            chunk_scales=chunk_scales,
            chunk_transforms=chunk_transforms,
        )
        _log.info("Saved alignment.npz: %d windows, overlap=%d",
                  len(chunk_scales), overlap)
    else:
        _log.info("No alignment metadata — single window or streaming mode")

    windows_meta = {
        'num_frames': int(state['num_frames']),
        'num_windows': len(window_boundaries),
        'overlap': overlap,
        'windows': [list(b) for b in window_boundaries],  # [[start, end], ...]
    }
    with open(os.path.join(dir_path, 'windows.json'), 'w') as f:
        _json.dump(windows_meta, f, indent=2)
    _log.info("Saved windows.json: %d windows, %d frames",
              windows_meta['num_windows'], windows_meta['num_frames'])

    # ── Meta (frame_type, is_keyframe, images) ──
    meta_dict = {}
    for key in ('frame_type', 'is_keyframe'):
        if key in predictions:
            val = predictions[key]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            meta_dict[key] = val
    if getattr(args, 'save_images', False) and 'images' in predictions:
        imgs = predictions['images']
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu().numpy()
        if imgs.ndim == 4 and imgs.shape[0] == state['num_frames']:
            meta_dict['images'] = imgs
    if meta_dict:
        np.savez_compressed(os.path.join(dir_path, 'meta.npz'), **meta_dict)

    # ── Legacy flat aligned copy (for backward-compatible comparison) ──
    aligned_dir = os.path.join(dir_path, 'aligned')
    _save_predictions_npz(predictions, aligned_dir)

    return dir_path


def _load_per_window_format(dir_path: str) -> dict:
    """Load predictions from the per-window NPZ format.

    Detected by the presence of ``window_000/``.  Loads each window's frames,
    applies cumulative alignment transforms, and stitches windows together
    (handling overlap deduplication).

    Returns a flat predictions dict (same structure as the legacy flat format).
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor
    from lingbot_map.utils.alignment import apply_alignment_to_frame

    # Load metadata
    with open(os.path.join(dir_path, 'windows.json'), 'r') as f:
        win_meta = _json.load(f)
    windows = win_meta['windows']  # [[start, end], ...]
    overlap = win_meta.get('overlap', 0)
    num_windows = win_meta['num_windows']
    num_frames = win_meta['num_frames']

    # Load alignment (transforms are already cumulative — each maps its
    # window directly into window 0's coordinate frame).
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

    # Load all per-window frames in parallel
    _log.info("Loading %d windows from per-window format...", num_windows)

    def _load_window(wi):
        wdir = os.path.join(dir_path, f"window_{wi:03d}")
        frame_files = sorted(glob.glob(os.path.join(wdir, 'frame_*.npz')))
        frames = []
        for ff in frame_files:
            data = np.load(ff, allow_pickle=False)
            frames.append({key: data[key] for key in data.files})
        return frames

    with ThreadPoolExecutor(max_workers=min(16, num_windows)) as pool:
        all_window_frames = list(pool.map(_load_window, range(num_windows)))

    # Apply alignment per window and build global frame list
    aligned_frames = []  # list of dicts, one per global frame
    for wi in range(num_windows):
        start, end = windows[wi]
        window_frames = all_window_frames[wi]
        s = float(cum_scales[wi])
        R = cum_R[wi]
        t_vec = cum_t[wi]

        is_last = wi == num_windows - 1

        for local_idx, fd in enumerate(window_frames):
            global_idx = start + local_idx

            # Overlap deduplication: non-final windows only contribute
            # frames before the overlap region.
            if not is_last and global_idx >= end - overlap:
                continue

            # Apply alignment transform
            result = apply_alignment_to_frame(
                fd.get('extrinsic', np.eye(3, dtype=np.float32)[:, :4]),
                fd.get('depth', np.zeros((1, 1), dtype=np.float16)),
                R, t_vec, s,
                depth_conf=fd.get('depth_conf'),
                intrinsic=fd.get('intrinsic'),
            )
            aligned_frames.append(result)

    # Stack into flat predictions dict
    all_keys = list(aligned_frames[0].keys())
    predictions = {}
    for key in all_keys:
        arr = np.stack([fd[key] for fd in aligned_frames], axis=0)
        if key in ('depth', 'depth_conf') and arr.dtype in (np.float16, np.uint8):
            arr = arr.astype(np.float32)
        predictions[key] = arr

    # Load meta.npz (frame_type, is_keyframe, images) if present
    meta_path = os.path.join(dir_path, 'meta.npz')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        for key in meta.files:
            predictions[key] = meta[key]

    _log.info("Loaded per-window predictions: %d frames, keys=%s",
              len(aligned_frames), sorted(predictions.keys()))
    return predictions


def _load_predictions_from_npz(input_path: str) -> dict:
    """Load predictions from NPZ directory or single file.

    Supports three formats:
      1. **Per-window** — ``window_000/`` subdirectories with deferred alignment
         (alignment applied at load time).
      2. **Legacy flat** — ``frame_*.npz`` files in a single directory.
      3. **Single .npz file** — classic combined NPZ.

    Returns a dict with numpy arrays.  Depth/depth_conf upcast to float32.
    """
    from concurrent.futures import ThreadPoolExecutor

    if os.path.isdir(input_path):
        # Detect per-window format
        if os.path.isdir(os.path.join(input_path, 'window_000')):
            return _load_per_window_format(input_path)

        # Legacy flat format
        frame_files = sorted(glob.glob(os.path.join(input_path, 'frame_*.npz')))
        if not frame_files:
            npy_files = sorted(glob.glob(os.path.join(input_path, '*.npy')))
            if npy_files:
                predictions = {}
                for path in tqdm(npy_files, desc="Loading .npy"):
                    key = os.path.splitext(os.path.basename(path))[0]
                    predictions[key] = np.load(path, allow_pickle=False)
                _log.info("Loaded predictions from %s (%d .npy files)", input_path, len(npy_files))
                return predictions
            raise ValueError(f"No frame_*.npz or *.npy files found in {input_path}")

        n_frames = len(frame_files)
        workers = min(32, n_frames)

        def _load_one(path):
            data = np.load(path, allow_pickle=False)
            return {key: data[key] for key in data.files}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            frame_dicts = list(tqdm(
                pool.map(_load_one, frame_files),
                total=n_frames,
                desc=f"Loading {n_frames} per-frame NPZs",
                unit="frame",
            ))

        all_keys = list(frame_dicts[0].keys())
        predictions = {}
        for key in all_keys:
            arr = np.stack([fd[key] for fd in frame_dicts], axis=0)
            # Upcast float16/uint8 depth/depth_conf to float32 for the viewer/renderer
            if key in ('depth', 'depth_conf') and arr.dtype in (np.float16, np.uint8):
                arr = arr.astype(np.float32)
            predictions[key] = arr

        # Load metadata if present
        meta_path = os.path.join(input_path, 'meta.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            for key in meta.files:
                predictions[key] = meta[key]
    else:
        data = np.load(input_path, allow_pickle=True)
        predictions = {key: data[key] for key in data.files}

    _log.info("Loaded predictions from %s (keys: %s)", input_path, sorted(predictions.keys()))
    return predictions


# =============================================================================
# Stage 8: GLB export
# =============================================================================

def _export_glb(predictions: dict, output_path: str, args) -> None:
    from lingbot_map.vis import predictions_to_glb
    scene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_threshold,
        filter_by_frames="all",
        mask_sky=args.mask_sky,
        target_dir=os.path.dirname(output_path),
        prediction_mode="Predicted Pointmap",
    )
    scene.export(output_path)
    _log.info("GLB saved to %s", output_path)


# =============================================================================
# Stage 9: Render pipeline (NPZ → video)
# =============================================================================

def _resolve_sky_artifact_dirs(args, artifact_name: str) -> tuple[str | None, str | None]:
    """Resolve per-scene sky mask dirs."""
    use_subdirs = getattr(args, "use_per_scene_sky_dirs", False)
    def _resolve(base, name):
        if base is None:
            return None
        return os.path.join(base, name) if use_subdirs else base
    return _resolve(args.sky_mask_dir, artifact_name), _resolve(
        args.sky_mask_visualization_dir, artifact_name,
    )


def _render_npz(npz_path: str, output_video: str, args, artifact_name: str | None = None) -> bool:
    """Render an NPZ directory/file to MP4 using the rgbd_render offline pipeline."""
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    from rgbd_render.config import (
        PipelineConfig, CameraSegment, _build_segments_from_flat_args, _set_nested,
    )
    from rgbd_render.camera import build_camera_path
    from rgbd_render.overlay import build_overlays
    from rgbd_render.pipeline.builder import SceneBuilder
    from rgbd_render.pipeline.offline import OfflinePipeline

    def _rlog(msg):
        tqdm.write(f"[render] {msg}")

    yaml_path = getattr(args, 'config', None)
    cfg = PipelineConfig.from_yaml(yaml_path) if yaml_path else PipelineConfig()
    user_set = getattr(args, '_user_supplied', set())

    def _apply(cfg_path, arg_name):
        val = getattr(args, arg_name, None)
        if val is None:
            return
        if yaml_path and arg_name not in user_set:
            return
        _set_nested(cfg, cfg_path, val)

    cfg.input = npz_path
    cfg.output = output_video
    cfg.fast_review = 0

    _apply('fps', 'video_fps')
    _apply('frame_stride', 'render_stride')
    _apply('render.width', 'video_width')
    _apply('render.height', 'video_height')
    _apply('render.point_size', 'point_size')
    _apply('preprocess.vis_threshold', 'vis_threshold')
    _apply('preprocess.conf_threshold', 'conf_threshold')
    _apply('preprocess.mask_sky', 'mask_sky')
    _apply('preprocess.sky_model', 'skyseg_model_path')

    artifact_name = artifact_name or os.path.splitext(os.path.basename(output_video))[0]
    sky_dir, sky_viz_dir = _resolve_sky_artifact_dirs(args, artifact_name)
    cfg.preprocess.sky_mask_dir = sky_dir
    cfg.preprocess.sky_mask_visualization_dir = sky_viz_dir

    # Camera segments
    _SEG_ARGS = {
        'camera_mode', 'smooth_window', 'back_offset', 'up_offset',
        'look_offset', 'follow_scale_frames', 'birdeye_start',
        'birdeye_duration', 'reveal_height_mult',
    }
    keep_yaml_segments = (
        yaml_path and bool(cfg.camera.segments) and not (user_set & _SEG_ARGS)
    )
    if not keep_yaml_segments:
        camera_mode = getattr(args, 'camera_mode', 'follow') or 'follow'
        has_birdeye = bool(getattr(args, 'birdeye_start', None)) \
            and bool(getattr(args, 'birdeye_duration', None))

        if camera_mode == 'follow' and has_birdeye:
            cfg.camera.segments = _build_segments_from_flat_args({
                'smooth_window': args.smooth_window,
                'back_offset': args.back_offset,
                'up_offset': args.up_offset,
                'look_offset': args.look_offset,
                'follow_scale_frames': args.follow_scale_frames,
                'birdeye_start': args.birdeye_start,
                'birdeye_duration': args.birdeye_duration,
                'reveal_height_mult': args.reveal_height_mult,
            })
        else:
            seg_kwargs = {'mode': camera_mode, 'frames': [0, -1]}
            if camera_mode == 'follow':
                seg_kwargs.update(
                    back_offset=args.back_offset,
                    up_offset=args.up_offset,
                    look_offset=args.look_offset,
                    scale_frames=args.follow_scale_frames,
                )
                if args.smooth_window is not None:
                    seg_kwargs['smooth_window'] = args.smooth_window
            elif camera_mode == 'birdeye' and args.reveal_height_mult is not None:
                seg_kwargs['reveal_height_mult'] = args.reveal_height_mult
            cfg.camera.segments = [CameraSegment(**seg_kwargs)]

    _apply('camera.fov', 'fov')
    _apply('camera.transition', 'birdeye_transition')
    _apply('scene.downsample', 'downsample_factor')
    cfg.scene.keyframes_only_points = bool(getattr(args, 'keyframes_only_points', False))

    # Overlay
    _apply('overlay.camera_vis', 'camera_vis')
    if getattr(args, 'trail_color_ramp', None):
        cfg.overlay.trail_color_ramp = args.trail_color_ramp
    for attr in ('trail_line_width', 'trail_tail_len', 'head_num_frames',
                 'head_point_size', 'head_frustum_scale', 'head_frustum_line_width',
                 'head_texture_alpha'):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg.overlay, attr, val)
    if getattr(args, 'head_frustum_color', None):
        cfg.overlay.head_frustum_color = args.head_frustum_color
    cfg.overlay.frame_tag = bool(getattr(args, 'frame_tag', False))
    if getattr(args, 'frame_tag_position', None):
        cfg.overlay.frame_tag_position = args.frame_tag_position

    _rlog("Building scene...")
    scene = SceneBuilder(cfg, log=_rlog).load().preprocess().voxelize().build()
    camera_path = build_camera_path(cfg.camera, scene)
    overlays, overlay_specs = build_overlays(cfg, scene)

    _rlog(f"{len(scene.sorted_xyz):,} voxels, {scene.num_frames} frames, "
          f"{len(camera_path)} camera frames")
    OfflinePipeline(scene, camera_path, overlays, cfg,
                    overlay_specs=overlay_specs, log=_rlog).run()
    scene.destroy()
    return True


# =============================================================================
# Stage 10: Viewer (viser 3D)
# =============================================================================

def _launch_viewer(predictions: dict, images_cpu, args, image_folder: str) -> None:
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
# Pipeline helpers
# =============================================================================

def _check_gpu_memory(device: torch.device) -> None:
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
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= _BF16_MIN_CAPABILITY else torch.float16
    else:
        dtype = torch.float32
    _log.info("Inference dtype: %s", dtype)
    return dtype


def _prepare_model(model, dtype: torch.dtype) -> None:
    if dtype != torch.float32 and model.aggregator is not None:
        _log.info("Casting aggregator to %s (heads kept in fp32)", dtype)
        model.aggregator = model.aggregator.to(dtype=dtype)


def _auto_keyframe_interval(args, num_frames: int) -> None:
    if args.keyframe_interval is not None:
        return
    if args.flow_threshold > 0:
        args.keyframe_interval = 1  # flow mode handles its own keyframes
        return
    if args.mode == "streaming" and num_frames > _KEYFRAME_AUTO_THRESHOLD:
        args.keyframe_interval = (
            (num_frames + _KEYFRAME_AUTO_THRESHOLD - 1) // _KEYFRAME_AUTO_THRESHOLD)
        _log.info("Auto --keyframe_interval=%d (num_frames=%d > %d)",
                  args.keyframe_interval, num_frames, _KEYFRAME_AUTO_THRESHOLD)
    else:
        args.keyframe_interval = 1


def _export_preprocessed(export_dir: str | None, images: torch.Tensor) -> None:
    if not export_dir:
        return
    os.makedirs(export_dir, exist_ok=True)
    _log.info("Exporting %d preprocessed images to %s ...", images.shape[0], export_dir)
    for i in range(images.shape[0]):
        img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(export_dir, f"{i:06d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    _log.info("Exported to %s", export_dir)


def _resolve_output_paths(args, scene_name: str) -> dict:
    """Determine output file paths for a scene."""
    base = args.output_folder or "."
    os.makedirs(base, exist_ok=True)
    paths = {}

    if args.save_predictions:
        npz_base = args.save_predictions if args.save_predictions != "." else os.path.join(base, scene_name)
        paths['npz'] = npz_base if npz_base.endswith('.npz') else npz_base

    if args.render is not None:
        render_base = args.render if args.render != "." else os.path.join(base, f"{scene_name}{args.video_suffix}.mp4")
        paths['video'] = render_base if render_base.endswith('.mp4') else os.path.join(render_base, f"{scene_name}{args.video_suffix}.mp4")

    if args.save_glb:
        paths['glb'] = os.path.join(base, f"{scene_name}.glb")

    return paths


# =============================================================================
# Main pipeline
# =============================================================================

def _run_single_scene(args, scene_name: str, image_folder: str,
                      model, device: torch.device, dtype: torch.dtype) -> int:
    """Run the full pipeline for one scene.  Returns exit code (0 = success)."""
    t_scene_start = time.time()
    _log.info("=" * 60)
    _log.info("Scene: %s", scene_name)
    _log.info("Source: %s", image_folder)
    if getattr(args, 'lazy_images', False):
        _log.info("Mode: lazy images (memmap, O(window) RAM)")
    _log.info("=" * 60)

    # Resolve output paths
    outs = _resolve_output_paths(args, scene_name)

    # ── Load images ──
    mmap_path = None  # for cleanup
    t_load = time.time()
    t_prep = 0.0
    # Temporarily set image_folder so load_images() works
    saved_folder = args.image_folder
    args.image_folder = image_folder
    try:
        if getattr(args, 'lazy_images', False):
            images, resolved_folder, mmap_path = _load_images_lazy(image_folder, args)
        else:
            images, resolved_folder = load_images(args)
    finally:
        args.image_folder = saved_folder

    num_frames = images.shape[0]
    t_load_elapsed = time.time() - t_load
    _log.info("Loaded %d frames (%.1f s)", num_frames, t_load_elapsed)
    _export_preprocessed(args.export_preprocessed, images)

    # ── Validate ──
    validate_frame_count(num_frames, args.mode)

    # ── Prepare ──
    # Keep images on CPU; inference methods move per-window slices to GPU just-in-time.
    if device.type == "cuda" and not getattr(args, 'lazy_images', False):
        images = images.pin_memory() if not images.is_pinned() else images
    _log.info("Images on %s, shape %s", images.device, tuple(images.shape))
    _auto_keyframe_interval(args, num_frames)

    # ── Compile warmup ──
    _warmup_compile(model, images, args, dtype, num_frames)

    # ── Inference (with incremental NPZ saving if requested) ──
    npz_dir = None
    if outs.get('npz'):
        # Set up incremental per-window saving via callback
        _inc_state = _setup_incremental_npz(outs['npz'], num_frames, args)
        per_window_cb = lambda w_pred, start, end: _on_window_complete(
            w_pred, start, end, _inc_state, args
        )
    else:
        per_window_cb = None

    predictions = _run_inference(model, images, args, dtype, per_window_cb)

    # ── Post-process ──
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    images_for_post = predictions["images"]
    predictions, images_cpu = _postprocess(predictions, images_for_post)

    # ── Export ──
    # Attach images to predictions for NPZ only if --save_images is set.
    if getattr(args, 'save_images', False):
        imgs = images_cpu
        if imgs.ndim == 5 and imgs.shape[0] == 1:
            imgs = imgs[0]  # strip batch dim
        predictions["images"] = imgs

    if outs.get('npz'):
        npz_dir = _finalize_incremental_npz(predictions, _inc_state, args)
    if outs.get('glb'):
        _export_glb(predictions, outs['glb'], args)

    # ── Render ──
    if outs.get('video'):
        if npz_dir is None:
            # Need to save NPZ temporarily for the render pipeline
            tmp_npz = tempfile.mkdtemp(prefix="lingbot_render_")
            npz_dir = _save_predictions_npz(predictions, tmp_npz)
            _log.info("Rendering video to %s ...", outs['video'])
            _render_npz(npz_dir, outs['video'], args, artifact_name=scene_name)
            shutil.rmtree(tmp_npz, ignore_errors=True)
        else:
            _log.info("Rendering video to %s ...", outs['video'])
            _render_npz(npz_dir, outs['video'], args, artifact_name=scene_name)

    # ── Viewer ──
    if not args.headless and not outs.get('video') and not outs.get('npz'):
        _launch_viewer(predictions, images_cpu, args, resolved_folder)

    # ── Cleanup ──
    if mmap_path and os.path.exists(mmap_path):
        try:
            os.unlink(mmap_path)
            _log.debug("Cleaned up lazy-image memmap: %s", mmap_path)
        except OSError:
            pass

    _log.info("Scene '%s' complete (total %.1f s).", scene_name,
              time.time() - t_scene_start)
    return 0


def _run_load_predictions_mode(args) -> int:
    """Handle --load_predictions: render to video, visualize sky masks, or launch interactive viewer."""

    # ── Sky mask only ──
    if args.visualize_sky_mask_only:
        from lingbot_map.vis.sky_segmentation import load_or_create_sky_masks
        _log.info("Sky mask visualization only mode")
        for npz_path in args.load_predictions:
            predictions = _load_predictions_from_npz(npz_path)
            name = os.path.splitext(os.path.basename(npz_path.rstrip('/')))[0]
            sky_dir, sky_viz_dir = _resolve_sky_artifact_dirs(args, name)
            _log.info("Generating sky masks for %s...", name)
            load_or_create_sky_masks(
                image_folder=None, image_paths=None,
                images=predictions.get("images"),
                skyseg_model_path=args.skyseg_model_path,
                sky_mask_dir=sky_dir,
                sky_mask_visualization_dir=sky_viz_dir,
                num_frames=predictions.get("images", np.empty((0,))).shape[0],
            )
            _log.info("Sky masks saved.")
        return 0

    # ── Render mode ──
    if args.render is not None:
        args.use_per_scene_sky_dirs = len(args.load_predictions) > 1
        for npz_path in args.load_predictions:
            if not os.path.exists(npz_path):
                _log.error("NPZ path not found: %s", npz_path)
                continue

            # Check that images are present (render pipeline requires them)
            predictions = _load_predictions_from_npz(npz_path)
            if "images" not in predictions:
                _log.error(
                    "NPZ '%s' has no images. Re-save with --save_images, or provide "
                    "--image_folder to reload originals.", npz_path)
                continue

            name = os.path.splitext(os.path.basename(npz_path.rstrip('/')))[0]
            base = args.output_folder or "."
            os.makedirs(base, exist_ok=True)
            video_path = args.render if args.render != "." else os.path.join(
                base, f"{name}{args.video_suffix}.mp4")
            _log.info("Rendering %s → %s", npz_path, video_path)
            _render_npz(npz_path, video_path, args, artifact_name=name)
        return 0

    # ── Interactive viewer mode (--load_predictions without --render) ──
    # Load the first NPZ path and launch the viewer.
    npz_path = args.load_predictions[0]
    if not os.path.exists(npz_path):
        _log.error("NPZ path not found: %s", npz_path)
        return 1

    _log.info("Loading predictions for interactive viewer...")
    predictions = _load_predictions_from_npz(npz_path)

    # If images are missing from NPZ, reload from --image_folder
    if "images" not in predictions:
        if not args.image_folder:
            _log.error(
                "NPZ has no images and --image_folder not provided. "
                "Either re-save with --save_images, or pass --image_folder to reload originals.")
            return 1
        _log.info("Images not in NPZ — reloading from %s...", args.image_folder)
        paths = _list_image_paths(args.image_folder, args.image_extension)
        paths = _apply_image_filters(paths, args.first_k, args.last_k, args.stride)
        if not paths:
            _log.error("No images found in %s", args.image_folder)
            return 1
        # Only load as many frames as the NPZ contains
        S = predictions["depth"].shape[0] if "depth" in predictions else len(paths)
        if len(paths) > S:
            _log.info("NPZ has %d frames, limiting reload to first %d of %d images", S, S, len(paths))
            paths = paths[:S]
        # Load and preprocess to match depth dimensions
        reloaded = load_and_preprocess_images(
            paths, mode="crop", image_size=args.image_size, patch_size=args.patch_size,
        )
        predictions["images"] = reloaded.numpy()
        _log.info("Reloaded %d images, shape %s", len(paths), reloaded.shape)

    # Upcast float16/uint8 depth/depth_conf to float32 for the viewer/renderer
    for key in ('depth', 'depth_conf'):
        if key in predictions and predictions[key].dtype in (np.float16, np.uint8):
            predictions[key] = predictions[key].astype(np.float32)
    images_cpu = predictions.get("images")
    if isinstance(images_cpu, torch.Tensor):
        images_cpu = images_cpu.detach().cpu()

    # Ensure images are in (S, 3, H, W) float32 [0,1] format for viewer
    if isinstance(images_cpu, np.ndarray):
        images_cpu = torch.from_numpy(images_cpu.astype(np.float32))
    if images_cpu.max() > 1.0:
        images_cpu = images_cpu / 255.0

    _launch_viewer(predictions, images_cpu, args, args.image_folder or "")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Track user-supplied args (for YAML config override logic)
    _arg_defaults = {a.dest: a.default for a in parser._actions if a.dest not in ('help',)}
    args._user_supplied = {
        k for k, v in vars(args).items()
        if k != '_user_supplied' and v != _arg_defaults.get(k, None)
    }

    # ── Determine mode ──
    # Mode 1: Load predictions (skip inference, render or visualize)
    if args.load_predictions is not None:
        return _run_load_predictions_mode(args)

    # Mode 2: Batch (--input_folder)
    if args.input_folder is not None:
        if not args.model_path:
            parser.error("--model_path is required for inference (batch mode)")
        scenes = discover_scenes(args)
        if args.scenes:
            scenes = [s for s in scenes if s[0] in args.scenes]
        if args.exclude_scenes:
            scenes = [s for s in scenes if s[0] not in args.exclude_scenes]
        if not scenes:
            _log.info("No scenes found.")
            return 0

        _log.info("Found %d scene(s):", len(scenes))
        for idx, (name, folder, count) in enumerate(scenes, 1):
            _log.info("  %d. %s (%d images) → %s", idx, name, count, folder)

        if args.dry_run:
            return 0

        args.use_per_scene_sky_dirs = len(scenes) > 1
        args.headless = True  # batch is always headless

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _check_gpu_memory(device)
        dtype = _select_dtype()

        model = _load_model(args, device)
        _prepare_model(model, dtype)

        results = []
        t_total = time.time()
        for scene_name, image_folder, _ in tqdm(scenes, desc="Processing scenes"):
            try:
                _run_single_scene(args, scene_name, image_folder, model, device, dtype)
                results.append({"scene": scene_name, "success": True})
            except Exception as exc:
                _log.error("Scene '%s' failed: %s", scene_name, exc)
                results.append({"scene": scene_name, "success": False, "error": str(exc)})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.time() - t_total
        _log.info("Batch complete: %d/%d scenes in %.1f s",
                  sum(1 for r in results if r['success']), len(results), elapsed)
        if args.output_folder:
            with open(os.path.join(args.output_folder, "batch_results.json"), "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
        return 0 if all(r['success'] for r in results) else 1

    # Mode 3: Single scene (--image_folder or --video_path)
    if not args.image_folder and not args.video_path:
        parser.error("Provide --image_folder, --video_path, --input_folder, or --load_predictions")

    if not args.model_path:
        parser.error("--model_path is required for inference")

    # Auto-headless when exporting
    if args.render is not None or args.save_predictions is not None:
        args.headless = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _check_gpu_memory(device)
    dtype = _select_dtype()

    model = _load_model(args, device)
    _prepare_model(model, dtype)

    scene_name = os.path.basename(
        args.image_folder or os.path.splitext(os.path.basename(args.video_path))[0]
    )
    return _run_single_scene(args, scene_name,
                             args.image_folder or args.video_path,
                             model, device, dtype)


if __name__ == "__main__":
    sys.exit(main())
