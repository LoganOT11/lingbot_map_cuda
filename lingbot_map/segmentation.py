"""Segmentation pipeline for LingBot-MAP.

Provides frame-by-frame segmentation and 3D mask lifting — decoupled
from both the inference processor and the web layer.

Usage::

    from lingbot_map.segmentation import SegmentationPipeline, lift_to_3d

    seg = SegmentationPipeline(model="skyseg")  # or "sam2.1_hiera_tiny"
    masks_dir = "/data/scenes/001/masks/"

    for idx, frame in enumerate(frames):
        mask = seg.segment_frame(frame)
        seg.save_mask(mask, idx, masks_dir)

    # After inference is complete:
    labeled = lift_to_3d(masks_dir, predictions_dir, target_classes=[1, 3])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# SegmentationPipeline
# ────────────────────────────────────────────────────────────────────────────


class SegmentationPipeline:
    """Frame-by-frame 2D segmentation with optional 3D lifting.

    Supports two backends:

    ``"skyseg"``
        The existing sky-segmentation ONNX model (skyseg.onnx, 176 MB).
        Produces soft sky/non-sky masks per frame.  Requires ``onnxruntime``.

    ``"sam2.1_hiera_tiny"`` (optional)
        Meta's Segment Anything Model 2.1, automatic grid-prompt mode.
        Requires ``segment_anything`` package.  Falls back gracefully if
        not installed.

    All segmenters yield uint8 masks of the same spatial size as the input
    frame.  Masks can be saved to disk and later lifted to 3D via
    :func:`lift_to_3d`.
    """

    def __init__(
        self,
        model: str = "skyseg",
        *,
        device: str = "cpu",
        skyseg_model_path: str = "models/skyseg.onnx",
        sam_config: str | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._skyseg_path = skyseg_model_path
        self._session = None  # ONNX inference session

        if model == "skyseg":
            self._init_skyseg()
        elif model.startswith("sam"):
            self._init_sam(sam_config)
        else:
            raise ValueError(
                f"Unknown segmentation model '{model}'. "
                f"Choose 'skyseg' or 'sam2.1_hiera_tiny'."
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Segment a single BGR or RGB frame.

        Args:
            frame: ``[H, W, 3]`` uint8 array, BGR (cv2 format) or RGB.

        Returns:
            ``[H, W]`` uint8 mask.  For skyseg: 0=sky, 1=non-sky.
            For SAM: integer class IDs per pixel.
        """
        if self._model_name == "skyseg":
            return self._segment_skyseg(frame)
        return self._segment_sam(frame)

    def segment_batch(self, frames: np.ndarray) -> np.ndarray:
        """Segment multiple frames at once.

        Args:
            frames: ``[N, H, W, 3]`` uint8 array.

        Returns:
            ``[N, H, W]`` uint8 masks.
        """
        masks = []
        for i in range(frames.shape[0]):
            masks.append(self.segment_frame(frames[i]))
        return np.stack(masks, axis=0)

    @staticmethod
    def save_mask(mask: np.ndarray, frame_idx: int, output_dir: str | Path) -> str:
        """Save a single-frame mask as a PNG file.

        Returns the path to the saved file.
        """
        import cv2

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(path), mask)
        return str(path)

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── Sky segmentation backend ────────────────────────────────────────────

    def _init_skyseg(self) -> None:
        """Initialise the ONNX sky segmentation session."""
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "onnxruntime is required for skyseg. "
                "Install with: pip install onnxruntime"
            )

        # Download model if not present
        if not os.path.exists(self._skyseg_path):
            self._download_skyseg_model()

        self._session = onnxruntime.InferenceSession(
            self._skyseg_path,
            providers=["CPUExecutionProvider"],
        )

    def _segment_skyseg(self, frame: np.ndarray) -> np.ndarray:
        """Run skyseg ONNX inference on a single frame.

        Returns a uint8 mask where 0=sky, 255=non-sky.
        """
        import cv2

        input_size = (320, 320)

        # Prepare input with ImageNet normalisation
        resized = cv2.resize(frame, input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if frame.shape[-1] == 3 else resized
        x = rgb.astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x / 255.0 - mean) / std
        x = x.transpose(2, 0, 1)
        x = x.reshape(1, 3, input_size[1], input_size[0]).astype("float32")

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        result = self._session.run([output_name], {input_name: x})[0]
        result = result.squeeze()

        # Normalise to [0, 255]
        result = (result - result.min()) / max(result.max() - result.min(), 1e-8)
        result = (result * 255).astype(np.uint8)

        # Resize back to original frame size
        h, w = frame.shape[:2]
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        return result

    @staticmethod
    def _download_skyseg_model() -> None:
        """Download skyseg.onnx from Hugging Face Hub."""
        import requests

        url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        print(f"Downloading skyseg.onnx from {url} ...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        os.makedirs("models", exist_ok=True)
        with open("models/skyseg.onnx", "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Downloaded skyseg.onnx to models/skyseg.onnx")

    # ── SAM backend (optional, graceful fallback) ───────────────────────────

    def _init_sam(self, config: str | None) -> None:
        """Try to load SAM 2.1.  If not installed, store a clear error
        message that will be raised on first use."""
        self._sam_available = False
        self._sam_error = None
        try:
            # SAM 2.1 is not a declared dependency — try to import
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            self._sam_available = True
            self._sam_predictor = None  # lazy init
        except ImportError as e:
            self._sam_error = (
                f"SAM 2.1 is not installed ({e}). "
                f"Install with: pip install segment-anything"
            )

    def _segment_sam(self, frame: np.ndarray) -> np.ndarray:
        """Run SAM auto-segmentation.  Raises if SAM is not available."""
        if not self._sam_available:
            raise RuntimeError(self._sam_error or "SAM not available")

        # Lazy-init the SAM predictor
        if self._sam_predictor is None:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            # Use the tiny Hiera model for speed
            model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            model.to(self._device)
            self._sam_predictor = SamAutomaticMaskGenerator(
                model,
                points_per_side=16,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
            )

        masks = self._sam_predictor.generate(frame)
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Combine masks into a single label image
        # Each mask gets a unique label (1, 2, 3, ...)
        label_image = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i, mask_data in enumerate(masks, start=1):
            label_image[mask_data["segmentation"]] = i

        return label_image


# ────────────────────────────────────────────────────────────────────────────
# 2D → 3D mask lifting
# ────────────────────────────────────────────────────────────────────────────


def lift_to_3d(
    masks_dir: str | Path,
    predictions_dir: str | Path,
    *,
    target_classes: list[int] | None = None,
    confidence_threshold: float = 0.5,
    voxel_size: float = 0.02,
) -> dict[int, np.ndarray]:
    """Lift 2D segmentation masks to 3D labeled point cloud.

    Reads per-frame masks (PNG) and per-frame predictions (NPZ), unprojects
    depth to world coordinates, and accumulates labeled points per class.

    Args:
        masks_dir: Directory containing ``frame_NNNNNN.png`` mask files.
        predictions_dir: Directory containing ``frame_NNNNNN.npz`` files
            with keys ``depth``, ``extrinsic``, ``intrinsic``.
        target_classes: If provided, only lift these class IDs.
        confidence_threshold: Minimum depth confidence to include a point.
        voxel_size: Deduplication voxel size in scene units (meters).
            Points within the same voxel are averaged.  Set to 0 to disable.

    Returns:
        ``{class_id: [N, 3] ndarray}`` mapping each class to its world-space
        XYZ points.
    """
    import glob as _glob

    from lingbot_map.utils.geometry import depth_to_world_coords_points

    masks_dir = Path(masks_dir)
    predictions_dir = Path(predictions_dir)

    mask_files = sorted(masks_dir.glob("frame_*.png"))
    pred_files = sorted(predictions_dir.glob("frame_*.npz"))

    if not mask_files:
        raise FileNotFoundError(f"No frame_*.png files in {masks_dir}")
    if not pred_files:
        raise FileNotFoundError(f"No frame_*.npz files in {predictions_dir}")

    # Accumulate per class
    from collections import defaultdict

    accumulator: dict[int, list[np.ndarray]] = defaultdict(list)

    for mask_path, pred_path in zip(mask_files, pred_files):
        mask = _read_mask(mask_path)
        data = np.load(pred_path)

        if "depth" not in data or "extrinsic" not in data:
            continue

        depth = data["depth"].squeeze()  # [H, W]
        extrinsic = data["extrinsic"]    # [3, 4]
        intrinsic = data["intrinsic"]    # [3, 3]
        conf = data.get("depth_conf", np.ones_like(depth))

        # Unproject depth → world
        world_xyz, _, valid = depth_to_world_coords_points(
            depth, extrinsic, intrinsic
        )

        # Resize mask to match depth if needed
        if mask.shape[:2] != depth.shape[:2]:
            import cv2
            mask = cv2.resize(
                mask, (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Filter by confidence
        conf_mask = (conf > confidence_threshold) & valid

        # Assign labels
        for class_id in np.unique(mask):
            if target_classes is not None and class_id not in target_classes:
                continue
            class_mask = (mask == class_id) & conf_mask
            if class_mask.sum() == 0:
                continue
            accumulator[int(class_id)].append(world_xyz[class_mask])

    # Concatenate per class and optionally deduplicate
    result: dict[int, np.ndarray] = {}
    for class_id, point_list in accumulator.items():
        points = np.concatenate(point_list, axis=0)
        if voxel_size > 0 and len(points) > 0:
            points = _voxel_deduplicate(points, voxel_size)
        result[class_id] = points

    return result


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask from PNG.  Returns uint8 [H, W]."""
    import cv2
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask


def _voxel_deduplicate(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deduplicate 3D points by averaging within voxel cells."""
    if len(points) == 0:
        return points

    voxel_indices = (points / voxel_size).astype(np.int64)
    # Use a dict to accumulate points per voxel
    voxel_map: dict[tuple, list] = {}
    for i, vi in enumerate(voxel_indices):
        key = tuple(vi)
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(points[i])

    # Average within each voxel
    deduped = np.array([np.mean(pts, axis=0) for pts in voxel_map.values()])
    return deduped


# ────────────────────────────────────────────────────────────────────────────
# Semantic GLB export
# ────────────────────────────────────────────────────────────────────────────


def export_semantic_glb(
    labeled_points: dict[int, np.ndarray],
    output_path: str | Path,
    *,
    class_names: dict[int, str] | None = None,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    downsample: int = 4,
) -> str:
    """Export labeled 3D points as a GLB file with per-class colors.

    Args:
        labeled_points: ``{class_id: [N, 3] ndarray}`` as returned by
            :func:`lift_to_3d`.
        output_path: Where to write the ``.glb`` file.
        class_names: Optional human-readable names per class.
        class_colors: Optional RGB colors ``(0-255, 0-255, 0-255)`` per
            class.  If not provided, a default colormap is used.
        downsample: Stride for spatial subsampling (1 = all points).

    Returns:
        The *output_path* as a string.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for GLB export. "
            "Install with: pip install trimesh"
        )

    # Default colormap
    if class_colors is None:
        import matplotlib.cm as cm
        unique_ids = sorted(labeled_points.keys())
        colormap = cm.get_cmap("tab20")
        class_colors = {}
        for i, cid in enumerate(unique_ids):
            rgba = colormap(i)
            class_colors[cid] = tuple(int(255 * v) for v in rgba[:3])

    scene = trimesh.Scene()

    for class_id, points in labeled_points.items():
        if len(points) < 10:
            continue

        # Downsample
        pts = points[::downsample]
        if len(pts) == 0:
            continue

        color = class_colors.get(class_id, (255, 255, 255))
        colors = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))

        name = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
        pc = trimesh.PointCloud(vertices=pts, colors=colors)

        # Store class info as metadata
        pc.metadata["class_id"] = class_id
        pc.metadata["class_name"] = name

        scene.add_geometry(pc, node_name=name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    return str(output_path)
