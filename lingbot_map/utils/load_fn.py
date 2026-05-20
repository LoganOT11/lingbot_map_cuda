# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image, ImageOps
from torchvision import transforms as TF
from tqdm.auto import tqdm
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)
        # Apply EXIF orientation so portrait photos (e.g. Orientation=6/8 from
        # most cameras and phones) are rotated to their displayed orientation
        # before any resize/crop. No-op when EXIF is absent.
        img = ImageOps.exif_transpose(img)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, fx=None, fy=None, cx=None, cy=None, mode="crop", image_size=512, patch_size=16):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

        

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    target_size = image_size
    to_tensor = TF.ToTensor()

    def _load_one(idx_path):
        i, image_path = idx_path
        img = Image.open(image_path)
        # Honor EXIF orientation (Sony / iPhone portrait shots store landscape
        # pixels + Orientation=6/8); without this the resize/crop below sees
        # raw landscape pixels and outputs a landscape frame.
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        fx_val = fy_val = cx_val = cy_val = None
        if fx is not None:
            fx_val = fx[i] * width
            fy_val = fy[i] * height
            cx_val = cx[i] * width
            cy_val = cy[i] * height

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / patch_size) * patch_size
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / patch_size) * patch_size
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        if fx is not None:
            fx_val = fx_val * new_width / width
            fy_val = fy_val * new_height / height
            cx_val = img.shape[2] / 2
            cy_val = img.shape[1] / 2

        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        return i, img, (fx_val, fy_val, cx_val, cy_val)

    # Parallel load with progress bar
    num_workers = min(16, len(image_path_list))
    results = [None] * len(image_path_list)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = pool.map(_load_one, enumerate(image_path_list))
        for i, img, calib in tqdm(futures, total=len(image_path_list), desc="Loading images"):
            results[i] = img
            if fx is not None:
                fx[i], fy[i], cx[i], cy[i] = calib

    images = results
    shapes = set((img.shape[1], img.shape[2]) for img in images)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    if fx is not None:
        return images, fx, fy, cx, cy
    return images


def load_and_preprocess_video_stream(
    video_source: str | bytes,
    *,
    fps: int = 5,
    image_size: int = 518,
    patch_size: int = 14,
    max_frames: int | None = None,
) -> "Iterator[tuple[int, torch.Tensor]]":
    """Stream-decode video frames one at a time, yielding preprocessed tensors.

    Unlike :func:`load_and_preprocess_images` which loads everything into a
    single tensor, this generator never holds more than one frame in memory.
    Ideal for long videos and web-upload processing.

    Args:
        video_source: File path (``str``) or in-memory bytes.
        fps: Target extraction frame rate.
        image_size: Width in pixels for the canonical preprocessed frame.
            Height is derived from the video's aspect ratio and rounded
            down to a multiple of *patch_size*.
        patch_size: ViT patch size for height alignment (default 14).
        max_frames: Stop after this many frames (``None`` = all).

    Yields:
        ``(global_frame_index, tensor)`` tuples where *tensor* has shape
        ``[1, 3, H, W]``, dtype float32, values in [0, 1], on CPU.

    Example:
        >>> for idx, frame in load_and_preprocess_video_stream("video.mp4", fps=5):
        ...     print(idx, frame.shape)
        0 torch.Size([1, 3, 294, 518])
        1 torch.Size([1, 3, 294, 518])
    """
    import io
    import os
    import tempfile

    import cv2

    # Accept both file paths and in-memory bytes
    if isinstance(video_source, bytes):
        tmpdir = tempfile.mkdtemp(prefix="lingbot_vstream_")
        tmpfile = os.path.join(tmpdir, "upload.mp4")
        try:
            with open(tmpfile, "wb") as f:
                f.write(video_source)
            yield from _stream_from_capture(
                tmpfile, fps, image_size, patch_size, max_frames
            )
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        yield from _stream_from_capture(
            video_source, fps, image_size, patch_size, max_frames
        )


def _stream_from_capture(
    video_path: str,
    fps: int,
    image_size: int,
    patch_size: int,
    max_frames: int | None,
) -> "Iterator[tuple[int, torch.Tensor]]":
    """Core streaming loop — shared by file and bytes paths."""
    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, round(src_fps / fps)) if fps > 0 else 1

        # Determine output resolution from the first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Video has no readable frames")

        h, w = first_frame.shape[:2]
        new_width = image_size
        new_height = round(h * (new_width / w) / patch_size) * patch_size

        yielded = 0
        frame_idx = 0

        # Preprocess the first frame we already read
        tensor = _preprocess_single_frame(
            first_frame, new_width, new_height, image_size
        )
        yield (yielded, tensor)
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            return

        # Process remaining frames
        while True:
            if frame_idx % interval == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                tensor = _preprocess_single_frame(
                    frame, new_width, new_height, image_size
                )
                yield (yielded, tensor)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            else:
                if not cap.grab():
                    break
            frame_idx += 1
    finally:
        cap.release()


def _preprocess_single_frame(
    bgr_frame: "np.ndarray",
    new_width: int,
    new_height: int,
    image_size: int,
) -> "torch.Tensor":
    """Resize + crop a single BGR frame to [1, 3, H, W] in [0, 1]."""
    import cv2
    import numpy as np
    import torch

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y : start_y + image_size, :]

    return tensor.unsqueeze(0)  # [1, 3, H, W]
