# LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction

**LingBot-Map** is a feed-forward 3D foundation model for **streaming 3D
reconstruction**. It processes images or video frame-by-frame with a paged KV
cache attention mechanism, producing camera poses, depth maps, and 3D point
clouds in real time.

- **Architecture**: Geometric Context Transformer (GCT) — unifies coordinate
  grounding, dense geometric cues, and long-range drift correction via anchor
  context, pose-reference window, and trajectory memory.
- **Performance**: ~20 FPS on 518×378 resolution, stable over 10,000+ frames.
- **Paper**: [arXiv:2604.14141](https://arxiv.org/abs/2604.14141)
- **License**: Apache 2.0

---

## ⚙️ Installation

**1. Create conda environment**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. Install PyTorch (CUDA 12.8)**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

**3. Install lingbot-map**

```bash
pip install -e .
```

**4. Install FlashInfer (recommended)**

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

If FlashInfer is not installed, the model falls back to SDPA (PyTorch native
attention) via `--use_sdpa`.

**5. Visualization dependencies (optional)**

```bash
pip install -e ".[vis]"
```

For sky masking support:

```bash
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # GPU (faster for large image sets)
```

---

## 📦 Model Download

| Model | Description |
|:---|:---|
| **lingbot-map-long** (`lingbot-map-long.pt`, 4.63 GB) | Best for long sequences / large-scale scenes (**recommended**) |
| lingbot-map | Balanced checkpoint for short + long sequences |
| lingbot-map-stage1 | Stage-1 training; can be loaded into VGGT for bidirectional inference |

Models are available on [HuggingFace](https://huggingface.co/robbyant/lingbot-map)
and [ModelScope](https://www.modelscope.cn/models/Robbyant/lingbot-map).

---

## 🚀 Quick Start

```bash
python demo.py --model_path lingbot-map-long.pt \
    --image_folder example/courthouse --mask_sky
```

This launches an interactive [viser](https://github.com/nerfstudio-project/viser)
3D viewer at `http://localhost:8080`. The viewer loops through all frames — press
Ctrl+C to stop, or use the "Playing" checkbox to pause/resume.

### Example Scenes

Four pre-packaged scenes in `example/`:

| Scene | Frames | Type | Notes |
|:---|:---|:---|:---|
| `example/courthouse` | 286 | Outdoor building | Use `--mask_sky` |
| `example/university` | — | Outdoor campus | Use `--mask_sky` |
| `example/loop` | — | Loop closure | — |
| `example/oxford` | — | Outdoor, large scale | Use `--mask_sky` |

### Working Commands

**Quick test (10 frames):**
```bash
python demo.py --model_path lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mask_sky --first_k 10
```
Model loads in ~10s, inference ~3s, GPU peaks at ~4.6 GB.

**Full run (286 frames, windowed mode):**
```bash
python demo.py --model_path lingbot-map-long.pt \
    --image_folder example/courthouse \
    --use_sdpa --mode windowed \
    --window_size 16 --overlap_size 4 --num_scale_frames 4 \
    --offload_to_cpu --mask_sky
```
Processes all 286 frames in ~1 minute. Peak GPU ~5.6 GB.

**Long sequences (>3000 frames):**
```bash
python demo.py --model_path lingbot-map-long.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128 --overlap_keyframes 16 --keyframe_interval 2
```

**Fast inference (reduced quality):**
```bash
python demo.py --model_path lingbot-map-long.pt \
    --image_folder example/courthouse --mask_sky --camera_num_iterations 1
```

**FlashInfer (requires ≥12 GB GPU):**
```bash
python demo.py --model_path lingbot-map-long.pt \
    --image_folder example/courthouse --mask_sky
```

---

## 🎮 Demo Flags Reference

### Inference Mode

| Flag | Purpose |
|:---|:---|
| `--mode streaming` | Default. Frame-by-frame with KV cache. Good for ≤320 frames. |
| `--mode windowed` | Overlapping windows + KV cache reset per window. Required for long sequences. |
| `--window_size N` | Keyframes per window in windowed mode. Controls KV cache memory. |
| `--overlap_size N` | Overlap between windows in actual frames. |
| `--overlap_keyframes N` | Overlap in keyframes (recommended when `keyframe_interval > 1`). |

### Memory Management

| Flag | Default | Purpose |
|:---|:---|:---|
| `--use_sdpa` | off | Use PyTorch SDPA instead of FlashInfer. **Required on ≤12 GB GPUs.** |
| `--offload_to_cpu` | **on** | Offload per-frame predictions to CPU during inference. |
| `--no-offload_to_cpu` | — | Keep predictions on GPU (needs more VRAM). |
| `--num_scale_frames N` | 8 | Bidirectional scale frames. Reduce to 2–4 to cut memory. |
| `--kv_cache_sliding_window N` | 64 | Max frames in KV cache window. Reduce to 32 to cut memory. |
| `--keyframe_interval N` | auto | Only cache every Nth frame. Reduces KV cache growth. |

### Performance

| Flag | Default | Purpose |
|:---|:---|:---|
| `--camera_num_iterations 1` | 4 | Faster inference (skips 3 refinement passes). |
| `--compile` | off | torch.compile hot modules. ~5 FPS faster. Adds 30–60s warmup. |

### Input

| Flag | Purpose |
|:---|:---|
| `--image_folder PATH` | Folder of images (`.jpg`, `.png`, `.JPG`) |
| `--video_path PATH` | Video file (extracts frames at `--fps` rate) |
| `--first_k N` | Only load first N frames (quick tests) |
| `--stride N` | Take every Nth frame |
| `--fps N` | FPS when extracting from video (default 10) |
| `--rotate_clockwise_90` | Rotate images 90° CW before preprocessing |
| `--export_preprocessed PATH` | Export preprocessed images for inspection |

### Visualization

| Flag | Default | Purpose |
|:---|:---|:---|
| `--port` | 8080 | Viser viewer port |
| `--conf_threshold` | 1.5 | Min confidence for point display |
| `--downsample_factor` | 10 | Spatial downsampling of point cloud |
| `--point_size` | 0.00001 | Point size in viewer |
| `--mask_sky` | off | Filter sky points from point cloud (outdoor scenes) |
| `--sky_mask_dir PATH` | auto | Cache directory for sky masks |
| `--sky_mask_visualization_dir PATH` | — | Save side-by-side mask visualizations |

---

## 🔑 Keyframe Interval

The KV cache stores at most 320 frames by default (the training RoPE range). When
`--keyframe_interval` is not set:
- `num_frames ≤ 320` → every frame cached (`interval = 1`)
- `num_frames > 320` → auto-selected `interval = ceil(num_frames / 320)`

For sequences longer than ~3000 frames, switch to `--mode windowed`.

---

## 🎥 Offline Rendering

For sequences too long for the interactive viewer, use `demo_render/batch_demo.py`
for headless point-cloud flythrough MP4 generation. Supports multiple camera modes
(follow, birdeye, static, pivot) configured via YAML presets in `demo_render/config/`.

Requires additional dependencies: open3d, kaolin, ffmpeg, CUDA extensions. See the
original README or source comments for setup instructions.

---

## 📖 Citation

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## ✨ Acknowledgments

This work builds upon several excellent open-source projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer)
