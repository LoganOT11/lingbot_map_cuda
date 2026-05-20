# Visualization, Gaussian Splatting & Segmentation for LingBot-MAP: Complete Production Analysis

> **End goal:** Take uploaded video → produce 3D Gaussian Splatting scene + semantically segmented point cloud → serve in a web viewer
> **Date:** 2026-05-20

---

## Executive Summary

**Yes, Gaussian Splatting from LingBot-MAP output is not only possible — it's an ideal input.** The model provides exactly what 3DGS needs (posed RGB images + dense depth for initialization) with better quality than the typical SfM→random-init pipeline. Segmentation can be pipelined frame-by-frame as frames arrive, with zero additional GPU memory pressure on the inference server if offloaded to a separate worker.

The combined pipeline (inference → segmentation → GS training → serve) is GPU-intensive but well-structured. The key insight: each stage has different GPU requirements and can run on separate machines or be time-sliced on one beefy GPU.

---

## 1. Current Visualization Capabilities

### 1.1 What Exists in the Repo

| Component | File | What it does |
|---|---|---|
| **Viser interactive viewer** | `lingbot_map/vis/point_cloud_viewer.py` | Python-side 3D viewer with camera frustums, frame animation, confidence filtering, sky masking. Runs its own HTTP server on configurable port. |
| **Viser thin wrapper** | `lingbot_map/vis/viser_wrapper.py` | Simplified version of above — just renders points + cameras, no GUI controls. |
| **GLB exporter** | `lingbot_map/vis/glb_export.py` | Converts predictions → `.glb` file with colored point cloud + camera pyramids. Uses trimesh. Supports confidence thresholding, sky masking, frame selection. |
| **Open3D offline renderer** | `apps/rgbd_render/renderer.py` | Headless Open3D rendering with octree LOD, Eye-Dome Lighting, point size control. Outputs rendered frames → encoded to MP4. |
| **Sky segmentation** | `lingbot_map/vis/sky_segmentation.py` | ONNX model (`skyseg.onnx`, 176 MB) runs per-frame. Produces soft sky/non-sky masks cached to disk. Used to filter sky points from point clouds. |
| **Camera path visualization** | `apps/rgbd_render/overlay.py` | Frustum wireframes, textured frustums, trajectory trails, frame tags. Rendered as Open3D overlays. |
| **Batch rendering pipeline** | `apps/rgbd_render/pipeline/` | Full offline pipeline: load NPZ → preprocess (confidence, sky mask, voxelize) → build scene → render frames → encode video. Supports parallel rendering via SharedMemory. |
| **NPZ → GLB converter** | `apps/viewer/npz_to_glb.py` | Standalone script to convert saved predictions to GLB. |

### 1.2 What's Missing for a Web Production Viewer

| Missing | Importance | Why |
|---|---|---|
| **Browser-based 3D viewer** | 🔴 Critical | Viser is Python-side — can't embed in a web app. Need Three.js + GLB or potree/deck.gl. |
| **Progressive loading** | 🟠 High | Current GLB is monolithic. For large scenes, need tiled/streaming formats (3D Tiles, potree octree). |
| **Interactive filtering** | 🟡 Medium | Confidence threshold, point size, frame range — must be adjustable in-browser without re-export. |
| **Camera path animation** | 🟡 Medium | Viser can animate cameras frame-by-frame. Need equivalent in Three.js. |
| **Measurements/annotations** | 🟢 Low | Distance measurements, point picking — nice-to-have for some use cases. |
| **Multi-resolution LOD** | 🟠 High | Current GLB is single-resolution. Large scenes need LOD for performance. |

---

## 2. Gaussian Splatting: Complete Feasibility Analysis

### 2.1 Why LingBot-MAP is Ideal for 3DGS

3D Gaussian Splatting (Kerbl et al., 2023) requires:

| 3DGS Input | LingBot-MAP Provides | Quality |
|---|---|---|
| **Posed RGB images** | ✅ `images` (preprocessed) or original images with `extrinsic`/`intrinsic` | Same as input |
| **Camera extrinsics** (world→camera or camera→world) | ✅ `extrinsic` [S, 3, 4] — world→camera format. Invert to camera→world for 3DGS. | Dense, per-frame, metric |
| **Camera intrinsics** | ✅ `intrinsic` [S, 3, 3] — fx, fy, cx, cy | Consistent across frames (fixed focal from FoV head) |
| **Initial point cloud** (SfM sparse) | ✅ **Dense depth maps** → unproject to dense world points. Far better than SfM sparse (hundreds of points) for initialization. | 100K+ points per frame |

**Comparison with typical 3DGS pipeline:**

```
Typical:    Images → COLMAP (SfM) → sparse points (100s) → random init → train 3DGS
                                                                              ↓
                                                               30,000 iterations, ~20 min

Ours:       Images → LingBot-MAP → dense depth + poses → unproject to dense points
                         ↓                                              ↓
               Initialize Gaussians at world points + RGB from images
                         ↓
               Train 3DGS: 7,000–15,000 iterations, ~5–10 min
```

**Why ours is faster:** Dense initialization means Gaussians start at approximately correct positions and colors. The optimizer only needs to refine shapes/opacities, not discover geometry from scratch. Expect **2-4× faster convergence**.

### 2.2 Gaussian Initialization from Depth

For each frame `i` with valid depth:

```python
# Unproject each pixel with valid depth
for frame_idx in range(S):
    depth = predictions["depth"][frame_idx]        # [H, W, 1]
    extrinsic = predictions["extrinsic"][frame_idx] # [3, 4] world→camera
    intrinsic = predictions["intrinsic"][frame_idx] # [3, 3]
    
    # Invert extrinsic: world→camera → camera→world
    c2w = invert_se3(extrinsic)  # [4, 4]
    
    # Unproject depth → world XYZ
    world_xyz = unproject(depth, extrinsic, intrinsic)  # [H, W, 3]
    
    # RGB from input image
    rgb = images[frame_idx]  # [H, W, 3] in [0,1]
    
    # Confidence filter
    conf = depth_conf[frame_idx]  # [H, W]
    mask = conf > conf_threshold
    
    # Initialize Gaussians at world positions with image colors
    for (x, y, z), (r, g, b) in zip(world_xyz[mask], rgb[mask]):
        gaussians.append(Gaussian(
            xyz=[x, y, z],
            rgb=[r, g, b],
            scale=initial_scale,      # ~distance to nearest neighbor
            opacity=conf[y, x] * 0.5, # confidence-scaled
            rotation=[1, 0, 0, 0],    # identity quaternion
        ))
```

### 2.3 Multi-View Deduplication

Since the same 3D point appears in multiple frames, initialize each Gaussian only once:

```python
# Voxel-grid-based deduplication
voxel_grid = {}  # key: (vx, vy, vz) → best Gaussian

for frame_idx in range(S):
    for each valid pixel (u, v) with depth d:
        world_xyz = unproject_pixel(u, v, d, ext[frame_idx], intr[frame_idx])
        voxel_key = tuple((world_xyz / voxel_size).astype(int))
        
        if voxel_key not in voxel_grid or conf[u,v] > voxel_grid[voxel_key].opacity:
            voxel_grid[voxel_key] = Gaussian(
                xyz=world_xyz,
                rgb=images[frame_idx, v, u],
                opacity=conf[u, v],
            )

gaussians = list(voxel_grid.values())  # ~1-5M Gaussians for a typical scene
```

### 2.4 Memory Budget for 3DGS Training

**Per-Gaussian storage (training):**

| Property | Elements | Bytes (float32) |
|---|---|---|
| xyz (position) | 3 | 12 |
| rgb (color, spherical harmonics DC) | 3 | 12 |
| scale (3D covariance) | 3 | 12 |
| rotation (quaternion) | 4 | 16 |
| opacity | 1 | 4 |
| **Total per Gaussian** | | **56 bytes** |

Plus SH coefficients (up to 48 per Gaussian for degree 3), optimizer states (momentum buffers ×4), and rendering buffers.

| Scene size | Gaussians | GS parameters | Optimizer states | Rendering buffers | **Total GPU** |
|---|---|---|---|---|---|
| Small room | 500K | 28 MB | 112 MB | ~500 MB | **~1 GB** |
| Medium building | 2M | 112 MB | 448 MB | ~2 GB | **~3 GB** |
| Large outdoor | 5M | 280 MB | 1.1 GB | ~4 GB | **~6 GB** |
| Full courthouse (300 frames) | 8M | 448 MB | 1.8 GB | ~6 GB | **~9 GB** |

**Training images on GPU:** Full-resolution images for loss computation. For 300 frames at 518×294: 300 × 3 × 518 × 294 × 4 bytes = ~550 MB.

**Total peak GPU for 3DGS training:** 3–12 GB depending on scene size. Fits on a 16-24 GB GPU for most scenes.

### 2.5 Training Time Estimates

| Scene | Gaussians | Iterations | RTX 4090 (24 GB) | A10G (24 GB) | RTX 4070 (12 GB) |
|---|---|---|---|---|---|
| Small (50 frames, room) | 1M | 7,000 | ~4 min | ~6 min | ~8 min |
| Medium (150 frames, building) | 3M | 15,000 | ~12 min | ~18 min | ⚠️ OOM risk |
| Large (300 frames, outdoor) | 5M | 20,000 | ~22 min | ~30 min | ❌ OOM |
| Full (300 frames, dense init) | 8M | 30,000 | ~35 min | ~45 min | ❌ OOM |

**With dense depth initialization:** Expect 2-4× fewer iterations for same quality → cut times by 50-75%.

### 2.6 GS Inference (Rendering)

**After training, the GS model is a `.ply` file + viewer:**

| Viewer | Gaussians supported | Memory | Web-friendly? |
|---|---|---|---|
| **gsplat.js** (WebGL) | ~1M | ~200 MB browser | ✅ Yes, but limited |
| **antimatter15/splat** (WebGL) | ~5M | ~500 MB browser | ✅ Yes, with compression |
| **Three.js + custom shader** | ~2M | ~300 MB browser | ✅ Yes, most flexible |
| **SIBR_viewer** (native) | 10M+ | GPU VRAM | ❌ Desktop only |
| **Unity/Unreal GS plugin** | 10M+ | GPU VRAM | ❌ Desktop only |
| **VRAM-Splat (WebGPU)** | ~10M | ~1 GB browser | ✅ Cutting edge |

**For web deployment:** The `.ply` file must be compressed. Options:
- **Sorted for α-blending** → already sorted by depth during training
- **Quantize positions** (float32 → float16): 50% reduction, minimal quality loss
- **Quantize SH coefficients** (float32 → float16 or uint8): 50-75% reduction
- **Remove low-opacity Gaussians** (< 0.01): 5-20% reduction
- **Compress .ply with gzip**: 3-5× compression ratio (positions + SH are smooth)

Target: **5M Gaussians → ~50-100 MB compressed `.ply.gz`** for browser delivery.

---

## 3. Segmentation: Frame-by-Frame Integration

### 3.1 Segmentation Models

Two tiers based on what you want to segment:

| Model | What it segments | GPU RAM | Speed (per frame, 518×294) | Output |
|---|---|---|---|---|
| **Sky segmentation** (skyseg.onnx, already in repo) | Sky vs non-sky | ~2 GB (ONNX CPU fallback: 0 GB GPU) | ~0.3s CPU / ~0.05s GPU | Binary mask |
| **SAM 2.1 (Segment Anything)** | Everything (promptable) | ~4-6 GB | ~0.5-1.0s | Multi-mask, per-pixel |
| **SAM 2.1-Hiera-Tiny** | Everything (faster) | ~2 GB | ~0.15-0.3s | Multi-mask |
| **SegFormer B5** (ADE20K) | 150 classes (stuff) | ~3 GB | ~0.1s | Per-pixel class labels |
| **Mask2Former** (COCO) | 80 classes (things + stuff) | ~4 GB | ~0.2s | Per-pixel instance masks |
| **MobileSAM / FastSAM** | Everything (lightweight) | ~1 GB | ~0.05-0.1s | Single mask per prompt |
| **YOLOv8-seg** | 80 classes (COCO) | ~2 GB | ~0.03s | Bbox + mask, instances |

### 3.2 Frame-by-Frame Pipeline

Segmentation can run **independently and in parallel** with LingBot-MAP inference, using the SAME input frames:

```
                    ┌─→ LingBot-MAP (GPU #0 or timesliced)
                    │   └→ pose_enc, depth, depth_conf
                    │
Upload → frames ────┤
                    │
                    └─→ Segmentation (GPU #1 or CPU, per-frame)
                        └→ mask[frame_idx, H, W]  (uint8 class labels or float32 confidence)
```

**No dependency between the two paths.** Segmentation can start as soon as the first frame is decoded, while LingBot-MAP needs at least 4-8 scale frames before producing output.

### 3.3 Lifting 2D Masks to 3D

Once both LingBot-MAP and segmentation complete, masks are lifted to 3D:

```python
# For each frame
for frame_idx in range(S):
    depth = predictions["depth"][frame_idx]
    mask_2d = segmentation_masks[frame_idx]  # [H, W] int (class labels)
    
    if mask_2d.shape != depth.shape[:2]:
        mask_2d = cv2.resize(mask_2d, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Unproject depth → world XYZ
    world_xyz = unproject(depth, extrinsic[frame_idx], intrinsic[frame_idx])
    
    # For each class, accumulate labeled points
    for class_id in np.unique(mask_2d):
        class_mask = mask_2d == class_id
        labeled_points[class_id].extend(world_xyz[class_mask])
```

### 3.4 Memory for Segmentation

| Component | Per-frame size | 9000 frames (all at once) | Frame-by-frame |
|---|---|---|---|
| Segmentation model weights | 0.3-2.5 GB (loaded once) | Same | Same |
| Input image (BGR, 518×W) | ~450 KB | ~4 GB | ~450 KB |
| Segmentation mask (uint8) | 150 KB | 1.35 GB | 150 KB |
| **Total per-frame GPU** | **2-6 GB** (incl model) | N/A | **~3-7 GB peak** |

**Frame-by-frame segmentation adds negligible memory.** The mask is just 150 KB/frame. If accumulated for all 9000 frames, masks alone are 1.35 GB — significant but manageable. Better to write masks to disk per-frame and load on demand.

### 3.5 Combining Segmentation with Gaussian Splatting

**Option A: Pre-filter Gaussians by class**
```python
# Only initialize Gaussians for "building" class, skip "sky" and "vegetation"
for each valid pixel:
    if mask_2d[v, u] in target_classes:
        create_gaussian(xyz, rgb, class_id=mask_2d[v, u])
```
→ Segmented GS scene with per-Gaussian semantic labels.

**Option B: Train full GS, then classify Gaussians**
```python
# Train 3DGS normally, then:
for each gaussian in trained_gaussians:
    # Render from each training view, check which class the Gaussian projects to
    gaussian.class_id = majority_vote(projected_classes)
```
→ Post-hoc semantic labeling. Noisier but doesn't constrain training.

**Option C: Semantic Gaussians (extended GS)**
Add a semantic logit head to each Gaussian alongside SH colors. Train jointly on RGB + cross-entropy loss. Requires modified GS rasterizer. Research-grade, but produces Gaussians that can be queried by class.

### 3.6 Sky Segmentation Integration

The existing sky segmentation (`skyseg.onnx`) can be repurposed:

```python
# Already runs per-frame, cached to disk
# Can be extended: instead of binary sky/non-sky, add more classes

# Current: sky_mask ∈ [0, 1]  → filter sky points
# Extended: sky_mask ∈ [0, 1]  → use as "sky probability" per Gaussian
#           → Gaussians with high sky prob get low opacity during training
#           → Sky regions naturally become transparent in the final model
```

---

## 4. End-to-End Production Pipeline Design

### 4.1 Single-Machine Architecture (Dev/MVP)

```
┌─────────────────────────────────────────────────────────┐
│  GPU Server (1× RTX 4090, 24 GB)                        │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ LingBot-MAP      │  │ Segmentation     │             │
│  │ (~3 GB + KV)     │  │ (~3 GB)          │             │
│  │ Time-sliced:     │  │ Runs per-frame   │             │
│  │ process window   │  │ immediately      │             │
│  │ → yield → next   │  │ after decode     │             │
│  └──────────────────┘  └──────────────────┘             │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
│  ┌──────────────────────────────────────┐               │
│  │ Scene Accumulator                     │               │
│  │ - depth + pose → disk                 │               │
│  │ - masks → disk (per-frame)            │               │
│  │ - original frames → disk              │               │
│  └──────────────────────────────────────┘               │
│                      │                                  │
│                      ▼ (after all frames complete)      │
│  ┌──────────────────────────────────────┐               │
│  │ 3DGS Trainer                          │               │
│  │ - Load depth + poses + frames + masks │               │
│  │ - Initialize Gaussians from depth     │               │
│  │ - Train 7K-30K iterations             │               │
│  │ - Compress → .ply.gz (50-100 MB)      │               │
│  └──────────────────────────────────────┘               │
│                      │                                  │
│                      ▼                                  │
│  ┌──────────────────────────────────────┐               │
│  │ Web Server (FastAPI)                  │               │
│  │ - Serve .ply.gz + viewer HTML         │               │
│  │ - Serve segmented point cloud (GLB)   │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

**Time budget for single-machine, 3-minute video (1800 frames @ 10fps):**

| Stage | Duration | GPU RAM |
|---|---|---|
| Frame extraction + preprocessing | ~30s | 0 GB (CPU) |
| LingBot-MAP inference (1800 frames, windowed) | ~6 min @ 5 FPS | ~12 GB |
| Segmentation (per-frame, interleaved) | +~3 min (overlaps with inference mostly) | ~3 GB |
| 3DGS initialization (unproject all depth) | ~30s | ~2 GB |
| 3DGS training (3M Gaussians, 15K iters) | ~10-15 min | ~8 GB |
| Compression + export | ~30s | 0 GB (CPU) |
| **Total** | **~20-25 min** | **~12 GB peak** |

### 4.2 Multi-GPU Architecture (Production)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ GPU #0      │     │ GPU #1      │     │ GPU #2      │
│ LingBot-MAP │     │ Segmentation│     │ 3DGS Trainer│
│ (inference) │     │ (SAM, etc.) │     │ (training)  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  + Storage  │
                    └─────────────┘
```

Each GPU does one thing. Inference and segmentation run in parallel on different GPUs with the same input frames. GS training starts as soon as both complete.

### 4.3 Streaming/Progressive Mode

For immediate feedback, don't wait for GS training:

```
Phase 1 (5 seconds):  Decode first 8 frames → run LingBot scale phase
                       → show initial point cloud preview (low-res GLB)

Phase 2 (30 seconds):  Continue streaming frames → update point cloud progressively
                       → segmentation masks streaming in parallel

Phase 3 (2 minutes):   All frames complete → start GS training
                       → user sees improving point cloud while GS trains

Phase 4 (10 minutes):  GS training complete → swap viewer from point cloud to GS
                       → seamless transition (same camera, better quality)
```

---

## 5. Memory Budget: Everything Combined

### 5.1 Per-Request Storage (Disk)

For a 3-minute video (1800 frames @ 10fps), 518×294 resolution:

| Artifact | Raw size | Compressed (NPZ) | Keep? |
|---|---|---|---|
| Original frames (JPEG, stored) | ~180 MB | 180 MB | ✅ For GS training |
| LingBot-MAP predictions (depth + pose + conf) | 2.2 GB | ~700 MB | ✅ Permanent |
| Segmentation masks (uint8, per frame) | 270 MB | ~50 MB (PNG) | ✅ Permanent |
| Dense point cloud (world XYZ + RGB + class) | 14 GB | N/A (computed on demand) | ❌ Compute on-demand |
| 3DGS .ply file (trained, 3M Gaussians) | 170 MB | ~60 MB (.ply.gz) | ✅ Permanent |
| Camera path JSON | 100 KB | 30 KB | ✅ Permanent |
| Preview GLB (1M points) | 20 MB | 15 MB (glb) | ✅ Permanent |
| **Total per scene** | | **~1.0 GB** | |

**Cleanup policy:** Delete original frames after GS training. Keep predictions + masks + GS file. ~800 MB/scene stored indefinitely.

### 5.2 Peak GPU RAM Timeline (Single GPU, Time-Sliced)

```
Time →
│  LingBot (12 GB)                    │                                     │
│  ├─ window 1 ─┤ ├─ w2 ─┤ ... ├─ w28 ─┤                                    │
│                                     │  GS training (8 GB)                 │
│  Seg (3 GB)     │                   │                                     │
│  ├─ frame 1 ─┤...├─ frame 1800 ─┤  │  (Segmentation finishes early)      │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
Peak: 15 GB (LingBot + Segmentation overlap) → fits on 24 GB GPU
      Without overlap: 12 GB peak (sequential)
```

### 5.3 Scaling to 15-Minute Videos

For a 15-minute video (9000 frames @ 10fps):

| Resource | 3-min video | 15-min video | Scaling |
|---|---|---|---|
| GPU VRAM (inference) | 12 GB | 12 GB | ✅ Constant (windowed mode) |
| GPU VRAM (GS training) | 8 GB | 20+ GB | ⚠️ More Gaussians = more memory |
| CPU RAM (streaming mode) | ~2 GB | ~2 GB | ✅ Constant (if incremental save) |
| Disk (predictions) | 700 MB | 3.5 GB | ⚠️ Linear |
| Disk (GS .ply.gz) | 60 MB | 200-400 MB | ⚠️ Sub-linear (deduplication) |
| Inference time | 6 min | 30 min | ⚠️ Linear |
| GS training time | 15 min | 45-90 min | ⚠️ Super-linear (more Gaussians) |

**For 15-minute videos with GS, a 24 GB GPU is borderline** for GS training. Options:
1. Limit Gaussians to 5M (downsample initialization) → keep training under 12 GB
2. Use two GPUs: one for inference, one for GS training
3. Accept longer training time with gradient accumulation and CPU offloading

---

## 6. Gaussian Splatting Quality Considerations

### 6.1 Where LingBot-MAP Depth Helps GS the Most

| GS challenge | Without depth | With LingBot-MAP depth |
|---|---|---|
| **Float-out artifacts** (Gaussians in empty space) | Frequent, requires many iterations to prune | Rare — Gaussians initialized on surfaces |
| **Thin structures** (poles, edges) | Often missed, need densification | Captured if visible in depth (patch-size limited) |
| **Texture-less walls** | Gaussians spread out, blurry | Depth constrains position, sharper |
| **View-dependent effects** (specular) | Gaussians learn to fake it | Depth provides correct geometry; SH handles appearance |
| **Convergence speed** | 30K iterations typical | 7-15K with good initialization |
| **Need for COLMAP** | Required (30-60 min preprocessing) | **Not needed at all** (LingBot replaces it) |

### 6.2 LingBot-MAP Limitations that Affect GS

| Limitation | Impact on GS | Mitigation |
|---|---|---|
| **Scale ambiguity** (metric scale unknown) | GS scene is arbitrarily scaled | Not a problem for visualization; fix with reference object if needed |
| **Depth noise in distant regions** | Gaussians at wrong depth → blur | Confidence filter drops low-confidence depth; GS optimizer corrects small errors |
| **Window boundary misalignment** | Discontinuity in poses → seam in GS | Alignment correction in windowed mode; GS training smooths seams |
| **Fixed resolution (518px width)** | Misses fine details | Use original-resolution images for GS appearance, only use depth at model resolution |
| **No depth for sky** | Sky Gaussians would be at wrong depth | Sky mask → don't initialize Gaussians in sky; sky rendered as background color |

### 6.3 Hybrid Approach: Depth for Geometry, GS for Appearance

LingBot-MAP provides **metric depth** (geometry). GS provides **photorealistic appearance**. Combine them:

```python
# Training loss with depth regularization
loss = rgb_loss + λ_depth * depth_loss + λ_ssim * ssim_loss

# Depth loss: encourages Gaussians to stay near the depth-initialized positions
depth_loss = || rendered_depth - lingbot_depth ||
```

This prevents GS from "explaining away" depth errors with spurious Gaussians.

---

## 7. Segmentation: Implementation Details

### 7.1 Frame-by-Frame Architecture

```python
# lingbot_map/segmentation.py  (NEW FILE)

class SegmentationPipeline:
    """Runs segmentation on frames as they become available.
    
    Designed for streaming: frames arrive in order, masks are produced
    immediately and written to disk. No accumulation needed.
    """
    
    def __init__(self, model_name: str = "sam2.1_hiera_tiny", device: str = "cuda"):
        self.model = self._load_model(model_name, device)
        self.device = device
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Segment a single frame (H, W, 3) BGR → (H, W) uint8 mask.
        
        Called per-frame as video is decoded. Non-blocking if on separate GPU.
        """
        mask = self.model.predict(frame)  # model-specific inference
        return mask
    
    def process_batch(self, frames: np.ndarray) -> np.ndarray:
        """Segment multiple frames at once for efficiency."""
        masks = self.model.predict_batch(frames)
        return masks
    
    def lift_to_3d(
        self,
        masks_2d: np.ndarray,     # [S, H, W] int
        depth: np.ndarray,         # [S, H, W, 1]
        extrinsic: np.ndarray,     # [S, 3, 4]
        intrinsic: np.ndarray,     # [S, 3, 3]
        target_classes: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Lift 2D segmentation masks to 3D labeled point cloud.
        
        Returns:
            dict mapping class_id → [N, 3] world XYZ points
        """
        labeled_points = defaultdict(list)
        
        for frame_idx in range(len(masks_2d)):
            world_xyz = unproject_depth_map_to_point_map(
                depth[frame_idx:frame_idx+1],
                extrinsic[frame_idx:frame_idx+1],
                intrinsic[frame_idx:frame_idx+1],
            )[0]  # [H, W, 3]
            
            mask = masks_2d[frame_idx]
            if mask.shape != world_xyz.shape[:2]:
                mask = cv2.resize(mask, (world_xyz.shape[1], world_xyz.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            
            for class_id in np.unique(mask):
                if target_classes and class_id not in target_classes:
                    continue
                class_mask = mask == class_id
                labeled_points[class_id].append(world_xyz[class_mask])
        
        return {k: np.concatenate(v, axis=0) for k, v in labeled_points.items()}
```

### 7.2 Integration Points

There are three places to hook segmentation into the existing pipeline:

| Hook | When | Pros | Cons |
|---|---|---|---|
| **A: During video decode** (`load_images_from_video`) | Immediately after `cap.read()` → decode BGR frame | Earliest possible, no GPU contention with LingBot | Segmentation slows down frame extraction |
| **B: In parallel with LingBot** (separate thread/GPU) | While LingBot processes window N, segment frames for window N+1 | Overlaps compute, no added latency | Requires coordination or separate GPU |
| **C: After all inference** (batch) | All frames + depth available | Simplest, can use batch inference | Adds post-processing time, no streaming benefit |

**Recommendation: Hook B for production, Hook C for MVP.**

### 7.3 Semantic Point Cloud Export

```python
def export_semantic_glb(
    labeled_points: dict[int, np.ndarray],
    class_names: dict[int, str],
    output_path: str,
    downsample: int = 4,
):
    """Export 3D semantic point cloud as GLB with per-class colors."""
    import trimesh
    
    # Predefined colormap for classes (or use ADE20K/COCO colors)
    class_colors = {
        0: [128, 128, 128],  # wall
        1: [0, 255, 0],      # floor
        2: [255, 0, 0],      # chair
        # ...
    }
    
    scene = trimesh.Scene()
    for class_id, points in labeled_points.items():
        if len(points) < 100:
            continue
        # Downsample
        points = points[::downsample]
        color = class_colors.get(class_id, [255, 255, 255])
        colors = np.tile(color, (len(points), 1))
        
        pc = trimesh.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc, node_name=class_names.get(class_id, f"class_{class_id}"))
    
    scene.export(output_path)
```

---

## 8. Integration with 3DGS Viewers

### 8.1 Browser-Based GS Viewers

| Viewer | URL / Package | Gaussians | Features |
|---|---|---|---|
| **antimatter15/splat** | `github.com/antimatter15/splat` | ~5M | WebGL, fast, sorted .ply |
| **gsplat.js** | `github.com/huggingface/gsplat.js` | ~1-2M | WebGL, Three.js compatible |
| **VRAM-Splat** | `github.com/nickgante/VRAM-Splat` | ~10M | WebGPU, cutting edge |
| **playcanvas-gs** | PlayCanvas extension | ~5M | Integrated engine |
| **Luma web viewer** | SaaS | Varies | Hosted, production-quality |

### 8.2 Serving GS from FastAPI

```python
@app.get("/api/scenes/{scene_id}/gaussian_splat")
async def get_gaussian_splat(scene_id: str):
    """Serve compressed Gaussian Splatting .ply.gz file."""
    ply_path = scene_store.get_gs_path(scene_id)
    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "Content-Disposition": f"inline; filename={scene_id}_gs.ply",
        }
    )

@app.get("/api/scenes/{scene_id}/semantic_glb")
async def get_semantic_glb(scene_id: str, classes: str = "all"):
    """Serve semantic point cloud as GLB."""
    glb_path = scene_store.get_semantic_glb_path(scene_id, classes)
    return FileResponse(glb_path, media_type="model/gltf-binary")
```

### 8.3 Client-Side Rendering (HTML/JS skeleton)

```html
<!-- Viewer page served by FastAPI -->
<canvas id="gs-canvas"></canvas>
<script type="module">
  import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest/dist/index.js";
  
  const sceneId = "{{ scene_id }}";
  
  // Load GS scene
  const response = await fetch(`/api/scenes/${sceneId}/gaussian_splat`);
  const buffer = await response.arrayBuffer();
  const splatData = SPLAT.parsePlyBuffer(buffer);
  
  // Initialize viewer
  const viewer = new SPLAT.Viewer({
    canvas: document.getElementById('gs-canvas'),
    camera: SPLAT.Camera.fromJson(await fetch(`/api/scenes/${sceneId}/cameras`).then(r=>r.json())),
  });
  viewer.addSplat(splatData);
  viewer.start();
  
  // Optional: toggle between GS and semantic point cloud
  document.getElementById('view-semantic').onclick = async () => {
    const glbUrl = `/api/scenes/${sceneId}/semantic_glb`;
    loadGLBViewer(glbUrl);  // separate Three.js viewer
  };
</script>
```

---

## 9. Implementation Roadmap

### Phase 1: Core GS Integration (2 weeks)

- [ ] **`lingbot_map/gs/initializer.py`** — Convert depth + poses → Gaussian initialization
- [ ] **`lingbot_map/gs/trainer.py`** — Wrap `gsplat` / `diff-gaussian-rasterization` training loop
- [ ] **`lingbot_map/gs/export.py`** — Export trained Gaussians to `.ply` + compress to `.ply.gz`
- [ ] **CLI:** `python -m lingbot_map.gs train --predictions scene.npz --output scene.ply`
- [ ] **Test:** Train on existing courthouse dataset, compare quality with COLMAP→GS baseline

### Phase 2: Segmentation Pipeline (1-2 weeks)

- [ ] **`lingbot_map/segmentation.py`** — `SegmentationPipeline` class (see §7.1)
- [ ] **SAM 2.1 integration** — Prompt-based (click-to-segment) + automatic (grid prompts)
- [ ] **Frame-by-frame streaming** — Hook into video decode
- [ ] **`lift_to_3d()`** — 2D masks → 3D labeled points
- [ ] **`export_semantic_glb()`** — Per-class colored GLB

### Phase 3: Server Integration (2 weeks)

- [ ] **`SceneProcessor`** extended with GS + segmentation dispatch
- [ ] **Background task:** after inference → trigger GS training (Celery / arq / asyncio task)
- [ ] **Progress reporting:** WebSocket updates for GS training progress (iterations, PSNR)
- [ ] **Storage management:** Cleanup original frames after GS training; keep `.ply.gz` + masks + predictions

### Phase 4: Web Viewer (parallel, frontend)

- [ ] **Three.js GS viewer** (antimatter15/splat or gsplat.js)
- [ ] **Semantic point cloud viewer** (Three.js + GLB, class toggle checkboxes)
- [ ] **Progressive loading:** Preview point cloud → GS warmup → full GS
- [ ] **Interactive segmentation:** Click-to-select, class filter, confidence slider

---

## 10. Key Decisions and Trade-offs

### 10.1 GS Training: On-Demand vs Pre-Computed

| | On-demand (train after upload) | Pre-computed (cache trained models) |
|---|---|---|
| **Latency** | 10-35 min after upload | Instant |
| **Storage per scene** | ~1 GB (predictions + masks + GS) | ~60 MB (GS .ply.gz only) |
| **Flexibility** | Can adjust training params per scene | Fixed quality |
| **GPU cost** | 10-35 min GPU time per upload | Amortized across views |

**Recommendation:** On-demand for MVP (users expect to wait for processing). Pre-computed + caching for popular/revisited scenes in production.

### 10.2 Segmentation: Server vs Client

| | Server-side | Client-side (in-browser) |
|---|---|---|
| **Model** | SAM 2.1 (6 GB) | MobileSAM / FastSAM (~200 MB) |
| **Latency** | Per-frame, parallel with inference | After download, per-frame |
| **Quality** | State-of-the-art | Slightly worse |
| **User interaction** | Pre-computed | Interactive (click-to-segment) |
| **Bandwidth** | Masks only (150 KB/frame) | Model weights + frames → masks |

**Recommendation:** Server-side semantic segmentation (automatic classes) + client-side SAM for interactive queries.

### 10.3 One GPU vs Multi-GPU

| | Single RTX 4090 (24 GB) | 2× RTX 4090 | 4× A10G (96 GB total) |
|---|---|---|---|
| **Inference + Seg + GS** | Time-sliced, 20-25 min/scene | Parallel, 15-20 min | Parallel, 10-15 min |
| **Concurrent users** | 1 (time-sliced) | 2 (one per GPU) | 4-6 |
| **Cost** | ~$2/hr (cloud) | ~$4/hr | ~$8/hr |
| **Complexity** | Low | Medium | High (K8s) |

---

## 11. Bottom Line

| Question | Answer |
|---|---|
| **Can we do Gaussian Splatting?** | ✅ **Yes, and it's an ideal input.** Depth-initialized GS converges 2-4× faster than COLMAP→random→GS. No COLMAP needed. |
| **Does GS increase memory?** | ✅ Yes. GS training adds 3-12 GB GPU, 10-35 min compute, and ~60 MB disk per scene. Separate from inference memory (doesn't run simultaneously on single GPU). |
| **Can segmentation run frame-by-frame?** | ✅ Yes. Segmentation is independent of LingBot-MAP. Can run on separate GPU or time-sliced. Per-frame mask is only 150 KB. |
| **What's the combined pipeline latency?** | ~20 min for 3-min video (inference 6 min + GS training 12 min + overhead). ~45 min for 15-min video. |
| **What's the storage footprint?** | ~1 GB/scene (predictions 700 MB + masks 50 MB + GS 60 MB + previews). Original frames deleted after GS training. |
| **Web viewer feasible?** | ✅ Yes. Compressed GS file is 50-100 MB for 5M Gaussians. gsplat.js or antimatter15/splat renders in-browser at 60 FPS. Semantic GLB adds 15-30 MB. |
| **Biggest implementation risk?** | GS training on 12 GB GPUs for scenes >2M Gaussians. Mitigation: limit Gaussian count, use density control carefully. |
| **Should we replace the point cloud viewer with GS?** | **No — complement.** Point cloud for immediate preview (5s latency). GS for final quality (10-35 min latency). Seamless transition between both in the same viewer. |
