# Point Cloud Deep Dive: What It Is, Where It Comes From, and How to Tame It

> **Correction to prior document:** The model does **NOT** produce `world_points` by default (`enable_point=False`). The point cloud is always computed from `depth + extrinsic + intrinsic` in post-processing (either in the Viser viewer, the GLB exporter, or the `rgbd_render` pipeline). The model only outputs `pose_enc` + `depth` + `depth_conf`. This is a good thing — depth is 1/3 the size of world_points.

---

## 1. How the Point Cloud is Generated

### 1.1 The Unprojection Pipeline

```
Model output:         depth[S,H,W,1]    pose_enc[S,9]
                         │                    │
Postprocess:              │           pose_encoding_to_extri_intri()
                         │                    │
                         │           extrinsic[S,3,4]    intrinsic[S,3,3]
                         │                    │               │
Unproject (per frame):    └────────────────────┴───────────────┘
                                        │
                         depth_to_world_coords_points(depth[i], ext[i], intr[i])
                                        │
                         world_points[i, H, W, 3]   ← 152,292 3D points per frame
```

### 1.2 The Math (Per Frame)

For a single frame with depth map `D[H, W]`, extrinsic `E[3,4]`, intrinsic `K[3,3]`:

```
Step 1: Pixel → Camera coordinates
  For each pixel (u, v) with depth d = D[v, u]:
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

Step 2: Camera → World coordinates
  R = E[:3, :3]  (rotation, world→camera)
  t = E[:3, 3]   (translation)

  cam_to_world = inverse([R|t])  →  R⁻¹ | -R⁻¹t

  world_xyz = R⁻¹ · [x_cam, y_cam, z_cam] + (-R⁻¹t)
```

This is a pure matrix multiply — **no GPU needed**. NumPy on CPU does it in ~0.5ms per frame.

### 1.3 Why Separate Depth from World Points?

The model predicts **depth in a canonical metric space**, and the camera head predicts **pose** (position + orientation + field of view). Combining them produces world-space points. This separation is intentional:

| | Depth-only | World points |
|---|---|---|
| Size per frame | 609 KB | 1.83 MB |
| Model output? | ✅ Yes | ❌ No (computed later) |
| Client needs? | Pose + intrinsics to unproject | Nothing else |
| Flexibility | Client chooses resolution, filtering | Fixed at unproject time |

---

## 2. Point Cloud Size Reality Check

### 2.1 Dense Point Count (Default)

At 518×294 resolution, every pixel with valid depth becomes a point:

```
Pixels per frame:  294 × 518 = 152,292
Valid depth pixels: ~60-80% (rest are sky, low confidence, or zero)
Typical points/frame: ~90,000–120,000
```

For a 15-minute video at 10 fps (9000 frames):

```
Total raw points:     9000 × 152,292 = 1.37 billion
Valid points (~70%):  ~960 million
Raw data (3×float32): 960M × 12 bytes = 11.5 GB
With RGB color:        +3×uint8 = +2.9 GB → 14.4 GB total
```

**This is far too large for any client-side viewer.** Even Three.js struggles above ~10M points. Cesium/potree can handle ~100M with level-of-detail but not billions.

### 2.2 Per-Frame Contribution

Not all frames contribute equally:

```
Frame 0-7 (scale):   Full density, highest quality (bidirectional attention)
Keyframes:           Full density, stored in KV cache, highest quality
Non-keyframes:       Full density, lower temporal stability (attend only to keyframes)
Overlap frames:      Duplicate coverage (same region from two windows)
```

---

## 3. Point Cloud Reduction Strategies

### 🥇 Strategy A: Confidence-Based Filtering (3–10× reduction, zero visual loss)

The model outputs `depth_conf[H, W]` for every pixel. Low-confidence points are typically sky, specular surfaces, or distant regions where depth is ambiguous.

```python
# Filter to top-N% most confident points
conf_threshold = np.percentile(depth_conf, 30)  # keep top 70%
mask = depth_conf > conf_threshold
filtered_points = world_points[mask]
```

| Percentile cutoff | Approx points retained | Visual impact |
|---|---|---|
| 0% (keep all) | 100% | Baseline |
| 20% | ~80% | Removes obvious noise, sky |
| 50% | ~50% | Removes most surfaces, keeps structure |
| 80% | ~20% | Sparse but recognizable |
| 95% | ~5% | Only very high-confidence points |

The GLB exporter already does this: `conf_thres=50` means it drops the bottom 50% of points by confidence.

### 🥈 Strategy B: Spatial Downsampling (4–100× reduction, minor visual loss)

Simple strided sampling:

```python
# Take every Nth pixel
stride = 4  # 16× fewer points
downsampled = world_points[::stride, ::stride, :]  # 74×37 → 19×10 grid
```

Or random subsampling:

```python
n_keep = 10000  # per frame
indices = np.random.choice(H*W, n_keep, replace=False)
sampled = world_points.reshape(-1, 3)[indices]
```

The demo already does this: `downsample_factor=10` → 100× reduction.

### 🥉 Strategy C: Keyframe-Only Point Cloud (frame_count × reduction)

Only unproject depth for keyframes (frames stored in KV cache). Non-keyframe poses are still available for camera trajectory visualization.

With `keyframe_interval=4` and `num_scale_frames=8`:
- 9000 frames → ~2256 keyframes (9000/4) + 8 scale per window
- Points reduced by 4×
- Camera path still shows full trajectory

The batch demo supports this with `--keyframes_only_points`.

### 4️⃣ Strategy D: Voxel Downsampling (5–20× reduction, uniform density)

Voxel grid filter — keeps one point per voxel cell:

```python
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 1cm voxels
```

The `rgbd_render` pipeline already does this with configurable `voxel_size`.

### 5️⃣ Strategy E: Temporal Deduplication (2–5× reduction for static scenes)

In static regions, consecutive frames produce nearly identical points. Deduplicate across time:

```python
# Approach: accumulate in a voxel grid across all frames
# Each voxel cell keeps only the highest-confidence point
# This automatically deduplicates static geometry
```

Not currently implemented but highly effective for architectural scenes.

### 6️⃣ Strategy F: Frame Skipping (N× reduction)

Skip frames entirely in the point cloud (keep poses for camera path):

```
Keep every 4th frame for depth unprojection → 4× smaller
Combine with keyframe-only for even more reduction
```

### Combined Effect

For a 9000-frame video, a reasonable web-friendly configuration:

| Step | Reduction | Cumulative points |
|---|---|---|
| All frames, full density | — | 960M (impossible) |
| Keyframes only (kf_int=4) | 4× | 240M |
| Spatial downsample 4× | 16× | 60M |
| Confidence filter (top 70%) | 1.4× | 43M |
| Voxel grid (2cm) | ~3× | ~14M |
| **Total** | **~70×** | **~14M points** ✅ |

14M points with RGB colors is ~170 MB — manageable for progressive loading in a web viewer.

---

## 4. Server vs Client Point Cloud Generation

### Option A: Server-Side Generation (Current Default)

```
Server: depth → unproject → world_points → filter → downsample → GLB/NPZ
Client: receives pre-computed point cloud
```

| Pros | Cons |
|---|---|
| Client is trivial (just load GLB) | Server CPU/RAM cost per request |
| Full control over filtering | 3× more data to transmit (XYZ vs depth) |
| User gets exactly what they see | Cannot re-filter without re-requesting |

### Option B: Client-Side Generation (Recommended for Web)

```
Server: depth + extrinsic + intrinsic → compressed → send to client
Client: unproject → filter → downsample → render
```

| Pros | Cons |
|---|---|
| 3× less bandwidth (depth vs XYZ) | Client needs to implement unprojection |
| User can re-filter interactively | Client CPU cost (~1ms/frame for unproject, ~5ms for filtering) |
| Server stateless, lower RAM | More client-side code |

### Option C: Hybrid (Best of Both)

```
Server: depth + extrinsic (small, always sent)
        Optionally pre-compute GLB at 3 quality levels (low/med/high)
        
Client: Progressive loading:
        1. Load camera path + low-res point cloud (instant, for preview)
        2. Stream higher-res point cloud as bandwidth allows
        3. Allow user to adjust confidence/density sliders
```

---

## 5. Serialization Formats for Point Clouds

| Format | Size (14M points, colored) | Streaming? | Browser support |
|---|---|---|---|
| **glTF/GLB** (binary) | 170–250 MB | ❌ Monolithic | ✅ Three.js, Cesium, model-viewer |
| **Draco glTF** | 30–60 MB | ❌ Monolithic | ✅ Three.js (with DRACOLoader) |
| **3D Tiles** (b3dm) | Per-tile, LOD levels | ✅ Progressive | ✅ Cesium, deck.gl |
| **Potree** (octree) | Per-node, LOD levels | ✅ Progressive | ✅ potree.js, Three.js |
| **LAS/LAZ** | 50–100 MB | ❌ Monolithic | ⚠️ Specialized (point cloud only) |
| **NPZ** (numpy) | 60–120 MB | ❌ Monolithic | ❌ Not browser-native |
| **Custom binary** (float32) | 14M × 12 = 168 MB | ✅ Trivial to stream | ✅ Fetch + ArrayBuffer |

### Recommendation Matrix

| Use Case | Format | Why |
|---|---|---|
| "Download my point cloud" | **LAS/LAZ** or **Draco glTF** | Industry standard, compact |
| "View in browser immediately" | **3D Tiles** or **Potree** | Progressive LOD, handles 100M+ points |
| "Quick preview / thumbnail" | **Low-res GLB** (1M points) | Instant load, <5 MB |
| "API integration" | **NPZ** (depth) + JSON (poses) | Machine-readable, re-processable |
| "Archive / reproduce" | **NPZ** (all predictions) | Full fidelity for re-rendering |

---

## 6. The RGB Question

The current viewer colors each point with the corresponding pixel color from the input image. This gives photorealistic point clouds but has trade-offs:

### 6.1 Color Size

```
Per point: 3 bytes (RGB uint8)
14M points: 42 MB  (significant but manageable)
960M points: 2.9 GB  (prohibitive)
```

### 6.2 Alternatives to Per-Point RGB

| Approach | Size | Visual quality |
|---|---|---|
| Per-point RGB (current) | 3 bytes/point | Photorealistic |
| Per-vertex normal (computed from depth) | 3 bytes/point | Shaded but colorless |
| Single global color | 3 bytes total | Monochrome point cloud |
| Confidence-mapped colormap | 1 byte/point (index into colormap) | Shows quality, not appearance |
| Disable color (just XYZ) | 0 bytes/point | Wireframe white |

The `rgbd_render` pipeline supports rendering colored points in Open3D — it's used for the offline video rendering, not for the web viewer.

---

## 7. Memory Profile for Point Cloud Operations

### 7.1 Unprojection Memory (Per Frame)

Computing world_points from a single frame:

```
Input:
  depth:         [294, 518]     float32 = 609 KB
  extrinsic:     [3, 4]         float32 = 48 B
  intrinsic:     [3, 3]         float32 = 36 B

Temporary:
  pixel grid:    [294, 518, 3]  float32 = 1.83 MB  (x, y, 1 coords)
  cam_coords:    [294, 518, 3]  float32 = 1.83 MB
  world_xyz:     [294, 518, 3]  float32 = 1.83 MB

Output:
  world_points:  [294, 518, 3]  float32 = 1.83 MB
  valid_mask:    [294, 518]     bool    = 152 KB

Peak per frame: ~4.3 MB
All 9000 frames at once: ~38 GB  ← DON'T DO THIS
One frame at a time:   ~4.3 MB  ← Fine
```

### 7.2 Incremental Point Cloud Building

The right pattern for large videos:

```python
# Process frame by frame, accumulate in a sparse structure
voxel_grid = SparseVoxelGrid(voxel_size=0.02)  # 2cm voxels

for frame_idx, depth_map in enumerate(stream_depths()):
    world_xyz = unproject_frame(depth_map, ext[frame_idx], intr[frame_idx])
    conf = depth_conf[frame_idx]
    colors = images[frame_idx]
    
    # Filter low confidence
    mask = conf > conf_threshold
    points = world_xyz[mask]
    rgbs = colors[mask]
    
    # Insert into voxel grid (auto-deduplicates)
    voxel_grid.insert(points, rgbs, conf[mask])
    
    # Free per-frame data
    del world_xyz, points, rgbs

# Export accumulated sparse cloud
voxel_grid.export_glb("scene.glb")
```

This approach:
- Peak memory: O(voxel_grid_size), not O(num_frames)
- Auto-deduplication of static geometry
- Natural LOD (voxel size determines density)
- Can be interrupted and resumed

---

## 8. Quality Considerations

### 8.1 Where the Point Cloud is Weakest

The model's depth predictions are weakest in these regions:

| Region | Cause | Mitigation |
|---|---|---|
| **Sky** | No geometry, model hallucinates depth | Sky segmentation mask (already implemented) |
| **Specular surfaces** (water, glass) | No consistent depth signal | Confidence filter drops these naturally |
| **Thin structures** (poles, wires) | Below patch resolution (14px) | Nothing — fundamental resolution limit |
| **Distant objects** (>100m) | Depth resolution degrades with distance | Accept higher uncertainty |
| **Fast camera motion** | Motion blur → depth ambiguity | Keyframe quality better than non-keyframe |
| **Window boundaries** | Scale misalignment between windows | Alignment correction mitigates |

### 8.2 Scale Ambiguity

The model predicts pose and depth up to an unknown global scale factor. This is inherent to monocular depth estimation. In windowed mode, the `scale_mode` alignment corrects this between windows. But the absolute scale (meters) is not guaranteed to match reality.

For a web viewer, this usually doesn't matter — relative proportions are correct. For measurement applications, a known reference distance (e.g., a 1m object in the scene) must be provided.

---

## 9. Concrete Recommendations for a Web App

### 9.1 Default Pipeline

```
1. Upload video → server extracts frames at ~5 fps
2. Run windowed inference (model: GPU, depth + pose output)
3. Save per-frame depth + extrinsic to NPZ on disk (NOT world_points)
4. Return scene_id immediately

On-demand:
5. GET /scenes/{id}/preview.glb  → Low-res GLB (1M points, instant)
6. GET /scenes/{id}/full.glb     → Medium-res GLB (10M points, ~30s)
7. GET /scenes/{id}/download.npz → Full predictions (depth + pose, for offline use)

Client viewer:
8. Three.js loads preview GLB immediately
9. Progressive background load of full GLB or 3D Tiles
10. Interactive confidence/density sliders re-filter server-side (or re-unproject)
```

### 9.2 Size Budget for a 3-Minute Video (1800 frames @ 5fps)

| Artifact | Size | Notes |
|---|---|---|
| Raw predictions (depth + pose) | 1800 × 609KB = 1.1 GB | Server disk, NPZ compressed → ~350 MB |
| Preview GLB (1M points) | ~15 MB | Instant client load |
| Full GLB (10M points) | ~140 MB | Background download |
| Full GLB Draco (10M points) | ~35 MB | Best for web |
| Camera path JSON | 1800 × 48B = 86 KB | Trivial |
| **Total server disk** | **~500 MB** | Clean up after 7 days |
| **Total client download** | **~50 MB** | Progressive, most users only load preview |

### 9.3 What NOT to Ship to the Client

- ❌ Full 1800-frame dense point cloud (1800 × 1.8 MB = 3.2 GB)
- ❌ Individual frame data without LOD (user won't see the difference)
- ❌ Full-resolution RGB textures per point (42 MB just for colors)
- ✅ Downsampled, voxel-filtered, confidence-filtered, Draco-compressed GLB

---

## 10. Bottom Line

| Question | Answer |
|---|---|
| Where does the point cloud come from? | Computed from `depth + extrinsic` AFTER inference, not during. The model only outputs depth maps. |
| How big is the raw point cloud? | 152K points/frame × 9000 frames = 1.37B points. Impractical for any viewer. |
| Can we make it web-friendly? | Yes. Combined filtering (keyframe-only + spatial downsample + confidence + voxel) reduces 960M → 14M points with minimal visual loss. |
| Server or client unprojection? | **Client-side unprojection** saves 3× bandwidth and enables interactive filtering. Send depth, not world_points. |
| Best serialization format? | **3D Tiles** or **Draco glTF** for progressive browser loading. **NPZ** for API/download. |
| Single biggest win? | **Don't ship world_points at all.** Ship depth + extrinsics (3× smaller), unproject on the client. Combined with keyframe-only unprojection, this is a 12× total reduction. |
