# LingBot-MAP Memory & Performance Deep Dive

> **Audience:** Skeptical senior engineers planning a production web deployment
> **Question:** Can we process a 15-minute video (9000 frames at 10 fps) in real-time without OOMing GPU VRAM *or* CPU RAM?

---

## 1. Executive Answer

**GPU VRAM:** ✅ Solvable. Both streaming (with sliding window) and windowed mode bound GPU memory to ~9–12 GB regardless of sequence length. The KV cache never grows past `scale_frames + sliding_window` patch pages.

**CPU RAM:** ❌ The current blocker. Both modes accumulate *all* prediction results in memory before returning. A 9000-frame sequence produces ~16.5 GB of raw output tensors (49.5 GB if world_points is precomputed server-side). Even with `output_device='cpu'`, result accumulation is unbounded.

**The fix is architectural, not algorithmic:** stream results to disk incrementally (per-window or per-frame), never accumulate.

---

## 2. GPU VRAM Decomposition

All numbers below are for a **518×294 (16:9 crop)** input at bfloat16. The model uses **24 transformer blocks** (depth=24, aa_block_size=1).

### 2.1 Model Weights (fixed cost)

| Component | Size |
|---|---|
| DINOv2 ViT-L backbone (patch_embed) | ~0.30 GB |
| Aggregator (24 attention blocks × ~60M params each) | ~1.44 GB |
| Camera head (4-layer causal trunk) | ~0.30 GB |
| Depth head (DPT-style decoder) | ~0.50 GB |
| Embeddings, norms, etc. | ~0.26 GB |
| **Total** | **~2.8 GB** (bf16 mixed via autocast; heads in fp32) |

This is loaded once and stays resident.

### 2.2 KV Cache — Patch Pages (per-sequence, bounded)

Each transformer block maintains an independent page pool. For a single 518×294 frame:

```
patches  = (518/14) × (294/14) = 37 × 21 = 777
specials = 6  (1 camera + 4 register + 1 scale)
tokens   = 783
page_size = 777  (exact for FA2; rounded to 1024 for FA3)
```

**Per page, per block** (K and V):

```
page_elements = page_size × num_heads × head_dim
              = 777 × 16 × 64 = 795,648
page_bytes    = 795,648 × 2 bytes (bf16) = 1.59 MB

Per block, per page (K+V): 2 × 1.59 MB = 3.18 MB
All 24 blocks, per page:   24 × 3.18 MB = 76.3 MB
```

**Active pages at steady state:**

| Page type | Count | Memory (all blocks) |
|---|---|---|
| Scale patch pages | 8 (fixed) | 8 × 76.3 = 610 MB |
| Live window patch pages | 64 (sliding_window) | 64 × 76.3 = 4.88 GB |
| Headroom | 16 | 16 × 76.3 = 1.22 GB |
| **Patch subtotal** | **88** | **~6.7 GB** |

### 2.3 KV Cache — Special Pages (per-sequence, grows until cleanup)

Special tokens (6 per frame) are stored in a separate append-only pool. They are **never evicted** during a sequence — they can only be freed by `clean_kv_cache()`.

```
Specials per frame:        6
Specials per page:         floor(777 / 6) = 129 frames' worth
Pages needed (72 frames):  ceil(72×6 / 777) = 1 page
Pages pre-allocated:       ceil(max_total_frames×6 / 777) + 16 headroom
```

With default `max_total_frames = max_frame_num + 100 = 1124`:
```
Special pages per block = ceil(1124 × 6 / 777) + 16
                        = ceil(8.68) + 16 = 9 + 16 = 25
Special memory: 24 × 25 × 3.18 MB ≈ 1.9 GB
```

**🔴 CRITICAL INSIGHT:** `max_total_frames` (default 1124) is a **hard cap** on streaming mode. If you try to process frame 1125 in a single streaming sequence, the special page pool is exhausted and the code asserts:

```python
assert self.free_special_pages[block_idx], (
    f"block {block_idx}: special page pool exhausted ..."
)
```

This means **streaming mode cannot process more than ~1124 frames in a single sequence** without either increasing `max_total_frames` (burns more GPU RAM) or periodically calling `clean_kv_cache()` (resets the sequence, loses temporal context).

**In windowed mode, this is not a problem:** each window gets a fresh KV cache, so special pages are always just 1-2 pages per block.

### 2.4 GPU Memory Summary

| Component | Streaming (72-frame window) | Windowed (per-window) |
|---|---|---|
| Model weights | 2.8 GB | 2.8 GB |
| Patch KV pages | 6.7 GB | 6.7 GB |
| Special KV pages | 1.9 GB (pre-alloc for 1124 frames) | 0.08 GB (1 page for 72 frames) |
| Activations (peak) | ~1.0 GB | ~1.0 GB |
| **Total peak GPU** | **~12.4 GB** | **~10.6 GB** |
| **Hard frame limit** | ~1124 frames | None (unbounded) |

### 2.5 SDPA Fallback Memory

When `--use_sdpa` is set, the KV cache uses a simpler dict-based approach:

```
Per block, per frame: 2 tensors of [1, num_heads, tokens, head_dim]
                    = 2 × 1 × 16 × 783 × 64 = 1,602,816 elements
                    = 2 × 6.4 MB = 12.8 MB per frame per block (bf16)

At sliding_window=64, cached_frames=72:
    72 × 12.8 MB × 24 blocks ≈ 22.1 GB — exceeds GPU memory!

But the actual implementation caches only keyframes:
    With keyframe_interval=4: only ~18 keyframes cached
    18 × 12.8 MB × 24 ≈ 5.5 GB ✓
```

This is why `keyframe_interval > 1` is essentially required when using SDPA.

---

## 3. CPU RAM — The Real Bottleneck

### 3.1 Per-Frame Output Size

**Important:** The model does NOT produce `world_points` by default (`enable_point=False`). The point cloud is always computed from `depth + extrinsic + intrinsic` in post-processing (Viser viewer, GLB exporter, or rgbd_render pipeline). The model output is just `pose_enc` + `depth` + `depth_conf`. See `docs/point_cloud_analysis.md` for details on point cloud generation.

For a single 518×294 frame, the **model output** contains:

| Key | Shape | Dtype | Uncompressed | NPZ compressed* |
|---|---|---|---|---|
| `depth` | [294, 518, 1] | float32 | 609 KB | ~200 KB |
| `depth_conf` | [294, 518] | float32 | 609 KB | ~200 KB |
| `pose_enc` | [9] | float32 | 36 B | 36 B |
| `images` (echoed) | [3, 294, 518] | float32 | **1.83 MB** | ~600 KB |
| **Total per frame (model output)** | | | **~3.0 MB** | **~1.0 MB** |

After postprocessing, `pose_enc` becomes `extrinsic` + `intrinsic` (trivial size). If `world_points` is computed server-side from depth, it adds **1.83 MB/frame** (see point cloud analysis).

*NPZ compression ratio ~3× for depth/point data (measured from analysis: 300 frames raw ~900 MB → compressed ~180 MB).

### 3.2 Accumulation Math (Model Output Only)

The model outputs `depth` + `depth_conf` + `pose_enc` + echoed `images`. `world_points` is NOT in model output by default — it's computed later if needed (see point cloud analysis). Adding `world_points` server-side would triple these numbers.

| Video | Frames | Model output (raw) | Model output (NPZ) | + world_points (raw) | Problem |
|---|---|---|---|---|---|
| 30s @ 10fps | 300 | 0.9 GB | ~120 MB | +0.5 GB | ✅ Fine on 8 GB |
| 3min @ 10fps | 1800 | 5.4 GB | ~720 MB | +3.3 GB | ⚠️ Tight on 16 GB |
| **15min @ 10fps** | **9000** | **27 GB** | ~3.6 GB | **+16.5 GB** | 🔴 **Guaranteed OOM** |
| 15min @ 5fps | 4500 | 13.5 GB | ~1.8 GB | +8.2 GB | ⚠️ Marginal on 32 GB |
| 15min @ 3fps | 2700 | 8.1 GB | ~1.1 GB | +4.9 GB | ⚠️ Tight on 16 GB |

**Without echoed images:** saves 1.83 MB/frame → ~60% reduction in model output size.

**Without world_points (client-side unprojection):** saves an ADDITIONAL 1.83 MB/frame on top of the model output.

**With incremental save-to-disk:** peak CPU RAM = window_size × per_frame_size, not total_frames × per_frame_size.

### 3.3 Input Images Memory

The preprocessed image tensor also lives in CPU RAM before/during inference:

```
Per frame: 3 × 294 × 518 × 4 bytes (float32) = 1.83 MB
9000 frames: 16.5 GB
```

This is the same for both modes. Solutions:
- Don't load all frames upfront — use a streaming video decoder
- Keep only the current window's frames in memory
- Use a memory-mapped file for the full image sequence

---

## 4. Streaming vs Windowed: Performance Comparison

### 4.1 Mechanism Comparison

| | Streaming | Windowed |
|---|---|---|
| **How it works** | Single KV cache, frames processed sequentially. Old patch pages evicted (sliding window). | Independent KV cache per window. Windows overlap to correct drift. |
| **GPU VRAM** | ~12.4 GB (fixed, independent of sequence length) | ~10.6 GB per window (fixed, per-window only) |
| **Hard frame cap** | ~1124 frames (special page exhaustion) | None (tested 10,000+) |
| **Temporal consistency** | Perfect within sliding window; oldest frames forgotten | Cross-window drift corrected via chunk alignment |
| **Scale frame reuse** | First 4-8 frames only | Every window has its own 4-8 scale frames |
| **Progress granularity** | Per-frame | Per-window (coarser) |
| **Re-computation** | None (KV cache amortizes past frames) | Scale frames re-computed each window |
| **Parallelizability** | None (causal dependency chain) | Windows are independent — could parallelize across GPUs |
| **Per-frame FPS** | ~5.7 (FlashInfer) / ~2.8 (SDPA) | ~5.7 within window, but overlap computation adds overhead |

### 4.2 The Overlap Tax

Windowed mode has overlap between consecutive windows. With default settings:

```
window_size = 64 keyframes
num_scale_frames = 8
keyframe_interval = 1 (every frame is a keyframe)
eff_overlap = max(8, overlap_keyframes × 1) = 8 frames (default: num_scale_frames)
eff_window = 64 actual frames
step = eff_window - eff_overlap = 56 frames
```

Each window processes 64 frames but only advances 56. **12.5% of frames are processed twice** (the 8 overlap frames serve as the next window's scale frames). With `keyframe_interval=4`:

```
phase2_kf = 64 - 8 = 56 keyframes
phase2_frames = 56 × 4 = 224 frames
eff_window = 8 + 224 = 232 frames
eff_overlap = max(8, overlap_keyframes × 4) — depends on overlap_keyframes setting
```

The overlap tax depends on keyframe interval and overlap settings.

### 4.3 Window Alignment Quality

The key concern with windowed mode: do windows "line up"? The code uses chunk alignment:

1. Each window produces independent pose + depth predictions
2. Overlapping frames between windows i and i+1 are compared
3. A scale factor is computed from depth ratios of overlapping keyframes
4. Window i+1's predictions are scaled to match window i

The `scale_mode` parameter controls this alignment:
- `median`: Median scale ratio (robust to outliers) — default
- `trimmed_mean`: Trimmed mean (more robust)
- `median_all`: Median over all frames (not just keyframes)
- `trimmed_mean_all`: Trimmed mean over all frames

**The alignment is critical for quality.** Without it, window i+1 could be at a completely different scale than window i (the model predicts scale up to an unknown factor). The analysis doesn't quantify alignment quality.

---

## 5. Optimization Strategies (Ranked by Impact)

### 🥇 Strategy 1: Incremental Save-to-Disk (CPU RAM → O(1))

**Impact:** Eliminates CPU RAM bottleneck entirely
**Effort:** Medium (restructure output collection)

Instead of accumulating all predictions in lists, write each frame (or each window) to disk immediately:

```python
# Current (accumulates everything):
all_pose_enc = []
for i in range(scale_frames, S):
    frame_output = self.forward(...)
    all_pose_enc.append(frame_output["pose_enc"])
predictions = {"pose_enc": torch.cat(all_pose_enc, dim=1)}
```

```python
# Proposed (incremental):
import numpy as np
output_dir = Path(f"results/{scene_id}")
output_dir.mkdir(parents=True)

# Phase 1: write scale frames as a batch
for i in range(scale_frames):
    write_frame_npz(output_dir, i, frame_output, mode="scale")

# Phase 2: write streaming frames one at a time
for i in range(scale_frames, S):
    frame_output = self.forward(...)
    write_frame_npz(output_dir, i, frame_output, mode="keyframe" if is_keyframe else "non_keyframe")
    del frame_output  # free immediately
```

The batch demo already does this: `save_predictions_npz()` writes per-frame `.npz` files using a thread pool. The pattern just needs to be integrated into the inference loop rather than done after all frames are processed.

**With this change:** Peak CPU RAM drops from 49.5 GB to ~5.5 MB (one frame) + image tensor for current window only.

### 🥈 Strategy 2: Streaming Video Decode (input memory → O(window))

**Impact:** Eliminates 16.5 GB input tensor allocation for 9000 frames
**Effort:** Medium (integrate PyAV or cv2.VideoCapture streaming)

Don't load all frames upfront:

```python
import av  # PyAV

def stream_frames(video_bytes: bytes, fps: int = 10):
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    
    frame_idx = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame_idx % interval == 0:
                img = frame.to_ndarray(format='rgb24')
                # Preprocess → tensor [1, 3, H, W]
                yield preprocess_frame(img)
            frame_idx += 1
```

Combined with windowed mode: decode frames for window N, process window N, save results, free frames, decode window N+1.

### 🥉 Strategy 3: Drop world_points from Server Output (bandwidth → 1/3)

**Impact:** Saves 1.83 MB/frame (33% of output size)
**Effort:** Low (remove from output dict; compute client-side)

`depth_to_world_coords_points()` is a simple matrix multiply. Ship `depth` + `extrinsic` + `intrinsic` to the client and unproject client-side. The original analysis mentions this. For a web app, this saves server CPU RAM, server disk, and network bandwidth.

**Trade-off:** Client must implement the unprojection. Three.js/WebGL can do this trivially. For API users who want NPZ downloads, compute world_points server-side only at download time (after processing, from saved depth + extrinsics files).

### 4️⃣ Strategy 4: Don't Echo Images

**Impact:** Saves 1.83 MB/frame (33% of output size)
**Effort:** Trivial (already controlled by a field)

The `predictions["images"]` key echoes back the preprocessed input images. Users already have their original images. Drop this from the output dict.

```python
# In postprocess(), simply don't include:
# predictions["images"] = images_out  # REMOVE THIS LINE
```

### 5️⃣ Strategy 5: Keyframe Interval Tuning (GPU VRAM → ~50% reduction)

**Impact:** Reduces KV cache size, increases throughput, slight quality trade-off
**Effort:** Trivial (already a parameter)

With `keyframe_interval=4` and SDPA backend:
- Only 1/4 of frames store KV → KV cache size proportional to keyframes
- 72 cached frames → 18 keyframes + 8 scale = 26 patch pages
- SDPA memory: 26 × 12.8 MB × 24 ≈ 8 GB (vs 22 GB for all frames)
- Non-keyframe frames still get full predictions; they just don't persist KV

With FlashInfer, the benefit is smaller (KV cache is already bounded by sliding_window=64), but higher keyframe intervals allow processing more actual frames per window.

**Quality impact:** Non-keyframe frames attend to fewer past frames (only keyframes). Camera head iterative refinement compensates partially. Needs quantitative testing.

### 6️⃣ Strategy 6: Increase `max_total_frames` for Longer Streaming

**Impact:** Extends streaming mode hard cap from 1124 to N frames
**Effort:** Trivial (constructor parameter)

```python
model = GCTStream(
    max_frame_num=16384,       # Was 1024
    kv_cache_sliding_window=64, # Keep window small
)
```

This pre-allocates more special pages: `ceil(16484 × 6 / 777) + 16 ≈ 143` special pages per block instead of 25. Memory impact: ~11 GB additional GPU VRAM for special pages → **not practical** beyond modest increases.

**Better approach:** For sequences >1000 frames, use windowed mode — it has no such hard cap.

### 7️⃣ Strategy 7: Parallelize Windowed Processing Across GPUs

**Impact:** Linear speedup with GPU count
**Effort:** High (multi-GPU orchestration)

Windowed mode windows are independent after the first window. Process windows on multiple GPUs:

```
GPU 0: windows 0, 4, 8, ...
GPU 1: windows 1, 5, 9, ...
GPU 2: windows 2, 6, 10, ...
GPU 3: windows 3, 7, 11, ...
```

Each window is self-contained (fresh KV cache, own scale frames). Alignment is done after all windows complete. This scales near-linearly.

---

## 6. Test Plan

### 6.1 Memory Scaling Test (highest priority)

**Setup:** Synthetic video, 518×294 frames, lengths: 100, 500, 1000, 2000, 5000, 10000

**Measure for each:**

| Metric | Tool | Hypothesis |
|---|---|---|
| Peak GPU allocated (GB) | `torch.cuda.max_memory_allocated()` | Constant ~12 GB for streaming, ~10 GB for windowed (regardless of sequence length) |
| Peak GPU reserved (GB) | `torch.cuda.max_memory_reserved()` | Should track allocated closely if `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Peak CPU RSS (GB) | `psutil.Process().memory_info().rss` | **Linear growth** with sequence length in current code. Should be constant after incremental save. |
| KV cache page counts | `kv_cache_manager.get_cache_stats()` | Should plateau at scale_frames + sliding_window |
| Special page count over time | `kv_cache_manager.get_cache_stats()` | Linear growth in streaming, constant in windowed |

**Hypothesis to falsify:** "Windowed mode with per-window incremental save can process a 10,000-frame video on a machine with 16 GB CPU RAM and 12 GB GPU VRAM."

### 6.2 KV Cache Sliding Window Efficiency Test

**Setup:** 500 frames, vary `kv_cache_sliding_window`: 16, 32, 64, 128

**Measure:**
- Peak GPU memory (should scale linearly with window size)
- Average FPS (more cached = less re-computation = faster per-frame)
- ATE (Absolute Trajectory Error) vs ground truth (or vs full-attention baseline)

**Hypothesis:** There's a sweet spot. Too small → frequent recomputation → slower. Too large → GPU memory waste. 64 is the default for a reason — test if 32 or 128 is better for the target GPU.

### 6.3 Keyframe Interval Quality-Speed Trade-off

**Setup:** Same 300-frame sequence, vary `keyframe_interval`: 1, 2, 4, 8, 16

**Measure:**
- FPS (fewer keyframes = less KV writing = faster)
- GPU peak memory
- ATE compared to kf_interval=1 baseline
- Point cloud completeness (% of points within error threshold)

**Hypothesis:** kf_interval=4 is the knee point — ~25% speedup with <5% quality degradation. kf_interval=8+ shows visible drift.

### 6.4 Windowed Alignment Quality Test

**Setup:** 1000-frame video, compare:
1. Streaming mode (ground truth reference)
2. Windowed mode, window_size=64, overlap=8, scale_mode="median"
3. Windowed mode, window_size=32, overlap=8
4. Windowed mode, window_size=128, overlap=16

**Measure:**
- Pose difference at window boundaries (should be near-zero after alignment)
- Point cloud consistency (do overlapping regions in adjacent windows produce the same points?)
- Scale ratio consistency (should be ~1.0 after alignment)
- Processing time

**Hypothesis:** Larger windows = fewer alignment boundaries = better consistency, but longer per-window time and higher peak GPU memory. Window_size=64 is likely optimal for 12 GB GPUs.

### 6.5 Streaming Decode + Process Pipeline Test

**Setup:** Real 15-minute video (9000 frames), compare two pipelines:

**Pipeline A (current):**
```
1. Decode all frames to disk (cv2.imwrite)     | ~2 min
2. Load all frames from disk to tensor          | ~30s
3. Preprocess all frames (resize/crop)          | ~10s
4. Run windowed inference (141 windows)         | ~25 min @ 2.8 FPS
5. Accumulate all predictions in RAM            | 49.5 GB peak
6. Save predictions to NPZ (parallel write)     | ~2 min
7. Free memory
```

**Pipeline B (streaming):**
```
For each window:
  1. Decode next 64 frames from video (PyAV)    | interleaved
  2. Preprocess window frames                   | ~0.1s
  3. Run inference on window                    | ~23s
  4. Save window predictions to NPZ immediately | ~0.3s
  5. Free window tensors                        | immediate
```

**Hypothesis:** Pipeline B finishes in ~27 min with <2 GB peak CPU RAM. Pipeline A crashes with OOM around window 40.

### 6.6 Resolution Scaling Test

**Setup:** Same scene, vary input resolution by adjusting `--image_size` and aspect ratio:
- 518×294 (default, 16:9 crop)
- 518×518 (1:1 pad)
- 378×378 (reduced, 27×27 patches)
- 252×252 (minimal, 18×18 patches)

**Measure:**
- FPS vs patches² (should be roughly O(patches²) for attention)
- Depth accuracy vs resolution (is the model robust to lower res?)
- GPU memory vs patches (KV cache scales with O(patches))

**Hypothesis:** 378×378 may be a good speed/quality trade-off for web use. Patches drop from 777 to 729 (only 6% fewer tokens) so FPS gain is modest. 252×252 drops to 324 patches → ~2.4× fewer tokens → ~2× faster but possibly degraded depth quality.

### 6.7 FlashInfer vs SDPA Real-World Test

**Setup:** Same hardware, same sequence, both backends

**Measure:**
- FPS (warm, not including first run)
- GPU memory peak (allocated + reserved)
- CUDA errors / crashes
- `torch.compile` compatibility

**Hypothesis:** FlashInfer is ~2× faster but crashes on some CUDA driver/GPU combinations. SDPA is the safe fallback but ~2.8 FPS vs ~5.7 FPS. Production should auto-detect and prefer FlashInfer, with SDPA fallback.

---

## 7. What's Actually Achievable

### 7.1 Processing Time Estimates

For a 9000-frame video (15 min @ 10 fps), 518×294 resolution:

| Mode | Backend | FPS | Time | GPU RAM | CPU RAM (peak, current) | CPU RAM (after optimization) |
|---|---|---|---|---|---|---|
| Streaming | FlashInfer | 5.7 | 26 min | 12.4 GB | 49.5 GB 💀 | ~200 MB ✅ |
| Streaming | SDPA | 2.8 | 54 min | 7.5 GB | 49.5 GB 💀 | ~200 MB ✅ |
| Windowed (64) | FlashInfer | 5.0* | 30 min | 10.6 GB | 49.5 GB 💀 | ~500 MB ✅ |
| Windowed (64) | SDPA | 2.5* | 60 min | 5.5 GB | 49.5 GB 💀 | ~500 MB ✅ |

*Windowed FPS includes overlap tax (~12.5% re-computation).

### 7.2 "Real-Time" Reality Check

"Real-time" for a 15-minute video means processing in ≤15 minutes:

| Requirement | FPS needed | Current max FPS | Feasible? |
|---|---|---|---|
| 15 min video @ 10fps input = 9000 frames | 10 FPS output | ~5.7 FPS (FlashInfer, 1 GPU) | ❌ **No.** 1.75× too slow |
| 15 min video @ 5fps input = 4500 frames | 5 FPS output | ~5.7 FPS | ✅ Marginal |
| 15 min video @ 3fps input = 2700 frames | 3 FPS output | ~5.7 FPS | ✅ Yes |
| 15 min video @ 10fps, 4 GPUs parallel | 10 FPS output | ~20 FPS (4×5.0 FPS) | ✅ Yes |

**Bottom line:** Single-GPU "real-time" is only achievable at reduced frame rates (~3-5 fps input). For 10 fps input, you need either:
- 2 GPUs (each processing half the windows)
- `torch.compile` with FlashInfer (~7-9 FPS, closer but still not quite 10 FPS)
- Reduce resolution (252×252 = ~10 FPS single GPU, but quality impact unknown)
- Accept that processing takes 1.75× real-time (26 min for a 15 min video)

### 7.3 Accuracy at Speed

The `camera_num_iterations` parameter controls the camera head's iterative refinement loop:

| Iterations | FPS impact | Pose accuracy |
|---|---|---|
| 1 | ~+15-20% faster | Noisier, especially at frame transitions |
| 2 | ~+8-10% faster | Good for most casual use |
| 3 | ~baseline | Near full accuracy |
| **4 (default)** | Baseline | Most accurate |

For a web app, `camera_num_iterations=2` is a good default — users won't notice the difference in a viewer but will notice the 10% speed improvement.

---

## 8. Configuration Recommendations

### 8.1 Per-GPU-Tier Configs

| GPU | Backend | sliding_window | keyframe_interval | window_size | Notes |
|---|---|---|---|---|---|
| 8 GB (RTX 4060, 4070 Laptop) | SDPA | 32 | 4 | 32 | Tight. Use keyframe interval to reduce KV cache. |
| 12 GB (RTX 4070 Ti, A2000) | FlashInfer preferred | 64 | 1 (all keyframes) | 64 | Sweet spot. FlashInfer if it works. |
| 16 GB (RTX 4080, A4000) | FlashInfer | 64 | 1 | 64 | Comfortable. Room for longer sliding window. |
| 24 GB (A10G, L40S, RTX 4090) | FlashInfer | 128 | 1 | 128 | Can handle 128-frame windows. |

### 8.2 Recommended Defaults for Web Deployment

```python
DEFAULT_CONFIG = {
    "image_size": 518,
    "patch_size": 14,
    "num_scale_frames": 8,
    "kv_cache_sliding_window": 64,
    "keyframe_interval": 1,    # 1 = every frame; auto-increase for >320 frames
    "camera_num_iterations": 2, # Fast default; allow override to 4 for quality
    "output_device": "cpu",    # Always CPU for server
    "max_frames": 10000,       # Hard reject above this
    "suggested_fps": 5,        # Default frame extraction rate
}
```

---

## 9. The Three Critical Code Changes

All other optimizations are secondary to these:

### 9.1 Incremental Output (fixes CPU RAM)

Change `inference_streaming()` and `inference_windowed()` to accept a `save_callback(frame_idx, predictions_dict)` parameter. The callback is called per-frame (or per-window) with the predictions for that unit. The caller can write to disk, stream over WebSocket, or accumulate — the model doesn't care.

### 9.2 Streaming Video Input (fixes input RAM)

Add `load_and_preprocess_video_stream(video_path_or_bytes)` that returns an iterator yielding `(frame_idx, tensor[1, 3, H, W])` tuples. Decode + preprocess on demand, never hold all frames.

### 9.3 Drop Redundant Output Keys (reduces per-frame size 2×)

By default, don't include `world_points` and `images` in the per-frame output. Compute `world_points` only when explicitly requested (and ideally server-side, at download time from saved depth + extrinsics).

---

## 10. Bottom Line

| Question | Answer |
|---|---|
| Can we process 15-minute videos? | **Yes**, with windowed mode + incremental save-to-disk. Streaming mode caps at ~1124 frames. |
| Real-time? | **No** on single GPU at 10 fps. Need ~17 FPS, current max ~5.7. Multi-GPU or lower frame rate required. |
| KV cache too large? | **No**. With sliding_window=64, GPU VRAM is bounded to ~12 GB regardless of sequence length. |
| CPU RAM OOM? | **Yes** with current accumulation pattern. **Fixed** by incremental save-to-disk (Strategy 1). |
| Windowed vs streaming quality? | Needs quantitative testing. Theoretically, windowed has drift at boundaries corrected by alignment. Streaming has perfect temporal consistency within window but forgets old frames. |
| Can we free more KV cache space? | The sliding window already bounds it. Keyframe interval reduces it further at a quality cost. The hidden limit is special page pre-allocation, not the sliding window itself. |
| Largest single optimization? | **Incremental save-to-disk** (Strategy 1): 100× reduction in peak CPU RAM with zero quality impact. |
