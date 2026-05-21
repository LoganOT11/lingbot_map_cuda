# Agent Prompt — LingBot-MAP Tier 3/4 Path Forward

> Generated 2026-05-21 after completing Tiers 1-2. Tier 3 (overlap recycling) was attempted and reverted due to an architectural conflict with the inter-window alignment algorithm.

## Current State

### Completed (Tiers 1-2)

| Tier | What | Status |
|---|---|---|
| NPZ compression | `np.savez_compressed` + float16 depth + uint8 depth_conf | ✓ 166→50 MB (−70%) |
| Tier 1 | Per-window callback → incremental NPZ save during inference | ✓ Survives crashes |
| Tier 2 | `--lazy_images` memmap loading → O(window) RAM for images | ✓ <1% overhead |

### Codebase map (files you'll touch)

```
lingbot_map_cuda/
├── apps/cli/demo.py                    # Main pipeline (CLI flags, _run_single_scene)
│   ├── _on_window_complete()           # Tier 1: per-window NPZ callback
│   ├── _setup_incremental_npz()        # Prepares output directory
│   ├── _finalize_incremental_npz()     # Overwrites with aligned predictions
│   └── _load_images_lazy()             # Tier 2: memmap-backed image loading
├── lingbot_map/models/
│   └── gct_stream_window.py            # inference_windowed() with per_window_callback
│       └── _align_and_stitch_windows() # THE KEY FUNCTION — computes alignment
├── lingbot_map/aggregator/
│   └── stream.py                       # AggregatorStream with SDPA KV cache
│       ├── snapshot_kv_cache()          # INFRA: deep-copy KV cache (5D tensors)
│       ├── restore_kv_cache(snapshot)   # INFRA: restore saved KV cache
│       ├── trim_kv_cache(n_frames, tpf) # INFRA: keep last N frames in cache
│       └── clean_kv_cache()            # Reset to empty
├── lingbot_map/layers/
│   └── attention.py                    # SDPAAttention — KV cache shape details
│       # KV shape: [B, num_heads, num_frames, tokens_per_frame, head_dim]
│       # Special KV: [B, num_heads, num_evicted_frames, num_special, head_dim]
├── lingbot_map/vis/
│   └── point_cloud_viewer.py           # Viser 3D viewer (for Tier 4)
│       # ViserPointCloudViewer.add_frames() can accept incremental point clouds
├── docs/
│   └── MEMORY_STREAMING_ANALYSIS.md    # Full analysis document
└── example/
    └── courthouse_npz_backup/          # 286-frame ground truth (166 MB original format)
```

### Key parameters (286-frame courthouse)
- Resolution: 518×294, patch_size=14 → 777 patches + 6 special tokens = 783 tokens/frame
- window_size=16 keyframes, num_scale_frames=4, overlap=4 frames
- 24 windows per 286-frame scene
- GPU: RTX 4070 Laptop (8.6 GB), SDPA backend (KV cache ~1.5 GB at window=16)
- Inference time: ~86s, NPZ size: 50.5 MB

---

## Path A — Deferred Alignment (Recommended)

### The problem Tier 3 hit
`_align_and_stitch_windows()` compares overlap predictions between adjacent windows to compute scale/rotation correction transforms. When overlap predictions are reused (identical), alignment computes identity transforms and can't correct inter-window drift.

### Solution: Align on load, not during inference
Instead of applying alignment during `inference_windowed()`, save EACH window's raw predictions to per-window NPZ files PLUS alignment metadata. Apply alignment when reading the NPZ files (in the viewer/renderer).

### Implementation plan

**Step 1 — Per-window NPZ format**

Create a new NPZ layout:
```
output_dir/
├── window_000/
│   ├── frame_000000.npz
│   ├── frame_000001.npz
│   └── ... (unaligned frames for this window)
├── window_001/
│   └── ...
├── alignment.npz    # chunk_scales [num_windows], chunk_transforms [num_windows, 4, 4]
└── meta.json
```

The `per_window_callback` already fires with `(w_pred, start, end)`. Save each window's frames to its own subdirectory as they complete.

**Step 2 — Save alignment metadata separately**

After `_align_and_stitch_windows()` computes `chunk_scales` and `chunk_transforms`, save them to `alignment.npz`. The per-window frames remain UNALIGNED on disk.

**Step 3 — Apply alignment on load**

Modify `_load_predictions_from_npz()` (demo.py line ~870) and `load_npz_data()` (rgbd_render/data/loader.py line ~140) to:
1. Load per-window frames
2. Load alignment metadata  
3. Apply `_warp_predictions(R, t, s)` per window before stacking
4. Return aligned predictions (same format as today)

The alignment math already exists in `_align_and_stitch_windows` and `_warp_predictions` (gct_stream_window.py line ~960). Extract it into a reusable utility.

**Step 4 — Enable overlap recycling**

With alignment deferred to load time, `inference_windowed()` no longer needs to align windows. This means:
- After window N completes, trim KV cache to `eff_overlap` frames using the existing `trim_kv_cache()` infrastructure
- Window N+1 restores trimmed cache, skips scale forward pass
- Overlap predictions from window N replace scale phase output
- Since alignment happens later, identical overlap predictions are fine

Expected savings: ~12% inference time (75s vs 86s), same quality.

### Key functions to extract/create

```python
# New utility in lingbot_map/utils/alignment.py
def compute_alignment(window_preds: list[dict], overlap: int) -> tuple:
    """Returns (scales, transforms) — extracted from _align_and_stitch_windows."""
    ...

def apply_alignment(frames: np.ndarray, scales, transforms, window_idx: int) -> np.ndarray:
    """Applies per-window alignment to depth/pose arrays."""
    ...
```

---

## Path B — Progressive Visualization (Tier 4)

### Goal
User sees point cloud accumulate in real-time as each window completes, instead of waiting 60+ seconds for all frames.

### Implementation plan

**Step 1 — Incremental viewer update callback**

The `per_window_callback` already fires with `(w_pred, start, end)`. Extend it or add a second callback:
```python
def _on_window_for_viewer(w_pred, start, end, viewer_state):
    """Post-process window frames and send to viser viewer."""
    frames = _postprocess_window(w_pred)
    viewer_state['viewer'].add_frames(frames, start, end)
```

The viewer's `add_frames()` method already exists (point_cloud_viewer.py). It needs per-frame: world_points (H×W×3), colors (H×W×3), confidence (H×W).

**Step 2 — Deferred viewer launch**

Currently the viewer launches AFTER all inference completes. Change demo.py `_run_single_scene` to:
1. Launch viewer early (empty scene)
2. Pass viewer handle into inference callback
3. Update viewer per window
4. Keep viewer alive after inference

**Step 3 — Alignment for progressive display**

Since alignment isn't known until all windows complete, progressive display shows UNALIGNED point clouds. Options:
- Accept minor visual seams between windows (they're typically invisible for adjacent windows)
- Apply approximate alignment using overlap region (heuristic)
- Defer to Path A (deferred alignment) — viewer shows aligned chunks as alignment becomes available

---

## Path C — Larger Window Size via FlashInfer Coaching

The GPU can't run FlashInfer currently (8.6 GB < 8.5 GB minimum). But window_size=32 with SDPA would use 6.6 GB (feasible). Going to 48 uses 7.8 GB (tight).

### Experiment
Run with `--window_size 32` and benchmark quality vs window_size=16:
```bash
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 32 --num_scale_frames 4 --overlap_size 4 --headless \
    --save_predictions outputs/ws32/
```

Fewer windows = less alignment overhead = possibly faster despite more KV cache per window. Compare NPZ diff against backup:
```bash
python scripts/compare_npz.py example/courthouse_npz_backup/ outputs/ws32/
```

---

## How to Test (immutable workflow)

Each change must pass against `example/courthouse_npz_backup/`:

```bash
# 1. Run inference with changes
rm -rf example/courthouse_npz
python apps/cli/demo.py --model_path models/lingbot-map-long.pt \
    --image_folder example/courthouse --use_sdpa --mode windowed \
    --window_size 16 --overlap_size 4 --num_scale_frames 4 --headless \
    --save_predictions example/courthouse_npz

# 2. Compare against ground truth
python -c "
import numpy as np, os
d_new='example/courthouse_npz'; d_old='example/courthouse_npz_backup'
fn=sorted([f for f in os.listdir(d_new) if f.startswith('frame_')])
fo=sorted([f for f in os.listdir(d_old) if f.startswith('frame_')])
for i in [0,50,100,150,200,250,285]:
    n=np.load(f'{d_new}/{fn[i]}'); o=np.load(f'{d_old}/{fo[i]}')
    dok=np.allclose(n['depth'].astype('f4'),o['depth'].astype('f4'),rtol=0,atol=0)
    dce=np.abs(n['depth_conf'].astype('f4')-o['depth_conf'].astype('f4')).max()
    eok=np.allclose(n['extrinsic'],o['extrinsic'],rtol=1e-5)
    iok=np.allclose(n['intrinsic'],o['intrinsic'],rtol=1e-5)
    ok=dok and eok and iok and dce<=0.5
    print(f'frame_{i}: {\"OK\" if ok else \"FAIL\"} depth={dok} conf={dce:.3f} extr={eok} intr={iok}')
"
# All frames must show OK
```

---

## Constraints & Gotchas

1. **SDPA KV cache shape**: 5D tensors `[B, num_heads, num_frames, tokens_per_frame, head_dim]` for regular, `[B, H, num_frames, num_special, D]` for special tokens. The `trim_kv_cache()` method handles this correctly.

2. **Quality is non-negotiable**: Depth must be bit-identical against backup. depth_conf ±0.5. Extrinsic/intrinsic rtol=1e-5. If any frame fails, the change is invalid.

3. **Alignment is the bottleneck for Tier 3**: `_align_and_stitch_windows()` requires DIFFERENT overlap predictions to compute correction transforms. Any approach that makes them identical breaks quality. Deferred alignment (Path A) is the solution.

4. **Model loading takes ~20s**: Factor this into timing comparisons. Only compare inference time, not total scene time including model load.

5. **GPU throttles on laptop**: Run-to-run variance of ±3s is normal. Compare averages of 2-3 runs for reliable timing.

6. **The backup NPZ format**: 166 MB, float16 depth, float16 depth_conf, float32 extrinsic/intrinsic, no compression. New format: 50 MB, float16 depth, uint8 depth_conf, compressed.
