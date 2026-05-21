# Memory & Streaming Optimization Analysis

> **Status**: Tier 1 ✓ complete · Tier 2 ✓ complete · Tier 3-4 planned

## Current Architecture (demo.py path)

```
load_images() → [S, 3, H, W]     ← ALL frames in CPU RAM (498 MB for 286)
     ↓
model.inference_windowed()       ← ALL windows processed, returns FULL dict
     ↓                              Overlap frames computed TWICE
_postprocess(all_frames)         ← ALL predictions in RAM (332 MB)
     ↓
_save_predictions_npz(all)       ← ALL frames saved at once
```

**Peak memory**: ~830 MB CPU + ~5.1 GB GPU (SDPA). Frames only hit disk after 100% of inference completes.

## Tier 1 — Incremental NPZ Export ✓

**Implemented**: Per-window callback in `inference_windowed()` saves raw predictions to disk as each window completes. Final aligned predictions overwrite at end. Survives crashes.

- `gct_stream_window.py`: Added `per_window_callback(w_dict, start, end)` parameter
- `demo.py`: `_on_window_complete()` saves per-window NPZ with dtype optimization
- NPZ compression: `np.savez_compressed` + float16 depth + uint8 depth_conf
- **Result**: 166.5 MB → 50.5 MB (−69.6%), depth lossless, depth_conf ±0.5 max error

## Tier 2 — Lazy Image Loading (memmap) ✓

**Implemented**: `--lazy_images` flag loads images via `np.memmap` backed by a temp file. The OS pages frames in/out as each window accesses them. Peak CPU RAM for images drops from O(num_frames) to O(window_size).

### Timing Comparison (286-frame courthouse, RTX 4070 Laptop)

| Metric | Regular (Tier 1) | Lazy (Tier 2) | Δ |
|---|---|---|---|
| Image load | 1.2 s | 2.3 s | +1.1 s |
| Inference | 86.1 s | 85.9 s | −0.2 s (noise) |
| GPU peak | 5.08 GB | 5.08 GB | 0 |
| **Image CPU RAM** | **498 MB** | **~28 MB** (O(window)) | **−470 MB** |
| Total scene time | ~89 s | 89.7 s | +0.7 s (+0.8%) |
| NPZ size | 50.5 MB | 50.5 MB | 0 |
| Output quality vs backup | ✓ passes | ✓ passes | identical |

**Key finding**: Lazy loading adds negligible overhead (+0.8% total time) while eliminating the per-frame RAM cost. Critical for sequences with thousands of frames where pre-loading all images is impossible.

### How It Works

```
_load_images_lazy()               ← Preprocess images in batches → np.memmap (disk)
     ↓                              Returns torch.from_numpy(mmap) — looks like real tensor
model.inference_windowed()       ← Accesses images[:, start:end]
     ↓                              → OS faults in only those pages
     ↓                              → After window done, pages can be evicted
     ↓                              Peak RAM: O(window_size) ≈ 28 MB for images
```

The memmap tensor supports `.shape`, `.device`, `.pin_memory()` — the model treats it exactly like a regular tensor. No model changes needed.

## Tier 3 — Overlap Recycling & KV Reuse (planned)

- Cache overlap predictions from window N, reuse in window N+1
- Saves ~20% inference compute (4/20 frames per window)
- KV cache reuse for overlap region (architecturally complex)

## Tier 4 — Progressive Visualization (planned)

- After each window: `viser_viewer.add_frames(global_idx_start, window_frames)`
- User sees point cloud growing in real-time
- Hooks into existing `per_window_callback` infrastructure

## Why SDPA + Windowed Mode

| Backend | Min KV overhead | Scaling | Fits 8.6 GB? |
|---|---|---|---|
| FlashInfer | 4.6 GB (fixed special-page pool) | Moderate | ✗ (starts at 8.5 GB) |
| SDPA | 0 GB | Linear | ✓ (5.3 GB at window=16) |

Windowed mode bounds the KV cache to `window_size + scale_frames` regardless of sequence length. Streaming mode would keep all frames in cache → OOM for 286 frames.

### Max Window Size (SDPA, RTX 4070 Laptop)

| window_size | KV cache | Total GPU | Fits? |
|---|---|---|---|
| 16 (current) | 1.5 GB | 5.3 GB | ✓ |
| 32 | 2.8 GB | 6.6 GB | ✓ |
| 48 | 4.0 GB | 7.8 GB | ✓ tight |
| 64 | 5.2 GB | 9.0 GB | ✗ OOM |

Lazy loading does **not** increase window_size headroom — it saves CPU RAM, not GPU VRAM. To use larger windows, a GPU with ≥12 GB VRAM is needed.
