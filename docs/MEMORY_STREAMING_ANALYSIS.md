# Memory & Streaming Optimization Analysis

> **Status**: Tier 1 ✓ complete · Tier 2 ✓ complete · Tier 3 ✗ (architecturally blocked) · Tier 4 planned

## Final Timing Summary (286-frame courthouse, RTX 4070 Laptop)

| Metric | Original | Tier 1 | Tier 1+2 (lazy) |
|---|---|---|---|
| Image load | 1.2 s | 1.2 s | 2.3 s |
| Inference | 86.1 s | 86.1 s | 85.9 s |
| GPU peak | 5.08 GB | 5.08 GB | 5.08 GB |
| Image CPU RAM | 498 MB | 498 MB | ~28 MB |
| NPZ size | 166.5 MB | 50.5 MB (−70%) | 50.5 MB (−70%) |
| Total time | ~90 s | ~90 s | 89.7 s (+0.8%) |
| Quality vs backup | baseline | ✓ all frames pass | ✓ all frames pass |

## Tier 1 — Incremental NPZ Export ✓

Per-window callback in `inference_windowed()` saves raw predictions to disk as each window completes. Final aligned predictions overwrite at end. Survives crashes.

- `gct_stream_window.py`: Added `per_window_callback(w_dict, start, end)` parameter
- `demo.py`: `_on_window_complete()` saves per-window NPZ with dtype optimization
- NPZ compression: `np.savez_compressed` + float16 depth + uint8 depth_conf
- **Result**: 166.5 MB → 50.5 MB (−69.6%), depth lossless, depth_conf ±0.5 max error

## Tier 2 — Lazy Image Loading (memmap) ✓

`--lazy_images` flag loads images via `np.memmap` backed by a temp file. The OS pages frames in/out as each window accesses them. Peak CPU RAM drops from O(num_frames) to O(window_size). Adds <1% overhead.

## Tier 3 — Overlap Recycling ✗ (architecturally blocked)

**Attempted**: Reuse overlap predictions from window N as scale frames for window N+1, skipping the scale forward pass. Achieved 12% inference speedup (86.1 → 75.4s).

**Why it failed**: The inter-window alignment algorithm (`_align_and_stitch_windows`) compares overlap predictions between adjacent windows to compute scale correction transforms. When overlap predictions are reused (identical in both windows), the alignment computes identity transforms, failing to correct for drift between windows. This caused depth errors in all frames beyond the first window.

**Infrastructure kept**: `AggregatorStream.snapshot_kv_cache()`, `restore_kv_cache()`, `trim_kv_cache()` methods remain in the codebase for future use.

**Path forward**: Overlap recycling requires either:
1. **Deferred alignment**: Save raw window predictions, apply alignment at load time (requires alignment metadata to be persisted separately)
2. **KV-only reuse**: Reuse KV cache entries but still run the scale forward pass (no compute savings, but better context for scale frames)
3. **Model architecture change**: Decouple alignment from overlap comparisons

## Tier 4 — Progressive Visualization (planned)

- After each window: `viser_viewer.add_frames(global_idx_start, window_frames)`
- User sees point cloud growing in real-time
- Hooks into existing `per_window_callback` infrastructure

## Why SDPA + Windowed Mode

| Backend | Min KV overhead | Scaling | Fits 8.6 GB? |
|---|---|---|---|
| FlashInfer | 4.6 GB (fixed special-page pool) | Moderate | ✗ (starts at 8.5 GB) |
| SDPA | 0 GB | Linear | ✓ (5.3 GB at window=16) |

Windowed mode bounds the KV cache to `window_size + scale_frames` regardless of sequence length.

### Max Window Size (SDPA, RTX 4070 Laptop)

| window_size | KV cache | Total GPU | Fits? |
|---|---|---|---|
| 16 (current) | 1.5 GB | 5.3 GB | ✓ |
| 32 | 2.8 GB | 6.6 GB | ✓ |
| 48 | 4.0 GB | 7.8 GB | ✓ tight |
| 64 | 5.2 GB | 9.0 GB | ✗ OOM |

Lazy loading (Tier 2) saves CPU RAM, not GPU VRAM — window_size headroom is unchanged.
