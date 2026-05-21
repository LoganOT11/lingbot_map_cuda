#!/usr/bin/env python3
"""Compare two NPZ exports frame-by-frame.  Useful for quantifying the impact
of model changes, preprocessing tweaks, or inference parameter ablations.

Usage:
    python scripts/compare_npz.py outputs/baseline/ outputs/experiment/
    python scripts/compare_npz.py outputs/baseline/ outputs/experiment/ --metric depth
"""
import argparse, glob, os, sys
import numpy as np
from tqdm import tqdm

def load_frame(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in d.files:
        arr = d[k]
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        out[k] = arr
    return out

def main():
    p = argparse.ArgumentParser(description="Compare two per-frame NPZ exports")
    p.add_argument("dir_a", help="Baseline NPZ directory")
    p.add_argument("dir_b", help="Experiment NPZ directory")
    p.add_argument("--metric", choices=["all", "depth", "extrinsic", "intrinsic"],
                   default="all", help="Which prediction key to compare (default: all)")
    p.add_argument("--frames", type=str, default=None,
                   help="Frame range: start:end (e.g. 0:50)")
    args = p.parse_args()

    frames_a = sorted(glob.glob(os.path.join(args.dir_a, "frame_*.npz")))
    frames_b = sorted(glob.glob(os.path.join(args.dir_b, "frame_*.npz")))

    if len(frames_a) != len(frames_b):
        print(f"WARNING: frame count mismatch — {len(frames_a)} vs {len(frames_b)}")
        n = min(len(frames_a), len(frames_b))
        frames_a, frames_b = frames_a[:n], frames_b[:n]
    else:
        n = len(frames_a)

    if args.frames:
        parts = args.frames.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else n
        frames_a = frames_a[start:end]
        frames_b = frames_b[start:end]
        n = len(frames_a)

    keys = ["depth", "extrinsic", "intrinsic"] if args.metric == "all" else [args.metric]

    print(f"Comparing {n} frames × {len(keys)} keys")
    print(f"  A (baseline): {args.dir_a}")
    print(f"  B (experiment): {args.dir_b}")
    print()

    # Per-key accumulators
    for key in keys:
        abs_diffs = []
        rel_diffs = []
        max_abs = 0.0
        max_frame = 0

        for i in tqdm(range(n), desc=f"  {key}", unit="frame"):
            da = load_frame(frames_a[i])
            db = load_frame(frames_b[i])

            if key not in da or key not in db:
                print(f"  WARNING: key '{key}' missing at frame {i}")
                continue

            a, b = da[key], db[key]
            if a.shape != b.shape:
                print(f"  WARNING: shape mismatch at frame {i}: {a.shape} vs {b.shape}")
                continue

            abs_diff = np.abs(a - b)
            abs_mean = float(np.mean(abs_diff))
            abs_max = float(np.max(abs_diff))
            abs_diffs.append(abs_mean)

            # Relative diff (avoid div-by-zero)
            denom = np.maximum(np.abs(a), 1e-8)
            rel = np.mean(np.abs(a - b) / denom)
            rel_diffs.append(float(rel))

            if abs_max > max_abs:
                max_abs = abs_max
                max_frame = i

        abs_diffs = np.array(abs_diffs)
        rel_diffs = np.array(rel_diffs)

        print(f"  {key}:")
        print(f"    Mean  |Δ|:  {abs_diffs.mean():.6f}  (±{abs_diffs.std():.6f})")
        print(f"    Max   |Δ|:  {max_abs:.6f}  (frame {max_frame})")
        print(f"    Mean  |Δ|%: {rel_diffs.mean()*100:.4f}% (±{rel_diffs.std()*100:.4f}%)")
        print(f"    Max   |Δ|%: {rel_diffs.max()*100:.4f}%  (frame {rel_diffs.argmax()})")

        # Per-frame quartiles
        q25, q50, q75 = np.percentile(abs_diffs, [25, 50, 75])
        print(f"    |Δ| quartiles:  Q1={q25:.6f}  median={q50:.6f}  Q3={q75:.6f}")
        print()

    # Frame-count summary
    print(f"Frames compared: {n}")
    print(f"Output keys available in A: {sorted(load_frame(frames_a[0]).keys())}")
    print(f"Output keys available in B: {sorted(load_frame(frames_b[0]).keys())}")


if __name__ == "__main__":
    main()
