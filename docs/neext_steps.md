**LingBot-MAP Visualization Pipeline**

Technical Analysis, Research Review & Next-Steps Roadmap

Date: May 20, 2026

# **1\. How the Visualization Is Produced**

The LingBot-MAP visualization pipeline is a multi-stage process that transforms raw video input into an interactive 3D scene. Each stage has distinct data flow, GPU requirements, and output artifacts.

## **1.1 Inference: Depth & Pose Estimation**

The pipeline begins with the LingBot-MAP model (GCTStream), a transformer-based architecture built on a DINOv2 ViT-L backbone with 24 attention blocks. Given a sequence of video frames, the model outputs:

- depth\[S, H, W, 1\] - metric depth maps, one per frame (609 KB/frame at 518x294)
- depth_conf\[S, H, W\] - per-pixel confidence scores
- pose_enc\[S, 9\] - compact camera pose encoding (extrinsics + intrinsics)

The model operates in two inference modes:

- Streaming mode: Processes frames one-by-one with a sliding KV cache window (default: 64 frames). Hard cap at ~1,124 frames per sequence.
- Windowed mode: Processes overlapping windows independently, then aligns them using a median scale correction. No frame limit - suitable for long videos.

GPU memory at 518x294 resolution with FlashInfer backend:

| **Component**    | **Streaming**      | **Windowed** |
| ---------------- | ------------------ | ------------ |
| Model weights    | 2.8 GB             | 2.8 GB       |
| Patch KV pages   | 6.7 GB             | 6.7 GB       |
| Special KV pages | 1.9 GB (pre-alloc) | ~0.08 GB     |
| Peak activations | ~1.0 GB            | ~1.0 GB      |
| Total peak GPU   | ~12.4 GB           | ~10.6 GB     |

## **1.2 Point Cloud Generation (Post-Inference)**

The model does NOT output world-space 3D points directly. Point clouds are always computed in post-processing via an unprojection pipeline:

For each frame i, the unprojection math is:

- Step 1 - Pixel to camera coords: x_cam = (u - cx) \* d / fx, y_cam = (v - cy) \* d / fy, z_cam = d
- Step 2 - Camera to world coords: world_xyz = R^-1 · \[x_cam, y_cam, z_cam\] + (-R^-1 · t)

This produces ~90,000-120,000 valid 3D points per frame (out of 152,292 pixels at 518x294). For a 15-minute video at 10 fps (9,000 frames), the raw unfiltered output is ~960 million points - far too large for any web viewer.

The pipeline applies multiple reduction strategies in sequence:

| **Strategy**                  | **Reduction Factor** | **Notes**                                        |
| ----------------------------- | -------------------- | ------------------------------------------------ |
| Keyframes only (interval=4)   | 4x                   | Only unproject depth for KV-cached keyframes     |
| Spatial downsample (stride=4) | 16x                  | Sample every 4th pixel in both dimensions        |
| Confidence filter (top 70%)   | 1.4x                 | Drop low-confidence sky/specular points          |
| Voxel grid (2 cm cells)       | ~3x                  | Spatially deduplicate overlapping frame coverage |
| Combined effect               | ~70x                 | ~14M points for a 15-min video - web-feasible    |

## **1.3 Gaussian Splatting (3DGS) Training**

LingBot-MAP is an ideal input for 3D Gaussian Splatting because it provides dense depth maps for Gaussian initialization - far superior to the sparse point clouds produced by traditional COLMAP pipelines.

| **Pipeline**         | **Initialization**                           | **Iterations** | **Training Time (RTX 4090)** |
| -------------------- | -------------------------------------------- | -------------- | ---------------------------- |
| Traditional (COLMAP) | Sparse SfM points (100s)                     | 30,000         | ~20 min                      |
| LingBot-MAP + 3DGS   | Dense depth-initialized points (100K+/frame) | 7,000-15,000   | ~5-10 min                    |

After training, Gaussians are exported as a compressed .ply.gz file (~50-100 MB for 5M Gaussians) and served to the browser for real-time rendering at 60 FPS.

## **1.4 Semantic Segmentation**

Segmentation runs independently of the depth/pose model and can be parallelized on a separate GPU or time-sliced. The current implementation uses:

- Sky segmentation: An ONNX model (skyseg.onnx, 176 MB) that produces soft sky/non-sky masks per frame. Used to filter sky points from the point cloud.
- SAM 2.1 (planned): Meta's Segment Anything Model for general-purpose object segmentation. Supports both automatic (grid prompts) and interactive (click-to-segment) modes.

Each 2D segmentation mask (~150 KB) is lifted to 3D by associating it with the corresponding depth-unprojected world points, producing a semantically labeled point cloud exportable as a GLB file.

## **1.5 Web Visualization Stack**

The current visualization tooling (Viser, Open3D, GLB exporter) is Python-based and not embeddable in a web app. The target production viewer combines:

- Gaussian Splat viewer: gsplat.js (antimatter15/splat) for real-time WebGL rendering of the .ply.gz output
- Point cloud viewer: Three.js with progressive GLB loading (preview at 1M points, full at 10M)
- FastAPI backend: Serves compressed scene artifacts and WebSocket streaming of inference progress
- Progressive loading strategy: Immediate point cloud preview (5s) → GS warmup → full GS scene (10-35 min)

## **1.6 End-to-End Pipeline Summary**

| **Stage**              | **Input**           | **Output**                                   | **Latency (3-min video)** |
| ---------------------- | ------------------- | -------------------------------------------- | ------------------------- |
| Video decode           | MP4/video bytes     | Frame tensors (5 fps)                        | ~15s                      |
| LingBot-MAP inference  | Frame tensors       | depth + pose NPZ (~350 MB)                   | ~6 min                    |
| Point cloud generation | depth + extrinsics  | Preview GLB (15 MB) / Full GLB (35 MB Draco) | ~30s                      |
| Sky segmentation       | RGB frames          | Per-frame masks (~50 MB total)               | ~2 min (parallel)         |
| 3DGS training          | Images + dense init | scene.ply.gz (~60 MB)                        | ~12 min                   |
| Total (single GPU)     | Raw video           | Interactive 3D scene                         | ~20 min                   |

# **2\. Relevant Research Papers & Open-Source Projects**

The following research and tools have direct applicability to optimizing each stage of the LingBot-MAP visualization pipeline. They are grouped by the pipeline stage they address.

## **2.1 Gaussian Splatting: Training Speed & Memory**

### **FastGS - Training 3DGS in 100 Seconds (CVPR 2026 Highlight)**

FastGS (Ren et al., 2025, arXiv:2511.04283) achieves a 3.32x training acceleration over the original 3DGS implementation, with comparable rendering quality on Mip-NeRF 360, Tanks & Temples, and Deep Blending benchmarks. It accomplishes this through smarter densification budgeting and pruning strategies that reduce redundant Gaussians without sacrificing visual fidelity.

- GitHub: github.com/fastgs/FastGS
- Relevance: Directly reduces GS training time from 12 min to ~4 min for medium scenes, making on-demand processing per user upload far more practical.

### **Faster-GS - Up to 5x Training Acceleration (Feb 2026)**

Faster-GS (arXiv:2602.09999) consolidates the most effective optimization strategies from prior 3DGS research and adds novel optimizations that exploit memory coalescence and fuse gradient computations. It establishes a new cost-effective baseline for 3DGS training.

- Relevance: Plug-and-play integration into an existing 3DGS training loop. Provides a testbed for fair comparison of optimization techniques.

### **GS-Scale - Memory-Efficient 3DGS via CPU Offloading**

GS-Scale (arXiv:2509.15645) enables high-quality 3DGS training on single consumer-grade GPUs by offloading Gaussian parameters to CPU host memory. It targets the Gaussian-related components (not activations) since only frustum-visible Gaussians participate in each forward pass.

- Relevance: Critical for supporting scenes that generate >5M Gaussians on 12-16 GB GPUs. Allows training of large courthouse-scale scenes (8M+ Gaussians) on an RTX 4070 Ti.

### **gsplat - Open-Source 3DGS Library (Apache 2.0)**

gsplat (Ye et al., JMLR 2025) is the reference open-source implementation for Gaussian Splatting, featuring Python/PyTorch bindings with highly optimized CUDA kernels. It achieves up to 10% less training time and 4x less memory than the original Kerbl et al. implementation.

- GitHub: github.com/nerfstudio-project/gsplat
- Relevance: Recommended backend for LingBot-MAP GS training (trainer.py). Actively maintained, Apache 2.0 licensed, integrates directly with depth-initialized Gaussian point clouds.

### **gsplat.js - Browser-Side Gaussian Splatting Viewer**

gsplat.js (Hugging Face) is a WebGL-based Gaussian Splatting renderer designed as a Three.js analog for .ply/.splat files. It supports loading scenes directly from URLs and renders up to ~2M Gaussians in real time.

- NPM: npmjs.com/package/gsplat
- Relevance: The recommended client-side renderer for serving the .ply.gz output from the LingBot-MAP GS pipeline. The compact .splat format (binary conversion of .ply) loads faster.

## **2.2 Semantic Segmentation & Scene Understanding**

### **OpenWorldSAM - SAM2 with Language Prompts (NeurIPS 2025 Spotlight)**

OpenWorldSAM (Xiao et al., arXiv:2507.05427) extends SAM2 to open-vocabulary scenarios by integrating multi-modal embeddings from a lightweight vision-language model. It trains only 4.5M parameters (frozen SAM2 + VLM) and achieves state-of-the-art zero-shot performance on unseen categories across semantic, instance, and panoptic segmentation benchmarks.

- GitHub: github.com/GinnyXiao/OpenWorldSAM
- Relevance: Replaces the need to pre-define segmentation classes. Users could type a text query (e.g., 'find all windows') and the model segments the scene accordingly - both in 2D frames and, via depth lifting, in 3D.

### **Semantic Gaussians - Open-Vocabulary 3D Scene Understanding**

Semantic Gaussians (Guo et al., arXiv:2403.15624) distills 2D pre-trained vision-language features (CLIP, OpenSeg) into 3D Gaussian primitives with no additional training. Each Gaussian gains a semantic embedding alongside its geometric properties, enabling text-query-driven 3D segmentation, scene editing, and spatiotemporal understanding.

- GitHub: github.com/sharinka0715/semantic-gaussians
- Relevance: Allows LingBot-MAP's GS scenes to be semantically queryable after training - users could click or type to isolate any object class in the 3D scene. Eliminates the need for a separate segmentation pass by embedding semantics during GS training.

### **Language Embedded 3D Gaussians (LEGaussians, IEEE TPAMI 2025)**

LEGaussians (Wang et al., 2025) introduces a memory-efficient representation for embedding language features into 3DGS scenes, addressing the prohibitive memory usage of naive feature embedding. It outperforms prior methods in rendering quality and semantic query accuracy while requiring significantly less GPU and disk storage.

- Project: buaavrcg.github.io/LEGaussians
- Relevance: If Semantic Gaussians proves memory-prohibitive for large LingBot-MAP scenes, LEGaussians provides a more efficient alternative with similar open-vocabulary querying capabilities.

## **2.3 Inference Acceleration**

### **FlashInfer - Paged KV Cache Attention (MLSys 2025 Best Paper)**

FlashInfer (arXiv:2501.01005) won Best Paper at MLSys 2025. It provides optimized attention kernels for paged and ragged KV caches, CUDAGraph and torch.compile compatibility, and dynamic scheduling for variable-length sequences. NVIDIA is now releasing its most performant inference kernels through the FlashInfer project.

- GitHub: github.com/flashinfer-ai/flashinfer
- Relevance: LingBot-MAP already uses FlashInfer for its paged KV cache. The ~2x throughput gain over SDPA is the difference between 5.7 FPS and 2.8 FPS - critical for meeting user latency expectations. The MLSys 2025 award signals continued active development and GPU driver support improvements.

### **torch.compile - PyTorch 2.7 Production Optimizer**

As of PyTorch 2.7, torch.compile delivers 1.5-2x speedups for inference workloads through kernel fusion, operator elimination, and specialized kernel selection. It is now compatible with CUDAGraph and FlashInfer (from FlashInfer v0.2+), enabling combined optimization of LingBot-MAP's inference loop.

- Relevance: Applying torch.compile to the GCTStream model's forward pass could push single-GPU throughput from ~5.7 FPS to ~7-9 FPS, narrowing the gap to real-time for 5 fps input video.

## **2.4 Web Point Cloud Streaming**

### **Potree - WebGL Point Cloud Renderer for Large Datasets**

Potree (TU Wien) is a free open-source WebGL point cloud renderer that handles arbitrarily large point clouds via octree-based level-of-detail streaming. PotreeConverter converts LAS/LAZ/PLY files into a streaming octree format that progressively loads only the points visible in the current viewport.

- GitHub: github.com/potree/potree
- Relevance: For scenes where the 14M-point compressed GLB is still too large for immediate load, Potree's progressive octree streaming provides a path to rendering hundreds of millions of points in a browser. The LingBot-MAP GLB exporter output can be converted via PotreeConverter.

### **3D Tiles - OGC Standard for Streaming 3D Geospatial Content**

3D Tiles is an OGC Community Standard for streaming massive heterogeneous 3D geospatial datasets (point clouds, meshes, building models). It uses a spatial hierarchy with LOD to stream only the tiles needed for the current view, and is supported natively by CesiumJS and deck.gl.

- Relevance: The highest-quality progressive loading option for large LingBot-MAP outdoor scenes. The conversion pipeline (GLB/LAS → 3D Tiles) is well-supported by tools like Cesium ion CLI and py3dtiles.

# **3\. Next Steps: Implementation Roadmap**

The roadmap is organized into four phases, each building on the previous. Phases 1 and 2 are prerequisites for production. Phases 3 and 4 can run in parallel and represent the full production target.

## **Phase 0: Pre-Flight & Safety Fixes (1-2 days) - IMMEDIATE**

These are blocking bugs and missing safety guards that must be addressed before any new feature work.

- Wrap \_set_skip_append in try/finally in GCTStream (5-line fix). If an exception occurs between \_set_skip_append(True) and \_set_skip_append(False), all future frames in the sequence produce corrupted output with no error message.
- Verify FlashInfer compatibility on the target production GPU. Test expandable_segments interaction with the current CUDA driver version. Document the SDPA fallback path and confirm it works end-to-end.
- Profile peak CPU RAM for 30s, 60s, and 300s videos to establish a baseline before optimization. This validates the memory analysis and sets the ceiling for the incremental-save fix.
- Add a hard frame limit (e.g., max_frames=10000) with a clear user-facing error message. Currently, submitting a 2-hour video would silently OOM the server.

## **Phase 1: Core Processor Class (3-5 days)**

The single highest-ROI engineering task: build a SceneProcessor class that encapsulates the model lifecycle, fixes all three memory/streaming blockers, and provides a clean interface for the web layer.

### **1.1 SceneProcessor Class (lingbot_map/processor.py)**

- \__init_\_(model_path, device, dtype, compile=False): Load model, optional torch.compile warmup
- process(images, config) -> dict: Blocking one-shot inference, returns full predictions
- process_streaming(images, config) -> Generator: Yields per-frame dicts - replaces the accumulation anti-pattern
- clean(): KV cache cleanup + torch.cuda.empty_cache() - must be called between every request
- estimate_resources(frame_count, resolution) -> dict: Pre-flight memory/time estimate for user feedback
- gpu_stats -> dict: VRAM usage, temperature, backend (FlashInfer vs SDPA) - for /api/health endpoint

### **1.2 Incremental Save-to-Disk (fixes CPU RAM OOM)**

Modify process_streaming() to accept a save_callback(frame_idx, frame_dict) parameter. The callback writes each frame's predictions to disk immediately after computation, keeping peak CPU RAM proportional to window size (~200 MB) rather than video length (~27 GB for 9,000 frames). This is a 100x reduction in peak memory with zero quality impact.

### **1.3 In-Memory Video Decode**

Add load_and_preprocess_video_stream(path_or_bytes) -> Iterator using PyAV. This replaces the current cv2.imwrite disk round-trip (which introduces JPEG quality loss and 5-10 seconds of avoidable I/O for 300 frames). The batch demo has already implemented this pattern - it just needs backporting.

### **1.4 Drop Redundant Output Keys**

Do not include world_points or echoed images in per-frame streaming output by default. This halves the per-frame output size. world_points can be computed on-demand from saved depth + extrinsics, either server-side or client-side.

## **Phase 2: Web API Layer (3-5 days)**

Build the FastAPI server on top of the SceneProcessor. All inference runs in a background worker; the HTTP layer is fully async and non-blocking.

### **2.1 FastAPI Application (webapp/app.py)**

| **Endpoint**                        | **Method** | **Description**                                                        |
| ----------------------------------- | ---------- | ---------------------------------------------------------------------- |
| POST /api/scenes                    | POST       | Upload video/images; returns scene_id immediately; enqueues processing |
| GET /api/scenes/{id}                | GET        | Status, progress (frames processed / total), result URLs               |
| GET /api/scenes/{id}/preview.glb    | GET        | Low-res GLB (1M points, <15 MB); available ~30s after upload           |
| GET /api/scenes/{id}/full.glb       | GET        | Full Draco-compressed GLB (10M points, ~35 MB)                         |
| GET /api/scenes/{id}/gaussian_splat | GET        | Compressed .ply.gz for GS viewer (~60 MB)                              |
| WS /api/scenes/{id}/stream          | WebSocket  | Per-frame streaming: pose JSON + depth + thumbnail as they complete    |
| GET /api/health                     | GET        | GPU VRAM, queue depth, active backend, current scene progress          |
| DELETE /api/scenes/{id}             | DELETE     | Cleanup artifacts from disk                                            |

### **2.2 Background Worker (webapp/worker.py)**

- Single FIFO queue; one GPU request at a time. No concurrent inference - the KV cache is single-sequence.
- process_scene_background(scene_id): Runs SceneProcessor.process_streaming(), calls save_callback per frame, updates scene metadata in real time for the status endpoint.
- After inference: trigger async tasks for GLB generation (immediate), GS training (background, 10-35 min), segmentation (parallel worker).

### **2.3 Scene Store (webapp/scene_store.py)**

- LocalDiskSceneStore for MVP: stores NPZ predictions, GLB files, metadata JSON, GS .ply.gz under /data/scenes/{scene_id}/
- S3SceneStore for production: same interface, backed by S3-compatible object storage
- Scene metadata: status (queued / processing / ready / error), timestamps, frame count, file sizes, processing config

## **Phase 3: 3DGS + Segmentation Integration (2-3 weeks, parallel tracks)**

### **Track A: 3DGS Pipeline**

- lingbot_map/gs/initializer.py: Convert depth + extrinsics -> Gaussian initialization using voxel-grid deduplication. Target: 1-5M Gaussians per scene.
- lingbot_map/gs/trainer.py: Wrap gsplat training loop. Configure densification budget per scene size. Apply FastGS pruning strategy for 2-3x training speedup.
- lingbot_map/gs/export.py: Export trained Gaussians to .ply, compress to .ply.gz. Convert to .splat format for gsplat.js.
- CLI test: python -m lingbot_map.gs train --predictions scene.npz --output scene.ply. Compare quality vs COLMAP baseline on courthouse dataset.
- Integrate GS-Scale CPU offloading for scenes >2M Gaussians on 12 GB GPUs.

### **Track B: Segmentation Pipeline**

- lingbot_map/segmentation.py: SegmentationPipeline class. Frame-by-frame processing with SAM 2.1 automatic mode (grid prompts for dense coverage).
- Add OpenWorldSAM integration for text-prompt-driven segmentation (e.g., 'segment all doors').
- lift_to_3d(): Associate 2D masks with 3D world points from depth unprojection. Produces semantically labeled point cloud.
- export_semantic_glb(): Per-class colored GLB with class toggle metadata for the frontend.
- Evaluate Semantic Gaussians for embedding CLIP features into the GS training loop - eliminates the separate segmentation pass for common query patterns.

## **Phase 4: Production Web Viewer (2-3 weeks, frontend)**

### **4.1 Gaussian Splat Viewer**

- Integrate gsplat.js (antimatter15/splat) as the primary 3D view mode
- Camera controls: orbit, fly-through, camera path animation along the inferred trajectory
- Progressive loading: point cloud preview while GS trains; seamless swap when GS is ready

### **4.2 Semantic Point Cloud Viewer**

- Three.js GLB loader with class toggle checkboxes (show/hide segmented classes)
- Confidence slider: client-side re-filtering of point cloud without server round-trip
- Click-to-segment: Forward click coordinates to server, run SAM 2.1 prompt, return new mask; update 3D viewer

### **4.3 Progressive Loading Strategy**

- T+0s: Upload acknowledged; skeleton UI shown
- T+30s: Low-res GLB point cloud preview (1M points) displayed in Three.js
- T+5min: Full Draco GLB available; viewer upgrades automatically
- T+20min: GS .ply available; viewer transitions to Gaussian Splat mode
- Throughout: WebSocket progress bar showing frames processed, current stage, ETA

### **4.4 Production Hardening**

- Multi-GPU worker pool for concurrent users (one inference sequence per GPU)
- File upload limits: max 500 MB, max 10,000 frames, type validation
- Result expiry: auto-cleanup after 7 days (configurable)
- Prometheus metrics: request latency, GPU VRAM watermark, queue depth, GS training time
- Docker container with CUDA 12.x base image; health check on startup

# **4\. Key Decisions & Trade-offs**

| **Decision**             | **Recommended Choice**                         | **Rationale**                                                                     |
| ------------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| GS training backend      | gsplat + FastGS pruning                        | Apache 2.0, 3.32x faster than baseline, depth-initialized init path supported     |
| Segmentation model       | SAM 2.1 (server) + OpenWorldSAM (text prompts) | SAM 2.1 for pre-computed classes; OpenWorldSAM for interactive open-vocab queries |
| Semantic GS embedding    | Semantic Gaussians (evaluate)                  | Eliminates separate segmentation pass; adds ~20% GS training overhead             |
| Web GS renderer          | gsplat.js (.splat format)                      | Lighter than .ply, broader device support; antimatter15/splat as fallback         |
| Point cloud format (web) | Draco GLB + Potree for large scenes            | Draco GLB for scenes <14M pts; Potree octree for larger outdoor scenes            |
| Inference backend        | FlashInfer with SDPA fallback                  | 2x throughput gain; SDPA fallback for driver compatibility issues                 |
| torch.compile            | Enable for production inference                | 1.5-2x speedup; verify no quality regression on test set first                    |
| Memory architecture      | Incremental save-to-disk (Strategy 1)          | 100x RAM reduction, zero quality impact - highest ROI single change               |
| GPU allocation           | Single GPU MVP, multi-GPU later                | FIFO queue on 1 GPU handles ~3 concurrent daily users; scale when needed          |
| Point cloud generation   | Client-side unprojection                       | 3x bandwidth reduction; ship depth + extrinsics, not world points                 |

# **5\. Summary: Critical Path to Production**

The three changes with the highest combined ROI, in priority order:

- Incremental save-to-disk (Phase 1.2): Reduces peak CPU RAM from ~27 GB to ~200 MB for a 15-minute video. Zero quality impact. Unblocks processing of all video lengths.
- SceneProcessor class (Phase 1): Encapsulates the model lifecycle, streaming, and cleanup. The single prerequisite for building a multi-user web server. Required before any web API work.
- gsplat + FastGS GS training (Phase 3A): Reduces per-scene GS training time by 3x (from ~12 min to ~4 min), making on-demand GS generation practical for a consumer-facing product.

With these three changes in place, the LingBot-MAP web app can:

- Process 15-minute videos without OOM on a 24 GB GPU
- Serve interactive 3D scenes to browser users within 20 minutes of upload
- Run 3DGS training on-demand without COLMAP - a key differentiator vs. existing 3D scanning tools
- Scale to multiple concurrent users by adding one GPU per inference queue slot

The research landscape (FastGS, Semantic Gaussians, OpenWorldSAM, FlashInfer v2+) is moving rapidly and strongly in the direction of this project's goals. The timing is favorable: training and segmentation tools that would have required custom research 12 months ago are now production-ready open-source libraries.