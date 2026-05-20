# Repo Structure Review — Skeptical Full-Stack Assessment

> Review of the outside contractor's monorepo recommendation against the current
> codebase, with a concrete plan for restructuring.

---

## TL;DR

The contractor is **directionally right but tactically wrong**. Monorepo: yes.
Their suggested layout: no — it ignores the existing package structure, would
break imports, creates empty directories for things that don't exist, and
misses the real structural problems already rotting in the codebase. Below is
a counter-proposal grounded in what's actually here.

---

## 1. What the contractor got right

| Point | Verdict |
|---|---|
| Monorepo for a hobby MVP with staged deployment | ✅ Correct. One repo, one issue tracker, one PR for cross-cutting changes. |
| Separate deployable units (frontend, API, worker) | ✅ Correct end-state. These should be independently buildable/deployable. |
| Shared contracts between services | ✅ Correct principle. API request/response types should be defined once. |
| `packages/core` concept for reusable inference wrapper | ✅ Good instinct — the model should be wrapped in a clean class, not scripted in `main()`. |
| Don't split into separate repos until justified by operational cost | ✅ Correct. Premature repo-splitting is a common over-engineering mistake. |

---

## 2. What the contractor got wrong — and why it matters

### 2.1 `packages/core` would break the existing package

The repo already has a proper Python package: `lingbot_map/`. It's pip-installable
via `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["lingbot_map*"]
```

Moving it to `packages/core/lingbot_map` would:
- Break `pip install -e .` (the `where` path changes)
- Break all internal `from lingbot_map.xxx import yyy` imports across 70+ files
- Require every file that currently does `import lingbot_map` to change
- Break `batch_demo.py`'s `from lingbot_map.vis.sky_segmentation import ...`
- Be a cosmetic rename with zero functional benefit

**The core library is already a package. Don't move it.** Adding nesting
(`packages/core/lingbot_map`) doesn't improve anything — it just buries the
most important code one level deeper and makes `pip install -e .` require
configuration changes for no reason.

### 2.2 Four empty app directories are speculative

The contractor proposes `apps/web`, `apps/api`, `apps/worker`, `apps/shared`.
None of these exist. Creating four directories that will sit empty for
weeks/months is premature. It also creates a false sense of structure — someone
will open `apps/web` expecting a React app and find nothing.

The repo should reflect what **currently exists** plus room for the **next
concrete step** (FastAPI server), not all possible future steps.

### 2.3 `packages/shared` is over-engineering at this stage

For a single-developer Python project where the API and worker are the same
Python process (see §5), "shared contracts" means ~5 Pydantic models and a
couple of TypedDicts. Creating a separate package with its own `pyproject.toml`,
build step, and import path for 200 lines of types is ceremony without value.

These should live in `lingbot_map/schemas.py` — the core package that both
the API and worker already import.

### 2.4 The layout ignores the current `demo_render/` mess

This is the biggest blind spot. `demo_render/` currently bundles **three
unrelated things**:

| Subdirectory | What it actually is |
|---|---|
| `rgbd_render/` | Point cloud render pipeline (full package with `__init__.py`, camera paths, scene builder, octree LOD, GPU culling, video encoding) |
| `render_cuda_ext/` | Compiled CUDA extension with its own `setup.py` and `torch.utils.cpp_extension` |
| `interactive_viewer/` | WebSocket-based interactive 3D viewer server |

These are three different deployable units jammed into one directory. The
contractor's layout doesn't address how to split them. Worse, `demo_render/`
has structural rot:

- **`batch_demo.py` imports from root `demo.py`** (`from demo import load_model, postprocess, prepare_for_visualization`) — tight coupling that would break under any restructuring
- **Three separate `sys.path.insert` hacks** to find `render_cuda_ext` and the project root
- **Code duplication**: `demo_render/rgbd_render/data/sky.py` explicitly says "Adapted from `lingbot_map.vis.sky_segmentation` to keep demo_render self-contained" — meaning the same sky segmentation logic exists in two places

### 2.5 No mention of binary assets

The repo root contains `lingbot-map-long.pt` (4.63 GB) and `skyseg.onnx` (176 MB).
These are not source code. They should not be in the root. The contractor's layout
says nothing about where large binary dependencies live.

---

## 3. The real structural problems — found during investigation

These are the things that will actually block a web app, regardless of how
directories are arranged:

### Problem 1: `demo.py` is both a script and a module

`batch_demo.py` does `from demo import load_model, postprocess, prepare_for_visualization`.
This means:

- `demo.py` must be importable as a Python module (no `if __name__ == "__main__"` guard issues... yet)
- `demo.py` has functions that are reused by other code, making it a de facto library
- Any change to `demo.py`'s imports or module-level code affects `batch_demo.py`

**Fix**: Extract `load_model`, `postprocess`, `prepare_for_visualization` into
`lingbot_map/` (where they belong — they operate on model internals). `demo.py`
becomes a thin CLI wrapper that imports from the library.

### Problem 2: `sys.path.insert` hacks

Three files modify `sys.path` at runtime to find packages:

```
demo_render/rgbd_scan_render.py:    sys.path.insert(0, ... 'render_cuda_ext')
demo_render/batch_demo.py:          sys.path.insert(0, ... "render_cuda_ext")
demo_render/interactive_viewer/server.py:  sys.path.insert(0, _PROJECT_ROOT)
```

This means the CUDA extension can't be imported normally, the interactive viewer
can't find its own parent package, and nothing is pip-installable.

**Fix**: Make `render_cuda_ext` a proper pip-installable package. Make
`interactive_viewer` a proper subpackage importable without path hacks.

### Problem 3: Sky segmentation code duplication

`demo_render/rgbd_render/data/sky.py` is an adapted copy of `lingbot_map/vis/sky_segmentation.py`.
When the sky segmentation model or logic changes, it must be updated in two
places. This is already bit-rotting.

**Fix**: Have `demo_render` import from `lingbot_map.vis` instead of maintaining
a fork. If the render pipeline needs a different interface, wrap it — don't copy it.

### Problem 4: Binary files in repo root

`lingbot-map-long.pt` (4.63 GB) and `skyseg.onnx` (176 MB) are in the root.
This makes `ls` painful, bloats the working directory, and means `git status`
is checking 5 GB of ignored files every time.

**Fix**: Move to `models/` with clear naming and a `.gitignore` entry for downloaded
model files. The code should reference them by configurable path, not hardcoded
relative to CWD.

---

## 4. Counter-proposal: what the structure should actually be

### Principles

1. **Don't move things that work.** `lingbot_map/` is a proper package. Leave it.
2. **Reflect what exists, not what might exist.** Add directories when you build
   the thing they contain.
3. **One clear home for each concern.** No code duplication, no `sys.path` hacks.
4. **Large binaries out of root.** Models, ONNX files, example data — all in
   dedicated directories.
5. **Entry points are thin.** CLI scripts import from the library, not vice versa.

### Proposed layout

```
lingbot_map_cuda/
│
├── lingbot_map/                   # Core library — NO CHANGES to internal structure
│   ├── models/                    # GCTStream, GCTBase, windowed variant
│   ├── aggregator/                # AggregatorStream (KV cache, FlashInfer/SDPA)
│   ├── layers/                    # Attention, blocks, RoPE, FlashInfer cache
│   ├── heads/                     # Camera head, DPT depth head
│   ├── utils/                     # Pose encoding, geometry, image loading
│   ├── vis/                       # GLB export, sky segmentation, Viser wrapper
│   ├── schemas.py                 # NEW — Pydantic models for API contracts
│   └── processor.py               # NEW — SceneProcessor class (from webapp_analysis.md §5.2)
│
├── apps/                          # Entry points — each is a deployable unit
│   ├── cli/                       # CLI demo (was root demo.py + gct_profile.py)
│   │   ├── demo.py                # Thin wrapper: parse args → processor.process()
│   │   └── profile.py             # Was gct_profile.py
│   │
│   ├── batch/                     # Batch processing (was demo_render/batch_demo.py)
│   │   └── main.py                # Was batch_demo.py
│   │
│   ├── render/                    # Point cloud → video pipeline (was demo_render/rgbd_render/)
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── scene.py
│   │   ├── renderer.py
│   │   ├── config.py
│   │   ├── overlay.py
│   │   ├── video.py
│   │   ├── data/
│   │   ├── geometry/
│   │   └── pipeline/
│   │
│   ├── viewer/                    # Interactive viewer (was demo_render/interactive_viewer/)
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── camera.py
│   │   └── npz_to_glb.py
│   │
│   └── cuda_ext/                  # CUDA kernels (was demo_render/render_cuda_ext/)
│       ├── setup.py
│       └── render_cuda_ext/
│           ├── __init__.py
│           └── _api.py
│
├── models/                        # Binary model files (NEW — moved from root)
│   ├── .gitkeep
│   └── README.md                  # Download links, expected paths
│
├── example/                       # Example data — unchanged
│   ├── courthouse/
│   ├── university/
│   ├── loop/
│   └── oxford/
│
├── assets/                        # Static assets — unchanged
│
├── config/                        # YAML configs (was demo_render/config/)
│   ├── default.yaml
│   ├── indoor.yaml
│   └── outdoor_large.yaml
│
├── docs/                          # Architecture & decisions
│   ├── agent.md                   # Was root agent.md
│   ├── webapp_analysis.md         # Already written
│   └── repo_structure_review.md   # This document
│
├── pyproject.toml                 # Core library build config — UPDATED
├── README.md
├── LICENSE.txt
├── .gitignore                     # UPDATED — model files, ONNX
│
└── Dockerfile                     # Single Dockerfile for the API+worker service
```

### What changes and why

| Change | Current | Proposed | Why |
|---|---|---|---|
| `demo.py` → `apps/cli/demo.py` | Root-level script imported as module by batch | Thin CLI in `apps/`, imports from `lingbot_map` | Stops scripts importing from other scripts. Clean dependency direction. |
| `gct_profile.py` → `apps/cli/profile.py` | Root-level script | In `apps/cli/` | Same home as demo. |
| `demo_render/batch_demo.py` → `apps/batch/main.py` | Imports from root `demo.py` | Imports from `lingbot_map` | Broken coupling. |
| `demo_render/rgbd_render/` → `apps/render/` | Flat under `demo_render/` | Top-level app package | Recognizable as a deployable unit. |
| `demo_render/interactive_viewer/` → `apps/viewer/` | Hidden inside demo_render | Top-level app | Same. |
| `demo_render/render_cuda_ext/` → `apps/cuda_ext/` | `sys.path.insert` hacks to find it | Proper pip-installable package under `apps/` | No runtime path manipulation. |
| `demo_render/config/` → `config/` | Configs buried in demo_render | Top-level config | Shared across apps. |
| Binary files → `models/` | Root directory | `models/` | Clean root. `.gitignore` catches downloads. |
| `agent.md` → `docs/` | Root | `docs/` | Documentation in one place. |
| `lingbot_map/schemas.py` | Doesn't exist | NEW — Pydantic models | Types shared between API and worker without a separate package. |
| `lingbot_map/processor.py` | Doesn't exist | NEW — SceneProcessor | Reusable model wrapper (see webapp_analysis.md). |

### What does NOT change

- **`lingbot_map/` internal structure** — models, layers, aggregator, heads, utils,
  vis all stay exactly where they are. They work.
- **`pyproject.toml`** — only `packages.find.include` may need updating if
  `apps/` packages are added to the install.
- **`example/` and `assets/`** — unchanged.
- **All internal imports** — `from lingbot_map.models.gct_stream import GCTStream`
  still works everywhere.

### Dependency direction after restructuring

```
         ┌─────────────────────┐
         │   lingbot_map/      │  ← Core library (no deps on apps/)
         │   models, layers,   │
         │   heads, utils, vis │
         │   schemas.py        │
         │   processor.py      │
         └────────┬────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ apps/cli │ │apps/batch│ │apps/viewer│  ← Import from lingbot_map only
└──────────┘ └──────────┘ └──────────┘
      ▲           ▲           ▲
      │           │           │
      └───────────┼───────────┘
                  │
         ┌────────┴────────┐
         │   apps/render/  │  ← Also imports from lingbot_map
         │                 │     (no more sky.py duplication)
         └─────────────────┘
```

No app imports from another app. No `sys.path.insert`. No `from demo import ...`.

---

## 5. What the contractor's layout would look like if actually applied — and why it fails

If we applied the contractor's layout literally:

```
packages/core/lingbot_map/    ← would break pyproject.toml + all imports
packages/shared/              ← empty for months; 200 lines of types don't need a package
apps/web/                     ← empty for months
apps/api/                     ← empty for months (the API is the same process as the worker!)
apps/worker/                  ← empty for months
```

The `apps/api` and `apps/worker` split is particularly misguided at this stage.
On a single GPU, the API server IS the worker. Splitting them into separate
services requires:

- A message queue (Redis, RabbitMQ)
- A job persistence layer
- Separate Docker containers
- Network communication between services
- Serialization/deserialization of multi-GB tensors over a network boundary

This is **production infrastructure for a problem you don't have yet**. When
you have multiple GPUs or need to scale horizontally, then split. Until then,
a single FastAPI process that loads the model and processes requests sequentially
is simpler, faster, and easier to debug.

The contractor is correct that the end-state should be separate services. They're
wrong that you need separate **directories** for them today. You need one `app.py`
that can be split later, not three empty folders.

---

## 6. Incremental migration plan — how to get there without breaking everything

This migration should happen in stages, each independently testable.

### Stage 0 (now): Extract shared functions from demo.py

**Before any file moves**, fix the import direction:

1. Move `load_model()`, `postprocess()`, `prepare_for_visualization()` from
   `demo.py` into `lingbot_map/` (new `lingbot_map/inference.py` or into
   existing modules).
2. Update `batch_demo.py` to import from `lingbot_map.inference` instead of `demo`.
3. Verify both demo and batch still work.

This is a pure refactor — no files move, only imports change.

### Stage 1: Create `apps/cli/`, move entry points

1. Create `apps/cli/`.
2. Move `demo.py` → `apps/cli/demo.py`. Strip it down: import from
   `lingbot_map`, parse args, call processor, launch viewer.
3. Move `gct_profile.py` → `apps/cli/profile.py`.
4. Update any internal references.
5. Verify: `python apps/cli/demo.py --image_folder example/courthouse ...`

### Stage 2: Extract render pipeline

1. Move `demo_render/rgbd_render/` → `apps/render/`.
2. Move `demo_render/config/` → `config/`.
3. Update `batch_demo.py` imports.
4. Fix sky segmentation duplication: make `apps/render/data/sky.py` import
   from `lingbot_map.vis.sky_segmentation` instead of maintaining a fork.
5. Verify batch processing still works.

### Stage 3: Extract CUDA extension and viewer

1. Move `demo_render/render_cuda_ext/` → `apps/cuda_ext/`.
2. Make `apps/cuda_ext` pip-installable (it already has `setup.py` — ensure
   the path works from the new location).
3. Move `demo_render/interactive_viewer/` → `apps/viewer/`.
4. Remove all `sys.path.insert` hacks.
5. Verify: `python apps/viewer/server.py` and render pipeline still import
   CUDA extension cleanly.

### Stage 4: Move binaries and docs

1. Create `models/`, move `.pt` and `.onnx` files.
2. Update `.gitignore` to exclude `models/*.pt`, `models/*.onnx` (track with
   DVC or document download links in `models/README.md`).
3. Move `agent.md` → `docs/agent.md`.
4. Move `webapp_analysis.md` → `docs/webapp_analysis.md`.

### Stage 5 (when building the web app): Add API

1. Create `app.py` at project root (or `apps/api/main.py`).
2. Import `SceneProcessor` from `lingbot_map.processor`.
3. Define FastAPI routes using `lingbot_map.schemas` Pydantic models.
4. Add `Dockerfile`.
5. The API IS the worker — same process, same GPU, sequential queue.

When you later need separate worker processes (multiple GPUs, high concurrency),
extract the worker into `apps/worker/` and add a message queue. The `SceneProcessor`
class doesn't change — only the orchestration layer does.

---

## 7. Comparison: contractor vs. counter-proposal

| Concern | Contractor | Counter-proposal | Winner |
|---|---|---|---|
| Core library location | `packages/core/lingbot_map` | `lingbot_map/` (unchanged) | **Counter** — no breakage |
| Entry points | `apps/web`, `apps/api`, `apps/worker` (all empty) | `apps/cli/`, `apps/batch/`, `apps/render/`, `apps/viewer/` (all populated) | **Counter** — reflects reality |
| Shared types | `packages/shared` (separate package) | `lingbot_map/schemas.py` (in core) | **Counter** — simpler, fewer packages |
| Binary assets | Not addressed | `models/` with `.gitignore` | **Counter** — actually handles the 5 GB of model files |
| Current structural rot | Not addressed | Fixed: `sys.path` hacks, sky.py duplication, `from demo import` coupling | **Counter** — fixes real problems |
| Future web app | Pre-created empty directories | `app.py` added when built | **Counter** — no empty folders |
| Migration risk | High — moves core package, breaks imports | Low — staged, each stage testable | **Counter** — won't break working code |

---

## 8. Bottom line

The contractor's monorepo recommendation is the right strategy but the wrong
layout. Their `packages/core` rename would break the existing Python package
for no benefit. Their four `apps/` directories are speculative — three of them
would sit empty while the one that has code (`apps/web`) doesn't exist yet.
They completely missed the `demo_render/` bundling problem, the `sys.path`
hacks, the sky segmentation duplication, and the 5 GB of binaries in the root.

The counter-proposal above:
- Keeps `lingbot_map/` exactly where it is (zero import breakage)
- Splits `demo_render/` into its actual concerns (render pipeline, CUDA ext,
  interactive viewer → three separate `apps/`)
- Removes all `sys.path.insert` hacks and the `from demo import` coupling
- Puts binaries in `models/` and docs in `docs/`
- Creates space for the web app without pre-creating empty directories
- Migrates incrementally in five stages, each independently testable

Start with Stage 0 (extract shared functions from `demo.py`). That alone fixes
the worst structural problem and takes 30 minutes.
