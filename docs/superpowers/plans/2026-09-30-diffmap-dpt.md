# Diffusion map + DPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diffusion maps on any `.obsp` connectivity graph and diffusion pseudotime from a chosen root (stemFinder-aware), end to end: pure module, adaptor, routes, codegen, ScanpyModal operation, Pseudotime modal.

**Architecture:** `backend/xcell/diffusion.py` wraps `sc.tl.diffmap` / `sc.tl.dpt` on a throwaway AnnData holding only the graph, adding symmetrisation, isolated-cell masking and inf→NaN. The adaptor mirrors `run_umap` for scope/naming and keeps a registry in `uns['xcell_diffmaps']` so DPT can find each map's graph. Frontend: diffusion map is a generic ScanpyModal op; DPT is a dedicated global modal.

**Tech Stack:** Python 3.11, scanpy 1.11.5, scipy sparse, FastAPI/Pydantic; React + TypeScript + Zustand, vitest.

**Spec:** `docs/superpowers/specs/2026-09-30-diffmap-dpt-design.md`

## Global Constraints

- Work only in the worktree `/private/tmp/claude-501/xcell-wt` (branch `feat/diffmap-dpt`); never edit the main repo mid-session (his `--reload` backend).
- Backend tests: `cd backend && "/Users/pcahan/Dropbox (Personal)/Code/xcell/.pixi/envs/default/bin/python" -m pytest …` (worktree's `xcell/` shadows the editable install because of cwd).
- Frontend: run `npx tsc --noEmit` / `npx vitest run` from `<wt>/frontend`.
- No `inf` / `NaN` across the API: NaN → `null` (existing `_nullable` for coordinates; obs columns already serialise NaN as null).
- No `None` values written into `uns` (h5ad cannot write them) — omit the key instead.
- `STATIONARY_EVAL = 0.9994` (scanpy's cut).
- Symmetrise asymmetric graphs as `(W + Wᵀ) / 2`; pass symmetric graphs through untouched.
- Output names: `X_diffmap[_<graph suffix>]`, subset `X_diffmap_<subset>[_<graph suffix>]`; DPT `dpt_pseudotime` + the map key's tail after `X_diffmap`.
- stemFinder orientation: `stemfinder*` lowest = most potent; `diffometer*` highest = most potent.
- Every new `_log_action` name needs a `codegen.REGISTRY` entry in the same commit.

## Review Focus

1. A squidpy spatial graph (asymmetric, possibly isolated cells, one component per section) — must give finite coordinates on covered cells, NaN elsewhere, a warning, never a crash. → Task 2 test `test_diffmap_on_an_asymmetric_spatial_graph_with_isolated_cells`.
2. A subset-scoped diffmap whose subset's slice of the graph is disconnected — DPT must give NaN (not inf) outside the root's component and the response must be JSON-serialisable. → Task 2 test `test_dpt_response_is_json_safe_on_a_disconnected_subset`.
3. A custom output name that collides with an existing embedding or column (`X_pca`, `leiden`) — refused with 400, nothing overwritten. → Task 2 tests `test_diffmap_refuses_to_overwrite_another_embedding`, `test_dpt_refuses_to_overwrite_another_column`.
4. Re-running DPT into the same column with another root — overwrites cleanly, the plot recolours. → Task 2 `test_dpt_rerun_overwrites`; Task 5 clears `colorBy` before selecting.
5. Switching to a diffmap embedding later (pane picker) — opens on DC1×DC2, no refetch loop on a 2-column map. → Task 3 store tests.

---

### Task 1: Pure module `xcell/diffusion.py`

**Files:**
- Create: `backend/xcell/diffusion.py`
- Test: `backend/tests/test_diffusion.py` (already written: 22 tests — parity with scanpy, isolated cells, symmetrisation, components, DPT NaN, root_from_cells)

**Interfaces — Produces:**
- `STATIONARY_EVAL: float = 0.9994`
- `PreparedGraph(matrix: csr_matrix, kept: np.ndarray, n_isolated: int, symmetrized: bool, component_labels: np.ndarray, component_sizes: list[int])`, property `n_components`
- `prepare_graph(conn) -> PreparedGraph` (ValueError: non-square, negative weights, no edges)
- `diffusion_map(conn, n_comps=15, random_state=0) -> dict` keys `X` (n×n_comps float32, NaN rows), `evals` (float32), `kept`, `n_cells_used`, `n_isolated`, `symmetrized`, `n_components`, `component_sizes`, `n_stationary`, `view_dims` (list[int] of 2), `warnings` (list[str])
- `dpt_pseudotime(conn, X, evals, root, n_dcs=10) -> dict` keys `pseudotime` (float64, NaN), `n_unreachable`, `n_dcs`, `root`
- `dpt_space(X, evals, n_dcs) -> np.ndarray`
- `root_from_cells(X, evals, cells, n_dcs) -> int`

- [ ] **Step 1:** Run `pytest tests/test_diffusion.py -q` → fails (`ImportError: cannot import name 'diffusion'`).
- [ ] **Step 2:** Implement `diffusion.py`:

```python
"""Diffusion maps and diffusion pseudotime on any connectivity graph.

Pure: a graph and arrays in, plain dicts out; no adaptor. The computation is
scanpy's (``sc.tl.diffmap`` / ``sc.tl.dpt`` on a throwaway AnnData that holds
only the graph), so on a symmetric, connected kNN the results are scanpy's bit
for bit. What this module adds is for graphs that are not a scanpy kNN:

- an isolated cell makes scanpy's density normalisation divide by zero, and
  sparse algebra then places the cell at 0 on every component — a position,
  silently. Isolated cells are left out and get NaN.
- ``eigsh`` assumes a symmetric matrix; squidpy's kNN spatial graph is not.
- across disconnected components DPT returns inf, which JSON cannot carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

#: Eigenvalues at or above this are stationary (one per connected component).
#: scanpy's own cut in ``Neighbors._get_dpt_row``, chosen for float32 precision.
STATIONARY_EVAL = 0.9994

_NEIGHBORS_KEY = '_xcell_diffusion'


@dataclass(frozen=True)
class PreparedGraph:
    matrix: sparse.csr_matrix
    kept: np.ndarray
    n_isolated: int
    symmetrized: bool
    component_labels: np.ndarray
    component_sizes: list[int]

    @property
    def n_components(self) -> int:
        return len(self.component_sizes)


def prepare_graph(conn) -> PreparedGraph:
    ...  # see implementation
```

(Full bodies: symmetry test with relative tolerance 1e-6, `setdiag(0)` only when the diagonal is non-zero, degrees from row sums, `connected_components(directed=False)`, pass the input through unchanged when nothing needed fixing so scanpy parity is exact; `diffusion_map` validates `3 <= n_comps <= n_kept - 1`, builds `_scanpy_frame`, runs `sc.tl.diffmap(..., neighbors_key=_NEIGHBORS_KEY)`, scatters into NaN rows, computes `n_stationary`, `view_dims = [f, f+1]` with `f = min(max(n_stationary, 1), n_comps - 2)`, warnings for isolated / symmetrised / disconnected; `dpt_pseudotime` slices to covered rows, re-prepares (an isolated covered cell means the graph changed → ValueError), runs `sc.tl.dpt` with `uns['iroot']`, maps non-finite to NaN; `dpt_space` scales non-stationary components by λ/(1−λ) and keeps stationary ones unscaled — exactly DPT's distance; `root_from_cells` = nearest the centroid in that space among covered cells.)

- [ ] **Step 3:** `pytest tests/test_diffusion.py -q` → all pass.
- [ ] **Step 4:** Commit `feat(diffusion): diffusion maps and DPT on any connectivity graph`.

### Task 2: Adaptor, routes, codegen, subset tree

**Files:**
- Modify: `backend/xcell/adaptor.py` (constants near `CELL_SUBSETS_UNS`; `check_prerequisites` prereqs; `_subset_record_step` / `_subset_forget_key` / `_subset_derived_keys` / `_subset_summary` / `delete_cell_subset`; new methods after `run_leiden`: `_default_graph_key`, `_diffmap_registry`, `_diffmap_info`, `run_diffmap`, `list_diffmaps`, `_dpt_root`, `run_dpt`)
- Modify: `backend/xcell/stemfinder.py` (`potency_columns(columns, dtypes)`)
- Modify: `backend/xcell/api/routes.py` (`DiffmapRequest`, `DptRequest`, `POST /scanpy/diffmap`, `POST /scanpy/dpt`, `GET /scanpy/diffmaps`)
- Modify: `backend/xcell/codegen.py` (`diffmap`, `dpt` entries)
- Test: `backend/tests/test_diffmap_adaptor.py`

**Interfaces — Consumes:** Task 1. **Produces:**
- `run_diffmap(n_comps=15, graph_key=None, key_added=None, active_cell_indices=None, cell_subset=None) -> dict` with `embedding_name, n_comps, graph_key, eigenvalues, n_stationary, view_dims, n_cells_used, n_isolated, n_components, component_sizes, symmetrized, warnings, cell_subset?`
- `run_dpt(diffmap_key='X_diffmap', n_dcs=10, root_mode='cells', root_cells=None, root_column=None, root_value=None, root_component=1, key_added=None) -> dict` with `key_added, diffmap_key, n_dcs, root_index, root_name, root_rule, n_cells, n_unreachable, warnings, cell_subset?`
- `list_diffmaps() -> {'diffmaps': [{key, graph_key, n_comps, n_cells, cell_subset, eigenvalues, n_stationary, view_dims}], 'potency': [{column, direction, label}]}`
- Routes: `POST /api/scanpy/diffmap`, `POST /api/scanpy/dpt`, `GET /api/scanpy/diffmaps`
- Subset summary `derived` gains `diffmap: list[str]`, `dpt: list[str]`.

Tests to write first (`test_diffmap_adaptor.py`, self-contained `_adata()` grid with expression blocks + spatial coords, `_adaptor()`):
- default graph → `X_diffmap`, `uns['diffmap_evals']`, registry entry, result `view_dims == [1, 2]`
- spatial graph → `X_diffmap_spatial`, registry graph `spatial_connectivities`
- `test_diffmap_on_an_asymmetric_spatial_graph_with_isolated_cells` (radius graph): finite on covered, NaN on isolated, `symmetrized`, warnings
- ad-hoc `active_cell_indices` → NaN outside, writes `X_diffmap`
- named subset → `X_diffmap_<name>`, dataset `X_diffmap` untouched; explicit `graph_key='connectivities'` still suffixes; registry `cell_subset`; subset `derived['diffmap']`; summary `embeddings` includes it
- `test_diffmap_refuses_to_overwrite_another_embedding` (`key_added='X_pca'` → ValueError, X_pca unchanged)
- unknown graph → ValueError; no graph at all → ValueError prerequisites
- DPT: each root mode (`cells` one/many, `obs_min`, `obs_max`, `group`, `dc_min`, `dc_max`); stemfinder orientation via a fake `stemfinder` column; root restricted to covered cells; unknown mode / missing column / non-numeric column / empty group → ValueError; unknown map → KeyError
- DPT default names: `X_diffmap` → `dpt_pseudotime`, `X_diffmap_spatial` → `dpt_pseudotime_spatial`, subset → `dpt_pseudotime_<name>`
- `test_dpt_rerun_overwrites`; `test_dpt_refuses_to_overwrite_another_column`
- `test_dpt_response_is_json_safe_on_a_disconnected_subset` (`json.dumps(result, allow_nan=False)`, obs NaN outside)
- DPT on a scanpy-written `X_diffmap` (no registry, `uns['diffmap_evals']`) works
- DPT after the map's graph is deleted → ValueError mentioning re-run
- logged step replays the resolved root: `history[-1]['params']['root_cells'] == [root_index]`
- subset cascade delete drops diffmap + dpt + registry entries
- `list_diffmaps` lists both and potency columns in the right direction
- routes: 200, 400 (bad n_comps), 404 (unknown map) via `TestClient` with `routes.set_adaptor`
- codegen: `codegen.REGISTRY['diffmap'|'dpt']` translate a recorded step to `xa.run_diffmap(...)` / `xa.run_dpt(..., root_cells=[i])`

- [ ] Steps: write tests → run (fail) → implement → run (pass) → full backend suite → commit `feat(diffusion): diffusion maps and DPT in the adaptor, routes and notebook export`.

### Task 3: Frontend helpers, store dims, axis labels, subset tree

**Files:**
- Create: `frontend/src/lib/diffusion.ts`, `frontend/src/lib/diffusion.test.ts`
- Modify: `frontend/src/store.ts` (`setSelectedEmbedding` default dims), `frontend/src/store.test.ts`
- Modify: `frontend/src/App.tsx` (DimensionPicker `DC{i}` labels)
- Modify: `frontend/src/lib/cellSubsets.ts` (+test): `CellSubsetDerived.diffmap/dpt`, `SUBSET_SCOPED_OPS` + `'diffmap'`, `subsetDiffmapKey`, `scopedOutput('diffmap')`, `derivedBadges`
- Modify: `frontend/src/lib/subsetTree.ts` if it enumerates derived kinds

**Produces (`lib/diffusion.ts`):**
- `isDiffmapKey(name: string): boolean` — `/^X_diffmap(_|$)/`
- `diffmapAxisLabel(i: number): string` — `DC${i}`
- `defaultDiffmapDims(ncols: number): {x: number, y: number} | null` — `{1,2}` when ncols ≥ 3 else null
- `dptOutputName(diffmapKey: string): string`
- `type RootMode = 'potency' | 'obs_min' | 'obs_max' | 'group' | 'cells' | 'dc_min' | 'dc_max'`
- `interface DptInputs { diffmapKey; nDcs: string; rootMode: RootMode; potencyColumn; potencyDirection: 'min'|'max'; column; groupValue; component: string; selection: number[]; keyAdded }`
- `dptParams(i: DptInputs): Record<string, unknown>` (potency → obs_min/obs_max)
- `dptBlocker(i: DptInputs): string | null`

- [ ] Steps: vitest for each helper (fail) → implement → store test "selecting a ≥3-column X_diffmap with no dims sets {1,2}; a 2-column one does not; existing dims kept" → implement → `npx tsc --noEmit` → commit `feat(diffusion): frontend helpers, DC axis labels, diffmap default axes`.

### Task 4: ScanpyModal — Diffusion map operation

**Files:** Modify `frontend/src/components/ScanpyModal.tsx`:
- `cell_analysis.functions.diffmap` after `leiden`: params `graph_key` (`graph_select`, `emptyLabel: 'Dataset default graph'`), `key_added` (text, ''), `n_comps` (number, 15)
- needs-graphs list (~1041) + `'diffmap'`
- default-name effect (~1057-1088) and graph preselection (~1098) handle `'diffmap'` like `'umap'` (`base = 'X_diffmap'`, blank on default graph)
- prerequisite bypass (~1152) includes `'diffmap'` when a graph is chosen
- result message: warnings + "Diffusion map X (n comps, λ…)" before the generic `embedding_name` branch
- refresh allowlist (~1749) + `'diffmap'`; embedding switch (~1770) + `'diffmap'` and `setEmbeddingDims(name, view_dims[0], view_dims[1])` before `setSelectedEmbedding`
- `custom` entry `dpt` ('Pseudotime (DPT)') with a button that opens the Pseudotime modal (like stemfinder at ~2937)

- [ ] Steps: implement → `npx tsc --noEmit` → commit `feat(diffusion): Diffusion map in Analyze → Cells`.

### Task 5: PseudotimeModal

**Files:**
- Create: `frontend/src/components/PseudotimeModal.tsx`
- Modify: `frontend/src/store.ts` (`isPseudotimeModalOpen` + setter, global)
- Modify: `frontend/src/App.tsx` (mount)

Behaviour: on open (and dataset change) GET `/api/scanpy/diffmaps`; reset state (global modals never unmount). Diffusion map select (label: key · graph · n cells); components used (default `min(10, n_comps)`); root mode radio list — *Most potent (stemFinder)* only when `potency` non-empty (default then, else *Tip of a diffusion component* DC1 min); obs column pickers from `obsSummaries`/schema dtypes; group values from the column's categories; *Current selection* shows `selectedCellIndices.length` and is disabled at 0; output name placeholder `dptOutputName(key)`. Run: POST `/api/scanpy/dpt` with `dptParams(...)`; on success `refreshSchema()`, `refreshObsSummaries()`, `setColorBy(null)`, `selectColorColumn(key)`, `setColorMode('metadata')`; show root rule, cell count, unreachable/warnings. Escape and click-outside close.

- [ ] Steps: implement → `npx tsc --noEmit`, `npx vitest run` → commit `feat(diffusion): Pseudotime (DPT) modal with stemFinder root`.

### Task 6: Verify in the browser, docs

- [ ] Isolated stack: backend :8100 from the worktree, Vite :5273 with `XCELL_BACKEND`.
- [ ] Load `ad_Grosswendt_demo.h5ad` (700 cells): normalize → log1p → HVG → PCA → neighbors → Diffusion map → plot shows DC1×DC2 with DC axis labels; stemFinder → DPT with stemFinder root → coloured by `dpt_pseudotime`; Epiblast low pseudotime.
- [ ] Toy spatial dataset: spatial neighbors → Diffusion map on the spatial graph → `X_diffmap_spatial`, warnings shown, no failed requests (`window.fetch` recorder).
- [ ] Subset: save a subset, run diffmap → `X_diffmap_<name>`, CoverageNotice visible, DPT → `dpt_pseudotime_<name>`.
- [ ] README feature list + CHANGELOG entry; commit `docs: diffusion maps and DPT`.
- [ ] Whole-branch review (requesting-code-review), fix, full suites, stop servers, merge per the live-dataset check.
