# Diffusion map + diffusion pseudotime (DPT) — design

Analyze → Cells gains two operations next to UMAP and Leiden:

- **Diffusion map** — the diffusion components (Coifman & Lafon 2006, Haghverdi
  et al. 2015) of *any* connectivity graph in `.obsp`: the expression kNN, the
  spatial graph, a Combine Neighbors result, or a subset's own graph.
- **Pseudotime (DPT)** — diffusion pseudotime (Haghverdi et al. 2016, scanpy's
  implementation) from a chosen root, on an existing diffusion map. The root can
  come from stemFinder, so the potency ranking that already exists in xcell
  orients the trajectory.

Decisions were made without a review round (the session ran on "proceed
without my feedback unless needed"); each is marked **[decision]** with its
reason so it can be revisited.

## Why scanpy, wrapped

`sc.tl.diffmap` / `sc.tl.dpt` run on a **throwaway AnnData** holding only the
graph (an `.obsp` matrix plus a minimal `uns` neighbors entry). Measured: the
coordinates, eigenvalues and pseudotime are bit-identical to scanpy run on the
full object. The wrapper exists for three things scanpy gets silently wrong on
graphs that are not a scanpy kNN — all three were reproduced, not guessed:

1. **Isolated cells.** A zero-degree row makes the density normalisation divide
   by zero; sparse algebra skips the empty row, so the cell is placed at **0 on
   every component** — a fake position, no error. A squidpy radius graph on 500
   random points left 120 cells isolated; a subset-sliced graph can do the same.
   → isolated cells (degree 0 after dropping self-loops) are left out of the
   eigenproblem and get **NaN** coordinates and NaN pseudotime.
2. **Asymmetric graphs.** `eigsh` assumes a symmetric matrix. squidpy's
   `spatial_neighbors(n_neighs=k)` adjacency is **not** symmetric (kNN is not
   mutual). → an asymmetric graph is symmetrised as `(W + Wᵀ) / 2` (the
   convention of scikit-learn's `spectral_embedding`) and the result says so.
   **[decision]** average rather than max/union: it keeps the weights of a
   weighted graph meaningful, and for a 0/1 kNN it only halves one-way edges.
   A symmetric graph is passed through untouched, so the scanpy path is exact.
3. **Disconnected components.** Several eigenvalues ≈ 1 whose eigenvectors only
   say which component a cell is in; DPT gives `inf` to every cell outside the
   root's component. A spatial graph over several tissue sections is
   disconnected by construction. → the result reports component count and
   sizes, the number of stationary components (λ ≥ 0.9994, scanpy's own cut in
   `Neighbors._get_dpt_row`), and suggests viewing the first non-stationary
   pair; DPT maps `inf` to NaN (JSON has no inf) and reports how many cells
   were unreachable. **[decision]** no per-component diffusion maps: running on a
   saved subset (one section, one lineage) already gives a connected graph, and
   stitching per-component eigenvectors yields axes that are not comparable.

Negative weights are rejected (400): the transition matrix is undefined.

## Backend

### Pure module `backend/xcell/diffusion.py`

No adaptor import; scanpy imported inside the functions.

```python
STATIONARY_EVAL = 0.9994

prepare_graph(conn) -> PreparedGraph
    # symmetrise if needed, drop the diagonal, find isolated rows;
    # .matrix (kept × kept), .kept (indices), .n_isolated, .symmetrized,
    # .component_labels, .component_sizes
diffusion_map(conn, n_comps=15, random_state=0) -> dict
    # X (n × n_comps float32, NaN rows for isolated), evals, kept, n_isolated,
    # symmetrized, n_components, component_sizes, n_stationary, view_dims, warnings
dpt_pseudotime(conn, X, evals, root, n_dcs=10) -> dict
    # pseudotime (n, NaN where undefined), n_unreachable, root
root_from_cells(X, evals, cells, n_dcs) -> int     # nearest the centroid
```

`view_dims`: the first two non-stationary components (scanpy's `X_diffmap`
column 0 is the stationary state; with k components the first k are). The UI
switches to these after a run.

**Root rule for a group of cells** **[decision]**: the cell nearest the group's
centroid in DPT space — diffusion components scaled by λ/(1−λ), the metric
DPT itself uses, so "nearest" means what pseudotime will measure. A medoid
would be the same idea at O(g²); an extreme (tip) cell is sensitive to one
noisy outlier.

### Adaptor

`run_diffmap(n_comps=15, graph_key=None, key_added=None, active_cell_indices=None, cell_subset=None)`

Mirrors `run_umap`: `_resolve_cell_scope`; a subset with no graph_key uses its
own graph when it exists; `_require_graph` / `check_prerequisites('diffmap')`;
names via `_default_output_name('X_diffmap', graph_key)` /
`_subset_output_name(...)` (so `X_diffmap`, `X_diffmap_spatial`,
`X_diffmap_<subset>`, `X_diffmap_<subset>_<graph>`); a scoped run slices the
graph to the scope and writes NaN outside. Synchronous like UMAP — 1.9 s for
the eigenproblem at 50 k cells.

Writes `.obsm[key]` and a registry entry
`uns['xcell_diffmaps'][key] = {graph_key, n_comps, evals, symmetrized, cell_subset?}`
(no `None` values — h5ad cannot write them). For `X_diffmap` it also writes
scanpy's `uns['diffmap_evals']`, so `sc.tl.dpt` works on an exported file.

`run_dpt(diffmap_key, n_dcs=10, root_mode='cells', root_cells=None, root_column=None, root_value=None, root_component=1, key_added=None)`

- The diffusion map must exist. Its registry entry gives the graph; a bare
  `X_diffmap` with `uns['diffmap_evals']` (a file scanpy wrote) is accepted as
  the default graph over all cells, for interop.
- The cells are the diffusion map's non-NaN rows; the graph is re-sliced to
  them (the same slice the map was computed on). A graph that has since gone
  is a 400 that says to re-run the diffusion map.
- `n_dcs` is clamped to the map's component count, and must be ≥ 2.
- Root modes, all restricted to cells the map covers:
  - `cells` — `root_cells` (one cell → that cell; several → nearest centroid).
    The UI sends the current selection.
  - `obs_min` / `obs_max` — extreme of a numeric `.obs` column (NaN ignored).
    The stemFinder preset is `obs_min` of `stemfinder` (lower = less
    differentiated, see the stemFinder spec) — or `obs_max` of `diffometer`.
  - `group` — a categorical column = value → nearest that group's centroid.
  - `dc_min` / `dc_max` — tip of a diffusion component (the classic
    Haghverdi heuristic when nothing is annotated).
- Output column: `key_added`, else derived from the map's key:
  `X_diffmap` → `dpt_pseudotime`, `X_diffmap_spatial` → `dpt_pseudotime_spatial`,
  `X_diffmap_<subset>` → `dpt_pseudotime_<subset>`. Re-running overwrites (like
  Leiden), since trying another root is the expected loop.
- Writes `.obs[key]` (float, NaN where undefined) and
  `uns['xcell_dpt'][key] = {diffmap_key, n_dcs, root_index, root_name, root_rule}`.
- Logged with the **resolved** root (`root_mode='cells', root_cells=[i]`), so the
  exported notebook replays the same root whatever produced it; the prose
  summary keeps the rule ("lowest `stemfinder`").

Both record into the subset tree when their scope is a subset
(`_subset_record_step(name, 'diffmap' | 'dpt', key, params)`); the registry,
`_subset_forget_key`, `_subset_derived_keys` (+ naming fallback),
`_subset_summary.embeddings` and `delete_cell_subset(drop_derived=True)` learn
the two new kinds. Deleting drops `uns['xcell_diffmaps'][key]` / `uns['xcell_dpt'][col]` too.

`check_prerequisites`: `'diffmap': ['neighbors']`.

### Routes

`POST /api/scanpy/diffmap` (`DiffmapRequest`) and `POST /api/scanpy/dpt`
(`DptRequest`), thin, `KeyError → 404`, `ValueError → 400`, both with
`dataset` as the last query parameter. `GET /api/scanpy/diffmaps` lists the
diffusion maps (key, graph_key, n_comps, n_cells, cell_subset, evals,
n_stationary) plus the `.obs` columns that can orient a root (`potency`:
`stemfinder*` → lowest, `diffometer*` / `stemfinder_raw*` → highest) for the
DPT modal.

### Codegen

`diffmap` and `dpt` are `xcell`-fidelity `_direct` calls
(`xa.run_diffmap(...)`, `xa.run_dpt(...)`). **[decision]** not `exact`: the
isolated-cell and symmetrisation handling are xcell's, and an `exact` line
that differs on a spatial graph is the failure the record exists to prevent.

## Frontend

- **Diffusion map** is a generic ScanpyModal operation on the Cells tab, after
  Leiden — the same shape as UMAP: `graph_select` (with the
  "graph chosen → skip the default-graph prerequisite" bypass), output name
  derived like UMAP's (blank on the default graph, `X_diffmap_<subset>` on a
  subset), components (default 15). Joins `SUBSET_SCOPED_OPS` with a
  `scopedOutput` case, and the needs-graphs / refresh-schema / switch-embedding
  allowlists. After a run the plot switches to the new key on `view_dims`
  (`setEmbeddingDims` before `setSelectedEmbedding`, as ScoreGeneSets does) and
  the result's warnings are shown.
- **Pseudotime (DPT)** is a `custom` entry that opens a dedicated
  `PseudotimeModal` (global modal, like StemFinder) **[decision]**: the root
  picker needs inputs that depend on each other (a category's values, the live
  selection count, the stemFinder preset), which the generic param form
  cannot express. It reads `GET /api/scanpy/diffmaps`; root choices:
  *Most potent cell (stemFinder)* when a potency column exists (default then),
  *Lowest / highest value of a column*, *Cell group*, *Current selection (N
  cells)*, *Tip of a diffusion component*. It never sends the mask: DPT's
  cells are its map's. After a run: refresh schema + obs summaries and colour
  by the new column (clearing `colorBy` first so a re-run into the same column
  refetches). Request body and blocker are pure helpers in `lib/diffusion.ts`.
- Axis pickers label diffusion-map columns `DC0…DC(k−1)` (scanpy numbering, so
  `DC1` here is `X_diffmap[:, 1]` in a notebook). `setSelectedEmbedding`
  opens a diffusion map with no chosen axes on `(1, 2)` — guarded on the map
  having ≥ 3 columns, since the backend clamps out-of-range dims and
  `useEmbedding` would refetch forever on the mismatch.
- Subset tree: `CellSubsetDerived` gains `diffmap` / `dpt`; badges and chips
  list them.

## Testing

Pure module: bit-parity with `sc.tl.diffmap` / `sc.tl.dpt` on a connected
symmetric kNN; isolated cells get NaN not zeros; an asymmetric graph is
symmetrised and flagged; a two-component graph reports two stationary
eigenvalues and DPT gives NaN (not inf) across components; centroid root.
Adaptor: names per graph/subset, scope NaN, registry, `diffmap_evals`
compat, every root mode, `stemfinder` orientation, subset tree record and
cascade delete, codegen entry (`test_codegen` requires one per action).
Routes: 400/404 mapping. Frontend: request builders and default-name /
dims helpers in `lib/diffusion.ts` with vitest. Browser: a real dataset on the
isolated stack — diffusion map on the expression graph and the spatial graph,
DPT from stemFinder.
