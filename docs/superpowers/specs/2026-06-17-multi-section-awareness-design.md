# Multi-section awareness — exploration + section-aware (multi)contour design

**Date:** 2026-06-17
**Status:** Design — section-aware (multi)contour approved; broader roadmap for review

## Background

A spatial dataset can hold several **distinct tissue sections** in one coordinate
space — e.g. three cuts of an E11.5 forelimb, or any set combined via
`combine_spatial_h5ads` (which lays sections left-to-right with a gap and tags
each cell with `obs['sample']`). The new bundled `toy_spatial_3sections.h5ad`
(`obs['section']` = section_1/2/3) is the test vehicle.

**The core invariant such data violates:** Euclidean distance *between* cells on
*different* sections is meaningless. Any analysis that treats the global `(x, y)`
plane as one continuous tissue will silently couple distinct sections across the
gaps. Today **no** xcell analysis consults the `section`/`sample` column.

## Exploration — implications across analyses

Audited every operation that uses spatial coordinates or cross-cell
distance/neighbor graphs. Grouped by the *mechanism* of cross-section coupling.

### A. Geometry-bridging (HIGH severity — silently wrong)

| Operation | File:line | Mechanism of cross-section coupling |
|---|---|---|
| **contourize** | `adaptor.py:_contour_score_field` (~136), `run_contourize` | `griddata(..., 'cubic')` builds one grid over `[x.min,x.max]×[y.min,y.max]` → **interpolates expression across the gap**; `gaussian_filter` then diffuses it further. Manufactures continuity that isn't there. |
| **multicontour** | `multicontour.py:score_module`, `adaptor.finalize_multicontour` | Same griddata/smoothing in Phase 1; Phase 2 conflict resolution picks neighbors from the spatial graph **or** a global coordinate-kNN — both can pull neighbors from another section. |
| **spatial_neighbors** | `adaptor.prepare_spatial_neighbors` (~5397) | `squidpy.gr.spatial_neighbors` on full coords. **Delaunay** triangulates the convex hull → always bridges sections; **radius ≥ gap** connects edge spots; **kNN** can pick cross-section neighbors at edges. This graph is the root dependency for ↓. |
| **spatial_autocorr** | `adaptor.prepare_spatial_autocorr` (~5559) | Moran's I / Geary's C over `spatial_connectivities` → inherits every spurious cross-section edge; statistic is corrupted. |
| **run_smooth** (on a spatial graph) | `adaptor.run_smooth` (~4210) | Averages expression over neighbor edges; cross-section edges bleed signal between sections. |
| **combine_neighbor_graphs** | `adaptor.combine_neighbor_graphs` (~4066) | If `spatial_connectivities` is a source, the combined graph inherits its bad edges (weights don't remove them). |

### B. Line-geometry (MEDIUM — context-dependent)

| Operation | File:line | Mechanism |
|---|---|---|
| `_project_cells_onto_line` | `adaptor.py:~1695` | Projects cells onto a polyline in an embedding. A line spanning the gap (or coincidental position overlap between sections) assigns cross-section-comparable positions/distances. |
| `test_line_association`, `test_multi_line_association` | `adaptor.py:~2246/~2374` | Fit one (pooled) B-spline of expression vs. line position across all selected cells → conflates within-section gradient with between-section variation. Multi-line *pools across sections* by design, which is right only if the same axis is meant in each. |
| heatmap line ordering | `heatmap.py:~215` | Orders cells by line position/distance; can interleave cells from different sections. |

### C. Expression-space (LOW — a batch concern, not a distance bug)

`run_neighbors` (PCA-kNN), `leiden`, `umap`, `diffexp`, `marker_genes`,
`cluster_gene_set`, gene graph/`cluster_genes` — these use expression, not
geometry, so the gap doesn't bridge them. Multi-section data can still introduce
**batch effects** (a section's technical signal dominating PCs), but that's a
distinct problem (batch correction / Harmony-style) and out of scope here.

### Takeaways

1. **One root fix has outsized leverage:** making `spatial_neighbors` build a
   *block-diagonal* (per-section) graph fixes spatial_autocorr, run_smooth,
   combine_neighbor_graphs, and multicontour's conflict step at once.
2. **(multi)contour needs its own fix** because it interpolates directly from
   coordinates, not from the neighbor graph.
3. **Convention:** a single optional `section_col` (an `obs` categorical),
   auto-detected from `section` then `sample`, threads through all spatial ops.
   `None`/absent → today's global behavior (full backward compatibility).

## Approved design — section-aware (multi)contour

### Principle

Keep **value semantics global**, make the **spatial step per-section**:
gene normalization (log1p → percentile-clip → min-max) and the band thresholds
stay computed over all cells (so "high" means the same thing in every section
and bands are comparable), but the **grid interpolation + Gaussian smoothing run
independently within each section**, so nothing bleeds across a gap.

### Backend

**`_contour_score_field(coords, gene_expr, ..., sections=None)`** (`adaptor.py`):
- Steps 1–2 (normalize per gene, average → per-cell `summary`) unchanged, global.
- If `sections is None`: current single-grid path.
- Else: for each unique section label, take that section's cell indices, run
  griddata-cubic + gaussian_filter + nearest-sample **over only those cells'
  coords/bbox**, and scatter the results back into a full-length `cell_vals`.
- `vmax = max(cell_vals)` over all cells (global) so thresholds are comparable.
- **Small-section guard:** a section with `< 4` cells (cubic needs a
  triangulation) skips interpolation and uses its raw `summary` values; griddata
  failures fall back `cubic → linear → nearest`.

**`multicontour.score_module(adata, genes, ..., sections=None)`** passes
`sections` through to `_contour_score_field`.

**`multicontour.assign_tissue(..., sections=None)`** + `_neighbor_lists`:
when `sections` given, drop any neighbor whose section differs from the query
spot's (both for the `spatial_connectivities` path and the coordinate-kNN
fallback). Conflict resolution then only ever votes with same-section neighbors.

**Adaptor methods** gain `section_col: str | None = None`:
`prepare_contourize`, `run_contourize`, `prepare_multicontour`,
`finalize_multicontour`. When set, resolve `sections = adata.obs[section_col]`
(validate it exists and is categorical-ish); pass arrays down. `finalize` already
has `params`/cache — store `section_col` there so finalize uses the same sections
as prepare.

### API

Add `section_col: str | None = None` to `ContourizeRequest`,
`MultiContourPrepareRequest`, `MultiContourFinalizeRequest`. No new endpoints.

### Frontend (Contour modal)

- A **Section column** dropdown in the select phase: options = categorical `.obs`
  columns + "(treat as one tissue)". **Default:** auto-select `section`, else
  `sample`, else none — with a one-line hint ("Detected 3 sections — contoured
  independently"). Thread the chosen value into the prepare/contourize/finalize
  payloads.

### Testing (TDD)

In `test_multicontour.py` / `test_contour_score.py`, on a 2-section synthetic
(two blobs separated by a gap, a gene high in section A only):
- Without `sections`: a section-B cell near the gap gets a non-trivial score
  (bleed) — characterizes the current behavior.
- With `sections`: section-B cells score ≈ 0 (no bleed); section-A scores
  unchanged vs. running A alone.
- `assign_tissue` with `sections`: a conflict spot never adopts a label whose
  only supporting neighbors are in another section.
- Small-section guard: a 2-cell section doesn't crash and returns finite scores.

### Out of scope (this spec)

- Per-section value normalization (global is the default; revisit if batch
  intensity differences mask "high" in dim sections).
- Section-aware `spatial_neighbors` and the rest of group A — see roadmap.

## Roadmap for the rest (recommended order, not yet approved)

1. **Section-aware `spatial_neighbors`** (highest leverage): add `section_col`;
   build the graph per section and block-diagonal-concatenate (or post-filter
   cross-section edges from squidpy's output). Auto-fixes autocorr, smooth,
   combine. ~Medium effort, big payoff.
2. **Guardrails on line ops + heatmap ordering**: warn (or optionally restrict)
   when a line's projected cells span multiple sections; document multi-line
   pooling semantics.
3. **Docs**: a "Working with multi-section data" section enumerating which
   analyses are section-aware and which still assume one tissue.
4. (Later) batch-effect handling for expression-space analyses — separate effort.
