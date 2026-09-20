# A first-class subset tree in AnnData — with its embeddings and decorations

Date: 2026-09-20. Status: agreed direction (the tree); extended the same day to
cover embeddings and embedding-specific decorations, and built without further
review at Patrick's request. Decisions he did not make are marked **[decision]**.
Follows `2026-09-19-named-cell-subsets-design.md` (shipped, `main` at `8aa3715`).

## Why

Patrick's workflow is recursive: cluster the whole dataset, then for chosen
top-level clusters run subset → HVG → PCA → (drop PCs) → kNN → Leiden → UMAP,
and repeat inside the finer clusters. A survey on 2026-09-20 found nothing in
the AnnData ecosystem built for this. Closest: MuData with `axis=-1`
("data subsets" sharing obs and var, one AnnData per subset, `.h5mu`);
Scarf's Zarr store (results namespaced by cell key + feature key); CHOIR and
IDclust (R only; CHOIR recomputes features and a reduction per subtree and
stores `P0_reduction`, `P1_reduction`… in Seurat `misc`); TileDB-SOMA
(nested collections); treedata (trees over obs, no per-node matrices).

**Decision:** keep AnnData/h5ad as the truth and make the hierarchy explicit
in a registry, the CHOIR `misc` record in AnnData terms. Yesterday's suffixed
keys stay; the registry records the tree instead of inferring it from names.

## The extension: embeddings and decorations

A subset node owns embeddings — `X_pca_<name>`, `X_umap_<name>`, a UMAP over
another graph (`X_umap_<name>_spatial`), and PC subsets `X_pca_<name>_<suffix>`
— and the things drawn *on* those embeddings. xcell has three kinds of
embedding-specific decoration:

| decoration | where it lives today | keyed by |
|---|---|---|
| shapes and lines (`drawnLines`) | browser store only; `uns['xcell_lines_json']` at export | `embeddingName` (+ `dimX`/`dimY`) |
| territories | `uns['xcell_territories_json']` | `embedding` |
| embedding transforms | applied to `obsm` in place; shapes follow via `transformShapesForEmbedding` | embedding name |

(Contours, sections, highlight overlays, label overlays and snapshots are
either cell-keyed or ephemeral view state; none names an embedding.)

**Decorations belong to embeddings, not to subsets.** The registry never
copies them. The tree resolves "this node's decorations" by key at listing
time, so a shape on the parent's UMAP is never shown on the child's (which is
already how rendering works), and nothing can drift.

**Shapes and lines persist like territories.** They are lost on reload today,
which would make the tree's account of them hollow across sessions. The
existing `POST /api/lines` sync now writes `uns['xcell_lines_json']` (the full
line: id, name, embedding, dims, points, smoothed points, draw type, closed,
visibility, stroke, fill — not projections, which are recomputed), the adaptor
hydrates from it on load, `GET /api/lines` returns it, and the browser hydrates
its store when a dataset loads and syncs, debounced, on every change.

**Deleting a subset with its results cascades.** Dropping `X_umap_<name>`
also drops territories and shapes on it (their geometry; a territory's
`territory_<type>` obs columns stay, as `delete_territories` already leaves
them), and the subset's PC subsets. The confirm row names what goes:
"X_umap_chondro, 2 shapes, 1 territory type".

**A vanished primary embedding is repaired.** Today the picker is left naming
a key the schema no longer lists (the split pane already repairs itself).
`setSchema` now re-picks by the load-time preference rule when the selected
embedding is gone.

## The registry

`uns['xcell_cell_subsets'][name]` stays a plain dict (readable in scanpy);
arrays that h5ad makes of empty lists are normalised back on read.

```
{
  obs_key, created_at, description,
  parent: '<name>'           # absent for a root
  origin: {kind: 'selection', embedding: '<name on screen when saved>'},
  derived: {
    hvg:   {key, params},              # one per subset
    pca:   {key, params},
    graph: {key, params},
    umap:  {'<obsm key>': params, …},  # keyed by output, so a re-run overwrites
    leiden:{'<obs col>': params, …},
    pca_subsets: {'<obsm key>': {dropped_pcs: [...]}},
  }
}
```

- **[decision] Parent is inferred by containment.** At creation, the smallest
  registered subset containing every cell of the new one is its parent (ties
  → most recently created); none → root. An explicit `parent` is accepted and
  validated (it must contain the cells) for scripts and the notebook. The
  browser clears `activeSubsetName` on any mask edit, so it cannot say where a
  mask came from; containment answers the same question and is right for the
  recursive workflow by construction.
- **[decision] Origin records only the embedding on screen.** The store does
  not track how a mask was built (lasso, category, threshold); the embedding
  it was saved on is what the tree can act on ("view where it was drawn").
  Other origin kinds can be added when a flow produces them.
- `derived` is written *at run time* by each scoped operation, with the flat
  scalar params that ran (no nested neighbours meta). The naming-based
  detection stays as the fallback for files made before this, merged with the
  record and filtered to keys that still exist.

`list_cell_subsets` returns, per subset: the existing summary (`derived` with
`umap` now a list of every UMAP the subset owns) plus `parent`, `children`,
`depth`, `origin`, `steps` (the recorded params), `embeddings` (every obsm key
it owns that exists) and `decorations: {lines: [names], territories: [types]}`.

## PCA Loadings and PC subsets on a subset

`get_pca_loadings(cell_subset=)` reads `PCs_<name>` / `uns['pca_<name>']`;
the route takes `?cell_subset=` and `usePcaLoadings` passes the active subset;
the `pca_with_loadings` prerequisite accepts the subset's keys.
`create_pca_subset(cell_subset=)` derives from `X_pca_<name>` and writes
`X_pca_<name>_<suffix>` / `PCs_<name>_<suffix>` /
`uns['pca_<name>']['variance_ratio_<suffix>']` and `['subsets'][suffix]`, and
records `derived.pca_subsets`. `list_pca_subsets(cell_subset=)` lists a
subset's own; the dataset's list no longer shows subset PCAs as PC subsets
(today `X_pca_chondro` appears there with nothing dropped — a bug).
`delete_pca_subset` finds the owner by prefix. `run_pca` on the dataset spares
anything under a registered subset's prefix; `run_pca` on a subset clears that
subset's own PC subsets, as the dataset run clears the dataset's. The PC-source
picker lists the active subset's PC subsets.

## Cells panel

Subset rows form a tree: depth-first order, indented per depth. Each row keeps
name, count, *Activate* and ✕, and its badges become chips: HVG · PCA · kNN
(plain), one chip per embedding (click switches the plot to it), one per
Leiden column (click colours by it; a ⋯ offers *Refine parent labels with
this…*, which opens the existing Transfer Labels modal with the parent's
Leiden column — or the dataset's — as the target and this column as the
source), and `N shapes` / `N territories` chips that switch to the embedding
they are on. The "+ results" confirm names the embeddings, shapes and
territories that will go.

## Notebook

`create_cell_subset` emits `parent=` when set; scoped steps already emit
`xa.run_*(…, cell_subset=…)`; `create_pca_subset(…, cell_subset=…)` likewise.

## Delete an `.obs` column (separate request, same session)

*Delete* appears next to *Hide* on hover for categorical and continuous rows
and on hidden rows, with an inline confirm like the subset delete. The backend
`delete_obs_column` drops the column and its `uns['<col>_colors']`, removes
the registry entry when the column is a subset's, logs the step (the notebook
emits `del adata.obs['…']`), and is reachable at `DELETE /api/obs/{column}`
(the old `/annotations/{name}` route stays). The browser refreshes both
channels and clears colour-by, the label overlay and hidden state if they
named the column, and refreshes subsets if it was one.

## Later

`POST /api/export/mudata` writing one AnnData per subset into an `.h5mu` with
`axis=-1`, each node a normal scanpy object.

## Constraints learned the hard way

- A subset run must always write `<key>_<name>` whatever `graph_key` arrives
  (`_subset_output_name`); the modal sends `connectivities` explicitly.
- `sc.tl.pca` re-applies `mask_var='highly_variable'` inside an explicit gene
  subset; pass `mask_var=None` there and clamp `n_comps` to genes used.
- anndata 0.12 writes `None` and empty lists into uns, but an empty list reads
  back as `array([], dtype=float64)` — normalise on read, never trust the type.
- Marker genes and the heatmap still ignore the mask (no cell-index list).
- Verify in the browser on the isolated stack (`/private/tmp/claude-501/xcell-wt`,
  `:8100` / `:5273`); his `--reload` backend on `:8000` restarts on any
  main-repo `backend/*.py` write. Check `GET :8000/api/schema` before merging.
