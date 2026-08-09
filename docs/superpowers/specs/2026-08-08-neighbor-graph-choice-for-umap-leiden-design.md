# Choosing the neighbor graph for UMAP and Leiden

**Date:** 2026-08-08
**Status:** Approved, ready to implement

## Problem

xcell can build several cell-cell connectivity graphs over the same dataset:

| graph | `obsp` key | built by |
|---|---|---|
| expression kNN | `connectivities` | Neighbors (`sc.pp.neighbors`) |
| spatial kNN | `spatial_connectivities` | Spatial Neighbors (`sq.gr.spatial_neighbors`) |
| weighted blend | `<name>_connectivities` | Combine Neighbors |

`list_neighbor_graphs()` already enumerates them, `/api/scanpy/neighbor_graphs`
already serves them, and Smooth already lets you pick one via the
`graph_select` field type.

UMAP and Leiden cannot. Both read whatever sits in `obsp['connectivities']`:
`run_umap` calls `sc.tl.umap(self.adata, ...)` with no `neighbors_key`, and
`run_leiden` calls `sc.tl.leiden(self.adata, ...)` with no `obsp`. So:

- **You cannot embed or cluster on the spatial graph at all.** Spatial domain
  detection — cluster spots by who they sit next to — is a standard analysis
  and is currently unreachable.
- **The only way to use a combined graph is to destroy the expression one.**
  `combine_neighbor_graphs(target_key='connectivities')` overwrites it in
  place. That is the documented workaround, and it means you cannot compare an
  expression clustering against a spatially-informed one without re-running
  Neighbors.
- **Nothing records which graph produced the current `X_umap`.** After a
  Combine Neighbors run that targeted `connectivities`, `X_umap` silently means
  something different than it did before, with no trace in the name.

## Approach

Add a `graph_key` parameter to both operations, and name the output after the
graph so several can coexist.

### `sc.tl.umap` needs its metadata repaired first

scanpy exposes `neighbors_key`, which names a key in `uns` holding
`{connectivities_key, distances_key, params}`. Two things break when you point
it at the graphs xcell actually has (verified against scanpy 1.11.5, squidpy):

1. **`uns['spatial_neighbors']` → `KeyError: 'method'`.** scanpy's `tl.umap`
   evaluates `neighbors["params"]["method"] != "umap"` to decide whether to
   warn. squidpy's params dict holds `n_neighbors`, `coord_type`, `radius`,
   `transform` — no `method`. The lookup is unguarded, so the warning check
   raises before UMAP starts.
2. **A bare `obsp` key → `KeyError: 'distances_key'`.** `combine_neighbor_graphs`
   writes only the matrix, no `uns` entry. Constructing a minimal one still
   fails: `NeighborsView` requires `distances_key` to be present. It does *not*
   require the matrix it names to exist — UMAP reads connectivities only — so
   the fix is about the shape of the dict, not about materializing distances.

So `run_umap` synthesizes a complete entry, uses it, and removes it:

```python
_GRAPH_META_KEY = '_xcell_graph'   # module-level constant, shared with codegen

self.adata.uns[_GRAPH_META_KEY] = self._umap_neighbors_meta(graph_key)
try:
    sc.tl.umap(self.adata, ..., neighbors_key=_GRAPH_META_KEY, key_added=name)
finally:
    self.adata.uns.pop(_GRAPH_META_KEY, None)
```

`_umap_neighbors_meta` starts from any existing `uns` entry whose
`connectivities_key` equals `graph_key` — so squidpy's recorded `n_neighbors`
and `coord_type` survive into the exported notebook rather than being replaced
by invented values — then force-fills the three fields scanpy reads unguarded:

- `connectivities_key` = `graph_key`
- `distances_key` = the sibling `<prefix>_distances`, whether or not it exists
- `params['method'] = 'umap'`

`use_rep` is left alone if the source entry had one; omitted otherwise, in
which case scanpy's `_choose_representation` falls back to `X_pca`. Verified:
a synthesized entry with `params={'method': 'umap'}` and a `distances_key`
naming a nonexistent matrix runs to completion.

The temp key never persists. It is deleted in a `finally`, so a UMAP that
raises does not leave a bogus neighbors entry behind for the next tool to find.

### Leiden needs none of this

`sc.tl.leiden` takes `obsp=` directly and reads that matrix as the adjacency.
`run_leiden` passes it through. (`obsp` and `neighbors_key` are mutually
exclusive in scanpy; only `obsp` is ever passed.)

### Naming: one rule, one place

`list_neighbor_graphs()` gains a `suffix` field per graph, derived by a single
private helper:

| `obsp` key | suffix | UMAP default | Leiden default |
|---|---|---|---|
| `connectivities` | `''` | `X_umap` | `leiden` |
| `spatial_connectivities` | `spatial` | `X_umap_spatial` | `leiden_spatial` |
| `expr_plus_space_connectivities` | `expr_plus_space` | `X_umap_expr_plus_space` | `leiden_expr_plus_space` |

The frontend prefills the name field from `suffix`; the backend derives the
same default when `key_added` is omitted. The rule lives in one helper that
both call, so they cannot drift.

The default graph keeps writing `X_umap` and `leiden`. Nothing changes for a
user who never opens the new dropdown — including the codegen output and the
Analysis Record, which stay byte-identical for existing recordings.

### Prerequisites

`check_prerequisites('umap')` and `('leiden')` both require a prior `neighbors`
run, i.e. that `obsp['connectivities']` exists. A Visium dataset where the user
ran only Spatial Neighbors has no such graph — and that is precisely the case
this feature exists for. When a valid `graph_key` is supplied, the prerequisite
check is skipped and the graph's own existence and shape are validated instead.

### Cell selections

Both methods, when given `active_cell_indices`, build a subset AnnData by
slicing `obsp` — and both hardcode `['connectivities', 'distances']`. With a
`graph_key` set, that slices the wrong matrix (or, on a spatial-only dataset,
nothing at all) and the run either uses stale data or fails. Both loops slice
the chosen graph instead.

## API

**Adaptor**

```python
def run_umap(self, min_dist=0.5, spread=1.0, n_components=2,
             graph_key=None, key_added=None, active_cell_indices=None)

def run_leiden(self, resolution=1.0, key_added='leiden',
               graph_key=None, active_cell_indices=None)
```

`graph_key=None` reproduces today's behaviour exactly, down to the recorded
params. Result dicts gain `graph_key` and report the real output name in the
existing `embedding_name` / `key_added` fields.

**Validation** (`ValueError` → 400, raised before any work):

- `graph_key` not in `obsp` → message lists what is available
- graph shape ≠ `(n_cells, n_cells)`
- `key_added` blank or whitespace-only

**Routes.** `UmapRequest` gains `graph_key: str | None = None` and
`key_added: str | None = None`; `LeidenRequest` gains `graph_key: str | None =
None`. `/api/scanpy/neighbor_graphs` entries gain `suffix`.

## Codegen

The exported notebook must run standalone, so `neighbors_key='<temp>'` cannot
be emitted on its own — the temp entry does not exist there. The `umap`
ActionSpec emits the synthesis as real lines ahead of the call:

```python
adata.uns['_xcell_graph'] = {
    'connectivities_key': 'spatial_connectivities',
    'distances_key': 'spatial_distances',
    'params': {'method': 'umap', 'n_neighbors': 8},
}
sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2,
           neighbors_key='_xcell_graph', key_added='X_umap_spatial')
```

That is what the adaptor really does, so fidelity stays `exact`. With no
`graph_key`, the emitted line is unchanged from today. `leiden` appends
`obsp='<key>'` and stays a one-liner.

## Frontend

`ScanpyModal.tsx`:

- `umap` params gain `graph_key` (`graph_select`) and `key_added` (`text`).
- `leiden` params gain `graph_key`.
- The `needsGraphs` allowlist (~line 883) gains `umap` and `leiden`, so the
  dropdown is populated when those tabs open.
- `graph_select` gains an optional `emptyLabel` on the param def. Smooth keeps
  "— pick a graph —" (a graph is required there); UMAP and Leiden show
  "Expression neighbors (default)", because `''` is a valid choice meaning
  "whatever `uns['neighbors']` points at".
- Changing `graph_key` re-prefills the name field from the graph's `suffix`,
  unless the user has typed over it. Tracked with a ref holding the last
  auto-generated value: if the field still equals it, replace it; otherwise
  leave it alone.

Refresh channels are already correct — the existing allowlists include `umap`
(schema refresh, for the new `.obsm` entry) and `leiden` (obs summaries, for
the new column). The result banner reads the output name from the response
rather than assuming `X_umap`.

## Tests

New self-contained `backend/tests/test_graph_choice.py`:

- UMAP on `spatial_connectivities` writes `obsm['X_umap_spatial']`, leaves
  `X_umap` untouched, and leaves no temp key in `uns`.
- UMAP on a combined graph that has **no** `uns` entry at all — the
  `distances_key` case — succeeds.
- The temp key is removed even when `sc.tl.umap` raises.
- Leiden on a spatial graph writes `leiden_spatial` with ≥2 clusters and does
  not touch `leiden`.
- `graph_key=None` produces byte-identical recorded params to today.
- Unknown `graph_key`, wrong-shape graph, and blank `key_added` each raise
  `ValueError`.
- Suffix derivation for all three key shapes.
- Subset path slices the chosen graph, not `connectivities`.
- Works on a dataset with spatial neighbors and **no** `connectivities` — the
  prerequisite-skip case.

Plus `test_codegen.py` cases for both emissions, and route tests for the new
request fields and the `suffix` field on `/neighbor_graphs`.

## Out of scope

- Migrating `combine_neighbor_graphs` to write proper `uns` entries. The
  synthesis handles graphs that lack them, which is what makes this work for
  graphs that already exist in a user's saved h5ad.
- Diffusion map, PAGA, and other neighbor-graph consumers. Same limitation,
  same fix shape, not asked for.
