# Visium HD `feature_slice.h5` loader — design

**Date:** 2026-06-16
**Status:** Approved, proceeding to implementation

## Problem

xcell loads 10x `.h5` files via `scanpy.read_10x_h5`, which expects a standard
feature-barcode matrix (`/matrix/...`). 10x's public Visium HD downloads ship a
compact all-in-one **`*_feature_slice.h5`** instead (root attr
`filetype == "feature_slice"`, produced by `spaceranger`). This file is not a
feature-barcode matrix and `read_10x_h5` rejects it, so xcell cannot open it.

A real example (`Visium_HD_Mouse_Embryo_feature_slice.h5`, 700 MB) ships with
only an images-only `spatial/` folder — no `binned_outputs/`, no
`filtered_feature_bc_matrix.h5`, no `tissue_positions`. So `sc.read_visium`
is not an option; everything needed lives inside the feature-slice file.

## `feature_slice.h5` structure (verified)

- Root attrs: `filetype="feature_slice"`, `metadata_json` (contains `nrows`,
  `ncols`, `spot_pitch` = 2.0 µm, transforms), `software_version`.
- `feature_slices/<gene_index>/{row,col,data}` — per-gene sparse expression on
  the 2 µm grid (`nrows×ncols` = 3350×3350). `data` = integer counts.
- `features/{id,name,feature_type,genome}` — 32,285 genes (gene order == the
  `<gene_index>` used under `feature_slices/`).
- `masks/square_{NNN}um/{row,col,data}` — which bins are tissue, on the
  **bin-size grid** (8 µm → row 0–814, col 0–837, ~344k bins; 16 µm → ~88k).
  `data == 1` for tissue.
- `umis/total/{row,col,data}` — total UMIs per 2 µm bin (not required).
- `secondary_analysis/clustering/square_{NNN}um_gene_expression_{method}/{row,col,data}`
  — precomputed clustering on the bin grid; `data` = cluster label
  (graphclust + kmeans 2..10), at 8 µm and 16 µm.

An `bin_size` µm bin maps from 2 µm coordinates as `(row // factor, col // factor)`
where `factor = bin_size // 2`.

## Approach

New module `backend/xcell/visium_hd.py` (keeps `adaptor.py`, already ~4k lines,
from growing; converter is independently testable). The adaptor's existing `.h5`
branch peeks at the root `filetype` attr and routes feature-slice files here.

### Public API

- `is_feature_slice(path) -> bool` — open file, check
  `attrs.get('filetype') == b'feature_slice'` (reads attrs only, cheap).
- `feature_slice_to_anndata(path, bin_size=8, import_clusters=True) -> AnnData`
  — the pure converter.
- `load_feature_slice_cached(path, bin_size=8, import_clusters=True) -> AnnData`
  — cache wrapper.

### Detection & routing (adaptor.py)

```python
elif self.filepath.suffix == '.h5':
    if is_feature_slice(self.filepath):
        self.adata = load_feature_slice_cached(self.filepath, bin_size=8)
    else:
        self.adata = sc.read_10x_h5(self.filepath)
    self.adata.var_names_make_unique()
```

No changes to `/browse` (already lists `.h5`) or `load_dataset` (already accepts
`.h5`). `bin_size` defaults to 8; a load-time selector is out of scope for this
pass — the function supports the parameter, the route hardcodes 8.

### Conversion algorithm

1. Parse `metadata_json`; `factor = bin_size // 2`; bin grid dims
   `n_bin_rows = ceil(nrows/factor)`, `n_bin_cols = ceil(ncols/factor)`.
2. Read `features/*` → `var` (index = `id`, columns `name`, `feature_type`,
   `genome`). Gene count N defines matrix columns.
3. Read `masks/square_{bin:03d}um` → set of tissue bins `(br, bc)`. Assign each
   a contiguous obs index via `flat = br * n_bin_cols + bc` → ordered obs.
   These bins are the observations.
4. For each gene slice i: read `row,col,data`; `br = row//factor`,
   `bc = col//factor`; map flat bin → obs index (drop entries whose bin is not
   in the tissue mask); accumulate COO triplets `(obs_idx, i, count)`.
5. `scipy.sparse.coo_matrix((data,(obs,gene)), shape=(n_obs, N)).tocsr()`;
   `sum_duplicates()` → integer counts in `.X`.
6. `obsm['spatial']` = bin centroids in µm: `x = bc * bin_size`,
   `y = -(br * bin_size)` (y negated so tissue renders upright, matching image
   orientation). obs_names = `"{br}_{bc}"`.
7. `uns['spatial']` provenance: sample_id, bin_size_um, source filename.

Notes: reading ~32k HDF5 datasets is the slow step (~1–3 min first convert →
cached). Peak memory at 8 µm ~1 GB during COO assembly; 16 µm ~4× lighter.

### Precomputed clusters → obs

When `import_clusters`, for each
`secondary_analysis/clustering/square_{bin}um_*`: read `(row,col,label)`, map to
obs index, fill an int array (missing → `"NA"`), store as an **ordered
categorical** obs column named by the method (`graphclust`, `kmeans_2`..
`kmeans_10`). Immediately colorable in xcell without re-running Leiden.

### Caching

`load_feature_slice_cached`:
- Cache path `<source_stem>.xcell.square_{bin:03d}um.h5ad` next to the source.
- If cache exists and `cache.mtime >= source.mtime` → `read_h5ad` and return.
- Else convert, `write_h5ad`, return. If source dir not writable, fall back to a
  temp-dir cache (log a warning).

## Testing

Synthetic-fixture unit tests (`backend/tests/test_visium_hd.py`): a helper
writes a minimal `feature_slice.h5` (few genes, ~6×6 2 µm grid, an 8 µm mask,
one clustering array) to a temp path. Assertions:
- `is_feature_slice` true for fixture, false for a plain 10x h5 / h5ad.
- Converted shape == (#tissue bins, #genes); hand-computed rebinned counts match.
- `obsm['spatial']` present with one row per obs.
- Cluster obs column present and categorical.
- Cache file created on first load and reused on second (converter not re-run —
  verified via a spy/monkeypatch or mtime).

Plus one manual smoke-load of the real 700 MB file end-to-end (outside the
suite) to confirm it opens in the adaptor.

## Out of scope

- Load-time bin-size selector in the UI/route (function supports it; route uses 8).
- Tissue-image overlay / registration to hires image.
- `sc.read_visium` folder ingestion (not applicable to this file shape).
