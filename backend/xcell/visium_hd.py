"""Load 10x Visium HD ``*_feature_slice.h5`` files into AnnData.

10x's public Visium HD downloads ship a compact, self-contained
``feature_slice.h5`` (root attr ``filetype == "feature_slice"``) rather than a
standard feature-barcode matrix. ``scanpy.read_10x_h5`` cannot read it. This
module rebins the per-gene 2 um expression slices up to a chosen bin size
(8 um by default), keeps only tissue bins, attaches spatial coordinates, and
optionally imports the precomputed clustering — producing an AnnData xcell can
display like any other dataset.

Public API:
    is_feature_slice(path) -> bool
    feature_slice_to_anndata(path, bin_size=8, import_clusters=True) -> AnnData
    load_feature_slice_cached(path, bin_size=8, import_clusters=True) -> AnnData
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def is_feature_slice(path: str | Path) -> bool:
    """Return True if ``path`` is a 10x Visium HD feature_slice.h5.

    Reads only the root attributes, so it is cheap even for large files.
    """
    try:
        with h5py.File(path, "r") as f:
            ft = f.attrs.get("filetype")
    except (OSError, KeyError):
        return False
    if isinstance(ft, bytes):
        ft = ft.decode()
    return ft == "feature_slice"


def _decode(arr) -> list[str]:
    """Decode an HDF5 byte-string dataset into a list of str."""
    out = []
    for v in arr[:]:
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def feature_slice_to_anndata(
    path: str | Path,
    bin_size: int = 8,
    import_clusters: bool = True,
) -> anndata.AnnData:
    """Convert a feature_slice.h5 into a binned AnnData.

    Args:
        path: Path to the feature_slice.h5 file.
        bin_size: Square bin edge length in microns (must be a multiple of the
                  2 um base pitch, e.g. 8 or 16).
        import_clusters: If True, import precomputed clustering arrays for this
                         bin size as ordered-categorical obs columns.

    Returns:
        AnnData with integer counts in ``.X`` (CSR), gene metadata in ``.var``,
        bin centroids (microns) in ``obsm['spatial']``, and provenance in
        ``uns['spatial']``.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        nrows, ncols = int(meta["nrows"]), int(meta["ncols"])
        pitch = float(meta["spot_pitch"])

        factor = int(round(bin_size / pitch))
        if factor < 1 or not math.isclose(factor * pitch, bin_size):
            raise ValueError(
                f"bin_size {bin_size} um is not a multiple of the {pitch} um pitch"
            )
        n_bin_cols = math.ceil(ncols / factor)

        # --- genes (var) ---
        feats = f["features"]
        gene_ids = _decode(feats["id"])
        n_genes = len(gene_ids)
        var = pd.DataFrame(
            {
                "name": _decode(feats["name"]),
                "feature_type": _decode(feats["feature_type"]),
                "genome": _decode(feats["genome"]),
            },
            index=pd.Index(gene_ids, name=None),
        )

        # --- tissue bins (obs), ordered by flattened bin index ---
        mask_key = f"square_{bin_size:03d}um"
        if mask_key not in f["masks"]:
            available = list(f["masks"].keys())
            raise ValueError(
                f"No mask for bin_size {bin_size} um ('{mask_key}'); "
                f"available: {available}"
            )
        mg = f["masks"][mask_key]
        mrow = np.asarray(mg["row"][:], dtype=np.int64)
        mcol = np.asarray(mg["col"][:], dtype=np.int64)
        flat = mrow * n_bin_cols + mcol
        order = np.argsort(flat, kind="stable")
        flat_sorted = flat[order]
        br_sorted = mrow[order]
        bc_sorted = mcol[order]
        n_obs = flat_sorted.size

        # map flattened bin index -> contiguous obs index
        flat_to_obs = {int(fl): i for i, fl in enumerate(flat_sorted)}

        # --- accumulate expression into COO triplets ---
        obs_idx_parts: list[np.ndarray] = []
        gene_idx_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        fs = f["feature_slices"]
        for gi in range(n_genes):
            grp = fs.get(str(gi))
            if grp is None:
                continue
            row = np.asarray(grp["row"][:], dtype=np.int64)
            if row.size == 0:
                continue
            col = np.asarray(grp["col"][:], dtype=np.int64)
            data = np.asarray(grp["data"][:], dtype=np.int64)
            bflat = (row // factor) * n_bin_cols + (col // factor)
            mapped = np.array([flat_to_obs.get(int(b), -1) for b in bflat])
            keep = mapped >= 0
            if not keep.any():
                continue
            obs_idx_parts.append(mapped[keep])
            gene_idx_parts.append(np.full(int(keep.sum()), gi, dtype=np.int64))
            data_parts.append(data[keep])

        if data_parts:
            obs_idx = np.concatenate(obs_idx_parts)
            gene_idx = np.concatenate(gene_idx_parts)
            data = np.concatenate(data_parts)
        else:
            obs_idx = np.empty(0, dtype=np.int64)
            gene_idx = np.empty(0, dtype=np.int64)
            data = np.empty(0, dtype=np.int64)

        X = coo_matrix(
            (data, (obs_idx, gene_idx)),
            shape=(n_obs, n_genes),
            dtype=np.int32,
        ).tocsr()
        X.sum_duplicates()

        # --- obs: names + spatial coords ---
        obs_names = [f"{int(r)}_{int(c)}" for r, c in zip(br_sorted, bc_sorted)]
        obs = pd.DataFrame(index=pd.Index(obs_names, name=None))
        # x = col * bin_size, y = -(row * bin_size) so tissue renders upright.
        spatial = np.column_stack(
            [bc_sorted * bin_size, -(br_sorted * bin_size)]
        ).astype(np.float64)

        # --- precomputed clustering -> obs categoricals ---
        if import_clusters and "secondary_analysis" in f:
            clustering = f["secondary_analysis"].get("clustering")
            if clustering is not None:
                prefix = f"square_{bin_size:03d}um_gene_expression_"
                for name in clustering.keys():
                    if not name.startswith(prefix):
                        continue
                    cg = clustering[name]
                    crow = np.asarray(cg["row"][:], dtype=np.int64)
                    ccol = np.asarray(cg["col"][:], dtype=np.int64)
                    clabel = np.asarray(cg["data"][:])
                    cflat = crow * n_bin_cols + ccol
                    values = np.array(["NA"] * n_obs, dtype=object)
                    for fl, lab in zip(cflat, clabel):
                        oi = flat_to_obs.get(int(fl))
                        if oi is not None:
                            values[oi] = str(int(lab))
                    col_name = name[len(prefix):]
                    obs[col_name] = pd.Categorical(values, ordered=True)

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.uns["spatial"] = {
        "sample_id": meta.get("sample_id"),
        "bin_size_um": bin_size,
        "source": path.name,
    }
    return adata


def _cache_path(path: Path, bin_size: int) -> Path:
    return path.with_name(f"{path.stem}.xcell.square_{bin_size:03d}um.h5ad")


def load_feature_slice_cached(
    path: str | Path,
    bin_size: int = 8,
    import_clusters: bool = True,
) -> anndata.AnnData:
    """Convert a feature_slice.h5, caching the result as an .h5ad.

    The cache lives next to the source file; if the source directory is not
    writable, falls back to the system temp directory. A cache is reused only
    when it is at least as new as the source file.
    """
    path = Path(path)
    cache = _cache_path(path, bin_size)

    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return anndata.read_h5ad(cache)

    adata = feature_slice_to_anndata(
        path, bin_size=bin_size, import_clusters=import_clusters
    )

    try:
        adata.write_h5ad(cache)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / cache.name
        print(
            f"[xcell] feature_slice cache dir not writable; caching to {fallback}"
        )
        try:
            adata.write_h5ad(fallback)
        except OSError as e:
            print(f"[xcell] feature_slice cache write skipped: {e}")
    return adata
