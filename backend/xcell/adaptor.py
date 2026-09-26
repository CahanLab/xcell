"""DataAdaptor class for wrapping AnnData objects.

This module provides a clean interface for accessing single-cell data,
following the adaptor pattern used in excellxgene. It abstracts away
direct AnnData access and is designed for easy integration with scanpy
analysis functions.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from .analysis_record import AnalysisRecord
from .diffexp import compute_diffexp


def _nullable(arr: np.ndarray) -> list:
    """Coordinates as nested lists, with non-finite values as ``None``.

    NaN is a real value in an embedding — ``run_pca`` / ``run_umap`` on a cell
    selection leave inactive cells NaN, and Localize leaves unplaceable cells
    NaN rather than inventing a position. It is not valid JSON, though, and
    starlette serializes strictly, so returning it raised
    ``ValueError: Out of range float values are not JSON compliant`` and the
    embedding could not be displayed at all. ``null`` is the honest wire
    representation of "this cell has no position".
    """
    a = np.asarray(arr, dtype=float)
    if np.isfinite(a).all():
        return a.tolist()          # the common path, untouched
    out = a.tolist()
    if a.ndim == 1:
        return [None if not np.isfinite(v) else v for v in out]
    return [[None if not np.isfinite(v) else v for v in row] for row in out]


def _combine_spatial_key(a: "anndata.AnnData") -> str | None:
    """Which .obsm array holds this input's spatial coordinates, if any.

    Same precedence the adaptor's own detection uses: an explicit
    ``uns['xcell_spatial_key']`` (an xcell export says which array it meant),
    then the ``spatial`` convention, then ``X_spatial``.
    """
    candidates = []
    explicit = a.uns.get('xcell_spatial_key')
    if isinstance(explicit, str):
        candidates.append(explicit)
    candidates += ['spatial', 'X_spatial']
    for key in candidates:
        if key in a.obsm:
            arr = a.obsm[key]
            if getattr(arr, 'ndim', 0) == 2 and arr.shape[1] >= 2:
                return key
    return None


def _read_10x_mex(matrix: Path, barcodes: Path, features: Path) -> anndata.AnnData:
    """Read a 10x MEX trio from explicit paths to its three files.

    scanpy's ``read_10x_mtx`` infers the layout from file names and accepts
    exactly two: uncompressed ``matrix.mtx`` + ``genes.tsv`` (Cell Ranger v2)
    or gzipped ``matrix.mtx.gz`` + ``features.tsv.gz`` (v3+). GEO gzips v2
    output as submitted, giving ``genes.tsv.gz``, which fits neither branch
    and sends scanpy looking for a ``features.tsv.gz`` that does not exist.
    Reading the files directly makes compression and the genes/features name
    independent; the feature file's column count says which vintage it is.

    Otherwise mirrors scanpy's defaults so var_names are stable across the
    two readers: symbols as var_names (made unique before any filtering),
    ids in ``var['gene_ids']``, and only 'Gene Expression' rows kept when a
    feature-type column is present.
    """
    import pandas as pd

    a = sc.read(matrix).T  # MEX is genes x cells; keep scanpy's transposed layout
    # dtype=str + keep_default_na so a gene literally called "NA" survives.
    feats = pd.read_csv(features, header=None, sep='\t', dtype=str, keep_default_na=False)
    if feats.shape[1] < 2:
        raise ValueError(
            f"{features.name}: expected at least 2 tab-separated columns "
            f"(gene id, symbol), found {feats.shape[1]}")
    if len(feats) != a.n_vars:
        raise ValueError(
            f"{features.name} lists {len(feats)} genes but {matrix.name} has {a.n_vars} rows")
    a.var_names = pd.Index(feats[1].values)
    a.var['gene_ids'] = feats[0].values
    a.var_names_make_unique()
    if feats.shape[1] >= 3:
        a.var['feature_types'] = feats[2].values
        a = a[:, a.var['feature_types'] == 'Gene Expression'].copy()
    bars = pd.read_csv(barcodes, header=None, sep='\t', dtype=str, keep_default_na=False)
    if len(bars) != a.n_obs:
        raise ValueError(
            f"{barcodes.name} lists {len(bars)} barcodes but {matrix.name} has {a.n_obs} columns")
    a.obs_names = pd.Index(bars[0].values)
    return a


def load_dataset_file(path: Path) -> tuple[anndata.AnnData, str]:
    """Load a dataset in any on-disk format xcell supports.

    The single sniffing chain behind both File -> Load and File -> Combine:
    10x CellRanger matrix directory, prefixed *_matrix.mtx(.gz) trio,
    .h5 (Visium HD feature_slice.h5 rebinned to 8 um, or a plain 10x
    feature-barcode matrix), and .h5ad for everything else. (.rds arrives
    here already converted — the R subprocess lives at the route layer.)

    Returns:
        (adata, source_kind) where source_kind is one of '10x_mtx',
        '10x_h5', 'h5ad' — the key the exported notebook uses to re-open
        the file the same way.
    """
    path = Path(path)
    if path.is_dir():
        files = DataAdaptor._find_10x_dir_files(path)
        if files is None:
            raise ValueError(
                f"{path.name} is not a 10x matrix folder: needs matrix.mtx, "
                "barcodes.tsv and features.tsv or genes.tsv (each optionally .gz)")
        return _read_10x_mex(*files), '10x_mtx'
    trio = DataAdaptor._find_10x_trio_files(path)
    if trio is not None:
        return _read_10x_mex(*trio), '10x_mtx'
    if path.suffix == '.h5':
        from xcell.visium_hd import is_feature_slice, load_feature_slice_cached
        if is_feature_slice(path):
            # 10x Visium HD feature_slice.h5 — not a feature-barcode
            # matrix; rebin to 8 um and attach spatial coords/clusters.
            return load_feature_slice_cached(path, bin_size=8), '10x_h5'
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        return a, '10x_h5'
    return anndata.read_h5ad(path), 'h5ad'


#: What a policy value may be, per annotation axis.
_OBS_POLICIES = ('merge', 'separate', 'drop')
_VAR_POLICIES = ('first', 'separate', 'drop')

#: Integer columns with at most this many distinct values read as cluster ids
#: rather than measurements, so they default to being kept apart.
_LABEL_LIKE_MAX_CARDINALITY = 50


def _column_kind(series: pd.Series) -> str:
    """Coarse type used to guess whether a column is a label or a measurement."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return 'category'
    if pd.api.types.is_bool_dtype(series):
        return 'bool'
    if pd.api.types.is_float_dtype(series):
        return 'numeric'
    if pd.api.types.is_integer_dtype(series):
        return 'integer'
    return 'string'


def _suggest_obs_policy(datasets, kind, n_unique) -> tuple[str, str]:
    """Default handling for one .obs column, with the reason to show the user.

    Merging is only safe when a value means the same thing in every dataset. A
    per-cell measurement does; a cluster id from an independent clustering does
    not, and pooling those invents a comparison nobody made.
    """
    if len(datasets) < 2:
        return 'merge', f"only {datasets[0]} has it — nothing to collide with"
    if kind in ('category', 'string'):
        return 'separate', 'labels from independent analyses are not comparable'
    if kind == 'integer' and n_unique is not None and n_unique <= _LABEL_LIKE_MAX_CARDINALITY:
        return 'separate', f'looks like ids, not a measurement ({n_unique} distinct values)'
    return 'merge', 'a per-cell measurement, comparable across datasets'


def _suggest_var_policy(datasets, identical) -> tuple[str, str]:
    """Default handling for one .var column. Genes are shared, so unlike .obs
    the values can be compared directly — and if they already agree everywhere,
    one copy is the whole story."""
    if len(datasets) < 2:
        return 'first', f"only {datasets[0]} has it"
    if identical:
        return 'first', 'identical in every dataset'
    return 'separate', 'each dataset annotates these genes differently'


class _NameAllocator:
    """Hands out column names, never reusing one already taken."""

    def __init__(self, reserved):
        self.taken = set(reserved)

    def take(self, name: str) -> str:
        if name not in self.taken:
            self.taken.add(name)
            return name
        i = 1
        while f'{name}_{i}' in self.taken:
            i += 1
        out = f'{name}_{i}'
        self.taken.add(out)
        return out


def _read_annotations(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(.obs, .var) of a dataset without pulling the matrix into memory."""
    if Path(path).suffix.lower() == '.h5ad':
        backed = anndata.read_h5ad(path, backed='r')
        try:
            return backed.obs.copy(), backed.var.copy()
        finally:
            if backed.file is not None:
                backed.file.close()
    a, _kind = load_dataset_file(Path(path))
    return a.obs.copy(), a.var.copy()


def describe_combine_columns(
    file_paths: list[Path], labels: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Every .obs / .var column across the inputs, with a suggested policy.

    Feeds the column picker so the choice is made before anything is combined.
    Reads annotations only — an h5ad is opened backed, so this stays cheap on
    matrices too large to load twice.

    Returns ``{'obs': [...], 'var': [...]}`` where each entry carries ``name``,
    the ``datasets`` holding it, its ``kind``, the ``suggested`` policy and a
    human ``reason``; .var entries add ``identical`` (do the datasets already
    agree over the shared genes).
    """
    labels = [str(lbl) for lbl in labels]
    obs_frames: list[pd.DataFrame] = []
    var_frames: list[pd.DataFrame] = []
    for path in file_paths:
        try:
            o, v = _read_annotations(Path(path))
        except Exception as e:
            raise ValueError(f"Could not read {Path(path).name}: {e}")
        obs_frames.append(o)
        var_frames.append(v)

    shared_genes = None
    for v in var_frames:
        idx = set(v.index)
        shared_genes = idx if shared_genes is None else (shared_genes & idx)
    shared = sorted(shared_genes or set())

    obs_out: list[dict[str, Any]] = []
    for name in sorted({c for f in obs_frames for c in f.columns}):
        holders = [lbl for lbl, f in zip(labels, obs_frames) if name in f.columns]
        first = next(f[name] for f in obs_frames if name in f.columns)
        kind = _column_kind(first)
        n_unique = None
        if kind in ('integer', 'category', 'string'):
            n_unique = int(max(
                f[name].nunique(dropna=True) for f in obs_frames if name in f.columns
            ))
        suggested, reason = _suggest_obs_policy(holders, kind, n_unique)
        obs_out.append({
            'name': name, 'datasets': holders, 'kind': kind,
            'n_unique': n_unique, 'suggested': suggested, 'reason': reason,
        })

    var_out: list[dict[str, Any]] = []
    for name in sorted({c for f in var_frames for c in f.columns}):
        holders = [lbl for lbl, f in zip(labels, var_frames) if name in f.columns]
        held = [f for f in var_frames if name in f.columns]
        # Compare over the shared genes only — the rest are dropped anyway, so
        # disagreement outside the intersection is not a reason to split.
        rendered = [f[name].reindex(shared).astype(str).tolist() for f in held]
        identical = all(r == rendered[0] for r in rendered[1:]) if shared else True
        first = held[0][name]
        suggested, reason = _suggest_var_policy(holders, identical)
        var_out.append({
            'name': name, 'datasets': holders, 'kind': _column_kind(first),
            'identical': bool(identical), 'suggested': suggested, 'reason': reason,
        })

    return {'obs': obs_out, 'var': var_out}


def _resolve_policy(
    frames, labels, requested, axis: str, shared=None
) -> dict[str, str]:
    """Fill in defaults for every column the caller did not decide on."""
    allowed = _OBS_POLICIES if axis == 'obs' else _VAR_POLICIES
    requested = dict(requested or {})
    known = {c for f in frames for c in f.columns}
    for name, value in requested.items():
        if name not in known:
            raise ValueError(
                f"No input dataset has a .{axis} column named '{name}'; "
                f"available: {sorted(known)}"
            )
        if value not in allowed:
            raise ValueError(
                f"Unknown .{axis} policy '{value}' for '{name}'; "
                f"expected one of {list(allowed)}"
            )

    resolved: dict[str, str] = {}
    for name in sorted(known):
        if name in requested:
            resolved[name] = requested[name]
            continue
        holders = [lbl for lbl, f in zip(labels, frames) if name in f.columns]
        if axis == 'obs':
            first = next(f[name] for f in frames if name in f.columns)
            kind = _column_kind(first)
            n_unique = None
            if kind in ('integer', 'category', 'string'):
                n_unique = int(max(
                    f[name].nunique(dropna=True) for f in frames if name in f.columns
                ))
            resolved[name] = _suggest_obs_policy(holders, kind, n_unique)[0]
        else:
            held = [f for f in frames if name in f.columns]
            rendered = [f[name].reindex(shared).astype(str).tolist() for f in held]
            identical = all(r == rendered[0] for r in rendered[1:]) if shared else True
            resolved[name] = _suggest_var_policy(holders, identical)[0]
    return resolved


def combine_datasets(
    file_paths: list[Path],
    labels: list[str],
    gap_fraction: float = 0.05,
    *,
    obs_policy: dict[str, str] | None = None,
    var_policy: dict[str, str] | None = None,
) -> anndata.AnnData:
    """Combine multiple datasets into one adata, spatially aware when possible.

    Two modes, chosen by what the inputs carry:

    **spatial** — every input has spatial coordinates (``spatial`` or
    ``X_spatial``, or whatever ``uns['xcell_spatial_key']`` names). Sections
    are laid out left-to-right along x with a gap proportional to the mean
    section width; the result keeps only ``X_spatial`` in .obsm (per-file
    UMAPs/PCAs are dropped — re-run scanpy on the combined data).

    **concat** — anything else. Rows are concatenated with no geometry
    invented: .obsm arrays present in *every* input are kept (for dissociated
    data those embeddings are the only geometry there is), everything
    one-sided is dropped.

    Either way the result has a ``sample`` categorical .obs column naming each
    cell's source file, the intersection of input gene indices, collision-safe
    obs_names, and ``uns['xcell_combine'] = {'mode': ..., 'labels': [...]}``
    recording what was done. Per-file ``.raw`` / ``.varm`` / ``.obsp`` /
    ``.uns`` are dropped in both modes.

    Args:
        file_paths: List of >=2 absolute paths in any format
            ``load_dataset_file`` accepts (.h5ad, 10x .h5, Visium HD
            feature_slice.h5, 10x matrix directory, *_matrix.mtx trio).
        labels: Per-file labels for the new ``sample`` column. Must match
                ``file_paths`` length; collisions are de-duplicated by suffix.
        gap_fraction: Horizontal gap between adjacent sections, as a fraction
                      of mean section width (default 0.05 = 5%). Spatial mode
                      only.
        obs_policy: Per .obs column, one of ``merge`` (one column, every cell
            keeps its own value), ``separate`` (``<col>__<label>`` per dataset,
            empty for the others) or ``drop``. Columns left out follow
            :func:`describe_combine_columns`' suggestion — measurements merge,
            labels stay apart, because pooling two independent clusterings into
            one column invents a comparison nobody made. ``sample`` is always
            written and cannot be overridden.
        var_policy: Per .var column, one of ``first`` (one copy), ``separate``
            or ``drop``. Genes are shared, so the default is ``first`` when
            every dataset already agrees and ``separate`` when they do not.

    Raises:
        ValueError: if fewer than 2 files, no shared genes, or an input
            that fails to load (the message names the file).
    """
    if len(file_paths) < 2:
        raise ValueError("At least 2 files required for combination")
    if len(labels) != len(file_paths):
        raise ValueError("labels must match file_paths length")

    # De-duplicate labels (filename collisions, etc.)
    seen: dict[str, int] = {}
    unique_labels: list[str] = []
    for lbl in labels:
        base = (lbl or "section").strip() or "section"
        if base in seen:
            seen[base] += 1
            unique_labels.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 1
            unique_labels.append(base)

    adatas: list[anndata.AnnData] = []
    spatial_keys: list[str | None] = []
    for p in file_paths:
        try:
            a, _kind = load_dataset_file(p)
        except Exception as e:
            raise ValueError(f"Could not load {Path(p).name}: {e}")
        adatas.append(a)
        spatial_keys.append(_combine_spatial_key(a))

    mode = 'spatial' if all(k is not None for k in spatial_keys) else 'concat'

    # Resolve the column policy before anything is rewritten, so a bad request
    # fails before any work is done.
    obs_frames = [a.obs for a in adatas]
    var_frames = [a.var for a in adatas]
    shared_genes: set[str] | None = None
    for v in var_frames:
        idx = set(v.index)
        shared_genes = idx if shared_genes is None else (shared_genes & idx)
    shared = sorted(shared_genes or set())
    # `sample` is written by the concat itself; a policy for it is meaningless.
    obs_requested = {k: v for k, v in (obs_policy or {}).items() if k != 'sample'}
    obs_resolved = _resolve_policy(obs_frames, unique_labels, obs_requested, 'obs')
    var_resolved = _resolve_policy(
        var_frames, unique_labels, var_policy, 'var', shared=shared
    )

    # Allocate final .obs names up front: merged columns keep their own name,
    # split ones become `<col>__<label>`, and nothing may land on a name that
    # is already spoken for.
    obs_alloc = _NameAllocator({'sample'})
    merged_obs: dict[str, str] = {}
    for name in sorted(obs_resolved):
        if obs_resolved[name] == 'merge':
            merged_obs[name] = obs_alloc.take(name)
    split_obs: dict[tuple[str, str], str] = {}
    for name in sorted(obs_resolved):
        if obs_resolved[name] != 'separate':
            continue
        for lbl, f in zip(unique_labels, obs_frames):
            if name in f.columns:
                split_obs[(name, lbl)] = obs_alloc.take(f'{name}__{lbl}')

    # Categorical columns lose their dtype when reindexed in with NaN, so the
    # categories are carried across the concat and restored afterwards.
    obs_categories: dict[str, list] = {}
    for name, policy in obs_resolved.items():
        for lbl, f in zip(unique_labels, obs_frames):
            if name not in f.columns or policy == 'drop':
                continue
            col = f[name]
            if not isinstance(col.dtype, pd.CategoricalDtype):
                continue
            final = merged_obs[name] if policy == 'merge' else split_obs[(name, lbl)]
            prev = obs_categories.setdefault(final, [])
            for c in col.cat.categories:
                if c not in prev:
                    prev.append(c)

    def _apply_obs_policy(b: anndata.AnnData, lbl: str) -> None:
        new_obs = pd.DataFrame(index=b.obs.index)
        for name, policy in obs_resolved.items():
            if policy == 'drop' or name not in b.obs.columns:
                continue
            final = merged_obs[name] if policy == 'merge' else split_obs[(name, lbl)]
            new_obs[final] = b.obs[name].values
        b.obs = new_obs

    cleaned: list[anndata.AnnData] = []
    if mode == 'spatial':
        coords = [
            np.asarray(a.obsm[k][:, :2], dtype=np.float64).copy()
            for a, k in zip(adatas, spatial_keys)
        ]
        widths = [float(sp[:, 0].max() - sp[:, 0].min()) for sp in coords]
        gap = float(np.mean(widths)) * gap_fraction

        # Shift each section's spatial x so they lay out left-to-right.
        current_offset = 0.0
        for a, sp, w, lbl in zip(adatas, coords, widths, unique_labels):
            sp[:, 0] = sp[:, 0] - sp[:, 0].min() + current_offset
            b = a.copy()
            # Reset obsm/obsp/varm/uns; drop .raw. The concat below operates
            # on .X via the intersection of var indices — keeping per-file
            # .raw with different gene spaces would conflict on concat.
            b.obsm = {'X_spatial': sp}
            b.obsp = {}
            b.varm = {}
            b.uns = {}
            b.raw = None
            _apply_obs_policy(b, lbl)
            cleaned.append(b)
            current_offset += w + gap
    else:
        for a, lbl in zip(adatas, unique_labels):
            b = a.copy()
            # .obsm is left alone: anndata.concat keeps the keys every input
            # shares and drops the rest, which is exactly the honest outcome —
            # a one-sided embedding cannot describe the combined cells.
            b.obsp = {}
            b.varm = {}
            b.uns = {}
            b.raw = None
            _apply_obs_policy(b, lbl)
            cleaned.append(b)

    # join='inner' below drops any .obs column not present in every input, and
    # a split column is by construction present in exactly one — so line the
    # frames up on the union first.
    all_obs_cols = sorted({c for b in cleaned for c in b.obs.columns})
    for b in cleaned:
        b.obs = b.obs.reindex(columns=all_obs_cols)

    combined = anndata.concat(
        cleaned,
        axis=0,
        join='inner',          # intersection of var.index
        label='sample',
        keys=unique_labels,
        index_unique='-',      # collision-safe obs_names
        merge=None,            # .var is rebuilt below, under var_policy
    )
    if combined.n_vars == 0:
        raise ValueError(
            "No shared genes across the provided files — cannot combine. "
            "Check that each h5ad uses the same gene identifiers (consider "
            "Gene IDs swap in xcell before exporting)."
        )
    for col, cats in obs_categories.items():
        if col in combined.obs.columns:
            combined.obs[col] = pd.Categorical(combined.obs[col], categories=cats)

    # Every combined gene is present in every input (join='inner'), so these
    # reindexes are exact and a boolean flag stays boolean.
    var_alloc = _NameAllocator(set(combined.var.columns))
    for name in sorted(var_resolved):
        if var_resolved[name] != 'first':
            continue
        src = next(f for f in var_frames if name in f.columns)
        combined.var[var_alloc.take(name)] = src[name].reindex(combined.var_names).values
    for name in sorted(var_resolved):
        if var_resolved[name] != 'separate':
            continue
        for lbl, f in zip(unique_labels, var_frames):
            if name in f.columns:
                combined.var[var_alloc.take(f'{name}__{lbl}')] = (
                    f[name].reindex(combined.var_names).values
                )

    # Ensure the `sample` column is a clean categorical for downstream UI.
    combined.obs['sample'] = pd.Categorical(combined.obs['sample'], categories=unique_labels, ordered=False)
    combined.uns['xcell_combine'] = {
        'mode': mode,
        'labels': list(unique_labels),
        'obs_policy': dict(obs_resolved),
        'var_policy': dict(var_resolved),
    }
    return combined


def _drop_cross_section_edges(adata, sections):
    """Zero out edges between cells in different sections in the spatial graph.

    Makes ``obsp['spatial_connectivities']`` / ``['spatial_distances']``
    block-diagonal so no neighborhood spans the gap between sections.
    """
    from scipy.sparse import csr_matrix

    sections = np.asarray(sections)
    for key in ('spatial_connectivities', 'spatial_distances'):
        if key not in adata.obsp:
            continue
        m = adata.obsp[key].tocoo()
        keep = sections[m.row] == sections[m.col]
        adata.obsp[key] = csr_matrix(
            (m.data[keep], (m.row[keep], m.col[keep])), shape=m.shape)


def _interp_smooth_subset(coords, summary, grid_res, smooth_sigma):
    """Grid-interpolate + Gaussian-smooth ``summary`` over ``coords`` and sample
    back at each point. Returns (cell_vals, grid_vmax).

    Degenerate inputs (fewer than 4 points, or collinear) can't be cubic-gridded,
    so the raw ``summary`` is returned unchanged — this is the per-section
    small-section guard.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    coords = np.asarray(coords)
    summary = np.asarray(summary, dtype=float)
    n = summary.shape[0]
    if n == 0:
        return summary.copy(), 0.0
    if n < 4 or np.ptp(coords[:, 0]) == 0 or np.ptp(coords[:, 1]) == 0:
        return summary.copy(), float(summary.max())

    x, y = coords[:, 0], coords[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    try:
        Zi = griddata((x, y), summary, (Xi, Yi), method='cubic', fill_value=0.0)
    except Exception:
        Zi = griddata((x, y), summary, (Xi, Yi), method='linear', fill_value=0.0)
    Zi_s = gaussian_filter(Zi, sigma=smooth_sigma, mode='nearest')
    vmax = float(np.nanmax(Zi_s))

    pts = np.vstack((Xi.ravel(), Yi.ravel())).T
    cell_vals = griddata(pts, Zi_s.ravel(), (x, y), method='nearest')
    return np.asarray(cell_vals, dtype=float), vmax


def _contour_score_field(coords, gene_expr, log_transform, clip_percentiles,
                         grid_res, smooth_sigma, sections=None):
    """Continuous per-spot contour score (the value before banding).

    Shared by single contourize and multi-contour module scoring.

    Gene normalization and the averaged ``summary`` are computed globally (so a
    "high" expresser means the same thing everywhere). The spatial interpolation
    + smoothing run per ``sections`` group when given, so expression never bleeds
    across the gap between sections; ``vmax`` is the max grid value across groups
    (global) so band thresholds stay comparable.

    Args:
        coords: (n, 2) spatial coordinates.
        gene_expr: dict mapping gene -> (n,) expression vector.
        log_transform: apply log1p before normalizing each gene.
        clip_percentiles: (lo, hi) percentile clip range per gene.
        grid_res: interpolation grid size per axis.
        smooth_sigma: Gaussian smoothing strength (grid-pixel units).
        sections: optional (n,) array of section labels; None → one global grid.

    Returns:
        (score, vmax): score is an (n,) float array in [0, vmax].
    """
    coords = np.asarray(coords)

    normed = []
    for vals in gene_expr.values():
        v = np.asarray(vals, dtype=float).copy()
        if log_transform:
            v = np.log1p(v)
        lo, hi = np.percentile(v, clip_percentiles)
        clipped = np.clip(v, lo, hi)
        normed.append((clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped))
    summary = np.mean(np.column_stack(normed), axis=1)

    if sections is None:
        return _interp_smooth_subset(coords, summary, grid_res, smooth_sigma)

    sections = np.asarray(sections)
    n = summary.shape[0]
    cell_vals = np.zeros(n, dtype=float)
    vmax = 0.0
    for s in np.unique(sections):
        idx = np.where(sections == s)[0]
        sub_vals, sub_vmax = _interp_smooth_subset(
            coords[idx], summary[idx], grid_res, smooth_sigma)
        cell_vals[idx] = sub_vals
        vmax = max(vmax, sub_vmax)
    return cell_vals, vmax


_CONN_SUFFIX = '_connectivities'


def _graph_suffix(graph_key: str | None) -> str:
    """Short name for a connectivity graph, used to name what it produces.

    'connectivities' is the default graph, so it contributes no suffix and
    UMAP/Leiden keep writing X_umap / leiden — nothing changes for a user who
    never touches the picker. Every other graph is named after its prefix so
    results from different graphs sit side by side instead of overwriting.
    """
    if not graph_key or graph_key == 'connectivities':
        return ''
    if graph_key.endswith(_CONN_SUFFIX):
        return graph_key[: -len(_CONN_SUFFIX)]
    return graph_key


def _default_output_name(base: str, graph_key: str | None) -> str:
    """'X_umap' + spatial_connectivities -> 'X_umap_spatial'."""
    suffix = _graph_suffix(graph_key)
    return f'{base}_{suffix}' if suffix else base


def _subset_output_name(base: str, subset_name: str, graph_key: str | None) -> str:
    """Where a run on a named cell subset writes by default.

    Always carries the subset's name, so it can never land on the dataset's
    own key whatever graph was chosen — the browser sends the default graph
    explicitly as 'connectivities', which contributes no suffix of its own.
    The subset's own graph adds nothing more ('leiden_chondro'); any other
    named graph is appended ('leiden_chondro_spatial').
    """
    suffix = _graph_suffix(graph_key)
    if not suffix or suffix == subset_name:
        return f'{base}_{subset_name}'
    return f'{base}_{subset_name}_{suffix}'


# Where the synthesized neighbors entry lives while sc.tl.umap reads it. Also
# emitted by codegen, so the exported notebook uses the same name.
_GRAPH_META_KEY = '_xcell_graph'

# Named cell subsets: the persisted form of the browser's active cell mask.
# Membership is a boolean .obs column (it survives a reload, exports with the
# h5ad, and reads naturally in scanpy: ``adata[adata.obs['subset_x']]``); the
# .uns registry carries what the column cannot. Everything the clustering chain
# computes on a subset is suffixed with its name — the shape
# prepare_cluster_cells_by_gene_set already uses — so the dataset's own X_pca /
# connectivities / X_umap / leiden are never overwritten by a sub-clustering.
CELL_SUBSETS_UNS = 'xcell_cell_subsets'
SUBSET_OBS_PREFIX = 'subset_'

# Drawn shapes and lines. They belong to an embedding (``embeddingName``) and
# used to live only in the browser, reaching this key at export time; now the
# browser's every sync writes it and a load reads it back, the way territories
# already work. A JSON string, since ragged point lists do not survive h5ad as
# nested uns.
LINES_UNS = 'xcell_lines_json'
_LINE_DEFAULTS: dict[str, Any] = {
    'dimX': 0, 'dimY': 1, 'smoothedPoints': None, 'drawType': 'pencil',
    'closed': False, 'visible': True, 'strokeColor': '#4ecdc4',
    'strokeWidth': 2, 'fillColor': None,
}


def _umap_neighbors_meta(adata, graph_key: str) -> dict[str, Any]:
    """A ``uns['neighbors']``-shaped entry that ``sc.tl.umap`` will accept.

    scanpy reads three fields without guarding them, and the graphs xcell
    produces are each missing one:

    - squidpy's ``uns['spatial_neighbors']`` has no ``params['method']``, and
      ``tl.umap`` evaluates it to decide whether to warn -> ``KeyError``.
    - ``combine_neighbor_graphs`` writes only the obsp matrix, no ``uns`` entry
      at all, and ``NeighborsView`` requires ``distances_key`` to be present.
      The matrix it names is never read — UMAP uses connectivities only — so a
      key pointing at nothing is fine.

    Start from any real entry that already points at this graph, so squidpy's
    recorded ``n_neighbors`` reaches the exported notebook instead of an
    invented one, then fill the gaps.
    """
    source: Mapping = {}
    for entry in adata.uns.values():
        if isinstance(entry, Mapping) and entry.get('connectivities_key') == graph_key:
            source = entry
            break

    params = dict(source.get('params', {}))
    params['method'] = 'umap'

    # Without a representation scanpy falls through to _get_pca_or_small_x,
    # which *computes and stores* a PCA when X_pca is missing and n_vars > 50.
    # That would leave an embedding in the object that no recorded step made,
    # so the exported notebook would not reproduce it. Reading .X costs nothing
    # here: for method='umap' the matrix is not what drives the layout.
    if 'use_rep' not in params and 'X_pca' not in adata.obsm:
        params['use_rep'] = 'X'

    suffix = _graph_suffix(graph_key)
    return {
        'connectivities_key': graph_key,
        'distances_key': (source.get('distances_key')
                          or (f'{suffix}_distances' if suffix else 'distances')),
        'params': params,
    }


def _umap_call(adata, *, min_dist, spread, n_components, meta, key_added):
    """``sc.tl.umap``, with a synthesized neighbors entry when one is needed."""
    kwargs = {'min_dist': min_dist, 'spread': spread, 'n_components': n_components}
    # key_added='X_umap' would move the params to uns['X_umap']; omitting it
    # keeps scanpy's own uns['umap'], exactly as before this parameter existed.
    if key_added != 'X_umap':
        kwargs['key_added'] = key_added

    if meta is None:
        sc.tl.umap(adata, **kwargs)
        return

    adata.uns[_GRAPH_META_KEY] = meta
    try:
        sc.tl.umap(adata, neighbors_key=_GRAPH_META_KEY, **kwargs)
    finally:
        # A failed run must not leave a fabricated neighbors entry behind for
        # the next tool that reads uns.
        adata.uns.pop(_GRAPH_META_KEY, None)


class DataAdaptor:
    """Wraps an AnnData object and provides accessor methods.

    This class follows the adaptor pattern to:
    - Provide clean accessor methods for embeddings, metadata, etc.
    - Enable future scanpy integration for analysis features
    - Abstract away AnnData implementation details from API routes

    Attributes:
        adata: The underlying AnnData object
        filepath: Path to the loaded h5ad file
    """

    def __init__(self, filepath: str | Path, *, adata: anndata.AnnData | None = None):
        """Load an h5ad, 10x h5, or 10x mtx directory and initialize the adaptor.

        Args:
            filepath: Path to the .h5ad/.h5 file, 10x CellRanger matrix directory,
                      or a *_matrix.mtx(.gz) file from a prefixed file trio. Used
                      for display only when ``adata`` is provided.
            adata: Optional pre-loaded AnnData. When supplied, the constructor
                   skips file I/O and uses this object directly. Used by the
                   combine-spatial route which builds the AnnData in memory.
        """
        self.filepath = Path(filepath)
        # How the matrix got here, so the exported notebook opens it the same
        # way. 'memory' means there is no file to re-read (combine_spatial, tests).
        source_kind = 'memory'
        if adata is not None:
            self.adata = adata
        else:
            self.adata, source_kind = load_dataset_file(self.filepath)
        self._normalized_adata: anndata.AnnData | None = None
        self._drawn_lines: list[dict[str, Any]] = self._restore_lines()
        self._action_history: list[dict[str, Any]] = []  # Track scanpy operations
        self._embedding_undo_stacks: dict[str, list[np.ndarray]] = {}  # Undo stacks for quilt transforms
        # Gene mask state — None means no mask is active.
        # See get_gene_mask / set_gene_mask / clear_gene_mask.
        self._gene_mask_config: dict[str, Any] | None = None
        self._visible_gene_mask: np.ndarray | None = None  # bool array, shape (n_genes,)
        # Multi-contour: cache of per-module scores between prepare and finalize,
        # keyed by an opaque token. See prepare_multicontour / finalize_multicontour.
        self._multicontour_cache: dict[str, dict[str, Any]] = {}
        # UCell rank matrices, keyed by (resolved_layer, max_rank). Transient:
        # validated against id(self.adata); never written into adata.
        self._ucell_rank_cache: dict[tuple[str, int], Any] = {}
        self._ucell_rank_cache_adata_id: int | None = None

        # Exportable provenance. Restored from the h5ad when one is present, so
        # re-opening an exported dataset continues its history rather than
        # starting a second one.
        self.analysis_record = self._restore_analysis_record()
        self._record_source(self.filepath, source_kind)

        # Defensively preserve raw counts: if .X looks like integer counts and
        # a "counts" layer isn't already present, snapshot .X into
        # adata.layers["counts"] so downstream tooling that wants raw counts
        # (e.g. integer-required methods) can still reach them after the user
        # runs normalize_total / log1p. Default expression paths still read
        # from .X — this is purely additive.
        self._maybe_snapshot_counts_layer()

    def _maybe_snapshot_counts_layer(self) -> None:
        """If ``adata.X`` looks like integer counts, copy it to ``layers['counts']``.

        Safe to call many times — no-op if ``layers['counts']`` already exists.
        Uses a random sample to avoid scanning huge matrices.
        """
        if self.adata is None:
            return
        if 'counts' in self.adata.layers:
            return

        X = self.adata.X
        from scipy.sparse import issparse

        try:
            # Sample a limited number of values for the integer check. For sparse,
            # use the stored nonzero data directly (which represents observed
            # expression); for dense, sample a flat slice.
            MAX_SAMPLE = 50_000
            if issparse(X):
                data = X.data
                if data.size == 0:
                    return  # All-zero matrix; nothing meaningful to snapshot
                if data.size > MAX_SAMPLE:
                    rng = np.random.default_rng(0)
                    sample = rng.choice(data, size=MAX_SAMPLE, replace=False)
                else:
                    sample = data
            else:
                arr = np.asarray(X)
                flat = arr.ravel()
                if flat.size == 0:
                    return
                if flat.size > MAX_SAMPLE:
                    rng = np.random.default_rng(0)
                    idx = rng.integers(0, flat.size, size=MAX_SAMPLE)
                    sample = flat[idx]
                else:
                    sample = flat

            sample = np.asarray(sample)
            if not np.issubdtype(sample.dtype, np.number):
                return

            # Ignore NaNs if any (rare for expression matrices but be safe).
            sample = sample[~np.isnan(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
            if sample.size == 0:
                return

            # Counts must be non-negative and integer-valued.
            if np.any(sample < 0):
                return
            is_integer_like = np.all(np.equal(np.mod(sample, 1), 0))
            if not is_integer_like:
                return

            # Copy .X into the 'counts' layer. Use .copy() so later in-place
            # edits to .X (e.g. normalize_total) don't touch the stored counts.
            self.adata.layers['counts'] = X.copy()
            self.adata.uns.setdefault('xcell', {})['counts_inferred'] = True
            print(
                "[xcell] .X looked like integer counts — snapshot to adata.layers['counts']"
            )
        except Exception as e:
            # Detection is best-effort; never block loading on it.
            print(f"[xcell] counts snapshot skipped: {e}")

    @staticmethod
    def _locate_10x_companions(parent: Path, prefix: str) -> tuple[Path, Path] | None:
        """Find ``<prefix>barcodes.tsv[.gz]`` and ``<prefix>features|genes.tsv[.gz]``.

        Compression is decided per file, not per trio, because GEO uploads
        gzip whatever the submitter had — a gzipped matrix next to plain
        companions is common.
        """
        barcodes = None
        for ext in ('.tsv.gz', '.tsv'):
            candidate = parent / f'{prefix}barcodes{ext}'
            if candidate.exists():
                barcodes = candidate
                break

        features = None
        for feat in ('features', 'genes'):
            for ext in ('.tsv.gz', '.tsv'):
                candidate = parent / f'{prefix}{feat}{ext}'
                if candidate.exists():
                    features = candidate
                    break
            if features:
                break

        if barcodes and features:
            return (barcodes, features)
        return None

    @staticmethod
    def _find_10x_dir_files(directory: Path) -> tuple[Path, Path, Path] | None:
        """A CellRanger matrix folder's (matrix, barcodes, features), else None."""
        for name in ('matrix.mtx.gz', 'matrix.mtx'):
            matrix = directory / name
            if matrix.exists():
                break
        else:
            return None
        companions = DataAdaptor._locate_10x_companions(directory, '')
        if companions is None:
            return None
        return (matrix, *companions)

    @staticmethod
    def _find_10x_trio_files(filepath: Path) -> tuple[Path, Path, Path] | None:
        """Check if filepath is a prefixed 10x matrix file with companion files.

        Detects GEO-style file trios like prefix_barcodes.tsv.gz,
        prefix_features.tsv.gz, prefix_matrix.mtx.gz.

        Returns:
            (matrix, barcodes, features) paths if valid trio, else None.
        """
        import re
        m = re.match(r'^(.+)_matrix\.mtx(\.gz)?$', filepath.name)
        if not m:
            return None
        companions = DataAdaptor._locate_10x_companions(filepath.parent, f'{m.group(1)}_')
        if companions is None:
            return None
        return (filepath, *companions)

    @property
    def n_cells(self) -> int:
        """Number of cells (observations) in the dataset."""
        return self.adata.n_obs

    @property
    def normalized_adata(self) -> anndata.AnnData:
        """Get normalized and log1p-transformed version of the data.

        Lazily computes and caches a copy of the AnnData with:
        - sc.pp.normalize_total (count depth scaling)
        - sc.pp.log1p transformation

        Returns:
            AnnData object with normalized expression values
        """
        if self._normalized_adata is None:
            # Create a copy to avoid modifying original data
            self._normalized_adata = self.adata.copy()
            # Apply count depth normalization (scales each cell to same total counts)
            sc.pp.normalize_total(self._normalized_adata)
            # Apply log1p transformation: log(x + 1)
            sc.pp.log1p(self._normalized_adata)
        return self._normalized_adata

    @property
    def n_genes(self) -> int:
        """Number of genes (variables) in the dataset."""
        return self.adata.n_vars

    def get_schema(self) -> dict[str, Any]:
        """Get dataset schema including available embeddings and metadata.

        Returns:
            Dictionary containing:
            - n_cells: Number of cells
            - n_genes: Number of genes
            - embeddings: List of available embedding names from .obsm
            - obs_columns: List of cell metadata column names from .obs
            - obs_dtypes: Dictionary mapping column names to their dtypes
        """
        # Get embedding names (keys in obsm that are 2D array-likes) and how many
        # columns each has (so the UI can offer a column picker for >2-dim ones).
        embeddings = []
        embedding_dims: dict[str, int] = {}
        for key in self.adata.obsm.keys():
            arr = self.adata.obsm[key]
            if hasattr(arr, 'shape') and len(arr.shape) == 2 and arr.shape[1] >= 2:
                embeddings.append(key)
                embedding_dims[key] = int(arr.shape[1])

        # Get obs column info
        obs_columns = list(self.adata.obs.columns)
        obs_dtypes = {}
        for col in obs_columns:
            dtype = self.adata.obs[col].dtype
            if pd.api.types.is_categorical_dtype(dtype):
                obs_dtypes[col] = "category"
            elif pd.api.types.is_numeric_dtype(dtype):
                obs_dtypes[col] = "numeric"
            else:
                obs_dtypes[col] = "string"

        # Score matrices → their column names, so the UI can offer columns for
        # color-by-score and embed-from-two-scores. Sources are the xcell
        # registry in .uns and any .obsm stored as a labelled DataFrame (how
        # PySingleCellNet and friends write theirs).
        score_matrices: dict[str, list[str]] = {}
        for key in self.adata.obsm.keys():
            cols = self._score_matrix_columns(str(key))
            if cols:
                score_matrices[str(key)] = cols

        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_genes_visible": (
                int(self._visible_gene_mask.sum())
                if self._visible_gene_mask is not None
                else self.n_genes
            ),
            "embeddings": embeddings,
            "embedding_dims": embedding_dims,
            "obs_columns": obs_columns,
            "obs_dtypes": obs_dtypes,
            "score_matrices": score_matrices,
            "filename": self.filepath.name,
        }

    def _clamp_dims(self, name: str, dim_x: int, dim_y: int) -> tuple[int, int]:
        """Validate/clamp a requested (dim_x, dim_y) column pair for an .obsm matrix.

        Out-of-range indices fall back to 0 / 1 so a stale request (e.g. after the
        data changed) degrades to the first two columns instead of erroring.
        """
        arr = self.adata.obsm[name]
        ncols = arr.shape[1] if getattr(arr, 'ndim', 1) == 2 else 1
        dx = int(dim_x) if 0 <= int(dim_x) < ncols else 0
        dy = int(dim_y) if 0 <= int(dim_y) < ncols else min(1, ncols - 1)
        return dx, dy

    def _clamp_one_dim(self, name: str, dim: int) -> int:
        """Clamp a single .obsm column index into [0, ncols-1] (fallback 0)."""
        arr = self.adata.obsm[name]
        ncols = arr.shape[1] if getattr(arr, 'ndim', 1) == 2 else 1
        d = int(dim)
        return d if 0 <= d < ncols else 0

    def _obsm_array(self, name: str) -> np.ndarray:
        """An .obsm entry as a positionally-indexable float array.

        DataFrames land in .obsm whenever a tool that labels its columns wrote
        them (PySingleCellNet's SCN_score, for one). Everything downstream of
        here indexes by column number, so unwrap once at the boundary.
        """
        arr = self.adata.obsm[name]
        if isinstance(arr, pd.DataFrame):
            return arr.to_numpy(dtype=np.float64, copy=False)
        return np.asarray(arr)

    def _view_coords(self, name: str, dim_x: int = 0, dim_y: int = 1) -> np.ndarray:
        """The two chosen columns of an .obsm matrix as an (n_cells, 2) float array."""
        dx, dy = self._clamp_dims(name, dim_x, dim_y)
        return np.asarray(self._obsm_array(name)[:, [dx, dy]], dtype=np.float64)

    def _line_view_coords(self, line: dict[str, Any]) -> np.ndarray:
        """Embedding coordinates for a drawn line, on the columns it was drawn on.

        A line records which two .obsm columns (``dimX``/``dimY``, default 0/1) it
        was drawn against, so projection/association use the same axes the user saw.
        """
        name = line.get('embeddingName', '')
        return self._view_coords(name, int(line.get('dimX', 0)), int(line.get('dimY', 1)))

    def get_embedding(self, name: str, dim_x: int = 0, dim_y: int = 1,
                      dim_z: int | None = None) -> dict[str, Any]:
        """Get embedding coordinates by name, viewing two chosen columns.

        Args:
            name: Name of the embedding (e.g., 'X_umap', 'X_pca')
            dim_x, dim_y: Which .obsm columns to use as x / y (default first two).
                For a >2-column matrix (PCA, gene-set scores) this lets the caller
                view any pair of columns. Out-of-range values fall back to 0 / 1.
            dim_z: Optional third .obsm column to view as z. When provided, the
                result additionally has "z" (list[float], one per cell) and
                "dim_z" (the clamped column index). When None, the result is
                unchanged from the 2-D case.

        Returns:
            Dictionary with: name, coordinates (list of [x, y]), dim_x, dim_y,
            and (when dim_z is not None) z, dim_z.

        Raises:
            KeyError: If embedding name not found in .obsm
        """
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {list(self.adata.obsm.keys())}")

        arr = self._obsm_array(name)
        dx, dy = self._clamp_dims(name, dim_x, dim_y)
        coords_2d = arr[:, [dx, dy]]

        result = {
            "name": name,
            "coordinates": _nullable(coords_2d),
            "dim_x": dx,
            "dim_y": dy,
        }
        if dim_z is not None:
            dz = self._clamp_one_dim(name, dim_z)
            result["z"] = _nullable(np.asarray(arr[:, dz], dtype=float))
            result["dim_z"] = dz
        return result

    def create_obs_embedding(
        self, col_x: str, col_y: str, log_axes: str = 'none',
        name: str | None = None,
    ) -> dict[str, Any]:
        """Build a 2-D embedding (obsm['X_...']) from two numeric .obs columns.

        log_axes in {'none','x','y','both'} applies log1p to the chosen axis
        (requires non-negative values). Raises ValueError on bad input or a
        duplicate embedding name.
        """
        obs = self.adata.obs
        for c in (col_x, col_y):
            if c not in obs.columns:
                raise ValueError(f"obs column '{c}' not found")
            if not pd.api.types.is_numeric_dtype(obs[c].dtype):
                raise ValueError(f"obs column '{c}' is not numeric")
        if log_axes not in ('none', 'x', 'y', 'both'):
            raise ValueError(f"log_axes must be none/x/y/both, got {log_axes!r}")
        x = obs[col_x].to_numpy(dtype=float)
        y = obs[col_y].to_numpy(dtype=float)
        if log_axes in ('x', 'both'):
            if np.nanmin(x) < 0:
                raise ValueError(f"log requires non-negative values; '{col_x}' has negatives")
            x = np.log1p(x)
        if log_axes in ('y', 'both'):
            if np.nanmin(y) < 0:
                raise ValueError(f"log requires non-negative values; '{col_y}' has negatives")
            y = np.log1p(y)
        if name:
            key = name if name.startswith('X_') else f'X_{name}'
        else:
            key = f'X_{col_x}_vs_{col_y}'
        if key in self.adata.obsm:
            raise ValueError(f"embedding '{key}' already exists")
        self.adata.obsm[key] = np.column_stack([x, y]).astype(float)
        result = {"embedding_name": key, "n_cells": self.n_cells}
        self._log_action('create_obs_embedding', {
            'col_x': col_x, 'col_y': col_y, 'log_axes': log_axes, 'name': name,
        }, result)
        return result

    def score_gene_sets_matrix(
        self,
        sets: list[dict[str, Any]],
        per_gene_norm: str = 'zscore_mad',
        per_gene_clip: float = 0.0,
        aggregation: str = 'mean',
        obsm_name: str = 'geneset_scores',
        layer: str | None = None,
        transform: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Score every gene set in a folder into one ``.obsm`` matrix.

        Each set is scored with the same mean pipeline used for coloring
        (``_aggregate_gene_set_scores``: per-gene normalize across cells → aggregate
        across genes per cell), honoring the active gene mask exactly like
        ``get_multi_gene_expression`` (via ``_filter_to_visible``). Kept sets are
        stacked column-wise into ``adata.obsm[obsm_name]`` (n_cells × n_kept_sets),
        and the column→set names plus scoring params are recorded in
        ``adata.uns['xcell_score_matrices'][obsm_name]`` (``.obsm`` arrays are
        unnamed, so the registry is how columns resolve back to set names).

        Args:
            sets: list of ``{"name": str, "genes": [str, ...]}``.
            per_gene_norm/per_gene_clip/aggregation: mean-pipeline params.
            obsm_name: destination ``.obsm`` slot (auto-named upstream, editable).
            layer/transform: source resolution, same rules as coloring.
            overwrite: replace an existing slot of the same name.

        Returns dict with obsm_name, columns (kept set names in order), n_cells,
        n_sets, skipped (list of {name, reason}), and per_column stats.

        Raises ValueError on a bad/duplicate name or if no set could be scored.
        """
        if not obsm_name or not isinstance(obsm_name, str):
            raise ValueError("obsm_name must be a non-empty string")
        if obsm_name in self.adata.obsm and not overwrite:
            raise ValueError(f"obsm slot '{obsm_name}' already exists (pass overwrite=True to replace)")

        # Source matrix — identical resolution to get_multi_gene_expression.
        if layer is not None and layer != 'X':
            source_matrix = self._resolve_source_matrix(layer)
        elif transform == 'log1p':
            source_matrix = self.normalized_adata.X
        else:
            source_matrix = self.adata.X

        columns: list[str] = []
        cols_data: list[np.ndarray] = []
        skipped: list[dict[str, Any]] = []
        per_column: dict[str, Any] = {}
        for s in sets:
            name = str(s.get('name'))
            genes = list(s.get('genes') or [])
            valid = [g for g in genes if g in self.adata.var.index]
            valid, n_masked = self._filter_to_visible(valid)
            if len(valid) == 0:
                skipped.append({"name": name, "reason": "no usable genes (missing or masked)"})
                continue
            scores = self._aggregate_gene_set_scores(
                genes=valid,
                source_matrix=source_matrix,
                per_gene_norm=per_gene_norm,
                per_gene_clip=per_gene_clip,
                aggregation=aggregation,
            )
            columns.append(name)
            cols_data.append(np.asarray(scores, dtype=np.float64))
            valid_scores = scores[~np.isnan(scores)]
            per_column[name] = {
                "min": float(np.min(valid_scores)) if valid_scores.size else 0.0,
                "max": float(np.max(valid_scores)) if valid_scores.size else 0.0,
                "n_genes_used": len(valid),
                "n_masked_excluded": int(n_masked),
            }

        if not columns:
            raise ValueError("No sets could be scored (every set had no usable genes)")

        self.adata.obsm[obsm_name] = np.column_stack(cols_data).astype(np.float64)

        reg = self.adata.uns.get('xcell_score_matrices')
        reg = dict(reg) if isinstance(reg, dict) else {}
        reg[obsm_name] = {
            "columns": list(columns),
            "per_gene_norm": per_gene_norm,
            "per_gene_clip": per_gene_clip,
            "aggregation": aggregation,
            "layer": layer,
        }
        self.adata.uns['xcell_score_matrices'] = reg

        result = {
            "obsm_name": obsm_name,
            "columns": columns,
            "n_cells": self.n_cells,
            "n_sets": len(columns),
            "skipped": skipped,
            "per_column": per_column,
        }
        self._log_action('score_gene_sets_matrix', {
            'sets': [{'name': s.get('name'), 'genes': list(s.get('genes') or [])}
                     for s in sets],
            'per_gene_norm': per_gene_norm,
            'per_gene_clip': per_gene_clip,
            'aggregation': aggregation,
            'obsm_name': obsm_name,
            'layer': layer,
            'transform': transform,
        }, {'obsm_name': obsm_name, 'columns': columns, 'n_sets': len(columns),
            'skipped': skipped})
        return result

    def _score_matrix_columns(self, obsm_name: str) -> list[str] | None:
        """Registered column (set) names for a score matrix, or None.

        Falls back to a DataFrame's own column labels when the matrix carries
        them. Matrices written by other tools arrive that way — PySingleCellNet
        stores ``obsm['SCN_score']`` as a DataFrame of per-class scores — and
        those names are just as authoritative as the xcell registry's.
        """
        reg = self.adata.uns.get('xcell_score_matrices')
        if isinstance(reg, dict):
            meta = reg.get(obsm_name)
            if isinstance(meta, dict) and 'columns' in meta:
                return [str(c) for c in meta['columns']]
        arr = self.adata.obsm.get(obsm_name)
        if isinstance(arr, pd.DataFrame):
            return [str(c) for c in arr.columns]
        return None

    def _obsm_values(self, obsm_name: str, idx: int) -> np.ndarray:
        """One column of an .obsm matrix as a float array, DataFrame or not."""
        arr = self.adata.obsm[obsm_name]
        col = arr.iloc[:, idx] if isinstance(arr, pd.DataFrame) else arr[:, idx]
        return np.asarray(col, dtype=np.float64)

    def _resolve_obsm_column_index(self, obsm_name: str, column: str) -> int:
        """Column index for a named (registry) or integer-string column."""
        if obsm_name not in self.adata.obsm:
            raise KeyError(f"obsm '{obsm_name}' not found")
        cols = self._score_matrix_columns(obsm_name)
        if cols is not None and str(column) in cols:
            return cols.index(str(column))
        try:
            idx = int(column)
        except (TypeError, ValueError):
            raise KeyError(f"column '{column}' not found in obsm '{obsm_name}'")
        n = self.adata.obsm[obsm_name].shape[1]
        if idx < 0 or idx >= n:
            raise KeyError(f"column index {idx} out of range for obsm '{obsm_name}'")
        return idx

    def get_obsm_column(self, obsm_name: str, column: str) -> dict[str, Any]:
        """Per-cell values of a named column of an ``.obsm`` matrix (for coloring)."""
        idx = self._resolve_obsm_column_index(obsm_name, column)
        vals = self._obsm_values(obsm_name, idx)
        valid = vals[~np.isnan(vals)]
        return {
            "obsm_name": obsm_name,
            "column": str(column),
            "values": [float(v) if not np.isnan(v) else None for v in vals],
            "min": float(np.min(valid)) if valid.size else 0.0,
            "max": float(np.max(valid)) if valid.size else 0.0,
        }

    def _match_var_names(self, pattern: str, match_mode: str = 'prefix') -> np.ndarray:
        """Boolean mask over var_names matching a prefix or regex."""
        import re
        if not pattern:
            raise ValueError("pattern must be non-empty")
        if match_mode not in ('prefix', 'regex'):
            raise ValueError("match_mode must be 'prefix' or 'regex'")
        names = self.adata.var_names
        if match_mode == 'prefix':
            return np.asarray(names.str.startswith(pattern))
        rx = re.compile(pattern)
        return np.array([bool(rx.search(str(n))) for n in names])

    def add_var_boolean_column(
        self, name: str, pattern: str, match_mode: str = 'prefix',
    ) -> dict[str, Any]:
        """Add a boolean .var column flagging genes whose names match a
        prefix/regex (e.g. mitochondrial, a species of origin).

        Raises ValueError on empty name/pattern, no match, or a name that
        collides with an existing non-boolean .var column.
        """
        if not name:
            raise ValueError("column name must be non-empty")
        if name in self.adata.var.columns and self.adata.var[name].dtype != bool:
            raise ValueError(f"'{name}' already exists and is not boolean")
        mask = self._match_var_names(pattern, match_mode)
        n_matched = int(mask.sum())
        if n_matched == 0:
            raise ValueError(f"no genes match {match_mode} '{pattern}'")
        self.adata.var[name] = mask.astype(bool)
        result = {"name": name, "n_genes_matched": n_matched}
        self._log_action('add_var_boolean_column', {
            'name': name, 'pattern': pattern, 'match_mode': match_mode,
        }, result)
        return result

    def run_calculate_qc_metrics(
        self, qc_vars=None, percent_top=None, log1p: bool = True,
    ) -> dict[str, Any]:
        """Run sc.pp.calculate_qc_metrics(inplace=True).

        qc_vars: list or comma-separated string of boolean .var columns.
        percent_top: None (default; skips top-N columns) or list/comma-string
        of ints. log1p adds the log1p_* columns.
        """
        if isinstance(qc_vars, str):
            qc_vars = [c.strip() for c in qc_vars.split(',') if c.strip()]
        qc_vars = list(qc_vars or [])
        for v in qc_vars:
            if v not in self.adata.var.columns:
                raise ValueError(f"qc_var '{v}' not found in .var")
            if self.adata.var[v].dtype != bool:
                raise ValueError(f"qc_var '{v}' is not boolean")
        if isinstance(percent_top, str):
            percent_top = [int(x) for x in percent_top.split(',') if x.strip()] or None
        obs_before = set(self.adata.obs.columns)
        var_before = set(self.adata.var.columns)
        sc.pp.calculate_qc_metrics(
            self.adata, qc_vars=qc_vars, percent_top=percent_top,
            log1p=log1p, inplace=True,
        )
        result = {
            "qc_vars": qc_vars,
            "n_obs_columns": len(set(self.adata.obs.columns) - obs_before),
            "n_var_columns": len(set(self.adata.var.columns) - var_before),
        }
        self._log_action('calculate_qc_metrics', {
            'qc_vars': qc_vars, 'percent_top': percent_top, 'log1p': log1p,
        }, result)
        return result

    def sum_counts_by_pattern(
        self, pattern: str, match_mode: str = 'prefix',
        obs_name: str | None = None, layer: str = 'counts',
    ) -> dict[str, Any]:
        """Sum counts of genes whose names match a prefix/regex into .obs.

        match_mode in {'prefix','regex'}. Reads layers[layer] (default
        'counts') if present, else .X. Raises ValueError on empty pattern or
        no match.
        """
        import re
        import scipy.sparse as sp
        mask = self._match_var_names(pattern, match_mode)
        n_matched = int(mask.sum())
        if n_matched == 0:
            raise ValueError(f"no genes match {match_mode} '{pattern}'")
        src = layer if layer else 'counts'
        M = self.adata.layers[src] if (src != 'X' and src in self.adata.layers) else self.adata.X
        sub = M[:, mask]
        sums = (np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub)
                else np.asarray(sub).sum(axis=1).ravel())
        if not obs_name:
            base = re.sub(r'[^0-9A-Za-z]+', '', pattern) or 'species'
            obs_name = f"{base}_counts"
        self.adata.obs[obs_name] = sums.astype(float)
        result = {"obs_name": obs_name, "n_genes_matched": n_matched}
        self._log_action('sum_counts_by_pattern', {
            'pattern': pattern, 'match_mode': match_mode,
            'obs_name': obs_name, 'layer': layer,
        }, result)
        return result

    def _species_prefix_candidates(self) -> list[str]:
        """Per-gene genome-prefix candidate, aligned to var_names ('' if none).

        CellRanger multi-genome references prefix every symbol with the genome
        name followed by a run of underscores ("GRCh38_A1BG", "mm10___Xkr4").
        The match is non-greedy so the shortest leading token wins, and the
        lookahead keeps a trailing "ABC_" from swallowing the whole name.
        """
        import re
        rx = re.compile(r'^(.+?_+)(?=.)')
        out = []
        for n in self.adata.var_names:
            m = rx.match(str(n))
            out.append(m.group(1) if m else '')
        return out

    def detect_species_prefixes(self, min_fraction: float = 0.01) -> dict[str, Any]:
        """Report the genome/species prefixes present on var_names. Read-only.

        A candidate counts as a genome only if it covers at least
        ``min_fraction`` of genes (and at least 2), so ordinary symbols that
        merely contain an underscore (MT_ND1, HLA_A) are not mistaken for one.

        Returns {"prefixes": [{"prefix", "label", "n_genes"}, ...],
        "n_unprefixed": int}, prefixes ordered by gene count descending.
        """
        from collections import Counter
        cands = self._species_prefix_candidates()
        threshold = max(2, int(np.ceil(min_fraction * len(cands))))
        tally = Counter(c for c in cands if c)
        kept = {p: n for p, n in tally.items()
                if n >= threshold and p.rstrip('_')}
        prefixes = [{"prefix": p, "label": p.rstrip('_'), "n_genes": int(n)}
                    for p, n in sorted(kept.items(), key=lambda kv: (-kv[1], kv[0]))]
        return {
            "prefixes": prefixes,
            "n_unprefixed": int(sum(1 for c in cands if c not in kept)),
        }

    def _assign_species_prefixes(self, prefixes: list[str], labels: list[str]):
        """Match each gene to the longest prefix it starts with.

        Longest-wins matters when one genome's prefix is a prefix of another's
        ("mm10_" vs "mm10___"), which decides how much gets stripped.
        """
        order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
        matched_prefix, matched_label = [], []
        for n in self.adata.var_names:
            name = str(n)
            hit = ''
            lab = ''
            for i in order:
                if name.startswith(prefixes[i]) and len(name) > len(prefixes[i]):
                    hit, lab = prefixes[i], labels[i]
                    break
            matched_prefix.append(hit)
            matched_label.append(lab)
        return matched_prefix, matched_label

    def rename_genes(
        self, pattern: str, replacement: str = '', match_mode: str = 'regex',
        make_unique: bool = False,
    ) -> dict[str, Any]:
        """Find/replace across gene symbols. An empty replacement strips the
        match — the usual way to drop a species prefix ('^mm10___').

        match_mode 'regex' applies re.sub (so ^ anchors and \\1 backreferences
        work); 'literal' replaces the pattern as plain text everywhere it
        occurs. Originals are preserved in .var['gene_symbol_original'] the
        first time a rename happens.

        Refuses when the result would collide two genes onto one name (unless
        ``make_unique``), since that silently merges genes in every downstream
        lookup. Raises ValueError if nothing matches, so typos are loud.
        """
        import re
        if not pattern:
            raise ValueError("pattern must be non-empty")
        if match_mode not in ('regex', 'literal'):
            raise ValueError("match_mode must be 'regex' or 'literal'")

        old_names = [str(n) for n in self.adata.var_names]
        if match_mode == 'regex':
            try:
                rx = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"invalid regex '{pattern}': {e}")
            new_names = [rx.sub(replacement, n) for n in old_names]
        else:
            new_names = [n.replace(pattern, replacement) for n in old_names]

        changed = [(o, n) for o, n in zip(old_names, new_names) if o != n]
        if not changed:
            raise ValueError(f"no gene names match {match_mode} '{pattern}'")

        blank = [o for o, n in zip(old_names, new_names) if not n]
        if blank:
            raise ValueError(
                f"the replacement would leave {len(blank)} gene(s) with an "
                f"empty name (e.g. {blank[0]}) — narrow the pattern")

        dupes = pd.Index(new_names)
        dupes = dupes[dupes.duplicated()].unique().tolist()
        if dupes and not make_unique:
            shown = ', '.join(map(str, dupes[:5]))
            more = f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else ""
            raise ValueError(
                f"renaming would create {len(dupes)} duplicate gene name(s): "
                f"{shown}{more}. Narrow the pattern, or pass make_unique to "
                f"disambiguate.")

        # Keep the true originals: never overwrite on a repeat rename.
        if 'gene_symbol_original' not in self.adata.var.columns:
            self.adata.var['gene_symbol_original'] = old_names
        index_name = self.adata.var.index.name
        # swap_var_index names the index after a column it then drops, so the
        # name we just re-created a column for can collide. h5ad cannot store
        # an index whose name is also a column with different values.
        if index_name in self.adata.var.columns:
            index_name = None
        self.adata.var.index = pd.Index(new_names)
        self.adata.var.index.name = index_name
        if dupes:
            self.adata.var_names_make_unique()
        # The normalized copy still carries the old names; drop it. The UCell
        # rank cache is keyed by position, not name, and the matrix is
        # untouched by a rename, so it stays valid.
        self._normalized_adata = None

        result = {
            "n_renamed": len(changed),
            "n_genes": len(new_names),
            "n_duplicates": len(dupes),
            "examples": [{"before": o, "after": n} for o, n in changed[:5]],
        }
        self._log_action('rename_genes', {
            'pattern': pattern, 'replacement': replacement,
            'match_mode': match_mode, 'make_unique': make_unique,
        }, result)
        return result

    def add_var_species_column(
        self, species_column: str = 'species', prefixes=None, labels=None,
        min_fraction: float = 0.01, unknown_label: str = 'unknown',
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Annotate each gene with its species/genome in a .var column.

        The column is derived from CellRanger-style prefixes on var_names
        unless ``prefixes`` (comma-separated or list) names them explicitly.
        An existing column is kept as-is unless ``overwrite``, so re-running
        is safe.

        Run this before stripping prefixes with rename_genes: once species
        lives in .var, sum_counts_by_species keeps working on bare symbols.
        """
        detected = self.detect_species_prefixes(min_fraction)
        if prefixes is None:
            prefix_list = [p["prefix"] for p in detected["prefixes"]]
        else:
            if isinstance(prefixes, str):
                prefixes = [p.strip() for p in prefixes.split(',') if p.strip()]
            prefix_list = list(prefixes)
            if not prefix_list:
                raise ValueError("prefixes must be non-empty when provided")

        if labels is None:
            label_list = [p.rstrip('_') for p in prefix_list]
        else:
            if isinstance(labels, str):
                labels = [s.strip() for s in labels.split(',') if s.strip()]
            label_list = list(labels)
            if len(label_list) != len(prefix_list):
                raise ValueError(
                    f"labels ({len(label_list)}) must match the number of "
                    f"prefixes ({len(prefix_list)})")

        matched_prefix, matched_label = self._assign_species_prefixes(
            prefix_list, label_list)

        if prefixes is not None:
            unmatched = [p for p in prefix_list if p not in set(matched_prefix)]
            if unmatched:
                raise ValueError(
                    f"no genes start with prefix(es): {', '.join(unmatched)}")

        derive = overwrite or species_column not in self.adata.var.columns
        if derive:
            if not prefix_list:
                raise ValueError(
                    "no species prefix found on gene names — pass prefixes "
                    "explicitly, or the symbols may already be stripped")
            assigned = np.array(
                [lab or unknown_label for lab in matched_label], dtype=object)
            cats = [c for c in (list(dict.fromkeys(label_list)) + [unknown_label])
                    if c in set(assigned.tolist())]
            self.adata.var[species_column] = pd.Categorical(assigned, categories=cats)

        col = self.adata.var[species_column].astype(str)
        counts = {k: int(v) for k, v in col.value_counts().items() if v}
        return {
            "species_column": species_column,
            "counts": counts,
            "derived": bool(derive),
            "prefixes": detected["prefixes"],
            "n_unknown": int(counts.get(unknown_label, 0)),
        }

    def sum_counts_by_species(
        self, species_column: str = 'species', layer: str = 'counts',
        suffix: str = '_counts', include_unknown: bool = False,
        unknown_label: str = 'unknown',
    ) -> dict[str, Any]:
        """Sum per-cell UMIs for each species in a .var species column.

        Writes one .obs column per species ("GRCh38_counts", "mm10_counts"),
        which feed straight into assign_species. Unlike sum_counts_by_pattern
        this reads the .var annotation instead of re-matching gene names, so it
        still works once the genome prefixes have been stripped.
        """
        import scipy.sparse as sp
        var = self.adata.var
        if species_column not in var.columns:
            raise ValueError(
                f"var column '{species_column}' not found — "
                f"run add_var_species_column first")
        col = var[species_column].astype(str)
        src = layer if layer else 'counts'
        M = (self.adata.layers[src]
             if (src != 'X' and src in self.adata.layers) else self.adata.X)

        series = var[species_column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            ordered = [str(c) for c in series.cat.categories]
        else:
            ordered = [str(c) for c in pd.unique(col)]

        obs_columns: list[str] = []
        n_genes: dict[str, int] = {}
        for lab in ordered:
            if lab == unknown_label and not include_unknown:
                continue
            mask = (col == lab).to_numpy()
            if not mask.any():
                continue
            sub = M[:, mask]
            sums = (np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub)
                    else np.asarray(sub).sum(axis=1).ravel())
            name = f"{lab}{suffix}"
            self.adata.obs[name] = sums.astype(float)
            obs_columns.append(name)
            n_genes[lab] = int(mask.sum())

        if not obs_columns:
            raise ValueError(
                f"no species to count in .var['{species_column}']")
        result = {"obs_columns": obs_columns, "n_genes": n_genes, "layer": src}
        self._log_action('sum_counts_by_species', {
            'species_column': species_column, 'layer': layer, 'suffix': suffix,
            'include_unknown': include_unknown, 'unknown_label': unknown_label,
        }, result)
        return result

    def assign_species(
        self, count_columns, labels=None, obs_name: str = 'species',
        threshold: float = 0.9,
    ) -> dict[str, Any]:
        """Assign each cell a species from per-species count columns.

        For each cell, the argmax-fraction species is assigned iff its fraction
        >= threshold, else 'mixed'; zero-total cells are 'unassigned'.
        """
        if isinstance(count_columns, str):
            count_columns = [c.strip() for c in count_columns.split(',') if c.strip()]
        count_columns = list(count_columns)
        if len(count_columns) < 2:
            raise ValueError("assign_species needs at least 2 count columns")
        obs = self.adata.obs
        for c in count_columns:
            if c not in obs.columns:
                raise ValueError(f"obs column '{c}' not found")
            if not pd.api.types.is_numeric_dtype(obs[c].dtype):
                raise ValueError(f"obs column '{c}' is not numeric")
        if labels is None:
            labels = [c[:-7] if c.endswith('_counts') else c for c in count_columns]
        elif isinstance(labels, str):
            labels = [s.strip() for s in labels.split(',') if s.strip()]
        labels = list(labels)
        if len(labels) != len(count_columns):
            raise ValueError("labels must match count_columns length")
        if not (0 < threshold <= 1):
            raise ValueError("threshold must be in (0, 1]")
        mat = np.column_stack([obs[c].to_numpy(dtype=float) for c in count_columns])
        total = mat.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(total[:, None] > 0, mat / total[:, None], 0.0)
        argmax = frac.argmax(axis=1)
        labels_arr = np.array(labels, dtype=object)
        assigned = np.where(frac[np.arange(len(total)), argmax] >= threshold,
                            labels_arr[argmax], 'mixed').astype(object)
        assigned[total <= 0] = 'unassigned'
        cats = [c for c in (list(dict.fromkeys(labels)) + ['mixed', 'unassigned'])
                if c in set(assigned.tolist())]
        self.adata.obs[obs_name] = pd.Categorical(assigned, categories=cats)
        counts = {c: int((assigned == c).sum()) for c in cats}
        result = {"obs_name": obs_name, "counts": counts}
        self._log_action('assign_species', {
            'count_columns': list(count_columns), 'labels': list(labels),
            'obs_name': obs_name, 'threshold': threshold,
        }, result)
        return result

    def transform_embedding(
        self,
        name: str,
        rotation_degrees: float = 0,
        reflect_x: bool = False,
        reflect_y: bool = False,
        cell_indices: list[int] | None = None,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
        dim_x: int = 0,
        dim_y: int = 1,
    ) -> dict[str, Any]:
        """Apply rotation, reflection, and/or translation to an embedding in-place.

        Operates on the two currently-viewed columns (``dim_x``, ``dim_y``; default
        the first two) so a transform matches whatever the user is looking at.
        Transforms are applied around the centroid (of the subset if cell_indices
        is provided, otherwise of all cells): reflections first, then rotation,
        then translation.
        """
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {list(self.adata.obsm.keys())}")
        dx, dy = self._clamp_dims(name, dim_x, dim_y)
        cols = [dx, dy]

        if cell_indices is not None:
            # Snapshot the affected columns for undo (records which columns).
            stack = self._embedding_undo_stacks.setdefault(name, [])
            stack.append((dx, dy, np.array(self.adata.obsm[name][:, cols], copy=True)))

            # Subset mode: only transform specified cells
            idx = np.array(cell_indices, dtype=int)
            coords = np.array(self.adata.obsm[name][np.ix_(idx, cols)], dtype=np.float64)
            centroid = coords.mean(axis=0)

            coords -= centroid

            if reflect_y:
                coords[:, 0] *= -1
            if reflect_x:
                coords[:, 1] *= -1

            if rotation_degrees != 0:
                theta = np.radians(rotation_degrees)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                coords = coords @ rot.T

            coords += centroid

            # Apply translation
            coords[:, 0] += translate_x
            coords[:, 1] += translate_y

            # Write back only the subset
            self.adata.obsm[name][np.ix_(idx, cols)] = coords
        else:
            # Full-embedding mode
            coords = np.array(self.adata.obsm[name][:, cols], dtype=np.float64)
            centroid = coords.mean(axis=0)

            coords -= centroid

            if reflect_y:
                coords[:, 0] *= -1
            if reflect_x:
                coords[:, 1] *= -1

            if rotation_degrees != 0:
                theta = np.radians(rotation_degrees)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                coords = coords @ rot.T

            coords += centroid

            # Apply translation (for full embedding too, though less common)
            if translate_x != 0 or translate_y != 0:
                coords[:, 0] += translate_x
                coords[:, 1] += translate_y

            self.adata.obsm[name][:, cols] = coords

        # Clear normalized cache (it may share obsm references)
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        result = self.get_embedding(name, dim_x=dx, dim_y=dy)
        result["undo_depth"] = len(self._embedding_undo_stacks.get(name, []))
        return result

    def undo_transform_embedding(self, name: str, dim_x: int = 0, dim_y: int = 1) -> dict[str, Any]:
        """Undo the last transform for an embedding.

        Pops the most recent snapshot (which records the columns it captured) and
        restores those columns. Returns the view at the requested dims.
        """
        stack = self._embedding_undo_stacks.get(name, [])
        if not stack:
            raise ValueError(f"No undo history for embedding '{name}'")
        snap = stack.pop()
        # Back-compat: older snapshots were bare arrays (columns 0,1).
        if isinstance(snap, tuple):
            sdx, sdy, coords = snap
        else:
            sdx, sdy, coords = 0, 1, snap
        self.adata.obsm[name][:, [sdx, sdy]] = coords
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None
        result = self.get_embedding(name, dim_x=dim_x, dim_y=dim_y)
        result["undo_depth"] = len(stack)
        return result

    def get_obs_column(self, name: str) -> dict[str, Any]:
        """Get cell metadata column values.

        Args:
            name: Name of the column in .obs

        Returns:
            Dictionary containing:
            - name: The column name
            - values: List of values for each cell
            - dtype: Data type ('category', 'numeric', or 'string')
            - categories: List of category names (only for categorical columns)

        Raises:
            KeyError: If column name not found in .obs
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Column '{name}' not found. Available: {list(self.adata.obs.columns)}")

        series = self.adata.obs[name]
        dtype = series.dtype

        result: dict[str, Any] = {
            "name": name,
        }

        if pd.api.types.is_categorical_dtype(dtype):
            result["dtype"] = "category"
            result["values"] = series.cat.codes.tolist()
            # Stringify so the wire format matches /api/obs/summary/{col}, which
            # uses str(val). Without this, float-valued categoricals (e.g. from
            # contourize) yield raw floats here vs. strings in the summary, and
            # frontend lookups by category value silently fail.
            result["categories"] = [str(v) for v in series.cat.categories.tolist()]
            colors = self._get_category_colors(name)
            if colors is not None:
                result["colors"] = colors
        elif pd.api.types.is_numeric_dtype(dtype):
            result["dtype"] = "numeric"
            # Handle NaN values by converting to None
            values = series.tolist()
            result["values"] = [None if pd.isna(v) else v for v in values]
        else:
            result["dtype"] = "string"
            result["values"] = series.astype(str).tolist()

        return result

    def get_cell_indices(self) -> list[str]:
        """Get cell index names/barcodes.

        Returns:
            List of cell identifiers from .obs.index
        """
        return self.adata.obs.index.tolist()

    def get_gene_names(self) -> list[str]:
        """Get gene names.

        Returns:
            List of gene names from .var.index
        """
        return self.adata.var.index.tolist()

    def get_var_identifier_columns(self) -> dict[str, Any]:
        """Get .var columns that could serve as gene identifiers.

        Returns columns with string/object dtype and >90% unique values.
        Includes '_index' representing the current index.

        Returns:
            Dictionary with 'columns' (list of column names including '_index')
            and 'current' (name of the current index, or '_index' if unnamed).
        """
        candidates = ['_index']  # Always include current index
        n_genes = self.adata.n_vars
        if n_genes == 0:
            index_name = self.adata.var.index.name or '_index'
            return {'columns': candidates, 'current': index_name if index_name != '_index' else '_index'}

        for col in self.adata.var.columns:
            series = self.adata.var[col]
            # Must be string/object dtype (including categorical with string categories)
            if hasattr(series, 'cat'):
                if not pd.api.types.is_string_dtype(series.cat.categories):
                    continue
            elif series.dtype not in ('object', 'string', 'str'):
                if not pd.api.types.is_string_dtype(series):
                    continue
            # Skip boolean-like columns
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2 and set(str(v).lower() for v in unique_vals).issubset({'true', 'false', '0', '1', 'yes', 'no'}):
                continue
            # Must have >90% unique values
            n_unique = series.nunique()
            if n_unique / n_genes > 0.9:
                candidates.append(col)

        current = self.adata.var.index.name or '_index'
        return {'columns': candidates, 'current': current}

    def swap_var_index(self, column_name: str) -> dict[str, Any]:
        """Swap the .var index with another column.

        Moves the current index into a .var column and promotes the
        specified column to the index. Clears expression caches.

        Args:
            column_name: Name of the .var column to use as the new index.

        Returns:
            Updated schema dict (same format as get_schema()).

        Raises:
            KeyError: If column_name is not in .var columns.
        """
        if column_name not in self.adata.var.columns:
            raise KeyError(f"Column '{column_name}' not found in .var")

        # Save current index as a column
        old_index_name = self.adata.var.index.name or '_prev_index'
        # Avoid collision if column already exists
        save_name = old_index_name
        if save_name in self.adata.var.columns:
            save_name = f"{save_name}_orig"
        self.adata.var[save_name] = self.adata.var.index

        # Set new index
        self.adata.var.index = self.adata.var[column_name].values
        self.adata.var.index.name = column_name
        # Remove the column (it's now the index)
        self.adata.var.drop(columns=[column_name], inplace=True)

        # Handle duplicates
        self.adata.var_names_make_unique()

        # Clear caches
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        # Regenerate the visible-gene mask since .var axis may have changed.
        # If referenced columns no longer exist, the mask is cleared.
        self._regenerate_gene_mask_after_var_change()

        self._log_action('swap_var_index', {'column_name': column_name},
                         {'n_genes': int(self.n_genes)})
        return self.get_schema()

    # Columns whose name already says "these are symbols", checked before
    # offering to create another one.
    _SYMBOL_COLUMNS = ('gene_symbol', 'gene_symbols', 'symbol', 'SYMBOL',
                       'gene_name', 'gene_names', 'feature_name')

    def _gene_symbol_state(self) -> tuple[dict[str, Any], str | None]:
        """(species detection over var_names, an existing symbol column or None)."""
        from xcell import gene_symbols as gs

        ids = [str(n) for n in self.adata.var_names]
        existing = next(
            (c for c in self._SYMBOL_COLUMNS if c in self.adata.var.columns), None)
        return gs.detect_species(ids), existing

    def preview_gene_symbol_mapping(self) -> dict[str, Any]:
        """What mapping Ensembl ids to symbols would do here. Mutates nothing.

        The cost is shown while the decision is still the user's, the same
        contract as the gene-overlap panel in Localize. It is also how the
        feature says *you do not need this*.
        """
        from xcell import gene_symbols as gs

        detection, existing = self._gene_symbol_state()
        ids = [str(n) for n in self.adata.var_names]
        out: dict[str, Any] = {
            'species': detection['species'],
            'n_genes': int(len(ids)),
            'n_recognized': int(detection['n_recognized']),
            'existing_symbol_column': existing,
            'n_mapped': 0,
            'n_unmapped': 0,
            'n_duplicate_symbols': 0,
            'examples': [],
            'applicable': False,
            'reason': '',
        }

        if detection['species'] is None:
            if detection['unsupported']:
                out['reason'] = (
                    f"These look like {detection['unsupported']} Ensembl ids. "
                    'xcell ships symbol tables for human and mouse only.')
            else:
                out['reason'] = (
                    'These gene names are not Ensembl ids, so there is nothing '
                    'to map — they are already symbols, or another identifier.')
            return out

        mapped = gs.map_ids(ids, gs.load_table(detection['species']))
        out.update({
            'n_mapped': int(mapped['n_mapped']),
            'n_unmapped': int(mapped['n_unmapped']),
            'n_duplicate_symbols': int(mapped['n_duplicate_symbols']),
            'examples': mapped['examples'],
            'applicable': True,
        })
        if existing:
            # Not a refusal — the existing column may be stale or partial. Say
            # it is there so a user whose column is fine can just switch to it.
            out['reason'] = (
                f".var['{existing}'] already looks like symbols, so you may not "
                'need this — the Gene IDs picker can switch to it.')
        return out

    def map_gene_symbols(
        self, *, column: str = 'gene_symbol', set_as_index: bool = False,
    ) -> dict[str, Any]:
        """Write official symbols for Ensembl ids into ``.var[column]``.

        Unmapped ids keep their own id, so no gene becomes unaddressable.
        Duplicate symbols are reported rather than refused: unlike a rename
        pattern, where a collision means the user typed something too broad, a
        collision here is a fact about the annotation with nothing to correct.
        """
        from xcell import gene_symbols as gs

        if not column:
            raise ValueError('column must be a non-empty name')
        detection, _ = self._gene_symbol_state()
        if detection['species'] is None:
            if detection['unsupported']:
                raise ValueError(
                    f"These look like {detection['unsupported']} Ensembl ids; "
                    'xcell ships symbol tables for human and mouse only.')
            raise ValueError(
                'These gene names are not Ensembl ids, so there is nothing to '
                'map.')

        ids = [str(n) for n in self.adata.var_names]
        mapped = gs.map_ids(ids, gs.load_table(detection['species']))
        # Plain strings, not Categorical: with set_as_index this column becomes
        # the .var index, and AnnData wants that to be strings — a categorical
        # index warns on assignment and is a hazard on save.
        self.adata.var[column] = pd.Index(mapped['symbols'], dtype=object)

        result = {
            'species': detection['species'],
            'column': column,
            'n_genes': int(len(ids)),
            'n_mapped': int(mapped['n_mapped']),
            'n_unmapped': int(mapped['n_unmapped']),
            'n_duplicate_symbols': int(mapped['n_duplicate_symbols']),
            'examples': mapped['examples'],
            'set_as_index': bool(set_as_index),
        }
        self._log_action('map_gene_symbols',
                         {'column': column, 'set_as_index': set_as_index},
                         result)
        if set_as_index:
            # Name the outgoing index first. swap_var_index parks it in a
            # column called after it, falling back to '_prev_index' — which is
            # what the Gene IDs picker would then offer as the way back to the
            # ids. Naming it says what it is.
            if not self.adata.var.index.name:
                fallback = 'ensembl_id'
                if fallback not in self.adata.var.columns:
                    self.adata.var.index.name = fallback
            # The one implementation of this operation, cache invalidation and
            # gene-mask regeneration included. Re-doing it here is how caches
            # get out of step with the axis.
            self.swap_var_index(column)
        return result

    def search_genes(self, query: str, limit: int = 20) -> list[str]:
        """Search for genes by name prefix.

        Args:
            query: Search query (case-insensitive prefix match)
            limit: Maximum number of results to return

        Returns:
            List of matching gene names (restricted to visible genes when
            a gene mask is active)
        """
        query_lower = query.lower()
        gene_names = self.get_visible_gene_names()

        # Find genes that start with the query (case-insensitive)
        matches = [g for g in gene_names if g.lower().startswith(query_lower)]

        # If not enough prefix matches, also include substring matches
        if len(matches) < limit:
            substring_matches = [
                g for g in gene_names
                if query_lower in g.lower() and g not in matches
            ]
            matches.extend(substring_matches)

        return matches[:limit]

    def get_expression(
        self,
        gene: str,
        transform: str | None = None,
        clip_percentile: float = 0.0,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Get expression values for a single gene across all cells.

        Args:
            gene: Gene name
            transform: Optional transformation to apply. Supported values:
                - None: Raw expression values
                - "log1p": Apply normalize_total followed by log1p transformation
            clip_percentile: Symmetric percentile clip (0 = no clipping). When
                >0, values are clipped at [clip_percentile, 100-clip_percentile]
                and the returned min/max are those clipped anchors so the
                frontend colormap stretches across the bulk of the distribution.
            layer: Optional layer name. When set (and not 'X'), reads expression
                from ``adata.layers[layer]`` directly and ignores ``transform``
                — the layer is treated as already in the right scale.

        Returns:
            Dictionary containing:
            - gene: The gene name
            - values: List of expression values for each cell
            - min: Minimum expression value
            - max: Maximum expression value
            - transform: The transformation applied (if any)
            - layer: The layer name read from (if non-default)

        Raises:
            KeyError: If gene not found in .var
        """
        if gene not in self.adata.var.index:
            raise KeyError(f"Gene '{gene}' not found in dataset")

        # Get gene index
        gene_idx = self.adata.var.index.get_loc(gene)

        # Layer overrides transform: a layer is treated as authoritative source.
        if layer is not None and layer != 'X':
            X = self._resolve_source_matrix(layer)
        elif transform == "log1p":
            X = self.normalized_adata.X
        else:
            X = self.adata.X

        # Get expression values from the chosen matrix.
        if hasattr(X, 'toarray'):
            values = X[:, gene_idx].toarray().flatten()
        else:
            values = X[:, gene_idx].flatten()

        values = np.asarray(values, dtype=np.float64)

        # Optional symmetric percentile clip. We clip the values in-place so
        # the frontend colormap (min..max) stretches across the bulk of the
        # distribution instead of being dominated by a few outliers.
        if clip_percentile and clip_percentile > 0:
            valid = values[~np.isnan(values)]
            if valid.size > 0:
                lo = float(np.percentile(valid, clip_percentile))
                hi = float(np.percentile(valid, 100 - clip_percentile))
                if hi > lo:
                    values = np.clip(values, lo, hi)

        # Convert to regular Python floats and handle NaN
        values_list = []
        for v in values:
            if np.isnan(v):
                values_list.append(None)
            else:
                values_list.append(float(v))

        # Calculate min/max excluding None values
        valid_values = [v for v in values_list if v is not None]
        min_val = min(valid_values) if valid_values else 0
        max_val = max(valid_values) if valid_values else 0

        result = {
            "gene": gene,
            "values": values_list,
            "min": min_val,
            "max": max_val,
        }
        if transform:
            result["transform"] = transform
        if layer and layer != 'X':
            result["layer"] = layer
        return result

    @staticmethod
    def _normalize_gene_column(
        values: np.ndarray,
        method: str,
        clip_percentile: float = 0.0,
    ) -> np.ndarray:
        """Normalize a single gene's per-cell expression vector.

        Args:
            values: 1D float array, one entry per cell.
            method: 'none' | 'zscore_mad' | 'zscore_sd' | 'minmax' | 'rank'.
            clip_percentile: For 'minmax', symmetric percentile clip applied
                before the rescale (e.g. 1.0 = clip at 1st/99th). Ignored for
                other methods.
        """
        v = np.asarray(values, dtype=np.float64)
        if method == 'none':
            return v
        if method == 'zscore_mad':
            mean = np.nanmean(v)
            centered = v - mean
            mad = np.nanmedian(np.abs(centered - np.nanmedian(centered)))
            if mad > 0:
                return centered / (mad * 1.4826)
            sd = np.nanstd(v)
            if sd > 0:
                return centered / sd
            return np.zeros_like(v)
        if method == 'zscore_sd':
            mean = np.nanmean(v)
            sd = np.nanstd(v)
            if sd > 0:
                return (v - mean) / sd
            return np.zeros_like(v)
        if method == 'minmax':
            valid = v[~np.isnan(v)]
            if valid.size == 0:
                return np.zeros_like(v)
            if clip_percentile and clip_percentile > 0:
                lo = float(np.percentile(valid, clip_percentile))
                hi = float(np.percentile(valid, 100 - clip_percentile))
            else:
                lo = float(np.nanmin(valid))
                hi = float(np.nanmax(valid))
            if hi <= lo:
                return np.zeros_like(v)
            clipped = np.clip(v, lo, hi)
            return (clipped - lo) / (hi - lo)
        if method == 'rank':
            # Average ranks (handles ties) → divide by N to land in [0,1].
            # NaNs propagate as NaN.
            from scipy.stats import rankdata
            mask_valid = ~np.isnan(v)
            out = np.full_like(v, np.nan, dtype=np.float64)
            if mask_valid.any():
                ranks = rankdata(v[mask_valid], method='average')
                out[mask_valid] = ranks / max(ranks.size, 1)
            return out
        raise ValueError(f"Unknown per-gene normalization method: {method!r}")

    @staticmethod
    def _aggregate_across_genes(
        matrix: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Aggregate a (n_cells, n_genes) float matrix to a 1D per-cell score.

        method: 'mean' | 'median' | 'sum' | 'max'. NaN-aware.
        """
        if method == 'mean':
            return np.nanmean(matrix, axis=1)
        if method == 'median':
            return np.nanmedian(matrix, axis=1)
        if method == 'sum':
            return np.nansum(matrix, axis=1)
        if method == 'max':
            # all-NaN columns return NaN under np.nanmax with a runtime warning;
            # squelch by masking
            with np.errstate(all='ignore'):
                return np.nanmax(matrix, axis=1)
        raise ValueError(f"Unknown aggregation method: {method!r}")

    def _ucell_ranks(self, layer: str | None, max_rank: int):
        """Per-cell capped descending gene ranks as a sparse CSC matrix.

        Ranks each cell's genes by expression descending (rank 1 = highest),
        average ties, then caps at ``max_rank`` (also capped to n_genes, per
        UCell). Ranks >= max_rank are dropped to 0 (sparse), meaning "treat as
        max_rank" when scoring. Result shape (n_cells, n_genes), cached on the
        adaptor keyed by (resolved layer, max_rank) and invalidated whenever
        ``self.adata`` is reassigned. Source layer 'counts' falls back to X.
        """
        from scipy.stats import rankdata
        import scipy.sparse as sp

        if layer in (None, 'counts'):
            resolved = 'counts' if 'counts' in self.adata.layers else 'X'
        else:
            resolved = layer
        n_cells, n_genes = self.adata.shape
        eff_max_rank = int(min(max_rank, n_genes))
        key = (resolved, eff_max_rank)

        if self._ucell_rank_cache_adata_id != id(self.adata):
            self._ucell_rank_cache = {}
            self._ucell_rank_cache_adata_id = id(self.adata)
        if key in self._ucell_rank_cache:
            return self._ucell_rank_cache[key]

        if resolved == 'X':
            M = self.adata.X
        elif resolved in self.adata.layers:
            M = self.adata.layers[resolved]
        else:
            raise ValueError(f"Layer '{resolved}' not found for UCell ranking")

        chunk = 2000
        blocks = []
        for start in range(0, n_cells, chunk):
            block = M[start:start + chunk]
            dense = block.toarray() if sp.issparse(block) else np.asarray(block)
            dense = dense.astype(np.float64, copy=False)
            r = rankdata(-dense, method='average', axis=1)
            r[r >= eff_max_rank] = 0.0
            blocks.append(sp.csr_matrix(r.astype(np.float32)))
        ranks = sp.vstack(blocks).tocsc() if blocks else sp.csc_matrix((0, n_genes))
        self._ucell_rank_cache[key] = ranks
        return ranks

    def _ucell_score_one(self, up_idx, down_idx, ranks, max_rank, w_neg):
        """Per-cell UCell score for one signature given the cached rank matrix.

        ``ranks`` is the sparse CSC from ``_ucell_ranks`` (0 means rank>=max_rank).
        Returns a float64 array of shape (n_cells,).
        """
        n_cells = ranks.shape[0]

        def u_stat(idx):
            n = len(idx)
            if n == 0:
                return np.zeros(n_cells, dtype=np.float64)
            sub = ranks[:, idx]
            sum_stored = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
            nnz = sub.getnnz(axis=1).astype(np.float64)
            # stored entries hold the true rank (<max_rank); missing -> max_rank
            rank_sum = n * max_rank - max_rank * nnz + sum_stored
            rank_sum_min = n * (n + 1) / 2.0
            denom = n * max_rank - rank_sum_min
            if denom <= 0:
                return np.ones(n_cells, dtype=np.float64)
            return 1.0 - (rank_sum - rank_sum_min) / denom

        u_p = u_stat(up_idx)
        u_n = u_stat(down_idx) if down_idx else 0.0
        return np.maximum(u_p - w_neg * u_n, 0.0)

    def ucell_score_values(
        self, up: list[str], down: list[str] | None = None,
        layer: str = 'counts', max_rank: int = 1500, w_neg: float = 1.0,
    ) -> dict[str, Any]:
        """Compute (non-persisted) per-cell UCell scores for one signature.

        Filters up/down to genes present in .var (missing skipped). A signature
        with no usable up-genes scores 0 everywhere (UCell property). Returns
        values + min/max + counts of genes used.
        """
        down = down or []
        var_index = self.adata.var.index
        up_g = [g for g in up if g in var_index]
        down_g = [g for g in down if g in var_index]
        n_genes = self.adata.shape[1]
        eff_max_rank = int(min(max(max_rank, len(up_g), len(down_g), 1), n_genes))
        ranks = self._ucell_ranks(layer, eff_max_rank)
        up_idx = [var_index.get_loc(g) for g in up_g]
        down_idx = [var_index.get_loc(g) for g in down_g]
        if not up_idx:
            scores = np.zeros(self.adata.shape[0], dtype=np.float64)
        else:
            scores = self._ucell_score_one(up_idx, down_idx, ranks, eff_max_rank, w_neg)
        return {
            "values": [float(v) for v in scores],
            "min": float(scores.min()) if scores.size else 0.0,
            "max": float(scores.max()) if scores.size else 0.0,
            "n_up_used": len(up_idx),
            "n_down_used": len(down_idx),
            "max_rank": eff_max_rank,
        }

    @staticmethod
    def _sanitize_obs_name(name: str) -> str:
        import re
        base = re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_') or 'set'
        return f"UCell_{base}"

    def _unique_obs_name(self, base: str) -> str:
        if base not in self.adata.obs.columns:
            return base
        i = 1
        while f"{base}_{i}" in self.adata.obs.columns:
            i += 1
        return f"{base}_{i}"

    def score_gene_sets_ucell(
        self, sets: list[dict[str, Any]], layer: str = 'counts',
        max_rank: int = 1500, w_neg: float = 1.0,
    ) -> dict[str, Any]:
        """Score directional gene sets with UCell and write .obs columns.

        Each set is {name, up:[...], down:[...]}. All sets share one rank matrix
        (one eff_max_rank for comparability). Sets with no usable up-genes are
        skipped (would score 0). Writes obs[UCell_<name>] (collision-safe) and
        returns per-set metadata.
        """
        var_index = self.adata.var.index
        n_genes = self.adata.shape[1]
        prepared = []
        for s in sets:
            name = s.get("name") or "set"
            up = [g for g in (s.get("up") or []) if g in var_index]
            down = [g for g in (s.get("down") or []) if g in var_index]
            prepared.append((name, up, down))
        longest = max([len(u) for _, u, _ in prepared]
                      + [len(d) for _, _, d in prepared] + [1])
        eff_max_rank = int(min(max(max_rank, longest), n_genes))
        ranks = self._ucell_ranks(layer, eff_max_rank)

        results = []
        for name, up, down in prepared:
            if not up:
                results.append({"name": name, "skipped": "no up-genes present in dataset"})
                continue
            up_idx = [var_index.get_loc(g) for g in up]
            down_idx = [var_index.get_loc(g) for g in down]
            scores = self._ucell_score_one(up_idx, down_idx, ranks, eff_max_rank, w_neg)
            col = self._unique_obs_name(self._sanitize_obs_name(name))
            self.adata.obs[col] = scores.astype(np.float64)
            results.append({
                "name": name, "obs_column": col,
                "min": float(scores.min()), "max": float(scores.max()),
                "n_up_used": len(up_idx), "n_down_used": len(down_idx),
            })
        out = {"results": results, "max_rank": eff_max_rank, "layer": layer}
        self._log_action('score_gene_sets_ucell', {
            'sets': [{'name': s.get('name'),
                      'genes': list(s.get('genes') or []),
                      'genes_down': list(s.get('genes_down') or [])} for s in sets],
            'layer': layer, 'max_rank': max_rank, 'w_neg': w_neg,
        }, {
            'max_rank': eff_max_rank,
            'obs_columns': [r['obs_column'] for r in results if 'obs_column' in r],
            'n_skipped': sum(1 for r in results if 'skipped' in r),
        })
        return out

    def _aggregate_gene_set_scores(
        self,
        genes: list[str],
        source_matrix,
        var_index=None,
        per_gene_norm: str = 'zscore_mad',
        per_gene_clip: float = 0.0,
        aggregation: str = 'mean',
    ) -> np.ndarray:
        """Build per-cell summary score for a gene set.

        Pulls each gene's column from ``source_matrix`` (a 2-D matrix shaped
        like ``adata.X``: n_cells × n_genes), normalizes per-gene according to
        ``per_gene_norm`` (with optional ``per_gene_clip`` for the 'minmax'
        path), then aggregates across genes via ``aggregation``. Returns a 1D
        float64 array of shape ``(n_cells,)``.

        ``var_index`` defaults to ``self.adata.var.index``. Pass an explicit
        index only if you're operating on a non-default matrix whose columns
        align differently — for the standard `.X` and `adata.layers[*]` cases
        the column order matches ``self.adata.var`` so the default works.
        """
        if var_index is None:
            var_index = self.adata.var.index
        gene_indices = [var_index.get_loc(g) for g in genes]
        if hasattr(source_matrix, 'toarray'):
            expr_matrix = source_matrix[:, gene_indices].toarray().astype(np.float64)
        else:
            expr_matrix = np.asarray(source_matrix[:, gene_indices], dtype=np.float64)

        if per_gene_norm == 'none':
            normed = expr_matrix
        else:
            cols = []
            for j in range(expr_matrix.shape[1]):
                cols.append(self._normalize_gene_column(
                    expr_matrix[:, j], per_gene_norm, per_gene_clip,
                ))
            normed = np.column_stack(cols)

        return self._aggregate_across_genes(normed, aggregation)

    def get_multi_gene_expression(
        self,
        genes: list[str],
        transform: str | None = None,
        per_gene_norm: str = 'zscore_mad',
        per_gene_clip: float = 0.0,
        aggregation: str = 'mean',
        clip_percentile: float = 1.0,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated expression values for multiple genes across all cells.

        The aggregation pipeline is:
            source → per-gene normalize → aggregate across genes
            → optional symmetric percentile clip (cell-level)

        Args:
            genes: List of gene names.
            transform: Optional source transformation. ``None`` reads
                ``adata.X``; ``"log1p"`` reads ``self.normalized_adata.X``
                (lazy ``normalize_total`` + ``log1p``).
            per_gene_norm: How each gene's column is normalized before
                aggregation. One of ``'none'``, ``'zscore_mad'`` (default —
                mean-center, MAD-scale, fallback SD, fallback zeros),
                ``'zscore_sd'`` (mean-center, SD-scale), ``'minmax'`` (clip
                at ``per_gene_clip`` percentile then rescale to [0, 1]), or
                ``'rank'`` (average-rank, divided by N → [0, 1]).
            per_gene_clip: Percentile clip used when ``per_gene_norm='minmax'``.
                Ignored for other methods. 0 means use raw min/max.
            aggregation: How per-gene values are combined per cell. One of
                ``'mean'`` (default), ``'median'``, ``'sum'``, ``'max'``.
            clip_percentile: Symmetric percentile clip applied to the
                aggregated per-cell scores (0 = no clip). Anchors the
                returned ``min``/``max`` for the colormap.

        Returns:
            Dictionary with: genes (used), values, min, max, transform (if any),
            n_masked_excluded, plus the per_gene_norm / per_gene_clip /
            aggregation / clip_percentile actually used.
        """
        # Filter to only genes present in the dataset (silently skip missing),
        # then drop any genes currently masked by the gene mask.
        valid_genes = [g for g in genes if g in self.adata.var.index]
        valid_genes, n_masked_excluded = self._filter_to_visible(valid_genes)

        if len(valid_genes) == 0:
            return {
                "genes": [],
                "values": [0.0] * self.n_cells,
                "min": 0.0,
                "max": 0.0,
                "n_masked_excluded": n_masked_excluded,
                "per_gene_norm": per_gene_norm,
                "per_gene_clip": per_gene_clip,
                "aggregation": aggregation,
                "clip_percentile": clip_percentile,
            }

        genes = valid_genes

        # Layer overrides transform: a layer is treated as authoritative source.
        if layer is not None and layer != 'X':
            source_matrix = self._resolve_source_matrix(layer)
        elif transform == "log1p":
            source_matrix = self.normalized_adata.X
        else:
            source_matrix = self.adata.X

        scores = self._aggregate_gene_set_scores(
            genes=genes,
            source_matrix=source_matrix,
            per_gene_norm=per_gene_norm,
            per_gene_clip=per_gene_clip,
            aggregation=aggregation,
        )

        # Optional cross-cell symmetric percentile clip — anchors the color
        # ramp to the bulk of the distribution.
        if clip_percentile and clip_percentile > 0:
            valid_scores = scores[~np.isnan(scores)]
            if valid_scores.size > 0:
                lo = float(np.percentile(valid_scores, clip_percentile))
                hi = float(np.percentile(valid_scores, 100 - clip_percentile))
                if hi > lo:
                    scores = np.clip(scores, lo, hi)
                    min_val = lo
                    max_val = hi
                else:
                    min_val = float(np.nanmin(scores)) if valid_scores.size else 0.0
                    max_val = float(np.nanmax(scores)) if valid_scores.size else 0.0
            else:
                min_val, max_val = 0.0, 0.0
        else:
            valid_scores = scores[~np.isnan(scores)]
            min_val = float(np.nanmin(valid_scores)) if valid_scores.size else 0.0
            max_val = float(np.nanmax(valid_scores)) if valid_scores.size else 0.0

        values_list = [
            float(v) if not np.isnan(v) else None
            for v in scores
        ]

        result = {
            "genes": genes,
            "values": values_list,
            "min": min_val,
            "max": max_val,
            "n_masked_excluded": n_masked_excluded,
            "per_gene_norm": per_gene_norm,
            "per_gene_clip": per_gene_clip,
            "aggregation": aggregation,
            "clip_percentile": clip_percentile,
        }
        if transform:
            result["transform"] = transform
        if layer and layer != 'X':
            result["layer"] = layer
        return result

    def get_bivariate_expression(
        self,
        genes1: list[str],
        genes2: list[str],
        transform: str | None = None,
        per_gene_norm: str = 'zscore_mad',
        per_gene_clip: float = 0.0,
        aggregation: str = 'mean',
        clip_percentile: float = 1.0,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Get [0, 1]-normalized scores for two gene sets for bivariate coloring.

        Same per-gene-norm + aggregation pipeline as
        :meth:`get_multi_gene_expression`, run independently per axis. The
        bivariate visualization needs both axes in [0, 1], so the per-cell
        scores are then symmetrically clipped at ``clip_percentile`` and
        rescaled to [0, 1] (``(score - lo) / (hi - lo)``).

        Args:
            genes1: Gene names for the first axis (e.g. red / x).
            genes2: Gene names for the second axis (e.g. blue / y).
            transform: ``None`` or ``"log1p"`` (route through normalized_adata).
            per_gene_norm: 'none' | 'zscore_mad' | 'zscore_sd' | 'minmax' | 'rank'.
            per_gene_clip: Percentile clip used by the 'minmax' per-gene norm.
            aggregation: 'mean' | 'median' | 'sum' | 'max'.
            clip_percentile: Symmetric percentile clip on the per-cell scores
                before the [0, 1] rescale (0 = use raw min/max).
        """
        genes1 = [g for g in genes1 if g in self.adata.var.index]
        genes2 = [g for g in genes2 if g in self.adata.var.index]

        if len(genes1) == 0 or len(genes2) == 0:
            raise ValueError("No valid genes found in one or both gene sets after filtering to dataset genes")

        if layer is not None and layer != 'X':
            source_matrix = self._resolve_source_matrix(layer)
        elif transform == "log1p":
            source_matrix = self.normalized_adata.X
        else:
            source_matrix = self.adata.X

        def summarize(genes: list[str]) -> list[float]:
            scores = self._aggregate_gene_set_scores(
                genes=genes,
                source_matrix=source_matrix,
                per_gene_norm=per_gene_norm,
                per_gene_clip=per_gene_clip,
                aggregation=aggregation,
            )
            valid = scores[~np.isnan(scores)]
            if valid.size == 0:
                return [0.5] * len(scores)

            if clip_percentile and clip_percentile > 0:
                lo = float(np.percentile(valid, clip_percentile))
                hi = float(np.percentile(valid, 100 - clip_percentile))
            else:
                lo = float(np.nanmin(valid))
                hi = float(np.nanmax(valid))

            if hi > lo:
                clipped = np.clip(scores, lo, hi)
                normalized = (clipped - lo) / (hi - lo)
            else:
                normalized = np.full_like(scores, 0.5)
            return [float(v) if not np.isnan(v) else 0.5 for v in normalized]

        result = {
            "genes1": genes1,
            "genes2": genes2,
            "values1": summarize(genes1),
            "values2": summarize(genes2),
            "per_gene_norm": per_gene_norm,
            "per_gene_clip": per_gene_clip,
            "aggregation": aggregation,
            "clip_percentile": clip_percentile,
        }
        if transform:
            result["transform"] = transform
        if layer and layer != 'X':
            result["layer"] = layer
        return result

    def _get_category_colors(self, name: str) -> list[str] | None:
        """Read positional hex colors from adata.uns[f'{name}_colors'].

        Follows the scanpy convention: a list aligned to .cat.categories. Returns
        None if the column isn't categorical, the uns key is missing, the length
        doesn't match, or any entry isn't a string. Numpy str_ values pass since
        they subclass str.
        """
        series = self.adata.obs.get(name)
        if series is None or not pd.api.types.is_categorical_dtype(series.dtype):
            return None
        raw = self.adata.uns.get(f"{name}_colors")
        if raw is None:
            return None
        try:
            colors = list(raw)
        except TypeError:
            return None
        if len(colors) != len(series.cat.categories):
            return None
        out: list[str] = []
        for c in colors:
            if not isinstance(c, str):
                return None
            out.append(str(c))
        return out

    def crosstab(
        self, column_a: str, column_b: str,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Count cells by two .obs columns at once, for the stacked barplot.

        ``active_cell_indices`` restricts the count to those cells, so a
        composition drawn while a mask is active describes the cells the user
        is looking at rather than the whole dataset.

        A stacked bar claims a composition — "38% of cluster 3 is proximal" —
        so every cell in a bar has to land in exactly one box or the bar lies
        about the cluster. Two consequences:

        - Cells with no value for `column_b` are counted into an explicit
          '(none)' bucket rather than dropped. Territories leave a cell blank
          when it has no coordinate, which is common and not an error.
        - Categories no cell uses are kept. A cluster emptied by a filter still
          belongs on the axis; dropping it renumbers everything to its right.

        Returns counts as rows of `a_categories` by columns of `b_categories`,
        plus each column's scanpy colors when the dataset carries them.
        """
        if column_a == column_b:
            raise ValueError(
                "Splitting a column by the same column gives one block per bar. "
                "Choose two different columns."
            )
        for name in (column_a, column_b):
            if name not in self.adata.obs.columns:
                raise KeyError(f"Column '{name}' not found in .obs")

        def categories(name: str) -> tuple[pd.Series, list[str], list[str] | None]:
            series = self.adata.obs[name]
            if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
                raise ValueError(
                    f"'{name}' is continuous. Bin it into a categorical column first — "
                    "grouping it here would invent categories nobody chose."
                )
            colors = self._get_category_colors(name)
            if pd.api.types.is_categorical_dtype(series.dtype):
                levels = [str(c) for c in series.cat.categories]
            else:
                # A plain string/bool column has no stored order, so use a
                # stable one rather than whatever order the rows happen to be in.
                levels = sorted({str(v) for v in series.dropna().unique()})
            return series, levels, colors

        series_a, cats_a, colors_a = categories(column_a)
        series_b, cats_b, colors_b = categories(column_b)
        indices = self._validate_cell_indices(active_cell_indices)
        if indices is not None:
            series_a = series_a.iloc[indices]
            series_b = series_b.iloc[indices]

        MISSING = "(none)"
        a_vals = series_a.astype("object").where(series_a.notna(), MISSING).astype(str)
        b_vals = series_b.astype("object").where(series_b.notna(), MISSING).astype(str)
        if a_vals.eq(MISSING).any():
            cats_a = [*cats_a, MISSING]
            colors_a = [*colors_a, None] if colors_a is not None else None
        if b_vals.eq(MISSING).any():
            cats_b = [*cats_b, MISSING]
            colors_b = [*colors_b, None] if colors_b is not None else None

        index_b = {name: j for j, name in enumerate(cats_b)}
        counts = [[0] * len(cats_b) for _ in cats_a]
        rows = {name: i for i, name in enumerate(cats_a)}
        for a, b in zip(a_vals, b_vals):
            i, j = rows.get(a), index_b.get(b)
            if i is not None and j is not None:
                counts[i][j] += 1

        return {
            "a": column_a,
            "b": column_b,
            "a_categories": cats_a,
            "b_categories": cats_b,
            "a_colors": colors_a,
            "b_colors": colors_b,
            "counts": counts,
            "n_cells": int(len(series_a)),
            "n_total": int(self.adata.n_obs),
        }

    def get_obs_column_summary(self, name: str) -> dict[str, Any]:
        """Get summary statistics for a cell metadata column.

        For categorical columns: returns categories with cell counts.
        For numeric columns: returns min, max, mean.

        Args:
            name: Name of the column in .obs

        Returns:
            Dictionary containing:
            - name: The column name
            - dtype: Data type ('category', 'numeric', or 'string')
            - For categorical: categories (list of {value, count} objects)
            - For numeric: min, max, mean

        Raises:
            KeyError: If column name not found in .obs
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Column '{name}' not found. Available: {list(self.adata.obs.columns)}")

        series = self.adata.obs[name]
        dtype = series.dtype

        result: dict[str, Any] = {
            "name": name,
        }

        if pd.api.types.is_categorical_dtype(dtype):
            result["dtype"] = "category"
            value_counts = series.value_counts()
            colors = self._get_category_colors(name)
            color_by_value: dict[str, str] = {}
            if colors is not None:
                for cat, c in zip(series.cat.categories.tolist(), colors):
                    color_by_value[str(cat)] = c
            entries: list[dict[str, Any]] = []
            for val, count in value_counts.items():
                entry: dict[str, Any] = {"value": str(val), "count": int(count)}
                hex_color = color_by_value.get(str(val))
                if hex_color is not None:
                    entry["color"] = hex_color
                entries.append(entry)
            result["categories"] = entries
        elif pd.api.types.is_numeric_dtype(dtype):
            result["dtype"] = "numeric"
            result["min"] = float(series.min()) if not pd.isna(series.min()) else None
            result["max"] = float(series.max()) if not pd.isna(series.max()) else None
            result["mean"] = float(series.mean()) if not pd.isna(series.mean()) else None
        else:
            result["dtype"] = "string"
            # For string columns, get unique values with counts
            value_counts = series.value_counts()
            result["categories"] = [
                {"value": str(val), "count": int(count)}
                for val, count in value_counts.head(50).items()  # Limit to 50 for strings
            ]

        return result

    def get_all_obs_summaries(self) -> list[dict[str, Any]]:
        """Get summary statistics for all cell metadata columns.

        Returns:
            List of summary dictionaries for each obs column.
        """
        summaries = []
        for col in self.adata.obs.columns:
            try:
                summary = self.get_obs_column_summary(col)
                summaries.append(summary)
            except Exception:
                # Skip columns that fail
                pass
        return summaries

    # =========================================================================
    # User annotation methods
    # =========================================================================

    def create_annotation(self, name: str, default_value: str = "unassigned") -> dict[str, Any]:
        """Create a new categorical annotation column.

        Args:
            name: Name of the new annotation column
            default_value: Default value for all cells

        Returns:
            Dictionary with the new column summary

        Raises:
            ValueError: If column already exists
        """
        if name in self.adata.obs.columns:
            raise ValueError(f"Annotation '{name}' already exists")

        # Create categorical column with default value
        self.adata.obs[name] = pd.Categorical(
            [default_value] * self.n_cells,
            categories=[default_value]
        )

        self._log_action('create_annotation',
                         {'name': name, 'default_value': default_value},
                         {'name': name, 'n_cells': self.n_cells})
        return self.get_obs_column_summary(name)

    def add_label_to_annotation(self, annotation: str, label: str) -> dict[str, Any]:
        """Add a new label/category to an existing annotation column.

        Args:
            annotation: Name of the annotation column
            label: New label to add

        Returns:
            Updated column summary

        Raises:
            KeyError: If annotation doesn't exist
        """
        if annotation not in self.adata.obs.columns:
            raise KeyError(f"Annotation '{annotation}' not found")

        series = self.adata.obs[annotation]
        if not pd.api.types.is_categorical_dtype(series):
            raise ValueError(f"Annotation '{annotation}' is not categorical")

        # Add new category if it doesn't exist
        if label not in series.cat.categories:
            self.adata.obs[annotation] = series.cat.add_categories([label])

        return self.get_obs_column_summary(annotation)

    def label_cells(
        self, annotation: str, label: str, cell_indices: list[int]
    ) -> dict[str, Any]:
        """Assign a label to specific cells in an annotation column.

        Args:
            annotation: Name of the annotation column
            label: Label to assign
            cell_indices: List of cell indices to label

        Returns:
            Updated column summary

        Raises:
            KeyError: If annotation doesn't exist
        """
        if annotation not in self.adata.obs.columns:
            raise KeyError(f"Annotation '{annotation}' not found")

        series = self.adata.obs[annotation]
        if not pd.api.types.is_categorical_dtype(series):
            raise ValueError(f"Annotation '{annotation}' is not categorical")

        # Add label as category if needed
        if label not in series.cat.categories:
            self.adata.obs[annotation] = series.cat.add_categories([label])

        # Assign label to specified cells
        self.adata.obs.iloc[cell_indices, self.adata.obs.columns.get_loc(annotation)] = label

        # The selection is the whole content of this step — without it, "labelled
        # 412 cells 'cortex'" is unreproducible prose.
        self._log_action('label_cells',
                         {'annotation': annotation, 'label': label},
                         {'n_cells_labelled': len(cell_indices)},
                         subset=cell_indices)
        return self.get_obs_column_summary(annotation)

    def delete_obs_column(self, column: str) -> dict[str, Any]:
        """Drop an .obs column, its scanpy colour list, and — if it was a
        named subset's membership column — the subset's registry entry, since
        a subset with no column can never be activated again.

        Raises:
            KeyError: If the column doesn't exist
        """
        if column not in self.adata.obs.columns:
            raise KeyError(f"Column '{column}' not found in .obs")

        self.adata.obs.drop(columns=[column], inplace=True)
        dropped_colors = self.adata.uns.pop(f'{column}_colors', None) is not None
        subset_removed = None
        registry = self._subset_registry()
        for name, entry in registry.items():
            if entry.get('obs_key', SUBSET_OBS_PREFIX + name) == column:
                subset_removed = name
                break
        if subset_removed is not None:
            registry.pop(subset_removed)
            self.adata.uns[CELL_SUBSETS_UNS] = registry
        result = {'column': column, 'dropped_colors': dropped_colors,
                  'subset_removed': subset_removed}
        self._log_action('delete_obs_column', {'column': column}, result)
        return result

    def delete_annotation(self, name: str) -> None:
        """Older name for delete_obs_column; the /annotations route still calls it."""
        self.delete_obs_column(name)

    def rename_obs_label(self, column: str, old_label: str, new_label: str) -> dict[str, Any]:
        """Rename a single category value in a categorical or string .obs column.

        Args:
            column: Name of the .obs column.
            old_label: Existing category value (matched against str(category) for
                categorical columns, exact match for string columns).
            new_label: Replacement label. Must be non-empty after trimming.

        Returns:
            Dict with column, old_label, new_label, n_cells_renamed.

        Raises:
            KeyError: If column not found.
            ValueError: If column is not categorical/string, old_label not found,
                new_label is empty, or new_label collides with an existing label.
        """
        new_label = (new_label or "").strip()
        if not new_label:
            raise ValueError("new_label cannot be empty")
        if column not in self.adata.obs.columns:
            raise KeyError(f"Column '{column}' not found")

        series = self.adata.obs[column]

        if pd.api.types.is_categorical_dtype(series.dtype):
            cats = series.cat.categories.tolist()
            cat_strs = [str(c) for c in cats]
            if old_label not in cat_strs:
                raise ValueError(f"Label '{old_label}' not found in column '{column}'")
            if new_label != old_label and new_label in cat_strs:
                raise ValueError(
                    f"A label '{new_label}' already exists in column '{column}'. "
                    "Use merge_obs_labels to combine them instead."
                )
            old_cat = cats[cat_strs.index(old_label)]
            n_cells = int((series == old_cat).sum())
            new_cats = [new_label if str(c) == old_label else str(c) for c in cats]
            self.adata.obs[column] = series.cat.rename_categories(new_cats)
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            mask = series.astype(str) == old_label
            n_cells = int(mask.sum())
            if n_cells == 0:
                raise ValueError(f"Label '{old_label}' not found in column '{column}'")
            existing = set(series.astype(str).unique())
            if new_label != old_label and new_label in existing:
                raise ValueError(
                    f"A label '{new_label}' already exists in column '{column}'. "
                    "Use merge_obs_labels to combine them instead."
                )
            new_series = series.astype(str).where(~mask, new_label)
            self.adata.obs[column] = new_series
        else:
            raise ValueError(
                f"Column '{column}' is not categorical or string (dtype={series.dtype})"
            )

        result = {
            "column": column,
            "old_label": old_label,
            "new_label": new_label,
            "n_cells_renamed": n_cells,
        }
        self._log_action('rename_obs_label', {
            'column': column, 'old_label': old_label, 'new_label': new_label,
        }, result)
        return result

    def merge_obs_labels(
        self, column: str, labels: list[str], new_label: str
    ) -> dict[str, Any]:
        """Merge two or more category values into a single new label.

        For categorical columns, all rows whose stringified value is in `labels`
        are reassigned to `new_label`, then unused categories are dropped. If
        `new_label` matches an existing category that is not in `labels`, the
        merge folds into it. For string columns, a vectorized replace is used.

        Args:
            column: Name of the .obs column.
            labels: List (length >= 2) of existing labels to merge.
            new_label: Target label. May reuse one of `labels` or be a fresh name.

        Returns:
            Dict with column, merged_labels, new_label, n_cells_merged.

        Raises:
            KeyError: If column not found.
            ValueError: On invalid args, dtype, or unknown labels.
        """
        new_label = (new_label or "").strip()
        if not new_label:
            raise ValueError("new_label cannot be empty")
        if not isinstance(labels, list) or len(labels) < 2:
            raise ValueError("Provide at least 2 labels to merge")
        if len(set(labels)) != len(labels):
            raise ValueError("Labels list contains duplicates")
        if column not in self.adata.obs.columns:
            raise KeyError(f"Column '{column}' not found")

        series = self.adata.obs[column]

        if pd.api.types.is_categorical_dtype(series.dtype):
            cats = series.cat.categories.tolist()
            cat_strs = [str(c) for c in cats]
            missing = [l for l in labels if l not in cat_strs]
            if missing:
                raise ValueError(
                    f"Labels not found in column '{column}': {missing}"
                )
            label_set = set(labels)
            old_cats = [cats[cat_strs.index(l)] for l in labels]
            n_cells = int(series.isin(old_cats).sum())
            ordered = bool(getattr(series.cat, 'ordered', False))
            # Build new category list: keep non-merged categories as-is (stringified),
            # insert new_label in place of the first merged category, drop the rest.
            new_cats: list[str] = []
            inserted = False
            for c, cs in zip(cats, cat_strs):
                if cs in label_set:
                    if not inserted:
                        new_cats.append(new_label)
                        inserted = True
                    # else: skip — folded into new_label
                else:
                    if cs == new_label and not inserted:
                        # new_label is an existing non-merged category; merge folds
                        # into it without duplicating.
                        new_cats.append(new_label)
                        inserted = True
                    elif cs == new_label:
                        # Already inserted; avoid duplicate.
                        continue
                    else:
                        new_cats.append(cs)
            if not inserted:
                new_cats.append(new_label)
            # Snapshot existing per-category colors BEFORE mutating obs so we can
            # rebuild adata.uns[f'{column}_colors'] aligned to the new categories.
            old_colors_list = self._get_category_colors(column)
            old_color_by_cs = (
                dict(zip(cat_strs, old_colors_list)) if old_colors_list else None
            )
            # Map original values to stringified, remap merged ones to new_label,
            # then build a fresh Categorical with the new category set.
            current_str = series.astype(str)
            remapped = current_str.where(~current_str.isin(label_set), new_label)
            # Categories must be unique; new_cats already deduped above.
            self.adata.obs[column] = pd.Categorical(
                remapped, categories=new_cats, ordered=ordered
            )
            # Update uns[f'{column}_colors'] to stay aligned to new categories.
            # The merged label inherits the color of an existing fold-into target
            # if applicable, otherwise the color of the first merged label.
            if old_color_by_cs is not None:
                if new_label in old_color_by_cs and new_label not in label_set:
                    new_label_color = old_color_by_cs[new_label]
                else:
                    new_label_color = old_color_by_cs.get(labels[0], "#888888")
                self.adata.uns[f"{column}_colors"] = [
                    new_label_color if cs == new_label else old_color_by_cs.get(cs, "#888888")
                    for cs in new_cats
                ]
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            current_str = series.astype(str)
            mask = current_str.isin(labels)
            n_cells = int(mask.sum())
            if n_cells == 0:
                raise ValueError(
                    f"None of the labels were found in column '{column}'"
                )
            self.adata.obs[column] = current_str.where(~mask, new_label)
        else:
            raise ValueError(
                f"Column '{column}' is not categorical or string (dtype={series.dtype})"
            )

        result = {
            "column": column,
            "merged_labels": labels,
            "new_label": new_label,
            "n_cells_merged": n_cells,
        }
        self._log_action('merge_obs_labels', {
            'column': column, 'labels': list(labels), 'new_label': new_label,
        }, result)
        return result

    def transfer_obs_labels(
        self,
        target_column: str,
        source_column: str,
        out_column: str,
        rename_mode: str = "replace",
        sep: str = ".",
        prefix: str = "",
        unassigned_values: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fold labels from a (partial) source column into a parent target column.

        For every cell that carries a "real" label in ``source_column`` (a value
        that is neither NaN nor one of ``unassigned_values``), the result takes
        that source label -- optionally renamed -- so it overrides the parent
        label. Cells the source did not touch keep their ``target_column`` value.

        This is the common "subcluster a masked subset, then refine the parent
        annotation" workflow: subclustering leaves masked-out cells as
        "unassigned", and those cells should retain their original parent label.

        Args:
            target_column: Parent categorical/string .obs column to refine.
            source_column: Column holding the new (e.g. subcluster) labels, with
                unassigned cells marked NaN or one of ``unassigned_values``.
            out_column: Where to write the result. May equal ``target_column`` to
                refine in place, or a name not already in .obs (creates a new one).
            rename_mode: How to name incoming source labels:
                - "replace": use the source label verbatim.
                - "parent_prefix": f"{parent_label}{sep}{source_label}" -- keeps
                  provenance and avoids collisions from bare "0"/"1" labels.
                - "custom_prefix": f"{prefix}{source_label}".
            sep: Separator for "parent_prefix" mode.
            prefix: Prefix string for "custom_prefix" mode.
            unassigned_values: Source values treated as "no new label" (cell keeps
                its parent label). Defaults to ["unassigned"]. NaN is always so.

        Returns:
            Dict with out_column, n_overridden, n_kept, n_new_labels, categories.

        Raises:
            KeyError: If target or source column not found.
            ValueError: On bad rename_mode, empty out_column, or an out_column
                that collides with an unrelated existing column.
        """
        if rename_mode not in ("replace", "parent_prefix", "custom_prefix"):
            raise ValueError(f"Unknown rename_mode '{rename_mode}'")
        if target_column not in self.adata.obs.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        if source_column not in self.adata.obs.columns:
            raise KeyError(f"Source column '{source_column}' not found")
        out_column = (out_column or "").strip()
        if not out_column:
            raise ValueError("out_column cannot be empty")
        if out_column in self.adata.obs.columns and out_column != target_column:
            raise ValueError(
                f"Column '{out_column}' already exists. Choose a new name or set "
                f"it to the target column '{target_column}' to refine in place."
            )

        if unassigned_values is None:
            unassigned_values = ["unassigned"]
        unassigned_set = {str(v) for v in unassigned_values}

        target = self.adata.obs[target_column]
        source = self.adata.obs[source_column]
        target_str = target.astype(str)
        source_str = source.astype(str)

        # A source value overrides the parent when it is not NaN and not a sentinel.
        source_isna = source.isna().to_numpy()
        has_new = (~source_isna) & (~source_str.isin(unassigned_set).to_numpy())

        parent_str = target_str.to_numpy().astype(object)
        src = source_str.to_numpy().astype(object)
        if rename_mode == "replace":
            renamed = src
        elif rename_mode == "parent_prefix":
            renamed = np.array(
                [f"{p}{sep}{s}" for p, s in zip(parent_str, src)], dtype=object
            )
        else:  # custom_prefix
            renamed = np.array([f"{prefix}{s}" for s in src], dtype=object)
        result = np.where(has_new, renamed, parent_str).astype(object)

        # Category ordering + colors: surviving parent labels keep their original
        # order/colors; new labels follow in source order, colored from the source.
        target_is_cat = pd.api.types.is_categorical_dtype(target.dtype)
        source_is_cat = pd.api.types.is_categorical_dtype(source.dtype)
        target_colors = self._get_category_colors(target_column)
        source_colors = self._get_category_colors(source_column)
        target_color_by = (
            dict(zip([str(c) for c in target.cat.categories], target_colors))
            if (target_colors and target_is_cat) else {}
        )
        source_color_by = (
            dict(zip([str(c) for c in source.cat.categories], source_colors))
            if (source_colors and source_is_cat) else {}
        )

        result_list = result.tolist()
        result_set = set(result_list)
        if target_is_cat:
            parent_order = [str(c) for c in target.cat.categories]
        else:
            parent_order = list(dict.fromkeys(parent_str.tolist()))
        ordered_cats = [c for c in parent_order if c in result_set]

        # Map each new label back to the source category it came from (for color).
        new_label_src: dict[str, str] = {}
        for keep, r, s in zip(has_new.tolist(), result_list, src.tolist()):
            if keep and r not in ordered_cats and r not in new_label_src:
                new_label_src[r] = s
        if source_is_cat:
            src_order = [str(c) for c in source.cat.categories]
        else:
            src_order = list(dict.fromkeys(src.tolist()))
        new_labels = sorted(
            new_label_src.keys(),
            key=lambda lbl: (
                src_order.index(new_label_src[lbl])
                if new_label_src[lbl] in src_order else len(src_order),
                lbl,
            ),
        )
        ordered_cats = ordered_cats + [c for c in new_labels if c not in ordered_cats]

        self.adata.obs[out_column] = pd.Categorical(result, categories=ordered_cats)

        if target_color_by or source_color_by:
            colors_out: list[str] = []
            for c in ordered_cats:
                if c in target_color_by:
                    colors_out.append(target_color_by[c])
                elif c in new_label_src and new_label_src[c] in source_color_by:
                    colors_out.append(source_color_by[new_label_src[c]])
                else:
                    colors_out.append("#888888")
            self.adata.uns[f"{out_column}_colors"] = colors_out
        elif f"{out_column}_colors" in self.adata.uns:
            # Stale colors from a prior column of this name would mis-align.
            del self.adata.uns[f"{out_column}_colors"]

        n_overridden = int(has_new.sum())
        result_dict = {
            "out_column": out_column,
            "target_column": target_column,
            "source_column": source_column,
            "n_overridden": n_overridden,
            "n_kept": int(self.n_cells - n_overridden),
            "n_new_labels": len(new_labels),
            "categories": ordered_cats,
        }
        self._log_action(
            "transfer_obs_labels",
            {
                "target_column": target_column,
                "source_column": source_column,
                "out_column": out_column,
                "rename_mode": rename_mode,
            },
            result_dict,
        )
        return result_dict

    def get_user_annotations(self) -> list[str]:
        """Get list of user-created annotation columns.

        For now, returns all categorical columns. In the future,
        could track which columns were created by users.

        Returns:
            List of annotation column names
        """
        return [
            col for col in self.adata.obs.columns
            if pd.api.types.is_categorical_dtype(self.adata.obs[col])
        ]

    def export_annotations(self, columns: list[str] | None = None) -> str:
        """Export cell annotations as TSV string.

        Args:
            columns: List of column names to export. If None, exports all.

        Returns:
            TSV-formatted string with cell indices and annotation values
        """
        if columns is None:
            df = self.adata.obs.copy()
        else:
            # Validate columns exist
            missing = [c for c in columns if c not in self.adata.obs.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            df = self.adata.obs[columns].copy()

        return df.to_csv(sep="\t")

    # =========================================================================
    # Differential expression analysis
    # =========================================================================

    def run_diffexp(
        self,
        group1_indices: list[int],
        group2_indices: list[int],
        top_n: int = 10,
        method: str = "wilcoxon",
        corr_method: str = "benjamini-hochberg",
        min_fold_change: float | None = None,
        min_in_group_fraction: float | None = None,
        max_out_group_fraction: float | None = None,
        max_pval_adj: float | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run differential expression analysis between two cell groups.

        Uses scanpy's rank_genes_groups with Wilcoxon rank-sum test.

        Args:
            group1_indices: Cell indices for group 1
            group2_indices: Cell indices for group 2
            top_n: Number of top genes to return for each direction
            method: Statistical method for rank_genes_groups
            corr_method: P-value correction method
            min_fold_change: Minimum fold change for filtering
            min_in_group_fraction: Minimum fraction of cells in group expressing gene
            max_out_group_fraction: Maximum fraction of cells outside group expressing gene
            max_pval_adj: Maximum adjusted p-value for filtering
            gene_subset: Gene filtering specification (str column name, list of genes, or dict spec)

        Returns:
            Dictionary containing:
            - positive: Top N genes upregulated in group1
            - negative: Top N genes upregulated in group2
            - group1_count: Number of cells in group 1
            - group2_count: Number of cells in group 2

        Raises:
            ValueError: If indices are invalid or groups too small
        """
        # Validate indices
        max_idx = self.n_cells - 1
        for idx in group1_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid cell index: {idx}")
        for idx in group2_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid cell index: {idx}")

        # Check for overlap
        set1 = set(group1_indices)
        set2 = set(group2_indices)
        overlap = set1 & set2
        if overlap:
            raise ValueError(f"Groups have {len(overlap)} overlapping cells")

        # Resolve gene subset
        if gene_subset is not None or self._visible_gene_mask is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            work_adata = self.adata[:, gene_mask].copy()
        else:
            work_adata = self.adata
            subset_type = 'all'

        result = compute_diffexp(
            adata=work_adata,
            group1_indices=group1_indices,
            group2_indices=group2_indices,
            top_n=top_n,
            method=method,
            corr_method=corr_method,
            min_fold_change=min_fold_change,
            min_in_group_fraction=min_in_group_fraction,
            max_out_group_fraction=max_out_group_fraction,
            max_pval_adj=max_pval_adj,
        )
        result['gene_subset_type'] = subset_type
        result['n_genes_tested'] = work_adata.n_vars
        self._log_action('diffexp', {
            'n_group1': len(group1_indices), 'n_group2': len(group2_indices),
            'top_n': top_n, 'method': method, 'corr_method': corr_method,
            'min_fold_change': min_fold_change,
            'min_in_group_fraction': min_in_group_fraction,
            'max_out_group_fraction': max_out_group_fraction,
            'max_pval_adj': max_pval_adj, 'gene_subset': gene_subset,
        }, {
            'n_genes_tested': work_adata.n_vars,
            'gene_subset_type': subset_type,
            'n_upregulated': len(result.get('upregulated') or []),
            'n_downregulated': len(result.get('downregulated') or []),
        })
        return result

    # =========================================================================
    # Drawn lines / trajectory methods
    # =========================================================================

    def _restore_lines(self) -> list[dict[str, Any]]:
        """Read drawn shapes out of .uns, or start with none.

        Tolerates the pre-2026-09 export shape (``embedding`` rather than
        ``embeddingName``, no styling) by filling defaults, and drops anything
        without an embedding or points rather than let one bad entry hide the
        rest. A corrupt blob must not block the load.
        """
        raw = self.adata.uns.get(LINES_UNS)
        if not raw:
            return []
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            return []
        out: list[dict[str, Any]] = []
        for i, line in enumerate(stored if isinstance(stored, list) else []):
            if not isinstance(line, dict) or not line.get('points'):
                continue
            emb = line.get('embeddingName') or line.get('embedding')
            if not emb:
                continue
            fixed = {**_LINE_DEFAULTS,
                     **{k: v for k, v in line.items() if k not in ('embedding', 'smoothed_points')},
                     'embeddingName': emb}
            if line.get('smoothed_points') and not fixed.get('smoothedPoints'):
                fixed['smoothedPoints'] = line['smoothed_points']
            fixed.setdefault('id', f'line_restored_{i}')
            fixed.setdefault('name', f'Line {i + 1}')
            out.append(fixed)
        return out

    def set_lines(self, lines: list[dict[str, Any]]) -> None:
        """Store the browser's drawn shapes, in memory and in .uns."""
        self._drawn_lines = [dict(line) for line in lines]
        self.adata.uns[LINES_UNS] = json.dumps(self._drawn_lines)

    def get_lines(self) -> list[dict[str, Any]]:
        """The stored drawn shapes."""
        return self._drawn_lines

    def _lines_on(self, embeddings: set[str]) -> list[dict[str, Any]]:
        return [line for line in self._drawn_lines if line.get('embeddingName') in embeddings]

    def _territories_on(self, embeddings: set[str]) -> list[str]:
        return [name for name, spec in self.get_territories().items()
                if isinstance(spec, Mapping) and spec.get('embedding') in embeddings]

    def _project_cells_onto_line(
        self,
        line_points: list[list[float]],
        embedding_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project all cells onto a line and compute position/distance.

        Uses the smoothed line if available, otherwise raw points.
        For each cell, finds the closest point on the polyline and computes:
        - Position along line (0 = start, 1 = end, normalized by arc length)
        - Perpendicular distance to the line

        Args:
            line_points: List of [x, y] points defining the line
            embedding_coords: Embedding coordinates for all cells (n_cells x 2)

        Returns:
            Tuple of (positions, distances) arrays, each of shape (n_cells,)
        """
        if len(line_points) < 2:
            return np.zeros(len(embedding_coords)), np.full(len(embedding_coords), np.nan)

        line_pts = np.array(line_points)
        n_cells = len(embedding_coords)

        # Compute cumulative arc length along line
        segment_lengths = np.sqrt(np.sum(np.diff(line_pts, axis=0) ** 2, axis=1))
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        positions = np.zeros(n_cells)
        distances = np.zeros(n_cells)

        for i, cell_pt in enumerate(embedding_coords):
            best_dist = np.inf
            best_pos = 0.0

            # Check each line segment
            for j in range(len(line_pts) - 1):
                p1 = line_pts[j]
                p2 = line_pts[j + 1]
                seg_vec = p2 - p1
                seg_len_sq = np.dot(seg_vec, seg_vec)

                if seg_len_sq < 1e-12:
                    # Degenerate segment
                    t = 0.0
                else:
                    # Project point onto segment
                    t = max(0, min(1, np.dot(cell_pt - p1, seg_vec) / seg_len_sq))

                # Closest point on this segment
                closest = p1 + t * seg_vec
                dist = np.sqrt(np.sum((cell_pt - closest) ** 2))

                if dist < best_dist:
                    best_dist = dist
                    # Position along total line
                    if total_length > 1e-12:
                        best_pos = (cumulative_lengths[j] + t * segment_lengths[j]) / total_length
                    else:
                        best_pos = 0.0

            positions[i] = best_pos
            distances[i] = best_dist

        return positions, distances

    def compute_line_projections(self) -> dict[str, dict[str, np.ndarray]]:
        """Compute cell projections for all stored lines.

        Returns:
            Dictionary mapping line name to {positions, distances} arrays
        """
        projections = {}

        for line in self._drawn_lines:
            embedding_name = line.get('embeddingName', '')
            line_name = line.get('name', 'unnamed')

            # Get the appropriate embedding coordinates
            if embedding_name not in self.adata.obsm:
                continue

            coords = self._line_view_coords(line)

            # Use smoothed points if available, otherwise raw points
            line_points = line.get('smoothedPoints') or line.get('points', [])
            if not line_points:
                continue

            positions, distances = self._project_cells_onto_line(line_points, coords)
            projections[line_name] = {
                'positions': positions,
                'distances': distances,
            }

        return projections

    def prepare_export_with_lines(self) -> anndata.AnnData:
        """Prepare AnnData for export, including line data.

        Stores line metadata as JSON string in .uns['xcell_lines_json'] and
        cell projections in .obsm['X_{line_name}_projection'].

        The JSON contains an array of line objects with:
        - name: Line name
        - embedding: Embedding name the line was drawn on
        - points: Array of [x, y] coordinates
        - smoothed_points: Array of smoothed [x, y] coordinates (if exists)

        Also stores the analysis record as a JSON string in
        .uns['xcell_analysis_record'], so an exported dataset carries its own
        provenance and re-opening it continues the history.

        Returns:
            Copy of adata with lines, projections and the analysis record added
        """
        # Work with a copy to avoid modifying the live data
        adata_export = self.adata.copy()

        # h5ad refuses to write a frame whose index.name is also a column with
        # different values. The name is cosmetic, so drop it rather than let a
        # whole export fail; the column keeps its data either way.
        for frame in (adata_export.var, adata_export.obs):
            if frame.index.name in frame.columns:
                frame.index.name = None

        adata_export.uns[self.ANALYSIS_RECORD_UNS_KEY] = self._serialize_record()

        if not self._drawn_lines:
            return adata_export

        # The full shapes, as the browser holds them, so a re-opened export
        # shows them again (the copy carries the key already; this keeps the
        # projections below in step with what is stored).
        adata_export.uns[LINES_UNS] = json.dumps(self._drawn_lines)

        # Compute and store projections
        projections = self.compute_line_projections()

        for line_name, proj_data in projections.items():
            # Sanitize line name for use as key (replace spaces, special chars)
            safe_name = line_name.replace(' ', '_').replace('-', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')

            key = f'X_{safe_name}_projection'

            # Store as n_cells x 2 matrix: [position, distance]
            proj_matrix = np.column_stack([
                proj_data['positions'],
                proj_data['distances'],
            ])
            adata_export.obsm[key] = proj_matrix

        return adata_export

    def _run_spline_association(
        self,
        test_values: np.ndarray,
        cell_indices: np.ndarray,
        gene_mask: np.ndarray,
        n_spline_knots: int = 5,
        fdr_threshold: float = 0.05,
        top_n: int | None = None,
        cluster_genes: bool = False,
    ) -> dict[str, Any]:
        """Core spline regression engine for line association analysis.

        Fits cubic B-spline models for each gene against the provided test
        values and tests significance via F-test (spline model vs intercept-only).

        This is a reusable helper called by both test_line_association (single
        line) and test_multi_line_association (pooled multi-line).

        Args:
            test_values: 1-D array of test variable values (e.g. position along
                        line), one per cell. Should be in [0, 1].
            cell_indices: 1-D integer array of cell indices into self.adata.
            gene_mask: Boolean array of length n_genes selecting which genes
                      to test.
            n_spline_knots: Number of interior knots for the B-spline basis.
            fdr_threshold: FDR threshold for significance.
            top_n: Optional cap on how many genes to return per direction
                (per module when clustering). None returns every
                significant gene; a cap keeps the highest-scoring ones.
            cluster_genes: If True, cluster significant genes by expression
                profile shape and return modules. If False, skip clustering
                and return only positive/negative lists (modules will be empty).

        Returns:
            Dict with keys: positive, negative, modules, n_cells,
            n_significant, n_positive, n_negative, n_modules, fdr_threshold,
            diagnostics. Callers should add line_name, test_variable, etc.
        """
        from scipy.interpolate import BSpline
        from scipy.stats import f as f_dist
        from statsmodels.stats.multitest import multipletests

        n_cells_used = len(cell_indices)

        # Get expression matrix for selected cells and genes
        # (trusts that user has already preprocessed: normalize, log1p, etc.)
        X = self.adata.X[cell_indices][:, gene_mask]

        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        n_genes = X.shape[1]
        gene_names = self.adata.var_names[gene_mask].tolist()

        # Build B-spline basis matrix
        # Use quantile-based knots for even coverage of cells
        pos = test_values

        # Create knot vector for cubic B-spline
        # Interior knots at quantiles, plus boundary knots
        degree = 3
        n_interior = n_spline_knots
        interior_knots = np.quantile(pos, np.linspace(0, 1, n_interior + 2)[1:-1])

        # Full knot vector: degree+1 copies at boundaries, interior knots in between
        knots = np.concatenate([
            np.repeat(0.0, degree + 1),
            interior_knots,
            np.repeat(1.0, degree + 1),
        ])

        # Number of basis functions
        n_basis = len(knots) - degree - 1

        # Evaluate B-spline basis at all positions
        B = np.zeros((n_cells_used, n_basis))
        for i in range(n_basis):
            # Create coefficient vector with 1 at position i
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            B[:, i] = spline(pos)

        # Design matrix: B-spline basis only (no separate intercept).
        # B-spline basis functions satisfy the partition of unity (sum to 1),
        # so the constant function is already in their span.
        design = B
        k = n_basis - 1  # Extra df beyond intercept (one basis function spans the constant)

        # Solve OLS for all genes at once: beta = (X'X)^{-1} X' Y
        # where Y is expression matrix (n_cells x n_genes)
        try:
            XtX = design.T @ design
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ design.T @ X  # n_basis x n_genes
        except np.linalg.LinAlgError:
            raise ValueError("Singular design matrix. Try fewer spline knots.")

        # Compute residuals and RSS
        predicted = design @ beta
        residuals = X - predicted
        rss_full = np.sum(residuals ** 2, axis=0)  # RSS per gene

        # Null model: intercept only
        gene_means = X.mean(axis=0)
        rss_null = np.sum((X - gene_means) ** 2, axis=0)

        # F-test: F = [(RSS_null - RSS_full) / k] / [RSS_full / (n - k - 1)]
        df1 = k  # Numerator df (spline coefficients)
        df2 = n_cells_used - k - 1  # Denominator df

        # Avoid division by zero
        rss_full_safe = np.maximum(rss_full, 1e-10)

        f_stat = ((rss_null - rss_full) / df1) / (rss_full_safe / df2)
        f_stat = np.maximum(f_stat, 0)  # F-stat can't be negative

        # P-values from F distribution
        p_values = 1 - f_dist.cdf(f_stat, df1, df2)

        # FDR correction
        _, fdr, _, _ = multipletests(p_values, method='fdr_bh')

        # Compute effect size metrics
        # R-squared: variance explained
        r_squared = 1 - rss_full / np.maximum(rss_null, 1e-10)
        r_squared = np.clip(r_squared, 0, 1)

        # Amplitude: range of predicted expression along the line
        amplitude = predicted.max(axis=0) - predicted.min(axis=0)

        # Direction: correlation of predicted with position (for ranking)
        # Positive = expression increases along line, Negative = decreases
        pos_centered = pos - pos.mean()
        pred_centered = predicted - predicted.mean(axis=0)
        direction = np.zeros(n_genes)
        for g in range(n_genes):
            if np.std(pred_centered[:, g]) > 1e-10:
                corr = np.corrcoef(pos_centered, pred_centered[:, g])[0, 1]
                direction[g] = corr if not np.isnan(corr) else 0
            else:
                direction[g] = 0

        # Build results DataFrame
        results = pd.DataFrame({
            'gene': gene_names,
            'f_stat': f_stat,
            'pval': p_values,
            'fdr': fdr,
            'r_squared': r_squared,
            'amplitude': amplitude,
            'direction': direction,
        })

        # Sort by significance (combining FDR and amplitude)
        results['score'] = -np.log10(results['fdr'] + 1e-300) * results['amplitude']

        # Get significant genes
        sig_mask = results['fdr'] < fdr_threshold
        n_significant = sig_mask.sum()

        # Split into positive (increasing) and negative (decreasing) direction
        # (kept for backward compatibility)
        pos_mask = sig_mask & (results['direction'] > 0)
        neg_mask = sig_mask & (results['direction'] < 0)

        # top_n is an optional cap: None returns every significant gene, which
        # is what a user who has already set an FDR threshold usually meant.
        cap = len(results) if top_n is None else max(0, int(top_n))
        cols = ['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']

        positive_genes = (
            results[pos_mask].nlargest(cap, 'score')[cols].to_dict('records')
        )

        negative_genes = (
            results[neg_mask].nlargest(cap, 'score')[cols].to_dict('records')
        )

        # ---- Module-based clustering of ALL significant genes ----
        # Evaluate spline profiles at evenly-spaced positions
        n_profile_points = 50
        profile_positions = np.linspace(0.0, 1.0, n_profile_points)
        profile_design = np.zeros((n_profile_points, n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            profile_design[:, i] = spline(profile_positions)
        profile_design_full = profile_design

        sig_indices = np.where(sig_mask.values)[0]
        modules = []

        if cluster_genes and len(sig_indices) > 0:
            # Predicted profiles for significant genes (n_profile_points x n_sig_genes)
            sig_profiles = profile_design_full @ beta[:, sig_indices]

            # Min-max normalize each gene's profile to [0, 1]
            prof_min = sig_profiles.min(axis=0)
            prof_max = sig_profiles.max(axis=0)
            prof_range = prof_max - prof_min
            prof_range[prof_range < 1e-10] = 1.0  # avoid division by zero
            norm_profiles = (sig_profiles - prof_min) / prof_range  # (n_points, n_sig)

            if len(sig_indices) == 1:
                # Single gene: one module
                cluster_labels = np.array([0])
            else:
                # Hierarchical clustering with correlation distance
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist

                # Transpose so each row is a gene's profile
                profile_matrix = norm_profiles.T  # (n_sig, n_points)

                # Correlation distance; clip to avoid numerical issues
                dists = pdist(profile_matrix, metric='correlation')
                dists = np.clip(dists, 0, 2)

                Z = linkage(dists, method='average')
                # Cut tree: use distance threshold of 0.5 (correlation-based)
                # This gives reasonable module granularity
                cluster_labels = fcluster(Z, t=0.5, criterion='distance') - 1  # 0-indexed

            sig_results = results.iloc[sig_indices].reset_index(drop=True)
            n_modules = int(cluster_labels.max()) + 1

            # Compute per-gene peak positions for all significant genes
            gene_peak_positions = np.argmax(norm_profiles, axis=0) / max(n_profile_points - 1, 1)

            for mod_idx in range(n_modules):
                member_mask = cluster_labels == mod_idx
                member_genes = sig_results[member_mask]
                member_profiles = norm_profiles[:, member_mask]  # (n_points, n_members)
                member_peak_positions = gene_peak_positions[member_mask]

                # Representative profile: mean of normalized profiles in this module
                rep_profile = member_profiles.mean(axis=1)

                # Classify pattern shape
                pattern = self._classify_profile_pattern(rep_profile, profile_positions)

                # Sort genes within module by peak position along the line
                member_genes = member_genes.copy()
                member_genes['peak_position'] = member_peak_positions
                if top_n is not None and len(member_genes) > cap:
                    # Cap by score, not by peak position — truncating the
                    # peak-ordered list would keep one end of the line only.
                    member_genes = member_genes.nlargest(cap, 'score')
                member_genes_sorted = member_genes.sort_values('peak_position')

                # Build gene records with per-gene profiles
                gene_records = []
                for row_idx, (orig_sig_idx, row) in enumerate(member_genes_sorted.iterrows()):
                    # orig_sig_idx is the index into norm_profiles columns
                    # (sig_results was built with reset_index, so index = column position)
                    gene_profile = norm_profiles[:, orig_sig_idx].tolist()
                    gene_records.append({
                        'gene': row['gene'],
                        'f_stat': row['f_stat'],
                        'pval': row['pval'],
                        'fdr': row['fdr'],
                        'r_squared': row['r_squared'],
                        'amplitude': row['amplitude'],
                        'direction': row['direction'],
                        'profile': gene_profile,
                        'peak_position': float(row['peak_position']),
                    })

                modules.append({
                    'module_id': mod_idx,
                    'pattern': pattern,
                    'n_genes': len(gene_records),
                    'representative_profile': rep_profile.tolist(),
                    'profile_positions': profile_positions.tolist(),
                    'genes': gene_records,
                })

            # Sort modules: increasing first, then decreasing, then peak, trough, complex
            pattern_order = {'increasing': 0, 'decreasing': 1, 'peak': 2, 'trough': 3, 'complex': 4}
            modules.sort(key=lambda m: (pattern_order.get(m['pattern'], 5), -m['n_genes']))

        # Compute diagnostic statistics
        n_pval_below_05 = int((p_values < 0.05).sum())
        n_pval_below_01 = int((p_values < 0.01).sum())

        # Check expression matrix properties
        expr_min = float(X.min())
        expr_max = float(X.max())
        expr_mean = float(X.mean())
        n_zero_genes = int((X.sum(axis=0) == 0).sum())

        # Position statistics
        pos_min = float(pos.min())
        pos_max = float(pos.max())
        pos_std = float(pos.std())

        # Full per-gene table (stats for EVERY gene tested) — exposed so the
        # client can download a CSV for GSEA / external analysis. Kept in
        # gene-order (matches adata.var_names[gene_mask]) so the user can
        # re-sort however they like.
        all_genes_records = (
            results[['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']]
            .to_dict('records')
        )

        return {
            'positive': positive_genes,
            'negative': negative_genes,
            'modules': modules,
            'all_genes': all_genes_records,
            'n_cells': n_cells_used,
            'n_significant': int(n_significant),
            'n_positive': int(pos_mask.sum()),
            'n_negative': int(neg_mask.sum()),
            'n_modules': len(modules),
            'fdr_threshold': fdr_threshold,
            'diagnostics': {
                'n_genes_tested': n_genes,
                'n_pval_below_05': n_pval_below_05,
                'n_pval_below_01': n_pval_below_01,
                'position_range': [pos_min, pos_max],
                'position_std': pos_std,
                'expression_range': [expr_min, expr_max],
                'expression_mean': expr_mean,
                'n_zero_genes': n_zero_genes,
                'spline_df': k,
            },
        }

    def prepare_line_association(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int | None = None,
        cluster_genes: bool = False,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare line association computation (cancellable).

        Validates that the named line exists (fail fast), then returns a pair of
        functions: ``compute_fn`` (calls test_line_association, read-only) and
        ``apply_fn`` (no-op since this operation doesn't write to adata).

        Args:
            Same as test_line_association.

        Returns:
            Tuple of (compute_fn, apply_fn)

        Raises:
            ValueError: If line not found
        """
        # Fail fast: validate line exists
        line_found = False
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line_found = True
                break
        if not line_found:
            raise ValueError(f"Line '{line_name}' not found")

        # Resolve the subset now and discard the mask: an unknown column or a
        # gene set that overlaps nothing should be a 400 the modal can show,
        # not a background task that fails a second later.
        self._resolve_gene_mask(gene_subset)

        # Snapshot parameters
        snap_line_name = line_name
        snap_cell_indices = cell_indices
        snap_gene_subset = gene_subset
        snap_test_variable = test_variable
        snap_n_spline_knots = n_spline_knots
        snap_min_cells = min_cells
        snap_fdr_threshold = fdr_threshold
        snap_top_n = top_n
        snap_cluster_genes = cluster_genes

        def compute_fn() -> dict[str, Any]:
            return self.test_line_association(
                line_name=snap_line_name,
                cell_indices=snap_cell_indices,
                gene_subset=snap_gene_subset,
                test_variable=snap_test_variable,
                n_spline_knots=snap_n_spline_knots,
                min_cells=snap_min_cells,
                fdr_threshold=snap_fdr_threshold,
                top_n=snap_top_n,
                cluster_genes=snap_cluster_genes,
            )

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            return result  # Read-only operation, result is already serializable

        return compute_fn, apply_fn

    def test_line_association(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int | None = None,
        cluster_genes: bool = False,
    ) -> dict[str, Any]:
        """Test genes for association with position along or distance from a line.

        Uses cubic B-spline regression to model gene expression as a function
        of a spatial variable derived from the line, then tests whether the
        spline model explains significantly more variance than an intercept-only
        model (F-test).

        Args:
            line_name: Name of the line to test against
            cell_indices: Optional list of cell indices to use. If None, uses
                         all cells (projected based on distance threshold).
            gene_subset: Optional gene filter. Can be a boolean column name (str),
                        a list of gene names, or a dict for combining columns.
            test_variable: 'position' to test against position along the line,
                          'distance' to test against perpendicular distance from
                          the line.
            n_spline_knots: Number of interior knots for the B-spline basis.
                           Total df = n_spline_knots + 2 (for cubic splines).
            min_cells: Minimum number of cells required for testing.
            fdr_threshold: FDR threshold for significance.
            top_n: Optional cap on how many genes to return per direction
                (per module when clustering). None returns every
                significant gene; a cap keeps the highest-scoring ones.

        Returns:
            Dict containing:
            - modules: Gene modules clustered by expression profile shape
            - positive: Genes with expression increasing along variable
            - negative: Genes with expression decreasing along variable
            - n_cells: Number of cells used
            - n_significant: Total significant genes at FDR threshold
            - line_name: The line name used

        Raises:
            ValueError: If line not found or too few cells
        """
        # Find the line
        line = None
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line = l
                break

        if line is None:
            raise ValueError(f"Line '{line_name}' not found")

        embedding_name = line.get('embeddingName', '')
        if embedding_name not in self.adata.obsm:
            raise ValueError(f"Embedding '{embedding_name}' not found")

        # Get line points (smoothed if available)
        line_points = line.get('smoothedPoints') or line.get('points', [])
        if len(line_points) < 2:
            raise ValueError("Line must have at least 2 points")

        # Get embedding coordinates
        coords = self._line_view_coords(line)

        # Project cells onto line
        positions, distances = self._project_cells_onto_line(line_points, coords)

        # Determine which cells to use
        if cell_indices is not None:
            # Use provided cell indices
            cell_mask = np.zeros(self.n_cells, dtype=bool)
            cell_mask[cell_indices] = True
        else:
            # Use all cells (could filter by distance threshold in future)
            cell_mask = np.ones(self.n_cells, dtype=bool)

        # Get positions and distances for selected cells
        selected_indices = np.where(cell_mask)[0]
        selected_positions = positions[cell_mask]
        selected_distances = distances[cell_mask]
        n_cells_used = len(selected_indices)

        if n_cells_used < min_cells:
            raise ValueError(
                f"Too few cells ({n_cells_used}). Need at least {min_cells}."
            )

        # Select the test variable
        if test_variable == 'distance':
            # Normalize distances to [0, 1] for spline fitting
            d_min = selected_distances.min()
            d_max = selected_distances.max()
            if d_max - d_min < 1e-10:
                raise ValueError(
                    "All cells have the same distance from the line. "
                    "Cannot test distance association."
                )
            test_values = (selected_distances - d_min) / (d_max - d_min)
        else:
            test_values = selected_positions

        # Resolve gene subset if provided
        gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(gene_subset)

        # Delegate to the shared spline regression engine
        result = self._run_spline_association(
            test_values=test_values,
            cell_indices=selected_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
            cluster_genes=cluster_genes,
        )

        # Add line-specific metadata
        result['line_name'] = line_name
        result['test_variable'] = test_variable
        result['gene_subset'] = self._gene_subset_summary(subset_type, subset_metadata)

        return result

    def prepare_multi_line_association(
        self,
        lines: list[dict[str, Any]],
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int | None = None,
        cluster_genes: bool = False,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare multi-line association computation (cancellable).

        Validates that all named lines exist (fail fast), then returns a pair of
        functions: ``compute_fn`` (calls test_multi_line_association, read-only)
        and ``apply_fn`` (no-op since this operation doesn't write to adata).

        Args:
            Same as test_multi_line_association.

        Returns:
            Tuple of (compute_fn, apply_fn)

        Raises:
            ValueError: If any line not found
        """
        # Fail fast: validate all lines exist
        line_names_set = {l.get('name') for l in self._drawn_lines}
        for entry in lines:
            if entry['name'] not in line_names_set:
                raise ValueError(f"Line '{entry['name']}' not found")

        # Same reason as the single-line case: a bad subset is a 400, not a
        # task that fails after the modal has already closed.
        self._resolve_gene_mask(gene_subset)

        # Snapshot parameters
        snap_lines = lines
        snap_gene_subset = gene_subset
        snap_test_variable = test_variable
        snap_n_spline_knots = n_spline_knots
        snap_min_cells = min_cells
        snap_fdr_threshold = fdr_threshold
        snap_top_n = top_n
        snap_cluster_genes = cluster_genes

        def compute_fn() -> dict[str, Any]:
            return self.test_multi_line_association(
                lines=snap_lines,
                gene_subset=snap_gene_subset,
                test_variable=snap_test_variable,
                n_spline_knots=snap_n_spline_knots,
                min_cells=snap_min_cells,
                fdr_threshold=snap_fdr_threshold,
                top_n=snap_top_n,
                cluster_genes=snap_cluster_genes,
            )

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            return result  # Read-only operation, result is already serializable

        return compute_fn, apply_fn

    def test_multi_line_association(
        self,
        lines: list[dict[str, Any]],
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int | None = None,
        cluster_genes: bool = False,
    ) -> dict[str, Any]:
        """Test genes for association across multiple lines (pooled analysis).

        Projects cells from each line entry onto their respective line geometry,
        normalizes positions per-line (optionally reversing direction), pools all
        cells, and runs the shared spline regression engine.

        Args:
            lines: List of dicts, each with:
                - name (str): Name of a drawn line in self._drawn_lines
                - cell_indices (list[int]): Cell indices to use for this line
                - reversed (bool): If True, flip positions (1 - pos) for this line
            gene_subset: Optional gene filter (boolean column name, gene list, or dict).
            test_variable: 'position' or 'distance'.
            n_spline_knots: Number of interior knots for the B-spline basis.
            min_cells: Minimum number of pooled cells required.
            fdr_threshold: FDR threshold for significance.
            top_n: Optional cap on how many genes to return per direction
                (per module when clustering). None returns every
                significant gene; a cap keeps the highest-scoring ones.

        Returns:
            Dict with spline association results plus multi-line metadata:
            line_name, test_variable, n_lines, lines_used.

        Raises:
            ValueError: If any line not found, embedding missing, or too few cells.
        """
        all_test_values = []
        all_cell_indices = []
        lines_used = []

        for entry in lines:
            line_name = entry['name']
            entry_cell_indices = entry['cell_indices']
            is_reversed = entry.get('reversed', False)

            # Look up line geometry
            line = None
            for l in self._drawn_lines:
                if l.get('name') == line_name:
                    line = l
                    break
            if line is None:
                raise ValueError(f"Line '{line_name}' not found")

            embedding_name = line.get('embeddingName', '')
            if embedding_name not in self.adata.obsm:
                raise ValueError(f"Embedding '{embedding_name}' not found")

            line_points = line.get('smoothedPoints') or line.get('points', [])
            if len(line_points) < 2:
                raise ValueError(f"Line '{line_name}' must have at least 2 points")

            coords = self._line_view_coords(line)

            # Project all cells onto the line
            positions, distances = self._project_cells_onto_line(line_points, coords)

            # Select this entry's cells
            idx_array = np.array(entry_cell_indices, dtype=int)
            if test_variable == 'distance':
                selected = distances[idx_array]
                d_min = selected.min()
                d_max = selected.max()
                if d_max - d_min < 1e-10:
                    raise ValueError(
                        f"All cells for line '{line_name}' have the same distance. "
                        "Cannot test distance association."
                    )
                vals = (selected - d_min) / (d_max - d_min)
            else:
                vals = positions[idx_array]

            # Reverse direction if requested
            if is_reversed:
                vals = 1.0 - vals

            all_test_values.append(vals)
            all_cell_indices.append(idx_array)
            lines_used.append(line_name)

        # Pool across all lines
        pooled_test_values = np.concatenate(all_test_values)
        pooled_cell_indices = np.concatenate(all_cell_indices)

        if len(pooled_cell_indices) < min_cells:
            raise ValueError(
                f"Too few pooled cells ({len(pooled_cell_indices)}). "
                f"Need at least {min_cells}."
            )

        # Resolve gene subset
        gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(gene_subset)

        # Run spline association on pooled data
        result = self._run_spline_association(
            test_values=pooled_test_values,
            cell_indices=pooled_cell_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
            cluster_genes=cluster_genes,
        )

        # Add multi-line metadata
        result['line_name'] = ' + '.join(lines_used)
        result['test_variable'] = test_variable
        result['n_lines'] = len(lines_used)
        result['lines_used'] = lines_used
        result['gene_subset'] = self._gene_subset_summary(subset_type, subset_metadata)

        return result

    @staticmethod
    def _classify_profile_pattern(
        profile: np.ndarray,
        positions: np.ndarray,
    ) -> str:
        """Classify the shape of a gene expression profile along a line.

        Args:
            profile: Normalized expression profile (0-1 scale), shape (n_points,)
            positions: Corresponding position values along the line (0-1)

        Returns:
            One of: 'increasing', 'decreasing', 'peak', 'trough', 'complex'
        """
        # Correlation with position
        corr = np.corrcoef(positions, profile)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Strong monotonic trend
        if corr > 0.7:
            return 'increasing'
        if corr < -0.7:
            return 'decreasing'

        # Check for peak or trough: location of max/min relative to endpoints
        argmax = np.argmax(profile)
        argmin = np.argmin(profile)
        n = len(profile)
        interior_fraction = 0.15  # consider first/last 15% as "edges"
        edge_low = int(n * interior_fraction)
        edge_high = int(n * (1 - interior_fraction))

        max_is_interior = edge_low <= argmax <= edge_high
        min_is_interior = edge_low <= argmin <= edge_high

        # Peak: max in interior and higher than both endpoints
        edge_mean = (profile[:edge_low].mean() + profile[edge_high:].mean()) / 2
        center_val = profile[argmax] if max_is_interior else profile[argmin]

        if max_is_interior and profile[argmax] > edge_mean + 0.2:
            return 'peak'
        if min_is_interior and profile[argmin] < edge_mean - 0.2:
            return 'trough'

        # Fallback: use monotonicity for weak trends
        if corr > 0.3:
            return 'increasing'
        if corr < -0.3:
            return 'decreasing'

        return 'complex'

    def create_line_projection_embedding(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create an embedding based on cell projections onto a line.

        This creates a new embedding in .obsm where:
        - X-axis: position along the line (0 = start, 1 = end)
        - Y-axis: distance from the line (with jitter for visualization)

        Only cells that are projected are included; others get NaN.

        Args:
            line_name: Name of the line to create embedding from
            cell_indices: Optional cell indices to include. If None and line
                         has projections stored, uses those. Otherwise uses all.

        Returns:
            Dict with embedding name and cell count
        """
        # Find the line
        line = None
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line = l
                break

        if line is None:
            raise ValueError(f"Line '{line_name}' not found")

        embedding_name = line.get('embeddingName', '')
        if embedding_name not in self.adata.obsm:
            raise ValueError(f"Embedding '{embedding_name}' not found")

        # Get line points (smoothed if available)
        line_points = line.get('smoothedPoints') or line.get('points', [])
        if len(line_points) < 2:
            raise ValueError("Line must have at least 2 points")

        # Get embedding coordinates
        coords = self._line_view_coords(line)

        # Project all cells onto line
        positions, distances = self._project_cells_onto_line(line_points, coords)

        # Determine which cells to include
        if cell_indices is not None:
            cell_mask = np.zeros(self.n_cells, dtype=bool)
            cell_mask[cell_indices] = True
        else:
            # Use all cells
            cell_mask = np.ones(self.n_cells, dtype=bool)

        # Create the projection embedding
        # X = position along line (0-1), Y = distance from line (normalized to 0-1)
        proj_embedding = np.full((self.n_cells, 2), np.nan)
        proj_embedding[cell_mask, 0] = positions[cell_mask]

        # Normalize distances to 0-1 range
        masked_distances = distances[cell_mask]
        dist_min = masked_distances.min()
        dist_max = masked_distances.max()
        if dist_max > dist_min:
            normalized_distances = (masked_distances - dist_min) / (dist_max - dist_min)
        else:
            # All cells same distance from line
            normalized_distances = np.zeros_like(masked_distances)
        proj_embedding[cell_mask, 1] = normalized_distances

        # Sanitize line name for embedding key
        safe_name = line_name.replace(' ', '_').replace('-', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        emb_key = f'X_{safe_name}_proj'

        # Store in adata.obsm
        self.adata.obsm[emb_key] = proj_embedding

        n_cells_projected = int(cell_mask.sum())

        return {
            'embedding_name': emb_key,
            'n_cells': n_cells_projected,
            'position_range': [float(positions[cell_mask].min()), float(positions[cell_mask].max())],
            'distance_range_original': [float(dist_min), float(dist_max)],
            'distance_range_normalized': [0.0, 1.0],
        }

    # =========================================================================
    # Scanpy analysis methods
    # =========================================================================

    # =========================================================================
    # Analysis record (exportable provenance) — see xcell/analysis_record.py
    # =========================================================================

    ANALYSIS_RECORD_UNS_KEY = 'xcell_analysis_record'

    def _restore_analysis_record(self) -> AnalysisRecord:
        """Read a record out of .uns, or start a fresh one.

        An h5ad can come from anywhere, so anything unreadable under that key
        is discarded rather than allowed to block the load.
        """
        raw = self.adata.uns.get(self.ANALYSIS_RECORD_UNS_KEY)
        if raw is not None:
            try:
                return AnalysisRecord.from_dict(json.loads(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return AnalysisRecord()

    def _record_source(self, path: Path, kind: str) -> None:
        """Record what was loaded, as the record's first step and its source.

        Without this the exported notebook has nothing to open — the single
        biggest gap in the old action history.
        """
        self.analysis_record.source = {
            'path': str(path),
            'kind': kind,
            'n_cells': int(self.n_cells),
            'n_genes': int(self.n_genes),
        }
        self.analysis_record.add_step(
            'load_dataset',
            {'path': str(path), 'kind': kind},
            {'n_cells': int(self.n_cells), 'n_genes': int(self.n_genes)},
        )

    def _serialize_record(self) -> str:
        """The record as a JSON string for .uns, without figure payloads.

        h5ad copes badly with deeply nested heterogeneous dicts, so the whole
        thing goes in as one string. Figures are dropped: base64 PNGs would
        bloat every exported dataset, and they belong to the notebook, not to
        the matrix.
        """
        payload = self.analysis_record.to_dict()
        payload['figures'] = {}
        for step in payload['steps']:
            step['figure_ids'] = []
        return json.dumps(payload)

    def set_source(self, path: Path | str, kind: str) -> None:
        """Correct the recorded source after construction.

        The load route converts .rds to a temporary h5ad before handing it to
        the adaptor; the record should name the file the user actually chose.
        """
        self.filepath = Path(path)
        self.analysis_record.source.update({'path': str(path), 'kind': kind})
        for step in self.analysis_record.steps:
            if step.action == 'load_dataset':
                step.params.update({'path': str(path), 'kind': kind})
                break

    def get_action_history(self) -> list[dict[str, Any]]:
        """Get the history of scanpy operations performed."""
        return self._action_history

    def _log_action(
        self,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any],
        *,
        subset: Any = None,
    ) -> None:
        """Log an action to the history and the exportable analysis record.

        ``subset`` is the *resolved* active-cell selection — what
        ``_validate_cell_indices`` / ``_get_active_adata`` returned, so it is
        already None when the caller passed no selection or one covering every
        cell. Passing it explicitly at each call site rather than stashing it on
        self keeps a selection from being misattributed to whichever operation
        happens to log next.
        """
        import datetime
        self._action_history.append({
            'action': action,
            'params': params,
            'result': result,
            'timestamp': datetime.datetime.now().isoformat(),
        })
        self.analysis_record.add_step(
            action, params, result,
            selection=None if subset is None else [int(i) for i in subset],
            n_total=self.n_cells,
        )

    # =========================================================================
    # Named cell subsets
    # =========================================================================

    @staticmethod
    def _sanitize_subset_name(name: str) -> str:
        """A subset name suffixes obsm/obsp/obs keys, so it must be key-safe."""
        import re
        clean = re.sub(r'[^A-Za-z0-9_]+', '_', str(name or '')).strip('_')
        if not clean:
            raise ValueError("Subset name must contain at least one letter or digit.")
        if clean.lower() in {'unassigned', 'nan', 'none', 'null'}:
            raise ValueError(f"'{clean}' is reserved; choose another subset name.")
        return clean

    @staticmethod
    def _plain(value: Any) -> Any:
        """Give a uns value back its JSON shape.

        h5ad turns an empty list into ``array([], dtype=float64)``, a list of
        strings into a string array and a nested dict into an h5 group, so a
        registry read back from disk compares unequal to the one written and
        will not json.dumps. Normalising on read is what lets the registry
        stay a plain dict — readable in scanpy — rather than a JSON blob.
        """
        if isinstance(value, np.ndarray):
            return [DataAdaptor._plain(v) for v in value.tolist()]
        if isinstance(value, Mapping):
            return {str(k): DataAdaptor._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DataAdaptor._plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _subset_registry(self) -> dict[str, dict[str, Any]]:
        reg = self.adata.uns.get(CELL_SUBSETS_UNS)
        if not isinstance(reg, Mapping):
            return {}
        return {str(k): dict(v) for k, v in self._plain(reg).items() if isinstance(v, Mapping)}

    def _subset_tree(
        self, registry: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, list[str]], dict[str, int], list[str]]:
        """Children, depth and a depth-first order over the registry.

        A ``parent`` naming a subset that is no longer registered counts as
        none, so a column dropped in scanpy orphans its children rather than
        hiding them. Roots and siblings sort by creation time, then name.
        """
        def sort_key(n: str) -> tuple[str, str]:
            return (registry[n].get('created_at') or '', n)

        children: dict[str, list[str]] = {n: [] for n in registry}
        roots: list[str] = []
        for n, e in registry.items():
            parent = e.get('parent')
            if parent and parent in registry and parent != n:
                children[parent].append(n)
            else:
                roots.append(n)
        for kids in children.values():
            kids.sort(key=sort_key)
        roots.sort(key=sort_key)

        depth: dict[str, int] = {}
        order: list[str] = []
        stack = [(n, 0) for n in reversed(roots)]
        while stack:
            n, d = stack.pop()
            if n in depth:   # a cycle in a hand-edited file
                continue
            depth[n] = d
            order.append(n)
            stack.extend((k, d + 1) for k in reversed(children[n]))
        for n in registry:      # anything only reachable through a cycle
            if n not in depth:
                depth[n] = 0
                order.append(n)
        return children, depth, order

    def _infer_parent(self, idx: np.ndarray, exclude: str | None = None) -> str | None:
        """The smallest registered subset containing every cell of ``idx``.

        A subset with exactly these cells is a twin, not a parent. Ties go to
        the most recently created, which in the recursive workflow is the one
        the user was just working in.
        """
        registry = self._subset_registry()
        best: tuple[int, str, str] | None = None
        for name, entry in registry.items():
            if name == exclude:
                continue
            key = entry.get('obs_key', SUBSET_OBS_PREFIX + name)
            if key not in self.adata.obs.columns:
                continue
            mask = self.adata.obs[key].values.astype(bool)
            n = int(mask.sum())
            if n <= len(idx) or not mask[idx].all():
                continue
            cand = (n, entry.get('created_at') or '', name)
            if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
        return best[2] if best else None

    def _subset_mask(self, name: str) -> np.ndarray:
        """Boolean membership of a registered subset; KeyError if unknown."""
        clean = self._sanitize_subset_name(name)
        key = SUBSET_OBS_PREFIX + clean
        if clean not in self._subset_registry() or key not in self.adata.obs.columns:
            known = sorted(self._subset_registry())
            raise KeyError(
                f"No cell subset named '{clean}'. "
                f"Known subsets: {known if known else 'none'}")
        return self.adata.obs[key].values.astype(bool)

    def _subset_record_step(
        self, name: str, step: str, key: str,
        params: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Write what a scoped operation just produced into the registry.

        ``hvg`` / ``pca`` / ``graph`` have one output per subset; ``umap`` and
        ``leiden`` are keyed by output name so a re-run overwrites and a run
        over another graph sits beside the first; ``pca_subsets`` keeps the
        dropped PCs. Only flat scalars are kept — the notebook already gets
        the full params from the analysis record.
        """
        registry = self._subset_registry()
        entry = registry.get(name)
        if entry is None:
            return
        derived = entry.setdefault('derived', {})
        clean = {k: (v.item() if isinstance(v, np.generic) else v)
                 for k, v in (params or {}).items()
                 if v is None or isinstance(v, (str, int, float, bool, np.generic))}
        if step in ('hvg', 'pca', 'graph'):
            derived[step] = {'key': key, 'params': clean}
        elif step in ('umap', 'leiden'):
            derived.setdefault(step, {})[key] = clean
        elif step == 'pca_subsets':
            derived.setdefault('pca_subsets', {})[key] = dict(extra or {})
        registry[name] = entry
        self.adata.uns[CELL_SUBSETS_UNS] = registry

    def _subset_forget_key(self, name: str, key: str) -> None:
        """Drop any record of ``key`` from a subset's derived entry."""
        registry = self._subset_registry()
        entry = registry.get(name)
        if entry is None:
            return
        derived = entry.get('derived') or {}
        for step in ('hvg', 'pca', 'graph'):
            if (derived.get(step) or {}).get('key') == key:
                derived.pop(step)
        for step in ('umap', 'leiden', 'pca_subsets'):
            if isinstance(derived.get(step), dict):
                derived[step].pop(key, None)
        entry['derived'] = derived
        registry[name] = entry
        self.adata.uns[CELL_SUBSETS_UNS] = registry

    def _subset_derived_keys(self, name: str, entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What exists of this subset's results: the recorded keys, plus
        anything the naming convention finds (files from before the record),
        filtered to keys that are actually there."""
        derived = (entry or {}).get('derived') or {}

        def recorded(step: str) -> str | None:
            rec = derived.get(step)
            return rec.get('key') if isinstance(rec, Mapping) else None

        def recorded_keys(step: str) -> list[str]:
            rec = derived.get(step)
            return list(rec.keys()) if isinstance(rec, Mapping) else []

        hvg = recorded('hvg') or f'highly_variable__{name}'
        pca = recorded('pca') or f'X_pca_{name}'
        graph = recorded('graph') or f'{name}_connectivities'
        umaps = recorded_keys('umap')
        for k in self.adata.obsm.keys():
            if (k == f'X_umap_{name}' or k.startswith(f'X_umap_{name}_')) and k not in umaps:
                umaps.append(k)
        leidens = recorded_keys('leiden')
        for c in self.adata.obs.columns:
            if (c == f'leiden_{name}' or c.startswith(f'leiden_{name}_')) and c not in leidens:
                leidens.append(c)
        pc_subsets = recorded_keys('pca_subsets')
        for k in self.adata.obsm.keys():
            if k.startswith(f'X_pca_{name}_') and k not in pc_subsets:
                pc_subsets.append(k)
        return {
            'hvg': hvg if hvg in self.adata.var.columns else None,
            'pca': pca if pca in self.adata.obsm else None,
            'graph': graph if graph in self.adata.obsp else None,
            'umap': [k for k in umaps if k in self.adata.obsm],
            'leiden': [c for c in leidens if c in self.adata.obs.columns],
            'pca_subsets': [k for k in pc_subsets if k in self.adata.obsm],
        }

    def _subset_summary(
        self, name: str, registry: Mapping[str, Mapping[str, Any]],
        tree: tuple[dict[str, list[str]], dict[str, int], list[str]] | None = None,
    ) -> dict[str, Any]:
        entry = registry[name]
        children, depth, _ = tree if tree is not None else self._subset_tree(registry)
        mask = self.adata.obs[SUBSET_OBS_PREFIX + name].values.astype(bool)
        parent = entry.get('parent')
        origin = entry.get('origin')
        derived = self._subset_derived_keys(name, entry)
        embeddings = ([derived['pca']] if derived['pca'] else []) + derived['pca_subsets'] + derived['umap']
        return {
            'name': name,
            'obs_key': SUBSET_OBS_PREFIX + name,
            'n_cells': int(mask.sum()),
            'n_total': self.n_cells,
            'created_at': entry.get('created_at'),
            'description': entry.get('description') or None,
            'parent': parent if parent and parent in registry else None,
            'children': list(children.get(name, [])),
            'depth': int(depth.get(name, 0)),
            'origin': dict(origin) if isinstance(origin, Mapping) and origin else None,
            'derived': derived,
            'steps': entry.get('derived') or {},
            'embeddings': embeddings,
            'decorations': {
                'lines': [line.get('name', '') for line in self._lines_on(set(embeddings))],
                'territories': self._territories_on(set(embeddings)),
            },
        }

    def create_cell_subset(
        self,
        name: str,
        cell_indices: list[int],
        *,
        description: str | None = None,
        overwrite: bool = False,
        parent: str | None = None,
        origin: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a cell selection as a named subset.

        Writes ``obs['subset_<name>']`` (bool) and a registry entry in
        ``uns['xcell_cell_subsets']``. The subset is what the clustering-chain
        operations take as ``cell_subset=`` to run on those cells alone while
        writing to suffixed keys.

        Args:
            parent: The subset this one nests under. Left unset it is inferred
                as the smallest registered subset containing every cell —
                which, in the recursive sub-clustering workflow, is the one
                the selection was made inside. Given, it must contain them.
            origin: How the selection was made, e.g. ``{'kind': 'selection',
                'embedding': 'X_umap'}``; stored as given, ``None`` values
                dropped.
        """
        import datetime
        clean = self._sanitize_subset_name(name)
        if cell_indices is None or len(cell_indices) == 0:
            raise ValueError("A subset needs at least one cell.")
        idx = np.unique(np.asarray(cell_indices, dtype=int))
        if idx.min() < 0 or idx.max() >= self.n_cells:
            raise ValueError(
                f"Cell indices out of range: max index {int(idx.max())}, n_cells {self.n_cells}")
        if len(idx) == self.n_cells:
            raise ValueError(
                "The selection covers every cell — that is the dataset itself, not a subset.")
        key = SUBSET_OBS_PREFIX + clean
        registry = self._subset_registry()
        if not overwrite and (clean in registry or key in self.adata.obs.columns):
            raise ValueError(
                f"A subset named '{clean}' already exists. Choose another name or overwrite it.")

        if parent is not None:
            parent = self._sanitize_subset_name(parent)
            if parent not in registry:
                raise ValueError(f"No cell subset named '{parent}' to be the parent.")
            pkey = registry[parent].get('obs_key', SUBSET_OBS_PREFIX + parent)
            if pkey not in self.adata.obs.columns or not self.adata.obs[pkey].values.astype(bool)[idx].all():
                raise ValueError(
                    f"Subset '{parent}' does not contain every cell of '{clean}', "
                    f"so it cannot be its parent.")
        else:
            parent = self._infer_parent(idx, exclude=clean)

        mask = np.zeros(self.n_cells, dtype=bool)
        mask[idx] = True
        self.adata.obs[key] = mask
        entry: dict[str, Any] = {
            'obs_key': key,
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'description': str(description) if description else '',
        }
        if parent:
            entry['parent'] = parent
        clean_origin = {str(k): v for k, v in (origin or {}).items() if v is not None}
        if clean_origin:
            entry['origin'] = clean_origin
        registry[clean] = entry
        self.adata.uns[CELL_SUBSETS_UNS] = registry

        out = self._subset_summary(clean, registry)
        params: dict[str, Any] = {'name': clean}
        if parent:
            params['parent'] = parent
        self._log_action('create_cell_subset', params,
                         {'n_cells': out['n_cells'], 'obs_key': key}, subset=idx)
        return out

    def list_cell_subsets(self) -> list[dict[str, Any]]:
        """Every registered subset with its live cell count and derived keys.

        An entry whose column has vanished (a user dropped it in scanpy) is
        pruned rather than reported, so the list never names a subset that
        cannot be activated.
        """
        registry = self._subset_registry()
        live = {n: e for n, e in registry.items()
                if e.get('obs_key', SUBSET_OBS_PREFIX + n) in self.adata.obs.columns}
        if len(live) != len(registry):
            self.adata.uns[CELL_SUBSETS_UNS] = live
        tree = self._subset_tree(live)
        return [self._subset_summary(n, live, tree) for n in tree[2]]

    def get_cell_subset_indices(self, name: str) -> list[int]:
        return [int(i) for i in np.where(self._subset_mask(name))[0]]

    def delete_cell_subset(self, name: str, *, drop_derived: bool = False) -> dict[str, Any]:
        """Remove a subset; optionally everything computed on it too — its
        results, and the shapes and territories drawn on its embeddings."""
        clean = self._sanitize_subset_name(name)
        self._subset_mask(clean)  # KeyError if unknown
        dropped: list[str] = []
        dropped_lines: list[str] = []
        dropped_territories: list[str] = []
        key = SUBSET_OBS_PREFIX + clean
        if key in self.adata.obs.columns:
            del self.adata.obs[key]
            dropped.append(key)
        registry = self._subset_registry()
        gone = registry.pop(clean, None) or {}
        # Its children move up to its parent, so the tree stays connected and
        # nothing under the deleted node is lost or hidden.
        for entry in registry.values():
            if entry.get('parent') == clean:
                if gone.get('parent'):
                    entry['parent'] = gone['parent']
                else:
                    entry.pop('parent', None)
        self.adata.uns[CELL_SUBSETS_UNS] = registry

        if drop_derived:
            derived = self._subset_derived_keys(clean, gone)
            if derived['hvg']:
                del self.adata.var[derived['hvg']]
                dropped.append(derived['hvg'])
            if derived['pca']:
                del self.adata.obsm[derived['pca']]
                dropped.append(derived['pca'])
                self.adata.varm.pop(f'PCs_{clean}', None)
                self.adata.uns.pop(f'pca_{clean}', None)
            if derived['graph']:
                for k in (derived['graph'], f'{clean}_distances'):
                    if k in self.adata.obsp:
                        del self.adata.obsp[k]
                        dropped.append(k)
                self.adata.uns.pop(clean, None)
            for k in derived['pca_subsets']:
                del self.adata.obsm[k]
                self.adata.varm.pop('PCs_' + k[len('X_pca_'):], None)
                dropped.append(k)
            for k in derived['umap']:
                del self.adata.obsm[k]
                dropped.append(k)
            for col in derived['leiden']:
                del self.adata.obs[col]
                dropped.append(col)

            # Shapes and territories drawn on an embedding that is gone have
            # no coordinates left to live in, so they go with it.
            gone_embeddings = {k for k in dropped if k.startswith('X_')}
            for line in self._lines_on(gone_embeddings):
                dropped_lines.append(line.get('name', ''))
            if dropped_lines:
                self.set_lines([line for line in self._drawn_lines
                                if line.get('embeddingName') not in gone_embeddings])
            dropped_territories = self._territories_on(gone_embeddings)
            if dropped_territories:
                stored = self.get_territories()
                for t in dropped_territories:
                    stored.pop(t, None)
                self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)

        result = {'name': clean, 'dropped': dropped,
                  'dropped_lines': dropped_lines, 'dropped_territories': dropped_territories}
        self._log_action('delete_cell_subset',
                         {'name': clean, 'drop_derived': bool(drop_derived)}, result)
        return result

    def _resolve_cell_scope(
        self, cell_subset: str | None, active_cell_indices: list[int] | None,
    ) -> tuple[np.ndarray | None, str | None]:
        """The cells an operation runs on, and the subset name if it came from one.

        A named subset wins over ad-hoc indices: the name is what scopes the
        output keys, and the indices the browser holds for it are the same
        cells anyway.
        """
        if cell_subset:
            clean = self._sanitize_subset_name(cell_subset)
            return np.where(self._subset_mask(clean))[0], clean
        return self._validate_cell_indices(active_cell_indices), None

    def _validate_cell_indices(
        self, active_cell_indices: list[int] | None,
    ) -> np.ndarray | None:
        """Validate active_cell_indices and return as numpy array.

        Returns None if no subsetting is needed (indices is None or covers all cells).
        """
        if active_cell_indices is None:
            return None

        indices = np.array(active_cell_indices)

        if len(indices) == 0:
            raise ValueError("active_cell_indices is empty — no cells selected.")

        if indices.max() >= self.n_cells or indices.min() < 0:
            raise ValueError(
                f"active_cell_indices out of range: max index {indices.max()}, "
                f"n_cells {self.n_cells}"
            )

        if len(indices) == self.n_cells:
            return None

        return indices

    def _get_active_adata(
        self, active_cell_indices: list[int] | None,
    ) -> tuple[anndata.AnnData, np.ndarray | None]:
        """Get an AnnData subset copy for active cells.

        Args:
            active_cell_indices: List of cell indices to include, or None for all.

        Returns:
            Tuple of (adata_subset_or_full, indices_array_or_None).
            If indices is None or covers all cells, returns (self.adata, None).
        """
        indices = self._validate_cell_indices(active_cell_indices)
        if indices is None:
            return self.adata, None
        return self.adata[indices].copy(), indices

    def check_prerequisites(
        self, action: str, cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Check if prerequisites are met for a scanpy action.

        Args:
            action: The scanpy action to check
            cell_subset: A named subset's own keys (``X_pca_<name>``,
                ``<name>_connectivities``) satisfy the check as well as the
                dataset's, since an operation on that subset can use either.

        Returns:
            Dict with 'satisfied' (bool) and 'missing' (list of missing prereqs)
        """
        prereqs = {
            # Cell analysis
            'filter_genes': [],
            'exclude_genes': [],
            'filter_cells': [],
            'normalize_total': [],
            'log1p': [],
            'pca': [],
            'neighbors': ['pca'],
            'umap': ['neighbors'],
            'leiden': ['neighbors'],
            'pca_loadings': ['pca_with_loadings'],
            # Gene analysis
            'gene_pca': [],
            'gene_neighbors': [],
            'find_similar_genes': ['gene_neighbors'],
            'cluster_genes': ['gene_neighbors'],
            'build_gene_graph': [],  # Convenience function, no prereqs
            # Spatial analysis
            'spatial_neighbors': ['has_spatial'],
            'spatial_autocorr': ['spatial_neighbors'],
            'contourize': ['has_spatial'],
            'multicontour': ['has_spatial'],  # X_pca additionally checked in prepare
            'neighborhood': ['has_spatial'],
        }

        required = prereqs.get(action, [])
        missing = []

        for prereq in required:
            if prereq == 'pca':
                keys = ['X_pca'] + ([f'X_pca_{cell_subset}'] if cell_subset else [])
                if not any(k in self.adata.obsm for k in keys):
                    missing.append('pca')
            elif prereq == 'neighbors':
                keys = ['connectivities'] + ([f'{cell_subset}_connectivities'] if cell_subset else [])
                if not any(k in self.adata.obsp for k in keys):
                    missing.append('neighbors')
            elif prereq == 'gene_pca':
                if 'X_gene_pca' not in self.adata.varm:
                    missing.append('gene_pca')
            elif prereq == 'gene_neighbors':
                if 'gene_connectivities' not in self.adata.varp:
                    missing.append('gene_neighbors')
            elif prereq == 'pca_with_loadings':
                own = 'pca' in self.adata.uns and 'PCs' in self.adata.varm
                subsets_own = bool(cell_subset) and (
                    f'pca_{cell_subset}' in self.adata.uns and f'PCs_{cell_subset}' in self.adata.varm)
                if not own and not subsets_own:
                    missing.append('pca_with_loadings')
            elif prereq == 'has_spatial':
                if not self._has_spatial_coordinates():
                    missing.append('has_spatial')
            elif prereq == 'spatial_neighbors':
                if 'spatial_connectivities' not in self.adata.obsp:
                    missing.append('spatial_neighbors')

        return {
            'satisfied': len(missing) == 0,
            'missing': missing,
        }

    # Where an explicit choice of coordinate array is remembered. In .uns so it
    # travels with the dataset, and namespaced because it is xcell's, not a
    # convention any other tool will recognize.
    SPATIAL_KEY_UNS = 'xcell_spatial_key'

    def _is_coordinate_array(self, key: str) -> bool:
        """Could this .obsm entry be a set of 2-D positions?"""
        if key not in self.adata.obsm:
            return False
        arr = self.adata.obsm[key]
        return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2

    def _has_spatial_coordinates(self) -> bool:
        """Whether any array is acting as this dataset's spatial coordinates."""
        return self._get_spatial_key() is not None

    def _get_spatial_key(self) -> str | None:
        """The .obsm key holding this dataset's spatial coordinates, if any.

        An explicit choice wins, then the two conventional names. The explicit
        step exists because Localize writes its map to ``X_spatial_pred`` (or
        whatever ``key_added`` names), which auto-detection could not see — so
        every spatial tool refused to run on coordinates the user had just
        produced.

        Widening auto-detection instead would have to *guess*: a query localized
        under several settings carries several predicted maps, and picking one
        would silently decide which analysis the user meant. It also matters
        that predicted coordinates are a weaker claim than measured ones —
        keeping the choice explicit is what lets the UI say so.

        A stored key naming an array that no longer exists falls back to
        auto-detection rather than stranding the dataset, since .uns survives a
        save and the .obsm it names may not.
        """
        chosen = self.adata.uns.get(self.SPATIAL_KEY_UNS)
        if isinstance(chosen, str) and self._is_coordinate_array(chosen):
            return chosen
        for key in ('spatial', 'X_spatial'):
            if self._is_coordinate_array(key):
                return key
        return None

    def set_spatial_key(self, key: str | None) -> dict[str, Any]:
        """Choose which .obsm array acts as this dataset's spatial coordinates.

        Args:
            key: an .obsm key, or None to go back to auto-detection.
        """
        if key is None:
            self.adata.uns.pop(self.SPATIAL_KEY_UNS, None)
        else:
            key = str(key)
            if key not in self.adata.obsm:
                raise ValueError(
                    f"'{key}' not found in .obsm. "
                    f'Available: {list(self.adata.obsm.keys())}'
                )
            if not self._is_coordinate_array(key):
                arr = self.adata.obsm[key]
                shape = getattr(arr, 'shape', None)
                raise ValueError(
                    f"'{key}' cannot be spatial coordinates: it needs at least "
                    f'two dimensions per cell, and has shape {shape}.'
                )
            self.adata.uns[self.SPATIAL_KEY_UNS] = key

        self._log_action('set_spatial_key', {'key': key}, {'spatial_key': key})
        return self.spatial_key_options()

    def spatial_key_options(self) -> dict[str, Any]:
        """Every .obsm array that could serve as coordinates, and which is active.

        ``predicted`` is read from the companion ``<key>_confidence`` column
        Localize writes beside its map, so maps made before this picker existed
        are still recognized as predictions. ``n_missing`` counts cells with no
        coordinate — ``min_confidence`` leaves those as NaN deliberately, and a
        spatial graph built over them is meaningless, so the number belongs in
        front of the choice rather than behind it.
        """
        options = []
        for key in self.adata.obsm.keys():
            if not self._is_coordinate_array(key):
                continue
            arr = np.asarray(self.adata.obsm[key], dtype=float)
            options.append({
                'key': str(key),
                'n_dims': int(arr.shape[1]),
                'predicted': f'{key}_confidence' in self.adata.obs.columns,
                'n_missing': int((~np.isfinite(arr[:, :2]).all(axis=1)).sum()),
            })

        explicit = self.adata.uns.get(self.SPATIAL_KEY_UNS)
        return {
            'current': self._get_spatial_key(),
            'explicit': explicit if isinstance(explicit, str) else None,
            'options': options,
        }

    SPATIAL_SCALE_UNS = 'xcell_spatial_scale'

    def merge_spots(
        self,
        region_indices: list[int],
        params: Any,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        layer: str | None = 'counts',
        purity_column: str | None = None,
        dry_run: bool = True,
    ) -> tuple[dict[str, Any], Any]:
        """Merge neighbouring spots in a region into single large cells.

        Read-only on self. The merged rows never existed, so every embedding and
        derived column computed on the originals would be stale — the result is
        a separate AnnData for the caller to register as its own dataset.

        Args:
            region_indices: The spots to merge, from the user's selection.
            params: A spot_merge.MergeParams.
            gene_subset: Genes the veto correlates over. Any form
                ``_resolve_gene_mask`` accepts, so a Gene Panel gene set
                arrives as an explicit gene list.
            layer: Matrix to sum. Must be integer-like.
            purity_column: Categorical column ``merge_purity`` votes on.
            dry_run: When True, compute the statistics and build nothing.

        Returns:
            (stats, merged_adata). merged_adata is None when dry_run.

        Raises:
            ValueError: No coordinates, no scale, too small a region, or a
                non-integer matrix.
        """
        from xcell.spot_merge import merge_spots as _merge

        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError(
                "This dataset has no spatial coordinates, so there are no "
                "neighbouring spots to merge."
            )
        indices = np.asarray(sorted({int(i) for i in region_indices}), dtype=int)
        if len(indices) < 2:
            raise ValueError("Merging needs a region of at least two spots.")

        scale = self.spatial_scale()
        if scale.get('um_per_unit') is None:
            raise ValueError(
                "Set the µm per coordinate unit before merging — a cell "
                "diameter in microns means nothing without it."
            )
        um = float(scale['um_per_unit'])
        coords = np.asarray(
            self.adata.obsm[spatial_key], dtype=float
        )[indices, :2] * um

        source = (
            self.adata.layers[layer]
            if layer and layer in self.adata.layers
            else self.adata.X
        )
        region_counts = source[indices]
        if hasattr(region_counts, 'toarray'):
            region_counts = region_counts.toarray()
        region_counts = np.asarray(region_counts, dtype=float)
        if not np.all(np.equal(np.mod(region_counts, 1), 0)):
            raise ValueError(
                "Merging sums raw counts, and this matrix is not integer-like. "
                "Name a counts layer with the `layer` parameter."
            )

        # The veto reads normalized profiles even though the merge sums raw
        # counts: on raw counts any two spots correlate highly, because both are
        # dominated by the same few high expressors, and the veto never fires.
        # Structural: this rebuilds the cells, so it reads the whole dataset
        # rather than the Gene Panel's current view.
        gene_mask, subset_type, _ = self._resolve_gene_mask(
            gene_subset, apply_visible_mask=False)
        result = _merge(coords, region_counts, region_counts[:, gene_mask], params)

        stats: dict[str, Any] = {
            'n_region_spots': int(len(indices)),
            'n_merged_spots': int(result.n_groups),
            'n_spots_before': int(self.n_cells),
            'n_spots_after': int(self.n_cells - len(indices) + result.n_groups),
            'pitch_um': float(result.pitch_um),
            'max_spots': int(result.max_spots),
            'size_histogram': {str(k): v for k, v in result.size_histogram.items()},
            'n_vetoed': int(result.n_vetoed),
            'correlation_quantiles': result.correlation_quantiles or None,
            'gene_subset': self._gene_subset_summary(
                subset_type, {'n_genes': int(gene_mask.sum())}
            ),
            'median_counts_before': float(np.median(region_counts.sum(axis=1))),
            '_labels': [int(v) for v in result.labels],
        }
        if dry_run:
            return stats, None

        merged = self._build_merged_adata(
            indices, result.labels, spatial_key, purity_column, params, stats,
        )
        stats['median_counts_after'] = float(
            np.median(np.asarray(merged.layers['counts'].sum(axis=1)).ravel())
        )
        return stats, merged

    def _build_merged_adata(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        spatial_key: str,
        purity_column: str | None,
        params: Any,
        stats: dict[str, Any],
    ):
        """Build the merged dataset: untouched rows first, then one per group.

        Numeric and QC .obs are dropped rather than averaged: a mean of
        per-spot scores is an approximation the summed counts cannot justify,
        and keeping them only outside the region would leave a column that is
        populated in half the tissue and blank in the other.
        """
        from scipy.sparse import csr_matrix, vstack

        counts_src = (
            self.adata.layers['counts']
            if 'counts' in self.adata.layers
            else self.adata.X
        )
        counts_src = csr_matrix(counts_src)
        in_region = np.zeros(self.n_cells, dtype=bool)
        in_region[indices] = True
        outside = np.where(~in_region)[0]

        cat_cols = [
            c for c in self.adata.obs.columns
            if isinstance(self.adata.obs[c].dtype, pd.CategoricalDtype)
            or self.adata.obs[c].dtype == object
        ]
        if purity_column is None:
            purity_column = cat_cols[0] if cat_cols else None

        source_coords = np.asarray(self.adata.obsm[spatial_key], dtype=float)
        rows, coords_out, obs_rows, names = [], [], [], []

        for i in outside:
            rows.append(counts_src[i])
            coords_out.append(source_coords[i, :2])
            row = {c: self.adata.obs[c].iloc[i] for c in cat_cols}
            row.update(merged_n_spots=1, merge_group='', merge_purity=np.nan)
            obs_rows.append(row)
            names.append(str(self.adata.obs_names[i]))

        for gid in range(int(labels.max()) + 1):
            members = indices[labels == gid]
            rows.append(csr_matrix(counts_src[members].sum(axis=0)))
            coords_out.append(source_coords[members, :2].mean(axis=0))
            row: dict[str, Any] = {
                'merged_n_spots': int(len(members)),
                'merge_group': f'g{gid}',
            }
            purity = np.nan
            for c in cat_cols:
                values = self.adata.obs[c].iloc[members]
                tally = values.value_counts()
                row[c] = tally.index[0] if len(tally) else None
                if c == purity_column and len(values):
                    purity = float(tally.iloc[0] / len(values))
            row['merge_purity'] = purity
            obs_rows.append(row)
            names.append(f'merged_g{gid}')

        X = vstack(rows, format='csr').astype(np.float32)
        obs = pd.DataFrame(obs_rows, index=pd.Index(names))
        if purity_column is None:
            obs = obs.drop(columns=['merge_purity'])
        for c in cat_cols:
            obs[c] = pd.Categorical(obs[c])
        obs['merge_group'] = pd.Categorical(obs['merge_group'])

        merged = anndata.AnnData(X=X, obs=obs, var=self.adata.var.copy())
        merged.layers['counts'] = merged.X.copy()
        merged.obsm[spatial_key] = np.asarray(coords_out, dtype=float)
        merged.uns['xcell_spot_merge'] = {
            'params': {
                'max_diameter_um': float(params.max_diameter_um),
                'min_correlation': float(params.min_correlation),
                'eligibility': params.eligibility,
                'max_spots': int(stats['max_spots']),
            },
            'source_file': str(self.filepath.name),
            'n_spots_before': int(self.n_cells),
            'n_spots_after': int(merged.n_obs),
            'gene_subset': stats['gene_subset'],
        }
        if self.SPATIAL_SCALE_UNS in self.adata.uns:
            merged.uns[self.SPATIAL_SCALE_UNS] = dict(
                self.adata.uns[self.SPATIAL_SCALE_UNS]
            )
        return merged

    def spatial_scale(self) -> dict[str, Any]:
        """Physical size of one spatial coordinate unit, in µm.

        The h5ad itself carries no units — Curio Seeker writes µm, plain
        Visium writes full-res image pixels, a Localize map inherits whatever
        its reference used — so this is an explicit per-dataset setting. The
        one safe auto-detection: the Visium HD loader builds its coordinates
        in µm and leaves ``bin_size_um`` in ``uns['spatial']`` saying so.
        """
        stored = self.adata.uns.get(self.SPATIAL_SCALE_UNS)
        if isinstance(stored, dict) and stored.get('um_per_unit') is not None:
            return {'um_per_unit': float(stored['um_per_unit']), 'source': 'user'}
        spatial_meta = self.adata.uns.get('spatial')
        if isinstance(spatial_meta, dict) and spatial_meta.get('bin_size_um') is not None:
            return {'um_per_unit': 1.0, 'source': 'visium_hd'}
        return {'um_per_unit': None, 'source': None}

    def spatial_scale_set(self, um_per_unit: float | None) -> dict[str, Any]:
        """Set (or clear, with None) how many µm one coordinate unit spans."""
        if um_per_unit is None:
            self.adata.uns.pop(self.SPATIAL_SCALE_UNS, None)
        else:
            value = float(um_per_unit)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f'µm per coordinate unit must be a positive number, got {um_per_unit}'
                )
            self.adata.uns[self.SPATIAL_SCALE_UNS] = {'um_per_unit': value}
        out = self.spatial_scale()
        self._log_action('set_spatial_scale', {'um_per_unit': um_per_unit}, out)
        return out

    def _split_present_genes(self, genes: list[str]) -> tuple[list[str], list[str]]:
        """Partition gene names into (present in .var_names, missing).

        Used by the contour paths so an imported gene set with a few genes not in
        the active dataset still runs on the genes that are present, reporting the
        dropped ones rather than failing outright.
        """
        var_names = self.adata.var_names
        present, missing = [], []
        for g in genes:
            (present if g in var_names else missing).append(g)
        return present, missing

    def _resolve_sections(self, section_col: str | None) -> np.ndarray | None:
        """Resolve a section obs-column name to a per-cell label array.

        Returns None when ``section_col`` is None (single-tissue behavior).
        Raises ValueError if the column is missing.
        """
        if section_col is None:
            return None
        if section_col not in self.adata.obs.columns:
            raise ValueError(f"Section column '{section_col}' not found in .obs")
        return np.asarray(self.adata.obs[section_col].astype(str).values)

    def _column_to_bool_array(self, col_name: str) -> np.ndarray:
        """Convert a .var column to a boolean numpy array.

        Accepts bool columns and numeric 0/1 columns (matching the same
        rules as get_var_boolean_columns). Raises ValueError if the
        column is not bool-like.
        """
        if col_name not in self.adata.var.columns:
            raise ValueError(f"Column '{col_name}' not found in .var")
        series = self.adata.var[col_name]
        dtype = series.dtype
        if dtype == bool:
            return np.asarray(series.values, dtype=bool)
        if pd.api.types.is_numeric_dtype(dtype):
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                return np.asarray((series == 1) | (series == True), dtype=bool)
        raise ValueError(f"Column '{col_name}' is not a boolean-like column")

    def _compute_visible_mask(
        self,
        keep_columns: list[str],
        hide_columns: list[str],
        keep_combine_mode: str,
    ) -> np.ndarray:
        """Compute the final visible mask from a config.

        Formula:
            visible = keep_mask AND NOT hide_mask
            keep_mask = all-True if no keep columns
                      = OR(columns) if keep_combine_mode == 'or'
                      = AND(columns) if keep_combine_mode == 'and'
            hide_mask = all-False if no hide columns
                      = OR(columns) otherwise
        """
        n = self.n_genes
        if keep_columns:
            arrays = [self._column_to_bool_array(c) for c in keep_columns]
            if keep_combine_mode == 'and':
                keep_mask = arrays[0].copy()
                for a in arrays[1:]:
                    keep_mask &= a
            else:  # 'or' (default)
                keep_mask = arrays[0].copy()
                for a in arrays[1:]:
                    keep_mask |= a
        else:
            keep_mask = np.ones(n, dtype=bool)

        if hide_columns:
            arrays = [self._column_to_bool_array(c) for c in hide_columns]
            hide_mask = arrays[0].copy()
            for a in arrays[1:]:
                hide_mask |= a
        else:
            hide_mask = np.zeros(n, dtype=bool)

        return keep_mask & ~hide_mask

    def get_gene_mask(self) -> dict[str, Any]:
        """Return the current mask config + counts.

        Always returns a dict, even when no mask is active.
        """
        n_total = self.n_genes
        if self._gene_mask_config is None or self._visible_gene_mask is None:
            return {
                'active': False,
                'keep_columns': [],
                'hide_columns': [],
                'keep_combine_mode': 'or',
                'n_visible': n_total,
                'n_total': n_total,
                'visible_gene_names': None,
            }
        n_visible = int(self._visible_gene_mask.sum())
        visible_gene_names = self.adata.var.index[self._visible_gene_mask].tolist()
        return {
            'active': True,
            'keep_columns': list(self._gene_mask_config.get('keep_columns', [])),
            'hide_columns': list(self._gene_mask_config.get('hide_columns', [])),
            'keep_combine_mode': self._gene_mask_config.get('keep_combine_mode', 'or'),
            'n_visible': n_visible,
            'n_total': n_total,
            'visible_gene_names': visible_gene_names,
        }

    def set_gene_mask(
        self,
        keep_columns: list[str],
        hide_columns: list[str],
        keep_combine_mode: str = 'or',
    ) -> dict[str, Any]:
        """Apply a gene mask.

        - Validates all referenced columns exist and are bool-like.
        - Empty keep_columns + empty hide_columns clears the mask.
        - Raises ValueError if the resulting mask leaves 0 visible genes.

        Returns the same shape as get_gene_mask().
        """
        if keep_combine_mode not in ('or', 'and'):
            raise ValueError(f"keep_combine_mode must be 'or' or 'and', got {keep_combine_mode!r}")

        # Empty config = clear
        if not keep_columns and not hide_columns:
            return self.clear_gene_mask()

        # Validate columns (raises ValueError if any are missing/non-bool)
        for c in keep_columns:
            self._column_to_bool_array(c)
        for c in hide_columns:
            self._column_to_bool_array(c)

        mask = self._compute_visible_mask(keep_columns, hide_columns, keep_combine_mode)
        if mask.sum() == 0:
            raise ValueError("Gene mask would leave 0 visible genes")

        self._gene_mask_config = {
            'keep_columns': list(keep_columns),
            'hide_columns': list(hide_columns),
            'keep_combine_mode': keep_combine_mode,
        }
        self._visible_gene_mask = mask
        return self.get_gene_mask()

    def clear_gene_mask(self) -> dict[str, Any]:
        """Clear the gene mask state."""
        self._gene_mask_config = None
        self._visible_gene_mask = None
        return self.get_gene_mask()

    def _regenerate_gene_mask_after_var_change(self) -> bool:
        """Rebuild _visible_gene_mask against the current .var axis.

        Called after operations that drop genes from .var or change the
        gene index. Behaviour:
          - No active mask → returns False, no-op.
          - All referenced columns still exist → recomputes the mask
            in place; returns False.
          - One or more referenced columns are gone → clears the mask;
            returns True.
        """
        if self._gene_mask_config is None:
            return False
        cfg = self._gene_mask_config
        try:
            self._visible_gene_mask = self._compute_visible_mask(
                keep_columns=cfg['keep_columns'],
                hide_columns=cfg['hide_columns'],
                keep_combine_mode=cfg['keep_combine_mode'],
            )
            # If everything got filtered out, clear rather than crash.
            if self._visible_gene_mask.sum() == 0:
                self.clear_gene_mask()
                return True
            return False
        except ValueError:
            # A referenced column no longer exists — clear the mask.
            self.clear_gene_mask()
            return True

    def get_visible_gene_names(self) -> list[str]:
        """Return gene names where _visible_gene_mask is True.

        Returns all gene names when no mask is active.
        """
        if self._visible_gene_mask is None:
            return self.adata.var.index.tolist()
        return self.adata.var.index[self._visible_gene_mask].tolist()

    def _filter_to_visible(self, genes: list[str]) -> tuple[list[str], int]:
        """Split a gene list into (visible_genes, n_excluded).

        When no mask is active, returns (genes, 0) unchanged.
        """
        if self._visible_gene_mask is None:
            return list(genes), 0
        visible_set = set(self.adata.var.index[self._visible_gene_mask].tolist())
        kept = [g for g in genes if g in visible_set]
        return kept, len(genes) - len(kept)

    def get_var_boolean_columns(self) -> list[dict[str, Any]]:
        """Get list of boolean columns in .var that can be used for gene filtering.

        Returns:
            List of dicts with column name, description, and count of True values
        """
        bool_columns = []
        for col in self.adata.var.columns:
            # Check if column is boolean or can be treated as boolean
            dtype = self.adata.var[col].dtype
            if dtype == bool or (dtype == 'bool'):
                n_true = self.adata.var[col].sum()
                bool_columns.append({
                    'name': col,
                    'n_true': int(n_true),
                    'n_total': self.n_genes,
                })
            # Also check for columns that look boolean (0/1 or True/False)
            elif pd.api.types.is_numeric_dtype(dtype):
                unique_vals = self.adata.var[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                    n_true = int((self.adata.var[col] == 1).sum() | (self.adata.var[col] == True).sum())
                    bool_columns.append({
                        'name': col,
                        'n_true': n_true,
                        'n_total': self.n_genes,
                    })
        return bool_columns

    def column_to_gene_names(self, column: str) -> list[str]:
        """Gene names where a boolean .var column is True (in .var order).

        Raises ValueError if the column is absent or not boolean-like (the
        allow-list is exactly what get_var_boolean_columns reports).
        """
        valid = {c['name'] for c in self.get_var_boolean_columns()}
        if column not in valid:
            raise ValueError(f"'{column}' is not a boolean .var column")
        mask = self._column_to_bool_array(column)
        return self.adata.var_names[mask].tolist()

    #: Missing-gene lists reported by gene_set_overlap are truncated here so a
    #: 2,000-gene library set does not ship 2,000 names per row of a table.
    MAX_REPORTED_OVERLAP_MISSING = 100

    def gene_set_overlap(self, sets: list[dict[str, Any]],
                         columns: list[str] | None = None) -> dict[str, Any]:
        """How much of each given gene set is in this dataset, and where.

        Names match exactly first, then case-insensitively (a human-symbol
        library against mouse data: ``COL1A1`` -> ``Col1a1``), and the resolved
        lists carry the dataset's own spelling so an import never has to guess
        again. ``columns`` are boolean ``.var`` columns (``highly_variable``,
        ``spatially_variable``…); each set reports how many of its *present*
        members fall in each. Directional sets keep ``genesDown`` separate.
        Pure read — nothing is written.
        """
        var_names = [str(g) for g in self.adata.var_names]
        exact = set(var_names)
        upper_to_var: dict[str, str] = {}
        for g in var_names:
            upper_to_var.setdefault(g.upper(), g)

        columns = [str(c) for c in (columns or [])]
        valid = {c['name'] for c in self.get_var_boolean_columns()}
        column_members: dict[str, set[str]] = {}
        for col in columns:
            if col not in valid:
                raise ValueError(f"'{col}' is not a boolean .var column")
            column_members[col] = set(self.column_to_gene_names(col))

        def resolve(genes: Any) -> tuple[list[str], list[str], int, int]:
            resolved: list[str] = []
            missing: list[str] = []
            seen: set[str] = set()
            n_exact = n_ci = 0
            for raw in genes if isinstance(genes, list) else []:
                g = str(raw).strip()
                if not g:
                    continue
                if g in exact:
                    hit, n_exact = g, n_exact + 1
                else:
                    hit = upper_to_var.get(g.upper())
                    if hit is None:
                        missing.append(g)
                        continue
                    n_ci += 1
                if hit not in seen:
                    seen.add(hit)
                    resolved.append(hit)
            return resolved, missing, n_exact, n_ci

        out_sets: list[dict[str, Any]] = []
        for s in sets:
            up_raw = s.get('genes') if isinstance(s, dict) else None
            down_raw = s.get('genesDown') if isinstance(s, dict) else None
            up, miss_up, e1, c1 = resolve(up_raw)
            down, miss_dn, e2, c2 = resolve(down_raw)
            present = up + down
            missing = miss_up + miss_dn
            n_genes = len(up_raw or []) + len(down_raw or [])
            out_sets.append({
                'name': str(s.get('name', '')) if isinstance(s, dict) else '',
                'n_genes': n_genes,
                'n_present': len(present),
                'n_exact': e1 + e2,
                'n_case_insensitive': c1 + c2,
                'n_missing': len(missing),
                'genes_resolved': up,
                'genes_down_resolved': down,
                'genes_missing': missing[:self.MAX_REPORTED_OVERLAP_MISSING],
                'columns': {col: sum(1 for g in present if g in column_members[col]) for col in columns},
            })
        return {'sets': out_sets, 'n_genes_dataset': len(var_names), 'columns': columns}

    # ------------------------------------------------------------------
    # Gene-set enrichment (overlap + preranked GSEA)
    # ------------------------------------------------------------------
    ENRICHMENT_UNS_KEY = 'xcell_enrichment'

    def _enrichment_sets(self, libraries: list[dict[str, Any]] | None,
                         sets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Cached libraries plus inline sets, each tagged with a library label."""
        from xcell import gene_set_sources as gss  # noqa: PLC0415
        out: list[dict[str, Any]] = []
        for ref in libraries or []:
            source = str(ref.get('source') or '')
            lib_id = str(ref.get('id') or '')
            lib = gss.find_library(source, lib_id, ref.get('species'))
            if lib is None:
                raise ValueError(
                    f"Library '{lib_id}' from {source or '?'} is not cached; "
                    "fetch it from the Gene set library first")
            label = str(lib.get('name') or lib_id)
            for s in lib.get('sets') or []:
                if isinstance(s, dict):
                    out.append({
                        'name': s.get('name', ''), 'genes': list(s.get('genes') or []),
                        'genes_down': list(s.get('genes_down') or []), 'library': label,
                        'description': s.get('description', ''), 'url': s.get('url', ''),
                    })
        for s in sets or []:
            if isinstance(s, dict):
                out.append({
                    'name': s.get('name', ''), 'genes': list(s.get('genes') or []),
                    'genes_down': list(s.get('genes_down') or s.get('genesDown') or []),
                    'library': 'My gene sets', 'description': '', 'url': '',
                })
        if not out:
            raise ValueError('No gene sets to test: choose at least one library or gene set')
        return out

    def _enrichment_store(self) -> dict[str, str]:
        raw = self.adata.uns.get(self.ENRICHMENT_UNS_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def _store_enrichment(self, key_hint: str, result: dict[str, Any]) -> str:
        """Persist as JSON under a never-colliding key; returns the key used.

        JSON strings, not nested dicts: a result is a list of records and
        h5ad cannot write an object array of dicts (same reason drawn lines
        and territories are stored this way).
        """
        base = self._sanitize_subset_name(key_hint)
        store = self._enrichment_store()
        key, n = base, 1
        while key in store:
            n += 1
            key = f'{base}_{n}'
        result['key'] = key
        store[key] = json.dumps(result)
        self.adata.uns[self.ENRICHMENT_UNS_KEY] = store
        return key

    def get_enrichment_results(self) -> list[dict[str, Any]]:
        out = []
        for raw in self._enrichment_store().values():
            try:
                r = json.loads(raw)
            except (TypeError, ValueError):
                continue
            out.append({k: r.get(k) for k in
                        ('key', 'kind', 'label', 'n_sets_tested', 'n_significant', 'created_at')})
        out.sort(key=lambda s: s.get('created_at') or '', reverse=True)
        return out

    def get_enrichment_result(self, key: str) -> dict[str, Any]:
        store = self._enrichment_store()
        if key not in store:
            raise KeyError(f"No enrichment result named '{key}'")
        return json.loads(store[key])

    def delete_enrichment_result(self, key: str) -> dict[str, Any]:
        store = self._enrichment_store()
        if key not in store:
            raise KeyError(f"No enrichment result named '{key}'")
        del store[key]
        self.adata.uns[self.ENRICHMENT_UNS_KEY] = store
        self._log_action('enrichment_delete', {'key': key}, {'deleted': key})
        return {'deleted': key}

    def run_overlap_enrichment(self, genes: list[str], *, name: str | None = None,
                               libraries: list[dict[str, Any]] | None = None,
                               sets: list[dict[str, Any]] | None = None,
                               gene_subset: Any = None, min_set_size: int = 5,
                               max_set_size: int = 500, min_overlap: int = 2,
                               key: str | None = None) -> dict[str, Any]:
        """Hypergeometric over-representation of ``genes`` in each library set.

        The universe is the dataset's genes after the session gene mask and
        ``gene_subset`` — the same genes the Gene Panel shows.
        """
        from datetime import datetime, timezone  # noqa: PLC0415
        from xcell import enrichment as en  # noqa: PLC0415
        if min_set_size < 1 or max_set_size < min_set_size:
            raise ValueError('Set size range must satisfy 1 <= min <= max')
        raw_sets = self._enrichment_sets(libraries, sets)
        mask, subset_type, _meta = self._resolve_gene_mask(gene_subset)
        universe = [str(g) for g in self.adata.var_names[mask]]
        q_idx, missing = en.resolve_symbols(list(genes or []), universe)
        if len(q_idx) < 2:
            raise ValueError(
                f'Need at least 2 query genes present in the universe; {len(q_idx)} of '
                f'{len(genes or [])} resolved')
        resolved, rmeta = en.resolve_sets(raw_sets, universe, min_size=min_set_size,
                                          max_size=max_set_size, directional='union')
        if not resolved:
            raise ValueError(
                f'No gene set has between {min_set_size} and {max_set_size} members in the universe')
        records = en.overlap_enrichment(q_idx, resolved, len(universe), min_overlap=min_overlap)
        for r in records:
            r['genes'] = [universe[i] for i in r['genes']]
        label = name or 'gene list'
        params = {
            'genes': list(genes or []), 'name': label, 'libraries': list(libraries or []),
            'sets': list(sets or []), 'gene_subset': gene_subset,
            'min_set_size': min_set_size, 'max_set_size': max_set_size, 'min_overlap': min_overlap,
        }
        result = {
            'kind': 'ora', 'label': f'Overlap: {label}',
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'query': {'name': label, 'n_input': len(genes or []), 'n_in_universe': len(q_idx),
                      'genes_missing': missing[:self.MAX_REPORTED_OVERLAP_MISSING]},
            'universe_size': len(universe), 'gene_subset_type': subset_type,
            'n_sets_input': rmeta['n_input'], 'n_sets_tested': len(records),
            'n_significant': sum(1 for r in records if r['padj'] <= 0.05),
            'results': records, 'params': params,
        }
        stored_key = self._store_enrichment(key or f'ora_{label}', result)
        self._log_action('enrichment_ora', params, {
            'key': stored_key, 'n_sets_tested': len(records), 'n_significant': result['n_significant']})
        return result

    def _gsea_ranking_snapshot(self, ranking: dict[str, Any], universe: list[str],
                               universe_mask: np.ndarray
                               ) -> tuple[dict[str, Any], Callable[[], np.ndarray], str, str]:
        """Validate a ranking spec now; return (clean spec, builder, label, key hint).

        The builder runs inside the background task and returns one score per
        universe gene (NaN = unranked). Everything it needs is snapshotted here
        so a later mutation of the live AnnData cannot leak in.
        """
        from xcell import enrichment as en  # noqa: PLC0415
        kind = str(ranking.get('kind') or '')
        cell_subset = ranking.get('cell_subset') or None
        cell_mask = self._subset_mask(cell_subset) if cell_subset else None   # KeyError if unknown
        suffix = f' [{cell_subset}]' if cell_subset else ''
        excluded = '\x00excluded'

        if kind == 'diffexp':
            col = str(ranking.get('obs_column') or '')
            if col not in self.adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in .obs")
            values = self.adata.obs[col].astype(str).values
            if cell_mask is not None:
                values = np.where(cell_mask, values, excluded)
            group = str(ranking.get('group') or '')
            reference = str(ranking.get('reference') or 'rest')
            method = str(ranking.get('method') or 'wilcoxon')
            metric = str(ranking.get('metric') or 'score')
            present = set(values[values != excluded])
            if group not in present:
                raise ValueError(f"Group '{group}' not found in column '{col}'" + (f" within subset '{cell_subset}'" if cell_subset else ''))
            if reference == group:
                raise ValueError('reference must differ from group')
            if reference != 'rest' and reference not in present:
                raise ValueError(f"Reference group '{reference}' not found in column '{col}'")
            if method not in ('wilcoxon', 't-test'):
                raise ValueError("method must be 'wilcoxon' or 't-test'")
            if metric not in ('score', 'log2fc'):
                raise ValueError("metric must be 'score' or 'log2fc'")
            in_group = values == group
            in_ref = (values != group) & (values != excluded) if reference == 'rest' else values == reference
            if in_group.sum() < 2 or in_ref.sum() < 2:
                raise ValueError('Each side of the contrast needs at least 2 cells')
            cells = np.flatnonzero(in_group | in_ref)
            labels = np.where(in_group[cells], 'group', 'reference')
            X = self.adata.X[cells][:, universe_mask]
            X = X.copy() if hasattr(X, 'copy') else np.array(X)

            def build() -> np.ndarray:
                import anndata as _ad  # noqa: PLC0415
                import scanpy as sc  # noqa: PLC0415
                tmp = _ad.AnnData(X=X)
                tmp.var_names = universe
                tmp.obs['g'] = pd.Categorical(labels, categories=['group', 'reference'])
                sc.tl.rank_genes_groups(tmp, groupby='g', groups=['group'], reference='reference',
                                        method=method, use_raw=False, key_added='r')
                r = tmp.uns['r']
                names = [str(g) for g in r['names']['group']]
                field = 'scores' if metric == 'score' else 'logfoldchanges'
                vals = np.asarray(r[field]['group'], dtype=float)
                pos = {g: i for i, g in enumerate(universe)}
                out = np.full(len(universe), np.nan)
                for g, v in zip(names, vals):
                    out[pos[g]] = v
                return out

            clean = {'kind': 'diffexp', 'obs_column': col, 'group': group, 'reference': reference,
                     'method': method, 'metric': metric, 'cell_subset': cell_subset}
            return clean, build, f'{col}: {group} vs {reference}{suffix}', f'gsea_{col}_{group}_vs_{reference}'

        if kind == 'pca':
            comp = ranking.get('component')
            pcs_key = f'PCs_{cell_subset}' if cell_subset else 'PCs'
            if pcs_key not in self.adata.varm:
                raise ValueError(f"No PCA loadings ('{pcs_key}' missing from .varm); run PCA first")
            pcs = np.asarray(self.adata.varm[pcs_key], dtype=float)
            if not isinstance(comp, int) or isinstance(comp, bool) or comp < 0 or comp >= pcs.shape[1]:
                raise ValueError(f'component must be an integer in [0, {pcs.shape[1] - 1}]')
            loading = pcs[universe_mask, comp].copy()
            loading[loading == 0] = np.nan   # genes outside the PCA's gene mask carry no loading
            clean = {'kind': 'pca', 'component': int(comp), 'cell_subset': cell_subset}
            return (clean, (lambda: loading), f'PC{comp + 1} loading{suffix}',
                    f'gsea_pca{comp + 1}' + (f'_{cell_subset}' if cell_subset else ''))

        if kind == 'scores':
            genes = list(ranking.get('genes') or [])
            scores = list(ranking.get('scores') or [])
            if len(genes) != len(scores) or not genes:
                raise ValueError('genes and scores must be non-empty lists of the same length')
            arr = np.full(len(universe), np.nan)
            exact, upper = en._symbol_lookup(universe)
            for g, v in zip(genes, scores):   # first occurrence of a symbol wins
                sym = str(g).strip()
                i = exact.get(sym)
                if i is None:
                    i = upper.get(sym.upper())
                if i is not None and np.isnan(arr[i]):
                    arr[i] = float(v)
            if np.isfinite(arr).sum() < 2:
                raise ValueError('Fewer than 2 of the scored genes are in the universe')
            clean = {'kind': 'scores', 'n_genes': len(genes), 'cell_subset': None}
            return clean, (lambda: arr), 'custom scores', 'gsea_scores'

        raise ValueError("ranking kind must be 'diffexp', 'pca' or 'scores'")

    def prepare_gsea(self, ranking: dict[str, Any], *, libraries: list[dict[str, Any]] | None = None,
                     sets: list[dict[str, Any]] | None = None, gene_subset: Any = None,
                     n_perm: int = 1000, min_set_size: int = 15, max_set_size: int = 500,
                     weight: float = 1.0, seed: int = 0, key: str | None = None):
        """Preranked GSEA as a background task: (compute_fn(report), apply_fn(result))."""
        from datetime import datetime, timezone  # noqa: PLC0415
        from xcell import enrichment as en  # noqa: PLC0415
        if n_perm < 10:
            raise ValueError('n_perm must be at least 10')
        if min_set_size < 1 or max_set_size < min_set_size:
            raise ValueError('Set size range must satisfy 1 <= min <= max')
        raw_sets = self._enrichment_sets(libraries, sets)
        mask, subset_type, _meta = self._resolve_gene_mask(gene_subset)
        universe = [str(g) for g in self.adata.var_names[mask]]
        clean_ranking, build, label, key_hint = self._gsea_ranking_snapshot(dict(ranking), universe, mask)
        resolved, rmeta = en.resolve_sets(raw_sets, universe, min_size=min_set_size,
                                          max_size=max_set_size, directional='split')
        if not resolved:
            raise ValueError(
                f'No gene set has between {min_set_size} and {max_set_size} members in the universe')
        params = {
            'ranking': clean_ranking, 'libraries': list(libraries or []), 'sets': list(sets or []),
            'gene_subset': gene_subset, 'n_perm': int(n_perm), 'min_set_size': int(min_set_size),
            'max_set_size': int(max_set_size), 'weight': float(weight), 'seed': int(seed),
        }

        def compute_fn(report):
            report(0.02, 'Building the ranking…')
            scores = build()
            report(0.15, 'Running GSEA…')
            out = en.preranked_gsea(
                scores, resolved, n_perm=n_perm, min_size=min_set_size, max_size=max_set_size,
                weight=weight, seed=seed,
                report=lambda f, m: report(0.15 + 0.85 * f, m))
            out['scores'] = scores
            return out

        def apply_fn(out):
            order = np.asarray(out['order'])
            scores = np.asarray(out['scores'])
            for r in out['results']:
                r['leading_edge'] = [universe[i] for i in r['leading_edge']]
            result = {
                'kind': 'gsea', 'label': f'GSEA: {label}',
                'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'ranking': {'kind': clean_ranking['kind'], 'label': label, 'n_ranked': int(out['n_ranked']),
                            'genes': [universe[i] for i in order],
                            'scores': [float(scores[i]) for i in order]},
                'universe_size': len(universe), 'gene_subset_type': subset_type,
                'n_sets_input': rmeta['n_input'], 'n_sets_tested': int(out['n_sets_tested']),
                'n_significant': sum(1 for r in out['results'] if r['padj'] <= 0.05),
                'n_perm': int(out['n_perm']), 'results': out['results'], 'params': params,
            }
            stored_key = self._store_enrichment(key or key_hint, result)
            self._log_action('enrichment_gsea', params, {
                'key': stored_key, 'n_sets_tested': result['n_sets_tested'],
                'n_significant': result['n_significant']})
            return result

        return compute_fn, apply_fn

    def guess_species(self) -> dict[str, Any]:
        """Species guess from the current var index (Ensembl prefix, else symbol case)."""
        from xcell import gene_symbols as gs  # noqa: PLC0415

        out = gs.guess_species([str(g) for g in self.adata.var_names])
        out['n_genes'] = int(self.n_genes)
        return out

    def _resolve_source_matrix(self, layer: str | None):
        """Return the (n_cells, n_genes) expression matrix to read from.

        ``layer is None`` or ``'X'`` → ``adata.X``. Otherwise reads the named
        layer; raises ValueError if the layer doesn't exist. Used by gene-side
        analyses that support an optional ``layer=`` override (for routing
        through e.g. a kNN-smoothed layer produced by ``run_smooth``).
        """
        if layer is None or layer == 'X':
            return self.adata.X
        if layer not in self.adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. "
                f"Available: ['X'] + {sorted(self.adata.layers.keys())}"
            )
        return self.adata.layers[layer]

    # A pathological request (an entire var index of names from another
    # species) shouldn't put tens of thousands of strings on the wire; the
    # exact count is always n_requested - n_genes.
    MAX_REPORTED_MISSING_GENES = 100

    @classmethod
    def _gene_subset_summary(
        cls,
        subset_type: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Describe a resolved gene subset for an API response.

        Always the same five keys, whatever kind of subset it was, so the UI
        renders one shape instead of branching. ``n_requested`` is None unless
        the subset was an explicit gene list; ``n_hidden_by_gene_mask`` is 0
        unless the active gene mask narrowed the selection.
        """
        missing = metadata.get('genes_missing') or []
        return {
            'type': subset_type,
            'n_genes': int(metadata.get('n_genes', 0)),
            'n_requested': metadata.get('genes_requested'),
            'genes_missing': list(missing[:cls.MAX_REPORTED_MISSING_GENES]),
            'n_hidden_by_gene_mask': int(metadata.get('n_hidden_by_gene_mask', 0)),
        }

    def _resolve_gene_mask(
        self,
        gene_subset: str | list[str] | dict[str, Any] | None,
        *,
        apply_visible_mask: bool = True,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Resolve a gene_subset specification into a boolean mask.

        The result is intersected with the active ``.var`` gene mask, so an
        operation that reports genes works in the universe the Gene Panel
        shows. Callers that build cell-space structure (PCA and the rest of
        the scanpy chain), rebuild the cells (spot merging), or hand genes to
        another dataset (the Localize reference) pass
        ``apply_visible_mask=False``: a session-only view must not silently
        change an embedding that gets written into the file.

        Args:
            gene_subset: Gene subset specification. Can be:
                - None: all genes
                - str: single boolean column name from .var (e.g., 'highly_variable')
                - list[str]: explicit list of gene names
                - dict: {'columns': [...], 'operation': 'intersection'|'union'}
            apply_visible_mask: Intersect with the active gene mask (default).

        Returns:
            Tuple of (boolean mask, subset_type string, metadata dict). When
            the gene mask removed anything, the metadata carries
            ``n_hidden_by_gene_mask`` and ``n_genes`` counts what survived.

        Raises:
            ValueError: If the subset is empty, or the gene mask hides all of it.
        """
        mask, subset_type, metadata = self._resolve_gene_subset_spec(gene_subset)
        if not apply_visible_mask or self._visible_gene_mask is None:
            return mask, subset_type, metadata

        hidden = int(np.sum(mask & ~self._visible_gene_mask))
        if hidden == 0:
            return mask, subset_type, metadata

        visible = mask & self._visible_gene_mask
        if not visible.any():
            raise ValueError(
                "The active gene mask hides every gene in this selection. "
                "Clear the gene mask, or widen it, and run again."
            )
        metadata = {
            **metadata,
            'n_genes': int(visible.sum()),
            'n_hidden_by_gene_mask': hidden,
        }
        return visible, subset_type, metadata

    def _resolve_gene_subset_spec(
        self,
        gene_subset: str | list[str] | dict[str, Any] | None,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """The gene_subset spec on its own, before the gene mask narrows it."""
        if gene_subset is None:
            return (
                np.ones(self.n_genes, dtype=bool),
                'all',
                {'n_genes': self.n_genes},
            )

        def _from_gene_names(names: list[str]) -> tuple[np.ndarray, str, dict[str, Any]]:
            """Mask from an explicit gene list, recording what was not found.

            Curated sets routinely name genes this dataset lacks (different
            species, symbol vintage, prior filtering). Dropping them quietly
            changes how the result should be read, so the names come back with
            the mask and callers can surface them.
            """
            mask = self.adata.var_names.isin(names)
            if mask.sum() == 0:
                raise ValueError("None of the specified genes found in dataset")
            present = set(self.adata.var_names)
            return (
                mask,
                'gene_list',
                {
                    'n_genes': int(mask.sum()),
                    'genes_requested': len(names),
                    'genes_missing': [g for g in names if g not in present],
                },
            )

        # An empty list reaches here from a UI combination that resolved to
        # nothing (an intersection of disjoint sets, say). Without this it
        # falls through to the catch-all below and reports a type error.
        if isinstance(gene_subset, list) and len(gene_subset) == 0:
            raise ValueError("Empty gene list — no genes to test.")

        # Single column name
        if isinstance(gene_subset, str):
            if gene_subset not in self.adata.var.columns:
                raise ValueError(f"Column '{gene_subset}' not found in .var")
            col_values = self.adata.var[gene_subset]
            # Convert to boolean mask
            if col_values.dtype == bool:
                mask = col_values.values
            else:
                mask = (col_values == 1) | (col_values == True)
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                raise ValueError(f"No genes found with {gene_subset}=True")
            return (
                mask,
                f'column:{gene_subset}',
                {'column': gene_subset, 'n_genes': int(mask.sum())},
            )

        # Explicit list of gene names
        if isinstance(gene_subset, list) and len(gene_subset) > 0 and isinstance(gene_subset[0], str):
            # A list of strings is ambiguous: gene names, or .var column names to
            # combine? Only a list where *every* entry names a column is read as
            # columns — anything else is gene names, however partial the overlap.
            all_columns = all(c in self.adata.var.columns for c in gene_subset)

            if all_columns:
                # Treat as column list with intersection
                gene_subset = {'columns': gene_subset, 'operation': 'intersection'}
            else:
                # Gene names — wholly or partly present, either way a gene list.
                return _from_gene_names(gene_subset)

        # Dict with columns and operation
        if isinstance(gene_subset, dict):
            columns = gene_subset.get('columns', [])
            operation = gene_subset.get('operation', 'intersection')

            if not columns:
                raise ValueError("No columns specified in gene_subset")

            if operation not in ('intersection', 'union'):
                raise ValueError("operation must be 'intersection' or 'union'")

            # Validate columns exist
            missing = [c for c in columns if c not in self.adata.var.columns]
            if missing:
                raise ValueError(f"Columns not found in .var: {missing}")

            # Build combined mask
            masks = []
            for col in columns:
                col_values = self.adata.var[col]
                if col_values.dtype == bool:
                    m = col_values.values
                else:
                    m = (col_values == 1) | (col_values == True)
                masks.append(np.asarray(m, dtype=bool))

            if operation == 'intersection':
                combined_mask = np.all(masks, axis=0)
                op_symbol = 'AND'
            else:  # union
                combined_mask = np.any(masks, axis=0)
                op_symbol = 'OR'

            if combined_mask.sum() == 0:
                raise ValueError(f"No genes found matching {op_symbol} of {columns}")

            return (
                combined_mask,
                f'{operation}:{"+".join(columns)}',
                {
                    'columns': columns,
                    'operation': operation,
                    'n_genes': int(combined_mask.sum()),
                    'individual_counts': {c: int(m.sum()) for c, m in zip(columns, masks)},
                },
            )

        raise ValueError(
            "gene_subset must be None, a column name (str), a list of gene names, "
            "or a dict with 'columns' and 'operation'"
        )

    def run_filter_genes(
        self,
        min_counts: int | None = None,
        max_counts: int | None = None,
        min_cells: int | None = None,
        max_cells: int | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Filter genes based on counts or number of cells expressing.

        Args:
            min_counts: Minimum total counts for a gene
            max_counts: Maximum total counts for a gene
            min_cells: Minimum number of cells expressing the gene
            max_cells: Maximum number of cells expressing the gene
            active_cell_indices: If provided, compute gene stats using only these cells

        Returns:
            Dict with before/after gene counts
        """
        n_genes_before = self.n_genes

        # Build kwargs for scanpy
        kwargs = {}
        if min_counts is not None:
            kwargs['min_counts'] = min_counts
        if max_counts is not None:
            kwargs['max_counts'] = max_counts
        if min_cells is not None:
            kwargs['min_cells'] = min_cells
        if max_cells is not None:
            kwargs['max_cells'] = max_cells

        indices = None  # also read by _log_action below
        if kwargs:
            adata_sub, indices = self._get_active_adata(active_cell_indices)
            if indices is not None:
                # Run filter on subset to find surviving genes
                # (adata_sub is already a copy from _get_active_adata)
                sc.pp.filter_genes(adata_sub, **kwargs)
                surviving = set(adata_sub.var_names)
                self.adata = self.adata[:, self.adata.var_names.isin(surviving)].copy()
            else:
                sc.pp.filter_genes(self.adata, **kwargs)

        n_genes_after = self.n_genes

        # Invalidate normalized cache since data changed
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        mask_cleared = self._regenerate_gene_mask_after_var_change()

        result = {
            'n_genes_before': n_genes_before,
            'n_genes_after': n_genes_after,
            'n_genes_removed': n_genes_before - n_genes_after,
            'gene_mask_cleared': mask_cleared,
        }
        self._log_action('filter_genes', kwargs, result, subset=indices)
        return result

    def run_exclude_genes(
        self,
        gene_names: list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Remove genes by exact name or regex pattern.

        Args:
            gene_names: List of gene names to remove (exact match)
            patterns: List of regex patterns to match against gene names
                (e.g. "^mt-" for mitochondrial, "^Gm\\d+" for predicted genes)

        Returns:
            Dict with before/after gene counts and removed gene names
        """
        import re

        n_genes_before = self.n_genes
        names = self.adata.var_names

        mask = np.zeros(len(names), dtype=bool)

        # Exact name matches
        if gene_names:
            name_set = set(gene_names)
            mask |= names.isin(name_set)

        # Pattern matches
        if patterns:
            for pattern in patterns:
                try:
                    mask |= names.str.match(pattern)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        removed_genes = names[mask].tolist()
        n_removed = int(mask.sum())

        if n_removed > 0:
            self.adata = self.adata[:, ~mask].copy()
            self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        mask_cleared = self._regenerate_gene_mask_after_var_change()

        result = {
            'n_genes_before': n_genes_before,
            'n_genes_after': self.n_genes,
            'n_genes_removed': n_removed,
            'removed_genes': removed_genes[:100],  # Cap list for large removals
            'removed_genes_total': n_removed,
            'gene_mask_cleared': mask_cleared,
        }
        self._log_action('exclude_genes', {
            'gene_names': gene_names,
            'patterns': patterns,
        }, result)
        return result

    def _filter_cells_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell total counts and expressed-gene counts, from .X.

        Same quantities ``sc.pp.filter_cells`` thresholds on: the sum of a
        cell's row, and how many of its entries are > 0.
        """
        X = self.adata.X
        counts = np.ravel(np.asarray(X.sum(axis=1))).astype(np.float64)
        genes = np.ravel(np.asarray((X > 0).sum(axis=1))).astype(np.int64)
        return counts, genes

    def filter_cells_qc(self) -> dict[str, Any]:
        """The distributions Filter Cells thresholds against, per cell.

        Lets the UI draw counts/genes histograms before anything is removed.
        Values are plain python numbers; a non-finite sum (pathological .X)
        becomes None rather than NaN, which is not JSON.
        """
        counts, genes = self._filter_cells_metrics()
        return {
            'counts': [float(v) if np.isfinite(v) else None for v in counts],
            'genes': [int(g) for g in genes],
        }

    def run_filter_cells(
        self,
        min_counts: int | None = None,
        max_counts: int | None = None,
        min_genes: int | None = None,
        max_genes: int | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Filter cells based on counts or number of genes expressed.

        All given thresholds apply together (a cell must satisfy every one),
        with scanpy's inclusive semantics. Not delegated to
        ``sc.pp.filter_cells``, which accepts exactly one threshold per call;
        the mask below is what its sequential calls would compute.

        Args:
            min_counts: Minimum total counts for a cell
            max_counts: Maximum total counts for a cell
            min_genes: Minimum number of genes expressed in the cell
            max_genes: Maximum number of genes expressed in the cell
            active_cell_indices: If provided, only evaluate these cells for filtering

        Returns:
            Dict with before/after cell counts
        """
        n_cells_before = self.n_cells

        kwargs = {}
        if min_counts is not None:
            kwargs['min_counts'] = min_counts
        if max_counts is not None:
            kwargs['max_counts'] = max_counts
        if min_genes is not None:
            kwargs['min_genes'] = min_genes
        if max_genes is not None:
            kwargs['max_genes'] = max_genes

        indices = None  # also read by _log_action below
        if kwargs:
            indices = self._validate_cell_indices(active_cell_indices)
            counts, genes = self._filter_cells_metrics()
            keep = np.ones(self.n_cells, dtype=bool)
            if min_counts is not None:
                keep &= counts >= min_counts
            if max_counts is not None:
                keep &= counts <= max_counts
            if min_genes is not None:
                keep &= genes >= min_genes
            if max_genes is not None:
                keep &= genes <= max_genes
            if indices is not None:
                # Cells outside the active set are not evaluated and survive.
                mask = np.ones(self.n_cells, dtype=bool)
                mask[indices] = keep[indices]
                keep = mask
            if not keep.all():
                self.adata = self.adata[keep].copy()

        n_cells_after = self.n_cells

        # Invalidate normalized cache since data changed
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        result = {
            'n_cells_before': n_cells_before,
            'n_cells_after': n_cells_after,
            'n_cells_removed': n_cells_before - n_cells_after,
        }
        self._log_action('filter_cells', kwargs, result, subset=indices)
        return result

    def delete_cells(
        self,
        cell_indices: list[int],
    ) -> dict[str, Any]:
        """Permanently remove specific cells from the dataset.

        Args:
            cell_indices: List of cell indices to remove.

        Returns:
            Dict with before/after cell counts.
        """
        if not cell_indices:
            raise ValueError("No cell indices provided.")

        indices = np.array(cell_indices)
        if indices.max() >= self.n_cells or indices.min() < 0:
            raise ValueError(
                f"Cell indices out of range [0, {self.n_cells - 1}]."
            )

        n_cells_before = self.n_cells

        # Build keep mask (True for cells to keep)
        keep_mask = np.ones(self.n_cells, dtype=bool)
        keep_mask[indices] = False

        self.adata = self.adata[keep_mask].copy()

        # Invalidate normalized cache
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        n_cells_after = self.n_cells

        result = {
            'n_cells_before': n_cells_before,
            'n_cells_after': n_cells_after,
            'n_cells_deleted': n_cells_before - n_cells_after,
        }
        self._log_action('delete_cells', {'n_indices': len(cell_indices)}, result,
                         subset=cell_indices)
        return result

    def run_normalize_total(
        self,
        target_sum: float | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Normalize total counts per cell.

        Args:
            target_sum: Target sum of counts per cell. If None, uses median.
            active_cell_indices: If provided, only normalize these cells

        Returns:
            Dict with operation status
        """
        from scipy import sparse

        kwargs = {}
        if target_sum is not None:
            kwargs['target_sum'] = target_sum

        adata_sub, indices = self._get_active_adata(active_cell_indices)
        if indices is not None:
            sc.pp.normalize_total(adata_sub, **kwargs)
            # Write back — normalize_total only scales rows, preserving sparsity
            if sparse.issparse(self.adata.X):
                csr = self.adata.X.tocsr()
                sub_csr = adata_sub.X.tocsr() if sparse.issparse(adata_sub.X) else sparse.csr_matrix(adata_sub.X)
                # Vectorized: flag data entries belonging to active rows
                row_flag = np.zeros(csr.shape[0], dtype=bool)
                row_flag[indices] = True
                entry_mask = np.repeat(row_flag, np.diff(csr.indptr))
                csr.data[entry_mask] = sub_csr.data
                self.adata.X = csr
            else:
                self.adata.X[indices] = adata_sub.X if not sparse.issparse(adata_sub.X) else adata_sub.X.toarray()
        else:
            sc.pp.normalize_total(self.adata, **kwargs)

        # Invalidate normalized cache
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        result = {'status': 'completed', 'target_sum': target_sum}
        self._log_action('normalize_total', kwargs, result, subset=indices)
        return result

    def run_log1p(
        self,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Apply log1p transformation to the data.

        Args:
            active_cell_indices: If provided, only transform these cells

        Returns:
            Dict with operation status
        """
        from scipy import sparse

        # Validate indices without copying (log1p operates in-place)
        indices = self._validate_cell_indices(active_cell_indices)
        if indices is not None:
            # In-place transform — log1p(0)=0 so sparsity is preserved
            if sparse.issparse(self.adata.X):
                csr = self.adata.X.tocsr()
                row_flag = np.zeros(csr.shape[0], dtype=bool)
                row_flag[indices] = True
                entry_mask = np.repeat(row_flag, np.diff(csr.indptr))
                np.log1p(csr.data[entry_mask], out=csr.data[entry_mask])
                self.adata.X = csr
            else:
                np.log1p(self.adata.X[indices], out=self.adata.X[indices])
        else:
            sc.pp.log1p(self.adata)

        # Invalidate normalized cache
        self._normalized_adata = None
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None

        result = {'status': 'completed'}
        self._log_action('log1p', {}, result, subset=indices)
        return result

    def _run_hvg_split(
        self, *, split_by, n_top_genes, min_mean, max_mean, min_disp, flavor,
        n_bins, active_cell_indices, add_union, add_intersection,
        min_cells_per_group,
    ) -> dict[str, Any]:
        """HVG detection within each group of ``split_by``. See the caller."""
        groups, skipped = self._prepare_hvg_groups(
            split_by, min_cells_per_group, active_cell_indices,
        )

        columns: list[str] = []
        per_group: dict[str, Any] = {}
        masks: list[np.ndarray] = []
        for name, rows in groups:
            mask = self._hvg_mask_for_cells(
                self.adata[rows].copy(), n_top_genes=n_top_genes,
                min_mean=min_mean, max_mean=max_mean, min_disp=min_disp,
                flavor=flavor, n_bins=n_bins,
            )
            col = f'highly_variable__{name}'
            self.adata.var[col] = mask
            columns.append(col)
            masks.append(mask)
            per_group[name] = {
                'n_cells': int(rows.size),
                'n_highly_variable': int(mask.sum()),
                'column': col,
            }

        stacked = np.vstack(masks)
        union, intersection = stacked.any(axis=0), stacked.all(axis=0)
        if add_union:
            self.adata.var['highly_variable__union'] = union
            columns.append('highly_variable__union')
        if add_intersection:
            self.adata.var['highly_variable__intersection'] = intersection
            columns.append('highly_variable__intersection')

        result = {
            'status': 'completed',
            'split_by': split_by,
            'groups': per_group,
            'columns': columns,
            'skipped': skipped,
            'n_union': int(union.sum()),
            'n_intersection': int(intersection.sum()),
            'n_total_genes': self.n_genes,
            'flavor': flavor,
        }
        self._log_action('highly_variable_genes', {
            'n_top_genes': n_top_genes,
            'min_mean': min_mean,
            'max_mean': max_mean,
            'min_disp': min_disp,
            'flavor': flavor,
            'subset': False,
            'split_by': split_by,
            'add_union': add_union,
            'add_intersection': add_intersection,
            'min_cells_per_group': min_cells_per_group,
        }, result, subset=active_cell_indices)
        return result

    def _hvg_mask_for_cells(
        self, adata_cells, *, n_top_genes, min_mean, max_mean, min_disp,
        flavor, n_bins,
    ) -> np.ndarray:
        """Boolean HVG mask over the *full* gene index, from a cell subset.

        Genes with no counts in the subset are dropped before scanpy sees them
        — they produce degenerate dispersion bins — and come back as not
        variable, which is the honest answer for a gene nobody expressed.
        """
        totals = np.asarray(adata_cells.X.sum(axis=0)).ravel()
        expressed = totals > 0
        sub = adata_cells[:, expressed].copy()
        sc.pp.highly_variable_genes(
            sub, n_top_genes=n_top_genes, min_mean=min_mean, max_mean=max_mean,
            min_disp=min_disp, flavor=flavor, n_bins=n_bins, subset=False,
        )
        mask = pd.Series(False, index=self.adata.var_names)
        mask.loc[sub.var_names] = sub.var['highly_variable'].values
        return mask.values.astype(bool)

    def _prepare_hvg_groups(
        self, split_by: str, min_cells_per_group: int, active_cell_indices,
    ):
        """(usable groups, skipped) for a split HVG run — validated eagerly."""
        if split_by not in self.adata.obs.columns:
            raise ValueError(f"Column '{split_by}' not found in .obs")
        series = self.adata.obs[split_by]
        if pd.api.types.is_float_dtype(series):
            raise ValueError(
                f"Column '{split_by}' is continuous; splitting HVG detection "
                "needs a categorical column naming samples, sections or donors"
            )
        labels = np.asarray(series.astype(str).values)
        mask = np.ones(self.n_cells, dtype=bool)
        if active_cell_indices is not None:
            mask = np.zeros(self.n_cells, dtype=bool)
            mask[np.asarray(active_cell_indices, dtype=np.int64)] = True

        if isinstance(series.dtype, pd.CategoricalDtype):
            order = [str(c) for c in series.cat.categories]
        else:
            _, first = np.unique(labels, return_index=True)
            order = [str(labels[i]) for i in sorted(first)]

        groups, skipped = [], []
        for name in order:
            rows = np.flatnonzero((labels == name) & mask)
            if rows.size == 0:
                continue          # a leftover category no cell uses
            if rows.size < min_cells_per_group:
                skipped.append({
                    'group': name, 'n_cells': int(rows.size),
                    'reason': f'fewer than {min_cells_per_group} cells',
                })
                continue
            groups.append((name, rows))
        if len(groups) < 2:
            raise ValueError(
                f"Column '{split_by}' yields {len(groups)} usable group(s); "
                f"splitting HVG detection needs at least 2 "
                f"(groups under {min_cells_per_group} cells are skipped)"
            )
        return groups, skipped

    def run_highly_variable_genes(
        self,
        n_top_genes: int | None = None,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
        flavor: str = 'seurat',
        n_bins: int = 20,
        subset: bool = False,
        active_cell_indices: list[int] | None = None,
        split_by: str | None = None,
        add_union: bool = False,
        add_intersection: bool = False,
        min_cells_per_group: int = 10,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Identify highly variable genes.

        Adds 'highly_variable' boolean column to .var. On a named
        ``cell_subset`` the result goes to ``highly_variable__<name>`` instead
        and the pooled column is left untouched, so sub-clustering a
        population does not change which genes the whole dataset's PCA uses.

        Args:
            n_top_genes: Number of top genes to select (overrides min/max thresholds)
            min_mean: Minimum mean expression threshold
            max_mean: Maximum mean expression threshold
            min_disp: Minimum dispersion threshold
            flavor: Method ('seurat', 'cell_ranger', 'seurat_v3')
            n_bins: Number of bins for dispersion normalization
            subset: If True, subset adata to only highly variable genes (destructive)
            active_cell_indices: If provided, compute HVGs using only these cells
            split_by: Categorical .obs column. When set, HVGs are detected
                *within* each group separately and written to
                ``highly_variable__<group>`` — pooled detection on a
                multi-sample dataset scores genes that vary *between* samples
                as readily as genes that vary within them, so a batch effect
                reads as biology. The pooled ``highly_variable`` column is left
                exactly as it was, since every gene_subset picker points at it.
            add_union: With split_by, also write ``highly_variable__union``
                (variable in any group).
            add_intersection: With split_by, also write
                ``highly_variable__intersection`` (variable in every group) —
                the conservative choice for multi-sample work.
            min_cells_per_group: Groups smaller than this are skipped and
                reported rather than contributing a noisy answer.

        Returns:
            Dict with operation status and number of HVGs. A split run returns
            ``groups`` (per-group counts), ``columns`` (everything written),
            ``skipped``, and the union / intersection sizes.
        """
        if split_by is not None:
            if subset:
                raise ValueError(
                    "subset=True cannot be combined with split_by: there is no "
                    "single gene set to subset to. Run the split first, then "
                    "subset on the union or intersection column."
                )
            if cell_subset:
                raise ValueError(
                    "split_by cannot be combined with a cell subset: the per-group "
                    "columns are named after the groups, not the subset. Run the "
                    "split on the whole dataset, or the subset on its own."
                )
            return self._run_hvg_split(
                split_by=split_by, n_top_genes=n_top_genes, min_mean=min_mean,
                max_mean=max_mean, min_disp=min_disp, flavor=flavor,
                n_bins=n_bins, active_cell_indices=active_cell_indices,
                add_union=add_union, add_intersection=add_intersection,
                min_cells_per_group=min_cells_per_group,
            )

        indices, subset_name = self._resolve_cell_scope(cell_subset, active_cell_indices)
        if subset_name is not None and subset:
            raise ValueError(
                "subset=True (drop non-variable genes) cannot be combined with a cell "
                "subset: the genes belong to the whole dataset.")
        out_column = 'highly_variable' if subset_name is None else f'highly_variable__{subset_name}'
        if indices is not None:
            from scipy import sparse
            adata_sub = self.adata[indices].copy()
            # Drop genes with zero expression in the subset to avoid
            # degenerate bin edges in scanpy's HVG binning step
            if sparse.issparse(adata_sub.X):
                gene_totals = np.asarray(adata_sub.X.sum(axis=0)).ravel()
            else:
                gene_totals = np.asarray(adata_sub.X.sum(axis=0)).ravel()
            expressed_mask = gene_totals > 0
            adata_hvg = adata_sub[:, expressed_mask].copy()

            # Compute HVGs on subset (always with subset=False to get annotations)
            sc.pp.highly_variable_genes(
                adata_hvg,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                flavor=flavor,
                n_bins=n_bins,
                subset=False,
            )
            if subset_name is not None:
                # Only the flag: the pooled means / dispersions stay the
                # dataset's, as with the split-by-group columns.
                full_col = pd.Series(False, index=self.adata.var_names)
                full_col.loc[adata_hvg.var_names] = adata_hvg.var['highly_variable'].values
                self.adata.var[out_column] = full_col.values.astype(bool)
            else:
                # Map results back to full gene set — unexpressed genes are not HVG
                for col in ['highly_variable', 'means', 'dispersions', 'dispersions_norm']:
                    if col in adata_hvg.var.columns:
                        default = False if col == 'highly_variable' else 0.0
                        full_col = pd.Series(default, index=self.adata.var_names, dtype=adata_hvg.var[col].dtype)
                        full_col.loc[adata_hvg.var_names] = adata_hvg.var[col]
                        self.adata.var[col] = full_col.values
                # Apply subset on full adata if requested
                if subset:
                    self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
        else:
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                flavor=flavor,
                n_bins=n_bins,
                subset=subset,
            )

        n_hvg = int(self.adata.var[out_column].sum())

        result = {
            'status': 'completed',
            'n_highly_variable': n_hvg,
            'n_total_genes': self.n_genes,
            'flavor': flavor,
        }
        params: dict[str, Any] = {
            'n_top_genes': n_top_genes,
            'min_mean': min_mean,
            'max_mean': max_mean,
            'min_disp': min_disp,
            'flavor': flavor,
            'subset': subset,
        }
        if subset_name is not None:
            result['column'] = out_column
            result['cell_subset'] = subset_name
            params['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'hvg', out_column, {
                'n_top_genes': n_top_genes, 'min_mean': min_mean, 'max_mean': max_mean,
                'min_disp': min_disp, 'flavor': flavor})
        self._log_action('highly_variable_genes', params, result, subset=indices)
        return result

    def run_pca(
        self,
        n_comps: int = 50,
        svd_solver: str = 'arpack',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        use_highly_variable: bool | None = None,
        active_cell_indices: list[int] | None = None,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Run PCA dimensionality reduction.

        Args:
            n_comps: Number of principal components to compute
            svd_solver: SVD solver to use ('arpack', 'randomized', 'auto')
            gene_subset: Subset of genes to use. Can be:
                - None: use default behavior (highly_variable if available, else all)
                - str: boolean column name from .var
                - list[str]: explicit list of gene names
                - dict: {'columns': [...], 'operation': 'intersection'|'union'}
            use_highly_variable: Deprecated, use gene_subset instead.
                If True and gene_subset is None, uses 'highly_variable' column.
            active_cell_indices: If provided, compute PCA on these cells only;
                inactive cells get NaN in X_pca.
            cell_subset: A named subset. The embedding goes to
                ``obsm['X_pca_<name>']`` (NaN outside the subset), loadings to
                ``varm['PCs_<name>']`` and metadata to ``uns['pca_<name>']``;
                the dataset's own ``X_pca`` is not touched. With no
                ``gene_subset`` the subset's own ``highly_variable__<name>``
                column is used when it exists, then the pooled one.

        Returns:
            Dict with operation status and variance explained
        """
        cell_indices, subset_name = self._resolve_cell_scope(cell_subset, active_cell_indices)

        # Handle legacy use_highly_variable parameter
        if gene_subset is None and use_highly_variable is True:
            if 'highly_variable' in self.adata.var.columns:
                gene_subset = 'highly_variable'

        auto_label: str | None = None
        if gene_subset is None and subset_name is not None:
            own = f'highly_variable__{subset_name}'
            if own in self.adata.var.columns:
                gene_subset = own
                auto_label = f'{own} (auto)'

        # Resolve gene subset
        if gene_subset is not None:
            # The gene mask is a session-only Gene-Panel view; it must not
            # silently change an embedding that gets written into the file.
            gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(
                gene_subset, apply_visible_mask=False)
            if auto_label:
                subset_type = auto_label
            n_genes_used = int(gene_mask.sum())

            # Create a temporary subset for PCA
            adata_pca = self.adata[:, gene_mask].copy()
        else:
            # Default scanpy behavior: use highly_variable if present
            adata_pca = self.adata
            subset_type = 'default'
            n_genes_used = self.n_genes
            if 'highly_variable' in self.adata.var.columns:
                n_genes_used = int(self.adata.var['highly_variable'].sum())
                subset_type = 'highly_variable (auto)'

        # Apply cell mask
        if cell_indices is not None:
            # Subset cells from the (possibly gene-subsetted) adata
            if gene_subset is not None:
                adata_pca = adata_pca[cell_indices].copy()
            else:
                adata_pca = self.adata[cell_indices].copy()

        # Limit n_comps to valid range. n_genes_used, not n_vars: on the
        # default path scanpy masks to the highly_variable column, so a
        # 76-gene dataset with 40 HVGs can give at most 39 components.
        max_comps = min(adata_pca.n_obs - 1, n_genes_used - 1)
        n_comps = min(n_comps, max_comps)

        # Run PCA on subset. An explicit gene subset must be used whole:
        # scanpy's mask_var defaults to the 'highly_variable' column whenever
        # one exists, which silently intersected 'spatially_variable' (or a
        # subset's own HVG column) with the pooled HVGs — and failed outright
        # when the two did not overlap.
        pca_kwargs: dict[str, Any] = {'n_comps': n_comps, 'svd_solver': svd_solver}
        if gene_subset is not None:
            pca_kwargs['mask_var'] = None
        sc.tl.pca(adata_pca, **pca_kwargs)

        if subset_name is not None:
            pca_key, pcs_key, uns_key = (f'X_pca_{subset_name}', f'PCs_{subset_name}',
                                         f'pca_{subset_name}')
        else:
            pca_key, pcs_key, uns_key = 'X_pca', 'PCs', 'pca'

        # Copy results back to main adata
        if cell_indices is not None:
            # Store the embedding with NaN for inactive cells
            full_pca = np.full((self.n_cells, n_comps), np.nan)
            full_pca[cell_indices] = adata_pca.obsm['X_pca']
            self.adata.obsm[pca_key] = full_pca
        else:
            self.adata.obsm[pca_key] = adata_pca.obsm['X_pca']
        self.adata.uns[uns_key] = adata_pca.uns['pca']

        # Copy gene loadings back as a full-size (n_genes, n_comps) matrix
        # with NaN rows for genes not included in the subset. Downstream
        # code (get_pca_loadings, create_pca_subset) expects varm['PCs']
        # to be present and correctly shaped for self.adata.n_vars.
        if 'PCs' in adata_pca.varm:
            full_pcs = np.full((self.n_genes, n_comps), np.nan)
            if gene_subset is not None:
                full_pcs[gene_mask, :] = adata_pca.varm['PCs']
            else:
                full_pcs[:, :] = adata_pca.varm['PCs']
            self.adata.varm[pcs_key] = full_pcs
            self.adata.uns[uns_key]['gene_subset'] = {
                'type': subset_type,
                'n_genes': n_genes_used,
            }

        # Get variance explained
        variance_ratio = self.adata.uns[uns_key]['variance_ratio'][:10].tolist()

        result = {
            'status': 'completed',
            'n_comps': n_comps,
            'variance_explained_top10': variance_ratio,
            'embedding_name': pca_key,
            'gene_subset_type': subset_type,
            'n_genes_used': n_genes_used,
        }
        # Clear derived PC subsets — they reference columns of the previous
        # X_pca and become stale on re-run. obsm and varm are NOT touched by
        # sc.tl.pca (only 'X_pca' and 'PCs' are overwritten), so we scan here.
        # sc.tl.pca does replace adata.uns['pca'] wholesale, so variance_ratio_*
        # and the 'subsets' metadata dict are already gone — the uns pops below
        # are defensive against stale obsm keys from externally loaded h5ad
        # files and to keep the invariant explicit. A named cell subset's PCA
        # shares the X_pca_<suffix> shape but is not derived from X_pca, so it
        # is skipped; and a subset run clears nothing, since X_pca is unchanged.
        cleared_subsets: list[str] = []
        if subset_name is None:
            subset_names = set(self._subset_registry())
            for key in list(self.adata.obsm.keys()):
                if key.startswith('X_pca_') and key != 'X_pca':
                    suffix = key[len('X_pca_'):]
                    # A subset's PCA, and the PC subsets under it, are its own.
                    if suffix in subset_names or any(
                            suffix.startswith(n + '_') for n in subset_names):
                        continue
                    self.adata.obsm.pop(key, None)
                    self.adata.varm.pop(f"PCs_{suffix}", None)
                    if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
                        self.adata.uns['pca'].pop(f"variance_ratio_{suffix}", None)
                        subsets_meta = self.adata.uns['pca'].get('subsets', {})
                        if isinstance(subsets_meta, dict):
                            subsets_meta.pop(suffix, None)
                    cleared_subsets.append(key)
        else:
            # The subset's own PC subsets reference columns of its previous
            # PCA. (uns['pca_<name>'] was replaced wholesale above, so their
            # variance ratios and metadata are already gone.)
            for key in list(self.adata.obsm.keys()):
                if key.startswith(f'{pca_key}_'):
                    self.adata.obsm.pop(key, None)
                    self.adata.varm.pop(f"{pcs_key}_{key[len(pca_key) + 1:]}", None)
                    self._subset_forget_key(subset_name, key)
                    cleared_subsets.append(key)
        if cleared_subsets:
            result['cleared_subsets'] = cleared_subsets

        params: dict[str, Any] = {
            'n_comps': n_comps,
            'svd_solver': svd_solver,
            'gene_subset': gene_subset,
        }
        if subset_name is not None:
            result['cell_subset'] = subset_name
            params['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'pca', pca_key, {
                'n_comps': n_comps, 'svd_solver': svd_solver,
                'gene_subset_type': subset_type, 'n_genes_used': n_genes_used})
        self._log_action('pca', params, result, subset=cell_indices)
        return result

    def _pca_slots(self, cell_subset: str | None) -> tuple[str, str, str, str | None]:
        """Where a PCA lives: ``(obsm key, varm key, uns key, subset name)``.

        The dataset's own is ``X_pca`` / ``PCs`` / ``uns['pca']``; a named
        subset's is each of those suffixed with its name. KeyError for an
        unknown subset, so a route maps it to 404.
        """
        if not cell_subset:
            return 'X_pca', 'PCs', 'pca', None
        clean = self._sanitize_subset_name(cell_subset)
        self._subset_mask(clean)
        return f'X_pca_{clean}', f'PCs_{clean}', f'pca_{clean}', clean

    def _pca_subset_owner(self, obsm_key: str) -> tuple[str | None, str]:
        """Which subset a PC-subset key belongs to, and its suffix.

        ``X_pca_chondro_noPC1`` is the subset chondro's; ``X_pca_noPC1`` is
        the dataset's. The longest registered name wins, so a subset named
        ``a`` does not claim ``X_pca_ab_noPC1``. A subset's *own* PCA is not a
        PC subset at all.
        """
        names = sorted(self._subset_registry(), key=len, reverse=True)
        for name in names:
            if obsm_key == f'X_pca_{name}':
                raise ValueError(
                    f"'{obsm_key}' is the PCA of the subset '{name}' itself, not a "
                    f"PC subset; delete the subset (with its results) instead.")
            if obsm_key.startswith(f'X_pca_{name}_'):
                return name, obsm_key[len(f'X_pca_{name}_'):]
        return None, obsm_key[len('X_pca_'):]

    def get_pca_loadings(self, top_n: int = 10, cell_subset: str | None = None) -> dict[str, Any]:
        """Return top +/- loading genes per computed PC.

        Reads varm['PCs'] and uns['pca']['variance_ratio'] — or, for a named
        subset, ``PCs_<name>`` and ``uns['pca_<name>']``, the subset's own.
        Gene rows containing NaN loadings (from subset-PCA runs) are excluded
        from per-PC rankings; up to top_n valid genes are returned per side.

        Raises:
            ValueError: if PCA has not been run or loadings are missing.

        Returns:
            {
              'n_pcs': int,
              'top_n': int,
              'pcs': [
                {
                  'index': 0,                 # zero-based
                  'variance_ratio': 0.127,
                  'positive': [{'gene': 'MALAT1', 'loading': 0.18}, ...],
                  'negative': [{'gene': 'MT-CO1', 'loading': -0.15}, ...],
                }, ...
              ]
            }
        """
        pca_key, pcs_key, uns_key, subset_name = self._pca_slots(cell_subset)
        where = f" on subset '{subset_name}'" if subset_name else ''
        if uns_key not in self.adata.uns:
            raise ValueError(f"PCA has not been run{where}. Run pca first.")
        if pcs_key not in self.adata.varm:
            raise ValueError(
                f"PC loadings are unavailable (varm['{pcs_key}'] missing). Re-run PCA{where}.")

        pcs_matrix = np.asarray(self.adata.varm[pcs_key])
        if pcs_matrix.ndim != 2:
            raise ValueError(f"Unexpected varm['{pcs_key}'] shape: {pcs_matrix.shape}")

        n_genes, n_comps = pcs_matrix.shape
        var_ratio = np.asarray(self.adata.uns[uns_key].get('variance_ratio', []))
        gene_names = list(self.adata.var_names)
        top_n = max(1, int(top_n))
        # Count genes with finite loadings on PC1 — mirrors the row-count a
        # subset PCA (e.g. HVG) actually contributed. Equal to n_genes on a
        # default PCA run; lower when gene_subset was active.
        n_genes_loaded = int(np.sum(~np.isnan(pcs_matrix[:, 0]))) if n_comps > 0 else 0

        pcs_out = []
        for i in range(n_comps):
            col = pcs_matrix[:, i]
            valid = ~np.isnan(col)
            valid_indices = np.where(valid)[0]
            valid_loadings = col[valid_indices]

            # Positive side: sort descending, take top_n
            pos_order = valid_indices[np.argsort(-valid_loadings)][:top_n]
            positive = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in pos_order
                if col[int(j)] > 0
            ]

            # Negative side: sort ascending, take top_n
            neg_order = valid_indices[np.argsort(valid_loadings)][:top_n]
            negative = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in neg_order
                if col[int(j)] < 0
            ]

            pcs_out.append({
                'index': i,
                'variance_ratio': float(var_ratio[i]) if i < len(var_ratio) else None,
                'positive': positive,
                'negative': negative,
            })

        return {
            'n_pcs': n_comps,
            'top_n': top_n,
            'n_genes_loaded': n_genes_loaded,
            'n_genes_total': n_genes,
            'pcs': pcs_out,
            'cell_subset': subset_name,
            'embedding': pca_key,
        }

    def create_pca_subset(
        self,
        drop_pc_indices: list[int],
        suffix: str | None = None,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Create derived PCA slots that exclude specific 1-indexed PCs.

        Writes:
          - obsm[f'X_pca_{suffix}'] — base embedding with dropped columns removed.
          - varm[f'PCs_{suffix}'] — matching loadings with dropped columns removed.
          - uns['pca'][f'variance_ratio_{suffix}'] — matching variance ratios.
          - uns['pca']['subsets'][suffix] = {'dropped_pcs': [i, j, ...]}
            (round-trips exact indices regardless of suffix).

        On a named subset the base is its own PCA and every key carries its
        name first — ``X_pca_<name>_<suffix>``, ``PCs_<name>_<suffix>``,
        ``uns['pca_<name>']`` — and the registry records it, so the subset
        owns the result and a dataset PCA re-run leaves it alone.

        Raises:
            ValueError: missing PCA, empty indices, out-of-range, all-dropped.
            ValueError: suffix collision with existing obsm key.
        """
        pca_key, pcs_key, uns_key, subset_name = self._pca_slots(cell_subset)
        if pca_key not in self.adata.obsm:
            where = f" on subset '{subset_name}'" if subset_name else ''
            raise ValueError(f"PCA has not been run{where}. Run pca first.")
        if not drop_pc_indices:
            raise ValueError("drop_pc_indices must contain at least one PC.")

        base_embed = np.asarray(self.adata.obsm[pca_key])
        n_cells, n_pcs = base_embed.shape

        # Convert from 1-indexed user-facing to 0-indexed column positions.
        idx = np.asarray(drop_pc_indices, dtype=int) - 1
        if (idx < 0).any():
            raise ValueError("drop_pc_indices must be >= 1 (PC numbers are 1-indexed).")
        if (idx >= n_pcs).any():
            raise ValueError(
                f"drop_pc_indices contains entries > {n_pcs} (total PCs available)."
            )
        idx = np.unique(idx)
        keep = np.setdiff1d(np.arange(n_pcs), idx, assume_unique=False)
        if keep.size == 0:
            raise ValueError("Cannot drop all PCs.")

        dropped_1indexed = sorted(int(i + 1) for i in idx)

        if suffix is None or suffix == '':
            suffix = f"noPC{'_'.join(str(i) for i in dropped_1indexed)}"

        new_obsm_key = f"{pca_key}_{suffix}"
        if new_obsm_key in self.adata.obsm:
            raise ValueError(f"A PC subset named '{suffix}' already exists.")

        # Write the three companion slots.
        self.adata.obsm[new_obsm_key] = base_embed[:, keep]

        varm_key = None
        if pcs_key in self.adata.varm:
            varm_key = f"{pcs_key}_{suffix}"
            self.adata.varm[varm_key] = np.asarray(self.adata.varm[pcs_key])[:, keep]

        var_ratio_key = None
        if uns_key in self.adata.uns and isinstance(self.adata.uns[uns_key], dict):
            if 'variance_ratio' in self.adata.uns[uns_key]:
                var_ratio_key = f"variance_ratio_{suffix}"
                self.adata.uns[uns_key][var_ratio_key] = np.asarray(
                    self.adata.uns[uns_key]['variance_ratio']
                )[keep]
            # Record the dropped indices for round-tripping in list_pca_subsets.
            subsets_meta = self.adata.uns[uns_key].setdefault('subsets', {})
            subsets_meta[suffix] = {'dropped_pcs': dropped_1indexed}

        result = {
            'obsm_key': new_obsm_key,
            'varm_key': varm_key,
            'variance_ratio_key': var_ratio_key,
            'suffix': suffix,
            'n_pcs_kept': int(keep.size),
            'dropped_pcs': dropped_1indexed,
            'cell_subset': subset_name,
        }
        params: dict[str, Any] = {'drop_pc_indices': dropped_1indexed, 'suffix': suffix}
        if subset_name is not None:
            params['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'pca_subsets', new_obsm_key,
                                     extra={'dropped_pcs': dropped_1indexed})
        self._log_action('create_pca_subset', params, result)
        return result

    def list_pca_subsets(self, cell_subset: str | None = None) -> list[dict[str, Any]]:
        """List the derived PC subsets of the dataset's PCA, or of a subset's.

        For each, reports obsm_key, suffix, n_pcs_kept, and dropped_pcs (from
        the owning uns entry's ``subsets`` when present, otherwise []). The
        dataset's list leaves out every key under a registered subset's
        prefix — a subset's own ``X_pca_<name>`` shares the shape of a PC
        subset but is not one.
        """
        pca_key, _, uns_key, subset_name = self._pca_slots(cell_subset)
        out: list[dict[str, Any]] = []
        subsets_meta = {}
        if uns_key in self.adata.uns and isinstance(self.adata.uns[uns_key], Mapping):
            subsets_meta = self.adata.uns[uns_key].get('subsets', {}) or {}
        prefix = f'{pca_key}_'
        names = set(self._subset_registry())

        for key in sorted(self.adata.obsm.keys()):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if subset_name is None and (
                    suffix in names or any(suffix.startswith(n + '_') for n in names)):
                continue
            arr = np.asarray(self.adata.obsm[key])
            n_pcs_kept = int(arr.shape[1]) if arr.ndim == 2 else 0
            meta = subsets_meta.get(suffix, {})
            dropped = [int(x) for x in meta.get('dropped_pcs', [])]
            out.append({
                'obsm_key': key,
                'suffix': suffix,
                'n_pcs_kept': n_pcs_kept,
                'dropped_pcs': dropped,
            })
        return out

    def delete_pca_subset(self, obsm_key: str) -> None:
        """Delete a derived PC subset's obsm, varm, variance_ratio, and
        uns['pca']['subsets'] entries.

        Raises:
            ValueError: if obsm_key == 'X_pca', is missing, or doesn't start
                with 'X_pca_'.
        """
        if obsm_key == 'X_pca':
            raise ValueError("Cannot delete the base X_pca embedding.")
        if not obsm_key.startswith('X_pca_'):
            raise ValueError(f"'{obsm_key}' is not a derived PC subset.")
        owner, suffix = self._pca_subset_owner(obsm_key)
        if obsm_key not in self.adata.obsm:
            raise ValueError(f"'{obsm_key}' not found in obsm.")

        _, pcs_key, uns_key, _ = self._pca_slots(owner)
        self.adata.obsm.pop(obsm_key, None)
        self.adata.varm.pop(f"{pcs_key}_{suffix}", None)
        if uns_key in self.adata.uns and isinstance(self.adata.uns[uns_key], dict):
            self.adata.uns[uns_key].pop(f"variance_ratio_{suffix}", None)
            subsets_meta = self.adata.uns[uns_key].get('subsets', {})
            if isinstance(subsets_meta, dict):
                subsets_meta.pop(suffix, None)
        if owner is not None:
            self._subset_forget_key(owner, obsm_key)
        self._log_action('delete_pca_subset', {'obsm_key': obsm_key}, None)

    def run_neighbors(
        self,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        metric: str = 'euclidean',
        use_rep: str | None = None,
        active_cell_indices: list[int] | None = None,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Compute neighborhood graph.

        Args:
            n_neighbors: Number of neighbors to use
            n_pcs: Number of PCs to use (None = use all)
            metric: Distance metric
            use_rep: obsm key to use as the representation (e.g. 'X_pca_noPC2_5').
                None or 'X_pca' preserves the default scanpy path (uses X_pca).
                Any other value must exist in adata.obsm.
            active_cell_indices: If provided, compute neighbors on these cells only;
                results are remapped into full-size sparse matrices.
            cell_subset: A named subset. The graph goes to
                ``obsp['<name>_connectivities']`` / ``['<name>_distances']``
                with scanpy's neighbours entry at ``uns['<name>']``, so UMAP
                and Leiden can pick it by ``graph_key``; the dataset's own
                graph is not touched. With no ``use_rep`` the subset's own
                ``X_pca_<name>`` is used when it exists, then ``X_pca``.

        Returns:
            Dict with operation status
        """
        from scipy.sparse import coo_matrix

        cell_indices, subset_name = self._resolve_cell_scope(cell_subset, active_cell_indices)

        # Check prerequisites
        prereq = self.check_prerequisites('neighbors', cell_subset=subset_name)
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        kwargs = {
            'n_neighbors': n_neighbors,
            'metric': metric,
        }
        if n_pcs is not None:
            kwargs['n_pcs'] = n_pcs

        # Resolve and validate use_rep. None / 'X_pca' preserve the existing
        # default path. Any other value must exist in adata.obsm.
        rep_key = use_rep if use_rep and use_rep != 'X_pca' else None
        # Unasked, a subset run uses its own PCA when it has one. An explicit
        # 'X_pca' is a choice — the dataset's embedding sliced to the subset.
        if not use_rep and subset_name is not None and f'X_pca_{subset_name}' in self.adata.obsm:
            rep_key = f'X_pca_{subset_name}'
        if rep_key is not None:
            if rep_key not in self.adata.obsm:
                raise ValueError(
                    f"use_rep '{rep_key}' not found in obsm. "
                    f"Create it via /api/scanpy/pca_subsets first."
                )
            kwargs['use_rep'] = rep_key

        if cell_indices is not None:
            # Build a subset AnnData with PCA from the active cells
            import anndata as ad
            source_key = rep_key if rep_key is not None else 'X_pca'
            pca_full = np.asarray(self.adata.obsm[source_key])
            pca_sub = pca_full[cell_indices]
            if np.isnan(pca_sub).any():
                n_bad = int(np.isnan(pca_sub).any(axis=1).sum())
                raise ValueError(
                    f"'{source_key}' has no values for {n_bad:,} of the selected cells "
                    f"(it was computed on a different selection). Run PCA on this "
                    f"selection first.")
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            adata_sub.obsm[source_key] = pca_sub

            # Name the representation explicitly. This AnnData holds only the
            # obsm block — it has no .X, on purpose — and scanpy's use_rep=None
            # default picks by adata.n_vars, which here is 0, so it falls
            # through to .X and dies on None. Without this, Neighbors on a cell
            # selection fails for every dataset.
            sub_kwargs = {**kwargs, 'use_rep': source_key}

            sc.pp.neighbors(adata_sub, **sub_kwargs)

            # Remap sparse obsp matrices to full size
            n_full = self.n_cells
            prefix = f'{subset_name}_' if subset_name is not None else ''
            for key in ['connectivities', 'distances']:
                if key in adata_sub.obsp:
                    sub_coo = adata_sub.obsp[key].tocoo()
                    full_rows = cell_indices[sub_coo.row]
                    full_cols = cell_indices[sub_coo.col]
                    full_mat = coo_matrix(
                        (sub_coo.data, (full_rows, full_cols)),
                        shape=(n_full, n_full),
                    )
                    self.adata.obsp[prefix + key] = full_mat.tocsr()

            if subset_name is not None:
                # scanpy's own key_added shape, so tl.umap / tl.leiden find the
                # graph through neighbors_key and list_neighbor_graphs sees it.
                meta = dict(adata_sub.uns['neighbors'])
                meta['connectivities_key'] = f'{subset_name}_connectivities'
                meta['distances_key'] = f'{subset_name}_distances'
                self.adata.uns[subset_name] = meta
            else:
                # Copy uns['neighbors'] metadata
                self.adata.uns['neighbors'] = adata_sub.uns['neighbors']
        else:
            sc.pp.neighbors(self.adata, **kwargs)

        result: dict[str, Any] = {'status': 'completed', 'n_neighbors': n_neighbors}
        params = dict(kwargs)
        if subset_name is not None:
            result['graph_key'] = f'{subset_name}_connectivities'
            result['use_rep'] = rep_key if rep_key is not None else 'X_pca'
            result['cell_subset'] = subset_name
            params['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'graph', result['graph_key'], {
                'n_neighbors': n_neighbors, 'n_pcs': n_pcs, 'metric': metric,
                'use_rep': result['use_rep']})
        self._log_action('neighbors', params, result, subset=cell_indices)
        return result

    def list_neighbor_graphs(self) -> list[dict[str, Any]]:
        """List available cell-cell connectivity graphs in obsp.

        Returns entries for any obsp key that looks like a connectivity graph
        ('connectivities' or '<prefix>_connectivities'), so they can be
        combined via combine_neighbor_graphs.
        """
        from scipy.sparse import issparse

        graphs: list[dict[str, Any]] = []
        n = self.n_cells
        for key in self.adata.obsp.keys():
            if key == 'connectivities' or key.endswith('_connectivities'):
                mat = self.adata.obsp[key]
                shape = tuple(mat.shape)
                if shape != (n, n):
                    continue
                nnz = int(mat.nnz) if issparse(mat) else int(np.count_nonzero(mat))
                if key == 'connectivities':
                    label = 'Expression neighbors'
                elif key == 'spatial_connectivities':
                    label = 'Spatial neighbors'
                else:
                    prefix = key[:-len('_connectivities')]
                    label = (f'Subset {prefix} neighbors'
                             if prefix in self._subset_registry() else f'{prefix} neighbors')
                graphs.append({
                    'key': key,
                    'label': label,
                    'n_edges': nnz,
                    'suffix': _graph_suffix(key),
                })
        graphs.sort(key=lambda g: (g['key'] != 'connectivities', g['key']))
        return graphs

    def combine_neighbor_graphs(
        self,
        sources: list[dict[str, Any]],
        target_key: str = 'connectivities',
    ) -> dict[str, Any]:
        """Combine multiple cell connectivity graphs with user-defined weights.

        Each source is ``{'key': str, 'weight': float}`` where key is an obsp
        connectivity graph (e.g. 'connectivities', 'spatial_connectivities').
        Each graph is max-normalized (so its largest edge weight becomes 1)
        before weighting, so graphs with very different edge scales contribute
        proportionally to their weight rather than their raw magnitude. The
        combined graph is symmetrized and written to ``obsp[target_key]``.

        If ``target_key`` is 'connectivities', ``uns['neighbors']`` is updated
        so downstream tools (Leiden, UMAP) pick up the combined graph with no
        further configuration.

        Args:
            sources: List of {key, weight} dicts. At least 2 entries required.
                Missing/None weights default to 1.0. If all weights are 0,
                equal weights are assumed.
            target_key: obsp key to store the combined connectivities in.

        Returns:
            Dict with status, target_key, sources_used, n_edges, and the
            normalized weights actually applied.
        """
        from scipy.sparse import issparse, csr_matrix

        if not isinstance(sources, list) or len(sources) < 2:
            raise ValueError("At least 2 source graphs are required to combine.")

        n = self.n_cells
        resolved: list[tuple[str, float]] = []
        for src in sources:
            if not isinstance(src, dict) or 'key' not in src:
                raise ValueError("Each source must be a dict with a 'key' field.")
            key = src['key']
            if key not in self.adata.obsp:
                raise ValueError(f"Connectivity graph '{key}' not found in obsp.")
            mat = self.adata.obsp[key]
            if tuple(mat.shape) != (n, n):
                raise ValueError(
                    f"Graph '{key}' has shape {mat.shape}, expected ({n}, {n})."
                )
            weight = src.get('weight', 1.0)
            try:
                weight = float(weight) if weight is not None else 1.0
            except (TypeError, ValueError):
                raise ValueError(f"Invalid weight for '{key}': {src.get('weight')!r}")
            if weight < 0:
                raise ValueError(f"Weight for '{key}' must be non-negative.")
            resolved.append((key, weight))

        # Normalize weights to sum to 1; if all zero, use equal weights.
        total = sum(w for _, w in resolved)
        if total <= 0:
            norm_weights = [1.0 / len(resolved)] * len(resolved)
        else:
            norm_weights = [w / total for _, w in resolved]

        combined = None
        applied: list[dict[str, Any]] = []
        for (key, _), w in zip(resolved, norm_weights):
            mat = self.adata.obsp[key]
            if issparse(mat):
                mat = mat.tocsr().astype(np.float64)
                max_val = float(mat.data.max()) if mat.nnz > 0 else 0.0
            else:
                mat = np.asarray(mat, dtype=np.float64)
                max_val = float(mat.max()) if mat.size > 0 else 0.0
            scale = (1.0 / max_val) if max_val > 0 else 0.0
            contrib = mat * (w * scale)
            combined = contrib if combined is None else combined + contrib
            applied.append({'key': key, 'weight': w, 'max_value': max_val})

        # Symmetrize: (C + C^T) / 2
        if issparse(combined):
            combined = (combined + combined.T) * 0.5
            combined = csr_matrix(combined)
            combined.eliminate_zeros()
            n_edges = int(combined.nnz)
        else:
            combined = (combined + combined.T) * 0.5
            n_edges = int(np.count_nonzero(combined))

        self.adata.obsp[target_key] = combined

        # Update uns['neighbors'] so leiden/umap find the combined graph when
        # target_key is 'connectivities'. We write a minimal but valid entry.
        if target_key == 'connectivities':
            neighbors_meta = dict(self.adata.uns.get('neighbors', {}))
            params = dict(neighbors_meta.get('params', {}))
            params['method'] = 'combined'
            params['sources'] = [a['key'] for a in applied]
            neighbors_meta['params'] = params
            neighbors_meta['connectivities_key'] = 'connectivities'
            # Leave distances_key alone; leiden only needs connectivities.
            self.adata.uns['neighbors'] = neighbors_meta

        result = {
            'status': 'completed',
            'target_key': target_key,
            'n_edges': n_edges,
            'sources_used': applied,
        }
        self._log_action('combine_neighbors', {
            'sources': [{'key': k, 'weight': w} for (k, _), w in zip(resolved, norm_weights)],
            'target_key': target_key,
        }, result)
        return result

    def list_layers(self) -> list[dict[str, Any]]:
        """List the readable expression matrices for downstream gene analyses.

        Always returns the synthetic 'X' entry first (the current ``adata.X``)
        followed by every key in ``adata.layers``. Used by the frontend to
        populate "Source matrix" dropdowns on Gene PCA / Gene Neighbors /
        Cluster Genes — picking a non-default layer routes the computation
        through ``adata.layers[layer]`` instead of ``adata.X``.

        Each entry also carries a ``scale`` block from :mod:`xcell.layer_scale`
        saying whether the matrix looks like raw counts, library-size
        normalized values, log-transformed values, or z-scores — plus the
        evidence behind that call and any provenance recorded by scanpy or by
        xcell's own preprocessing history. Datasets rarely document which of
        these they hold, and picking the wrong source silently corrupts
        anything rank- or count-based downstream.
        """
        from scipy.sparse import issparse

        from xcell import layer_scale as ls

        out: list[dict[str, Any]] = []
        n_obs, n_var = self.adata.shape

        def _info(name: str, mat) -> dict[str, Any]:
            sparse_flag = issparse(mat)
            nnz = int(mat.nnz) if sparse_flag else int(np.count_nonzero(mat))
            density = nnz / max(n_obs * n_var, 1)
            scale = ls.assess_matrix_scale(mat)
            scale['provenance'] = (
                ls.provenance_from_adata(self.adata, name)
                + ls.provenance_from_history(self._action_history, name)
            )
            return {
                'name': name,
                'shape': list(mat.shape),
                'nnz': nnz,
                'density': float(density),
                'sparse': bool(sparse_flag),
                'scale': scale,
            }

        out.append({**_info('X', self.adata.X), 'is_default': True})
        for key in sorted(self.adata.layers.keys()):
            out.append({**_info(key, self.adata.layers[key]), 'is_default': False})
        return out

    SMOOTH_POST_TRANSFORMS = ('none', 'cp10k', 'lognorm')

    def run_smooth(
        self,
        graph_key: str,
        n_steps: int = 1,
        source_layer: str | None = None,
        output_layer: str = 'smoothed',
        self_loop_weight: float = 1.0,
        post_transform: str = 'none',
    ) -> dict[str, Any]:
        """Smooth expression over a kNN graph and store the result in a layer.

        Algorithm — one step is:  ``S ← D⁻¹·(A + α·I)·X``
        where A is the obsp connectivity matrix specified by ``graph_key`` and
        α is ``self_loop_weight``. The self-loop term keeps each cell's own
        value in the average (α=1 weights it equally with its neighbors;
        α=0 is pure neighbor average; large α barely smooths). Repeating
        ``n_steps`` times applies the same operator iteratively, which spreads
        the influence of distant neighbors at the cost of more over-smoothing.

        The result is written to ``adata.layers[output_layer]`` as a sparse
        matrix (the smoothing operator densifies relative to a sparse input,
        but for typical kNN smoothing the output stays sparse enough to be
        worth keeping in CSR form). ``adata.X`` is **not** modified — the
        smoothed matrix is intended as an alternative source for downstream
        gene-side analyses (Gene PCA / Gene Neighbors / Cluster Genes), each
        of which gained a ``layer`` parameter to opt in.

        Args:
            graph_key: Key in ``adata.obsp`` whose connectivities define the
                kNN structure. Typically ``connectivities`` (expression),
                ``spatial_connectivities``, or a combined graph from
                ``combine_neighbors``.
            n_steps: Number of smoothing iterations. Default 1.
            source_layer: Layer to read the input matrix from. None → ``adata.X``.
            output_layer: Layer to write the smoothed matrix to. Existing
                layer with the same name is overwritten.
            self_loop_weight: Weight α for the self-loop term. Default 1.0.
            post_transform: What to do to the smoothed matrix before storing it.
                ``'none'`` (default) stores the average as-is; ``'cp10k'``
                rescales each cell to 10,000; ``'lognorm'`` does that and takes
                log1p.

                This exists because averaging does not commute with the log.
                Smoothing a log-normalized matrix averages logs, and the mean of
                logs is not the log of the mean — on E11.5 limb ST data the
                stored values come out 70–83% below log1p of the smoothed
                expression, the gap widening with expression level. Held-out
                molecular cross-validation on the same data ranks the pipelines
                counts → smooth → normalize (best) < normalize → smooth <
                log → smooth < sqrt → smooth, with the spread widening as
                smoothing strengthens. So the recommended route to a denoised,
                directly usable matrix is to smooth **counts** and transform
                afterwards, which is what ``'lognorm'`` does in one step.
                See docs/measurements/2026-08-21-smoothing-transform.md.
        """
        from scipy.sparse import csr_matrix, diags, eye, issparse

        if graph_key not in self.adata.obsp:
            raise ValueError(
                f"Graph '{graph_key}' not found in obsp. "
                f"Available: {list(self.adata.obsp.keys())}"
            )
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if self_loop_weight < 0:
            raise ValueError(
                f"self_loop_weight must be >= 0, got {self_loop_weight}"
            )
        if post_transform not in self.SMOOTH_POST_TRANSFORMS:
            raise ValueError(
                f"post_transform must be one of "
                f"{list(self.SMOOTH_POST_TRANSFORMS)}, got '{post_transform}'"
            )

        n = self.n_cells
        A = self.adata.obsp[graph_key]
        if tuple(A.shape) != (n, n):
            raise ValueError(
                f"Graph '{graph_key}' has shape {A.shape}, expected ({n}, {n})."
            )
        if not issparse(A):
            A = csr_matrix(A)
        else:
            A = A.tocsr().astype(np.float64)

        if self_loop_weight > 0:
            A = A + eye(n, format='csr') * self_loop_weight

        # Row-normalize so each cell's smoothed values are a weighted average
        # of its (own + neighbors') values, not a sum that depends on degree.
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        A_norm = diags(1.0 / deg) @ A

        # Source matrix — None means read .X.
        if source_layer is None or source_layer == 'X':
            X = self.adata.X
            source_label = 'X'
        else:
            if source_layer not in self.adata.layers:
                raise ValueError(
                    f"Source layer '{source_layer}' not found. "
                    f"Available: {sorted(self.adata.layers.keys())}"
                )
            X = self.adata.layers[source_layer]
            source_label = source_layer
        if not issparse(X):
            X = csr_matrix(X)

        S = X
        for _ in range(n_steps):
            S = A_norm @ S
        # scipy's sparse @ sparse keeps CSR; ensure that explicitly.
        if not issparse(S):
            S = csr_matrix(S)
        S = S.tocsr().astype(np.float32)
        S.eliminate_zeros()

        if post_transform in ('cp10k', 'lognorm'):
            totals = np.asarray(S.sum(axis=1)).ravel()
            # A cell whose neighbourhood is empty has nothing to rescale;
            # dividing by its zero total would write NaN into the layer.
            totals[totals == 0] = 1.0
            S = (diags((1e4 / totals).astype(np.float32)) @ S).tocsr()
            if post_transform == 'lognorm':
                S.data = np.log1p(S.data)
            S = S.astype(np.float32)

        self.adata.layers[output_layer] = S

        result = {
            'status': 'completed',
            'graph_key': graph_key,
            'n_steps': n_steps,
            'source_layer': source_label,
            'output_layer': output_layer,
            'self_loop_weight': float(self_loop_weight),
            'post_transform': post_transform,
            'output_nnz': int(S.nnz),
            'output_density': float(S.nnz / max(n * S.shape[1], 1)),
        }
        self._log_action('smooth', {
            'graph_key': graph_key,
            'n_steps': n_steps,
            'source_layer': source_label,
            'output_layer': output_layer,
            'self_loop_weight': float(self_loop_weight),
            'post_transform': post_transform,
        }, result)
        return result

    def _require_graph(self, graph_key: str) -> None:
        """Validate an obsp connectivity graph before any work starts."""
        if graph_key not in self.adata.obsp:
            available = [k for k in self.adata.obsp
                         if k == 'connectivities' or k.endswith(_CONN_SUFFIX)]
            raise ValueError(
                f"Graph '{graph_key}' not found in obsp. "
                f"Available: {available if available else 'none'}"
            )
        n = self.n_cells
        shape = tuple(self.adata.obsp[graph_key].shape)
        if shape != (n, n):
            raise ValueError(
                f"Graph '{graph_key}' has shape {shape}, expected ({n}, {n})."
            )

    def run_umap(
        self,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        graph_key: str | None = None,
        key_added: str | None = None,
        active_cell_indices: list[int] | None = None,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Compute a UMAP embedding.

        Args:
            min_dist: Minimum distance between points.
            spread: Spread of the embedding.
            n_components: Number of dimensions.
            graph_key: obsp connectivity graph to embed. None uses whatever
                ``uns['neighbors']`` points at — the expression kNN, and the
                behaviour before this parameter existed. Any other graph
                (spatial, or a Combine Neighbors result) reaches scanpy through
                a synthesized entry; see :func:`_umap_neighbors_meta`.
            key_added: obsm key for the result. None or blank derives it from
                the graph: ``X_umap`` for the default, ``X_umap_<prefix>``
                otherwise, so embeddings from different graphs coexist.
            active_cell_indices: If provided, compute UMAP on these cells only;
                inactive cells get NaN coordinates.
            cell_subset: A named subset. Embeds the subset's own graph
                (``<name>_connectivities``) when it exists and no graph_key is
                given, otherwise the chosen or default graph sliced to the
                subset; the result goes to ``X_umap_<name>`` unless key_added
                names it, and the dataset's own ``X_umap`` is not touched.

        Returns:
            Dict with status, embedding_name, n_components, graph_key.
        """
        cell_indices, subset_name = self._resolve_cell_scope(cell_subset, active_cell_indices)
        if subset_name is not None and not graph_key:
            own_graph = f'{subset_name}_connectivities'
            if own_graph in self.adata.obsp:
                graph_key = own_graph
        if graph_key:
            # A named graph carries its own prerequisite. check_prerequisites
            # looks for obsp['connectivities'], which a spatial-only dataset
            # does not have — and that is the case this parameter exists for.
            self._require_graph(graph_key)
        else:
            prereq = self.check_prerequisites('umap')
            if not prereq['satisfied']:
                raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        name = (key_added or '').strip()
        if not name:
            name = (_subset_output_name('X_umap', subset_name, graph_key)
                    if subset_name is not None
                    else _default_output_name('X_umap', graph_key))
        meta = _umap_neighbors_meta(self.adata, graph_key) if graph_key else None

        if cell_indices is not None:
            import anndata as ad
            adata_sub = ad.AnnData(
                obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            # sc.tl.umap reads the representation the neighbours entry names
            # (to seed the layout), so the slice needs that key, not just X_pca.
            wanted_reps = {'X_pca'}
            if meta is not None and meta['params'].get('use_rep') not in (None, 'X'):
                wanted_reps.add(meta['params']['use_rep'])
            for rep in wanted_reps:
                if rep in self.adata.obsm:
                    adata_sub.obsm[rep] = np.asarray(self.adata.obsm[rep])[cell_indices]

            # Slice the graph this run actually uses. Hardcoding
            # 'connectivities' here sliced the wrong matrix for any other
            # choice, and nothing at all on a spatial-only dataset.
            if meta is not None:
                wanted = (meta['connectivities_key'], meta['distances_key'])
            else:
                wanted = ('connectivities', 'distances')
            for key in wanted:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if meta is None and 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sub_meta = dict(meta) if meta is not None else None
            if sub_meta is not None and sub_meta['params'].get('use_rep') == 'X':
                # adata_sub deliberately has no .X — only the obsm block — so
                # 'X' is not a readable representation here. X_pca is the only
                # thing a subset run can offer.
                if 'X_pca' not in adata_sub.obsm:
                    raise ValueError(
                        "UMAP on a cell selection needs X_pca — run PCA first."
                    )
                sub_meta['params'] = {**sub_meta['params'], 'use_rep': 'X_pca'}

            _umap_call(adata_sub, min_dist=min_dist, spread=spread,
                       n_components=n_components, meta=sub_meta, key_added='X_umap')

            full_umap = np.full((self.n_cells, n_components), np.nan)
            full_umap[cell_indices] = adata_sub.obsm['X_umap']
            self.adata.obsm[name] = full_umap
        else:
            _umap_call(self.adata, min_dist=min_dist, spread=spread,
                       n_components=n_components, meta=meta, key_added=name)

        result = {
            'status': 'completed',
            'embedding_name': name,
            'n_components': n_components,
            'graph_key': graph_key,
        }
        # Only record what differs from the default, so existing recordings and
        # their exported notebooks stay byte-identical.
        params: dict[str, Any] = {
            'min_dist': min_dist, 'spread': spread, 'n_components': n_components,
        }
        if graph_key:
            params['graph_key'] = graph_key
            # Record the entry that was actually installed, rather than letting
            # codegen re-derive it. The exported notebook then emits what really
            # ran — including use_rep, without which the notebook would compute
            # the same phantom PCA this method exists to avoid.
            params['neighbors_meta'] = meta
        if name != 'X_umap':
            params['key_added'] = name
        if subset_name is not None:
            params['cell_subset'] = subset_name
            result['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'umap', name, {
                'min_dist': min_dist, 'spread': spread, 'n_components': n_components,
                'graph_key': graph_key})
        self._log_action('umap', params, result, subset=cell_indices)
        return result

    def run_leiden(
        self,
        resolution: float = 1.0,
        key_added: str = 'leiden',
        graph_key: str | None = None,
        active_cell_indices: list[int] | None = None,
        cell_subset: str | None = None,
    ) -> dict[str, Any]:
        """Run Leiden clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters).
            key_added: obs column for the labels. Left at its default, a
                non-default graph derives its own name (``leiden_spatial``) so
                clusterings from different graphs coexist.
            graph_key: obsp connectivity graph to cluster. None uses
                ``obsp['connectivities']`` via ``uns['neighbors']``, as before.
            active_cell_indices: If provided, cluster only these cells;
                inactive cells are labeled 'unassigned'.
            cell_subset: A named subset. Clusters the subset's own graph when
                it exists and no graph_key is given, otherwise the chosen or
                default graph sliced to the subset; labels go to
                ``leiden_<name>`` (``unassigned`` outside) unless key_added
                names the column, and the dataset's own ``leiden`` is not touched.

        Returns:
            Dict with status, key_added, n_clusters, resolution, graph_key.
        """
        cell_indices, subset_name = self._resolve_cell_scope(cell_subset, active_cell_indices)
        if subset_name is not None and not graph_key:
            own_graph = f'{subset_name}_connectivities'
            if own_graph in self.adata.obsp:
                graph_key = own_graph
        if graph_key:
            self._require_graph(graph_key)
        else:
            prereq = self.check_prerequisites('leiden')
            if not prereq['satisfied']:
                raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # key_added defaults to a string rather than None, so an explicit
        # 'leiden' is indistinguishable from an unset one — treat the default
        # value as unset and let the graph name the column.
        if key_added != 'leiden':
            name = key_added
        elif subset_name is not None:
            name = _subset_output_name('leiden', subset_name, graph_key)
        else:
            name = _default_output_name('leiden', graph_key)
        # sc.tl.leiden takes the adjacency directly; unlike UMAP it needs no
        # synthesized uns entry. obsp and neighbors_key are mutually exclusive,
        # so only one is ever passed.
        graph_kwargs = {'obsp': graph_key} if graph_key else {}

        if cell_indices is not None:
            # Build subset AnnData with neighbor graph
            import anndata as ad
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))

            # Slice the graph this run actually uses. Hardcoding
            # 'connectivities' sliced the wrong matrix for any other choice,
            # and nothing at all on a spatial-only dataset.
            wanted = (graph_key,) if graph_key else ('connectivities', 'distances')
            for key in wanted:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if not graph_key and 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sc.tl.leiden(adata_sub, resolution=resolution, key_added=name,
                         **graph_kwargs)

            # Map labels back with 'unassigned' for inactive cells
            sub_categories = list(adata_sub.obs[name].cat.categories)
            all_categories = sub_categories + ['unassigned']
            full_labels = ['unassigned'] * self.n_cells
            for i, idx in enumerate(cell_indices):
                full_labels[idx] = str(adata_sub.obs[name].iloc[i])
            self.adata.obs[name] = pd.Categorical(
                full_labels, categories=all_categories,
            )

            n_clusters = len(sub_categories)
        else:
            sc.tl.leiden(self.adata, resolution=resolution, key_added=name,
                         **graph_kwargs)
            n_clusters = len(self.adata.obs[name].cat.categories)

        result = {
            'status': 'completed',
            'key_added': name,
            'n_clusters': n_clusters,
            'resolution': resolution,
            'graph_key': graph_key,
        }
        params: dict[str, Any] = {'resolution': resolution, 'key_added': name}
        if graph_key:
            params['graph_key'] = graph_key
        if subset_name is not None:
            params['cell_subset'] = subset_name
            result['cell_subset'] = subset_name
            self._subset_record_step(subset_name, 'leiden', name, {
                'resolution': resolution, 'graph_key': graph_key})
        self._log_action('leiden', params, result, subset=cell_indices)
        return result

    # =========================================================================
    # Gene analysis methods
    # =========================================================================

    @staticmethod
    def _find_elbow_kneedle(values: np.ndarray, sensitivity: float = 1.0) -> int:
        """Find elbow point using Kneedle algorithm.

        Args:
            values: Array of values (e.g., variance ratios)
            sensitivity: Sensitivity parameter (higher = more sensitive)

        Returns:
            Index of the elbow point
        """
        n = len(values)
        if n < 2:
            return 0

        # Normalize x and y to [0, 1]
        x = np.arange(n)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)

        # Calculate differences from the diagonal line
        # For a decreasing curve, we look for max distance below the line
        differences = y_norm - (1 - x_norm)

        # Apply sensitivity - look for where the curve deviates significantly
        threshold = sensitivity * np.std(differences)

        # Find the elbow: first point where difference drops below threshold
        # after the initial steep decline
        for i in range(1, n - 1):
            if differences[i] < -threshold:
                # Check if we're past the steep part
                local_slope = values[i] - values[i - 1]
                next_slope = values[i + 1] - values[i] if i + 1 < n else 0
                if abs(next_slope) < abs(local_slope) * 0.5:
                    return i

        # Fallback: use maximum curvature
        if n > 2:
            second_derivative = np.diff(np.diff(values))
            return int(np.argmax(np.abs(second_derivative))) + 1

        return min(n - 1, 10)

    def run_gene_pca(
        self,
        n_comps: int | None = None,
        scale: bool = True,
        use_kneedle: bool = True,
        max_comps: int = 100,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        active_cell_indices: list[int] | None = None,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Run PCA on genes (transposed expression matrix).

        Computes gene embeddings based on their expression patterns across cells.
        Results are stored in .varm['X_gene_pca'] and variance info in .uns['gene_pca'].

        Args:
            n_comps: Number of components. If None and use_kneedle=True, auto-detect.
            scale: Whether to z-score scale genes before PCA (recommended)
            use_kneedle: Whether to use Kneedle algorithm for auto PC selection
            max_comps: Maximum components to compute before Kneedle selection
            gene_subset: Subset of genes to use. Can be:
                - None: use all genes
                - str: boolean column name from .var (e.g., 'highly_variable', 'spatially_variable')
                - list[str]: explicit list of gene names
                - dict: {'columns': ['col1', 'col2'], 'operation': 'intersection'|'union'}
                  for combining multiple boolean columns with AND/OR logic
            active_cell_indices: If provided, use only these cells for gene PCA
            layer: Optional layer name in adata.layers to read expression from
                instead of adata.X. Pass the output of run_smooth here to use a
                kNN-smoothed matrix as the basis for gene PCA.

        Returns:
            Dict with operation status, n_comps used, and variance explained
        """
        from scipy import sparse
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Resolve gene subset to boolean mask
        gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(gene_subset)

        # Store the gene subset info for downstream use
        self.adata.uns['gene_pca_subset'] = {
            'type': subset_type,
            'genes': self.adata.var_names[gene_mask].tolist(),
            'n_genes': int(gene_mask.sum()),
            **subset_metadata,
        }

        # Resolve source matrix (X or a layer).
        source_matrix = self._resolve_source_matrix(layer)

        # Subset cells if active_cell_indices provided, then genes, then transpose
        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            X = source_matrix[cell_indices][:, gene_mask].T
        else:
            X = source_matrix[:, gene_mask].T
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional scaling (z-score each gene's expression across cells)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Determine number of components to compute
        n_genes, n_cells = X.shape
        max_possible = min(n_genes - 1, n_cells - 1, max_comps)

        if n_comps is not None:
            # User specified
            n_comps_compute = min(n_comps, max_possible)
            n_comps_final = n_comps_compute
        elif use_kneedle:
            # Compute more PCs, then use Kneedle to select
            n_comps_compute = max_possible
        else:
            # Default to 50
            n_comps_compute = min(50, max_possible)
            n_comps_final = n_comps_compute

        # Run PCA
        pca = PCA(n_components=n_comps_compute)
        gene_pcs = pca.fit_transform(X)
        variance_ratio = pca.explained_variance_ratio_

        # Apply Kneedle if needed
        if n_comps is None and use_kneedle:
            elbow_idx = self._find_elbow_kneedle(variance_ratio)
            n_comps_final = max(elbow_idx + 1, 5)  # At least 5 PCs
            n_comps_final = min(n_comps_final, n_comps_compute)
        elif n_comps is None:
            n_comps_final = n_comps_compute

        # Store truncated results
        # If using a subset, store full-sized array with NaN for excluded genes
        full_gene_pcs = np.full((self.n_genes, n_comps_final), np.nan)
        full_gene_pcs[gene_mask, :] = gene_pcs[:, :n_comps_final]
        self.adata.varm['X_gene_pca'] = full_gene_pcs

        # Store variance info and subset metadata
        self.adata.uns['gene_pca'] = {
            'variance_ratio': variance_ratio.tolist(),
            'variance': pca.explained_variance_.tolist(),
            'n_comps': n_comps_final,
            'n_comps_computed': n_comps_compute,
            'scaled': scale,
            'elbow_index': elbow_idx if (n_comps is None and use_kneedle) else None,
            'gene_subset_type': subset_type,
            'n_genes_used': int(gene_mask.sum()),
        }

        cumulative_var = float(np.sum(variance_ratio[:n_comps_final]))

        result = {
            'status': 'completed',
            'n_comps': n_comps_final,
            'n_comps_computed': n_comps_compute,
            'cumulative_variance': cumulative_var,
            'scaled': scale,
            'elbow_detected': elbow_idx if (n_comps is None and use_kneedle) else None,
            'gene_subset_type': subset_type,
            'n_genes_used': int(gene_mask.sum()),
        }
        self._log_action('gene_pca', {
            'n_comps': n_comps,
            'scale': scale,
            'use_kneedle': use_kneedle,
            'gene_subset': gene_subset,
        }, result, subset=cell_indices)
        return result

    def get_cell_pca_variance(self) -> dict[str, Any]:
        """Get cell PCA variance information for visualization.

        Returns:
            Dict with variance ratios, cumulative variance, and elbow point
        """
        if 'pca' not in self.adata.uns:
            raise ValueError("Cell PCA has not been computed. Run pca first.")

        info = self.adata.uns['pca']
        variance_ratio = np.array(info['variance_ratio'])
        cumulative = np.cumsum(variance_ratio)

        # Try elbow detection
        elbow_idx = None
        if len(variance_ratio) > 2:
            elbow_idx = self._find_elbow_kneedle(variance_ratio)

        return {
            'variance_ratio': variance_ratio.tolist(),
            'cumulative_variance': cumulative.tolist(),
            'n_comps_used': len(variance_ratio),
            'n_comps_computed': len(variance_ratio),
            'elbow_index': elbow_idx,
        }

    def get_gene_pca_variance(self) -> dict[str, Any]:
        """Get gene PCA variance information for visualization.

        Returns:
            Dict with variance ratios, cumulative variance, and elbow point
        """
        if 'gene_pca' not in self.adata.uns:
            raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

        info = self.adata.uns['gene_pca']
        variance_ratio = np.array(info['variance_ratio'])
        cumulative = np.cumsum(variance_ratio)

        return {
            'variance_ratio': variance_ratio.tolist(),
            'cumulative_variance': cumulative.tolist(),
            'n_comps_used': info['n_comps'],
            'n_comps_computed': info['n_comps_computed'],
            'elbow_index': info.get('elbow_index'),
        }

    def prepare_gene_neighbors(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        basis: str = 'gene_pca',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        scale: bool = True,
        active_cell_indices: list[int] | None = None,
        layer: str | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare gene-gene kNN graph computation (cancellable).

        Validates inputs and snapshots data upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            n_neighbors: Number of neighbors per gene
            metric: Distance metric ('euclidean', 'cosine', 'pearson')
            basis: 'gene_pca' to use PCA embedding, 'expression' to use raw expression
            gene_subset: Gene filtering (only used when basis='expression').
                Can be str (boolean column), list[str] (gene names), or dict (multi-column spec).
            scale: Z-score scale genes before computing neighbors (only used when basis='expression')
            active_cell_indices: Optional cell subset (only used when basis='expression')

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy import sparse

        n_genes_total = self.adata.n_vars
        subset_type = 'all'
        # Only the 'expression' basis subsets cells; elsewhere None is the
        # honest answer for the analysis record.
        cell_indices = None

        if basis == 'gene_pca':
            if 'X_gene_pca' not in self.adata.varm:
                raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

            gene_pcs = self.adata.varm['X_gene_pca'].copy()
            valid_mask = ~np.isnan(gene_pcs[:, 0])
            valid_indices = np.where(valid_mask)[0]
            representation = gene_pcs[valid_mask, :]

            pca_info = self.adata.uns.get('gene_pca', {})
            subset_type = pca_info.get('gene_subset_type', 'all')

        elif basis == 'expression':
            from sklearn.preprocessing import StandardScaler

            if gene_subset is not None or self._visible_gene_mask is not None:
                gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            else:
                gene_mask = np.ones(self.adata.n_vars, dtype=bool)

            valid_mask = gene_mask
            valid_indices = np.where(valid_mask)[0]

            cell_indices = self._validate_cell_indices(active_cell_indices)
            source_matrix = self._resolve_source_matrix(layer)
            if cell_indices is not None:
                X = source_matrix[cell_indices][:, gene_mask].T
            else:
                X = source_matrix[:, gene_mask].T

            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            representation = X
        else:
            raise ValueError(f"Unknown basis: {basis}. Must be 'gene_pca' or 'expression'.")

        n_genes_valid = len(valid_indices)
        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings/expression")

        n_neighbors = min(n_neighbors, n_genes_valid - 1)

        # Snapshot all data needed by compute_fn
        snap_representation = representation.copy()
        snap_valid_indices = valid_indices.copy()
        snap_n_genes_total = n_genes_total
        snap_n_genes_valid = n_genes_valid
        snap_n_neighbors = n_neighbors
        snap_metric = metric
        snap_basis = basis
        snap_subset_type = subset_type

        def compute_fn() -> dict[str, Any]:
            from sklearn.neighbors import NearestNeighbors
            from scipy import sparse

            if snap_metric == 'pearson':
                corr_matrix = np.corrcoef(snap_representation)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                dist_full = 1.0 - corr_matrix
                np.fill_diagonal(dist_full, 0.0)
                nn = NearestNeighbors(n_neighbors=snap_n_neighbors + 1, metric='precomputed')
                nn.fit(dist_full)
                distances, indices = nn.kneighbors(dist_full)
            else:
                nn = NearestNeighbors(n_neighbors=snap_n_neighbors + 1, metric=snap_metric)
                nn.fit(snap_representation)
                distances, indices = nn.kneighbors(snap_representation)

            rows = []
            cols = []
            dists = []
            for i_valid in range(snap_n_genes_valid):
                i_original = snap_valid_indices[i_valid]
                for j_idx in range(1, snap_n_neighbors + 1):
                    j_valid = indices[i_valid, j_idx]
                    j_original = snap_valid_indices[j_valid]
                    rows.append(i_original)
                    cols.append(j_original)
                    dists.append(distances[i_valid, j_idx])

            dist_matrix = sparse.csr_matrix(
                (dists, (rows, cols)),
                shape=(snap_n_genes_total, snap_n_genes_total)
            )

            conn_weights = [1.0 / (1.0 + d) for d in dists]
            conn_matrix = sparse.csr_matrix(
                (conn_weights, (rows, cols)),
                shape=(snap_n_genes_total, snap_n_genes_total)
            )

            return {
                'dist_matrix': dist_matrix,
                'conn_matrix': conn_matrix,
                'n_neighbors': snap_n_neighbors,
                'metric': snap_metric,
                'basis': snap_basis,
                'n_genes_valid': snap_n_genes_valid,
                'n_genes_total': snap_n_genes_total,
                'subset_type': snap_subset_type,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            self.adata.varp['gene_distances'] = result['dist_matrix']
            self.adata.varp['gene_connectivities'] = result['conn_matrix']

            self.adata.uns['gene_neighbors'] = {
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
                'n_genes_in_graph': result['n_genes_valid'],
                'gene_subset_type': result['subset_type'],
            }

            status_result = {
                'status': 'completed',
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
                'n_genes': result['n_genes_valid'],
                'n_genes_total': result['n_genes_total'],
                'gene_subset_type': result['subset_type'],
            }
            self._log_action('gene_neighbors', {
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
            }, status_result, subset=cell_indices)
            return status_result

        return compute_fn, apply_fn

    def run_gene_neighbors(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        basis: str = 'gene_pca',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        scale: bool = True,
        active_cell_indices: list[int] | None = None,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Compute gene-gene kNN graph from gene PCA embedding or raw expression.

        Results are stored in .varp['gene_connectivities'] and .varp['gene_distances'].

        Args:
            n_neighbors: Number of neighbors per gene
            metric: Distance metric ('euclidean', 'cosine', 'pearson')
            basis: 'gene_pca' to use PCA embedding, 'expression' to use raw expression
            gene_subset: Gene filtering (only used when basis='expression').
                Can be str (boolean column), list[str] (gene names), or dict (multi-column spec).
            scale: Z-score scale genes before computing neighbors (only used when basis='expression')
            active_cell_indices: Optional cell subset (only used when basis='expression')
            layer: Optional layer name to read expression from instead of adata.X
                (only used when basis='expression'). Pass the output of run_smooth
                to compute neighbors over a kNN-smoothed expression matrix.

        Returns:
            Dict with operation status
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy import sparse

        n_genes_total = self.adata.n_vars
        subset_type = 'all'
        # Only the 'expression' basis subsets cells; elsewhere None is the
        # honest answer for the analysis record.
        cell_indices = None

        if basis == 'gene_pca':
            # Existing behavior: use gene PCA embedding
            if 'X_gene_pca' not in self.adata.varm:
                raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

            gene_pcs = self.adata.varm['X_gene_pca']
            valid_mask = ~np.isnan(gene_pcs[:, 0])
            valid_indices = np.where(valid_mask)[0]
            representation = gene_pcs[valid_mask, :]

            # Get subset info from gene_pca
            pca_info = self.adata.uns.get('gene_pca', {})
            subset_type = pca_info.get('gene_subset_type', 'all')

        elif basis == 'expression':
            # New path: use raw expression matrix (genes x cells)
            from sklearn.preprocessing import StandardScaler

            # Resolve gene subset
            if gene_subset is not None or self._visible_gene_mask is not None:
                gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            else:
                gene_mask = np.ones(self.adata.n_vars, dtype=bool)

            valid_mask = gene_mask
            valid_indices = np.where(valid_mask)[0]

            # Subset cells if provided
            cell_indices = self._validate_cell_indices(active_cell_indices)
            source_matrix = self._resolve_source_matrix(layer)
            if cell_indices is not None:
                X = source_matrix[cell_indices][:, gene_mask].T
            else:
                X = source_matrix[:, gene_mask].T

            # Densify if sparse
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Optional z-score scaling
            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            representation = X
        else:
            raise ValueError(f"Unknown basis: {basis}. Must be 'gene_pca' or 'expression'.")

        n_genes_valid = len(valid_indices)
        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings/expression")

        # Limit n_neighbors to valid range
        n_neighbors = min(n_neighbors, n_genes_valid - 1)

        # Compute kNN on valid genes
        if metric == 'pearson':
            # Pearson correlation distance: 1 - r
            corr_matrix = np.corrcoef(representation)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            dist_full = 1.0 - corr_matrix
            np.fill_diagonal(dist_full, 0.0)
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='precomputed')
            nn.fit(dist_full)
            distances, indices = nn.kneighbors(dist_full)
        else:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
            nn.fit(representation)
            distances, indices = nn.kneighbors(representation)

        # Build sparse distance matrix (exclude self)
        # Map back to original gene indices
        rows = []
        cols = []
        dists = []
        for i_valid in range(n_genes_valid):
            i_original = valid_indices[i_valid]
            for j_idx in range(1, n_neighbors + 1):  # Skip self (index 0)
                j_valid = indices[i_valid, j_idx]
                j_original = valid_indices[j_valid]
                rows.append(i_original)
                cols.append(j_original)
                dists.append(distances[i_valid, j_idx])

        dist_matrix = sparse.csr_matrix(
            (dists, (rows, cols)),
            shape=(n_genes_total, n_genes_total)
        )

        # Build connectivity matrix (1 / (1 + distance) for weights)
        conn_weights = [1.0 / (1.0 + d) for d in dists]
        conn_matrix = sparse.csr_matrix(
            (conn_weights, (rows, cols)),
            shape=(n_genes_total, n_genes_total)
        )

        # Store in .varp
        self.adata.varp['gene_distances'] = dist_matrix
        self.adata.varp['gene_connectivities'] = conn_matrix

        # Store metadata
        self.adata.uns['gene_neighbors'] = {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
            'n_genes_in_graph': n_genes_valid,
            'gene_subset_type': subset_type,
        }

        result = {
            'status': 'completed',
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
            'n_genes': n_genes_valid,
            'n_genes_total': n_genes_total,
            'gene_subset_type': subset_type,
        }
        self._log_action('gene_neighbors', {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
        }, result, subset=cell_indices)
        return result

    def run_find_similar_genes(
        self,
        gene: str,
        n_neighbors: int = 10,
        use: str = 'connectivities',
    ) -> dict[str, Any]:
        """Find genes with similar expression patterns.

        Args:
            gene: Query gene name
            n_neighbors: Number of similar genes to return
            use: 'connectivities' (similarity) or 'distances'

        Returns:
            Dict with list of similar genes and their scores
        """
        # Check prerequisites
        prereq = self.check_prerequisites('find_similar_genes')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if use not in ('connectivities', 'distances'):
            raise ValueError("`use` must be 'connectivities' or 'distances'")

        # Find gene index
        if gene not in self.adata.var_names:
            raise KeyError(f"Gene '{gene}' not found in dataset")

        gene_idx = self.adata.var_names.get_loc(gene)

        # Get the appropriate matrix
        if use == 'connectivities':
            matrix = self.adata.varp['gene_connectivities']
        else:
            matrix = self.adata.varp['gene_distances']

        # Extract row for query gene
        row = matrix.getrow(gene_idx)
        cols = row.indices
        values = row.data

        # Exclude self
        mask = cols != gene_idx
        cols = cols[mask]
        values = values[mask]

        if len(cols) == 0:
            return {
                'query_gene': gene,
                'similar_genes': [],
                'scores': [],
                'use': use,
            }

        # Sort by similarity (descending for connectivity, ascending for distance)
        if use == 'connectivities':
            order = np.argsort(-values)
        else:
            order = np.argsort(values)

        # Get top k
        top_k = min(n_neighbors, len(order))
        top_indices = cols[order[:top_k]]
        top_scores = values[order[:top_k]]

        similar_genes = [self.adata.var_names[i] for i in top_indices]

        result = {
            'query_gene': gene,
            'similar_genes': similar_genes,
            'scores': top_scores.tolist(),
            'use': use,
        }
        return result

    def run_cluster_genes(
        self,
        resolution: float = 0.5,
        key_added: str = 'gene_cluster',
    ) -> dict[str, Any]:
        """Cluster genes into co-expression modules using Leiden.

        Results are stored in .var[key_added] and .uns['gene_modules'].

        Args:
            resolution: Leiden resolution (higher = more clusters)
            key_added: Column name in .var for cluster labels

        Returns:
            Dict with cluster info and module composition
        """
        import igraph as ig
        import leidenalg

        # Check prerequisites
        prereq = self.check_prerequisites('cluster_genes')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Get connectivity matrix
        conn = self.adata.varp['gene_connectivities']

        # Make symmetric (required for Leiden)
        conn_sym = conn + conn.T
        conn_sym.data = conn_sym.data / 2

        # Identify genes that are in the neighbor graph (have connections)
        # Works for both gene_pca and expression basis
        row_nnz = np.diff(conn_sym.indptr)
        valid_mask = row_nnz > 0
        valid_indices = np.where(valid_mask)[0]
        n_genes_valid = len(valid_indices)

        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings to cluster")

        # Create mapping from original indices to subgraph indices
        original_to_sub = {orig: sub for sub, orig in enumerate(valid_indices)}
        sub_to_original = valid_indices

        # Extract subgraph for valid genes only
        conn_sub = conn_sym[valid_indices, :][:, valid_indices]
        sources_sub, targets_sub = conn_sub.nonzero()
        weights = np.array(conn_sub[sources_sub, targets_sub]).flatten()

        # Build igraph from the subgraph
        g = ig.Graph(directed=False)
        g.add_vertices(n_genes_valid)
        edges = list(zip(sources_sub.tolist(), targets_sub.tolist()))
        g.add_edges(edges)
        g.es['weight'] = weights.tolist()

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
        )

        # Extract cluster assignments for valid genes
        clusters_sub = np.array(partition.membership)
        n_clusters = len(set(clusters_sub))

        # Map back to full gene set: unclustered genes get label 'unclustered'
        cluster_labels = np.full(self.n_genes, 'unclustered', dtype=object)
        for sub_idx, orig_idx in enumerate(sub_to_original):
            cluster_labels[orig_idx] = str(clusters_sub[sub_idx])

        # Store in .var
        self.adata.var[key_added] = pd.Categorical(cluster_labels)

        # Build module dictionary (only for clustered genes)
        modules = {}
        for cluster_id in range(n_clusters):
            # Find genes in this cluster
            sub_mask = clusters_sub == cluster_id
            original_indices = sub_to_original[sub_mask]
            genes_in_cluster = self.adata.var_names[original_indices].tolist()
            modules[f'module_{cluster_id}'] = genes_in_cluster

        self.adata.uns['gene_modules'] = modules

        result = {
            'status': 'completed',
            'key_added': key_added,
            'n_clusters': n_clusters,
            'n_genes_clustered': n_genes_valid,
            'n_genes_unclustered': self.n_genes - n_genes_valid,
            'resolution': resolution,
            'module_sizes': {k: len(v) for k, v in modules.items()},
        }
        self._log_action('cluster_genes', {
            'resolution': resolution,
            'key_added': key_added,
        }, result)
        return result

    def _resolve_gene_indices(
        self, gene_names: list[str], *, use_gene_mask: bool = True
    ) -> tuple[list[str], list[int]]:
        """Map requested gene names to (found_names, .var row indices).

        Unknown names are dropped. When ``use_gene_mask`` (the default) and a
        .var gene mask is active, masked-out genes are skipped — the same rule
        ``_resolve_gene_mask`` applies to every gene-reporting operation. Pass
        False to read the whole .var axis regardless.
        """
        var_names = self.adata.var_names
        active_mask = (
            self._visible_gene_mask
            if (use_gene_mask and self._visible_gene_mask is not None)
            else None
        )
        found_genes: list[str] = []
        gene_idx: list[int] = []
        for name in gene_names:
            if name in var_names:
                loc = int(var_names.get_loc(name))
                if active_mask is not None and not active_mask[loc]:
                    continue
                found_genes.append(name)
                gene_idx.append(loc)
        return found_genes, gene_idx

    def _read_gene_matrix(
        self,
        gene_idx: list[int],
        *,
        cell_indices: list[int] | None = None,
        layer: str | None = None,
    ) -> np.ndarray:
        """Return a (n_genes, n_cells) profile matrix for the given gene indices.

        Source is a user-supplied layer (read directly, no extra normalization)
        or the lazy normalize_total + log1p snapshot of .X; optionally subset to
        ``cell_indices``.
        """
        if layer is not None and layer != 'X':
            source_matrix = self._resolve_source_matrix(layer)
        else:
            source_matrix = self.normalized_adata.X
        if cell_indices is not None:
            if len(cell_indices) == 0:
                raise ValueError("Cell subset is empty")
            cell_arr = np.asarray(cell_indices, dtype=np.int64)
            X = source_matrix[cell_arr, :][:, gene_idx]
        else:
            X = source_matrix[:, gene_idx]
        import scipy.sparse
        if scipy.sparse.issparse(X):
            X = X.toarray()
        # Transpose so each row is one gene's profile across the cells.
        return np.asarray(X).T

    def auto_coexpression_report(
        self,
        gene_names: list[str],
        *,
        cell_indices: list[int] | None = None,
        layer: str | None = None,
        use_gene_mask: bool = True,
        metric: str = 'bicor',
        min_genes: int = 5,
        merge_threshold: float = 0.8,
        purity_threshold: float = 0.5,
        max_split_depth: int = 2,
        min_module_corr: float = 0.2,
    ) -> dict[str, Any]:
        """Auto co-expression modules with diagnostics.

        Same matrix path as ``cluster_gene_set(method='auto')`` but returns the
        structured report ``{"modules", "unassigned", "diagnostics"}`` (modules
        and unassigned are gene-name lists).
        """
        found_genes, gene_idx = self._resolve_gene_indices(
            gene_names, use_gene_mask=use_gene_mask
        )
        if len(found_genes) < 2:
            raise ValueError(
                f"auto_coexpression_report: need at least 2 known genes, "
                f"got {len(found_genes)}"
            )
        X_genes = self._read_gene_matrix(
            gene_idx, cell_indices=cell_indices, layer=layer
        )
        from .gene_coexpression import auto_coexpression_report
        return auto_coexpression_report(
            X_genes, found_genes,
            metric=metric, min_genes=min_genes,
            merge_threshold=merge_threshold,
            purity_threshold=purity_threshold,
            max_split_depth=max_split_depth,
            min_module_corr=min_module_corr,
        )

    def cluster_gene_set(
        self,
        gene_names: list[str],
        method: str,
        k: int | None = None,
        cell_indices: list[int] | None = None,
        eps: float = 0.3,
        min_samples: int = 3,
        layer: str | None = None,
        use_gene_mask: bool = True,
        metric: str = 'bicor',
        min_genes: int = 5,
        merge_threshold: float = 0.8,
        purity_threshold: float = 0.5,
        max_split_depth: int = 2,
        min_module_corr: float = 0.2,
    ) -> list[list[str]]:
        """Cluster a set of genes by expression pattern across cells.

        Uses the normalized (normalize_total + log1p) expression matrix so
        results match the rest of the gene-analysis code path.

        Args:
            gene_names: Gene symbols to cluster. Unknown names are dropped.
            method: One of:
                * 'auto' — robust co-expression modules: a robust correlation
                  metric (``metric``), an automatic cluster count, then a
                  refinement pass that splits impure modules, merges
                  near-duplicates, and prunes modules below ``min_genes`` into a
                  trailing "unassigned" group. No K required.
                * 'hierarchical' — Ward linkage on correlation distance, cut
                  at K.
                * 'kmeans' — K-means on raw gene expression vectors.
                * 'dbscan' — density-based clustering on correlation distance.
                  Discovers the number of clusters from the data; genes that
                  don't reach the density threshold are returned as a trailing
                  "noise" cluster.
            k: Number of clusters. Required (>= 2) for hierarchical / kmeans.
                Ignored for dbscan.
            cell_indices: Optional subset of cells. None means all cells.
            eps: DBSCAN only — neighbor radius in correlation-distance space
                (range [0, 2]; lower = stricter co-expression required).
                Two genes are neighbors when ``1 - Pearson(g1, g2) <= eps``,
                i.e. their Pearson correlation is at least ``1 - eps``.
            min_samples: DBSCAN only — minimum genes (including the gene
                itself) within the eps-radius for a gene to be a core point.
                Smaller values find smaller programs but admit more noise.

        Returns:
            A list of clusters, each a list of gene names. Deterministic
            ordering by cluster id. K-means may return fewer than k groups if
            it collapses an empty cluster; DBSCAN may return any number of
            groups (and an additional noise group at the end if any genes
            were unassigned). The UI handles all of these.
        """
        if method not in ('hierarchical', 'kmeans', 'dbscan', 'auto'):
            raise ValueError(f"Unknown method: {method}")
        if method in ('hierarchical', 'kmeans'):
            if k is None or k < 2:
                raise ValueError(f"{method} requires k >= 2, got {k!r}")
        if method == 'dbscan':
            if not (0.0 < eps <= 2.0):
                raise ValueError(f"dbscan eps must be in (0, 2], got {eps!r}")
            if min_samples < 2:
                raise ValueError(
                    f"dbscan min_samples must be >= 2, got {min_samples!r}"
                )

        found_genes, gene_idx = self._resolve_gene_indices(
            gene_names, use_gene_mask=use_gene_mask
        )
        # Hierarchical / kmeans need at least k genes; DBSCAN needs at least
        # min_samples genes to seed any cluster; auto needs at least 2.
        if method == 'auto':
            min_required = 2
        elif method == 'dbscan':
            min_required = min_samples
        else:
            min_required = k
        if len(found_genes) < min_required:
            raise ValueError(
                f"cluster_gene_set: need at least {min_required} known genes, "
                f"got {len(found_genes)}"
            )

        # Shape (n_genes_found, n_cells_subset): each row one gene's profile.
        X_genes = self._read_gene_matrix(
            gene_idx, cell_indices=cell_indices, layer=layer
        )

        if method == 'auto':
            from .gene_coexpression import auto_coexpression_modules
            return auto_coexpression_modules(
                X_genes, found_genes,
                metric=metric, min_genes=min_genes,
                merge_threshold=merge_threshold,
                purity_threshold=purity_threshold,
                max_split_depth=max_split_depth,
                min_module_corr=min_module_corr,
            )

        if method == 'hierarchical':
            from scipy.spatial.distance import pdist
            from scipy.cluster.hierarchy import linkage, fcluster
            dist = pdist(X_genes, metric='correlation')
            # Guard against numerical issues (correlation distance can
            # produce tiny negatives or slight >2 values).
            dist = np.clip(dist, 0, 2)
            if not np.isfinite(dist).all():
                raise ValueError(
                    "Cannot cluster: one or more genes have zero variance "
                    "across the selected cells"
                )
            Z = linkage(dist, method='ward')
            labels = fcluster(Z, t=k, criterion='maxclust')  # 1-indexed
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(X_genes)  # 0-indexed
        else:  # dbscan
            from scipy.spatial.distance import pdist, squareform
            from sklearn.cluster import DBSCAN
            # Pre-compute the full distance matrix so we can guard against
            # zero-variance genes (which produce NaN correlation distances)
            # before handing it to DBSCAN, and to keep the metric identical
            # to the hierarchical path.
            dist_vec = pdist(X_genes, metric='correlation')
            dist_vec = np.clip(dist_vec, 0.0, 2.0)
            if not np.isfinite(dist_vec).all():
                raise ValueError(
                    "Cannot cluster: one or more genes have zero variance "
                    "across the selected cells"
                )
            dist_mat = squareform(dist_vec)
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(dist_mat)  # cluster ids; -1 for noise

        # Partition found_genes by cluster label. For DBSCAN, push the noise
        # bucket (-1) to the end as a single trailing group so the UI can
        # render it as "cluster N" alongside the real clusters.
        partition: dict[int, list[str]] = {}
        for gene, label in zip(found_genes, labels):
            partition.setdefault(int(label), []).append(gene)
        if method == 'dbscan' and -1 in partition:
            noise = partition.pop(-1)
            ordered = [partition[key] for key in sorted(partition.keys())]
            ordered.append(noise)
            return ordered
        return [partition[key] for key in sorted(partition.keys())]

    def run_build_gene_graph(
        self,
        n_pcs: int | None = None,
        scale: bool = True,
        use_kneedle: bool = True,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        active_cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience function: run gene_pca and gene_neighbors in one step.

        Args:
            n_pcs: Number of PCs (None for auto-detection)
            scale: Whether to scale genes before PCA
            use_kneedle: Whether to use Kneedle for PC selection
            n_neighbors: Number of neighbors for kNN graph
            metric: Distance metric for kNN
            active_cell_indices: If provided, use only these cells for gene PCA
            gene_subset: Gene filtering specification passed to gene_pca

        Returns:
            Dict with combined results from both steps
        """
        # Run gene PCA
        pca_result = self.run_gene_pca(
            n_comps=n_pcs,
            scale=scale,
            use_kneedle=use_kneedle,
            active_cell_indices=active_cell_indices,
            gene_subset=gene_subset,
        )

        # Run gene neighbors
        neighbors_result = self.run_gene_neighbors(
            n_neighbors=n_neighbors,
            metric=metric,
        )

        result = {
            'status': 'completed',
            'pca': pca_result,
            'neighbors': neighbors_result,
        }
        return result

    # =========================================================================
    # NMF gene programs
    # =========================================================================

    #: Correlation matrices are genes × genes; past this many genes the
    #: coherence check stops being instant, and no curated set is that large.
    MAX_COHERENCE_GENES = 3000

    def gene_set_coherence(self, genes: list[str], *, cell_indices: list[int] | None = None,
                           layer: str | None = None, metric: str = 'pearson') -> dict[str, Any]:
        """One number before decomposing: how much of the set one pattern explains.

        Pure read over the log-normalised matrix (or a named layer) on the
        chosen cells. See :func:`gene_coexpression.set_coherence`.
        """
        from xcell import gene_coexpression as gc  # noqa: PLC0415

        requested = [str(g) for g in genes]
        found, gene_idx = self._resolve_gene_indices(requested)
        if not found:
            raise ValueError("None of the specified genes found in dataset")
        if len(found) > self.MAX_COHERENCE_GENES:
            raise ValueError(f"Coherence is computed over at most {self.MAX_COHERENCE_GENES} genes; got {len(found)}")
        idx = self._validate_cell_indices(cell_indices)
        cells = None if idx is None else [int(i) for i in idx]
        X = self._read_gene_matrix(gene_idx, cell_indices=cells, layer=layer)
        out = gc.set_coherence(X, metric=metric)
        found_set = set(found)
        out.update({
            'n_genes_used': len(found),
            'genes_missing': [g for g in requested if g not in found_set][:100],
            'n_cells': int(X.shape[1]),
            'metric': metric,
        })
        return out

    def prepare_gene_set_decomposition(
        self,
        genes: list[str],
        *,
        key: str,
        method: str = 'pca',
        k: int = 3,
        loading_threshold: float = 0.2,
        layer: str | None = None,
        transform: str | None = 'log1p',
        cell_indices: list[int] | None = None,
        seed: int = 0,
        specificity_weight: float = 1.0,
        weight_explained: float = 0.5,
        overwrite: bool = False,
    ) -> tuple[Callable[[Callable], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Decompose one gene set into expression programs (PCA or NMF).

        Writes what ``prepare_gene_nmf`` writes, so the whole score-matrix UI
        applies: ``obsm[key]`` (cells × programs, NaN outside the cell scope)
        with a ``uns['xcell_score_matrices']`` entry, ``varm[f'{key}_loadings']``
        (all genes × programs, zero outside the set), and
        ``uns['xcell_gene_set_decomposition'][key]`` with the program gene
        lists, the variance / factor weights and the set's coherence.
        ``k`` is clamped to what the set allows rather than failing.
        """
        import re  # noqa: PLC0415

        from scipy import sparse as _sp  # noqa: PLC0415

        from xcell import gene_coexpression as gc  # noqa: PLC0415
        from xcell import gene_set_decomposition as gsd  # noqa: PLC0415

        key = re.sub(r'[^A-Za-z0-9_]+', '_', str(key or '')).strip('_')
        if not key:
            raise ValueError("key must contain at least one letter or digit")
        method = str(method or 'pca').lower()
        if method not in ('pca', 'nmf'):
            raise ValueError(f"method must be 'pca' or 'nmf', got '{method}'")
        requested = [str(g) for g in genes]
        found, gene_idx = self._resolve_gene_indices(requested)
        found_set = set(found)
        missing = [g for g in requested if g not in found_set]
        if not found:
            raise ValueError("None of the specified genes found in dataset")
        if len(found) < 2:
            raise ValueError(f"Need at least 2 genes present to decompose; found {len(found)}")
        idx = self._validate_cell_indices(cell_indices)
        cell_idx = np.arange(self.n_cells) if idx is None else np.asarray(idx, dtype=int)
        n_used = int(len(cell_idx))
        if n_used < 3:
            raise ValueError(f"Need at least 3 cells; got {n_used}")
        if key in self.adata.obsm and not overwrite:
            raise ValueError(f"A score matrix named '{key}' already exists in .obsm; choose another key or set overwrite")
        k_max = min(len(found) - 1, n_used - 1) if method == 'pca' else min(len(found), n_used)
        k_eff = int(max(1, min(int(k), k_max)))

        if layer:
            source = self._resolve_source_matrix(layer)
        elif transform == 'log1p':
            source = self.normalized_adata.X
        else:
            source = self.adata.X
        sub_X = source[cell_idx][:, gene_idx]
        sub_X = sub_X.toarray() if _sp.issparse(sub_X) else np.asarray(sub_X)
        sub_X = np.ascontiguousarray(sub_X, dtype=np.float32)

        snap_found, snap_missing = list(found), list(missing)
        snap_seed, snap_thr = int(seed), float(loading_threshold)
        snap_spec, snap_wexp = float(specificity_weight), float(weight_explained)
        snap_layer, snap_transform = layer, transform
        loadings_key = f'{key}_loadings'

        def compute_fn(report: Callable) -> dict[str, Any]:
            report(0.05, 'Measuring coherence…')
            coherence = gc.set_coherence(sub_X.T, metric='pearson')
            if method == 'pca':
                report(0.3, f'PCA ({k_eff} components)…')
                res = gsd.pca_programs(sub_X, snap_found, k=k_eff, loading_threshold=snap_thr, seed=snap_seed)
            else:
                def inner(frac: float, msg: str) -> None:
                    report(0.3 + 0.65 * float(frac), msg)
                res = gsd.nmf_programs(sub_X, snap_found, k=k_eff, seed=snap_seed, max_genes=len(snap_found),
                                       specificity_weight=snap_spec, weight_explained=snap_wexp,
                                       progress_callback=inner)
            res['coherence'] = coherence
            report(1.0, 'Done')
            return res

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            n = self.n_cells
            scores = np.asarray(result['scores'], dtype=np.float32)
            n_prog = int(scores.shape[1])
            names = [p['name'] for p in result['programs']]
            full = np.full((n, n_prog), np.nan, dtype=np.float32)
            full[cell_idx] = scores
            self.adata.obsm[key] = full
            reg = self.adata.uns.get('xcell_score_matrices')
            reg = dict(reg) if isinstance(reg, dict) else {}
            reg[key] = {'columns': names, 'source': 'gene_set_decomposition', 'method': method}
            self.adata.uns['xcell_score_matrices'] = reg

            L = np.zeros((self.adata.n_vars, n_prog), dtype=np.float32)
            L[gene_idx] = np.asarray(result['loadings'], dtype=np.float32)
            self.adata.varm[loadings_key] = L

            params = {
                'genes': snap_found, 'key': key, 'method': method, 'k': k_eff,
                'loading_threshold': snap_thr, 'layer': snap_layer, 'transform': snap_transform,
                'seed': snap_seed, 'specificity_weight': snap_spec, 'weight_explained': snap_wexp,
            }
            programs_map = {}
            for p in result['programs']:
                entry = {kk: vv for kk, vv in p.items() if kk != 'name'}
                programs_map[p['name']] = entry
            store = self.adata.uns.get('xcell_gene_set_decomposition')
            store = dict(store) if isinstance(store, dict) else {}
            store[key] = {
                'key': key, 'method': method, 'k': k_eff,
                'genes_used': snap_found, 'genes_missing': snap_missing[:100],
                'n_cells': n_used, 'program_names': names, 'programs': programs_map,
                'variance_ratio': list(result.get('variance_ratio') or []),
                'factor_weights': list(result.get('factor_weights') or []),
                'coherence': {kk: vv for kk, vv in result['coherence'].items() if vv is not None},
                'params': {kk: vv for kk, vv in params.items() if kk != 'genes' and vv is not None},
            }
            self.adata.uns['xcell_gene_set_decomposition'] = store

            out = {
                'key': key, 'obsm_key': key, 'loadings_key': loadings_key, 'method': method, 'k': k_eff,
                'program_names': names, 'programs': result['programs'],
                'variance_ratio': result.get('variance_ratio'),
                'factor_weights': result.get('factor_weights'),
                'n_dropped': result.get('n_dropped', 0),
                'n_genes_used': len(snap_found), 'genes_missing': snap_missing[:100],
                'n_missing': len(snap_missing), 'n_cells': n_used,
                'coherence': result['coherence'],
            }
            self._log_action('gene_set_decomposition', params, out,
                             subset=(None if idx is None else cell_idx))
            return out

        return compute_fn, apply_fn

    def list_gene_set_decompositions(self) -> list[dict[str, Any]]:
        store = self.adata.uns.get('xcell_gene_set_decomposition')
        if not isinstance(store, dict):
            return []
        out = []
        for key, rec in store.items():
            if not isinstance(rec, dict):
                continue
            out.append({
                'key': str(key), 'method': rec.get('method'),
                'n_programs': len(rec.get('program_names', [])),
                'n_genes_used': len(rec.get('genes_used', [])),
                'n_cells': int(rec.get('n_cells', 0)),
            })
        return out

    def get_gene_set_decomposition(self, key: str) -> dict[str, Any]:
        store = self.adata.uns.get('xcell_gene_set_decomposition')
        if not isinstance(store, dict) or key not in store:
            raise KeyError(f"No gene-set decomposition named '{key}'")
        rec = store[key]
        programs = []
        names = list(rec.get('program_names', []))
        pm = rec.get('programs', {})
        for name in names:
            entry = dict(pm.get(name, {}))
            entry['name'] = name
            for fld in ('genes', 'genes_down', 'weights', 'weights_down'):
                entry[fld] = [x.item() if hasattr(x, 'item') else x for x in list(entry.get(fld, []))]
            programs.append(entry)
        return {
            'key': str(key), 'obsm_key': str(key), 'method': rec.get('method'), 'k': int(rec.get('k', len(names))),
            'program_names': names, 'programs': programs,
            'variance_ratio': [float(v) for v in rec.get('variance_ratio', [])],
            'factor_weights': [float(v) for v in rec.get('factor_weights', [])],
            'genes_used': [str(g) for g in rec.get('genes_used', [])],
            'genes_missing': [str(g) for g in rec.get('genes_missing', [])],
            'n_cells': int(rec.get('n_cells', 0)),
            'coherence': {kk: (vv.item() if hasattr(vv, 'item') else (list(vv) if isinstance(vv, (list, tuple, np.ndarray)) else vv))
                          for kk, vv in dict(rec.get('coherence', {})).items()},
            'params': {kk: (vv.item() if hasattr(vv, 'item') else vv) for kk, vv in dict(rec.get('params', {})).items()},
        }

    #: A gene map stores a genes × genes float32 similarity; past this the
    #: matrix alone is 36 MB and the browser has nothing useful to draw.
    MAX_GENE_MAP_GENES = 3000

    def prepare_gene_map(
        self,
        *,
        key: str,
        genes: list[str] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        expression_weight: float = 1.0,
        expression_metric: str = 'bicor',
        annotation_weight: float = 1.0,
        annotation_libraries: list[dict[str, Any]] | None = None,
        string_weight: float = 0.0,
        string_species: str | None = None,
        string_required_score: int = 400,
        go_weight: float = 0.0,
        go_aspect: str = 'bp',
        go_species: str | None = None,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        embedding: str = 'umap',
        layer: str | None = None,
        cell_indices: list[int] | None = None,
        seed: int = 0,
        overwrite: bool = False,
    ) -> tuple[Callable[[Callable], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Build a gene map: similarity from expression, annotation, STRING and GO → modules + 2-D layout.

        Genes come from an explicit list or a ``gene_subset`` (a boolean
        ``.var`` column, a gene list, or a ``{columns, operation}`` spec).
        Annotation libraries must already be in the Gene set library cache;
        naming one that is not is a 400 here, not a failed task. STRING edges
        are fetched inside the task. The GO channel needs the ontology and the
        species GAF on disk — the ``go`` source's one library — and is checked
        here for the same reason. The result lives in
        ``uns['xcell_gene_maps'][key]`` with the similarity matrix itself.
        """
        import re  # noqa: PLC0415

        from xcell import gene_set_sources as gss  # noqa: PLC0415
        from xcell import go_semantic as gos  # noqa: PLC0415

        key = re.sub(r'[^A-Za-z0-9_]+', '_', str(key or '')).strip('_')
        if not key:
            raise ValueError("key must contain at least one letter or digit")
        if genes:
            requested = [str(g) for g in genes]
            found, gene_idx = self._resolve_gene_indices(requested)
            found_set = set(found)
            missing = [g for g in requested if g not in found_set]
            subset_type = 'gene_list'
        elif gene_subset is not None or self._visible_gene_mask is not None:
            mask, subset_type, _meta = self._resolve_gene_mask(gene_subset)
            gene_idx = [int(i) for i in np.flatnonzero(mask)]
            found = [str(self.adata.var_names[i]) for i in gene_idx]
            missing = []
        else:
            raise ValueError("Give either genes or gene_subset")
        if len(found) < 3:
            raise ValueError(f"A gene map needs at least 3 genes present; found {len(found)}")
        if len(found) > self.MAX_GENE_MAP_GENES:
            raise ValueError(f"A gene map holds at most {self.MAX_GENE_MAP_GENES} genes; got {len(found)} — narrow the subset")
        store = self.adata.uns.get('xcell_gene_maps')
        if isinstance(store, dict) and key in store and not overwrite:
            raise ValueError(f"A gene map named '{key}' already exists; choose another key or set overwrite")
        embedding = str(embedding or 'umap').lower()
        if embedding not in ('umap', 'mds'):
            raise ValueError("embedding must be 'umap' or 'mds'")

        idx = self._validate_cell_indices(cell_indices)
        cells = None if idx is None else [int(i) for i in idx]
        n_cells = len(cells) if cells is not None else int(self.n_cells)

        X_genes = None
        if float(expression_weight) > 0:
            X_genes = np.ascontiguousarray(self._read_gene_matrix(gene_idx, cell_indices=cells, layer=layer), dtype=np.float32)

        memberships: dict[str, list[str]] = {}
        libs_used: list[dict[str, Any]] = []
        if float(annotation_weight) > 0 and annotation_libraries:
            lower = {g.upper() for g in found}
            for spec in annotation_libraries:
                if not isinstance(spec, dict) or not spec.get('id'):
                    raise ValueError("Each annotation library needs {source, id}")
                lib = gss.find_library(str(spec.get('source', 'msigdb')), str(spec['id']), spec.get('species'))
                if lib is None:
                    raise ValueError(f"Library '{spec['id']}' ({spec.get('source', 'msigdb')}) has not been fetched yet — fetch it in the Gene set library first")
                n_sets = 0
                for st in lib.get('sets', []):
                    members = [str(m) for m in st.get('genes', []) if str(m).upper() in lower]
                    if members:
                        memberships[f"{lib['id']}:{st['name']}"] = members
                        n_sets += 1
                libs_used.append({'source': lib.get('source'), 'id': lib.get('id'), 'name': lib.get('name'),
                                  'species': lib.get('species'), 'n_sets_overlapping': n_sets})

        species = None
        if float(string_weight) > 0:
            species = string_species or self.guess_species().get('species')
            if species not in gss.SPECIES_TAXON:
                raise ValueError("STRING needs a species ('human' or 'mouse'); the dataset's could not be guessed")
            if len(found) > gss.MAX_STRING_IDENTIFIERS:
                raise ValueError(f"STRING accepts at most {gss.MAX_STRING_IDENTIFIERS} genes per query; got {len(found)}")

        go_sp: str | None = None
        if float(go_weight) > 0:
            go_sp = go_species or string_species or self.guess_species().get('species')
            if go_sp not in gos.GAF_URLS:
                raise ValueError("The GO channel needs a species ('human' or 'mouse'); the dataset's could not be guessed")
            if go_aspect not in gos.ASPECTS:
                raise ValueError(f"go_aspect must be one of {', '.join(gos.ASPECTS)}; got '{go_aspect}'")
            have = gos.availability(go_sp)
            if not (have['obo'] and have['gaf'].get(go_sp)):
                raise ValueError(
                    f"GO annotations for {go_sp} have not been fetched — fetch 'GO annotations ({go_sp})' "
                    "under Gene Ontology in the Gene set library first")

        if X_genes is None and not memberships and species is None and go_sp is None:
            raise ValueError('No similarity channel is available — enable expression, annotation (with a cached library), STRING or GO')

        snap_found, snap_missing = list(found), list(missing)
        snap_w = (float(expression_weight), float(annotation_weight), float(string_weight), float(go_weight))
        snap_go_aspect = str(go_aspect)
        snap_metric, snap_score = str(expression_metric), int(string_required_score)
        snap_nn, snap_res, snap_seed = int(n_neighbors), float(resolution), int(seed)
        snap_layer, snap_subset_type = layer, subset_type

        def compute_fn(report: Callable) -> dict[str, Any]:
            from xcell import gene_similarity as gsim  # noqa: PLC0415

            channels: dict[str, tuple[np.ndarray | None, float]] = {}
            info: dict[str, Any] = {}
            if X_genes is not None:
                report(0.1, f'Expression similarity ({snap_metric})…')
                channels['expression'] = (gsim.expression_similarity(X_genes, metric=snap_metric), snap_w[0])
                info['expression'] = {'weight': snap_w[0], 'metric': snap_metric, 'n_cells': n_cells, 'layer': snap_layer}
            if memberships:
                report(0.3, f'Annotation similarity over {len(memberships)} sets…')
                S_ann, n_terms = gsim.annotation_similarity(snap_found, memberships)
                channels['annotation'] = (S_ann, snap_w[1])
                info['annotation'] = {'weight': snap_w[1], 'libraries': libs_used, 'n_terms': len(memberships),
                                      'n_genes_annotated': int(sum(1 for t in n_terms if t > 0)), 'terms_per_gene': n_terms}
            if species is not None:
                report(0.45, 'Querying STRING…')
                net = gss.string_network(snap_found, species, required_score=snap_score)
                S_str, n_edges = gsim.string_similarity(snap_found, net.get('edges', []))
                channels['string'] = (S_str, snap_w[2])
                info['string'] = {'weight': snap_w[2], 'species': species, 'required_score': snap_score, 'n_edges': n_edges}
            if go_sp is not None:
                report(0.52, f'GO semantic similarity ({snap_go_aspect.upper()})…')
                S_go, go_meta = gsim.go_similarity(snap_found, go_sp, snap_go_aspect)
                channels['go'] = (S_go, snap_w[3])
                info['go'] = {'weight': snap_w[3], 'species': go_sp, 'aspect': snap_go_aspect,
                              'n_genes_annotated': int(go_meta['n_genes_annotated']), 'n_terms': int(go_meta['n_terms']),
                              'terms_per_gene': [int(t) for t in go_meta['terms_per_gene']]}
            report(0.6, 'Modules and layout…')
            out = gsim.build_gene_map(snap_found, channels=channels, n_neighbors=snap_nn,
                                      resolution=snap_res, embedding=embedding, seed=snap_seed)
            out['channels'] = info
            report(1.0, 'Done')
            return out

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            params = {
                'genes': snap_found, 'gene_subset': snap_subset_type, 'key': key,
                'expression_weight': snap_w[0], 'expression_metric': snap_metric,
                'annotation_weight': snap_w[1], 'annotation_libraries': libs_used,
                'string_weight': snap_w[2], 'string_species': species, 'string_required_score': snap_score,
                'go_weight': snap_w[3], 'go_aspect': snap_go_aspect if go_sp is not None else None, 'go_species': go_sp,
                'n_neighbors': snap_nn, 'resolution': snap_res, 'embedding': embedding,
                'layer': snap_layer, 'seed': snap_seed,
            }
            rec = {
                'key': key, 'genes': snap_found, 'genes_missing': snap_missing[:100],
                'coords': np.asarray(result['coords'], dtype=np.float32),
                'modules': [int(m) for m in result['modules']],
                'order': [int(i) for i in result['order']],
                'similarity': np.asarray(result['similarity'], dtype=np.float32),
                'module_sizes': [int(x) for x in result['module_sizes']],
                'n_modules': int(result['n_modules']),
                'channels': result['channels'],
                'channel_weights': {k: float(v) for k, v in result['channel_weights'].items()},
                'params': {k: v for k, v in params.items() if k != 'genes' and v is not None},
                'n_cells': n_cells,
            }
            store = self.adata.uns.get('xcell_gene_maps')
            store = dict(store) if isinstance(store, dict) else {}
            store[key] = rec
            self.adata.uns['xcell_gene_maps'] = store
            out = {
                'key': key, 'n_genes': len(snap_found), 'genes_missing': snap_missing[:100],
                'n_missing': len(snap_missing), 'n_modules': rec['n_modules'],
                'module_sizes': rec['module_sizes'], 'channel_weights': rec['channel_weights'],
                'channels': {k: {kk: vv for kk, vv in v.items() if kk != 'terms_per_gene'} for k, v in result['channels'].items()},
                'embedding': embedding, 'n_cells': n_cells,
            }
            self._log_action('gene_map', params, out, subset=(None if idx is None else np.asarray(cells)))
            return out

        return compute_fn, apply_fn

    def list_gene_maps(self) -> list[dict[str, Any]]:
        store = self.adata.uns.get('xcell_gene_maps')
        if not isinstance(store, dict):
            return []
        return [{
            'key': str(k), 'n_genes': len(r.get('genes', [])), 'n_modules': int(r.get('n_modules', 0)),
            'channel_weights': {kk: float(vv) for kk, vv in dict(r.get('channel_weights', {})).items()},
        } for k, r in store.items() if isinstance(r, dict)]

    def get_gene_map(self, key: str, *, include_similarity: bool = False) -> dict[str, Any]:
        store = self.adata.uns.get('xcell_gene_maps')
        if not isinstance(store, dict) or key not in store:
            raise KeyError(f"No gene map named '{key}'")
        r = store[key]

        def _plain(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {kk: _plain(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_plain(x) for x in v]
            if hasattr(v, 'item'):
                return v.item()
            return v

        out = {
            'key': str(key),
            'genes': [str(g) for g in r.get('genes', [])],
            'coords': [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(r.get('coords'), dtype=float)],
            'modules': [int(m) for m in r.get('modules', [])],
            'order': [int(i) for i in r.get('order', [])],
            'module_sizes': [int(x) for x in r.get('module_sizes', [])],
            'n_modules': int(r.get('n_modules', 0)),
            'channels': _plain(r.get('channels', {})),
            'channel_weights': _plain(r.get('channel_weights', {})),
            'params': _plain(r.get('params', {})),
            'n_cells': int(r.get('n_cells', 0)),
        }
        if include_similarity:
            S = np.asarray(r.get('similarity'), dtype=float)
            out['similarity'] = np.round(S, 3).tolist()
        return out

    def _annotation_species(self, species: str | None = None) -> str:
        from xcell import gene_annotations as ga  # noqa: PLC0415

        sp = species or self.guess_species().get('species')
        if sp not in ga.TAXON:
            hint = f"got '{species}'" if species else "the dataset's could not be guessed; pass species explicitly"
            raise ValueError(f"Gene annotations need a species of 'mouse' or 'human' — {hint}")
        return str(sp)

    def gene_annotations(self, genes: list[str], species: str | None = None,
                         refresh: bool = False) -> dict[str, Any]:
        """MyGene.info annotations for a few genes (cache first). Pure read."""
        from xcell import gene_annotations as ga  # noqa: PLC0415

        sp = self._annotation_species(species)
        recs = ga.get_annotations([str(g) for g in genes], sp, refresh=refresh)
        return {'species': sp, 'annotations': recs}

    def gene_annotation_status(self, species: str | None = None) -> dict[str, Any]:
        """How many of this dataset's genes are already annotated in the cache."""
        from xcell import gene_annotations as ga  # noqa: PLC0415

        genes = [str(g) for g in self.adata.var_names]
        try:
            sp = self._annotation_species(species)
        except ValueError as e:
            return {'species': None, 'n_genes': len(genes), 'n_cached': 0, 'error': str(e)}
        return {'species': sp, 'n_genes': len(genes), 'n_cached': ga.count_cached(sp, genes),
                'cache_path': ga.stats(sp)['path']}

    def prepare_gene_annotation_prefetch(self, species: str | None = None):
        """Warm the annotation cache for every gene in the dataset (background task).

        Nothing is written to the AnnData, so this is not logged as an analysis
        step; it only makes the ⓘ cards open instantly afterwards.
        """
        from xcell import gene_annotations as ga  # noqa: PLC0415

        sp = self._annotation_species(species)
        genes = [str(g) for g in self.adata.var_names]

        def compute_fn(report: Callable) -> dict[str, Any]:
            ga.get_annotations(genes, sp, report=report)
            return {'n_genes': len(genes)}

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            st = ga.stats(sp)
            return {'species': sp, 'n_genes': result['n_genes'],
                    'n_cached': ga.count_cached(sp, genes), 'cache_path': st['path']}

        return compute_fn, apply_fn

    def prepare_cluster_cells_by_gene_set(
        self,
        genes: list[str],
        *,
        key: str,
        n_comps: int = 20,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        run_umap: bool = True,
        scale: bool = True,
        layer: str | None = None,
        transform: str | None = 'log1p',
        cell_indices: list[int] | None = None,
        seed: int = 0,
        overwrite: bool = False,
    ) -> tuple[Callable[[Callable], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Cluster cells on one gene set's genes: PCA → kNN → Leiden (→ UMAP).

        Everything lands under suffixed keys so the dataset's own ``X_pca`` /
        ``neighbors`` / ``leiden`` are never touched: ``obsm['X_pca_<key>']``,
        ``obsp['<key>_connectivities']`` + ``uns['<key>']`` (scanpy's own
        neighbours entry, so the existing UMAP and Leiden routes can re-run on
        this graph through ``graph_key``), ``obs['leiden_<key>']`` and
        ``obsm['X_umap_<key>']``. Cells outside ``cell_indices`` are labelled
        ``unassigned`` and get NaN coordinates — ``run_leiden``'s convention.

        Expression is normalize_total + log1p unless a layer is named, then
        z-scored per gene, so a highly expressed member (a collagen) does not
        own PC1 by magnitude alone. ``n_comps`` and ``n_neighbors`` are clamped
        to what the set and cell count allow rather than failing: a 6-gene set
        has at most 5 components, and that is still a useful clustering.
        """
        import re  # noqa: PLC0415

        key = re.sub(r'[^A-Za-z0-9_]+', '_', str(key or '')).strip('_')
        if not key:
            raise ValueError("key must contain at least one letter or digit")
        requested = [str(g) for g in genes]
        found, gene_idx = self._resolve_gene_indices(requested)
        found_set = set(found)
        missing = [g for g in requested if g not in found_set]
        if not found:
            raise ValueError("None of the specified genes found in dataset")
        if len(found) < 2:
            raise ValueError(f"Need at least 2 genes present to cluster cells on; found {len(found)}")
        idx = self._validate_cell_indices(cell_indices)
        cell_idx = np.arange(self.n_cells) if idx is None else np.asarray(idx, dtype=int)
        n_used = int(len(cell_idx))
        if n_used < 3:
            raise ValueError(f"Need at least 3 cells to cluster; got {n_used}")

        obs_col, pca_key, umap_key = f'leiden_{key}', f'X_pca_{key}', f'X_umap_{key}'
        conn_key, dist_key = f'{key}_connectivities', f'{key}_distances'
        if not overwrite and (obs_col in self.adata.obs or pca_key in self.adata.obsm
                              or conn_key in self.adata.obsp):
            raise ValueError(
                f"A clustering named '{key}' already exists (obs['{obs_col}']); "
                "choose another key or set overwrite")

        k = int(max(1, min(int(n_comps), len(found) - 1, n_used - 1)))
        nn = int(max(2, min(int(n_neighbors), n_used - 1)))
        res = float(resolution)

        if layer:
            source = self._resolve_source_matrix(layer)
        elif transform == 'log1p':
            source = self.normalized_adata.X
        else:
            source = self.adata.X
        from scipy import sparse as _sp  # noqa: PLC0415

        sub_X = source[cell_idx][:, gene_idx]
        sub_X = sub_X.toarray() if _sp.issparse(sub_X) else np.asarray(sub_X)
        sub_X = np.ascontiguousarray(sub_X, dtype=np.float32)

        # Snapshot everything the closures read, so a later mutation of the
        # live AnnData cannot leak into a running task.
        snap_obs_names = [str(x) for x in self.adata.obs_names[cell_idx]]
        snap_found, snap_missing = list(found), list(missing)
        snap_scale, snap_umap, snap_seed = bool(scale), bool(run_umap), int(seed)
        snap_layer, snap_transform = layer, transform

        def compute_fn(report: Callable) -> dict[str, Any]:
            import anndata as _ad  # noqa: PLC0415

            sub = _ad.AnnData(X=sub_X.copy())
            sub.obs_names = snap_obs_names
            sub.var_names = snap_found
            if snap_scale:
                report(0.05, 'Scaling genes…')
                sc.pp.scale(sub, max_value=10)
            report(0.15, f'PCA ({k} components)…')
            sc.tl.pca(sub, n_comps=k, svd_solver='arpack', random_state=snap_seed)
            report(0.4, f'Neighbour graph (k={nn})…')
            sc.pp.neighbors(sub, n_neighbors=nn, n_pcs=k, key_added=key, random_state=snap_seed)
            report(0.6, f'Leiden (resolution {res:g})…')
            sc.tl.leiden(sub, resolution=res, key_added='leiden', neighbors_key=key,
                         flavor='igraph', n_iterations=2, directed=False, random_state=snap_seed)
            umap = None
            if snap_umap:
                report(0.75, 'UMAP…')
                sc.tl.umap(sub, neighbors_key=key, random_state=snap_seed)
                umap = np.asarray(sub.obsm['X_umap'], dtype=np.float32)
            report(1.0, 'Done')
            params = {}
            for pk, pv in dict(sub.uns[key].get('params', {})).items():
                params[pk] = pv.item() if hasattr(pv, 'item') else pv
            return {
                'pca': np.asarray(sub.obsm['X_pca'], dtype=np.float32),
                'variance_ratio': [float(v) for v in sub.uns['pca']['variance_ratio']],
                'labels': [str(x) for x in sub.obs['leiden']],
                'umap': umap,
                'conn': sub.obsp[f'{key}_connectivities'].tocsr(),
                'dist': sub.obsp[f'{key}_distances'].tocsr(),
                'neighbors': {'connectivities_key': conn_key, 'distances_key': dist_key, 'params': params},
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            n = self.n_cells
            pca_full = np.full((n, k), np.nan, dtype=np.float32)
            pca_full[cell_idx] = result['pca']
            self.adata.obsm[pca_key] = pca_full
            if result['umap'] is not None:
                um = np.full((n, 2), np.nan, dtype=np.float32)
                um[cell_idx] = result['umap']
                self.adata.obsm[umap_key] = um
            elif umap_key in self.adata.obsm:
                del self.adata.obsm[umap_key]

            labels = ['unassigned'] * n
            for i, ci in enumerate(cell_idx):
                labels[ci] = result['labels'][i]
            cats = sorted(set(result['labels']), key=lambda t: (len(t), t))
            if n_used < n:
                cats.append('unassigned')
            self.adata.obs[obs_col] = pd.Categorical(labels, categories=cats)

            for mat_key, mat in ((conn_key, result['conn']), (dist_key, result['dist'])):
                coo = mat.tocoo()
                self.adata.obsp[mat_key] = _sp.csr_matrix(
                    (coo.data, (cell_idx[coo.row], cell_idx[coo.col])), shape=(n, n))
            self.adata.uns[key] = result['neighbors']

            sizes = {str(c): int(v) for c, v in pd.Series(result['labels']).value_counts().items()}
            params = {
                'genes': snap_found, 'key': key, 'n_comps': k, 'n_neighbors': nn,
                'resolution': res, 'run_umap': snap_umap, 'scale': snap_scale,
                'layer': snap_layer, 'transform': snap_transform, 'seed': snap_seed,
            }
            reg = self.adata.uns.get('xcell_gene_set_clusterings')
            reg = dict(reg) if isinstance(reg, dict) else {}
            reg[key] = {
                'genes_used': snap_found, 'genes_missing': snap_missing[:100],
                'n_cells': n_used, 'n_clusters': len(sizes), 'cluster_sizes': sizes,
                'variance_ratio': result['variance_ratio'],
                'params': {pk: pv for pk, pv in params.items() if pk != 'genes'},
            }
            self.adata.uns['xcell_gene_set_clusterings'] = reg

            out = {
                'key': key, 'obs_column': obs_col,
                'embedding': umap_key if result['umap'] is not None else None,
                'pca_key': pca_key, 'graph_key': conn_key,
                'n_genes_used': len(snap_found), 'genes_missing': snap_missing[:100],
                'n_missing': len(snap_missing), 'n_cells': n_used,
                'n_comps': k, 'n_neighbors': nn, 'resolution': res,
                'n_clusters': len(sizes), 'cluster_sizes': sizes,
                'variance_ratio': result['variance_ratio'],
            }
            self._log_action('cluster_cells_by_gene_set', params, out,
                             subset=(None if idx is None else cell_idx))
            return out

        return compute_fn, apply_fn

    def prepare_gene_nmf(
        self,
        *,
        k: int = 10,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        cell_indices: list[int] | None = None,
        layer: str | None = None,
        transform: str = 'log1p',
        key: str = 'NMF',
        l1_w: float = 0.0,
        l1_h: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-4,
        seed: int = 0,
        specificity_weight: float = 1.0,
        weight_explained: float = 0.5,
        max_genes: int = 200,
        n_threads: int | None = None,
        overwrite: bool = False,
    ) -> tuple[Callable[[Callable], dict], Callable[[dict], dict]]:
        """Factorize expression into gene programs (GeneNMF, single sample).

        Validates synchronously, snapshots the expression submatrix (sparse
        stays sparse — a 250k-cell matrix is never densified), and returns
        (compute_fn, apply_fn) for the task manager. apply_fn persists:

          * ``.obsm[key]`` (cells x programs usage) + a score-matrix registry
            entry, so each program is colorable like any other score column and
            the matrix doubles as a k-dimensional embedding;
          * ``.varm[f'{key}_loadings']`` (all genes x programs), zero outside
            the fitted gene subset, mirroring how PCA stores ``varm['PCs']``;
          * ``.uns['xcell_gene_nmf'][key]`` with the program gene sets, their
            weights, and the run parameters.

        Args:
            k: number of programs to fit (>= 2).
            gene_subset: None (all genes), a boolean ``.var`` column name such
                as ``'highly_variable'``, an explicit gene list, or the dict
                form ``_resolve_gene_mask`` accepts. NMF is normally run on
                highly variable genes.
            cell_indices: fit on a subset; every other cell scores NaN.
            layer/transform: source matrix, resolved exactly like gene-set
                scoring — a named layer wins, else log1p-normalized or raw .X.
            specificity_weight/weight_explained/max_genes: gene-set extraction,
                see :func:`xcell.gene_nmf.program_genes`.
        """
        import scipy.sparse

        from xcell import gene_nmf as gnmf

        if not isinstance(k, (int, np.integer)) or k < 2:
            raise ValueError(f"k must be an integer >= 2, got {k!r}")
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        if key in self.adata.obsm and not overwrite:
            raise ValueError(
                f"'{key}' already exists (pass overwrite=True to replace it)"
            )

        gene_mask, subset_type, subset_meta = self._resolve_gene_mask(gene_subset)
        gene_idx = np.flatnonzero(gene_mask)
        gene_names = [str(g) for g in self.adata.var_names[gene_idx]]

        if layer is not None and layer != 'X':
            source = self._resolve_source_matrix(layer)
        elif transform == 'log1p':
            source = self.normalized_adata.X
        else:
            source = self.adata.X

        if cell_indices is None:
            cell_arr = None
            n_cells_used = self.n_cells
        else:
            cell_arr = np.asarray(cell_indices, dtype=np.int64)
            if cell_arr.size == 0:
                raise ValueError("Cell selection is empty")
            if cell_arr.min() < 0 or cell_arr.max() >= self.n_cells:
                raise ValueError(
                    f"Cell indices out of range for {self.n_cells} cells"
                )
            n_cells_used = int(cell_arr.size)

        if k > min(n_cells_used, len(gene_names)):
            raise ValueError(
                f"k={k} exceeds the data being factorized ({n_cells_used} "
                f"cells x {len(gene_names)} genes); k must be <= "
                f"{min(n_cells_used, len(gene_names))}"
            )

        # Snapshot: slice rows first (cheap on CSR), then columns, so the
        # background thread owns a private copy and can't observe later edits.
        sub = source if cell_arr is None else source[cell_arr, :]
        sub = sub[:, gene_idx]
        if scipy.sparse.issparse(sub):
            sub = sub.tocsr()
            values = sub.data
        else:
            sub = np.asarray(sub)
            values = sub
        if values.size and float(np.min(values)) < 0.0:
            raise ValueError(
                "NMF needs non-negative expression, but the selected matrix "
                "has negative values. Scaled or centered data will not work — "
                "use raw counts or a log-normalized layer."
            )

        snap = {
            'k': int(k), 'l1_w': float(l1_w), 'l1_h': float(l1_h),
            'max_iter': int(max_iter), 'tol': float(tol), 'seed': int(seed),
            'specificity_weight': float(specificity_weight),
            'weight_explained': float(weight_explained),
            'max_genes': int(max_genes),
        }
        snap_meta = {
            'key': key, 'layer': layer, 'transform': transform,
            'gene_subset': subset_type, 'n_genes_used': len(gene_names),
            'n_cells_used': n_cells_used,
        }

        def compute_fn(report: Callable) -> dict:
            return gnmf.run_gene_programs(
                sub, gene_names, n_threads=n_threads,
                progress_callback=report, **snap,
            )

        def apply_fn(result: dict) -> dict:
            programs = result['programs']
            names = [p['name'] for p in programs]

            scores = np.asarray(result['cell_scores'], dtype=np.float64)
            if cell_arr is None:
                full_scores = scores
            else:
                # Cells the model never saw stay NaN rather than 0 — a zero
                # here would read as "this program is absent", not "not fitted".
                full_scores = np.full(
                    (self.n_cells, scores.shape[1]), np.nan, dtype=np.float64
                )
                full_scores[cell_arr, :] = scores
            self.adata.obsm[key] = full_scores

            loadings = np.asarray(result['loadings'], dtype=np.float64)
            full_loadings = np.zeros(
                (self.n_genes, loadings.shape[1]), dtype=np.float64
            )
            full_loadings[gene_idx, :] = loadings
            self.adata.varm[f'{key}_loadings'] = full_loadings

            reg = self.adata.uns.get('xcell_score_matrices')
            reg = dict(reg) if isinstance(reg, dict) else {}
            reg[key] = {'columns': list(names), 'source': 'gene_nmf'}
            self.adata.uns['xcell_score_matrices'] = reg

            # Programs go in as a mapping, not a list of dicts: h5ad can write
            # nested dicts of lists but not an object array of records.
            record = {
                'params': {**snap, **snap_meta},
                'program_names': list(names),
                'programs': {
                    p['name']: {
                        'genes': list(p['genes']),
                        'weights': [float(w) for w in p['weights']],
                        'index': int(p['index']),
                        'factor_weight': float(result['factor_weights'][i]),
                    }
                    for i, p in enumerate(programs)
                },
                'metrics': {
                    'n_programs': int(result['n_programs']),
                    'n_dropped': int(result['n_dropped']),
                    'n_iter': int(result['n_iter']),
                    'converged': bool(result['converged']),
                    'reconstruction_error': float(result['reconstruction_error']),
                    'relative_error': (
                        float(result['relative_error'])
                        if result['relative_error'] is not None else float('nan')
                    ),
                },
            }
            runs = self.adata.uns.get('xcell_gene_nmf')
            runs = dict(runs) if isinstance(runs, dict) else {}
            runs[key] = record
            self.adata.uns['xcell_gene_nmf'] = runs

            self._log_action('gene_nmf', {**snap, **snap_meta}, {
                'obsm_key': key,
                'n_programs': int(result['n_programs']),
                'n_dropped': int(result['n_dropped']),
                'programs': names,
            })

            return {
                'obsm_key': key,
                'varm_key': f'{key}_loadings',
                'programs': programs,
                'n_programs': int(result['n_programs']),
                'n_dropped': int(result['n_dropped']),
                'n_iter': int(result['n_iter']),
                'converged': bool(result['converged']),
                'reconstruction_error': float(result['reconstruction_error']),
                'relative_error': result['relative_error'],
                'factor_weights': list(result['factor_weights']),
                **snap_meta,
                'params': dict(snap),
            }

        return compute_fn, apply_fn

    def get_gene_nmf_result(self, key: str = 'NMF') -> dict[str, Any] | None:
        """A stored NMF run for instant re-display, or None if there isn't one.

        h5ad round-trips lists as numpy arrays, so everything is re-listified
        and the programs come back in ``program_names`` order.
        """
        runs = self.adata.uns.get('xcell_gene_nmf')
        if not isinstance(runs, dict) or key not in runs:
            return None
        rec = runs[key]
        stored = rec.get('programs') or {}
        programs = []
        for name in [str(n) for n in rec.get('program_names', [])]:
            entry = stored.get(name)
            if entry is None:
                continue
            programs.append({
                'name': name,
                'index': int(entry['index']),
                'genes': [str(g) for g in entry['genes']],
                'weights': [float(w) for w in entry['weights']],
                'factor_weight': float(entry['factor_weight']),
            })
        metrics = {k: v for k, v in dict(rec.get('metrics', {})).items()}
        rel = metrics.get('relative_error')
        if rel is not None and not np.isfinite(rel):
            metrics['relative_error'] = None
        return {
            'key': key,
            'programs': programs,
            'params': dict(rec.get('params', {})),
            'metrics': metrics,
        }

    def prepare_meta_programs(
        self,
        *,
        sample_column: str,
        ks: list[int] | tuple[int, ...] = (4, 5, 6),
        n_mp: int = 10,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        layer: str | None = None,
        transform: str = 'log1p',
        key: str = 'MP',
        min_cells: int = 10,
        l1_w: float = 0.0,
        l1_h: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-4,
        seed: int = 0,
        specificity_weight: float = 5.0,
        weight_explained: float = 0.8,
        max_genes: int = 200,
        metric: str = 'cosine',
        min_confidence: float = 0.5,
        n_threads: int | None = None,
        overwrite: bool = False,
    ) -> tuple[Callable[[Callable], dict], Callable[[dict], dict]]:
        """Meta-programs: NMF per sample per rank, consolidated (GeneNMF).

        Factorizes each group of ``sample_column`` on its own at every rank in
        ``ks``, then clusters all the resulting programs into ``n_mp``
        consensus meta-programs. A program fitted on pooled data can be a batch
        effect; one that recurs independently across samples and ranks cannot,
        and ``sample_coverage`` is what separates the two.

        apply_fn persists ``.obsm[key]`` (per-cell meta-program score, weighted
        mean of the consensus genes) plus a score-matrix registry entry, and
        ``.uns['xcell_gene_nmf_meta'][key]`` with the gene sets, metrics,
        per-sample composition, and the program similarity matrix / clustering
        / leaf order the heatmap needs.
        """
        import scipy.sparse

        from xcell import gene_nmf as gnmf

        if sample_column not in self.adata.obs.columns:
            raise ValueError(f"Column '{sample_column}' not found in .obs")
        series = self.adata.obs[sample_column]
        if pd.api.types.is_float_dtype(series):
            raise ValueError(
                f"Column '{sample_column}' is continuous; meta-programs need a "
                "categorical column identifying samples, sections or donors"
            )
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = [str(c) for c in series.cat.categories]
        else:
            categories = [str(c) for c in pd.unique(series)]
        labels = np.asarray(series.astype(str).values)
        present = [c for c in categories if (labels == c).sum() > 0]
        if len(present) < 2:
            raise ValueError(
                f"Column '{sample_column}' has {len(present)} sample(s); "
                "meta-programs need at least 2 to find what recurs"
            )
        if len(present) > 100:
            raise ValueError(
                f"Column '{sample_column}' has {len(present)} values; that "
                "looks like a per-cell identifier, not a sample grouping"
            )

        ks_list = sorted({int(k) for k in (ks or ())})
        if not ks_list or any(k < 2 for k in ks_list):
            raise ValueError(f"ks must be one or more integers >= 2, got {list(ks)!r}")
        if not isinstance(n_mp, (int, np.integer)) or n_mp < 1:
            raise ValueError(f"n_mp must be an integer >= 1, got {n_mp!r}")
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        if key in self.adata.obsm and not overwrite:
            raise ValueError(
                f"'{key}' already exists (pass overwrite=True to replace it)"
            )

        gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
        gene_idx = np.flatnonzero(gene_mask)
        gene_names = [str(g) for g in self.adata.var_names[gene_idx]]

        if layer is not None and layer != 'X':
            source = self._resolve_source_matrix(layer)
        elif transform == 'log1p':
            source = self.normalized_adata.X
        else:
            source = self.adata.X

        sub = source[:, gene_idx]
        if scipy.sparse.issparse(sub):
            sub = sub.tocsr()
            values = sub.data
        else:
            sub = np.asarray(sub)
            values = sub
        if values.size and float(np.min(values)) < 0.0:
            raise ValueError(
                "NMF needs non-negative expression, but the selected matrix "
                "has negative values. Scaled or centered data will not work — "
                "use raw counts or a log-normalized layer."
            )

        snap = {
            'ks': ks_list, 'n_mp': int(n_mp), 'min_cells': int(min_cells),
            'l1_w': float(l1_w), 'l1_h': float(l1_h), 'max_iter': int(max_iter),
            'tol': float(tol), 'seed': int(seed),
            'specificity_weight': float(specificity_weight),
            'weight_explained': float(weight_explained),
            'max_genes': int(max_genes), 'metric': metric,
            'min_confidence': float(min_confidence),
        }
        snap_meta = {
            'key': key, 'sample_column': sample_column, 'layer': layer,
            'transform': transform, 'gene_subset': subset_type,
            'n_genes_used': len(gene_names), 'n_samples': len(present),
        }
        labels_snap = labels.copy()

        def compute_fn(report: Callable) -> dict:
            return gnmf.run_meta_programs(
                sub, gene_names, labels_snap, samples=present,
                n_threads=n_threads, progress_callback=report, **snap,
            )

        def apply_fn(result: dict) -> dict:
            mps = result['metaprograms']
            names = [mp['name'] for mp in mps]

            self.adata.obsm[key] = np.asarray(
                result['cell_scores'], dtype=np.float64
            )
            reg = self.adata.uns.get('xcell_score_matrices')
            reg = dict(reg) if isinstance(reg, dict) else {}
            reg[key] = {'columns': list(names), 'source': 'gene_nmf_meta'}
            self.adata.uns['xcell_score_matrices'] = reg

            record = {
                'params': {**snap, **snap_meta},
                'metaprogram_names': list(names),
                'metaprograms': {
                    mp['name']: {
                        'genes': list(mp['genes']),
                        'weights': [float(w) for w in mp['weights']],
                        'n_programs': int(mp['n_programs']),
                        'n_genes': int(mp['n_genes']),
                        'sample_coverage': float(mp['sample_coverage']),
                        'silhouette': float(mp['silhouette']),
                        'mean_similarity': float(mp['mean_similarity']),
                    }
                    for mp in mps
                },
                'composition': {
                    name: dict(counts)
                    for name, counts in result['composition'].items()
                },
                'samples': [str(s) for s in result['samples']],
                'program_labels': list(result['program_labels']),
                'program_samples': list(result['program_samples']),
                'program_ks': [int(k) for k in result['program_ks']],
                'clusters': [int(c) for c in result['clusters']],
                'order': [int(i) for i in result['order']],
                'similarity': np.asarray(result['similarity'], dtype=np.float32),
                'skipped': [dict(s) for s in result['skipped']],
                'ks_used': [int(k) for k in result['ks_used']],
                'n_dropped': int(result['n_dropped']),
                'n_programs': int(result['n_programs']),
            }
            runs = self.adata.uns.get('xcell_gene_nmf_meta')
            runs = dict(runs) if isinstance(runs, dict) else {}
            runs[key] = record
            self.adata.uns['xcell_gene_nmf_meta'] = runs

            self._log_action('gene_nmf_meta', {**snap, **snap_meta}, {
                'obsm_key': key,
                'n_metaprograms': len(names),
                'n_programs': int(result['n_programs']),
                'metaprograms': names,
            })

            return {
                'obsm_key': key,
                'metaprograms': mps,
                'composition': result['composition'],
                'samples': [str(s) for s in result['samples']],
                'program_labels': list(result['program_labels']),
                'program_samples': list(result['program_samples']),
                'program_ks': [int(k) for k in result['program_ks']],
                'clusters': [int(c) for c in result['clusters']],
                'order': [int(i) for i in result['order']],
                'similarity': np.asarray(result['similarity']).tolist(),
                'skipped': [dict(s) for s in result['skipped']],
                'ks_used': [int(k) for k in result['ks_used']],
                'n_dropped': int(result['n_dropped']),
                'n_programs': int(result['n_programs']),
                **snap_meta,
                'params': dict(snap),
            }

        return compute_fn, apply_fn

    def get_meta_programs_result(self, key: str = 'MP') -> dict[str, Any] | None:
        """A stored meta-program run for instant re-display, or None."""
        runs = self.adata.uns.get('xcell_gene_nmf_meta')
        if not isinstance(runs, dict) or key not in runs:
            return None
        rec = runs[key]
        stored = rec.get('metaprograms') or {}
        mps = []
        for name in [str(n) for n in rec.get('metaprogram_names', [])]:
            entry = stored.get(name)
            if entry is None:
                continue
            mps.append({
                'name': name,
                'genes': [str(g) for g in entry['genes']],
                'weights': [float(w) for w in entry['weights']],
                'n_programs': int(entry['n_programs']),
                'n_genes': int(entry['n_genes']),
                'sample_coverage': float(entry['sample_coverage']),
                'silhouette': float(entry['silhouette']),
                'mean_similarity': float(entry['mean_similarity']),
            })
        return {
            'key': key,
            'metaprograms': mps,
            'composition': {
                str(n): {str(s): int(v) for s, v in dict(c).items()}
                for n, c in dict(rec.get('composition', {})).items()
            },
            'samples': [str(s) for s in rec.get('samples', [])],
            'program_labels': [str(x) for x in rec.get('program_labels', [])],
            'program_samples': [str(x) for x in rec.get('program_samples', [])],
            'program_ks': [int(k) for k in rec.get('program_ks', [])],
            'clusters': [int(c) for c in rec.get('clusters', [])],
            'order': [int(i) for i in rec.get('order', [])],
            'similarity': np.asarray(rec.get('similarity', [])).tolist(),
            'skipped': [dict(s) for s in rec.get('skipped', [])],
            'ks_used': [int(k) for k in rec.get('ks_used', [])],
            'n_dropped': int(rec.get('n_dropped', 0)),
            'n_programs': int(rec.get('n_programs', 0)),
            'params': dict(rec.get('params', {})),
        }

    def list_meta_program_runs(self) -> list[str]:
        """Keys of every stored meta-program run."""
        runs = self.adata.uns.get('xcell_gene_nmf_meta')
        return sorted(str(k) for k in runs) if isinstance(runs, dict) else []

    def sample_column_candidates(self, max_samples: int = 100) -> list[dict[str, Any]]:
        """Categorical .obs columns that could identify samples.

        Continuous columns are out, and so is anything with one value or with
        so many that it is plainly a per-cell identifier.
        """
        out: list[dict[str, Any]] = []
        for col in self.adata.obs.columns:
            series = self.adata.obs[col]
            if pd.api.types.is_float_dtype(series):
                continue
            values = series.astype(str)
            counts = values.value_counts()
            counts = counts[counts > 0]
            if not (2 <= len(counts) <= max_samples):
                continue
            out.append({
                'name': str(col),
                'n_samples': int(len(counts)),
                'min_cells': int(counts.min()),
                'samples': [str(v) for v in counts.index[:100]],
            })
        return out

    def list_gene_nmf_runs(self) -> list[str]:
        """Keys of every stored NMF run, so the UI can offer them."""
        runs = self.adata.uns.get('xcell_gene_nmf')
        return sorted(str(k) for k in runs) if isinstance(runs, dict) else []

    # =========================================================================
    # Spatial Analysis Methods
    # =========================================================================

    def prepare_spatial_neighbors(
        self,
        n_neighs: int = 6,
        coord_type: str | None = None,
        spatial_key: str | None = None,
        delaunay: bool = False,
        n_rings: int = 1,
        radius: float | None = None,
        section_col: str | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial neighborhood graph computation (cancellable).

        Validates inputs and copies adata upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            n_neighs: Number of spatial neighbors (default 6 for hexagonal grids)
            coord_type: 'grid' for Visium, 'generic' for other, None for auto-detect
            spatial_key: Key in .obsm for spatial coordinates (auto-detected if None)
            delaunay: Use Delaunay triangulation instead of kNN
            n_rings: Number of rings of neighbors for grid coordinates
            radius: Radius for generic coordinates (optional)

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Check prerequisites (fail fast)
        prereq = self.check_prerequisites('spatial_neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Auto-detect spatial key if not provided
        if spatial_key is None:
            spatial_key = self._get_spatial_key()
            if spatial_key is None:
                raise ValueError("No spatial coordinates found in .obsm")

        # Copy adata for background computation (squidpy needs full adata)
        adata_copy = self.adata.copy()

        # Snapshot parameters
        snap_n_neighs = n_neighs
        snap_coord_type = coord_type
        snap_spatial_key = spatial_key
        snap_delaunay = delaunay
        snap_n_rings = n_rings
        snap_radius = radius
        snap_sections = self._resolve_sections(section_col)

        def compute_fn() -> dict[str, Any]:
            import squidpy as sq

            sq.gr.spatial_neighbors(
                adata_copy,
                n_neighs=snap_n_neighs,
                coord_type=snap_coord_type,
                spatial_key=snap_spatial_key,
                delaunay=snap_delaunay,
                n_rings=snap_n_rings,
                radius=snap_radius,
            )

            # Optionally make the graph block-diagonal across sections.
            if snap_sections is not None:
                _drop_cross_section_edges(adata_copy, snap_sections)

            return {
                'spatial_connectivities': adata_copy.obsp['spatial_connectivities'],
                'spatial_distances': adata_copy.obsp['spatial_distances'],
                'uns_spatial_neighbors': adata_copy.uns.get('spatial_neighbors', {}),
                'n_edges': adata_copy.obsp['spatial_connectivities'].nnz,
                'spatial_key': snap_spatial_key,
                'n_neighs': snap_n_neighs,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            self.adata.obsp['spatial_connectivities'] = result['spatial_connectivities']
            self.adata.obsp['spatial_distances'] = result['spatial_distances']

            # Copy any uns keys squidpy set
            if result['uns_spatial_neighbors']:
                self.adata.uns['spatial_neighbors'] = result['uns_spatial_neighbors']

            status_result = {
                'status': 'completed',
                'n_cells': self.n_cells,
                'n_edges': result['n_edges'],
                'spatial_key': result['spatial_key'],
                'n_neighs': result['n_neighs'],
            }
            self._log_action('spatial_neighbors', {
                'n_neighs': result['n_neighs'],
                'coord_type': snap_coord_type,
                'spatial_key': result['spatial_key'],
                'delaunay': snap_delaunay,
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_spatial_neighbors(
        self,
        n_neighs: int = 6,
        coord_type: str | None = None,
        spatial_key: str | None = None,
        delaunay: bool = False,
        n_rings: int = 1,
        radius: float | None = None,
        section_col: str | None = None,
    ) -> dict[str, Any]:
        """Compute spatial neighborhood graph using Squidpy.

        Builds a graph based on spatial proximity of cells/spots.
        Results stored in .obsp['spatial_connectivities'] and .obsp['spatial_distances'].

        Args:
            n_neighs: Number of spatial neighbors (default 6 for hexagonal grids)
            coord_type: 'grid' for Visium, 'generic' for other, None for auto-detect
            spatial_key: Key in .obsm for spatial coordinates (auto-detected if None)
            delaunay: Use Delaunay triangulation instead of kNN
            n_rings: Number of rings of neighbors for grid coordinates
            radius: Radius for generic coordinates (optional)

        Returns:
            Dict with operation status and graph info
        """
        import squidpy as sq

        # Check prerequisites
        prereq = self.check_prerequisites('spatial_neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Auto-detect spatial key if not provided
        if spatial_key is None:
            spatial_key = self._get_spatial_key()
            if spatial_key is None:
                raise ValueError("No spatial coordinates found in .obsm")

        # Build spatial neighbors graph
        sq.gr.spatial_neighbors(
            self.adata,
            n_neighs=n_neighs,
            coord_type=coord_type,
            spatial_key=spatial_key,
            delaunay=delaunay,
            n_rings=n_rings,
            radius=radius,
        )

        # Optionally make the graph block-diagonal across sections.
        sections = self._resolve_sections(section_col)
        if sections is not None:
            _drop_cross_section_edges(self.adata, sections)

        # Get graph stats
        n_edges = self.adata.obsp['spatial_connectivities'].nnz

        result = {
            'status': 'completed',
            'n_cells': self.n_cells,
            'n_edges': n_edges,
            'spatial_key': spatial_key,
            'n_neighs': n_neighs,
        }
        self._log_action('spatial_neighbors', {
            'n_neighs': n_neighs,
            'coord_type': coord_type,
            'spatial_key': spatial_key,
            'delaunay': delaunay,
        }, result)
        return result

    def prepare_spatial_autocorr(
        self,
        mode: str = 'moran',
        genes: list[str] | None = None,
        n_perms: int | None = 100,
        n_jobs: int = 1,
        corr_method: str = 'fdr_bh',
        pval_threshold: float = 0.05,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial autocorrelation computation (cancellable).

        Validates inputs and copies adata upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            mode: 'moran' for Moran's I, 'geary' for Geary's C
            genes: Explicit subset of gene names to test (takes priority over gene_subset)
            n_perms: Number of permutations for p-value (None for analytical only)
            n_jobs: Number of parallel jobs
            corr_method: Multiple testing correction method
            pval_threshold: Threshold for marking genes as spatially variable
            gene_subset: Gene filtering via boolean column (e.g. 'highly_variable'). Ignored if genes is set.

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Check prerequisites (fail fast)
        prereq = self.check_prerequisites('spatial_autocorr')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if mode not in ('moran', 'geary'):
            raise ValueError("mode must be 'moran' or 'geary'")

        # squidpy rejects n_perms=0. Treat 0 as "analytical only" (None).
        if n_perms is not None and n_perms <= 0:
            n_perms = None

        # Resolve gene_subset to gene list if no explicit genes provided
        subset_type = 'all'
        if genes is None and (gene_subset is not None or self._visible_gene_mask is not None):
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            genes = self.adata.var_names[gene_mask].tolist()

        # Copy adata for background computation
        adata_copy = self.adata.copy()

        # Snapshot parameters
        snap_mode = mode
        snap_genes = genes
        snap_n_perms = n_perms
        snap_n_jobs = n_jobs
        snap_corr_method = corr_method
        snap_pval_threshold = pval_threshold
        snap_subset_type = subset_type
        snap_gene_subset = gene_subset

        def compute_fn() -> dict[str, Any]:
            import squidpy as sq

            sq.gr.spatial_autocorr(
                adata_copy,
                mode=snap_mode,
                genes=snap_genes,
                n_perms=snap_n_perms,
                n_jobs=snap_n_jobs,
                corr_method=snap_corr_method,
            )

            # Extract results from .uns
            uns_key = 'moranI' if snap_mode == 'moran' else 'gearyC'
            stat_col = 'I' if snap_mode == 'moran' else 'C'
            results_df = adata_copy.uns[uns_key]

            # Determine p-value column
            if f'pval_{snap_corr_method}' in results_df.columns:
                pval_col = f'pval_{snap_corr_method}'
            elif 'pval_norm_fdr_bh' in results_df.columns:
                pval_col = 'pval_norm_fdr_bh'
            elif 'pval_norm' in results_df.columns:
                pval_col = 'pval_norm'
            else:
                pval_cols = [c for c in results_df.columns if 'pval' in c]
                pval_col = pval_cols[0] if pval_cols else None

            return {
                'uns_key': uns_key,
                'stat_col': stat_col,
                'results_df': results_df,
                'pval_col': pval_col,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            uns_key = result['uns_key']
            stat_col = result['stat_col']
            results_df = result['results_df']
            pval_col = result['pval_col']

            # Store full results DataFrame in .uns
            self.adata.uns[uns_key] = results_df

            # Initialize .var columns with defaults for genes not tested
            self.adata.var[uns_key] = np.nan
            self.adata.var['spatial_pval_adj'] = np.nan
            self.adata.var['spatially_variable'] = False

            # Fill in values for tested genes
            for gene in results_df.index:
                if gene in self.adata.var_names:
                    self.adata.var.loc[gene, uns_key] = results_df.loc[gene, stat_col]
                    if pval_col:
                        self.adata.var.loc[gene, 'spatial_pval_adj'] = results_df.loc[gene, pval_col]

            # Mark spatially variable genes
            if pval_col:
                sv_mask = self.adata.var['spatial_pval_adj'] < snap_pval_threshold
                if snap_mode == 'moran':
                    sv_mask = sv_mask & (self.adata.var[uns_key] > 0)
                else:
                    sv_mask = sv_mask & (self.adata.var[uns_key] < 1)
                self.adata.var['spatially_variable'] = sv_mask

            n_sv_genes = self.adata.var['spatially_variable'].sum()
            n_tested = len(results_df)

            # Get top spatially variable genes
            sv_genes = self.adata.var[self.adata.var['spatially_variable']].copy()
            if snap_mode == 'moran':
                sv_genes = sv_genes.sort_values(uns_key, ascending=False)
            else:
                sv_genes = sv_genes.sort_values(uns_key, ascending=True)
            top_sv_genes = sv_genes.head(20).index.tolist()

            status_result = {
                'status': 'completed',
                'mode': snap_mode,
                'n_tested': n_tested,
                'n_spatially_variable': int(n_sv_genes),
                'pval_threshold': snap_pval_threshold,
                'top_sv_genes': top_sv_genes,
                'gene_subset_type': snap_subset_type,
            }
            self._log_action('spatial_autocorr', {
                'mode': snap_mode,
                'n_perms': snap_n_perms,
                'corr_method': snap_corr_method,
                'pval_threshold': snap_pval_threshold,
                'gene_subset': snap_gene_subset,
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_spatial_autocorr(
        self,
        mode: str = 'moran',
        genes: list[str] | None = None,
        n_perms: int | None = 100,
        n_jobs: int = 1,
        corr_method: str = 'fdr_bh',
        pval_threshold: float = 0.05,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute spatial autocorrelation to identify spatially variable genes.

        Uses Squidpy to compute Moran's I or Geary's C statistics.
        Results stored in:
        - .uns['moranI'] or .uns['gearyC']: Full DataFrame with stats
        - .var['spatially_variable']: Boolean, True if gene passes threshold
        - .var['moranI'] or .var['gearyC']: The autocorrelation statistic
        - .var['spatial_pval_adj']: Adjusted p-value

        Args:
            mode: 'moran' for Moran's I, 'geary' for Geary's C
            genes: Explicit subset of gene names to test (takes priority over gene_subset)
            n_perms: Number of permutations for p-value (None for analytical only)
            n_jobs: Number of parallel jobs
            corr_method: Multiple testing correction method
            pval_threshold: Threshold for marking genes as spatially variable
            gene_subset: Gene filtering via boolean column (e.g. 'highly_variable'). Ignored if genes is set.

        Returns:
            Dict with operation status, number of spatially variable genes
        """
        import squidpy as sq

        # Check prerequisites
        prereq = self.check_prerequisites('spatial_autocorr')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if mode not in ('moran', 'geary'):
            raise ValueError("mode must be 'moran' or 'geary'")

        # squidpy rejects n_perms=0. Treat 0 as "analytical only" (None).
        if n_perms is not None and n_perms <= 0:
            n_perms = None

        # Resolve gene_subset to gene list if no explicit genes provided
        subset_type = 'all'
        if genes is None and (gene_subset is not None or self._visible_gene_mask is not None):
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            genes = self.adata.var_names[gene_mask].tolist()

        # Run spatial autocorrelation
        sq.gr.spatial_autocorr(
            self.adata,
            mode=mode,
            genes=genes,
            n_perms=n_perms,
            n_jobs=n_jobs,
            corr_method=corr_method,
        )

        # Get results from .uns
        uns_key = 'moranI' if mode == 'moran' else 'gearyC'
        stat_col = 'I' if mode == 'moran' else 'C'
        results_df = self.adata.uns[uns_key]

        # Determine p-value column (depends on correction method and n_perms)
        if f'pval_{corr_method}' in results_df.columns:
            pval_col = f'pval_{corr_method}'
        elif 'pval_norm_fdr_bh' in results_df.columns:
            pval_col = 'pval_norm_fdr_bh'
        elif 'pval_norm' in results_df.columns:
            pval_col = 'pval_norm'
        else:
            # Fallback - use first pval column
            pval_cols = [c for c in results_df.columns if 'pval' in c]
            pval_col = pval_cols[0] if pval_cols else None

        # Store in .var for easy filtering
        # Initialize columns with NaN for genes not tested
        self.adata.var[uns_key] = np.nan
        self.adata.var['spatial_pval_adj'] = np.nan
        self.adata.var['spatially_variable'] = False

        # Fill in values for tested genes
        for gene in results_df.index:
            if gene in self.adata.var_names:
                self.adata.var.loc[gene, uns_key] = results_df.loc[gene, stat_col]
                if pval_col:
                    self.adata.var.loc[gene, 'spatial_pval_adj'] = results_df.loc[gene, pval_col]

        # Mark spatially variable genes
        if pval_col:
            sv_mask = self.adata.var['spatial_pval_adj'] < pval_threshold
            # For Moran's I, positive values indicate clustering
            if mode == 'moran':
                sv_mask = sv_mask & (self.adata.var[uns_key] > 0)
            # For Geary's C, values < 1 indicate positive autocorrelation
            else:
                sv_mask = sv_mask & (self.adata.var[uns_key] < 1)
            self.adata.var['spatially_variable'] = sv_mask

        n_sv_genes = self.adata.var['spatially_variable'].sum()
        n_tested = len(results_df)

        # Get top spatially variable genes
        sv_genes = self.adata.var[self.adata.var['spatially_variable']].copy()
        if mode == 'moran':
            sv_genes = sv_genes.sort_values(uns_key, ascending=False)
        else:
            sv_genes = sv_genes.sort_values(uns_key, ascending=True)
        top_sv_genes = sv_genes.head(20).index.tolist()

        result = {
            'status': 'completed',
            'mode': mode,
            'n_tested': n_tested,
            'n_spatially_variable': int(n_sv_genes),
            'pval_threshold': pval_threshold,
            'top_sv_genes': top_sv_genes,
            'gene_subset_type': subset_type,
        }
        self._log_action('spatial_autocorr', {
            'mode': mode,
            'n_perms': n_perms,
            'corr_method': corr_method,
            'pval_threshold': pval_threshold,
            'gene_subset': gene_subset,
        }, result)
        return result

    def prepare_contourize(
        self,
        genes: list[str],
        contour_levels: int = 6,
        log_transform: bool = True,
        smooth_sigma: float = 2.0,
        grid_res: int = 200,
        clip_percentiles: tuple = (1, 99),
        annotation_key: str | None = None,
        section_col: str | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial expression contouring (cancellable).

        Validates inputs and snapshots data upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            genes: List of gene names defining the module
            contour_levels: Number of contour thresholds
            log_transform: Apply log1p before contouring
            smooth_sigma: Gaussian smoothing strength
            grid_res: Interpolation grid size per axis
            clip_percentiles: Percentile clipping range
            annotation_key: Name for the result .obs column (auto-generated if None)

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Validate spatial key (fail fast)
        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError("No spatial coordinates found")

        # Drop genes not in the dataset; proceed with present ones (fail fast
        # only if none remain).
        genes, missing_genes = self._split_present_genes(genes)
        if not genes:
            raise ValueError(
                f"None of the selected genes are present in the data: {missing_genes}")

        # Auto-generate annotation_key if None
        if annotation_key is None:
            annotation_key = f"contour_{genes[0]}_{len(genes)}"

        # Snapshot spatial coordinates and gene expression data
        coords_snap = self.adata.obsm[spatial_key].copy()
        sections_snap = self._resolve_sections(section_col)
        gene_expression_snap = {}
        for g in genes:
            xmat = self.adata[:, g].X
            if hasattr(xmat, 'toarray'):
                gene_expression_snap[g] = xmat.toarray().flatten().copy()
            else:
                gene_expression_snap[g] = np.asarray(xmat).flatten().copy()

        # Snapshot parameters
        snap_genes = list(genes)
        snap_contour_levels = contour_levels
        snap_log_transform = log_transform
        snap_smooth_sigma = smooth_sigma
        snap_grid_res = grid_res
        snap_clip_percentiles = clip_percentiles
        snap_annotation_key = annotation_key
        snap_n_cells = self.adata.n_obs
        snap_missing_genes = missing_genes

        def compute_fn() -> dict[str, Any]:
            # 1-6) Continuous per-cell contour score via the shared core
            gene_expr = {g: gene_expression_snap[g] for g in snap_genes}
            cell_vals, vmax = _contour_score_field(
                coords_snap, gene_expr, snap_log_transform,
                snap_clip_percentiles, snap_grid_res, snap_smooth_sigma,
                sections=sections_snap)

            N = snap_contour_levels
            thresholds = np.linspace(0, vmax, N + 2)[1:-1]

            # 7) Assign each cell the highest threshold it meets
            annotation = np.zeros(snap_n_cells, dtype=float)
            for t in sorted(thresholds):
                mask = cell_vals >= t
                annotation[mask] = t

            # 8) Build ordered categorical data
            cats = np.unique(np.concatenate(([0.0], thresholds)))

            return {
                'annotation': annotation,
                'categories': cats,
                'annotation_key': snap_annotation_key,
                'n_genes': len(snap_genes),
                'genes': snap_genes,
                'contour_levels': snap_contour_levels,
                'n_cells': snap_n_cells,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            annotation_cat = pd.Categorical(
                result['annotation'],
                categories=result['categories'],
                ordered=True,
            )
            self.adata.obs[result['annotation_key']] = annotation_cat

            status_result = {
                'status': 'completed',
                'annotation_key': result['annotation_key'],
                'n_genes': result['n_genes'],
                'genes': result['genes'],
                'missing_genes': snap_missing_genes,
                'contour_levels': result['contour_levels'],
                'n_cells': result['n_cells'],
            }
            self._log_action('contourize', {
                'genes': result['genes'],
                'contour_levels': result['contour_levels'],
                'log_transform': snap_log_transform,
                'smooth_sigma': snap_smooth_sigma,
                'grid_res': snap_grid_res,
                'annotation_key': result['annotation_key'],
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_contourize(
        self,
        genes: list[str],
        contour_levels: int = 6,
        log_transform: bool = True,
        smooth_sigma: float = 2.0,
        grid_res: int = 200,
        clip_percentiles: tuple = (1, 99),
        annotation_key: str | None = None,
        section_col: str | None = None,
    ) -> dict[str, Any]:
        """Compute spatial expression contours from a gene set and assign each cell a contour level.

        For each gene: extract expression, optionally log1p, percentile-clip,
        min-max normalize to [0,1]. Average across genes per cell. Interpolate
        onto a grid, Gaussian smooth, compute N thresholds, and assign each cell
        the highest threshold it meets. Result stored as an ordered categorical
        column in .obs.

        Args:
            genes: List of gene names defining the module
            contour_levels: Number of contour thresholds
            log_transform: Apply log1p before contouring
            smooth_sigma: Gaussian smoothing strength
            grid_res: Interpolation grid size per axis
            clip_percentiles: Percentile clipping range
            annotation_key: Name for the result .obs column (auto-generated if None)
            section_col: Optional .obs column; when set, interpolate per section so
                expression never bleeds across the gap between sections.

        Returns:
            Dict with status, annotation_key, n_genes, genes, missing_genes,
            contour_levels, n_cells
        """
        # Drop genes not in the dataset (e.g. from an imported gene set) and
        # proceed with the present ones; only fail if none remain.
        genes, missing_genes = self._split_present_genes(genes)
        if not genes:
            raise ValueError(
                f"None of the selected genes are present in the data: {missing_genes}")

        # Get spatial coordinates
        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError("No spatial coordinates found")
        coords = self.adata.obsm[spatial_key]
        n_cells = coords.shape[0]
        sections = self._resolve_sections(section_col)

        # Helper for sparse arrays
        def _get_array(xmat):
            return xmat.toarray().flatten() if hasattr(xmat, 'toarray') else xmat.flatten()

        # 1-6) Continuous per-cell contour score via the shared core
        gene_expr = {g: _get_array(self.adata[:, g].X) for g in genes}
        cell_vals, vmax = _contour_score_field(
            coords, gene_expr, log_transform, clip_percentiles, grid_res, smooth_sigma,
            sections=sections)

        N = contour_levels
        thresholds = np.linspace(0, vmax, N + 2)[1:-1]

        # 7) Assign each cell the highest threshold it meets
        annotation = np.zeros(n_cells, dtype=float)
        for t in sorted(thresholds):
            mask = cell_vals >= t
            annotation[mask] = t

        # 8) Store as ordered categorical
        cats = np.unique(np.concatenate(([0.0], thresholds)))
        annotation_cat = pd.Categorical(annotation, categories=cats, ordered=True)

        if annotation_key is None:
            annotation_key = f"contour_{genes[0]}_{len(genes)}"
        self.adata.obs[annotation_key] = annotation_cat

        result = {
            'status': 'completed',
            'annotation_key': annotation_key,
            'n_genes': len(genes),
            'genes': genes,
            'missing_genes': missing_genes,
            'contour_levels': contour_levels,
            'n_cells': n_cells,
        }
        self._log_action('contourize', {
            'genes': genes,
            'contour_levels': contour_levels,
            'log_transform': log_transform,
            'smooth_sigma': smooth_sigma,
            'grid_res': grid_res,
            'annotation_key': annotation_key,
        }, result)
        return result

    def suggest_contour_params(self) -> dict[str, Any]:
        """Data-aware suggested contour parameters for the current dataset.

        Returns a dict with ``grid_res`` (≈√n_spots) and ``smooth_sigma`` (from
        the median spot spacing). Used to prefill the contour UI for both single-
        and multi-gene-set contouring.

        Also returns ``n_spots``, ``median_spacing`` and ``extent``. Sigma is
        measured in grid pixels and a pixel is ``extent / grid_res``, so the UI
        cannot say whether a smoothing setting reaches the neighbouring spot —
        or spans a whole zone — without these. ``median_spacing`` and ``extent``
        are ``None`` when there is nothing to measure.

        Raises:
            ValueError: if no spatial coordinates are present.
        """
        from xcell import multicontour as mc

        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError("No spatial coordinates found")
        coords = self.adata.obsm[spatial_key]
        grid_res = mc.suggest_grid_res(self.adata.n_obs)
        smooth_sigma = mc.suggest_smooth_sigma(coords, grid_res)
        median_spacing, extent = mc.spot_geometry(coords)
        return {
            'grid_res': grid_res,
            'smooth_sigma': round(float(smooth_sigma), 2),
            'n_spots': int(self.adata.n_obs),
            # None rather than NaN: the UI drops the advice that needs these
            # rather than rendering a broken comparison.
            'median_spacing': round(median_spacing, 4) if median_spacing else None,
            'extent': round(extent, 4) if extent else None,
        }

    def check_multicontour_prereqs(self, gene_sets: dict[str, list[str]]) -> None:
        """Cheap up-front validation for multi-contour (raises ValueError).

        Called synchronously by the route so prerequisite failures surface as a
        400 immediately, and again inside :meth:`prepare_multicontour`.
        """
        if 'X_pca' not in self.adata.obsm:
            raise ValueError("Multi-contour requires X_pca — run PCA first.")
        if self._get_spatial_key() is None:
            raise ValueError("No spatial coordinates found")
        if len(gene_sets) < 2:
            raise ValueError("Select at least 2 gene sets for multi-contour")
        # A set with SOME genes missing still runs (on the present ones); only a
        # set with NO present genes is a hard error.
        for name, genes in gene_sets.items():
            present, _ = self._split_present_genes(genes)
            if not present:
                raise ValueError(f"Gene set '{name}' has no genes present in the data")

    def prepare_multicontour(
        self,
        gene_sets: dict[str, list[str]],
        contour_levels: int = 3,
        log_transform: bool = True,
        clip_percentiles: tuple = (1, 99),
        grid_res: int | None = None,
        smooth_sigma: float | None = None,
        section_col: str | None = None,
    ) -> dict[str, Any]:
        """Phase 1 of multi-contour: score each module, cache scores, return review.

        Computes a continuous spatial score and threshold bands for each gene-set
        module and returns a per-module review payload (histograms, auto cutoff)
        without writing anything to ``.obs``. The per-module band assignments are
        cached under an opaque token consumed by :meth:`finalize_multicontour`.

        Args:
            gene_sets: Mapping of tissue label -> list of gene names.
            contour_levels: Number of contour thresholds per module.
            log_transform: Apply log1p before contouring.
            clip_percentiles: Percentile clip range per gene.
            grid_res: Interpolation grid size; auto-suggested from spot count if None.
            smooth_sigma: Gaussian smoothing; auto-suggested from spacing if None.

        Returns:
            Dict with token, modules (list of review dicts), and params.

        Raises:
            ValueError: if X_pca is missing, fewer than 2 gene sets are given,
                        spatial coords are missing, or a gene set has unknown genes.
        """
        import uuid
        from xcell import multicontour as mc

        self.check_multicontour_prereqs(gene_sets)
        spatial_key = self._get_spatial_key()
        sections = self._resolve_sections(section_col)

        if grid_res is None:
            grid_res = mc.suggest_grid_res(self.adata.n_obs)
        if smooth_sigma is None:
            smooth_sigma = mc.suggest_smooth_sigma(
                self.adata.obsm[spatial_key], grid_res)

        modules: list[dict[str, Any]] = []
        scores: dict[str, dict[str, Any]] = {}
        filtered_gene_sets: dict[str, list[str]] = {}
        missing_genes: dict[str, list[str]] = {}
        for name, genes in gene_sets.items():
            present, missing = self._split_present_genes(genes)
            filtered_gene_sets[name] = present  # prereqs guarantee present is non-empty
            if missing:
                missing_genes[name] = missing
            r = mc.score_module(self.adata, present, contour_levels, log_transform,
                                tuple(clip_percentiles), grid_res, smooth_sigma,
                                sections=sections)
            scores[name] = {'bands': r['bands'], 'thresholds': r['thresholds']}
            modules.append({
                'name': name,
                'n_genes': len(present),
                'thresholds': [float(t) for t in r['thresholds']],
                'band_values': [float(v) for v in r['band_values']],
                'histogram': r['histogram'],
                'auto_cutoff': float(r['auto_cutoff']),
            })

        params = {
            'gene_sets': filtered_gene_sets,  # present-only, so finalize re-score is safe
            'contour_levels': contour_levels,
            'log_transform': log_transform,
            'clip_percentiles': list(clip_percentiles),
            'grid_res': grid_res,
            'smooth_sigma': smooth_sigma,
            'section_col': section_col,
        }
        token = uuid.uuid4().hex
        # Cap the cache (band arrays are sizable); evict oldest beyond a few.
        while len(self._multicontour_cache) >= 4:
            self._multicontour_cache.pop(next(iter(self._multicontour_cache)))
        self._multicontour_cache[token] = {'scores': scores, 'params': params}
        return {'token': token, 'modules': modules, 'params': params, 'missing_genes': missing_genes}

    def finalize_multicontour(
        self,
        token: str,
        cutoffs: dict[str, float],
        profile_k: int = 15,
        out_name: str = "tissue",
        save_qc: bool = False,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Phase 2 of multi-contour: binarize, assign, resolve, write column(s).

        Args:
            token: Token returned by :meth:`prepare_multicontour`.
            cutoffs: Mapping module name -> high cutoff (band value).
            profile_k: Nearest unambiguous spatial neighbors to vote over.
            out_name: Name of the result ``.obs`` categorical column.
            save_qc: Also write ``<out_name>_status`` and per-module ``<name>_high``.
            params: Fallback params to recompute scores if the token cache is gone.

        Returns:
            Dict with status, annotation_key, categories, counts, n_resolved.
        """
        from xcell import multicontour as mc

        cached = self._multicontour_cache.get(token)
        if cached is None:
            if params is None:
                raise ValueError("Score cache expired; resubmit with params to recompute")
            scores = {}
            sections = self._resolve_sections(params.get('section_col'))
            for name, genes in params['gene_sets'].items():
                r = mc.score_module(
                    self.adata, genes, params['contour_levels'],
                    params['log_transform'], tuple(params['clip_percentiles']),
                    params['grid_res'], params['smooth_sigma'], sections=sections)
                scores[name] = {'bands': r['bands'], 'thresholds': r['thresholds']}
        else:
            scores = cached['scores']
            params = cached['params']

        highs = {name: mc.binarize(scores[name]['bands'], cutoffs[name]) for name in scores}

        spatial_conn = self.adata.obsp.get('spatial_connectivities')
        coords = np.asarray(self.adata.obsm[self._get_spatial_key()])
        pca = np.asarray(self.adata.obsm['X_pca'])
        sections = self._resolve_sections(params.get('section_col'))
        labels, status = mc.assign_tissue(
            highs, self.adata, profile_k, spatial_conn, pca, coords, sections=sections)

        categories = list(scores.keys()) + ['unassigned']
        self.adata.obs[out_name] = pd.Categorical(labels, categories=categories, ordered=False)
        if save_qc:
            self.adata.obs[f'{out_name}_status'] = pd.Categorical(
                status, categories=['single', 'resolved', 'unassigned'], ordered=False)
            for name in scores:
                self.adata.obs[f'{name}_high'] = pd.Categorical(
                    np.where(highs[name], 'high', 'low'), categories=['low', 'high'], ordered=True)

        counts = {c: int(np.sum(labels == c)) for c in categories}
        result = {
            'status': 'completed',
            'annotation_key': out_name,
            'categories': categories,
            'counts': counts,
            'n_resolved': int(np.sum(status == 'resolved')),
        }
        self._log_action('multicontour', {
            'gene_sets': dict(params['gene_sets']),
            'cutoffs': cutoffs,
            'profile_k': profile_k,
            'out_name': out_name,
            'save_qc': save_qc,
            'params': {k: v for k, v in params.items() if k != 'gene_sets'},
        }, result)
        self._multicontour_cache.pop(token, None)
        return result

    # =========================================================================
    # Territories — hand-drawn regions that annotate cells by occupancy
    # See xcell/territories.py for the geometry.
    # =========================================================================

    # Ragged vertex lists do not survive an h5ad round trip as nested uns
    # structures, so the whole payload is one JSON string — the same choice
    # xcell_lines_json and xcell_analysis_record already make.
    TERRITORY_UNS_KEY = 'xcell_territories_json'

    def get_territories(self) -> dict[str, Any]:
        """Every saved territory type, or {} when none were ever drawn."""
        raw = self.adata.uns.get(self.TERRITORY_UNS_KEY)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            # A corrupt blob must not take the dataset down with it.
            return {}

    def save_territories(self, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Write one territory type into the live AnnData.

        Validated here rather than at draw time so a payload arriving from an
        import or a script cannot put geometry in that names an embedding this
        dataset does not have — which would assign an empty column and look
        like the drawing was wrong.
        """
        import datetime

        if not type_name or not str(type_name).strip():
            raise ValueError('Territory type needs a name')
        embedding = payload.get('embedding') or 'spatial'
        if embedding not in self.adata.obsm:
            raise ValueError(
                f"Territories reference embedding '{embedding}', which this "
                f"dataset does not have. Available: {sorted(self.adata.obsm)}"
            )
        sections = payload.get('sections') or {}
        if not sections:
            raise ValueError(f"Territory type '{type_name}' has no sections")
        for name, block in sections.items():
            if len(block.get('ring') or []) < 3:
                raise ValueError(f"Section '{name}' has no ring")

        stored = self.get_territories()
        stored[type_name] = {
            **payload,
            'embedding': embedding,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        result = {
            'type': type_name,
            'n_sections': len(sections),
            'n_cuts': sum(len(b.get('cuts') or []) for b in sections.values()),
            'n_named': len({a['name'] for b in sections.values()
                            for a in (b.get('anchors') or [])}),
        }
        self._log_action('save_territories', {'type': type_name}, result)
        return result

    def import_territories(self, payload: dict[str, Any], embedding: str) -> dict[str, Any]:
        """Adopt another dataset's territories, retargeted at a local embedding.

        The geometry belongs to the *reference's* coordinate space, which on
        this dataset is the predicted embedding — so the embedding key is
        rewritten, and the source is stamped so a later reader can tell drawn
        regions from borrowed ones.
        """
        import datetime

        if embedding not in self.adata.obsm:
            raise ValueError(
                f"Cannot import territories into '{embedding}': this dataset "
                f"has no such embedding."
            )
        stored = self.get_territories()
        imported: list[str] = []
        stamp = datetime.datetime.now().isoformat(timespec='seconds')
        for name, spec in (payload or {}).items():
            stored[name] = {
                **spec,
                'embedding': embedding,
                'source': f'imported:{self.filepath.name}@{stamp}',
            }
            imported.append(name)
        if imported:
            self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        self._log_action('import_territories',
                         {'embedding': embedding}, {'imported': imported})
        return {'imported': imported}

    TERRITORY_PREFIX = 'territory_'
    TERRITORY_UNASSIGNED = 'unassigned'

    def assign_territories(
        self,
        types: list[str],
        combine: bool = False,
        embedding: str | None = None,
    ) -> dict[str, Any]:
        """Label every cell by the territory it occupies, one column per type.

        Coordinates come from the embedding each type was drawn in, so a type
        imported from a spatial reference reads the *predicted* coordinates on
        this dataset while a locally drawn type reads the real ones.
        """
        from xcell import territories as terr

        stored = self.get_territories()
        missing = [t for t in types if t not in stored]
        if missing:
            raise KeyError(f"No territory type(s): {missing}")
        if combine and len(types) < 2:
            raise ValueError('Combining needs at least two types')

        out: dict[str, Any] = {'types': {}}
        written: list[str] = []

        for type_name in types:
            spec = stored[type_name]
            emb = embedding or spec.get('embedding') or 'spatial'
            if emb not in self.adata.obsm:
                raise ValueError(
                    f"Territory type '{type_name}' was drawn in embedding "
                    f"'{emb}', which this dataset does not have."
                )
            coords = np.asarray(self.adata.obsm[emb], dtype=float)[:, :2]
            placed = np.isfinite(coords).all(axis=1)

            labels = np.full(self.n_cells, self.TERRITORY_UNASSIGNED, dtype=object)
            section_col = spec.get('section_col')
            if section_col and section_col in self.adata.obs.columns:
                section_of = self.adata.obs[section_col].astype(str).values
            else:
                # No column to go on — either the geometry was drawn on one
                # tissue, or it was imported from a reference whose section
                # column this dataset does not have. Either way every section's
                # faces get a chance: the sections are spatially disjoint, so a
                # coordinate falls inside at most one of their rings. Pinning
                # every cell to whichever section came first would leave the
                # whole dataset unassigned whenever that one is not theirs.
                section_of = None

            for section_name, block in spec['sections'].items():
                mask = placed if section_of is None else (
                    (section_of == section_name) & placed)
                if not mask.any():
                    continue
                faces = terr.derive_faces(block['ring'], block.get('cuts') or [])
                names = terr.name_faces(faces, block.get('anchors') or [])
                found = terr.assign(
                    coords[mask], faces, names,
                    unassigned=self.TERRITORY_UNASSIGNED,
                )
                if section_of is None:
                    # Every section sees every cell here, so a later section
                    # must not overwrite a hit an earlier one already made.
                    slots = np.flatnonzero(mask)
                    keep = found != self.TERRITORY_UNASSIGNED
                    labels[slots[keep]] = found[keep]
                else:
                    labels[mask] = found

            series = pd.Series(labels, index=self.adata.obs_names, dtype=object)
            series[~placed] = np.nan          # no coordinate is not "outside"
            column = f'{self.TERRITORY_PREFIX}{type_name}'
            self.adata.obs[column] = pd.Categorical(series)
            written.append(column)

            counts = series.dropna().value_counts()
            out['types'][type_name] = {
                'column': column,
                'counts': {str(k): int(v) for k, v in counts.items()},
                'n_unplaced': int((~placed).sum()),
            }

        if combine:
            parts = [self.adata.obs[f'{self.TERRITORY_PREFIX}{t}'].astype(str)
                     for t in types]
            joined = parts[0]
            for p in parts[1:]:
                joined = joined.str.cat(p, sep='|')
            column = f'{self.TERRITORY_PREFIX}{"__".join(types)}'
            self.adata.obs[column] = pd.Categorical(joined)
            written.append(column)
            out['combined_column'] = column

        out['columns'] = written
        self._log_action('assign_territories',
                         {'types': list(types), 'combine': bool(combine)}, out)
        return out

    def delete_territories(self, type_name: str) -> dict[str, Any]:
        """Forget one territory type. Its .obs columns are left alone."""
        stored = self.get_territories()
        if type_name not in stored:
            raise KeyError(f"No territory type '{type_name}'")
        del stored[type_name]
        self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        self._log_action('delete_territories', {'type': type_name}, {'type': type_name})
        return {'type': type_name, 'remaining': sorted(stored)}

    # =========================================================================
    # Localize — predicting spatial coordinates from a spatial reference
    # See xcell/localize.py for the method and its failure modes.
    # =========================================================================

    # Below this many shared genes a prediction is not worth making: the map
    # still looks smooth and confident and means nothing. Failing loudly beats a
    # caveat nobody reads.
    MIN_LOCALIZE_GENES = 10

    def spatial_reference_bundle(
        self,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        layer: str | None = None,
        section_col: str | None = None,
    ) -> dict[str, Any]:
        """Hand this dataset over as a spatial reference, as plain arrays.

        Localize spans two datasets, which no other route does. Rather than let
        the route reach into this adaptor's AnnData — the one thing the
        architecture reserves for adaptors — the reference exports what the
        query needs and the query consumes plain data.
        """
        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError(
                'This dataset has no spatial coordinates, so it cannot act as a '
                "reference. Expected .obsm['spatial'] or .obsm['X_spatial']."
            )
        # These genes become the shared space with the query dataset, which
        # has its own mask (or none), so this reads the whole reference.
        mask, subset_type, _ = self._resolve_gene_mask(
            gene_subset, apply_visible_mask=False)
        sections = self._resolve_sections(section_col)

        matrix = self._resolve_source_matrix(layer)
        expr = matrix[:, mask]
        expr = expr.toarray() if hasattr(expr, 'toarray') else np.asarray(expr)

        return {
            'expr': np.asarray(expr, dtype=np.float32),
            'coords': np.asarray(self.adata.obsm[spatial_key], dtype=float)[:, :2],
            'genes': [str(g) for g in self.adata.var_names[mask]],
            'sections': sections,
            'n_cells': int(self.n_cells),
            'spatial_key': spatial_key,
            'gene_subset_type': subset_type,
            'section_col': section_col,
            'layer': layer or 'X',
            # Hand-drawn regions travel with the reference: the query's
            # predicted coordinates land in *this* dataset's space, so this
            # geometry is what describes them.
            'territories': self.get_territories(),
        }

    def _align_to_reference(
        self, bundle: dict[str, Any], layer: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Line this dataset's genes up with the reference's, **by name**.

        The pure module takes two aligned matrices and cannot know a panel was
        reordered or only partly shared — matching names is exactly the job that
        needs the AnnData, and getting it wrong produces a confident map of
        nothing.
        """
        ref_genes = list(bundle['genes'])
        positions = {g: i for i, g in enumerate(self.adata.var_names)}
        shared = [g for g in ref_genes if g in positions]
        missing = [g for g in ref_genes if g not in positions]

        if not shared:
            raise ValueError(
                'The query and the spatial reference share no gene names. '
                'Check that both use the same identifiers (symbols vs Ensembl IDs).'
            )
        matrix = self._resolve_source_matrix(layer)
        cols = [positions[g] for g in shared]
        query = matrix[:, cols]
        query = query.toarray() if hasattr(query, 'toarray') else np.asarray(query)

        ref_index = {g: i for i, g in enumerate(ref_genes)}
        ref = bundle['expr'][:, [ref_index[g] for g in shared]]
        return np.asarray(query, dtype=np.float32), ref, shared, missing

    def localize_gene_overlap(
        self, bundle: dict[str, Any], layer: str | None = None,
    ) -> dict[str, Any]:
        """How much of the reference panel this dataset actually carries.

        Reported before anything runs, following the same reasoning as the
        PySingleCellNet coverage check: silently proceeding on a handful of
        shared genes yields confident nonsense.
        """
        ref_genes = list(bundle['genes'])
        present = set(map(str, self.adata.var_names))
        shared = [g for g in ref_genes if g in present]
        missing = [g for g in ref_genes if g not in present]
        n_ref = len(ref_genes) or 1
        frac = len(shared) / n_ref
        return {
            'n_shared': len(shared),
            'n_reference_genes': len(ref_genes),
            'n_query_genes': int(self.n_genes),
            'frac_of_reference': float(frac),
            'missing_from_query': missing[:100],
            'n_missing': len(missing),
            'sufficient': len(shared) >= self.MIN_LOCALIZE_GENES,
            'severity': 'ok' if frac >= 0.5 else ('warn' if frac >= 0.2 else 'error'),
        }

    def prepare_localize(
        self,
        bundle: dict[str, Any],
        *,
        k: int = 15,
        metric: str = 'correlation',
        transform: str = 'zscore',
        aggregation: str = 'weighted_mean',
        min_confidence: float = 0.0,
        epsilon: float = 0.05,
        max_iterations: int = 300,
        layer: str | None = None,
        key_added: str = 'X_spatial_pred',
        import_territories: bool = False,
        assign_territories: bool = False,
    ) -> tuple[Callable[..., Any], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Predict a coordinate for every cell here from a spatial reference.

        Two-phase: validate now so bad input is a 400 rather than a background
        task that dies a minute later, compute with no side effects, then write.
        """
        from xcell import localize as lz

        if metric not in lz.METRICS:
            raise ValueError(f"metric must be one of {list(lz.METRICS)}")
        if transform not in lz.TRANSFORMS:
            raise ValueError(f"transform must be one of {list(lz.TRANSFORMS)}")
        if aggregation not in lz.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {list(lz.AGGREGATIONS)}")
        if aggregation == 'injective':
            # Same check the estimator makes, made here so an impossible run is
            # a 400 the user reads rather than a task that dies a minute in.
            problem = lz.injective_feasibility(
                int(self.n_cells), int(len(bundle['coords'])),
            )
            if problem:
                raise ValueError(problem)
        if aggregation == 'transport':
            # Same reason as above: caught here so a bad dial is a 400 rather
            # than a background task that dies once the matching is done.
            if epsilon <= 0:
                raise ValueError(f'epsilon must be positive, got {epsilon}')
            if max_iterations < 1:
                raise ValueError(
                    f'max_iterations must be at least 1, got {max_iterations}')
        if not key_added:
            raise ValueError('key_added must be a non-empty name')
        if assign_territories and not import_territories:
            raise ValueError(
                'Assigning territories needs the boundaries imported too — '
                'tick "import territory boundaries".'
            )
        snap_territories = dict(bundle.get('territories') or {})

        query_expr, ref_expr, shared, missing = self._align_to_reference(bundle, layer)
        if len(shared) < self.MIN_LOCALIZE_GENES:
            raise ValueError(
                f'Only {len(shared)} genes overlap between the query and the '
                f'spatial reference (need at least {self.MIN_LOCALIZE_GENES}). '
                'A prediction from this few genes would look confident and mean '
                'nothing.'
            )

        # Snapshot everything the closure needs, so it cannot observe state that
        # changed while the task was queued.
        snap = {
            'query': query_expr,
            'ref': ref_expr,
            'coords': np.asarray(bundle['coords'], dtype=float),
            'sections': bundle.get('sections'),
            'k': int(k),
            'metric': metric,
            'transform': transform,
            'aggregation': aggregation,
            'min_confidence': float(min_confidence),
            'epsilon': float(epsilon),
            'max_iterations': int(max_iterations),
            'key': key_added,
            'shared': shared,
            'missing': missing,
            'section_col': bundle.get('section_col'),
            'gene_subset_type': bundle.get('gene_subset_type'),
            'n_reference_cells': int(bundle.get('n_cells', len(bundle['coords']))),
        }

        def compute_fn(report: Callable[[float, str], None] | None = None) -> dict[str, Any]:
            projection = lz.project_knn(
                snap['query'], snap['ref'], snap['coords'],
                k=snap['k'], metric=snap['metric'], transform=snap['transform'],
                aggregation=snap['aggregation'], ref_sections=snap['sections'],
                min_confidence=snap['min_confidence'],
                epsilon=snap['epsilon'], max_iterations=snap['max_iterations'],
                progress=report,
            )
            return {'projection': projection}

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            projection = result['projection']
            key = snap['key']
            self.adata.obsm[key] = np.asarray(projection.coords, dtype=float)
            self.adata.obs[f'{key}_confidence'] = np.asarray(
                projection.confidence, dtype=float)
            self.adata.obs[f'{key}_similarity'] = np.asarray(
                projection.similarity, dtype=float)
            if projection.sections is not None:
                self.adata.obs[f'{key}_section'] = pd.Categorical(
                    projection.sections.astype(str))

            conf = np.asarray(projection.confidence, dtype=float)
            out = {
                'status': 'completed',
                'embedding_name': key,
                'n_cells': int(self.n_cells),
                'n_unplaced': int(projection.n_unplaced),
                'n_shared_genes': len(snap['shared']),
                'n_missing_genes': len(snap['missing']),
                'missing_genes': snap['missing'][:50],
                'n_reference_cells': snap['n_reference_cells'],
                # Which solver produced the assignment. 'candidate' is
                # near-optimal, and saying so is the difference between a
                # documented approximation and a silent one.
                'assignment': projection.assignment,
                'n_candidates': projection.n_candidates,
                # Which reference the coupling was solved against, and whether
                # it converged. A subsampled or budget-exhausted run is a
                # different answer, and the panel has to be able to say so.
                'transport': projection.transport,
                'n_ref_used': projection.n_ref_used,
                'sinkhorn_iterations': projection.sinkhorn_iterations,
                # None rather than NaN when there is nothing to report: this
                # crosses the API, and NaN is not JSON.
                'marginal_error': (
                    float(projection.marginal_error)
                    if projection.marginal_error is not None
                    and np.isfinite(projection.marginal_error)
                    else None
                ),
                'gene_subset_type': snap['gene_subset_type'],
                'section_col': snap['section_col'],
                # So the panel can say what a threshold would cost before it is
                # applied, rather than after.
                'confidence_distribution': {
                    'median': float(np.median(conf)),
                    'q25': float(np.percentile(conf, 25)),
                    'q75': float(np.percentile(conf, 75)),
                    'n_below_0.2': int((conf < 0.2).sum()),
                    'n_below_0.5': int((conf < 0.5).sum()),
                },
            }
            # Territories are carried over only when asked: importing rewrites
            # this dataset's saved geometry, which is not a side effect a
            # coordinate prediction should have by default.
            territory_report: dict[str, Any] = {'imported': [], 'assigned': []}
            if import_territories and snap_territories:
                territory_report.update(
                    self.import_territories(snap_territories, embedding=key))
                if assign_territories and territory_report['imported']:
                    assigned = self.assign_territories(
                        territory_report['imported'], embedding=key)
                    territory_report['assigned'] = assigned['columns']
            out['territories'] = territory_report

            self._log_action('localize', {
                'k': snap['k'], 'metric': snap['metric'],
                'transform': snap['transform'], 'aggregation': snap['aggregation'],
                'min_confidence': snap['min_confidence'],
                'key_added': key, 'section_col': snap['section_col'],
                'gene_subset_type': snap['gene_subset_type'],
            }, {
                'embedding_name': key,
                'n_cells': out['n_cells'],
                'n_unplaced': out['n_unplaced'],
                'n_shared_genes': out['n_shared_genes'],
                'n_reference_cells': out['n_reference_cells'],
                'median_confidence': out['confidence_distribution']['median'],
            })
            return out

        return compute_fn, apply_fn

    def prepare_localize_cross_validation(
        self,
        *,
        k: int = 15,
        metric: str = 'correlation',
        transform: str = 'zscore',
        aggregation: str = 'weighted_mean',
        holdout_fraction: float = 0.2,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        layer: str | None = None,
        section_col: str | None = None,
        groupby: str | None = None,
        seed: int = 0,
    ) -> tuple[Callable[..., Any], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Hold out part of *this* spatial dataset and predict it from the rest.

        What a user does when they have no ground truth for their query. The
        number it returns is optimistic — there is no platform gap to cross
        within one dataset — and the result says so via ``same_platform``.
        """
        from xcell import localize as lz

        bundle = self.spatial_reference_bundle(
            gene_subset=gene_subset, layer=layer, section_col=section_col,
        )
        groups = None
        if groupby is not None:
            if groupby not in self.adata.obs.columns:
                raise ValueError(f"obs column '{groupby}' not found")
            groups = np.asarray(self.adata.obs[groupby].astype(str).values)

        snap = {
            'expr': bundle['expr'], 'coords': bundle['coords'],
            'sections': bundle['sections'], 'groups': groups,
            'k': int(k), 'metric': metric, 'transform': transform,
            'aggregation': aggregation,
            'holdout_fraction': float(holdout_fraction), 'seed': int(seed),
        }

        def compute_fn(report: Callable[[float, str], None] | None = None) -> dict[str, Any]:
            return lz.cross_validate_localization(
                snap['expr'], snap['coords'],
                holdout_fraction=snap['holdout_fraction'],
                ref_sections=snap['sections'], groups=snap['groups'],
                seed=snap['seed'], progress=report,
                k=snap['k'], metric=snap['metric'], transform=snap['transform'],
                aggregation=snap['aggregation'],
            )

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            # Read-only: nothing is written into the AnnData, so there is
            # nothing to apply and nothing to log.
            return result

        return compute_fn, apply_fn

    def evaluate_localization_maps(
        self,
        bundle: dict[str, Any],
        embeddings: list[str],
        gene_sets: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Score one or more predicted maps against the spatial reference.

        Scoring several at once is the point: the value of these numbers is
        comparative, and a user arrives here with a handful of saved variants
        and no way to rank them.

        Args:
            bundle: from the reference's :meth:`spatial_reference_bundle`.
            embeddings: obsm keys on *this* (query) dataset holding predictions.
            gene_sets: marker name -> gene list. Genes absent from either
                dataset are dropped; a set left with none is skipped and named
                in ``skipped_gene_sets``, rather than failing the whole run.
        """
        from xcell import localize_metrics as lm

        if not embeddings:
            raise ValueError('Name at least one predicted embedding to score.')

        ref_coords = np.asarray(bundle['coords'], dtype=float)
        ref_expr = np.asarray(bundle['expr'], dtype=np.float32)
        ref_genes = {str(g): i for i, g in enumerate(bundle['genes'])}

        query_matrix = self._resolve_source_matrix(None)

        marker_sets: list[dict[str, Any]] = []
        skipped: list[str] = []
        for name, genes in gene_sets.items():
            present, _ = self._split_present_genes(list(genes))
            cols = [g for g in present if g in ref_genes]
            if not cols:
                skipped.append(str(name))
                continue
            q = query_matrix[:, [self.adata.var_names.get_loc(g) for g in cols]]
            q = q.toarray() if hasattr(q, 'toarray') else np.asarray(q)
            marker_sets.append({
                'name': name,
                'ref_scores': ref_expr[:, [ref_genes[g] for g in cols]].mean(axis=1),
                'pred_scores': np.asarray(q, dtype=float).mean(axis=1),
            })

        results = []
        for key in embeddings:
            if key not in self.adata.obsm:
                raise ValueError(
                    f"Embedding '{key}' not found in obsm. "
                    f'Available: {list(self.adata.obsm.keys())}'
                )
            pred = np.asarray(self.adata.obsm[key], dtype=float)[:, :2]
            results.append({'embedding': key,
                            **lm.evaluate_map(ref_coords, pred, marker_sets)})

        return {
            'maps': results,
            'skipped_gene_sets': skipped,
            'n_reference_cells': int(ref_coords.shape[0]),
        }

    def localize_reference_geometry(
        self, gene_sets: dict[str, list[str]], layer: str | None = None,
    ) -> dict[str, Any]:
        """Which of these populations would a mean-of-neighbours estimator lose?

        Called on the **reference** adaptor, about itself — there is no query and
        no prediction, which is what lets the answer be shown while the
        parameters are still being chosen. See
        :func:`xcell.localize_metrics.mean_collapse_risk` for what the number is.

        Args:
            gene_sets: population name -> gene list. Genes absent from this
                dataset are dropped; a set left with none is skipped and named
                in ``skipped_gene_sets`` rather than failing the whole call.
            layer: which matrix to score from; ``None`` is ``.X``.
        """
        from xcell import localize_metrics as lm

        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError(
                'This dataset has no spatial coordinates, so it cannot act as a '
                "reference. Expected .obsm['spatial'] or .obsm['X_spatial']."
            )
        coords = np.asarray(self.adata.obsm[spatial_key], dtype=float)[:, :2]
        matrix = self._resolve_source_matrix(layer)

        populations: list[dict[str, Any]] = []
        skipped: list[str] = []
        for name, genes in gene_sets.items():
            present, _ = self._split_present_genes(list(genes))
            if not present:
                skipped.append(str(name))
                continue
            block = matrix[:, [self.adata.var_names.get_loc(g) for g in present]]
            block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
            scores = np.asarray(block, dtype=float).mean(axis=1)
            populations.append({
                'name': str(name),
                'n_genes_used': len(present),
                **lm.mean_collapse_risk(coords, scores),
            })

        # Riskiest first: the UI names the top few, and which few matters.
        populations.sort(key=lambda p: (p['risk'] is None, -(p['risk'] or 0.0)))
        return {
            'populations': populations,
            'skipped_gene_sets': skipped,
            'n_reference_cells': int(coords.shape[0]),
        }

    def evaluate_localization(
        self,
        predicted_key: str,
        truth_key: str,
        *,
        groupby: str | None = None,
        confidence_column: str | None = None,
    ) -> dict[str, Any]:
        """Score a prediction against coordinates already known to be true.

        For a benchmark dataset, or for a spatial dataset deliberately used as a
        query to measure the method.
        """
        from xcell import localize as lz

        for key in (predicted_key, truth_key):
            if key not in self.adata.obsm:
                raise KeyError(
                    f"'{key}' not found in .obsm. Available: "
                    f'{list(self.adata.obsm.keys())}'
                )
        groups = None
        if groupby is not None:
            if groupby not in self.adata.obs.columns:
                raise ValueError(f"obs column '{groupby}' not found")
            groups = np.asarray(self.adata.obs[groupby].astype(str).values)

        column = confidence_column or f'{predicted_key}_confidence'
        confidence = (
            np.asarray(self.adata.obs[column], dtype=float)
            if column in self.adata.obs.columns else None
        )
        return lz.evaluate_localization(
            np.asarray(self.adata.obsm[predicted_key], dtype=float),
            np.asarray(self.adata.obsm[truth_key], dtype=float),
            confidence=confidence,
            groups=groups,
        )

    # =========================================================================
    # Ligand-receptor spatial signaling (CytoSignal-style)
    # =========================================================================
    def _ligrec_spatial_coords(self) -> np.ndarray:
        """Return (n_cells, 2) spatial coordinates or raise if absent."""
        key = self._get_spatial_key()
        if key is None:
            raise ValueError("No spatial coordinates found in .obsm (need 'spatial')")
        return np.asarray(self.adata.obsm[key])[:, :2]

    def suggest_ligrec_params(self) -> dict[str, Any]:
        """Data-driven default parameters for the ligand-receptor tool."""
        from xcell import ligrec
        coords = self._ligrec_spatial_coords()
        radius = ligrec.suggest_radius(coords)
        n_cells = self.adata.n_obs
        # ~100k null draws total; permutations = pool / n_cells, clamped.
        n_perm = int(min(200, max(20, round(100000 / max(1, n_cells)))))
        return {
            "radius": round(float(radius), 3),
            "n_perm": n_perm,
            "min_cells": max(3, int(0.005 * n_cells)),
            "p_thresh": 0.05,
            "median_nn_distance": round(float(ligrec.median_nn_distance(coords)), 4),
        }

    @staticmethod
    def _ligrec_obs_key(ligand: str, receptor: str) -> str:
        """Stable, readable .obs column name for an interaction's score."""
        return f"LR_{ligand}_to_{receptor}"

    def prepare_ligrec(
        self,
        *,
        pairs: list[dict[str, Any]] | None = None,
        radius: float | None = None,
        sigma: float | None = None,
        n_perm: int = 100,
        min_cells: int = 10,
        p_thresh: float = 0.05,
        recep_smooth: bool = False,
        smooth: bool = True,
        types: list[str] | None = None,
        section_col: str | None = None,
        max_pairs: int = 400,
        db_path: str | None = None,
        seed: int = 0,
        progress_callback: Any = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Phase 1 of ligand-receptor scoring: score + test every usable pair.

        Loads the ligand-receptor database (or uses ``pairs``), keeps pairs whose
        every subunit gene is present and expressed in >= ``min_cells`` cells,
        scores them with a spatial permutation null, and persists the full
        per-cell score + significance matrices into the AnnData (``.obsm`` +
        ``.uns['lrscore']``) so any interaction can be visualized later without
        re-running. Returns a ranked per-interaction summary for review.

        Args:
            progress_callback: Optional ``report(frac, message)`` for live progress.

        Returns:
            Dict with summary (ranked), params, n_tested, n_significant,
            n_dropped_capped, interactions.
        """
        from xcell import ligrec as lr

        coords = self._ligrec_spatial_coords()
        if pairs is None:
            pairs = lr.load_lr_database(db_path)
        if types:
            pairs = [p for p in pairs if p["type"] in set(types)]

        # Keep pairs whose every subunit gene is present and expressed enough.
        var_names_list = list(self.adata.var_names)
        var_index = {g: i for i, g in enumerate(var_names_list)}
        # Case-insensitive gene resolution: the bundled database uses human
        # UPPERCASE symbols, so match them to the dataset's actual var names
        # case-insensitively. This makes the human DB work on mouse data
        # (Pdgfb -> PDGFB), mirroring CytoSignal's uppercase-match. Each pair's
        # gene lists are rewritten to the dataset's real names for scoring.
        upper_to_var: dict[str, str] = {}
        for g in var_names_list:
            upper_to_var.setdefault(str(g).upper(), str(g))

        def resolve_gene(g: str) -> str | None:
            if g in var_index:
                return g
            return upper_to_var.get(str(g).upper())

        # Optional gene subset (e.g. highly/spatially-variable .var boolean
        # column, an explicit gene list, or a column-combination spec): restrict
        # eligible genes so only pairs whose subunits all fall in the subset run.
        allowed: set[str] | None = None
        if gene_subset is not None or self._visible_gene_mask is not None:
            mask, _subset_type, _ = self._resolve_gene_mask(gene_subset)
            allowed = {str(g) for g, keep in zip(var_names_list, mask) if keep}

        X = self.adata.X
        Xcsr = X.tocsr() if hasattr(X, "tocsr") else np.asarray(X)
        if hasattr(Xcsr, "getnnz"):
            per_gene_cells = np.asarray((Xcsr > 0).sum(axis=0)).ravel()
        else:
            per_gene_cells = (np.asarray(Xcsr) > 0).sum(axis=0)

        def resolve_genes(genes: list[str]) -> list[str] | None:
            """Resolve subunit genes to actual var names; None if any is missing,
            not expressed in >= min_cells cells, or excluded by the gene subset."""
            actual: list[str] = []
            for g in genes:
                a = resolve_gene(g)
                if a is None or per_gene_cells[var_index[a]] < min_cells:
                    return None
                if allowed is not None and a not in allowed:
                    return None
                actual.append(a)
            return actual

        kept: list[dict[str, Any]] = []
        for p in pairs:
            lig = resolve_genes(p["ligand"])
            rec = resolve_genes(p["receptor"])
            if lig is None or rec is None:
                continue
            kept.append({**p, "ligand": lig, "receptor": rec})
        if not kept:
            raise ValueError(
                "No ligand-receptor pairs have all genes present and expressed "
                f"in >= {min_cells} cells. The bundled database uses human "
                "(UPPERCASE) gene symbols, matched case-insensitively to your "
                "data; check that your dataset's gene names are standard symbols "
                "(e.g. Pdgfb / PDGFB) and lower min_cells if your data is sparse."
            )

        n_dropped_capped = 0
        if len(kept) > max_pairs:
            # Prioritize pairs by total expression of their genes (most-expressed
            # first) so the cap keeps the most informative interactions.
            def expr_weight(p: dict[str, Any]) -> float:
                idx = [var_index[g] for g in p["ligand"] + p["receptor"]]
                return float(per_gene_cells[idx].sum())
            kept = sorted(kept, key=expr_weight, reverse=True)
            n_dropped_capped = len(kept) - max_pairs
            kept = kept[:max_pairs]

        if radius is None:
            radius = lr.suggest_radius(coords)
        sections = self._resolve_sections(section_col)

        result = lr.compute_ligrec(
            Xcsr, list(self.adata.var_names), coords, kept,
            radius=float(radius), sigma=sigma, recep_smooth=recep_smooth,
            smooth=smooth, n_perm=n_perm, p_thresh=p_thresh, sections=sections,
            seed=seed, progress_callback=progress_callback,
        )

        n_significant = sum(1 for s in result["summary"] if s["n_signif"] > 0)
        params = {
            "radius": round(float(radius), 4),
            "n_perm": n_perm,
            "min_cells": min_cells,
            "p_thresh": p_thresh,
            "recep_smooth": recep_smooth,
            "smooth": smooth,
            "section_col": section_col,
            "gene_subset": gene_subset if isinstance(gene_subset, str) else None,
        }
        # Persist the full per-cell score + significance matrices to the AnnData
        # (cell x interaction) so any interaction can be visualized later without
        # re-running. .obsm is the right home (N x P); .obsp would be N x N.
        self.adata.obsm["lrscore"] = np.asarray(result["scores"], dtype=np.float32)
        self.adata.obsm["lrscore_significant"] = np.asarray(
            result["significant"], dtype=np.uint8
        )
        self.adata.uns["lrscore"] = {
            "interactions": list(result["interactions"]),
            "pairs": [
                {"interaction": p["interaction"], "ligand": p["ligand"],
                 "receptor": p["receptor"], "type": p["type"]}
                for p in result["pairs"]
            ],
            "summary": result["summary"],
            "params": params,
            "n_dropped_capped": n_dropped_capped,
        }
        self._log_action('ligrec_analyze', dict(params), {
            "n_tested": len(kept),
            "n_significant": n_significant,
            "n_dropped_capped": n_dropped_capped,
        })
        return {
            "summary": result["summary"],
            "params": params,
            "n_tested": len(kept),
            "n_significant": n_significant,
            "n_dropped_capped": n_dropped_capped,
            "interactions": list(result["interactions"]),
        }

    def get_ligrec_result(self) -> dict[str, Any] | None:
        """Return the stored ligand-receptor result for re-selection, or None.

        Lets the UI re-open the tool and pick different interactions to visualize
        without recomputing, as long as scoring parameters are unchanged.
        """
        stored = self.adata.uns.get("lrscore")
        if stored is None or "lrscore" not in self.adata.obsm:
            return None
        scores = self.adata.obsm["lrscore"]
        n_cells = scores.shape[0]
        summary = list(stored.get("summary", []))
        return {
            "summary": summary,
            "params": dict(stored.get("params", {})),
            "interactions": list(stored.get("interactions", [])),
            "n_tested": len(summary),
            "n_significant": sum(1 for s in summary if s.get("n_signif", 0) > 0),
            "n_dropped_capped": int(stored.get("n_dropped_capped", 0)),
            "n_cells": int(n_cells),
        }

    def finalize_ligrec(
        self,
        interactions: list[str],
        write_significance: bool = False,
    ) -> dict[str, Any]:
        """Phase 2: write selected interactions' score columns to .obs.

        Reads from the persisted ``.obsm['lrscore']`` / ``.uns['lrscore']`` so it
        can be called any number of times after one prepare_ligrec run.

        Args:
            interactions: Interaction ids to write as .obs columns.
            write_significance: Also write a categorical significant/ns column.

        Returns:
            Dict with written column names and annotation_key (first score column).
        """
        stored = self.adata.uns.get("lrscore")
        if stored is None or "lrscore" not in self.adata.obsm:
            raise ValueError("No ligand-receptor result found; run the analysis first")
        index = {it: i for i, it in enumerate(stored["interactions"])}
        pair_by = {p["interaction"]: p for p in stored["pairs"]}
        scores = self.adata.obsm["lrscore"]
        signif = self.adata.obsm.get("lrscore_significant")

        written: list[str] = []
        first_key: str | None = None
        for it in interactions:
            if it not in index:
                continue
            pr = pair_by[it]
            key = self._ligrec_obs_key(
                "_".join(pr["ligand"]), "_".join(pr["receptor"])
            )
            self.adata.obs[key] = np.asarray(scores[:, index[it]], dtype=float)
            written.append(key)
            if first_key is None:
                first_key = key
            if write_significance and signif is not None:
                sig = np.asarray(signif[:, index[it]]) > 0
                sig_key = f"{key}_sig"
                self.adata.obs[sig_key] = pd.Categorical(
                    np.where(sig, "significant", "ns"),
                    categories=["ns", "significant"],
                )
                written.append(sig_key)

        if first_key is None:
            raise ValueError("None of the requested interactions are in this result")

        result = {"written": written, "annotation_key": first_key}
        self._log_action("ligrec", {
            "interactions": interactions,
            "write_significance": write_significance,
        }, result)
        return result

    def prepare_neighborhood(
        self,
        *,
        column: str,
        mode: str = "knn",
        n_neighs: int = 10,
        radius: float | None = None,
        n_perms: int = 1000,
        section_col: str | None = None,
        seed: int = 0,
    ) -> tuple[Callable[[Callable], dict], Callable[[dict], dict]]:
        """Cell-type neighborhood composition + co-location enrichment.

        Validates synchronously, snapshots everything the job needs, and
        returns (compute_fn, apply_fn) for the task manager. compute_fn takes
        the task manager's ``report`` callback (permutations are the long
        part). apply_fn persists:

          * ``.obsm['neighborhood_composition']`` (cells x types neighbor
            fractions) + a score-matrix registry entry, so each type's
            neighborhood fraction is colorable like any score column;
          * ``.uns['xcell_neighborhood']`` with the types x types composition,
            z-score, p/q-value and log2 fold-change matrices for the heatmap.
        """
        prereq = self.check_prerequisites('neighborhood')
        if not prereq['satisfied']:
            raise ValueError(
                "Neighborhood analysis needs spatial coordinates "
                f"(missing: {prereq['missing']})"
            )
        if column not in self.adata.obs.columns:
            raise ValueError(f"Column '{column}' not found in .obs")
        series = self.adata.obs[column]
        if pd.api.types.is_float_dtype(series):
            raise ValueError(
                f"Column '{column}' is continuous; neighborhood enrichment "
                "needs a categorical column (clusters or cell types)"
            )
        if series.isna().any():
            raise ValueError(
                f"Column '{column}' has missing values; fill or subset first"
            )
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = [str(c) for c in series.cat.categories
                          if c in set(series.unique())]
        else:
            categories = [str(c) for c in pd.unique(series)]
        if len(categories) < 2:
            raise ValueError(
                f"Column '{column}' has {len(categories)} category; need >= 2"
            )
        if len(categories) > 200:
            raise ValueError(
                f"Column '{column}' has {len(categories)} categories; that "
                "looks like a per-cell identifier, not a clustering"
            )
        # Graph parameters fail here, synchronously, not inside the task.
        if mode not in ("knn", "radius"):
            raise ValueError(f"Unknown mode '{mode}' (expected 'knn' or 'radius')")
        if mode == "radius" and (radius is None or radius <= 0):
            raise ValueError("radius mode needs a positive radius")
        if mode == "knn" and n_neighs < 1:
            raise ValueError("n_neighs must be >= 1")

        # Snapshot: the background thread must not observe later mutations.
        coords = np.asarray(self.adata.obsm[self._get_spatial_key()])[:, :2].copy()
        labels = np.asarray(series.astype(str).values).copy()
        sections = self._resolve_sections(section_col)
        snap_params = {
            "column": column, "mode": mode, "n_neighs": n_neighs,
            "radius": radius, "n_perms": n_perms, "section_col": section_col,
            "seed": seed,
        }

        def compute_fn(report: Callable) -> dict:
            from xcell import neighborhood as nb
            return nb.compute_neighborhood(
                coords, labels, categories=categories, mode=mode,
                n_neighs=n_neighs, radius=radius, n_perms=n_perms,
                sections=sections, seed=seed, progress_callback=report,
            )

        def apply_fn(result: dict) -> dict:
            cell_comp = result.pop("cell_composition")
            self.adata.obsm["neighborhood_composition"] = np.asarray(
                cell_comp, dtype=np.float32
            )
            reg = self.adata.uns.get('xcell_score_matrices')
            reg = dict(reg) if isinstance(reg, dict) else {}
            reg["neighborhood_composition"] = {
                "columns": list(result["categories"]),
                "source_column": column,
            }
            self.adata.uns['xcell_score_matrices'] = reg

            stored = {**result, "params": dict(snap_params)}
            self.adata.uns["xcell_neighborhood"] = stored

            q = np.array(result["qvals"])
            z = np.array(result["zscores"])
            self._log_action('neighborhood_enrichment', dict(snap_params), {
                "n_types": len(result["categories"]),
                "n_edges": result["n_edges"],
                "n_attracted": int(((q < 0.05) & (z > 0)).sum()),
                "n_avoided": int(((q < 0.05) & (z < 0)).sum()),
            })
            return stored

        return compute_fn, apply_fn

    def get_neighborhood_result(self) -> dict[str, Any] | None:
        """The stored neighborhood result for instant re-display, or None."""
        stored = self.adata.uns.get("xcell_neighborhood")
        if stored is None or "neighborhood_composition" not in self.adata.obsm:
            return None
        out = dict(stored)
        # h5ad round-trips lists as numpy arrays; re-listify for JSON.
        for key in ("composition", "counts", "zscores", "pvals", "qvals",
                    "log2fc", "expected", "categories"):
            if key in out and hasattr(out[key], "tolist"):
                out[key] = out[key].tolist()
        if "params" in out:
            out["params"] = dict(out["params"])
        return out

    def get_spatially_variable_genes(
        self,
        top_n: int | None = None,
        pval_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Get list of spatially variable genes.

        Args:
            top_n: Return only top N genes (sorted by statistic)
            pval_threshold: Override threshold for filtering

        Returns:
            Dict with gene list and statistics
        """
        if 'spatially_variable' not in self.adata.var.columns:
            raise ValueError("spatial_autocorr has not been run")

        # Determine which statistic was used
        if 'moranI' in self.adata.var.columns:
            stat_col = 'moranI'
            ascending = False
        elif 'gearyC' in self.adata.var.columns:
            stat_col = 'gearyC'
            ascending = True
        else:
            raise ValueError("No spatial autocorrelation statistics found")

        # Filter genes
        if pval_threshold is not None:
            mask = (self.adata.var['spatial_pval_adj'] < pval_threshold)
            if stat_col == 'moranI':
                mask = mask & (self.adata.var[stat_col] > 0)
            else:
                mask = mask & (self.adata.var[stat_col] < 1)
        else:
            mask = self.adata.var['spatially_variable']

        sv_genes = self.adata.var[mask].copy()
        sv_genes = sv_genes.sort_values(stat_col, ascending=ascending)

        if top_n is not None:
            sv_genes = sv_genes.head(top_n)

        genes_list = []
        for gene in sv_genes.index:
            genes_list.append({
                'gene': gene,
                'statistic': float(sv_genes.loc[gene, stat_col]),
                'pval_adj': float(sv_genes.loc[gene, 'spatial_pval_adj']),
            })

        return {
            'genes': genes_list,
            'n_total': int(mask.sum()),
            'statistic_type': stat_col,
        }

    def get_gene_modules(self) -> dict[str, Any]:
        """Get gene cluster modules from the last cluster_genes run.

        Returns:
            Dict with modules (dict of module_name -> gene list)
        """
        if 'gene_modules' not in self.adata.uns:
            raise ValueError("cluster_genes has not been run")

        return {
            'modules': self.adata.uns['gene_modules'],
            'n_modules': len(self.adata.uns['gene_modules']),
        }

    def run_marker_genes(
        self,
        obs_column: str,
        groups: list[str] | None = None,
        top_n: int = 25,
        min_in_group_fraction: float | None = None,
        max_out_group_fraction: float | None = None,
        min_fold_change: float | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one-vs-rest marker gene analysis using scanpy.

        Identifies marker genes for each group in a categorical obs column
        using Wilcoxon rank-sum test (one-vs-rest).

        Args:
            obs_column: Name of a categorical column in .obs
            groups: Optional list of group names to include. If None, uses all groups.
            top_n: Number of top marker genes per group
            min_in_group_fraction: Min fraction of cells in group expressing the gene
            max_out_group_fraction: Max fraction of cells outside group expressing the gene
            min_fold_change: Minimum fold change threshold
            gene_subset: Gene filtering specification (str column name, list of genes, or dict spec)

        Returns:
            Dictionary with obs_column, results (per-group gene lists), and params

        Raises:
            ValueError: If column doesn't exist, isn't categorical, or has < 2 groups
        """
        # Validate obs_column exists
        if obs_column not in self.adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in .obs")

        # Validate it's categorical
        dtype = self.adata.obs[obs_column].dtype
        if not pd.api.types.is_categorical_dtype(dtype) and not pd.api.types.is_string_dtype(dtype):
            raise ValueError(f"Column '{obs_column}' is not categorical (dtype: {dtype})")

        # Resolve gene subset
        if gene_subset is not None or self._visible_gene_mask is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            work_adata = self.adata[:, gene_mask].copy()
        else:
            work_adata = self.adata.copy()
            subset_type = 'all'

        # Ensure the column is categorical
        if not pd.api.types.is_categorical_dtype(work_adata.obs[obs_column].dtype):
            work_adata.obs[obs_column] = pd.Categorical(work_adata.obs[obs_column])

        # If groups specified, subset to only those cells
        if groups is not None:
            all_categories = list(work_adata.obs[obs_column].cat.categories)
            invalid = [g for g in groups if g not in all_categories]
            if invalid:
                raise ValueError(f"Groups not found in column '{obs_column}': {invalid}")
            mask = work_adata.obs[obs_column].isin(groups)
            work_adata = work_adata[mask].copy()
            # Remove unused categories after subsetting
            work_adata.obs[obs_column] = work_adata.obs[obs_column].cat.remove_unused_categories()

        # Validate we have at least 2 groups
        n_groups = len(work_adata.obs[obs_column].cat.categories)
        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups for marker gene analysis, got {n_groups}")

        # Run rank_genes_groups (one-vs-rest). use_raw=False so we test against
        # the in-session adata.X (which the user has been preprocessing), not
        # adata.raw — which may carry an older/larger gene index than .X, leading
        # to marker names that don't exist in the current var_names and fail to
        # color/visualize. The two-group diffexp path already uses use_raw=False;
        # this matches that behavior.
        sc.tl.rank_genes_groups(
            work_adata,
            groupby=obs_column,
            method='wilcoxon',
            use_raw=False,
            key_added='marker_genes',
        )

        # Apply filters if specified
        has_filters = any(x is not None for x in [min_in_group_fraction, max_out_group_fraction, min_fold_change])
        if has_filters:
            filter_kwargs: dict[str, Any] = {'key': 'marker_genes', 'key_added': 'marker_genes_filtered'}
            if min_in_group_fraction is not None:
                filter_kwargs['min_in_group_fraction'] = min_in_group_fraction
            if max_out_group_fraction is not None:
                filter_kwargs['max_out_group_fraction'] = max_out_group_fraction
            if min_fold_change is not None:
                filter_kwargs['min_fold_change'] = min_fold_change
            sc.tl.filter_rank_genes_groups(work_adata, **filter_kwargs)

        # Extract results per group
        result_groups = []
        for group in work_adata.obs[obs_column].cat.categories:
            group_str = str(group)
            try:
                if has_filters:
                    df = sc.get.rank_genes_groups_df(work_adata, group=group_str, key='marker_genes_filtered')
                    # filter_rank_genes_groups sets filtered genes to NaN
                    df = df.dropna(subset=['names'])
                else:
                    df = sc.get.rank_genes_groups_df(work_adata, group=group_str, key='marker_genes')

                # Take top N
                df = df.head(top_n)

                genes = []
                for _, row in df.iterrows():
                    genes.append({
                        'gene': str(row['names']),
                        'log2fc': float(row['logfoldchanges']),
                        'pval': float(row['pvals']),
                        'pval_adj': float(row['pvals_adj']),
                    })

                result_groups.append({
                    'group': group_str,
                    'genes': genes,
                })
            except Exception:
                # If a group fails (e.g., too few cells), include empty result
                result_groups.append({
                    'group': group_str,
                    'genes': [],
                })

        self._log_action('marker_genes', {
            'obs_column': obs_column,
            'groups': groups,
            'top_n': top_n,
            'min_in_group_fraction': min_in_group_fraction,
            'max_out_group_fraction': max_out_group_fraction,
            'min_fold_change': min_fold_change,
            'gene_subset': gene_subset,
        }, {
            'n_groups': len(result_groups),
            'total_genes': sum(len(g['genes']) for g in result_groups),
        })

        return {
            'obs_column': obs_column,
            'results': result_groups,
            'gene_subset_type': subset_type,
            'n_genes_tested': work_adata.n_vars,
        }

    # ------------------------------------------------------------------
    # PySingleCellNet cell-type classification (optional dependency)
    # ------------------------------------------------------------------

    def pyscn_status(self) -> dict[str, Any]:
        """Whether PySingleCellNet is importable, plus this dataset's context.

        The UI calls this on modal open to decide between showing the feature
        and showing install instructions, and to pre-fill the pickers.
        """
        from xcell import pyscn

        status = dict(pyscn.availability())
        status['n_cells'] = self.n_cells
        status['n_genes'] = self.n_genes
        status['layers'] = self.list_layers()
        status['categorical_obs'] = [
            c for c in self.adata.obs.columns
            if pd.api.types.is_categorical_dtype(self.adata.obs[c].dtype)
        ]
        status['existing_runs'] = sorted(
            (self.adata.uns.get('pyscn') or {}).keys()
        ) if isinstance(self.adata.uns.get('pyscn'), dict) else []
        return status

    def pyscn_inspect_classifier(self, path: str) -> dict[str, Any]:
        """Describe a classifier file and how well it fits this dataset.

        Deliberately available whether or not PySingleCellNet is installed —
        unpickling and comparing gene lists needs only sklearn — so a user can
        check a classifier against their data before committing to the install.

        Raises:
            ValueError: The file is missing or is not a classifier.
        """
        from xcell import pyscn

        clf, meta = pyscn.load_classifier(path)
        return {
            'classifier': meta,
            'gene_overlap': pyscn.assess_gene_overlap(self.adata.var_names, clf),
            'colors': pyscn.classifier_colors_hex(clf),
        }

    def prepare_pyscn_classify(
        self,
        path: str,
        *,
        key: str = 'SCN',
        layer: str | None = None,
        case_insensitive: bool = False,
        categorize: bool = True,
        quantile: float = 0.05,
        chunk_size: int = 20_000,
    ) -> tuple[Any, Any]:
        """Classify this dataset's cells against a trained PySCN classifier.

        Returns the ``(compute_fn, apply_fn)`` pair the task manager expects.
        Validation happens here, synchronously, so a bad classifier or a
        missing layer is a 400 rather than a failed background task.

        The forest runs on a copy narrowed to the classifier's genes, so the
        live AnnData is untouched until ``apply_fn``.

        Args:
            path: Path to the pickled classifier.
            key: Prefix for everything written — ``<key>_class_argmax``,
                ``<key>_class_score``, ``<key>_class_type`` in .obs and
                ``<key>_score`` in .obsm. Re-running under a different key
                keeps both results side by side.
            layer: Expression source; None reads .X. The top-scoring-pair
                transform is invariant to per-cell monotone rescaling, so
                counts / CPM / log-normalized all behave identically — but
                z-scored data does not, which the UI warns about.
            case_insensitive: Match gene symbols ignoring case.
            categorize: Also derive ``<key>_class_type``
                (Singular / Ambiguous / None / Rand).
            quantile: Per-class score quantile used as the threshold, as in
                PySCN's ``comp_ct_thresh``.
            chunk_size: Cells per prediction call.

        Raises:
            ValueError: Bad classifier, missing layer, or no gene overlap.
        """
        from xcell import pyscn

        # Fail fast, on the request thread.
        pyscn.import_pyscn()
        clf, meta = pyscn.load_classifier(path)
        if not str(key).strip():
            raise ValueError("key must be a non-empty name")
        key = str(key).strip()
        overlap = pyscn.assess_gene_overlap(self.adata.var_names, clf)
        query = pyscn.build_query_adata(
            self.adata, clf, layer=layer, case_insensitive=case_insensitive,
        )
        colors = pyscn.classifier_colors_hex(clf)
        snap_path, snap_key = str(path), key
        snap_cat, snap_q, snap_chunk = bool(categorize), float(quantile), int(chunk_size)

        def compute_fn(report=None):
            def progress(frac, message=None):
                if report is not None:
                    report(frac, message)

            progress(0.02, 'Running the random forest…')
            scores = pyscn.classify_scores(
                query, clf, chunk_size=snap_chunk, progress=progress,
            )
            return {
                'scores': scores,
                'classes': [str(c) for c in scores.columns],
                'colors': colors,
                'gene_overlap': overlap,
                'key': snap_key,
                'categorize': snap_cat,
                'quantile': snap_q,
                'classifier_path': snap_path,
                'layer': layer or 'X',
                'classifier': meta,
            }

        return compute_fn, self._apply_pyscn_result

    def _apply_pyscn_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Write a classification into .obs / .obsm / .uns and summarize it.

        Results are stored the way xcell already stores score matrices — a
        float array in .obsm plus a column-name entry in
        ``uns['xcell_score_matrices']`` — so the existing "color by score" and
        "embed from two scores" UI picks them up with no new plumbing. Colors
        from the classifier go to ``uns['<column>_colors']``, scanpy's
        convention, which xcell's category palette already prefers over its
        generated one.
        """
        from xcell import pyscn

        scores = result['scores']
        key = result['key']
        classes = [str(c) for c in result['classes']]

        summary = pyscn.summarize_scores(scores)
        labels = summary['labels']

        argmax_col = f'{key}_class_argmax'
        score_col = f'{key}_class_score'
        type_col = f'{key}_class_type'
        obsm_key = f'{key}_score'

        # Order categories the way the classifier does, keeping only those
        # actually called — an unused class in the legend is noise.
        present = [c for c in classes if c in set(labels)]
        self.adata.obs[argmax_col] = pd.Categorical(
            labels, categories=present, ordered=False,
        )
        self.adata.obs[score_col] = np.asarray(summary['confidence'], dtype=np.float32)

        colors = result.get('colors') or {}
        self.adata.uns[f'{argmax_col}_colors'] = [
            colors.get(c, '#808080') for c in present
        ]

        thresholds: dict[str, float] = {}
        class_types: list[str] = []
        if result.get('categorize'):
            thresholds = pyscn.compute_thresholds(
                scores, labels, quantile=result.get('quantile', 0.05),
            )
            class_types = pyscn.derive_class_types(scores, labels, thresholds)
            present_types = [t for t in pyscn.CLASS_TYPE_ORDER if t in set(class_types)]
            self.adata.obs[type_col] = pd.Categorical(
                class_types, categories=present_types, ordered=True,
            )
            self.adata.uns[f'{type_col}_colors'] = [
                pyscn.CLASS_TYPE_COLORS[t] for t in present_types
            ]

        self.adata.obsm[obsm_key] = scores.to_numpy(dtype=np.float64, copy=True)
        reg = self.adata.uns.get('xcell_score_matrices')
        reg = dict(reg) if isinstance(reg, dict) else {}
        reg[obsm_key] = {
            'columns': classes,
            'source': 'pyscn',
            'layer': result.get('layer', 'X'),
        }
        self.adata.uns['xcell_score_matrices'] = reg

        runs = self.adata.uns.get('pyscn')
        runs = dict(runs) if isinstance(runs, dict) else {}
        runs[key] = {
            'classifier_path': result.get('classifier_path'),
            'classes': classes,
            'gene_overlap': result['gene_overlap'],
            'thresholds': thresholds,
            'quantile': result.get('quantile', 0.05),
            'layer': result.get('layer', 'X'),
            'obs_columns': [argmax_col, score_col] + ([type_col] if class_types else []),
            'obsm_key': obsm_key,
        }
        self.adata.uns['pyscn'] = runs

        # Composition, sorted by prevalence — the first thing anyone wants to
        # see after a classification run.
        counts = pd.Series(labels).value_counts()
        conf = pd.Series(summary['confidence'], dtype=float)
        composition = [
            {
                'name': str(name),
                'n_cells': int(n),
                'frac': float(n / len(labels)) if labels else 0.0,
                'mean_score': float(conf[pd.Series(labels) == name].mean()),
                'color': colors.get(str(name), '#808080'),
                'threshold': thresholds.get(str(name)),
            }
            for name, n in counts.items()
        ]

        type_counts = pd.Series(class_types).value_counts() if class_types else pd.Series(dtype=int)
        out = {
            'key': key,
            'n_cells': len(labels),
            'classes': classes,
            'composition': composition,
            'class_types': [
                {
                    'name': str(t),
                    'n_cells': int(type_counts.get(t, 0)),
                    'frac': float(type_counts.get(t, 0) / len(class_types)),
                    'color': pyscn.CLASS_TYPE_COLORS.get(str(t), '#808080'),
                }
                for t in pyscn.CLASS_TYPE_ORDER if type_counts.get(t, 0) > 0
            ],
            'gene_overlap': result['gene_overlap'],
            'obs_columns': [argmax_col, score_col] + ([type_col] if class_types else []),
            'argmax_column': argmax_col,
            'score_column': score_col,
            'type_column': type_col if class_types else None,
            'obsm_key': obsm_key,
            'thresholds': thresholds,
            'classifier': result.get('classifier'),
        }

        self._log_action('pyscn_classify', {
            'classifier_path': result.get('classifier_path'),
            'key': key,
            'layer': result.get('layer', 'X'),
        }, {
            'n_cells': out['n_cells'],
            'n_classes': len(classes),
            'gene_overlap': result['gene_overlap']['frac_found'],
        })
        return out

    def prepare_pyscn_train(
        self,
        groupby: str,
        out_path: str,
        *,
        n_cells_per_type: int | None = None,
        n_top_genes: int = 30,
        n_top_gene_pairs: int = 40,
        n_trees: int = 1000,
        n_rand: int | None = None,
        n_comps: int = 30,
        layer: str | None = None,
        source_scale: str | None = None,
    ) -> tuple[Any, Any]:
        """Train a classifier on this dataset's labels and write it to disk.

        Follows the PySCN quickstart: balance cells per type, normalize, pick
        highly variable genes, then ``tl.train_classifier``. Preprocessing runs
        on a copy, so training never mutates the loaded dataset — the training
        scale PySCN wants (log-normalized) is often not the scale the user is
        exploring in.

        Args:
            groupby: Categorical .obs column holding the cell type labels.
            out_path: Where to write the pickled classifier.
            n_cells_per_type: Cap per label, for class balance. None uses every
                cell, which biases the forest toward abundant types.
            n_top_genes, n_top_gene_pairs, n_trees, n_rand, n_comps: PySCN
                training parameters, same meanings as upstream.
            layer: Source matrix; None reads .X.
            source_scale: Override the detected scale of that matrix
                ('raw_counts', 'normalized_linear', 'log_normalized',
                'log_transformed'). None auto-detects. This decides what
                preprocessing is still owed — a reference distributed as
                log-normalized values must not be normalized and logged again.

        Raises:
            ValueError: Missing package, bad column, too few labels/cells, or
                an unwritable output path.
        """
        from xcell import pyscn

        pyscn.import_pyscn()

        if groupby not in self.adata.obs.columns:
            raise ValueError(f"obs column '{groupby}' not found")
        labels = self.adata.obs[groupby]
        n_labels = int(pd.Series(labels).nunique(dropna=True))
        if n_labels < 2:
            raise ValueError(
                f"'{groupby}' has {n_labels} distinct label(s); training needs "
                "at least 2 cell types."
            )
        if layer not in (None, '', 'X') and layer not in self.adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available: "
                f"['X'] + {sorted(self.adata.layers.keys())}"
            )

        # PySCN cannot encode a gene whose symbol contains '_'. Training drops
        # those; if that leaves nothing, say so now rather than after the user
        # waits through a forest fit.
        usable = self.n_genes - len(pyscn.underscore_gene_names(self.adata.var_names))
        if usable < pyscn.MIN_TRAINING_GENES:
            raise ValueError(
                f"Only {usable} of this dataset's {self.n_genes} gene symbols "
                "are free of underscores. PySingleCellNet encodes gene pairs "
                "as 'geneA_geneB' and splits on '_', so genes whose own names "
                "contain one cannot be used. Rename them (Genes panel → "
                "rename gene symbols) and train again."
            )

        out = Path(out_path).expanduser()
        if out.suffix not in ('.pkl', '.pickle'):
            out = out.with_suffix('.pkl')
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot write to {out.parent}: {e}")

        # Decide the preprocessing plan here, synchronously, so an untrainable
        # source scale (z-scored, binary) is a 400 rather than a failed task.
        from xcell import layer_scale as ls

        source = (
            self.adata.X if layer in (None, '', 'X') else self.adata.layers[layer]
        )
        detected = ls.assess_matrix_scale(source)['verdict']
        plan = pyscn.training_plan(detected, override=source_scale)

        # Copy now, on the request thread, so the background job can never see
        # a half-mutated AnnData.
        train_src = self.adata.copy()
        snap = {
            'groupby': groupby, 'out': out, 'n_cells_per_type': n_cells_per_type,
            'n_top_genes': int(n_top_genes), 'n_top_gene_pairs': int(n_top_gene_pairs),
            'n_trees': int(n_trees), 'n_rand': n_rand, 'n_comps': int(n_comps),
            'layer': layer, 'plan': plan,
        }

        def compute_fn(report=None):
            def progress(frac, message=None):
                if report is not None:
                    report(frac, message)

            return pyscn.train_and_save(train_src, snap, progress)

        def apply_fn(result):
            self._log_action('pyscn_train', {
                'groupby': snap['groupby'],
                'out_path': str(snap['out']),
                'n_trees': snap['n_trees'],
                'source_scale': plan['source_scale'],
            }, {
                'n_classes': len(result['classifier']['cell_type_classes']),
                'n_cells_used': result['n_cells_used'],
                'preprocessing': result['preprocessing']['reason'],
            })
            return result

        return compute_fn, apply_fn
