"""Turning a recorded step into prose and runnable Python.

Pure: a step in, a :class:`TranslatedStep` out. No AnnData, no adaptor, no I/O —
which is what lets every entry in the registry be tested against its expected
output directly.

The whole feature stands or falls on whether a reader can trust the emitted
code, so each step carries a **fidelity**:

``exact``
    The emitted line is the library call the adaptor really makes. Verified
    against ``adaptor.py``: ``run_normalize_total`` with no active selection
    literally executes ``sc.pp.normalize_total(self.adata, **kwargs)``.
``xcell``
    Reproducible only through xcell's own Python API. The notebook wraps the
    AnnData in a ``DataAdaptor`` and calls the method, with the parameters as
    recorded.
``manual``
    No code equivalent — described in prose and nothing more.

Orthogonally, a step that ran on an active cell selection is flagged in
``warnings``. We deliberately do not generate subset + write-back code: each
subset-capable method has its own write-back semantics (UMAP writes NaN for
inactive cells, Leiden writes ``'unassigned'``, ``normalize_total`` splices
sparse rows), and code that silently differs from what ran is the exact failure
this feature exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from xcell.analysis_record import Step

EXACT = 'exact'
XCELL = 'xcell'
MANUAL = 'manual'

SCANPY = ('import scanpy as sc',)
SQUIDPY = ('import squidpy as sq',)
NUMPY = ('import numpy as np',)
XCELL_API = ('from xcell.adaptor import DataAdaptor',)

# The variable names the exported notebook sets up. `xa` is a DataAdaptor
# wrapping `adata`; SELECTIONS is the sidecar of recorded cell selections.
ADATA = 'adata'
ADAPTOR = 'xa'


@dataclass(frozen=True)
class TranslatedStep:
    title: str
    summary: str
    code: list[str]
    fidelity: str
    imports: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Emission:
    """Code lines whose fidelity differs from the action's usual tier.

    Most actions sit in one tier, so their builders just return a list. A few
    depend on how they were called — PCA over a plain `.var` column is an exact
    scanpy call, but PCA over an ad-hoc gene list is only reproducible through
    xcell's own API — and those return this instead.
    """

    lines: list[str]
    fidelity: str


@dataclass(frozen=True)
class ActionSpec:
    label: str
    fidelity: str
    summary: Callable[[dict, dict], str]
    # Returns the code lines (or an Emission to override the tier), or None
    # when this particular step can't be reproduced — a missing selection, an
    # in-memory source. None downgrades the step to `manual`.
    code: Callable[[Step], list[str] | Emission | None] | None = None
    imports: tuple[str, ...] = ()


# --- literal rendering ----------------------------------------------------

def _lit(value: Any) -> str:
    """Render a recorded parameter as a Python literal.

    Everything here is pasted into a notebook the user will execute, so this
    goes through ``repr`` rather than string formatting — a gene name or file
    path containing a quote must not be able to escape its literal.
    """
    if isinstance(value, (str, bool)) or value is None:
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_lit(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{_lit(k)}: {_lit(v)}' for k, v in value.items()) + '}'
    return repr(str(value))


def _splat(
    params: dict,
    only: tuple[str, ...] | None = None,
    keep_none: tuple[str, ...] = (),
) -> str:
    """Render params as keyword arguments, dropping Nones.

    A recorded ``None`` usually means "the user did not ask for this", so
    omitting it leaves the library on its own default — the same thing the
    adaptor does when it builds a kwargs dict conditionally.

    ``keep_none`` is for the cases where that reasoning fails: where the adaptor
    passes ``None`` *explicitly* and the library's default is something else.
    ``sc.pp.calculate_qc_metrics(percent_top=None)`` means "no top-N columns",
    while scanpy's own default is ``(50, 100, 200, 500)`` — which raises
    IndexError on a dataset with fewer genes than that. Dropping that None
    produces a notebook that crashes where xcell did not.
    """
    items = params.items() if only is None else ((k, params[k]) for k in only if k in params)
    return ', '.join(
        f'{k}={_lit(v)}' for k, v in items if v is not None or k in keep_none
    )


def _call(
    fn: str,
    params: dict,
    only: tuple[str, ...] | None = None,
    target: str = ADATA,
    keep_none: tuple[str, ...] = (),
) -> str:
    args = _splat(params, only, keep_none)
    return f'{fn}({target}, {args})' if args else f'{fn}({target})'


def _xcall(method: str, params: dict, only: tuple[str, ...] | None = None) -> str:
    return f'{ADAPTOR}.{method}({_splat(params, only)})'


def _n(value: Any) -> str:
    """Format a recorded number for prose, tolerating a missing value.

    Counts arrive as ints, but parameters like ``target_sum`` are floats and
    read badly as "10,000.0", so whole floats lose their decimal.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return '?'
    if isinstance(value, float) and value.is_integer():
        return f'{int(value):,}'
    return f'{value:,}'


# --- per-action code builders ---------------------------------------------

_LOADERS = {
    'h5ad': 'sc.read_h5ad',
    '10x_h5': 'sc.read_10x_h5',
    '10x_mtx': 'sc.read_10x_mtx',
}


def _code_load(step: Step) -> list[str] | None:
    kind = step.params.get('kind', 'h5ad')
    path = step.params.get('path')
    reader = _LOADERS.get(kind)
    if reader is None or not path:
        # An in-memory AnnData (combine_spatial, tests) has no file to re-read.
        return None
    lines = [f'{ADATA} = {reader}({_lit(path)})']
    if kind != 'h5ad':
        lines.append(f'{ADATA}.var_names_make_unique()')
    return lines


def _code_pca(step: Step) -> list[str] | Emission:
    p = step.params
    args = _splat({'n_comps': p.get('n_comps'), 'svd_solver': p.get('svd_solver')})
    subset = p.get('gene_subset')
    if subset is None:
        return [f'sc.tl.pca({ADATA}, {args})']
    if isinstance(subset, str):
        # xcell copies out the gene subset, runs PCA there, and copies the
        # embedding back — varm['PCs'] is padded with NaN for excluded genes.
        return [
            f"_mask = {ADATA}.var[{_lit(subset)}].astype(bool).values",
            f'_sub = {ADATA}[:, _mask].copy()',
            f'sc.tl.pca(_sub, {args})',
            f"{ADATA}.obsm['X_pca'] = _sub.obsm['X_pca']",
            f"{ADATA}.uns['pca'] = _sub.uns['pca']",
        ]
    # A gene list or a multi-column mask expression. Resolving it the way xcell
    # does (intersection/union over .var columns) is not worth reimplementing —
    # hand it back to the adaptor, which is where that logic lives.
    return Emission(
        [_xcall('run_pca', p, ('n_comps', 'svd_solver', 'gene_subset'))], XCELL,
    )


def _code_exclude_genes(step: Step) -> list[str]:
    names = step.params.get('gene_names') or []
    patterns = step.params.get('patterns') or []
    lines = [f'_drop = {ADATA}.var_names.isin({_lit(list(names))})']
    for pattern in patterns:
        lines.append(f'_drop = _drop | {ADATA}.var_names.str.match({_lit(pattern)})')
    lines.append(f'{ADATA} = {ADATA}[:, ~_drop].copy()')
    return lines


def _code_delete_cells(step: Step) -> list[str] | None:
    if not step.selection:
        # Over SELECTION_CAP the indices were never kept; there is nothing
        # honest to emit.
        return None
    key = f'step_{step.index}'
    return [
        f'_drop = SELECTIONS[{_lit(key)}]',
        f'{ADATA} = {ADATA}[~np.isin(np.arange({ADATA}.n_obs), _drop)].copy()',
    ]


def _code_multicontour(step: Step) -> list[str]:
    p = step.params
    prep = dict(p.get('params') or {})
    prep['gene_sets'] = p.get('gene_sets')
    return [
        f'_mc = {_xcall("prepare_multicontour", prep)}',
        f"{ADAPTOR}.finalize_multicontour(_mc['token'], "
        f'{_splat(p, ("cutoffs", "profile_k", "out_name", "save_qc"))})',
    ]


def _code_pyscn_classify(step: Step) -> list[str] | None:
    path = step.params.get('classifier_path')
    if not path:
        return None
    args = _splat({'path': path, 'key': step.params.get('key'),
                   'layer': step.params.get('layer')})
    return [f'_xcell_run({ADAPTOR}.prepare_pyscn_classify({args}))']


def _two_phase(method: str, only: tuple[str, ...] | None = None):
    """A prepare_* method returning (compute_fn, apply_fn)."""
    def build(step: Step) -> list[str]:
        return [f'_xcell_run({ADAPTOR}.{method}({_splat(step.params, only)}))']
    return build


def _direct(method: str, only: tuple[str, ...] | None = None):
    def build(step: Step) -> list[str]:
        return [_xcall(method, step.params, only)]
    return build


def _scanpy(fn: str, only: tuple[str, ...] | None = None):
    def build(step: Step) -> list[str]:
        return [_call(fn, step.params, only)]
    return build


def _code_filter_cells(step: Step) -> list[str]:
    """One sc.pp.filter_cells call per threshold.

    scanpy accepts exactly one of min_counts/max_counts/min_genes/max_genes
    per call, so a single splatted call raises in the exported notebook when
    the UI sent several. Sequential calls compute the same AND the adaptor
    applies.
    """
    order = ('min_counts', 'max_counts', 'min_genes', 'max_genes')
    lines = [
        _call('sc.pp.filter_cells', {k: step.params[k]})
        for k in order
        if step.params.get(k) is not None
    ]
    return lines or [_call('sc.pp.filter_cells', step.params)]


# Must equal adaptor._GRAPH_META_KEY. Spelled twice rather than imported: this
# module promises no adaptor and no AnnData, and importing the constant would
# drag scanpy in behind it. test_codegen asserts the two stay equal.
_GRAPH_META_KEY = '_xcell_graph'


def _code_umap(step: Step) -> list[str]:
    """UMAP, plus the neighbors entry a non-default graph needs.

    ``sc.tl.umap`` has no ``obsp=``; it reads ``uns[neighbors_key]``. The
    adaptor builds that entry, hands it over, and deletes it — a notebook has
    to build it too, or ``neighbors_key`` names nothing. Emitting both is what
    keeps this ``exact``. The dict is the one that really ran, recorded at the
    time, rather than a re-derivation that could drift from it.
    """
    p = step.params
    call = _call('sc.tl.umap', p,
                 ('min_dist', 'spread', 'n_components', 'key_added'))
    meta = p.get('neighbors_meta')
    if not meta:
        return [call]
    return [
        f'{ADATA}.uns[{_lit(_GRAPH_META_KEY)}] = {_lit(meta)}',
        call[:-1] + f', neighbors_key={_lit(_GRAPH_META_KEY)})',
    ]


def _code_leiden(step: Step) -> list[str]:
    """Leiden takes the adjacency directly, so this stays a one-liner."""
    p = step.params
    call = _call('sc.tl.leiden', p, ('resolution', 'key_added'))
    graph_key = p.get('graph_key')
    if not graph_key:
        return [call]
    return [call[:-1] + f', obsp={_lit(graph_key)})']


# --- the registry ---------------------------------------------------------

REGISTRY: dict[str, ActionSpec] = {
    'load_dataset': ActionSpec(
        label='Load dataset',
        fidelity=EXACT,
        imports=SCANPY,
        code=_code_load,
        summary=lambda p, r: (
            f"Loaded `{p.get('path', 'the dataset')}` — "
            f"{_n(r.get('n_cells'))} cells x {_n(r.get('n_genes'))} genes."
        ),
    ),

    # --- preprocessing (exact scanpy) ---
    'filter_genes': ActionSpec(
        label='Filter genes', fidelity=EXACT, imports=SCANPY,
        code=_scanpy('sc.pp.filter_genes'),
        summary=lambda p, r: (
            f"Filtered genes ({_splat(p) or 'no thresholds'}): "
            f"{_n(r.get('n_genes_removed'))} removed, {_n(r.get('n_genes_after'))} remain."
        ),
    ),
    'filter_cells': ActionSpec(
        label='Filter cells', fidelity=EXACT, imports=SCANPY,
        code=_code_filter_cells,
        summary=lambda p, r: (
            f"Filtered cells ({_splat(p) or 'no thresholds'}): "
            f"{_n(r.get('n_cells_removed'))} removed, {_n(r.get('n_cells_after'))} remain."
        ),
    ),
    'exclude_genes': ActionSpec(
        label='Exclude genes', fidelity=EXACT, imports=(),
        code=_code_exclude_genes,
        summary=lambda p, r: (
            f"Excluded {_n(r.get('n_genes_removed'))} genes by name/pattern "
            f"({len(p.get('gene_names') or [])} names, {len(p.get('patterns') or [])} patterns)."
        ),
    ),
    'delete_cells': ActionSpec(
        label='Delete cells', fidelity=EXACT, imports=NUMPY,
        code=_code_delete_cells,
        summary=lambda p, r: (
            f"Removed {_n(r.get('n_cells_deleted'))} selected cells; "
            f"{_n(r.get('n_cells_after'))} remain."
        ),
    ),
    'normalize_total': ActionSpec(
        label='Normalize total counts', fidelity=EXACT, imports=SCANPY,
        code=_scanpy('sc.pp.normalize_total'),
        summary=lambda p, r: (
            f"Normalized each cell to {_n(p['target_sum'])} counts."
            if p.get('target_sum') is not None
            else 'Normalized each cell to the median total count.'
        ),
    ),
    'log1p': ActionSpec(
        label='Log-transform', fidelity=EXACT, imports=SCANPY,
        code=_scanpy('sc.pp.log1p'),
        summary=lambda p, r: 'Applied log1p to the expression matrix.',
    ),
    'highly_variable_genes': ActionSpec(
        label='Highly variable genes', fidelity=EXACT, imports=SCANPY,
        code=_scanpy('sc.pp.highly_variable_genes'),
        summary=lambda p, r: (
            f"Selected {_n(r.get('n_highly_variable'))} highly variable genes "
            f"of {_n(r.get('n_total_genes'))} (flavor {p.get('flavor', 'seurat')})."
        ),
    ),
    'pca': ActionSpec(
        label='PCA', fidelity=EXACT, imports=SCANPY,
        code=_code_pca,
        summary=lambda p, r: (
            f"PCA to {_n(r.get('n_comps', p.get('n_comps')))} components "
            f"over {_n(r.get('n_genes_used'))} genes "
            f"({r.get('gene_subset_type', 'default')} gene set)."
        ),
    ),
    'neighbors': ActionSpec(
        label='Neighbor graph', fidelity=EXACT, imports=SCANPY,
        code=_scanpy('sc.pp.neighbors'),
        summary=lambda p, r: (
            f"Built a k-nearest-neighbor graph with k={_n(p.get('n_neighbors'))} "
            f"({p.get('metric', 'euclidean')} distance)."
        ),
    ),
    'umap': ActionSpec(
        label='UMAP', fidelity=EXACT, imports=SCANPY,
        code=_code_umap,
        summary=lambda p, r: (
            f"UMAP embedding in {_n(p.get('n_components', 2))} dimensions "
            f"(min_dist {p.get('min_dist')}, spread {p.get('spread')})"
            + (f" over `{p['graph_key']}`" if p.get('graph_key') else '')
            + f" → `.obsm['{r.get('embedding_name', 'X_umap')}']`."
        ),
    ),
    'leiden': ActionSpec(
        label='Leiden clustering', fidelity=EXACT, imports=SCANPY,
        code=_code_leiden,
        summary=lambda p, r: (
            f"Leiden clustering at resolution {p.get('resolution')}"
            + (f" over `{p['graph_key']}`" if p.get('graph_key') else '')
            + f" → {_n(r.get('n_clusters'))} clusters in "
            f"`.obs['{p.get('key_added', 'leiden')}']`."
        ),
    ),

    # --- spatial (exact squidpy) ---
    'spatial_neighbors': ActionSpec(
        label='Spatial neighbor graph', fidelity=EXACT, imports=SQUIDPY,
        code=_scanpy('sq.gr.spatial_neighbors',
                     ('n_neighs', 'coord_type', 'spatial_key', 'delaunay', 'n_rings', 'radius')),
        summary=lambda p, r: (
            f"Spatial graph over `.obsm['{p.get('spatial_key', 'spatial')}']` "
            f"({p.get('coord_type', 'generic')}, {_n(p.get('n_neighs'))} neighbors"
            f"{', Delaunay' if p.get('delaunay') else ''}) → "
            f"{_n(r.get('n_edges'))} edges."
        ),
    ),
    'spatial_autocorr': ActionSpec(
        label='Spatial autocorrelation', fidelity=EXACT, imports=SQUIDPY,
        code=_scanpy('sq.gr.spatial_autocorr', ('mode', 'genes', 'n_perms', 'corr_method')),
        summary=lambda p, r: (
            f"{'Moran’s I' if p.get('mode') == 'moran' else 'Geary’s C'} "
            f"over {_n(r.get('n_genes_tested'))} genes "
            f"({_n(p.get('n_perms'))} permutations, {p.get('corr_method')} correction) → "
            f"{_n(r.get('n_significant'))} significant at p < {p.get('pval_threshold')}."
        ),
    ),

    # --- xcell's own analyses ---
    'create_pca_subset': ActionSpec(
        label='PC subset', fidelity=XCELL, imports=XCELL_API,
        code=_direct('create_pca_subset', ('drop_pc_indices', 'suffix')),
        summary=lambda p, r: (
            f"Derived a PC subset dropping PCs {p.get('drop_pc_indices')} → "
            f"`.obsm['{r.get('obsm_key', '?')}']`."
        ),
    ),
    'delete_pca_subset': ActionSpec(
        label='Delete PC subset', fidelity=XCELL, imports=XCELL_API,
        code=_direct('delete_pca_subset', ('obsm_key',)),
        summary=lambda p, r: f"Deleted the PC subset `{p.get('obsm_key')}`.",
    ),
    'combine_neighbors': ActionSpec(
        label='Combine neighbor graphs', fidelity=XCELL, imports=XCELL_API,
        code=_direct('combine_neighbor_graphs', ('sources', 'target_key')),
        summary=lambda p, r: (
            f"Combined {len(p.get('sources') or [])} neighbor graphs into "
            f"`.obsp['{p.get('target_key', 'connectivities')}']` "
            f"({_n(r.get('n_edges'))} edges)."
        ),
    ),
    'smooth': ActionSpec(
        label='Graph smoothing', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_smooth'),
        summary=lambda p, r: (
            f"Smoothed `{p.get('source_layer')}` over `{p.get('graph_key')}` for "
            f"{_n(p.get('n_steps'))} steps → `.layers['{p.get('output_layer')}']`."
        ),
    ),
    'gene_pca': ActionSpec(
        label='Gene PCA', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_gene_pca', ('n_comps', 'scale', 'use_kneedle', 'gene_subset')),
        summary=lambda p, r: (
            f"PCA in gene space to {_n(r.get('n_comps', p.get('n_comps')))} components."
        ),
    ),
    'gene_neighbors': ActionSpec(
        label='Gene neighbor graph', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_gene_neighbors', ('n_neighbors', 'metric', 'basis')),
        summary=lambda p, r: (
            f"Gene-gene kNN graph with k={_n(p.get('n_neighbors'))} "
            f"over `{p.get('basis')}` ({p.get('metric')})."
        ),
    ),
    'cluster_genes': ActionSpec(
        label='Gene clustering', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_cluster_genes', ('resolution', 'key_added')),
        summary=lambda p, r: (
            f"Leiden clustering of genes at resolution {p.get('resolution')} → "
            f"{_n(r.get('n_clusters'))} modules in `.var['{p.get('key_added')}']`."
        ),
    ),
    'contourize': ActionSpec(
        label='Expression contours', fidelity=XCELL, imports=XCELL_API,
        code=_two_phase('prepare_contourize'),
        summary=lambda p, r: (
            f"Contoured {len(p.get('genes') or [])} genes at "
            f"{_n(p.get('contour_levels'))} levels → `.obs['{p.get('annotation_key')}']`."
        ),
    ),
    'multicontour': ActionSpec(
        label='Multi-gene-set contours', fidelity=XCELL, imports=XCELL_API,
        code=_code_multicontour,
        summary=lambda p, r: (
            f"Contoured {len(p.get('gene_sets') or {})} gene sets into a tissue "
            f"annotation `.obs['{p.get('out_name')}']` "
            f"({_n(r.get('n_regions'))} regions)."
        ),
    ),
    'ligrec': ActionSpec(
        label='Ligand-receptor columns', fidelity=XCELL, imports=XCELL_API,
        code=_direct('finalize_ligrec', ('interactions', 'write_significance')),
        summary=lambda p, r: (
            f"Wrote {len(p.get('interactions') or [])} ligand-receptor interaction "
            f"scores to .obs ({len(r.get('written') or [])} columns)."
        ),
    ),
    'marker_genes': ActionSpec(
        label='Marker genes', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_marker_genes'),
        summary=lambda p, r: (
            f"Ranked marker genes for `{p.get('obs_column')}` "
            f"(top {_n(p.get('top_n'))} per group) → "
            f"{_n(r.get('total_genes'))} genes across {_n(r.get('n_groups'))} groups."
        ),
    ),
    'transfer_obs_labels': ActionSpec(
        label='Transfer labels', fidelity=XCELL, imports=XCELL_API,
        code=_direct('transfer_obs_labels',
                     ('target_column', 'source_column', 'out_column', 'rename_mode')),
        summary=lambda p, r: (
            f"Transferred labels from `{p.get('source_column')}` onto "
            f"`{p.get('target_column')}` → `.obs['{p.get('out_column')}']`."
        ),
    ),
    'pyscn_classify': ActionSpec(
        label='Cell-type classification (PySingleCellNet)', fidelity=XCELL, imports=XCELL_API,
        code=_code_pyscn_classify,
        summary=lambda p, r: (
            f"Classified {_n(r.get('n_cells'))} cells into {_n(r.get('n_classes'))} types "
            f"→ `.obs['{p.get('key', 'SCN')}_class']` "
            f"({(r.get('gene_overlap') or 0) * 100:.0f}% of classifier genes found)."
        ),
    ),
    'localize': ActionSpec(
        label='Localize (spatial projection)', fidelity=MANUAL,
        # Reproducing it needs the *other* dataset — the spatial reference —
        # which the record does not carry and a single-adata notebook cannot
        # reconstruct. Naming the parameters is honest; emitting a call that
        # silently used the wrong reference would not be.
        code=lambda s: None,
        summary=lambda p, r: (
            f"Predicted spatial coordinates for {_n(r.get('n_cells'))} cells "
            f"from a spatial reference of {_n(r.get('n_reference_cells'))} cells "
            f"(k={_n(p.get('k'))}, {p.get('metric')} on {p.get('transform')}-"
            f"transformed expression, {p.get('aggregation')} aggregation over "
            f"{_n(r.get('n_shared_genes'))} shared genes) → "
            f"`.obsm['{r.get('embedding_name')}']`. "
            f"Median confidence {r.get('median_confidence')}; "
            f"{_n(r.get('n_unplaced'))} cells left unplaced."
        ),
    ),
    'set_spatial_key': ActionSpec(
        label='Choose the spatial coordinates', fidelity=EXACT, imports=(),
        # A one-line assignment, but it belongs in the export: every spatial
        # result after it was computed over *this* array, and on a localized
        # dataset that array is a prediction rather than a measurement.
        code=lambda s: (
            [f"{ADATA}.uns['xcell_spatial_key'] = {_lit(s.params.get('key'))}"]
            if s.params.get('key')
            else [f"{ADATA}.uns.pop('xcell_spatial_key', None)"]
        ),
        summary=lambda p, r: (
            f"Used `.obsm['{p.get('key')}']` as the spatial coordinates for "
            f'everything below.'
            if p.get('key') else
            'Went back to auto-detecting the spatial coordinates.'
        ),
    ),
    'ligrec_analyze': ActionSpec(
        label='Ligand-receptor signaling', fidelity=XCELL, imports=XCELL_API,
        code=_direct('prepare_ligrec'),
        summary=lambda p, r: (
            f"Scored ligand-receptor interactions (radius {p.get('radius')}, "
            f"{_n(p.get('n_perm'))} permutations, p < {p.get('p_thresh')}): "
            f"{_n(r.get('n_tested'))} pairs tested, "
            f"{_n(r.get('n_significant'))} significant."
        ),
    ),
    'pyscn_train': ActionSpec(
        label='Train classifier (PySingleCellNet)', fidelity=XCELL, imports=XCELL_API,
        code=_two_phase('prepare_pyscn_train',
                        ('groupby', 'out_path', 'n_trees', 'source_scale')),
        summary=lambda p, r: (
            f"Trained a classifier on `{p.get('groupby')}` "
            f"({_n(r.get('n_classes'))} classes, {_n(r.get('n_cells_used'))} cells, "
            f"{_n(p.get('n_trees'))} trees) → `{p.get('out_path')}`."
        ),
    ),
    # --- annotation and metadata edits ---
    # These change what every downstream figure says, so a report that omits
    # them is misleading even when every computation in it is reproducible.
    'rename_obs_label': ActionSpec(
        label='Rename a label', fidelity=EXACT, imports=(),
        code=lambda s: [
            f"{ADATA}.obs[{_lit(s.params.get('column'))}] = "
            f"{ADATA}.obs[{_lit(s.params.get('column'))}].cat.rename_categories("
            f"{{{_lit(s.params.get('old_label'))}: {_lit(s.params.get('new_label'))}}})",
        ],
        summary=lambda p, r: (
            f"Renamed `{p.get('old_label')}` to `{p.get('new_label')}` in "
            f"`.obs['{p.get('column')}']` ({_n(r.get('n_cells_renamed'))} cells)."
        ),
    ),
    'merge_obs_labels': ActionSpec(
        label='Merge labels', fidelity=EXACT, imports=(),
        code=lambda s: [
            f"_col = {ADATA}.obs[{_lit(s.params.get('column'))}].astype(str)",
            f"_col[_col.isin({_lit(list(s.params.get('labels') or []))})] = "
            f"{_lit(s.params.get('new_label'))}",
            f"{ADATA}.obs[{_lit(s.params.get('column'))}] = _col.astype('category')",
        ],
        summary=lambda p, r: (
            f"Merged {p.get('labels')} into `{p.get('new_label')}` in "
            f"`.obs['{p.get('column')}']` ({_n(r.get('n_cells_merged'))} cells)."
        ),
    ),
    'create_annotation': ActionSpec(
        label='New annotation column', fidelity=EXACT, imports=('import pandas as pd',),
        code=lambda s: [
            f"{ADATA}.obs[{_lit(s.params.get('name'))}] = pd.Categorical("
            f"[{_lit(s.params.get('default_value'))}] * {ADATA}.n_obs)",
        ],
        summary=lambda p, r: (
            f"Created `.obs['{p.get('name')}']`, every cell "
            f"`{p.get('default_value')}`."
        ),
    ),
    'label_cells': ActionSpec(
        label='Label selected cells', fidelity=EXACT, imports=(),
        code=lambda s: (
            [
                f"_sel = SELECTIONS[{_lit(f'step_{s.index}')}]",
                f"{ADATA}.obs[{_lit(s.params.get('annotation'))}] = "
                f"{ADATA}.obs[{_lit(s.params.get('annotation'))}].cat.add_categories("
                f"[{_lit(s.params.get('label'))}])",
                f"{ADATA}.obs.iloc[_sel, {ADATA}.obs.columns.get_loc("
                f"{_lit(s.params.get('annotation'))})] = {_lit(s.params.get('label'))}",
            ]
            if s.selection else None
        ),
        summary=lambda p, r: (
            f"Labelled {_n(r.get('n_cells_labelled'))} hand-selected cells "
            f"`{p.get('label')}` in `.obs['{p.get('annotation')}']`."
        ),
    ),

    # --- derived columns and metadata ---
    'calculate_qc_metrics': ActionSpec(
        label='QC metrics', fidelity=EXACT, imports=SCANPY,
        code=lambda s: [_call('sc.pp.calculate_qc_metrics', {
            'qc_vars': s.params.get('qc_vars'),
            'percent_top': s.params.get('percent_top'),
            'log1p': s.params.get('log1p'),
            'inplace': True,
        }, keep_none=('percent_top',))],
        summary=lambda p, r: (
            f"Computed QC metrics over {p.get('qc_vars') or 'no'} gene flags → "
            f"{_n(r.get('n_obs_columns'))} new `.obs` and "
            f"{_n(r.get('n_var_columns'))} new `.var` columns."
        ),
    ),
    'add_var_boolean_column': ActionSpec(
        label='Flag genes', fidelity=XCELL, imports=XCELL_API,
        code=_direct('add_var_boolean_column', ('name', 'pattern', 'match_mode')),
        summary=lambda p, r: (
            f"Flagged {_n(r.get('n_genes_matched'))} genes matching "
            f"{p.get('match_mode')} `{p.get('pattern')}` in `.var['{p.get('name')}']`."
        ),
    ),
    'sum_counts_by_pattern': ActionSpec(
        label='Sum counts by pattern', fidelity=XCELL, imports=XCELL_API,
        code=_direct('sum_counts_by_pattern', ('pattern', 'match_mode', 'obs_name', 'layer')),
        summary=lambda p, r: (
            f"Summed counts of {_n(r.get('n_genes_matched'))} genes matching "
            f"`{p.get('pattern')}` into `.obs['{r.get('obs_name')}']`."
        ),
    ),
    'sum_counts_by_species': ActionSpec(
        label='Sum counts by species', fidelity=XCELL, imports=XCELL_API,
        code=_direct('sum_counts_by_species'),
        summary=lambda p, r: (
            f"Summed per-species counts from `.var['{p.get('species_column')}']` → "
            f"{r.get('obs_columns')}."
        ),
    ),
    'assign_species': ActionSpec(
        label='Assign species', fidelity=XCELL, imports=XCELL_API,
        code=_direct('assign_species', ('count_columns', 'labels', 'obs_name', 'threshold')),
        summary=lambda p, r: (
            f"Assigned each cell a species at {p.get('threshold')} purity → "
            f"`.obs['{p.get('obs_name')}']` ({r.get('counts')})."
        ),
    ),
    'rename_genes': ActionSpec(
        label='Rename genes', fidelity=XCELL, imports=XCELL_API,
        code=_direct('rename_genes', ('pattern', 'replacement', 'match_mode', 'make_unique')),
        summary=lambda p, r: (
            f"Renamed {_n(r.get('n_renamed'))} of {_n(r.get('n_genes'))} genes "
            f"({p.get('match_mode')} `{p.get('pattern')}` → `{p.get('replacement')}`)."
        ),
    ),
    'map_gene_symbols': ActionSpec(
        label='Add gene symbols', fidelity=XCELL, imports=XCELL_API,
        code=_direct('map_gene_symbols', ('column', 'set_as_index')),
        summary=lambda p, r: (
            f"Mapped {_n(r.get('n_mapped'))} of {_n(r.get('n_genes'))} "
            f"{r.get('species')} Ensembl ids to symbols → "
            f"`.var['{p.get('column')}']`"
            + (' and promoted them to the gene index.'
               if p.get('set_as_index') else '.')
        ),
    ),
    'swap_var_index': ActionSpec(
        label='Swap gene identifiers', fidelity=XCELL, imports=XCELL_API,
        code=_direct('swap_var_index', ('column_name',)),
        summary=lambda p, r: (
            f"Promoted `.var['{p.get('column_name')}']` to the gene index "
            f"({_n(r.get('n_genes'))} genes)."
        ),
    ),
    'create_obs_embedding': ActionSpec(
        label='Embedding from .obs columns', fidelity=XCELL, imports=XCELL_API,
        code=_direct('create_obs_embedding', ('col_x', 'col_y', 'log_axes', 'name')),
        summary=lambda p, r: (
            f"Built a 2-D embedding from `{p.get('col_x')}` and `{p.get('col_y')}` → "
            f"`.obsm['{r.get('embedding_name')}']`."
        ),
    ),

    # --- scoring and testing ---
    'score_gene_sets_matrix': ActionSpec(
        label='Score gene sets', fidelity=XCELL, imports=XCELL_API,
        code=_direct('score_gene_sets_matrix',
                     ('sets', 'per_gene_norm', 'per_gene_clip', 'aggregation',
                      'obsm_name', 'layer', 'transform')),
        summary=lambda p, r: (
            f"Scored {_n(r.get('n_sets'))} gene sets "
            f"({p.get('per_gene_norm')} per-gene, {p.get('aggregation')} across genes) "
            f"→ `.obsm['{r.get('obsm_name')}']`."
        ),
    ),
    'score_gene_sets_ucell': ActionSpec(
        label='UCell gene-set scores', fidelity=XCELL, imports=XCELL_API,
        code=_direct('score_gene_sets_ucell', ('sets', 'layer', 'max_rank', 'w_neg')),
        summary=lambda p, r: (
            f"UCell-scored {len(p.get('sets') or [])} gene sets "
            f"(max rank {_n(r.get('max_rank'))}) → {r.get('obs_columns')}."
        ),
    ),
    'diffexp': ActionSpec(
        label='Differential expression', fidelity=XCELL, imports=XCELL_API,
        # The two groups were picked in the UI, so the call needs their indices.
        code=lambda s: None,
        summary=lambda p, r: (
            f"{p.get('method')} test between two selections of "
            f"{_n(p.get('n_group1'))} and {_n(p.get('n_group2'))} cells over "
            f"{_n(r.get('n_genes_tested'))} genes → "
            f"{_n(r.get('n_upregulated'))} up, {_n(r.get('n_downregulated'))} down "
            f"({p.get('corr_method')} correction)."
        ),
    ),
}


def registered_actions() -> set[str]:
    return set(REGISTRY)


# --- translation ----------------------------------------------------------

def _unknown_summary(action: str, params: dict) -> str:
    if not params:
        return f'`{action}` (no parameters recorded).'
    rendered = ', '.join(f'{k}={_lit(v)}' for k, v in params.items())
    return f'`{action}` with {rendered}.'


def _warnings_for(step: Step, spec: ActionSpec | None) -> list[str]:
    out: list[str] = []
    if step.n_active is not None:
        out.append(
            f'xcell ran this on an active selection of {step.n_active:,} '
            f'of {step.n_total:,} cells. The code below runs on the whole dataset.'
            if step.n_total
            else f'xcell ran this on an active selection of {step.n_active:,} cells. '
                 f'The code below runs on the whole dataset.'
        )
    if step.action == 'spatial_neighbors' and step.params.get('section_col'):
        out.append(
            f"xcell additionally removed edges crossing sections defined by "
            f"`.obs['{step.params['section_col']}']`; the squidpy call alone "
            f'does not do that.'
        )
    return out


def translate(step: Step) -> TranslatedStep:
    """Render one recorded step as a title, prose, and (where honest) code."""
    spec = REGISTRY.get(step.action)
    warnings = _warnings_for(step, spec)

    if spec is None:
        return TranslatedStep(
            title=f'Unrecognized operation: {step.action}',
            summary=_unknown_summary(step.action, step.params),
            code=[],
            fidelity=MANUAL,
            imports=(),
            warnings=warnings,
        )

    try:
        summary = spec.summary(step.params, step.result)
    except Exception:  # a record restored from an older h5ad may lack keys
        summary = _unknown_summary(step.action, step.params)

    emitted = spec.code(step) if spec.code else None
    if isinstance(emitted, Emission):
        lines, fidelity = emitted.lines, emitted.fidelity
    else:
        lines, fidelity = emitted, spec.fidelity
    if not lines:
        return TranslatedStep(
            title=spec.label, summary=summary, code=[],
            fidelity=MANUAL, imports=(), warnings=warnings,
        )

    return TranslatedStep(
        title=spec.label,
        summary=summary,
        code=lines,
        fidelity=fidelity,
        # An xcell-tier step needs the adaptor regardless of what the spec
        # normally imports (see PCA over a gene list).
        imports=spec.imports if fidelity != XCELL else tuple({*spec.imports, *XCELL_API}),
        warnings=warnings,
    )
