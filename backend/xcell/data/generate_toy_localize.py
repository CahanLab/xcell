"""Generate the paired localization benchmark: a spatial reference and a
matched "dissociated" scRNA-seq query with exact ground truth.

    toy_localize_spatial.h5ad   the reference — expression + coordinates
    toy_localize_scrna.h5ad     the query — expression only, truth hidden in
                                .obsm['spatial_true'] and .obs['true_x'/'true_y']

Use it for Analyze -> Spatial -> Localize: load the spatial file into one slot,
the scRNA-seq file into the other, run the projection, then score it against the
stored truth.

The tissue is deliberately built to *break* kNN projection rather than flatter
it. Four region-restricted programs are the easy case; the other three
populations each attack a different assumption:

``Immune``      uniformly dispersed, one flat program. Its transcriptional
                neighbours are scattered over the whole tissue, so the centroid
                of them is meaningless — this is what the confidence score has
                to catch.
``Bipolar``     one program, two distant patches. A weighted mean lands in the
                empty gap between them; ``densest`` aggregation should pick a
                patch.
``Circulating`` present only in the query, absent from the reference. It will be
                placed *somewhere* regardless — the similarity score is what
                says not to believe it.

The two files also differ the way two platforms do: the query is sequenced ~4x
deeper with a different dropout rate and per-gene capture efficiencies, so the
choice of transform (per-dataset z-score / rank) actually matters.

Deterministic. Regenerate with::

    cd backend && pixi run -e dev python -m xcell.data.generate_toy_localize
"""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

OUT_DIR = Path(__file__).resolve().parent
SEED = 11

WIDTH, HEIGHT = 200.0, 160.0
N_SPATIAL = 1200
N_QUERY = 900

# Region-restricted programs: (name, centre, radius, n_marker_genes)
REGIONS = [
    ('Apical', (100.0, 140.0), 45.0),
    ('Basal', (100.0, 25.0), 45.0),
    ('Anterior', (35.0, 80.0), 42.0),
    ('Posterior', (165.0, 80.0), 42.0),
]
# The two Bipolar patches — far apart, same program.
BIPOLAR_PATCHES = [(45.0, 30.0), (160.0, 135.0)]
BIPOLAR_RADIUS = 18.0

N_MARKERS = 12      # marker genes per program
N_GRADIENT = 24     # genes forming smooth position gradients
N_NOISE = 12        # genes carrying no positional information


def _programs() -> tuple[list[str], dict[str, list[int]], int]:
    """Build the gene panel and the index blocks each program drives."""
    names: list[str] = []
    blocks: dict[str, list[int]] = {}
    for label, _, _ in REGIONS:
        blocks[label] = list(range(len(names), len(names) + N_MARKERS))
        names += [f'{label}_m{i}' for i in range(N_MARKERS)]
    for label in ('Immune', 'Bipolar', 'Circulating'):
        blocks[label] = list(range(len(names), len(names) + N_MARKERS))
        names += [f'{label}_m{i}' for i in range(N_MARKERS)]
    blocks['_gradient'] = list(range(len(names), len(names) + N_GRADIENT))
    names += [f'grad_{i}' for i in range(N_GRADIENT)]
    blocks['_noise'] = list(range(len(names), len(names) + N_NOISE))
    names += [f'noise_{i}' for i in range(N_NOISE)]
    return names, blocks, len(names)


def _sample_positions(rng, n):
    """Positions with population labels, inside an elliptical tissue."""
    cx, cy, rx, ry = WIDTH / 2, HEIGHT / 2, WIDTH / 2, HEIGHT / 2
    coords, labels = [], []

    def _inside(x, y):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    def _draw_in(centre, radius):
        while True:
            x, y = rng.normal(centre, radius / 2.2, 2)
            if _inside(x, y):
                return x, y

    # Proportions: the four regions carry the tissue, Immune is scattered
    # through it, Bipolar sits in its two patches.
    n_region = int(n * 0.62)
    n_immune = int(n * 0.22)
    n_bipolar = n - n_region - n_immune

    for i in range(n_region):
        label, centre, radius = REGIONS[i % len(REGIONS)]
        coords.append(_draw_in(centre, radius))
        labels.append(label)
    for _ in range(n_immune):
        while True:
            x, y = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
            if _inside(x, y):
                break
        coords.append((x, y))
        labels.append('Immune')
    for i in range(n_bipolar):
        coords.append(_draw_in(BIPOLAR_PATCHES[i % 2], BIPOLAR_RADIUS))
        labels.append('Bipolar')

    return np.asarray(coords, dtype=float), np.asarray(labels, dtype=object)


def _expression(rng, coords, labels, blocks, n_genes):
    """Latent expression: program markers plus smooth spatial gradients."""
    n = len(coords)
    expr = np.full((n, n_genes), 0.25, dtype=float)

    for label in set(labels.tolist()):
        idx = np.flatnonzero(labels == label)
        cols = blocks[label]
        expr[np.ix_(idx, cols)] += rng.gamma(6.0, 1.2, (len(idx), len(cols)))

    # Gradients: each gene a random plane over the tissue, so position is
    # recoverable even for a cell whose program is shared with others.
    u = (coords[:, 0] - WIDTH / 2) / (WIDTH / 2)
    v = (coords[:, 1] - HEIGHT / 2) / (HEIGHT / 2)
    planes = rng.normal(0, 1, (3, N_GRADIENT))
    grad = np.column_stack([u, v, np.ones(n)]) @ planes
    expr[:, blocks['_gradient']] += np.maximum(grad + 2.5, 0.0)

    expr[:, blocks['_noise']] += rng.gamma(2.0, 0.6, (n, N_NOISE))

    # Immune carries NO positional information — that is the whole point of it.
    immune = np.flatnonzero(labels == 'Immune')
    if len(immune):
        expr[np.ix_(immune, blocks['_gradient'])] = 0.25
    # Bipolar is identical in both patches, so its neighbourhood is bimodal.
    bipolar = np.flatnonzero(labels == 'Bipolar')
    if len(bipolar):
        expr[np.ix_(bipolar, blocks['_gradient'])] = 0.25
    return expr


def _sequence(rng, latent, depth, dropout, capture):
    """Turn latent expression into counts for one platform.

    Different depth, dropout and per-gene capture efficiency is what makes the
    two files behave like two technologies rather than two copies.
    """
    rate = latent * capture[None, :] * depth
    counts = rng.poisson(rate).astype(np.float32)
    counts *= (rng.random(counts.shape) > dropout)
    return counts


def _build(rng, n_cells, blocks, n_genes, *, depth, dropout, capture, extra_pop=None):
    coords, labels = _sample_positions(rng, n_cells)
    if extra_pop is not None:
        # A population that exists only in the query. Its "position" is real —
        # it came from somewhere — but nothing in the reference resembles it.
        n_extra = int(n_cells * 0.08)
        keep = rng.permutation(len(coords))[: len(coords) - n_extra]
        coords, labels = coords[keep], labels[keep]
        e_coords, _ = _sample_positions(rng, n_extra)
        coords = np.vstack([coords, e_coords])
        labels = np.concatenate([labels, np.array([extra_pop] * n_extra, dtype=object)])
    latent = _expression(rng, coords, labels, blocks, n_genes)
    counts = _sequence(rng, latent, depth, dropout, capture)
    return coords, labels, counts


def main() -> None:
    rng = np.random.default_rng(SEED)
    gene_names, blocks, n_genes = _programs()

    # Per-gene capture efficiency differs between platforms — the reason a
    # per-dataset transform is the default rather than an option.
    capture_spatial = rng.uniform(0.5, 1.5, n_genes)
    capture_query = rng.uniform(0.5, 1.5, n_genes)

    ref_coords, ref_labels, ref_counts = _build(
        rng, N_SPATIAL, blocks, n_genes,
        depth=1.0, dropout=0.35, capture=capture_spatial,
    )
    q_coords, q_labels, q_counts = _build(
        rng, N_QUERY, blocks, n_genes,
        depth=4.0, dropout=0.12, capture=capture_query,
        extra_pop='Circulating',
    )

    spatial = anndata.AnnData(X=csr_matrix(ref_counts))
    spatial.var_names = gene_names
    spatial.obs_names = [f'spot_{i}' for i in range(len(ref_coords))]
    spatial.obs['cell_type'] = pd.Categorical(ref_labels.astype(str))
    spatial.obsm['spatial'] = ref_coords
    spatial.obsm['X_spatial'] = ref_coords.copy()
    spatial.uns['xcell_toy'] = 'localize reference'

    query = anndata.AnnData(X=csr_matrix(q_counts))
    query.var_names = gene_names
    query.obs_names = [f'cell_{i}' for i in range(len(q_coords))]
    query.obs['cell_type'] = pd.Categorical(q_labels.astype(str))
    # Ground truth, kept out of .obsm['spatial'] so nothing treats this as a
    # spatial dataset by accident — it is the answer key, not an input.
    query.obsm['spatial_true'] = q_coords
    query.obs['true_x'] = q_coords[:, 0]
    query.obs['true_y'] = q_coords[:, 1]
    query.uns['xcell_toy'] = 'localize query (truth in obsm["spatial_true"])'

    spatial.write_h5ad(OUT_DIR / 'toy_localize_spatial.h5ad')
    query.write_h5ad(OUT_DIR / 'toy_localize_scrna.h5ad')

    print(f'spatial: {spatial.shape}  {dict(spatial.obs["cell_type"].value_counts())}')
    print(f'query:   {query.shape}  {dict(query.obs["cell_type"].value_counts())}')
    print(f'wrote {OUT_DIR / "toy_localize_spatial.h5ad"}')
    print(f'wrote {OUT_DIR / "toy_localize_scrna.h5ad"}')


if __name__ == '__main__':
    main()
