"""Ligand-receptor spatial signaling scores (CytoSignal-style, no velocity).

A from-scratch NumPy/SciPy port of the core LRscore + permutation statistics of
the CytoSignal R package (welch-lab/CytoSignal). It detects, for each cell and
each ligand-receptor interaction, how strongly the cell is receiving signaling,
using spatial neighborhoods of expression. Two modes mirror CytoSignal:

  * diffusion-dependent (secreted ligands): ligand is imputed over a Gaussian
    epsilon-ball neighborhood (the ligand "diffuses" to nearby cells);
  * contact-dependent (membrane ligands): ligand is imputed over direct Delaunay
    neighbors only.

The per-cell score for an interaction is, exactly as in CytoSignal's production
path (inferScoreLR):

    Lbar = DTmean( normalize( impute(ligand, neighborGraph) ) )
    Rbar = DTmean( normalize( receptor [optionally DT-smoothed] ) )
    score = (sum over ligand subunits of Lbar) * (sum over receptor subunits Rbar)

Significance comes from a spatial permutation null (shuffle the cell axis, keep
the graph) and an empirical per-cell p-value, BH-corrected per interaction.

Velocity / dynamics (VeloCytoSignal) are intentionally omitted.

The bundled database (lr_pairs.csv) is human (UPPERCASE HGNC symbols). Mouse data
is matched case-insensitively (Pdgfb -> PDGFB) in prepare_ligrec, mirroring
CytoSignal's uppercase-match. This is a heuristic, not true orthology.

TODO (future work): add a proper mouse->human ortholog table (e.g. from MGI
HOM_MouseHumanSequence / Ensembl BioMart) as an option, so mouse genes whose
symbols genuinely differ from their human ortholog (or 1:many cases) map
correctly instead of relying on symbol case alone. See docs/ROADMAP.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree, Delaunay

# Gaussian tail factor: sigma = radius / sqrt(-2 ln(thresh)), thresh=0.001.
# matches CytoSignal inferEpsParams (sqrt(-2 ln 0.001) ~= 3.7169).
_SIGMA_FACTOR = float(np.sqrt(-2.0 * np.log(0.001)))

LR_DB_PATH = Path(__file__).resolve().parent / "data" / "lr_pairs.csv"


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
def load_lr_database(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the shipped compact ligand-receptor table.

    Returns a list of dicts: {interaction, ligand:[genes], receptor:[genes],
    type:'diffusion'|'contact', classification}.
    """
    p = Path(path) if path is not None else LR_DB_PATH
    out: list[dict[str, Any]] = []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "interaction": row["interaction"],
                "ligand": [g for g in row["ligand"].split("_") if g],
                "receptor": [g for g in row["receptor"].split("_") if g],
                "type": row.get("type", "diffusion"),
                "classification": row.get("classification", ""),
            })
    return out


# ---------------------------------------------------------------------------
# spatial neighbor graphs
# ---------------------------------------------------------------------------
def median_nn_distance(coords: np.ndarray) -> float:
    """Median nearest-neighbor distance — the natural spatial length scale."""
    coords = np.asarray(coords, float)
    if coords.shape[0] < 2:
        return 1.0
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return float(np.median(d[:, 1]))


def suggest_radius(coords: np.ndarray, rings: float = 3.0) -> float:
    """Data-driven diffusion radius: a few nearest-neighbor distances."""
    return float(rings * median_nn_distance(coords))


def _section_mask(sections: np.ndarray | None, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    if sections is None:
        return np.ones(len(i), dtype=bool)
    sections = np.asarray(sections)
    return sections[i] == sections[j]


def gaussian_graph(
    coords: np.ndarray,
    radius: float,
    sigma: float | None = None,
    self_weight: str | float = "auto",
    sections: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Row-normalized Gaussian epsilon-ball graph (diffusion mode).

    A[i, j] is the (normalized) Gaussian weight of neighbor j for index cell i,
    for all j within `radius` of i, plus a self weight. Rows sum to 1, so
    ``A @ X`` imputes expression as a weighted neighborhood average.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    if sigma is None:
        sigma = radius / _SIGMA_FACTOR
    peak = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs):
        keep = _section_mask(sections, pairs[:, 0], pairs[:, 1])
        pairs = pairs[keep]
    if len(pairs):
        d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2)) * peak
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.concatenate([w, w])
    else:
        rows = np.array([], int)
        cols = np.array([], int)
        data = np.array([], float)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    if self_weight == "auto":
        sw = peak * 5.0
    elif self_weight == "sum_1":
        sw = peak
    else:
        sw = float(self_weight) * peak
    A = A + sparse.diags(np.full(n, sw))

    rs = np.asarray(A.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return (sparse.diags(1.0 / rs) @ A).tocsr()


def delaunay_graph(
    coords: np.ndarray,
    max_radius: float | None = None,
    sections: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Delaunay contact graph.

    Returns (A_weighted, A_binary). A_weighted has neighbor weights normalized to
    sum 1 plus self=1 (CytoSignal "weight_sum_2"), suitable for ``A @ X``
    imputation. A_binary is the symmetric 0/1 adjacency *including self-loops*,
    used to build the DT-mean averaging graph. Collinear/degenerate inputs yield
    self-only graphs.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    edges: set[tuple[int, int]] = set()
    if n >= 3:
        try:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                m = len(simplex)
                for a in range(m):
                    for b in range(a + 1, m):
                        i, j = int(simplex[a]), int(simplex[b])
                        edges.add((i, j) if i < j else (j, i))
        except Exception:
            edges = set()

    ei, ej = [], []
    for (i, j) in edges:
        if max_radius is not None and np.linalg.norm(coords[i] - coords[j]) > max_radius:
            continue
        ei.append(i)
        ej.append(j)
    ei = np.asarray(ei, int)
    ej = np.asarray(ej, int)
    if len(ei):
        keep = _section_mask(sections, ei, ej)
        ei, ej = ei[keep], ej[keep]

    if len(ei):
        rows = np.concatenate([ei, ej])
        cols = np.concatenate([ej, ei])
        off = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        off = sparse.csr_matrix((n, n))

    eye = sparse.identity(n, format="csr")
    A_bin = (off + eye).tocsr()
    A_bin.data[:] = 1.0

    deg = np.asarray(off.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    A_w = (sparse.diags(inv) @ off + eye).tocsr()
    return A_w, A_bin


def to_mean_graph(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-normalize a 0/1 adjacency so each row averages over its neighbors."""
    A = adjacency.tocsr().astype(float)
    deg = np.asarray((A > 0).sum(axis=1)).ravel().astype(float)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (sparse.diags(inv) @ A).tocsr()


# ---------------------------------------------------------------------------
# normalization + score
# ---------------------------------------------------------------------------
def normalize_log1p(M: np.ndarray, lib: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """Library-size normalize then log1p (Seurat-style), zero-lib cells -> 0."""
    M = np.asarray(M, float)
    lib = np.asarray(lib, float)
    out = np.zeros_like(M)
    nz = lib > 0
    if nz.any():
        out[nz] = np.log1p(M[nz] / lib[nz][:, None] * scale)
    return out


def score_block(
    X: np.ndarray,
    lib: np.ndarray,
    A_lig: sparse.csr_matrix,
    G_dt: sparse.csr_matrix,
    lig_idx: list[np.ndarray],
    rec_idx: list[np.ndarray],
    recep_smooth: bool = False,
    X_rec: np.ndarray | None = None,
    lib_rec: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the per-cell LR score for a block of pairs sharing one ligand graph.

    X is (n_cells, n_genes) raw counts used for the *ligand* imputation; X_rec
    (defaults to X) is the matrix used for the *receptor* side. They differ only
    in the permutation null, where the ligand neighborhood is shuffled while each
    cell keeps its own receptor. lig_idx[p] / rec_idx[p] are the gene-column
    indices of pair p's ligand / receptor subunits. Returns (n_cells, n_pairs).
    """
    X = np.asarray(X, float)
    lib = np.asarray(lib, float)
    Xr = X if X_rec is None else np.asarray(X_rec, float)
    libr = lib if lib_rec is None else np.asarray(lib_rec, float)

    L_imp = np.asarray(A_lig @ X)
    lib_imp = np.asarray(A_lig @ lib)
    if recep_smooth:
        R = np.asarray(G_dt @ Xr)
        lib_R = np.asarray(G_dt @ libr)
    else:
        R = Xr
        lib_R = libr

    Lnorm = normalize_log1p(L_imp, lib_imp)
    Rnorm = normalize_log1p(R, lib_R)
    Lbar = np.asarray(G_dt @ Lnorm)
    Rbar = np.asarray(G_dt @ Rnorm)

    n = X.shape[0]
    out = np.empty((n, len(lig_idx)), dtype=float)
    for p in range(len(lig_idx)):
        lig_sum = Lbar[:, lig_idx[p]].sum(axis=1)
        rec_sum = Rbar[:, rec_idx[p]].sum(axis=1)
        out[:, p] = lig_sum * rec_sum
    return out


# ---------------------------------------------------------------------------
# significance + orchestration
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values for a 1D array."""
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def section_permutation(
    n_cells: int,
    sections: np.ndarray | None,
    rng: np.random.Generator,
    groups: list[np.ndarray] | None = None,
) -> np.ndarray:
    """A permutation of cell indices for the spatial null.

    With no sections, a plain global shuffle. With ``sections``, each cell maps
    to a random cell in the SAME section (a section-stratified null), so the
    permutation never mixes ligand expression across sections. ``groups`` may be
    precomputed (one index array per section) to avoid recomputing per call.
    """
    if sections is None:
        return rng.permutation(n_cells)
    if groups is None:
        sec = np.asarray(sections)
        groups = [np.where(sec == s)[0] for s in np.unique(sec)]
    perm = np.arange(n_cells)
    for idx in groups:
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def _resolve_pair_indices(
    pairs: list[dict[str, Any]], var_index: dict[str, int]
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    """Keep pairs whose every subunit gene is present; return column indices."""
    kept: list[dict[str, Any]] = []
    lig_idx: list[np.ndarray] = []
    rec_idx: list[np.ndarray] = []
    for pr in pairs:
        lig = [var_index[g] for g in pr["ligand"] if g in var_index]
        rec = [var_index[g] for g in pr["receptor"] if g in var_index]
        if len(lig) != len(pr["ligand"]) or len(rec) != len(pr["receptor"]):
            continue
        kept.append(pr)
        lig_idx.append(np.asarray(lig, int))
        rec_idx.append(np.asarray(rec, int))
    return kept, lig_idx, rec_idx


def compute_ligrec(
    X: Any,
    var_names: list[str],
    coords: np.ndarray,
    pairs: list[dict[str, Any]],
    *,
    radius: float,
    sigma: float | None = None,
    max_radius: float | None = None,
    recep_smooth: bool = False,
    smooth: bool = True,
    n_perm: int = 100,
    p_thresh: float = 0.05,
    sections: np.ndarray | None = None,
    seed: int = 0,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Score every pair, test significance against a spatial permutation null.

    Args:
        X: (n_cells, n_genes) raw counts (dense or sparse).
        var_names: gene names aligned to X columns.
        coords: (n_cells, 2) spatial coordinates.
        pairs: list of {interaction, ligand:[genes], receptor:[genes], type}.
        radius: Gaussian epsilon-ball radius for diffusion-mode ligand spread.
        sigma: Gaussian bandwidth (defaults to radius / 3.717).
        max_radius: cap on Delaunay edge length for contact mode (defaults to
            ``radius``).
        recep_smooth: DT-smooth the receptor side too.
        smooth: spatially smooth the score field over the Gaussian neighborhood
            (CytoSignal's smoothScoreLR) -- strengthens contiguous signaling and
            damps isolated spikes. The null is smoothed identically.
        n_perm: number of spatial permutations for the null.
        p_thresh: BH-adjusted p cutoff for calling a cell significant.
        sections: optional per-cell section labels; edges never cross sections.
        seed: RNG seed.

    Returns:
        dict with interactions (kept pairs), scores, pvalues, padj, significant
        (all n_cells x n_pairs), and a per-interaction summary sorted by number
        of significant cells.
    """
    coords = np.asarray(coords, float)
    n_cells = coords.shape[0]
    var_index = {g: i for i, g in enumerate(var_names)}

    kept, lig_idx, rec_idx = _resolve_pair_indices(pairs, var_index)
    if not kept:
        raise ValueError("No ligand-receptor pairs have all genes present in the data")

    # Full library size from ALL genes (counts).
    Xcsr = X.tocsr() if sparse.issparse(X) else np.asarray(X, float)
    lib = (np.asarray(Xcsr.sum(axis=1)).ravel() if sparse.issparse(Xcsr)
           else Xcsr.sum(axis=1)).astype(float)

    if max_radius is None:
        max_radius = radius

    types = {pr["type"] for pr in kept}
    # A Gaussian graph is always built: for diffusion-mode ligand imputation and
    # for the spatial score smoothing step (used by every mode).
    A_gauss = gaussian_graph(coords, radius, sigma, sections=sections)
    if "contact" in types:
        A_dt_w, A_dt_bin = delaunay_graph(coords, max_radius=max_radius, sections=sections)
    else:
        _, A_dt_bin = delaunay_graph(coords, max_radius=max_radius, sections=sections)
        A_dt_w = None
    G_dt = to_mean_graph(A_dt_bin)
    A_smooth = A_gauss

    P = len(kept)
    diff_cols = [k for k, pr in enumerate(kept) if pr["type"] == "diffusion"]
    cont_cols = [k for k, pr in enumerate(kept) if pr["type"] == "contact"]

    # Reduce the dense matrix to just the genes used by the kept pairs and remap
    # subunit indices into that reduced space. Every imputation matmul then runs
    # on (cells x used-genes) instead of (cells x all-genes) -- the dominant
    # speedup for gene panels / whole-transcriptome data. Numerically identical:
    # A @ X[:, used] == (A @ X)[:, used].
    used = sorted({int(g) for p in range(P) for g in
                   list(lig_idx[p]) + list(rec_idx[p])})
    col_of = {g: j for j, g in enumerate(used)}
    Xsub = (Xcsr[:, used].toarray() if sparse.issparse(Xcsr)
            else np.asarray(Xcsr, float)[:, used])
    lig_r = [np.array([col_of[int(g)] for g in lig_idx[p]], int) for p in range(P)]
    rec_r = [np.array([col_of[int(g)] for g in rec_idx[p]], int) for p in range(P)]

    # The receptor side is fixed across permutations (each cell keeps its own
    # receptor), so precompute the per-cell receptor subunit sums once.
    if recep_smooth:
        Rbar = np.asarray(G_dt @ normalize_log1p(np.asarray(G_dt @ Xsub),
                                                 np.asarray(G_dt @ lib)))
    else:
        Rbar = np.asarray(G_dt @ normalize_log1p(Xsub, lib))
    rec_sum = np.empty((n_cells, P), dtype=float)
    for p in range(P):
        rec_sum[:, p] = Rbar[:, rec_r[p]].sum(axis=1)

    def _ligand_sums(perm: np.ndarray | None) -> np.ndarray:
        Xl = Xsub if perm is None else Xsub[perm]
        libl = lib if perm is None else lib[perm]
        out = np.zeros((n_cells, P), dtype=float)
        if diff_cols:
            Lbar = np.asarray(G_dt @ normalize_log1p(
                np.asarray(A_gauss @ Xl), np.asarray(A_gauss @ libl)))
            for p in diff_cols:
                out[:, p] = Lbar[:, lig_r[p]].sum(axis=1)
        if cont_cols:
            Lbar = np.asarray(G_dt @ normalize_log1p(
                np.asarray(A_dt_w @ Xl), np.asarray(A_dt_w @ libl)))
            for p in cont_cols:
                out[:, p] = Lbar[:, lig_r[p]].sum(axis=1)
        return out

    def _score(perm: np.ndarray | None = None) -> np.ndarray:
        # Score = (imputed-ligand subunit sum) x (receptor subunit sum). The null
        # permutes the ligand neighborhood only; the receptor side is precomputed.
        return _ligand_sums(perm) * rec_sum

    def _smooth(M: np.ndarray) -> np.ndarray:
        return np.asarray(A_smooth @ M) if smooth else M

    def _report(frac: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, message)

    _report(0.0, f"Scoring {len(kept)} interactions")
    scores = _smooth(_score())

    # Spatial permutation null: shuffle which cells supply ligand to each
    # neighborhood (within section if sections given), keep receptors fixed.
    rng = np.random.default_rng(seed)
    if sections is not None:
        sec_arr = np.asarray(sections)
        section_groups = [np.where(sec_arr == s)[0] for s in np.unique(sec_arr)]
    else:
        section_groups = None
    n_perm = max(1, n_perm)
    null_pool = [np.empty(0) for _ in range(P)]
    null_cols = [[] for _ in range(P)]
    for t in range(n_perm):
        perm = section_permutation(n_cells, sections, rng, groups=section_groups)
        ns = _smooth(_score(perm))
        for p in range(P):
            null_cols[p].append(ns[:, p])
        # Report every few permutations so the UI shows live progress + speed.
        if t % 5 == 0 or t == n_perm - 1:
            _report((t + 1) / n_perm, f"Permutation {t + 1}/{n_perm}")
    for p in range(P):
        null_pool[p] = np.sort(np.concatenate(null_cols[p]))

    pvalues = np.ones((n_cells, P), dtype=float)
    padj = np.ones((n_cells, P), dtype=float)
    for p in range(P):
        npool = null_pool[p]
        m = npool.size
        ge = m - np.searchsorted(npool, scores[:, p], side="left")
        pvalues[:, p] = (ge + 1.0) / (m + 1.0)
        padj[:, p] = benjamini_hochberg(pvalues[:, p])

    significant = (padj < p_thresh) & (scores > 0)

    summary = []
    for p, pr in enumerate(kept):
        col = scores[:, p]
        n_sig = int(significant[:, p].sum())
        summary.append({
            "interaction": pr["interaction"],
            "type": pr["type"],
            "ligand": "_".join(pr["ligand"]),
            "receptor": "_".join(pr["receptor"]),
            "classification": pr.get("classification", ""),
            "n_signif": n_sig,
            "frac_signif": float(n_sig / n_cells) if n_cells else 0.0,
            "mean_score": float(col.mean()),
            "max_score": float(col.max()) if col.size else 0.0,
        })
    order = sorted(range(P), key=lambda k: (summary[k]["n_signif"], summary[k]["mean_score"]), reverse=True)
    summary = [summary[k] for k in order]

    return {
        "interactions": [pr["interaction"] for pr in kept],
        "pairs": kept,
        "scores": scores,
        "pvalues": pvalues,
        "padj": padj,
        "significant": significant,
        "summary": summary,
    }
