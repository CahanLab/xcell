"""Generate ``toy_spatial_3sections.h5ad`` — a 3-section spatial toy dataset.

Mirrors the single-section ``toy_spatial.h5ad`` (same 76-gene panel and
``gene_category`` structure, same Mesen/Primor cell types and spatial subregion
markers) but contains **three distinct sections sampled from the same tissue** —
analogous to three cuts of a mouse E11.5 forelimb. The sections are laid out
left-to-right with a gap, so the Euclidean distance between spots on *different*
sections is large and not biologically meaningful.

Use it to exercise (multi)contour and other spatial analyses (spatial neighbors,
autocorrelation) on data where cross-section distances must be ignored: a global
grid interpolation or kNN that ignores the ``section`` label will incorrectly
bridge the gaps between sections.

Deterministic (fixed seed). Regenerate with::

    cd backend && pixi run -e dev python -m xcell.data.generate_toy_3sections

Reuses the gene schema from the bundled ``toy_spatial.h5ad`` so the exact same
gene sets/categories apply to both datasets.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent
SEED = 7
N_SECTIONS = 3
CELLS_PER_SECTION = 300
SECTION_WIDTH = 250.0   # proximal-distal extent (x) of one section
SECTION_HEIGHT = 360.0  # anterior-posterior extent (y)
SECTION_GAP = 90.0      # empty x-gap between consecutive sections


def _load_gene_schema():
    """Return (gene_names, gene_category Series, dict category -> [genes])."""
    toy = anndata.read_h5ad(DATA_DIR / "toy_spatial.h5ad")
    cats = toy.var["gene_category"].astype(str)
    by_cat: dict[str, list[str]] = {}
    for g, c in zip(toy.var_names, cats):
        by_cat.setdefault(c, []).append(g)
    return list(toy.var_names), cats.values, by_cat


def _section_cells(rng: np.random.Generator):
    """Sample one section: local coords in [0,W]x[0,H], cell types, and the
    normalized axes used to drive subregion markers.

    Geometry: an elliptical limb-bud blob. Proximal-distal axis = x (0 proximal,
    1 distal); the primordium (Primor) occupies the distal cap, mesenchyme the
    rest. Anterior-posterior axis = y.
    """
    cx, cy = SECTION_WIDTH / 2, SECTION_HEIGHT / 2
    rx, ry = SECTION_WIDTH / 2, SECTION_HEIGHT / 2

    xs, ys = [], []
    while len(xs) < CELLS_PER_SECTION:
        # rejection-sample inside the ellipse
        x = rng.uniform(0, SECTION_WIDTH)
        y = rng.uniform(0, SECTION_HEIGHT)
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
            xs.append(x)
            ys.append(y)
    x = np.array(xs)
    y = np.array(ys)

    u = x / SECTION_WIDTH          # proximal(0) -> distal(1)
    v = y / SECTION_HEIGHT         # anterior(0) -> posterior(1)
    # radial distance from section center, normalized to ~[0,1]
    r = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)

    # Primordium = distal cap (u high); rest = mesenchyme.
    is_primor = u > 0.6
    cell_type = np.where(is_primor, "Primor", "Mesen")

    return {
        "x": x, "y": y, "u": u, "v": v, "r": r,
        "cell_type": cell_type, "is_primor": is_primor,
    }


def _gene_lambda(gene_category: str, sec: dict) -> np.ndarray:
    """Poisson rate per cell for a gene of the given category within a section."""
    n = sec["x"].shape[0]
    u, v, r = sec["u"], sec["v"], sec["r"]
    is_primor = sec["is_primor"]
    is_mesen = ~is_primor
    base = np.full(n, 0.4)

    def bump(mask_strength):
        # map a 0..1 spatial weight to a count rate (peak ~9)
        return 0.4 + 9.0 * np.clip(mask_strength, 0, 1)

    if gene_category == "ubiquitous":
        return np.full(n, 5.0)
    if gene_category == "sparse":
        return np.full(n, 0.3)
    if gene_category == "mesen_specific":
        return np.where(is_mesen, 8.0, 0.5)
    if gene_category == "primor_specific":
        return np.where(is_primor, 8.0, 0.5)

    # --- spatial subregion markers (gated to their tissue) ---
    if gene_category == "mesen_anterior":
        return np.where(is_mesen, bump(1 - v), base)
    if gene_category == "mesen_posterior":
        return np.where(is_mesen, bump(v), base)
    if gene_category == "mesen_proximal":
        return np.where(is_mesen, bump(1 - u), base)
    if gene_category == "mesen_distal":
        return np.where(is_mesen, bump(u / 0.6), base)
    if gene_category == "primor_core":
        return np.where(is_primor, bump(1 - r), base)
    if gene_category == "primor_perimeter":
        return np.where(is_primor, bump(r), base)
    if gene_category == "primor_proximal":
        return np.where(is_primor, bump(1 - (u - 0.6) / 0.4), base)
    if gene_category == "primor_distal":
        return np.where(is_primor, bump((u - 0.6) / 0.4), base)

    return base


def build() -> anndata.AnnData:
    rng = np.random.default_rng(SEED)
    gene_names, gene_cats, _by_cat = _load_gene_schema()
    n_genes = len(gene_names)

    all_x, all_y, all_ct, all_sec = [], [], [], []
    sec_states = []
    for s in range(N_SECTIONS):
        sec = _section_cells(rng)
        x_off = s * (SECTION_WIDTH + SECTION_GAP)
        all_x.append(sec["x"] + x_off)
        all_y.append(sec["y"])
        all_ct.append(sec["cell_type"])
        all_sec.append(np.full(sec["x"].shape[0], f"section_{s + 1}"))
        sec_states.append(sec)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    cell_type = np.concatenate(all_ct)
    section = np.concatenate(all_sec)
    n_cells = x.shape[0]

    # Expression: per gene, concatenate section-wise Poisson draws.
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for gi, gcat in enumerate(gene_cats):
        col = []
        for sec in sec_states:
            lam = _gene_lambda(str(gcat), sec)
            col.append(rng.poisson(lam).astype(np.float32))
        X[:, gi] = np.concatenate(col)

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(cell_type, categories=["Mesen", "Primor"]),
            "section": pd.Categorical(section, categories=[f"section_{s + 1}" for s in range(N_SECTIONS)]),
        },
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame({"gene_category": pd.Categorical(gene_cats)}, index=gene_names)

    adata = anndata.AnnData(X=csr_matrix(X), obs=obs, var=var)
    coords = np.column_stack([x, y]).astype(np.float32)
    adata.obsm["spatial"] = coords
    adata.obsm["X_spatial"] = coords.copy()
    adata.uns["xcell"] = {
        "toy": "3-section limb-bud; sections laid out left-to-right with a gap; "
               "cross-section distances are not meaningful (see obs['section'])."
    }
    return adata


def main():
    adata = build()
    out = DATA_DIR / "toy_spatial_3sections.h5ad"
    adata.write_h5ad(out)
    print(f"Wrote {out}  shape={adata.shape}  "
          f"sections={list(adata.obs['section'].cat.categories)}")


if __name__ == "__main__":
    main()
