"""Generate a synthetic spatial dataset with known ligand-receptor signaling.

The tissue is a jittered 40x40 grid (1600 cells). Three small, dense ligand
sources are planted, each with its cognate receptor expressed broadly across the
tissue, so signaling fires in a ring around each source:

  * PDGFB   (diffusion) -> PDGFRA        source at (10, 10)
  * ADCYAP1 (diffusion) -> ADCYAP1R1     source at (30, 30)
  * DLL1    (contact)   -> NOTCH1        source at (10, 30)

Two negative controls are built in:
  * EFNA1 -> EPHA2: the ligand EFNA1 is expressed weakly everywhere (not
    localized), so no spatially-restricted signaling should be detected.
  * Receptors are broad, so a permuted (scattered) ligand cannot reconstruct the
    local source density -> the planted pairs are decisively significant.

Gene symbols are real (CellPhoneDB), so the shipped lr_pairs.csv database matches
them out of the box. Use it to exercise and demo Analyze -> Spatial -> Ligand-Receptor.

Run:  pixi run -e dev python -m xcell.data.generate_toy_ligrec
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix

SEED = 0
GRID = 40
OUT = Path(__file__).resolve().parent / "toy_spatial_ligrec.h5ad"

# (ligand, receptor, source_xy, mode)
SOURCES = [
    ("PDGFB", "PDGFRA", (10.0, 10.0), "diffusion"),
    ("ADCYAP1", "ADCYAP1R1", (30.0, 30.0), "diffusion"),
    ("DLL1", "NOTCH1", (10.0, 30.0), "contact"),
]
NEG_LIGAND, NEG_RECEPTOR = "EFNA1", "EPHA2"  # ligand not localized -> no signal
HOUSEKEEPING = ["GAPDH", "ACTB"]
NOISE = ["NOISE1", "NOISE2", "NOISE3"]


def build() -> anndata.AnnData:
    rng = np.random.default_rng(SEED)
    xs, ys = np.meshgrid(np.arange(GRID), np.arange(GRID))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    coords += rng.normal(0.0, 0.12, coords.shape)  # jitter -> robust Delaunay
    n = coords.shape[0]

    genes: list[str] = []
    cols: list[np.ndarray] = []

    def add(name: str, vec: np.ndarray) -> None:
        genes.append(name)
        cols.append(vec.astype(np.float32))

    cell_type = np.array(["stroma"] * n, dtype=object)
    blob_radius = 2.0

    for ligand, receptor, (cx, cy), _mode in SOURCES:
        d = np.linalg.norm(coords - np.array([cx, cy]), axis=1)
        src = d <= blob_radius
        # Ligand: dense in the small source, near-zero elsewhere (globally rare).
        lig = np.where(src, 25.0, 0.0) + rng.poisson(0.2, n)
        # Receptor: broad, moderate, so signaling fires where ligand reaches.
        rec = 12.0 + rng.poisson(2.0, n).astype(float)
        add(ligand, lig)
        add(receptor, rec)
        cell_type[src] = f"{ligand}_source"

    # Negative control: ligand expressed weakly everywhere (not localized).
    add(NEG_LIGAND, (3.0 + rng.poisson(2.0, n)).astype(float))
    add(NEG_RECEPTOR, (10.0 + rng.poisson(2.0, n)).astype(float))

    for hk in HOUSEKEEPING:
        add(hk, (20.0 + rng.poisson(5.0, n)).astype(float))
    for nz in NOISE:
        add(nz, rng.poisson(1.0, n).astype(float))

    X = np.column_stack(cols)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cell_type)},
        index=[f"cell_{i:04d}" for i in range(n)],
    )
    var = pd.DataFrame(index=genes)
    adata = anndata.AnnData(X=csr_matrix(X.astype(np.float32)), obs=obs, var=var)
    adata.obsm["spatial"] = coords.astype(np.float32)
    adata.obsm["X_spatial"] = coords.astype(np.float32)
    return adata


def main() -> None:
    adata = build()
    adata.write_h5ad(OUT)
    print(f"wrote {OUT}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"cell types: {dict(adata.obs['cell_type'].value_counts())}")


if __name__ == "__main__":
    main()
