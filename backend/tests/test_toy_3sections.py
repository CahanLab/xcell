"""Guard the bundled 3-section toy dataset's structure.

Regenerate with: cd backend && pixi run -e dev python -m xcell.data.generate_toy_3sections
"""

from pathlib import Path

import anndata
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "xcell" / "data" / "toy_spatial_3sections.h5ad"


@pytest.fixture(scope="module")
def adata():
    assert DATA.exists(), f"missing toy dataset: {DATA}"
    return anndata.read_h5ad(DATA)


def test_basic_shape(adata):
    assert adata.n_obs == 900
    assert adata.n_vars == 76


def test_obs_and_obsm(adata):
    assert list(adata.obs["section"].cat.categories) == ["section_1", "section_2", "section_3"]
    assert set(adata.obs["cell_type"].cat.categories) == {"Mesen", "Primor"}
    assert "spatial" in adata.obsm and "X_spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (900, 2)


def test_three_sections_each_300(adata):
    counts = adata.obs["section"].value_counts()
    assert all(counts[s] == 300 for s in ["section_1", "section_2", "section_3"])


def test_sections_are_spatially_separated(adata):
    # Each section's x-range must not overlap the next (gap between them).
    sp = np.asarray(adata.obsm["spatial"])
    sec = adata.obs["section"].values
    xmax_s1 = sp[sec == "section_1", 0].max()
    xmin_s2 = sp[sec == "section_2", 0].min()
    xmax_s2 = sp[sec == "section_2", 0].max()
    xmin_s3 = sp[sec == "section_3", 0].min()
    assert xmin_s2 > xmax_s1
    assert xmin_s3 > xmax_s2


def test_gene_category_present(adata):
    assert "gene_category" in adata.var
    cats = set(adata.var["gene_category"].astype(str))
    assert "primor_core" in cats and "mesen_anterior" in cats
