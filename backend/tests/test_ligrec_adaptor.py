"""Tests for the ligand-receptor adaptor methods (prepare/finalize)."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _spatial_adata():
    """30x30 grid: ligand L from a small dense source, receptor Rnear in a shell
    around it, Rfar far away, N noise. Mirrors the validated signaling field."""
    w = 30
    xs, ys = np.meshgrid(np.arange(w), np.arange(w))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    c = np.array([15.0, 15.0])
    d = np.linalg.norm(coords - c, axis=1)
    L = np.where(d <= 2.0, 20.0, 0.0)
    Rnear = np.where((d > 2.0) & (d <= 3.7), 20.0, 0.0)
    Rfar = np.where(d >= 9.0, 20.0, 0.0)
    rng = np.random.default_rng(0)
    N = rng.poisson(1.0, coords.shape[0]).astype(float)
    X = np.column_stack([L, Rnear, Rfar, N]).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["L", "Rnear", "Rfar", "N"]
    ad.obsm["spatial"] = coords
    return ad


_PAIRS = [
    {"interaction": "L->Rnear", "ligand": ["L"], "receptor": ["Rnear"], "type": "diffusion"},
    {"interaction": "L->Rfar", "ligand": ["L"], "receptor": ["Rfar"], "type": "diffusion"},
    {"interaction": "N->Rnear", "ligand": ["N"], "receptor": ["Rnear"], "type": "diffusion"},
]


def test_prepare_requires_spatial():
    ad = _spatial_adata()
    del ad.obsm["spatial"]
    a = DataAdaptor("x.h5ad", adata=ad)
    try:
        a.prepare_ligrec(pairs=_PAIRS, n_perm=20)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "spatial" in str(e).lower()


def test_prepare_ranks_true_pair_first():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    res = a.prepare_ligrec(pairs=_PAIRS, radius=4.0, n_perm=100, min_cells=2, seed=1)
    summary = res["summary"]
    assert summary[0]["interaction"] == "L->Rnear"
    assert summary[0]["n_signif"] > 0
    by = {s["interaction"]: s for s in summary}
    assert by["L->Rnear"]["n_signif"] > by["L->Rfar"]["n_signif"]
    assert by["L->Rnear"]["n_signif"] > by["N->Rnear"]["n_signif"]


def test_prepare_suggests_radius_when_omitted():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    res = a.prepare_ligrec(pairs=_PAIRS, n_perm=20, min_cells=2, seed=1)
    assert res["params"]["radius"] > 0


def test_finalize_writes_score_column():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    a.prepare_ligrec(pairs=_PAIRS, radius=4.0, n_perm=100, min_cells=2, seed=1)
    fin = a.finalize_ligrec(interactions=["L->Rnear"], write_significance=True)
    key = fin["annotation_key"]
    assert key in a.adata.obs
    vals = np.asarray(a.adata.obs[key].values, dtype=float)
    # The score is highest near the source (center), low far away.
    coords = a.adata.obsm["spatial"]
    d = np.linalg.norm(coords - np.array([15.0, 15.0]), axis=1)
    assert vals[d <= 4].mean() > vals[d > 8].mean()
    # significance column written too
    assert any(c.endswith("_sig") for c in fin["written"])


def test_prepare_persists_results_to_obsm_and_uns():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    assert a.get_ligrec_result() is None  # nothing before running
    a.prepare_ligrec(pairs=_PAIRS, radius=4.0, n_perm=60, min_cells=2, seed=1)
    assert a.adata.obsm["lrscore"].shape == (a.adata.n_obs, len(_PAIRS))
    assert a.adata.obsm["lrscore_significant"].shape == (a.adata.n_obs, len(_PAIRS))
    stored = a.get_ligrec_result()
    assert stored is not None
    assert len(stored["summary"]) == len(_PAIRS)
    assert set(stored["interactions"]) == {p["interaction"] for p in _PAIRS}


def test_finalize_can_be_called_repeatedly_for_reselection():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    a.prepare_ligrec(pairs=_PAIRS, radius=4.0, n_perm=60, min_cells=2, seed=1)
    a.finalize_ligrec(interactions=["L->Rnear"])
    # A second call with a different interaction still works (no token needed).
    fin2 = a.finalize_ligrec(interactions=["L->Rfar"])
    assert fin2["annotation_key"] in a.adata.obs


def test_finalize_without_prior_run_raises():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    try:
        a.finalize_ligrec(interactions=["L->Rnear"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "result" in str(e).lower()


def test_progress_callback_is_invoked():
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    seen = []
    a.prepare_ligrec(pairs=_PAIRS, radius=4.0, n_perm=40, min_cells=2, seed=1,
                     progress_callback=lambda frac, msg=None: seen.append((frac, msg)))
    assert seen  # at least one progress update
    assert seen[-1][0] == 1.0  # reaches 100%


def test_case_insensitive_gene_matching_for_mouse():
    # Mouse data uses Title-case symbols; the human DB uses UPPERCASE. The pair
    # should still match, and the written column should use the dataset's actual
    # (mouse) gene names, not the database's.
    ad = _spatial_adata()
    ad.var_names = ["Pdgfb", "Pdgfra", "Rfar", "N"]  # Pdgfb=source, Pdgfra=shell
    a = DataAdaptor("x.h5ad", adata=ad)
    pairs = [{"interaction": "PDGFB->-PDGFRA", "ligand": ["PDGFB"], "receptor": ["PDGFRA"], "type": "diffusion"}]
    res = a.prepare_ligrec(pairs=pairs, radius=4.0, n_perm=80, min_cells=2, seed=1)
    assert res["n_tested"] == 1
    assert res["summary"][0]["n_signif"] > 0
    fin = a.finalize_ligrec(interactions=["PDGFB->-PDGFRA"])
    assert "Pdgfb" in fin["annotation_key"]  # dataset gene name, not DB's


def test_min_cells_filters_unexpressed_pairs():
    # A pair whose gene is expressed in too few cells is dropped.
    a = DataAdaptor("x.h5ad", adata=_spatial_adata())
    pairs = _PAIRS + [
        {"interaction": "GHOST->Rnear", "ligand": ["GHOST"], "receptor": ["Rnear"], "type": "diffusion"},
    ]
    res = a.prepare_ligrec(pairs=pairs, radius=4.0, n_perm=20, min_cells=2, seed=1)
    interactions = [s["interaction"] for s in res["summary"]]
    assert "GHOST->Rnear" not in interactions  # GHOST gene absent -> dropped
