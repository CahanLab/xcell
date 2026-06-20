"""Regression test: the synthetic LR dataset's planted signals are detected."""

from pathlib import Path

import anndata
import pytest

from xcell.adaptor import DataAdaptor

TOY = Path(__file__).resolve().parents[1] / "xcell" / "data" / "toy_spatial_ligrec.h5ad"


@pytest.mark.skipif(not TOY.exists(), reason="toy_spatial_ligrec.h5ad not generated")
def test_planted_pairs_rank_top_and_negative_is_not_significant():
    adata = anndata.read_h5ad(TOY)
    a = DataAdaptor(str(TOY), adata=adata)
    res = a.prepare_ligrec(radius=4.0, n_perm=100, min_cells=5, seed=1)
    by = {s["interaction"]: s for s in res["summary"]}

    # All three planted pairs (two diffusion, one contact) fire.
    assert by["PDGFB->-PDGFRA"]["n_signif"] > 0
    assert by["ADCYAP1->-ADCYAP1R1"]["n_signif"] > 0
    assert by["DLL1->-NOTCH1"]["n_signif"] > 0

    # The non-localized ligand control fires on no cells, despite co-expression.
    assert by["EFNA1->-EPHA2"]["n_signif"] == 0

    # Planted pairs outrank the negative control.
    assert res["summary"][0]["interaction"] in {
        "PDGFB->-PDGFRA", "ADCYAP1->-ADCYAP1R1", "DLL1->-NOTCH1"
    }


@pytest.mark.skipif(not TOY.exists(), reason="toy_spatial_ligrec.h5ad not generated")
def test_finalize_writes_colorable_score_column():
    adata = anndata.read_h5ad(TOY)
    a = DataAdaptor(str(TOY), adata=adata)
    a.prepare_ligrec(radius=4.0, n_perm=80, min_cells=5, seed=1)
    fin = a.finalize_ligrec(interactions=["PDGFB->-PDGFRA"])
    key = fin["annotation_key"]
    assert key in a.adata.obs
    # numeric column suitable for continuous coloring
    assert a.adata.obs[key].dtype.kind == "f"
