"""Territory persistence and cell assignment on the adaptor.

Territories are written into the *live* adata, not only at export time — the
whole point is that a dataset carries its own regions.
"""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n=60, n_genes=8, seed=0, sections=True):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(1.0, (n, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    # A grid so membership is easy to reason about: x in [0,10), y in [0,10)
    xs = rng.uniform(0.5, 9.5, n)
    ys = rng.uniform(0.5, 9.5, n)
    ad.obsm["spatial"] = np.column_stack([xs, ys])
    if sections:
        ad.obs["section"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B"))
    return ad


def _adaptor(**kw):
    return DataAdaptor("t.h5ad", adata=_adata(**kw))


def _payload(section_col="section", sections=("A", "B")):
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    per = {
        s: {
            "ring": ring,
            "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
            "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0},
                        {"name": "distal", "x": 5.0, "y": 2.0}],
        }
        for s in sections
    }
    return {"embedding": "spatial", "section_col": section_col,
            "sections": per, "source": "drawn"}


# --- persistence ----------------------------------------------------------

def test_saving_writes_into_the_live_adata_not_only_on_export():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    assert DataAdaptor.TERRITORY_UNS_KEY in a.adata.uns
    stored = json.loads(a.adata.uns[DataAdaptor.TERRITORY_UNS_KEY])
    assert stored["prox-dist"]["sections"]["A"]["anchors"][0]["name"] == "proximal"


def test_saving_stamps_a_creation_date():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    assert a.get_territories()["prox-dist"]["created"]


def test_a_second_type_does_not_clobber_the_first():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    assert sorted(a.get_territories()) == ["ant-post", "prox-dist"]


def test_saving_the_same_name_replaces_it():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    p = _payload()
    p["sections"]["A"]["anchors"][0]["name"] = "renamed"
    a.save_territories("prox-dist", p)
    got = a.get_territories()["prox-dist"]["sections"]["A"]["anchors"][0]["name"]
    assert got == "renamed"


def test_get_returns_empty_when_nothing_was_ever_saved():
    assert _adaptor().get_territories() == {}


def test_delete_removes_one_type_and_leaves_the_rest():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    a.delete_territories("prox-dist")
    assert list(a.get_territories()) == ["ant-post"]


def test_deleting_an_unknown_type_is_an_error_not_a_silent_no_op():
    a = _adaptor()
    with pytest.raises(KeyError, match="nope"):
        a.delete_territories("nope")


def test_a_type_naming_an_embedding_the_dataset_lacks_is_refused():
    a = _adaptor()
    p = _payload()
    p["embedding"] = "X_not_here"
    with pytest.raises(ValueError, match="X_not_here"):
        a.save_territories("prox-dist", p)


def test_a_type_with_no_sections_is_refused():
    a = _adaptor()
    p = _payload()
    p["sections"] = {}
    with pytest.raises(ValueError, match="no sections"):
        a.save_territories("prox-dist", p)


def test_territories_survive_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    path = tmp_path / "round.h5ad"
    a.adata.write_h5ad(path)

    reloaded = DataAdaptor(str(path), adata=anndata.read_h5ad(path))
    assert reloaded.get_territories()["prox-dist"]["sections"]["A"]["cuts"][0]["points"] \
        == [[1.0, 5.0], [9.0, 5.0]]
