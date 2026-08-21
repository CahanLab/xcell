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


# --- assignment -----------------------------------------------------------

def test_assignment_writes_a_prefixed_categorical_column():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    col = a.adata.obs["territory_prox-dist"]
    assert isinstance(col.dtype, pd.CategoricalDtype)
    assert set(col.dropna().unique()) <= {"proximal", "distal", "unassigned"}


def test_every_cell_above_the_cut_is_proximal():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    ys = a.adata.obsm["spatial"][:, 1]
    labels = a.adata.obs["territory_prox-dist"].astype(str).values
    assert set(labels[ys > 5.0]) == {"proximal"}
    assert set(labels[ys < 5.0]) == {"distal"}


def test_the_result_reports_counts_per_territory():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    result = a.assign_territories(["prox-dist"])
    counts = result["types"]["prox-dist"]["counts"]
    assert counts["proximal"] + counts["distal"] == a.n_cells


def test_a_section_is_only_assigned_by_its_own_faces():
    """Sections sit side by side in one coordinate space; a section-blind
    implementation looks perfectly correct here and is wrong."""
    a = _adaptor()
    payload = _payload(sections=("A",))          # geometry for A only
    a.save_territories("prox-dist", payload)
    a.assign_territories(["prox-dist"])
    labels = a.adata.obs["territory_prox-dist"].astype(str).values
    is_b = (a.adata.obs["section"].astype(str) == "B").values
    assert set(labels[is_b]) == {"unassigned"}
    assert "proximal" in set(labels[~is_b])


def test_the_same_name_in_two_sections_yields_one_label():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    labels = a.adata.obs["territory_prox-dist"].astype(str)
    by_section = labels.groupby(a.adata.obs["section"].astype(str)).unique()
    assert set(by_section["A"]) == set(by_section["B"])


def test_a_dataset_without_the_section_column_uses_the_single_ring():
    ad = _adata(sections=False)
    a = DataAdaptor("t.h5ad", adata=ad)
    a.save_territories("prox-dist", _payload(section_col=None, sections=("all",)))
    a.assign_territories(["prox-dist"])
    assert "unassigned" not in set(a.adata.obs["territory_prox-dist"].astype(str))


def test_combining_two_types_writes_a_third_column():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    a.assign_territories(["prox-dist", "ant-post"], combine=True)
    combined = a.adata.obs["territory_prox-dist__ant-post"].astype(str)
    assert combined.iloc[0] == "|".join([
        a.adata.obs["territory_prox-dist"].astype(str).iloc[0],
        a.adata.obs["territory_ant-post"].astype(str).iloc[0],
    ])


def test_combining_needs_at_least_two_types():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    with pytest.raises(ValueError, match="at least two"):
        a.assign_territories(["prox-dist"], combine=True)


def test_assigning_an_unknown_type_is_an_error():
    a = _adaptor()
    with pytest.raises(KeyError, match="nope"):
        a.assign_territories(["nope"])


def test_a_cell_with_no_coordinate_gets_na_not_unassigned():
    """NaN means "no position"; unassigned means "positioned outside every
    territory". Collapsing them would hide unplaced cells."""
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    coords = a.adata.obsm["spatial"].copy()
    coords[0] = [np.nan, np.nan]
    a.adata.obsm["spatial"] = coords
    a.assign_territories(["prox-dist"])
    assert pd.isna(a.adata.obs["territory_prox-dist"].iloc[0])


def test_assignment_is_recorded_in_the_analysis_record():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    assert any(e["action"] == "assign_territories" for e in a._action_history)
