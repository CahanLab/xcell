"""Tests for transfer_obs_labels: using subcluster labels to refine a parent category.

Scenario these tests model: a user masks some cells, subclusters the rest, which
creates a new .obs column where the masked-out cells are "unassigned". They then
want to fold those new subcluster labels back into a parent annotation, keeping
the parent label for every cell the subcluster did not touch.
"""

import numpy as np
import anndata
import pandas as pd
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata():
    """6 cells: parent cell_type A (0-3) / B (4-5). Only the A cells were
    subclustered into 0/1; B cells are 'unassigned' in the subcluster column."""
    X = csr_matrix(np.ones((6, 2), dtype=np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = ["g0", "g1"]
    ad.obs["cell_type"] = pd.Categorical(["A", "A", "A", "A", "B", "B"])
    ad.obs["sub"] = pd.Categorical(
        ["0", "0", "1", "1", "unassigned", "unassigned"],
        categories=["0", "1", "unassigned"],
    )
    return ad


def test_parent_prefix_refines_parent_keeps_untouched_labels():
    a = DataAdaptor("x.h5ad", adata=_adata())
    res = a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="cell_type_refined",
        rename_mode="parent_prefix",
        sep=".",
    )
    out = a.adata.obs["cell_type_refined"].astype(str).tolist()
    assert out == ["A.0", "A.0", "A.1", "A.1", "B", "B"]
    assert res["n_overridden"] == 4
    assert res["n_kept"] == 2
    # Result is categorical with exactly the labels that appear.
    assert set(a.adata.obs["cell_type_refined"].cat.categories) == {"A.0", "A.1", "B"}


def test_replace_mode_uses_raw_source_labels():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="replace",
    )
    out = a.adata.obs["refined"].astype(str).tolist()
    assert out == ["0", "0", "1", "1", "B", "B"]


def test_custom_prefix_mode():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="custom_prefix",
        prefix="sub_",
    )
    out = a.adata.obs["refined"].astype(str).tolist()
    assert out == ["sub_0", "sub_0", "sub_1", "sub_1", "B", "B"]


def test_overwrite_target_in_place():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="cell_type",  # same as target -> in place
        rename_mode="parent_prefix",
    )
    out = a.adata.obs["cell_type"].astype(str).tolist()
    assert out == ["A.0", "A.0", "A.1", "A.1", "B", "B"]


def test_nan_in_source_is_treated_as_unassigned():
    ad = _adata()
    # Replace the 'unassigned' values with real NaN to confirm NaN is "no new label".
    ad.obs["sub"] = pd.Categorical(["0", "0", "1", "1", None, None], categories=["0", "1"])
    a = DataAdaptor("x.h5ad", adata=ad)
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="replace",
    )
    out = a.adata.obs["refined"].astype(str).tolist()
    assert out == ["0", "0", "1", "1", "B", "B"]


def test_custom_unassigned_values():
    ad = _adata()
    ad.obs["sub"] = pd.Categorical(
        ["0", "0", "1", "1", "NA", "NA"], categories=["0", "1", "NA"]
    )
    a = DataAdaptor("x.h5ad", adata=ad)
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="replace",
        unassigned_values=["NA"],
    )
    out = a.adata.obs["refined"].astype(str).tolist()
    assert out == ["0", "0", "1", "1", "B", "B"]


def test_kept_parent_label_colors_are_preserved():
    ad = _adata()
    ad.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]  # A -> red, B -> green
    a = DataAdaptor("x.h5ad", adata=ad)
    a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="parent_prefix",
    )
    cats = list(a.adata.obs["refined"].cat.categories)
    colors = a.adata.uns["refined_colors"]
    color_by_cat = dict(zip(cats, colors))
    # B was untouched -> keeps green.
    assert color_by_cat["B"] == "#00ff00"


def test_missing_target_raises():
    a = DataAdaptor("x.h5ad", adata=_adata())
    try:
        a.transfer_obs_labels(
            target_column="nope", source_column="sub", out_column="refined"
        )
        assert False, "expected error"
    except (ValueError, KeyError) as e:
        assert "nope" in str(e)


def test_out_column_collision_with_unrelated_column_raises():
    ad = _adata()
    ad.obs["other"] = pd.Categorical(["x"] * 6)
    a = DataAdaptor("x.h5ad", adata=ad)
    try:
        a.transfer_obs_labels(
            target_column="cell_type",
            source_column="sub",
            out_column="other",  # exists, not the target -> refuse to clobber
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "other" in str(e)


def test_nothing_overridden_when_source_all_unassigned():
    ad = _adata()
    ad.obs["sub"] = pd.Categorical(["unassigned"] * 6, categories=["unassigned"])
    a = DataAdaptor("x.h5ad", adata=ad)
    res = a.transfer_obs_labels(
        target_column="cell_type",
        source_column="sub",
        out_column="refined",
        rename_mode="replace",
    )
    out = a.adata.obs["refined"].astype(str).tolist()
    assert out == ["A", "A", "A", "A", "B", "B"]
    assert res["n_overridden"] == 0
