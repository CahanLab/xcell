"""Assembling the merged AnnData: what carries over and what does not.

The merged rows never existed, so anything derived from the originals is either
stale or an approximation the summed counts cannot justify. These tests pin down
what survives — counts, coordinates, categoricals, provenance — and, just as
importantly, what is deliberately absent.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.spot_merge import MergeParams


def _adata(n_side: int = 6, pitch: float = 15.0) -> anndata.AnnData:
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    n = len(coords)
    counts = (rng.poisson(5.0, size=(n, 8)) + 1).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.obs_names = [f"s{i}" for i in range(n)]
    ad.layers["counts"] = csr_matrix(counts)
    ad.obs["cell_type"] = pd.Categorical(["hyper"] * n)
    ad.obs["total_counts"] = counts.sum(axis=1)          # QC — must be dropped
    ad.obs["ucell_score"] = rng.random(n)                # numeric — dropped
    ad.obsm["spatial"] = coords
    ad.obsm["X_umap"] = rng.normal(size=(n, 2))          # must be dropped
    ad.uns["xcell_spatial_scale"] = {"um_per_unit": 1.0}
    return ad


def _adaptor() -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=_adata())


def _region(a: DataAdaptor) -> list[int]:
    """The lower-left 3x3 block of the 6x6 grid."""
    coords = a.adata.obsm["spatial"]
    return np.where((coords[:, 0] < 40) & (coords[:, 1] < 40))[0].tolist()


def test_dry_run_returns_stats_and_builds_nothing():
    a = _adaptor()
    stats, merged = a.merge_spots(_region(a), MergeParams(), dry_run=True)
    assert merged is None
    assert stats["n_region_spots"] == 9
    assert stats["n_merged_spots"] < 9
    assert stats["pitch_um"] == pytest.approx(15.0)
    assert stats["max_spots"] >= 1


def test_the_source_dataset_is_untouched():
    a = _adaptor()
    before = a.adata.n_obs
    a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert a.adata.n_obs == before
    assert "merge_group" not in a.adata.obs


def test_counts_are_conserved():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    # Merging moves counts between rows; it never creates or loses them.
    source_total = float(np.asarray(a.adata.layers["counts"].sum()))
    merged_total = float(np.asarray(merged.layers["counts"].sum()))
    assert merged_total == pytest.approx(source_total)


def test_merged_coordinates_are_the_member_centroid():
    a = _adaptor()
    region = _region(a)
    stats, merged = a.merge_spots(region, MergeParams(), dry_run=False)
    labels = np.asarray(stats["_labels"], dtype=int)
    groups = merged.obs["merge_group"].astype(str).to_numpy()
    src = a.adata.obsm["spatial"]
    for gid in range(labels.max() + 1):
        members = [region[i] for i in range(len(region)) if labels[i] == gid]
        row = int(np.where(groups == f"g{gid}")[0][0])
        np.testing.assert_allclose(
            merged.obsm["spatial"][row], src[members].mean(axis=0)
        )


def test_untouched_spots_keep_their_counts():
    a = _adaptor()
    region = set(_region(a))
    _, merged = a.merge_spots(sorted(region), MergeParams(), dry_run=False)
    outside = [i for i in range(a.adata.n_obs) if i not in region]
    singles = merged.obs["merged_n_spots"].to_numpy()
    assert (singles == 1).sum() >= len(outside)
    # The first untouched row is row 0 and must match its source exactly.
    np.testing.assert_allclose(
        np.asarray(merged.layers["counts"][0].todense()).ravel(),
        np.asarray(a.adata.layers["counts"][outside[0]].todense()).ravel(),
    )


def test_numeric_and_qc_obs_are_dropped_everywhere():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "ucell_score" not in merged.obs
    assert "total_counts" not in merged.obs


def test_categoricals_survive_by_majority_vote():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "cell_type" in merged.obs
    assert set(merged.obs["cell_type"].dropna().unique()) == {"hyper"}


def test_only_the_spatial_obsm_survives():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert set(merged.obsm.keys()) == {"spatial"}


def test_provenance_records_how_it_was_made():
    a = _adaptor()
    _, merged = a.merge_spots(
        _region(a), MergeParams(max_diameter_um=40.0), dry_run=False
    )
    prov = merged.uns["xcell_spot_merge"]
    assert prov["params"]["max_diameter_um"] == 40.0
    assert prov["n_spots_before"] == 36
    assert prov["n_spots_after"] == merged.n_obs


def test_purity_appears_when_there_is_something_to_vote_on():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "merge_purity" in merged.obs
    assert merged.obs["merge_purity"].max() <= 1.0


def test_a_region_of_one_spot_is_refused():
    a = _adaptor()
    with pytest.raises(ValueError, match="at least two"):
        a.merge_spots([0], MergeParams(), dry_run=True)


def test_missing_spatial_scale_is_refused():
    ad = _adata()
    del ad.uns["xcell_spatial_scale"]
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="coordinate unit"):
        a.merge_spots(list(range(9)), MergeParams(), dry_run=True)


def test_non_integer_matrix_is_refused():
    ad = _adata()
    del ad.layers["counts"]
    ad.X = csr_matrix(np.asarray(ad.X.todense()) * 0.37)
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="counts"):
        a.merge_spots(list(range(9)), MergeParams(), dry_run=True, layer=None)


def test_no_spatial_coordinates_is_refused():
    ad = _adata()
    del ad.obsm["spatial"]
    del ad.obsm["X_umap"]
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="spatial coordinates"):
        a.merge_spots(list(range(9)), MergeParams(), dry_run=True)
