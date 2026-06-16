"""Tests for the Visium HD ``feature_slice.h5`` loader (xcell.visium_hd).

A synthetic fixture writes a minimal feature_slice.h5 so the rebinned counts can
be hand-computed and asserted exactly. No real 700MB file needed.

Fixture layout (2 um grid is 8x8, bin_size 8 um -> factor 4 -> 2x2 bin grid):

  Tissue mask (8 um bins, data==1): (0,0), (0,1), (1,1)   [bin (1,0) is NOT tissue]

  Gene 0 "GeneA":  (r0,c0)=5, (r1,c1)=3  -> bin(0,0)=8 ;  (r4,c5)=2 -> bin(1,1)=2
  Gene 1 "GeneB":  (r0,c5)=4 -> bin(0,1)=4 ;  (r5,c4)=1 -> bin(1,1)=1
  Gene 2 "GeneC":  (r2,c2)=7 -> bin(0,0)=7 ;  (r6,c6)=9 -> bin(1,1)=9 ;
                   (r4,c0)=100 -> bin(1,0) which is NOT tissue -> DROPPED

  obs order (flat = br*n_bin_cols + bc, n_bin_cols=2): (0,0)<(0,1)<(1,1)

  Expected X (obs x gene):
            GeneA GeneB GeneC
    (0,0):    8     0     7
    (0,1):    0     4     0
    (1,1):    2     1     9

  Clustering graphclust labels: (0,0)->1, (0,1)->2, (1,1)->1
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import xcell.visium_hd as vh


def _bytes_arr(strings):
    return np.array([s.encode() for s in strings], dtype="S256")


def write_feature_slice(path: Path) -> None:
    """Write a minimal valid feature_slice.h5 to ``path``."""
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = b"feature_slice"
        f.attrs["software_version"] = b"spaceranger-test"
        f.attrs["metadata_json"] = json.dumps(
            {"sample_id": "test_sample", "nrows": 8, "ncols": 8, "spot_pitch": 2.0}
        )

        feats = f.create_group("features")
        feats.create_dataset("id", data=_bytes_arr(["ENSG1", "ENSG2", "ENSG3"]))
        feats.create_dataset("name", data=_bytes_arr(["GeneA", "GeneB", "GeneC"]))
        feats.create_dataset("feature_type", data=_bytes_arr(["Gene Expression"] * 3))
        feats.create_dataset("genome", data=_bytes_arr(["mm10"] * 3))

        fs = f.create_group("feature_slices")

        def add_gene(idx, rows, cols, data):
            g = fs.create_group(str(idx))
            g.create_dataset("row", data=np.array(rows, dtype="uint32"))
            g.create_dataset("col", data=np.array(cols, dtype="uint32"))
            g.create_dataset("data", data=np.array(data, dtype="uint32"))

        add_gene(0, [0, 1, 4], [0, 1, 5], [5, 3, 2])
        add_gene(1, [0, 5], [5, 4], [4, 1])
        add_gene(2, [2, 6, 4], [2, 6, 0], [7, 9, 100])

        masks = f.create_group("masks")
        m = masks.create_group("square_008um")
        # tissue bins (0,0),(0,1),(1,1) on the 8um grid; data==1
        m.create_dataset("row", data=np.array([0, 0, 1], dtype="int64"))
        m.create_dataset("col", data=np.array([0, 1, 1], dtype="int64"))
        m.create_dataset("data", data=np.array([1, 1, 1], dtype="int64"))

        sa = f.create_group("secondary_analysis").create_group("clustering")
        cl = sa.create_group("square_008um_gene_expression_graphclust")
        cl.create_dataset("row", data=np.array([0, 0, 1], dtype="int64"))
        cl.create_dataset("col", data=np.array([0, 1, 1], dtype="int64"))
        cl.create_dataset("data", data=np.array([1, 2, 1], dtype="int64"))


@pytest.fixture
def fs_path(tmp_path):
    p = tmp_path / "Sample_feature_slice.h5"
    write_feature_slice(p)
    return p


def test_is_feature_slice_true(fs_path):
    assert vh.is_feature_slice(fs_path) is True


def test_is_feature_slice_false_for_plain_h5(tmp_path):
    p = tmp_path / "plain.h5"
    with h5py.File(p, "w") as f:
        f.create_group("matrix")  # looks like a 10x feature-barcode file
    assert vh.is_feature_slice(p) is False


def test_converted_shape(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8)
    assert adata.shape == (3, 3)  # 3 tissue bins x 3 genes


def test_rebinned_counts(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8)
    X = adata.X.toarray()
    expected = np.array([[8, 0, 7], [0, 4, 0], [2, 1, 9]], dtype=X.dtype)
    np.testing.assert_array_equal(X, expected)


def test_var_gene_metadata(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8)
    assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3"]
    assert list(adata.var["name"]) == ["GeneA", "GeneB", "GeneC"]


def test_non_tissue_bin_dropped(fs_path):
    # The count of 100 lived in non-tissue bin (1,0); it must not appear anywhere.
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8)
    assert 100 not in set(np.asarray(adata.X.toarray()).ravel().tolist())
    assert "1_0" not in set(adata.obs_names)


def test_spatial_coords(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8)
    assert "spatial" in adata.obsm
    coords = np.asarray(adata.obsm["spatial"])
    assert coords.shape == (3, 2)
    # obs order (0,0),(0,1),(1,1); x=bc*8, y=-(br*8)
    expected = np.array([[0, 0], [8, 0], [8, -8]], dtype=coords.dtype)
    np.testing.assert_array_equal(coords, expected)
    assert list(adata.obs_names) == ["0_0", "0_1", "1_1"]


def test_clusters_imported_as_categorical(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8, import_clusters=True)
    assert "graphclust" in adata.obs
    col = adata.obs["graphclust"]
    assert str(col.dtype) == "category"
    assert list(col.astype(str)) == ["1", "2", "1"]


def test_clusters_skipped_when_disabled(fs_path):
    adata = vh.feature_slice_to_anndata(fs_path, bin_size=8, import_clusters=False)
    assert "graphclust" not in adata.obs


def test_cache_created_and_reused(fs_path, monkeypatch):
    adata1 = vh.load_feature_slice_cached(fs_path, bin_size=8)
    cache = fs_path.with_name("Sample_feature_slice.xcell.square_008um.h5ad")
    assert cache.exists()
    assert adata1.shape == (3, 3)

    # Second load must NOT re-run the converter (cache hit).
    calls = {"n": 0}
    real = vh.feature_slice_to_anndata

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(vh, "feature_slice_to_anndata", spy)
    adata2 = vh.load_feature_slice_cached(fs_path, bin_size=8)
    assert calls["n"] == 0
    assert adata2.shape == (3, 3)
