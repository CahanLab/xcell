"""UCell-style directional gene-set scoring."""
import numpy as np
import anndata
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 2 cells x 4 genes, hand-chosen so descending ranks are exact.
    # Cell0: A=10,B=5,C=1,D=0 -> ranks A=1,B=2,C=3,D=4
    # Cell1: A=0,B=1,C=5,D=10 -> ranks D=1,C=2,B=3,A=4
    X = np.array([[10, 5, 1, 0],
                  [0, 1, 5, 10]], dtype=np.float32)
    ad = anndata.AnnData(X=sp.csr_matrix(X))
    ad.var_names = ["A", "B", "C", "D"]
    return ad


def test_ucell_ranks_caps_to_n_genes_and_caches():
    a = DataAdaptor("x.h5ad", adata=_adata())
    ranks = a._ucell_ranks("X", 1500)   # maxRank auto-capped to n_genes=4
    assert sp.issparse(ranks)
    assert ranks.shape == (2, 4)
    # rank>=maxRank(4) dropped to 0; cell0 keeps A=1,B=2,C=3, drops D
    dense = ranks.toarray()
    assert dense[0, 0] == 1 and dense[0, 1] == 2 and dense[0, 2] == 3
    assert dense[0, 3] == 0   # D rank 4 == maxRank -> dropped
    # identical call returns the SAME cached object
    assert a._ucell_ranks("X", 1500) is ranks
