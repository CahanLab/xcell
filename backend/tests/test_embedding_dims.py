"""Viewing / transforming / associating on a chosen pair of .obsm columns."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata():
    rng = np.random.default_rng(3)
    n_cells, n_genes = 40, 6
    X = csr_matrix(rng.integers(0, 5, size=(n_cells, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obsm['X_pca'] = rng.normal(size=(n_cells, 4)).astype(np.float64)   # 4-D
    ad.obsm['X_spatial'] = rng.normal(size=(n_cells, 2)).astype(np.float64)
    return ad


def _a():
    return DataAdaptor("x.h5ad", adata=_adata())


def test_get_embedding_default_and_custom_dims():
    a = _a(); pca = a.adata.obsm['X_pca']
    d = a.get_embedding('X_pca')
    assert (d['dim_x'], d['dim_y']) == (0, 1)
    np.testing.assert_allclose([c[0] for c in d['coordinates']], pca[:, 0])
    np.testing.assert_allclose([c[1] for c in d['coordinates']], pca[:, 1])
    d2 = a.get_embedding('X_pca', dim_x=2, dim_y=3)
    assert (d2['dim_x'], d2['dim_y']) == (2, 3)
    np.testing.assert_allclose([c[0] for c in d2['coordinates']], pca[:, 2])
    np.testing.assert_allclose([c[1] for c in d2['coordinates']], pca[:, 3])


def test_get_embedding_out_of_range_clamps():
    a = _a()
    d = a.get_embedding('X_pca', dim_x=9, dim_y=9)
    assert (d['dim_x'], d['dim_y']) == (0, 1)


def test_schema_embedding_dims():
    a = _a(); s = a.get_schema()
    assert s['embedding_dims']['X_pca'] == 4
    assert s['embedding_dims']['X_spatial'] == 2


def test_transform_operates_on_selected_dims_only():
    a = _a(); before = a.adata.obsm['X_pca'].copy()
    a.transform_embedding('X_pca', rotation_degrees=90, dim_x=2, dim_y=3)
    after = a.adata.obsm['X_pca']
    np.testing.assert_allclose(after[:, 0], before[:, 0])   # col 0 untouched
    np.testing.assert_allclose(after[:, 1], before[:, 1])   # col 1 untouched
    assert not np.allclose(after[:, 2], before[:, 2])       # col 2 rotated
    # rotation preserves the centroid of the transformed columns
    np.testing.assert_allclose(after[:, [2, 3]].mean(0), before[:, [2, 3]].mean(0), atol=1e-9)


def test_undo_restores_selected_dims():
    a = _a(); before = a.adata.obsm['X_pca'].copy()
    a.transform_embedding('X_pca', translate_x=5.0, cell_indices=[0, 1, 2], dim_x=2, dim_y=3)
    assert not np.allclose(a.adata.obsm['X_pca'][:, 2], before[:, 2])
    a.undo_transform_embedding('X_pca', dim_x=2, dim_y=3)
    np.testing.assert_allclose(a.adata.obsm['X_pca'], before, atol=1e-9)


def test_line_view_coords_uses_line_dims():
    a = _a(); pca = a.adata.obsm['X_pca']
    line = {'name': 'L', 'embeddingName': 'X_pca', 'points': [[0, 0], [1, 1]], 'dimX': 1, 'dimY': 3}
    coords = a._line_view_coords(line)
    np.testing.assert_allclose(coords[:, 0], pca[:, 1])
    np.testing.assert_allclose(coords[:, 1], pca[:, 3])
    # no dims recorded → first two columns
    coords2 = a._line_view_coords({'name': 'L', 'embeddingName': 'X_pca', 'points': [[0, 0], [1, 1]]})
    np.testing.assert_allclose(coords2[:, 0], pca[:, 0])
    np.testing.assert_allclose(coords2[:, 1], pca[:, 1])


def test_line_association_projects_on_line_dims():
    """A line on cols (0,2) must project cells onto those columns, not (0,1)."""
    a = _a()
    a.set_lines([{'name': 'L', 'embeddingName': 'X_pca',
                  'points': [[-2.0, -2.0], [2.0, 2.0]], 'dimX': 0, 'dimY': 2}])
    projections = a.compute_line_projections()['L']
    pca = a.adata.obsm['X_pca']
    # positions come from projecting the (col0, col2) coords onto the line
    expected_pos, _ = a._project_cells_onto_line(
        [[-2.0, -2.0], [2.0, 2.0]], pca[:, [0, 2]])
    np.testing.assert_allclose(projections['positions'], expected_pos, atol=1e-9)
    # and NOT equal to projecting the (col0, col1) coords (unless degenerate)
    pos01, _ = a._project_cells_onto_line([[-2.0, -2.0], [2.0, 2.0]], pca[:, [0, 1]])
    assert not np.allclose(projections['positions'], pos01)
