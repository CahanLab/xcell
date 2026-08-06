"""Filesystem browsing for different kinds of file.

The Load dialog browses for datasets; the Cell Typing dialog browses for
pickled classifiers. Same navigation, different set of files worth showing —
so `kind` selects the filter and everything else stays shared.
"""
from fastapi.testclient import TestClient

from xcell.main import app


def _make_tree(tmp_path):
    (tmp_path / 'sub').mkdir()
    for name in (
        'data.h5ad', 'other.h5', 'seurat.rds',
        'clf.pkl', 'model.pickle',
        'notes.txt', 'script.py',
    ):
        (tmp_path / name).write_bytes(b'x' * 10)
    return tmp_path


def _names(resp):
    return {e['name'] for e in resp.json()['entries']}


def test_default_kind_lists_datasets_and_is_unchanged(tmp_path):
    """The Load dialog's existing behaviour must not shift."""
    client = TestClient(app)
    resp = client.get('/api/browse', params={'path': str(_make_tree(tmp_path))})
    assert resp.status_code == 200, resp.text
    names = _names(resp)
    assert {'data.h5ad', 'other.h5', 'seurat.rds'} <= names
    assert 'clf.pkl' not in names
    assert 'notes.txt' not in names


def test_classifier_kind_lists_pickles(tmp_path):
    client = TestClient(app)
    resp = client.get('/api/browse', params={
        'path': str(_make_tree(tmp_path)), 'kind': 'classifier',
    })
    assert resp.status_code == 200, resp.text
    names = _names(resp)
    assert {'clf.pkl', 'model.pickle'} <= names
    # Datasets are noise when you are looking for a classifier.
    assert 'data.h5ad' not in names
    assert 'notes.txt' not in names


def test_directories_are_listed_for_every_kind(tmp_path):
    """Navigation has to work regardless of what you are hunting for."""
    client = TestClient(app)
    tree = str(_make_tree(tmp_path))
    for kind in (None, 'data', 'classifier'):
        params = {'path': tree}
        if kind:
            params['kind'] = kind
        assert 'sub' in _names(client.get('/api/browse', params=params))


def test_file_sizes_are_reported_for_classifiers(tmp_path):
    """Classifier pickles run to hundreds of MB; the size is worth seeing."""
    client = TestClient(app)
    resp = client.get('/api/browse', params={
        'path': str(_make_tree(tmp_path)), 'kind': 'classifier',
    })
    entry = next(e for e in resp.json()['entries'] if e['name'] == 'clf.pkl')
    assert entry['size'] == 10
    assert entry['type'] == 'file'


def test_unknown_kind_is_rejected(tmp_path):
    client = TestClient(app)
    resp = client.get('/api/browse', params={
        'path': str(_make_tree(tmp_path)), 'kind': 'nonsense',
    })
    assert resp.status_code == 400
    assert 'kind' in resp.json()['detail'].lower()


def test_shortcuts_are_returned_for_every_kind(tmp_path):
    client = TestClient(app)
    resp = client.get('/api/browse', params={
        'path': str(_make_tree(tmp_path)), 'kind': 'classifier',
    })
    assert isinstance(resp.json()['shortcuts'], list)
