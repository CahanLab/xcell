"""POST /api/session/reset — back to a fresh session without restarting the server.

File → New session in the browser calls this and then reloads the page. The
point is that no AnnData from the previous analysis survives: every slot is
dropped (not just 'primary'), background tasks still computing on those
datasets are cancelled so their results are never applied, and the gene-set
store is emptied only when asked — gene sets are user-level and a new analysis
often reuses them.
"""
import threading
import time

import anndata
import numpy as np
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import gene_set_store
from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app
from xcell.task_manager import TaskManager


def _adata(n_cells=30, n_genes=8):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(
        X=csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(np.float32))
    )
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    return ad


def _install(*slots):
    for slot in slots:
        routes.set_adaptor(DataAdaptor(f'{slot}.h5ad', adata=_adata()), slot=slot)


def _clean():
    for slot in list(routes.list_adaptors()):
        routes.remove_adaptor(slot)
    gene_set_store.reset()


def _client():
    return TestClient(app)


def _wait_terminal(tm, task_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = tm.get_status(task_id)
        if entry is None or entry.status != 'running':
            return entry
        time.sleep(0.01)
    return tm.get_status(task_id)


# --- TaskManager.cancel_all ---------------------------------------------------

def test_cancel_all_cancels_every_running_task_and_skips_apply():
    tm = TaskManager(max_workers=2)
    gate = threading.Event()
    applied = []

    def compute():
        gate.wait(10)
        return {'x': 1}

    ids = [tm.submit(compute, lambda r: applied.append(r) or r) for _ in range(2)]
    assert tm.cancel_all() == 2
    gate.set()
    for tid in ids:
        assert _wait_terminal(tm, tid).status == 'cancelled'
    # The whole point: a result computed on a dropped dataset is never written.
    assert applied == []


def test_cancel_all_ignores_finished_tasks():
    tm = TaskManager(max_workers=1)
    tid = tm.submit(lambda: {'x': 1}, lambda r: r)
    assert _wait_terminal(tm, tid).status == 'completed'
    assert tm.cancel_all() == 0
    assert tm.get_status(tid).status == 'completed'


# --- the route ------------------------------------------------------------------

def test_reset_unloads_every_slot():
    _clean()
    _install('primary', 'secondary', 'slot3')
    r = _client().post('/api/session/reset', json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body['unloaded']) == ['primary', 'secondary', 'slot3']
    assert routes.list_adaptors() == {}
    # Nothing is loaded afterwards, so a schema request says so.
    assert _client().get('/api/schema').status_code == 503
    assert _client().get('/api/datasets').json() == {}


def test_reset_keeps_gene_sets_unless_asked():
    _clean()
    _install('primary')
    gene_set_store.set_gene_sets({'manual': {'geneSets': [{'name': 'a', 'genes': ['g1']}]}})
    body = _client().post('/api/session/reset', json={}).json()
    assert body['gene_sets_cleared'] is False
    assert gene_set_store.get_gene_sets() != {}


def test_reset_clears_gene_sets_when_asked():
    _clean()
    _install('primary')
    gene_set_store.set_gene_sets({'manual': {'geneSets': [{'name': 'a', 'genes': ['g1']}]}})
    body = _client().post('/api/session/reset', json={'clear_gene_sets': True}).json()
    assert body['gene_sets_cleared'] is True
    assert gene_set_store.get_gene_sets() == {}
    assert _client().get('/api/gene_sets').json() == {'gene_sets': {}}


def test_reset_with_nothing_loaded_is_not_an_error():
    _clean()
    r = _client().post('/api/session/reset', json={})
    assert r.status_code == 200
    assert r.json()['unloaded'] == []


def test_reset_cancels_running_tasks(monkeypatch):
    _clean()
    _install('primary')
    tm = TaskManager(max_workers=1)
    monkeypatch.setattr(routes, 'task_manager', tm)
    gate = threading.Event()
    tid = tm.submit(lambda: gate.wait(10) and {'x': 1}, lambda r: r)
    body = _client().post('/api/session/reset', json={}).json()
    gate.set()
    assert body['cancelled_tasks'] == 1
    assert _wait_terminal(tm, tid).status == 'cancelled'
