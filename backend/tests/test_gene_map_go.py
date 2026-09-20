"""The gene map's GO channel: gated on the fetched files, computed through go_semantic.

The similarity itself is pinned in test_go_semantic; here the toy ontology
annotates the planted gene blocks so the channel has a known block structure
and the adaptor's bookkeeping (params, channel info, JSON shape) is checked.
"""
import gzip
import time

import anndata
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell import go_semantic as gos
from xcell.adaptor import DataAdaptor

N_CELLS, N_GENES = 120, 30
GENES = [f"G{i:03d}" for i in range(N_GENES)]

OBO = """format-version: 1.2
data-version: releases/2026-07-26

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000010
name: process ten
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0000020
name: process twenty
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0000030
name: process thirty
namespace: biological_process
is_a: GO:0008150
"""


def _gaf():
    lines = ['!gaf-version: 2.2', '!date-generated: 2026-05-21']
    for i, g in enumerate(GENES):
        term = ['GO:0000010', 'GO:0000020', 'GO:0000030'][i // 10]
        lines.append('\t'.join(['MGI', f'MGI:{i}', g, 'involved_in', term, 'PMID:1', 'IDA', '', 'P', g, '',
                                'protein', 'taxon:10090', '20130101', 'MGI', '', f'MGI:MGI:{i}']))
    return '\n'.join(lines) + '\n'


@pytest.fixture
def go_files(tmp_path):
    gss.set_cache_dir(tmp_path / 'cache')
    gos.go_dir().mkdir(parents=True)
    gos.obo_path().write_text(OBO)
    with gzip.open(gos.gaf_path('mouse'), 'wt') as fh:
        fh.write(_gaf())
    gos._ontology_cache.clear(); gos._gaf_cache.clear(); gos._space_cache.clear()
    yield
    gss.set_cache_dir(None)


def _adaptor():
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(2.0, (N_CELLS, N_GENES)).astype(np.float32)))
    ad.var_names = GENES
    return DataAdaptor('x.h5ad', adata=ad)


def _run(a, **kw):
    compute_fn, apply_fn = a.prepare_gene_map(**kw)
    return apply_fn(compute_fn(lambda *args, **k: None))


def test_go_channel_needs_the_files_and_names_the_library_to_fetch(tmp_path):
    gss.set_cache_dir(tmp_path / 'empty')
    try:
        with pytest.raises(ValueError, match="fetch 'GO annotations \\(mouse\\)'"):
            _adaptor().prepare_gene_map(genes=GENES, key='m', expression_weight=0.0, go_weight=1.0, go_species='mouse')
    finally:
        gss.set_cache_dir(None)


def test_go_channel_alone_recovers_the_annotated_blocks(go_files):
    a = _adaptor()
    out = _run(a, genes=GENES, key='gomap', expression_weight=0.0, annotation_weight=0.0,
               go_weight=1.0, go_species='mouse', go_aspect='bp', n_neighbors=5, embedding='mds')
    assert set(out['channel_weights']) == {'go'} and out['channel_weights']['go'] == pytest.approx(1.0)
    assert out['channels']['go'] == {'weight': 1.0, 'species': 'mouse', 'aspect': 'bp',
                                     'n_genes_annotated': 30, 'n_terms': 4}
    rec = a.adata.uns['xcell_gene_maps']['gomap']
    S = np.asarray(rec['similarity'])
    # Same term → identical closed sets → 1; different leaves share only the root (IC 0) → 0.
    assert S[0, 1] == pytest.approx(1.0) and S[0, 10] == pytest.approx(0.0)
    assert out['n_modules'] == 3 and sorted(out['module_sizes']) == [10, 10, 10]
    assert rec['params']['go_aspect'] == 'bp' and rec['params']['go_species'] == 'mouse'
    assert rec['channels']['go']['terms_per_gene'] == [2] * 30
    # The read-back strips the per-gene list from the summary but keeps it in channels.
    got = a.get_gene_map('gomap')
    assert got['channels']['go']['n_genes_annotated'] == 30


def test_go_channel_combines_with_expression_and_is_recorded(go_files):
    a = _adaptor()
    out = _run(a, genes=GENES, key='both', expression_weight=1.0, expression_metric='pearson',
               go_weight=1.0, go_species='mouse', n_neighbors=5, embedding='mds')
    assert set(out['channel_weights']) == {'expression', 'go'}
    assert out['channel_weights']['go'] == pytest.approx(0.5)
    step = a.analysis_record.steps[-1]
    assert step.action == 'gene_map' and step.params['go_weight'] == 1.0


def test_a_bad_aspect_is_a_400_before_the_task(go_files):
    with pytest.raises(ValueError, match='go_aspect'):
        _adaptor().prepare_gene_map(genes=GENES, key='m', expression_weight=0.0, go_weight=1.0,
                                    go_species='mouse', go_aspect='zz')


def test_the_route_takes_the_go_parameters(go_files, monkeypatch):
    from xcell.api import routes
    from xcell.main import app
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    r = client.post('/api/gene_map/run', json={'genes': GENES, 'key': 'viaroute', 'expression_weight': 0.0,
                                               'annotation_weight': 0.0, 'go_weight': 1.0, 'go_species': 'mouse',
                                               'n_neighbors': 5, 'embedding': 'mds'})
    assert r.status_code == 202, r.text
    task_id = r.json()['task_id']
    for _ in range(200):
        st = client.get(f'/api/tasks/{task_id}').json()
        if st['status'] in ('completed', 'error'):
            break
        time.sleep(0.05)
    assert st['status'] == 'completed', st
    assert st['result']['channels']['go']['n_genes_annotated'] == 30
