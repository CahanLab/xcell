"""PySingleCellNet cell-type classification adapter.

These tests run whether or not pySingleCellNet is installed — that is the
point of the integration being optional. Everything that does not need the
package (classifier introspection, gene alignment, thresholding, error
messages, route behaviour when the package is absent) is tested
unconditionally; the handful of tests that actually invoke PySCN skip
themselves.
"""
import pickle
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse
from fastapi.testclient import TestClient

from xcell import pyscn
from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


HAVE_PYSCN = pyscn.availability()['available']
needs_pyscn = pytest.mark.skipif(not HAVE_PYSCN, reason='pySingleCellNet not installed')


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

CLASSES = ['B cell', 'T cell', 'rand']
CLF_GENES = ['Cd19', 'Ms4a1', 'Cd3e', 'Cd8a', 'Actb']


def _fake_clf(genes=CLF_GENES, classes=CLASSES):
    """A stand-in for a trained PySCN classifier dict.

    Mirrors the real shape returned by ``pySingleCellNet.tl.train_classifier``:
    the gene array used for the top-scoring-pair transform, the pair list, the
    fitted sklearn estimator, per-class colors, and the training arguments.
    """
    from sklearn.ensemble import RandomForestClassifier

    pairs = [f'{a}_{b}' for a in genes[:3] for b in genes[3:]]
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(30, len(pairs)))
    y = rng.choice(classes, size=30)
    est = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    return {
        'tpGeneArray': np.array(genes),
        'topPairs': np.array(pairs),
        'classifier': est,
        'diffExpGenes': {c: list(genes[:2]) for c in classes if c != 'rand'},
        'ctColors': {'B cell': (0.1, 0.2, 0.9), 'T cell': '#ff0000', 'rand': (0.5, 0.5, 0.5)},
        'argList': {'n_rand': 100, 'n_top_genes': 30, 'n_top_gene_pairs': 40, 'n_trees': 1000},
    }


def _adata(genes=CLF_GENES, n_cells=40, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.integers(0, 20, (n_cells, len(genes))).astype(np.float32)))
    ad.var_names = list(genes)
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    return ad


# --------------------------------------------------------------------------
# Availability — the integration must degrade cleanly, never crash on import
# --------------------------------------------------------------------------

def test_availability_reports_a_usable_shape():
    a = pyscn.availability()
    assert isinstance(a['available'], bool)
    assert 'install_hint' in a and 'pip install' in a['install_hint']
    if a['available']:
        assert a['version']
    else:
        # A user staring at a disabled button needs to know why.
        assert a['error']


def test_import_pyscn_raises_an_actionable_error_when_missing(monkeypatch):
    monkeypatch.setattr(pyscn, '_import_module', lambda: None)
    with pytest.raises(ValueError) as exc:
        pyscn.import_pyscn()
    msg = str(exc.value)
    assert 'pip install pySingleCellNet' in msg


# --------------------------------------------------------------------------
# Classifier loading and introspection
# --------------------------------------------------------------------------

def test_describe_classifier_surfaces_classes_and_training_params():
    d = pyscn.describe_classifier(_fake_clf())
    assert d['classes'] == CLASSES
    # 'rand' is an artifact of training, not a biological call — flag it.
    assert d['cell_type_classes'] == ['B cell', 'T cell']
    assert d['n_genes'] == len(CLF_GENES)
    assert d['n_gene_pairs'] > 0
    assert d['train_params']['n_trees'] == 1000


def test_load_classifier_round_trips_a_pickle(tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    clf, meta = pyscn.load_classifier(str(p))
    assert meta['classes'] == CLASSES
    assert set(clf) >= {'tpGeneArray', 'topPairs', 'classifier'}


def test_load_classifier_rejects_a_file_that_is_not_a_classifier(tmp_path):
    p = tmp_path / 'nope.pkl'
    with open(p, 'wb') as fh:
        pickle.dump({'hello': 'world'}, fh)
    with pytest.raises(ValueError) as exc:
        pyscn.load_classifier(str(p))
    assert 'tpGeneArray' in str(exc.value)


def test_load_classifier_reports_a_missing_file(tmp_path):
    with pytest.raises(ValueError) as exc:
        pyscn.load_classifier(str(tmp_path / 'absent.pkl'))
    assert 'not found' in str(exc.value).lower()


# --------------------------------------------------------------------------
# Gene overlap — PySCN fills missing genes with zeros, silently. This is the
# single most important guard in the integration.
# --------------------------------------------------------------------------

def test_gene_overlap_full_match():
    r = pyscn.assess_gene_overlap(CLF_GENES, _fake_clf())
    assert r['n_required'] == len(CLF_GENES)
    assert r['n_found'] == len(CLF_GENES)
    assert r['frac_found'] == 1.0
    assert r['missing'] == []
    assert r['severity'] == 'ok'


def test_gene_overlap_partial_is_flagged_and_lists_the_missing():
    r = pyscn.assess_gene_overlap(['Cd19', 'Ms4a1', 'Xyz'], _fake_clf())
    assert r['n_found'] == 2
    assert set(r['missing']) == {'Cd3e', 'Cd8a', 'Actb'}
    assert r['severity'] in ('warn', 'error')


def test_gene_overlap_severity_escalates_as_coverage_drops():
    clf = _fake_clf()
    ok = pyscn.assess_gene_overlap(CLF_GENES, clf)['severity']
    warn = pyscn.assess_gene_overlap(CLF_GENES[:4], clf)['severity']
    bad = pyscn.assess_gene_overlap(CLF_GENES[:1], clf)['severity']
    assert (ok, warn, bad) == ('ok', 'warn', 'error')


def test_gene_overlap_detects_a_pure_case_mismatch():
    """UPPERCASE query vs mixed-case classifier is a rename, not a real miss."""
    upper = [g.upper() for g in CLF_GENES]
    r = pyscn.assess_gene_overlap(upper, _fake_clf())
    assert r['n_found'] == 0
    assert r['n_found_case_insensitive'] == len(CLF_GENES)
    assert r['case_mismatch_only'] is True


def test_gene_overlap_does_not_claim_case_mismatch_when_genes_truly_differ():
    r = pyscn.assess_gene_overlap(['Aaa', 'Bbb'], _fake_clf())
    assert r['case_mismatch_only'] is False


# --------------------------------------------------------------------------
# Query matrix construction — align to the classifier's gene array, in order,
# without densifying the whole dataset.
# --------------------------------------------------------------------------

def test_build_query_adata_orders_columns_to_the_classifier():
    ad = _adata(genes=['Actb', 'Cd8a', 'Cd3e', 'Ms4a1', 'Cd19'])  # shuffled
    q = pyscn.build_query_adata(ad, _fake_clf())
    assert list(q.var_names) == CLF_GENES
    assert q.n_obs == ad.n_obs
    # Column content must follow the gene, not the position.
    src = ad[:, 'Cd3e'].X.toarray().ravel()
    got = q[:, 'Cd3e'].X.toarray().ravel()
    np.testing.assert_array_equal(src, got)


def test_build_query_adata_zero_fills_genes_the_dataset_lacks():
    ad = _adata(genes=['Cd19', 'Cd3e'])
    q = pyscn.build_query_adata(ad, _fake_clf())
    assert list(q.var_names) == CLF_GENES
    np.testing.assert_array_equal(q[:, 'Actb'].X.toarray().ravel(), np.zeros(ad.n_obs))


def test_build_query_adata_keeps_x_sparse():
    """PySCN's predict path calls .X.toarray() — a dense X would crash it."""
    q = pyscn.build_query_adata(_adata(), _fake_clf())
    assert issparse(q.X)


def test_build_query_adata_is_narrow_regardless_of_dataset_width():
    """Only the classifier's genes are materialized, so wide data is safe."""
    ad = _adata(genes=CLF_GENES + [f'filler{i}' for i in range(5000)])
    q = pyscn.build_query_adata(ad, _fake_clf())
    assert q.n_vars == len(CLF_GENES)


def test_build_query_adata_can_resolve_case_differences():
    ad = _adata(genes=[g.upper() for g in CLF_GENES])
    q = pyscn.build_query_adata(ad, _fake_clf(), case_insensitive=True)
    assert list(q.var_names) == CLF_GENES
    assert q.X.nnz > 0  # genes were actually matched, not zero-filled


def test_build_query_adata_reads_a_named_layer():
    ad = _adata()
    ad.layers['counts'] = csr_matrix(np.ones((ad.n_obs, ad.n_vars), dtype=np.float32))
    q = pyscn.build_query_adata(ad, _fake_clf(), layer='counts')
    np.testing.assert_array_equal(q.X.toarray(), np.ones((ad.n_obs, len(CLF_GENES))))


def test_build_query_adata_tolerates_duplicate_gene_names():
    ad = _adata(genes=['Cd19', 'Cd19', 'Cd3e', 'Cd8a', 'Actb'])
    q = pyscn.build_query_adata(ad, _fake_clf())
    assert list(q.var_names) == CLF_GENES  # first occurrence wins, no crash


def test_build_query_adata_refuses_when_nothing_overlaps():
    ad = _adata(genes=['Zzz1', 'Zzz2'])
    with pytest.raises(ValueError) as exc:
        pyscn.build_query_adata(ad, _fake_clf())
    assert 'overlap' in str(exc.value).lower()


# --------------------------------------------------------------------------
# Post-processing: thresholds and category calls
# --------------------------------------------------------------------------

def _scores():
    return pd.DataFrame(
        [
            [0.90, 0.05, 0.05],   # clean B cell
            [0.05, 0.90, 0.05],   # clean T cell
            [0.45, 0.45, 0.10],   # two types above threshold -> ambiguous
            [0.20, 0.20, 0.60],   # rand wins
            [0.30, 0.28, 0.42],   # nothing convincing
        ],
        columns=CLASSES,
        index=[f'c{i}' for i in range(5)],
    )


def test_argmax_labels_and_confidence():
    r = pyscn.summarize_scores(_scores())
    assert r['labels'] == ['B cell', 'T cell', 'B cell', 'rand', 'rand']
    np.testing.assert_allclose(r['confidence'], [0.90, 0.90, 0.45, 0.60, 0.42])


def test_class_types_split_singular_ambiguous_none_and_rand():
    scores = _scores()
    labels = pyscn.summarize_scores(scores)['labels']
    types = pyscn.derive_class_types(scores, labels, thresholds={'B cell': 0.4, 'T cell': 0.4})
    assert types == ['Singular', 'Singular', 'Ambiguous', 'Rand', 'Rand']


def test_thresholds_mirror_pyscn_comp_ct_thresh():
    """Per-class quantile of the scores of cells argmax-assigned to that class."""
    scores = _scores()
    labels = pyscn.summarize_scores(scores)['labels']
    th = pyscn.compute_thresholds(scores, labels, quantile=0.05)
    assert set(th) == {'B cell', 'T cell'}          # 'rand' is excluded
    # B cell is the argmax for rows 0 and 2 (scores .90 and .45).
    assert th['B cell'] == pytest.approx(np.quantile([0.90, 0.45], 0.05))


def test_thresholds_handle_a_class_no_cell_was_assigned_to():
    scores = _scores()
    labels = ['B cell'] * 5
    th = pyscn.compute_thresholds(scores, labels, quantile=0.05)
    assert th['T cell'] == 0.0


def test_colors_are_converted_to_hex_for_the_frontend():
    hexes = pyscn.classifier_colors_hex(_fake_clf())
    assert hexes['B cell'] == '#1a33e6'
    assert hexes['T cell'] == '#ff0000'
    assert all(v.startswith('#') and len(v) == 7 for v in hexes.values())


# --------------------------------------------------------------------------
# Adaptor integration
# --------------------------------------------------------------------------

def test_inspect_classifier_combines_metadata_and_overlap(tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    a = DataAdaptor('x.h5ad', adata=_adata())
    info = a.pyscn_inspect_classifier(str(p))
    assert info['classifier']['classes'] == CLASSES
    assert info['gene_overlap']['frac_found'] == 1.0


def test_apply_classification_writes_obs_obsm_and_registry():
    a = DataAdaptor('x.h5ad', adata=_adata(n_cells=5))
    a._apply_pyscn_result({
        'scores': _scores(),
        'classes': CLASSES,
        'colors': {'B cell': '#1a33e6', 'T cell': '#ff0000', 'rand': '#808080'},
        'gene_overlap': {'n_required': 5, 'n_found': 5, 'frac_found': 1.0,
                         'missing': [], 'severity': 'ok'},
        'key': 'SCN',
        'categorize': True,
        'quantile': 0.05,
        'classifier_path': '/tmp/clf.pkl',
    })
    ad = a.adata
    assert list(ad.obs['SCN_class_argmax'][:2]) == ['B cell', 'T cell']
    assert ad.obs['SCN_class_argmax'].dtype.name == 'category'
    assert 'SCN_class_score' in ad.obs and ad.obs['SCN_class_score'].iloc[0] == pytest.approx(0.9)
    assert 'SCN_class_type' in ad.obs
    assert ad.obsm['SCN_score'].shape == (5, 3)
    # Registered so the existing score-matrix UI can name the columns.
    assert ad.uns['xcell_score_matrices']['SCN_score']['columns'] == CLASSES
    # Colors flow to scanpy's per-column convention, which xcell already reads.
    assert ad.uns['SCN_class_argmax_colors'][0] == '#1a33e6'


def test_apply_classification_is_rerunnable_under_a_second_key():
    a = DataAdaptor('x.h5ad', adata=_adata(n_cells=5))
    common = {
        'scores': _scores(), 'classes': CLASSES,
        'colors': {c: '#888888' for c in CLASSES},
        'gene_overlap': {'n_required': 5, 'n_found': 5, 'frac_found': 1.0,
                         'missing': [], 'severity': 'ok'},
        'categorize': False, 'quantile': 0.05, 'classifier_path': 'p',
    }
    a._apply_pyscn_result({**common, 'key': 'SCN'})
    a._apply_pyscn_result({**common, 'key': 'PBMC'})
    assert {'SCN_class_argmax', 'PBMC_class_argmax'} <= set(a.adata.obs.columns)
    assert {'SCN_score', 'PBMC_score'} <= set(a.adata.obsm.keys())


def test_classification_result_is_recorded_in_uns_and_history():
    a = DataAdaptor('x.h5ad', adata=_adata(n_cells=5))
    a._apply_pyscn_result({
        'scores': _scores(), 'classes': CLASSES,
        'colors': {c: '#888888' for c in CLASSES},
        'gene_overlap': {'n_required': 5, 'n_found': 4, 'frac_found': 0.8,
                         'missing': ['Actb'], 'severity': 'warn'},
        'key': 'SCN', 'categorize': False, 'quantile': 0.05,
        'classifier_path': '/data/pbmc.pkl',
    })
    rec = a.adata.uns['pyscn']['SCN']
    assert rec['classifier_path'] == '/data/pbmc.pkl'
    assert rec['gene_overlap']['severity'] == 'warn'
    assert any(e['action'] == 'pyscn_classify' for e in a._action_history)


def test_composition_summary_counts_cells_per_class():
    a = DataAdaptor('x.h5ad', adata=_adata(n_cells=5))
    out = a._apply_pyscn_result({
        'scores': _scores(), 'classes': CLASSES,
        'colors': {c: '#888888' for c in CLASSES},
        'gene_overlap': {'n_required': 5, 'n_found': 5, 'frac_found': 1.0,
                         'missing': [], 'severity': 'ok'},
        'key': 'SCN', 'categorize': True, 'quantile': 0.05, 'classifier_path': 'p',
    })
    comp = {c['name']: c for c in out['composition']}
    assert comp['B cell']['n_cells'] == 2
    assert comp['T cell']['n_cells'] == 1
    assert out['n_cells'] == 5
    assert sum(c['n_cells'] for c in out['composition']) == 5
    # Types are derived from self-calibrated thresholds (comp_ct_thresh), so
    # which buckets appear depends on the data; every cell must land in one.
    assert sum(t['n_cells'] for t in out['class_types']) == 5
    assert {t['name'] for t in out['class_types']} <= set(pyscn.CLASS_TYPE_ORDER)
    assert out['class_types'][0]['name'] == 'Singular'  # ordered, not by count


# --- score matrices produced outside xcell (e.g. by PySCN itself) ----------

def test_dataframe_obsm_columns_are_readable():
    """An h5ad classified in PySCN stores obsm['SCN_score'] as a DataFrame."""
    ad = _adata(n_cells=5)
    ad.obsm['SCN_score'] = _scores()
    a = DataAdaptor('x.h5ad', adata=ad)
    assert a.get_schema()['score_matrices']['SCN_score'] == CLASSES
    col = a.get_obsm_column('SCN_score', 'T cell')
    np.testing.assert_allclose(col['values'], _scores()['T cell'].to_numpy())


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

def test_route_status_reports_availability(monkeypatch):
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: DataAdaptor('x.h5ad', adata=_adata()))
    client = TestClient(app)
    resp = client.get('/api/pyscn/status')
    assert resp.status_code == 200
    body = resp.json()
    assert body['available'] is HAVE_PYSCN
    assert 'install_hint' in body


def test_route_inspect_classifier(monkeypatch, tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    a = DataAdaptor('x.h5ad', adata=_adata())
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post('/api/pyscn/inspect_classifier', json={'path': str(p)})
    assert resp.status_code == 200, resp.text
    assert resp.json()['classifier']['n_gene_pairs'] > 0


def test_route_inspect_classifier_400s_on_a_bad_path(monkeypatch, tmp_path):
    a = DataAdaptor('x.h5ad', adata=_adata())
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post('/api/pyscn/inspect_classifier', json={'path': str(tmp_path / 'no.pkl')})
    assert resp.status_code == 400
    assert 'not found' in resp.json()['detail'].lower()


@pytest.mark.skipif(HAVE_PYSCN, reason='only meaningful when the package is absent')
def test_route_classify_400s_with_install_instructions_when_absent(monkeypatch, tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    a = DataAdaptor('x.h5ad', adata=_adata())
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post('/api/pyscn/classify', json={'path': str(p)})
    assert resp.status_code == 400
    assert 'pip install pySingleCellNet' in resp.json()['detail']


# --------------------------------------------------------------------------
# End-to-end, only when the package is actually installed
# --------------------------------------------------------------------------

@needs_pyscn
def test_classify_end_to_end(tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    a = DataAdaptor('x.h5ad', adata=_adata(n_cells=25))
    compute_fn, apply_fn = a.prepare_pyscn_classify(str(p), key='SCN')
    out = apply_fn(compute_fn(lambda frac, message=None: None))
    assert a.adata.obsm['SCN_score'].shape == (25, len(CLASSES))
    assert set(a.adata.obs['SCN_class_argmax'].unique()) <= set(CLASSES)
    assert out['n_cells'] == 25


@needs_pyscn
def test_chunked_classification_matches_single_pass(tmp_path):
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(_fake_clf(), fh)
    ad = _adata(n_cells=25)
    one = DataAdaptor('x.h5ad', adata=ad.copy())
    many = DataAdaptor('x.h5ad', adata=ad.copy())
    c1, a1 = one.prepare_pyscn_classify(str(p), key='SCN', chunk_size=10_000)
    c2, a2 = many.prepare_pyscn_classify(str(p), key='SCN', chunk_size=7)
    a1(c1(lambda f, message=None: None))
    a2(c2(lambda f, message=None: None))
    np.testing.assert_allclose(one.adata.obsm['SCN_score'], many.adata.obsm['SCN_score'])


@needs_pyscn
def test_train_then_classify_recovers_the_training_labels(tmp_path):
    """The full loop: train on labelled data, classify it back, get the labels.

    Perfect recall on the training set is not a generalization claim — it is
    the weakest check that the plumbing (balancing, preprocessing, pickling,
    reloading, gene alignment, score assembly) is wired up correctly. Any
    break in that chain shows up here as scrambled labels.
    """
    rng = np.random.default_rng(0)
    n_types, n_per, n_genes = 3, 40, 300
    blocks, labels = [], []
    for t in range(n_types):
        base = rng.gamma(1.0, 2.0, size=n_genes)
        base[t * 30:(t + 1) * 30] *= 12          # a marker block per type
        blocks.append(rng.poisson(base, size=(n_per, n_genes)))
        labels += [f'type{t}'] * n_per
    ad = anndata.AnnData(X=csr_matrix(np.vstack(blocks).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(ad.n_obs)]
    ad.obs['celltype'] = pd.Categorical(labels)

    a = DataAdaptor('x.h5ad', adata=ad)
    out_path = str(tmp_path / 'clf.pkl')
    train_c, train_a = a.prepare_pyscn_train(
        'celltype', out_path, n_cells_per_type=30, n_trees=100, n_comps=20,
    )
    trained = train_a(train_c(lambda f, message=None: None))
    assert trained['classifier']['cell_type_classes'] == ['type0', 'type1', 'type2']
    assert Path(out_path).exists()

    cls_c, cls_a = a.prepare_pyscn_classify(out_path, key='SCN')
    out = cls_a(cls_c(lambda f, message=None: None))
    predicted = a.adata.obs['SCN_class_argmax'].astype(str).to_numpy()
    assert (predicted == np.array(labels)).mean() > 0.9
    assert out['obsm_key'] == 'SCN_score'
    assert a.get_schema()['score_matrices']['SCN_score'] == out['classes']


@needs_pyscn
def test_training_drops_labels_with_too_few_cells(tmp_path):
    ad = _adata(genes=[f'g{i}' for i in range(120)], n_cells=60)
    labels = ['a'] * 29 + ['b'] * 29 + ['rare'] * 2
    ad.obs['celltype'] = pd.Categorical(labels)
    a = DataAdaptor('x.h5ad', adata=ad)
    c, ap = a.prepare_pyscn_train(
        'celltype', str(tmp_path / 'c.pkl'), n_cells_per_type=None,
        n_trees=50, n_comps=10,
    )
    res = ap(c(lambda f, message=None: None))
    assert res['dropped_labels'] == ['rare']
    assert res['classifier']['cell_type_classes'] == ['a', 'b']


@needs_pyscn
def test_training_refuses_a_column_with_one_label(tmp_path):
    ad = _adata(n_cells=20)
    ad.obs['only'] = pd.Categorical(['x'] * 20)
    a = DataAdaptor('x.h5ad', adata=ad)
    with pytest.raises(ValueError) as exc:
        a.prepare_pyscn_train('only', str(tmp_path / 'c.pkl'))
    assert 'at least 2 cell types' in str(exc.value)


# --------------------------------------------------------------------------
# Gene names containing '_'
#
# PySCN encodes each top-scoring pair as the string "geneA_geneB" and decodes
# it with split("_"). A symbol that itself contains an underscore therefore
# decodes into fragments that are not genes, and the transform fails deep
# inside pandas with an opaque KeyError. xcell has to see this coming.
# --------------------------------------------------------------------------

def test_underscore_gene_names_are_identified():
    bad = pyscn.underscore_gene_names(['Actb', 'Mesen_1', 'Cd3e', 'Ubiq_10'])
    assert bad == ['Mesen_1', 'Ubiq_10']


def test_underscore_names_are_absent_from_clean_symbols():
    assert pyscn.underscore_gene_names(['Actb', 'Cd3e', 'mt-Nd1']) == []


def test_inspect_flags_a_classifier_trained_on_underscore_genes(tmp_path):
    clf = _fake_clf(genes=['Mesen_1', 'Ubiq_2', 'Cd3e', 'Cd8a', 'Actb'])
    p = tmp_path / 'clf.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(clf, fh)
    with pytest.raises(ValueError) as exc:
        pyscn.load_classifier(str(p))
    msg = str(exc.value)
    assert 'underscore' in msg.lower()
    assert 'Mesen_1' in msg


@needs_pyscn
def test_training_drops_underscore_genes_and_says_so(tmp_path):
    genes = [f'g{i}' for i in range(150)] + [f'mod_{i}' for i in range(30)]
    ad = _adata(genes=genes, n_cells=60)
    ad.obs['celltype'] = pd.Categorical(['a'] * 30 + ['b'] * 30)
    a = DataAdaptor('x.h5ad', adata=ad)
    c, ap = a.prepare_pyscn_train(
        'celltype', str(tmp_path / 'c.pkl'), n_cells_per_type=None,
        n_trees=50, n_comps=10,
    )
    res = ap(c(lambda f, message=None: None))
    assert res['n_genes_dropped_underscore'] == 30
    assert all('_' not in g for g in res['classifier']['genes'])


def test_training_refuses_when_every_gene_has_an_underscore(tmp_path):
    """The toy spatial dataset is exactly this case — genes named Mesen_1 etc."""
    ad = _adata(genes=[f'mod_{i}' for i in range(40)], n_cells=40)
    ad.obs['celltype'] = pd.Categorical(['a'] * 20 + ['b'] * 20)
    a = DataAdaptor('x.h5ad', adata=ad)
    with pytest.raises(ValueError) as exc:
        a.prepare_pyscn_train('celltype', str(tmp_path / 'c.pkl'))
    msg = str(exc.value)
    assert 'underscore' in msg.lower()
    # Point at the fix rather than just the failure.
    assert 'rename' in msg.lower()


# --------------------------------------------------------------------------
# Training on data that is already normalized / logged
#
# Plenty of reference datasets are distributed only as log-normalized values.
# Running normalize_total + log1p over those again produces log1p(scale(log1p(x))),
# which distorts the fold changes that drive marker-gene selection and makes
# seurat_v3 HVG (which requires integer counts) meaningless. xcell decides what
# preprocessing is still owed from the detected scale of the source matrix.
# --------------------------------------------------------------------------

def test_training_plan_for_raw_counts_does_everything():
    p = pyscn.training_plan('raw_counts')
    assert (p['normalize'], p['log1p']) == (True, True)
    # seurat_v3 is the right flavor only when real counts are available.
    assert p['hvg_flavor'] == 'seurat_v3'
    assert p['snapshot_counts'] is True


def test_training_plan_for_linear_normalized_logs_but_does_not_renormalize():
    p = pyscn.training_plan('normalized_linear')
    assert (p['normalize'], p['log1p']) == (False, True)
    assert p['hvg_flavor'] == 'seurat'
    assert p['snapshot_counts'] is False


@pytest.mark.parametrize('verdict', ['log_normalized', 'log_transformed'])
def test_training_plan_leaves_log_scale_data_alone(verdict):
    p = pyscn.training_plan(verdict)
    assert (p['normalize'], p['log1p']) == (False, False)
    assert p['hvg_flavor'] == 'seurat'
    assert p['snapshot_counts'] is False
    assert 'already' in p['reason'].lower()


def test_training_plan_is_conservative_when_the_scale_is_unrecognized():
    """Unknown means unknown: don't transform, but say the guess was made."""
    p = pyscn.training_plan('unknown')
    assert (p['normalize'], p['log1p']) == (False, False)
    assert p['uncertain'] is True


@pytest.mark.parametrize('verdict,explains', [
    # Each refusal must say what is wrong with *that* scale specifically —
    # a generic "cannot train" leaves the user with no next step.
    ('z_scored', 'top-scoring-pair'),
    ('binary', 'rank two genes'),
    ('empty', 'no non-zero values'),
])
def test_training_plan_refuses_scales_that_cannot_be_trained_on(verdict, explains):
    with pytest.raises(ValueError) as exc:
        pyscn.training_plan(verdict)
    assert explains in str(exc.value)


def test_training_plan_accepts_an_explicit_override():
    """Detection is a heuristic; the user gets the last word."""
    p = pyscn.training_plan('raw_counts', override='log_normalized')
    assert (p['normalize'], p['log1p']) == (False, False)
    assert p['source_scale'] == 'log_normalized'
    assert p['overridden'] is True


def _lognorm_adata(n_genes=200, n_cells=120, seed=0):
    """A reference distributed the way many public ones are: log-normalized only."""
    rng = np.random.default_rng(seed)
    counts = np.vstack([
        rng.poisson(np.where(np.arange(n_genes) // 50 == t, 30, 2), size=(n_cells // 2, n_genes))
        for t in range(2)
    ]).astype(np.float32)
    sums = counts.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    x = np.log1p(counts / sums * 1e4).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(x))
    ad.var_names = [f'Gene{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(ad.n_obs)]
    ad.obs['celltype'] = pd.Categorical(['a'] * (n_cells // 2) + ['b'] * (n_cells // 2))
    return ad


def test_lognormalized_input_is_detected_as_such():
    """Guard the assumption the training plan rests on."""
    from xcell import layer_scale
    ad = _lognorm_adata()
    assert layer_scale.assess_matrix_scale(ad.X)['verdict'] == 'log_normalized'


@needs_pyscn
def test_training_on_lognormalized_data_does_not_transform_again(tmp_path):
    a = DataAdaptor('x.h5ad', adata=_lognorm_adata())
    c, ap = a.prepare_pyscn_train(
        'celltype', str(tmp_path / 'c.pkl'), n_cells_per_type=None,
        n_trees=50, n_comps=10,
    )
    res = ap(c(lambda f, message=None: None))
    prep = res['preprocessing']
    assert prep['source_scale'] == 'log_normalized'
    assert (prep['normalize'], prep['log1p']) == (False, False)
    # A 'counts' layer built from log values would be a lie that seurat_v3
    # would then silently consume.
    assert prep['snapshot_counts'] is False
    assert prep['hvg_flavor'] == 'seurat'


@needs_pyscn
def test_training_on_raw_counts_still_normalizes(tmp_path):
    ad = _adata(genes=[f'Gene{i}' for i in range(200)], n_cells=120)
    ad.obs['celltype'] = pd.Categorical(['a'] * 60 + ['b'] * 60)
    a = DataAdaptor('x.h5ad', adata=ad)
    c, ap = a.prepare_pyscn_train(
        'celltype', str(tmp_path / 'c.pkl'), n_cells_per_type=None,
        n_trees=50, n_comps=10,
    )
    res = ap(c(lambda f, message=None: None))
    prep = res['preprocessing']
    assert prep['source_scale'] == 'raw_counts'
    assert (prep['normalize'], prep['log1p']) == (True, True)


@needs_pyscn
def test_a_classifier_trained_on_lognormalized_data_works(tmp_path):
    """The point of not double-transforming: the classifier must still classify."""
    ad = _lognorm_adata()
    a = DataAdaptor('x.h5ad', adata=ad)
    out_path = str(tmp_path / 'c.pkl')
    c, ap = a.prepare_pyscn_train(
        'celltype', out_path, n_cells_per_type=None, n_trees=100, n_comps=10,
    )
    ap(c(lambda f, message=None: None))
    cc, ca = a.prepare_pyscn_classify(out_path, key='SCN')
    ca(cc(lambda f, message=None: None))
    predicted = a.adata.obs['SCN_class_argmax'].astype(str).to_numpy()
    truth = ad.obs['celltype'].astype(str).to_numpy()
    assert (predicted == truth).mean() > 0.9


@needs_pyscn
def test_explicit_scale_override_reaches_training(tmp_path):
    a = DataAdaptor('x.h5ad', adata=_lognorm_adata())
    c, ap = a.prepare_pyscn_train(
        'celltype', str(tmp_path / 'c.pkl'), n_cells_per_type=None,
        n_trees=50, n_comps=10, source_scale='normalized_linear',
    )
    res = ap(c(lambda f, message=None: None))
    assert res['preprocessing']['overridden'] is True
    assert res['preprocessing']['log1p'] is True


def test_training_refuses_a_z_scored_source(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (60, 200)).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(x))
    ad.var_names = [f'Gene{i}' for i in range(200)]
    ad.obs['celltype'] = pd.Categorical(['a'] * 30 + ['b'] * 30)
    a = DataAdaptor('x.h5ad', adata=ad)
    with pytest.raises(ValueError) as exc:
        a.prepare_pyscn_train('celltype', str(tmp_path / 'c.pkl'))
    assert 'scaled' in str(exc.value).lower() or 'z-scored' in str(exc.value).lower()


@needs_pyscn
def test_pair_features_are_invariant_to_per_cell_monotone_transforms():
    """Pins down what double-transforming does and does not break.

    normalize_total scales each cell by a positive constant and log1p is
    monotone, so neither changes the ordering of two genes *within* a cell —
    which is the only thing the top-scoring-pair transform reads. The features
    are therefore identical, and the harm of re-normalizing an already
    log-normalized reference is confined to gene selection (rank_genes_groups,
    PCA/dendrogram, HVG), not to how a cell is scored.

    If this ever fails, the reasoning behind `training_plan` needs revisiting.
    """
    from pySingleCellNet.tools.classifier import _query_transform

    rng = np.random.default_rng(0)
    genes = [f'G{i}' for i in range(30)]
    counts = pd.DataFrame(rng.poisson(8, (50, 30)).astype(float), columns=genes)
    pairs = np.array([f'G{a}_G{b}' for a in range(10) for b in range(10, 14)])

    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e4
    lognorm = np.log1p(cpm)
    double_logged = np.log1p(lognorm.div(lognorm.sum(axis=1), axis=0) * 1e4)

    base = _query_transform(counts, pairs)
    assert base.equals(_query_transform(lognorm, pairs))
    assert base.equals(_query_transform(double_logged, pairs))
