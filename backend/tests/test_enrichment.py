"""Pure tests for xcell.enrichment: BH, symbol resolution, overlap test, GSEA."""
import numpy as np
import pytest
from scipy.stats import fisher_exact, hypergeom

from xcell import enrichment as en


def test_bh_adjust_matches_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = en.bh_adjust(p)
    # sorted: 0.01,0.03,0.04,0.20 -> 0.04, 0.06, 0.0533 -> monotone 0.0533, 0.20
    assert adj.tolist() == pytest.approx([0.04, 0.05333333, 0.05333333, 0.20], rel=1e-6)
    assert en.bh_adjust(np.array([])).size == 0
    nan_adj = en.bh_adjust(np.array([0.5, np.nan]))
    assert np.isnan(nan_adj[1]) and nan_adj[0] == 0.5


def test_resolve_symbols_exact_then_case_insensitive_dedup():
    universe = ['Col1a1', 'Sox9', 'ACTB', 'Actb']
    idx, missing = en.resolve_symbols(['COL1A1', 'Sox9', 'sox9', 'Actb', 'Nope'], universe)
    assert idx == [0, 1, 3]            # exact 'Actb' wins over ci 'ACTB'
    assert missing == ['Nope']


def test_resolve_sets_union_and_split_and_size_filter():
    universe = [f'G{i}' for i in range(10)]
    sets = [
        {'name': 'A', 'genes': ['g0', 'G1', 'G2'], 'genes_down': ['G3'], 'library': 'L'},
        {'name': 'tiny', 'genes': ['G0']},
        {'name': 'huge', 'genes': universe},
    ]
    union, meta = en.resolve_sets(sets, universe, min_size=2, max_size=5)
    assert [s['name'] for s in union] == ['A']
    assert union[0]['indices'].tolist() == [0, 1, 2, 3] and union[0]['library'] == 'L'
    assert meta == {'n_input': 3, 'n_dropped_size': 2}
    split, _ = en.resolve_sets(sets, universe, min_size=1, max_size=5, directional='split')
    assert [s['name'] for s in split] == ['A', 'A (down)', 'tiny']
    assert split[1]['indices'].tolist() == [3]


def test_overlap_enrichment_matches_scipy_and_is_finite():
    universe_n = 100
    query = list(range(10))
    sets = [
        {'name': 'hit', 'library': 'L', 'description': '', 'url': '', 'n_input': 8,
         'indices': np.array([0, 1, 2, 3, 50, 51, 52, 53])},
        {'name': 'miss', 'library': 'L', 'description': '', 'url': '', 'n_input': 5,
         'indices': np.array([60, 61, 62, 63, 64])},
    ]
    res = en.overlap_enrichment(query, sets, universe_n, min_overlap=2)
    hit, miss = res[0], res[1]
    assert hit['name'] == 'hit' and hit['n_overlap'] == 4 and hit['n_set'] == 8
    assert hit['pval'] == pytest.approx(hypergeom.sf(3, universe_n, 8, 10))
    table = [[4, 6], [4, 86]]
    assert hit['pval'] == pytest.approx(fisher_exact(table, alternative='greater')[1])
    assert hit['genes'] == [0, 1, 2, 3]
    assert hit['expected'] == pytest.approx(10 * 8 / 100)
    assert hit['fold_enrichment'] == pytest.approx(4 / 0.8)
    assert miss['n_overlap'] == 0 and miss['below_min_overlap'] is True
    assert np.isfinite(miss['odds_ratio']) and miss['pval'] == 1.0
    assert all(np.isfinite(r['padj']) for r in res)
    assert hit['padj'] <= miss['padj']


# --- preranked GSEA ------------------------------------------------------------

def _naive_es(positions, weights, n_ranked):
    """Running-sum loop straight from Subramanian 2005, for cross-checking."""
    hits = dict(zip(positions, weights))
    wsum = sum(weights)
    n_miss = n_ranked - len(positions)
    run, best = 0.0, 0.0
    for i in range(n_ranked):
        run += hits[i] / wsum if i in hits else -1.0 / n_miss
        if abs(run) > abs(best):
            best = run
    return best


def _rset(name, indices):
    return {'name': name, 'library': 'L', 'description': '', 'url': '', 'n_input': len(indices),
            'indices': np.asarray(indices)}


def test_es_vectorised_equals_naive_loop():
    rng = np.random.default_rng(1)
    n = 40
    scores = np.sort(rng.normal(size=n))[::-1]
    for _ in range(20):
        pos = np.sort(rng.choice(n, size=6, replace=False))
        w = np.abs(scores[pos])
        es, _, _, _ = en._es_from_positions(pos[None, :], w[None, :], n)
        assert es[0] == pytest.approx(_naive_es(pos.tolist(), w.tolist(), n))


def test_gsea_planted_top_bottom_and_random():
    rng = np.random.default_rng(0)
    n = 2000
    scores = rng.normal(size=n)
    order = np.argsort(-scores)
    top = order[:30]
    bottom = order[-30:]
    rand = rng.choice(n, size=30, replace=False)
    sets = [_rset('top', top), _rset('bottom', bottom), _rset('rand', rand)]
    out = en.preranked_gsea(scores, sets, n_perm=200, min_size=10, max_size=500, seed=0)
    by = {r['name']: r for r in out['results']}
    assert by['top']['es'] > 0.9 and by['top']['nes'] > 1
    # The null is split by sign (Subramanian 2005), so the floor is 1/(1 + n_same_sign),
    # about 2/n_perm — not 1/(n_perm + 1).
    assert 1 / 201 <= by['top']['pval'] <= 1 / 60
    assert set(by['top']['leading_edge']) == set(top.tolist())
    assert by['bottom']['es'] < -0.9 and set(by['bottom']['leading_edge']) == set(bottom.tolist())
    assert by['rand']['pval'] > 0.05 and abs(by['rand']['nes']) < 1.6
    assert out['n_ranked'] == n and out['n_sets_tested'] == 3
    assert out['results'][0]['name'] in ('top', 'bottom')
    assert all(np.isfinite([r['es'], r['nes'], r['pval'], r['padj']]).all() for r in out['results'])


def test_gsea_is_deterministic_and_curve_only_for_top_n():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=500)
    sets = [_rset(f's{i}', rng.choice(500, size=20, replace=False)) for i in range(5)]
    a = en.preranked_gsea(scores, sets, n_perm=50, min_size=5, seed=3, curve_top_n=2)
    b = en.preranked_gsea(scores, sets, n_perm=50, min_size=5, seed=3, curve_top_n=2)
    assert [r['pval'] for r in a['results']] == [r['pval'] for r in b['results']]
    curves = [r['curve'] is not None for r in a['results']]
    assert curves == [True, True, False, False, False]
    c = a['results'][0]['curve']
    assert c[0] == [0, 0.0] and c[-1] == [499, 0.0] and len(c) == 2 + 2 * 20


def test_gsea_edge_cases_nan_scores_zero_weights_full_set_and_weight_zero():
    scores = np.array([3.0, 2.0, 1.0, 0.0, 0.0, np.nan, -1.0, -2.0])
    # 'zero' has only zero-score members; 'all' is every finite gene.
    sets = [_rset('zero', [3, 4]), _rset('all', [0, 1, 2, 3, 4, 6, 7]), _rset('nan_only', [5, 0])]
    out = en.preranked_gsea(scores, sets, n_perm=20, min_size=1, max_size=10, seed=0)
    by = {r['name']: r for r in out['results']}
    assert out['n_ranked'] == 7
    assert by['zero']['es'] == 0.0 and by['zero']['nes'] == 0.0 and by['zero']['pval'] == 1.0
    assert by['all']['es'] == pytest.approx(1.0)
    assert by['nan_only']['n_set'] == 1          # the NaN gene dropped out
    # weight 0 -> classic KS: planted top set of 2 in 7 genes
    ks = en.preranked_gsea(scores, [_rset('top2', [0, 1])], n_perm=10, min_size=1, weight=0.0, seed=0)
    assert ks['results'][0]['es'] == pytest.approx(1.0)


def test_gsea_report_called():
    calls = []
    rng = np.random.default_rng(0)
    scores = rng.normal(size=100)
    sets = [_rset('a', rng.choice(100, 10, replace=False)), _rset('b', rng.choice(100, 12, replace=False))]
    en.preranked_gsea(scores, sets, n_perm=10, min_size=5, report=lambda f, m: calls.append((f, m)))
    assert calls and 0.0 <= calls[0][0] <= 1.0 and calls[-1][0] == pytest.approx(1.0)


def test_null_positions_are_sorted_k_subsets_of_bounded_width():
    rng = np.random.default_rng(0)
    pos = en._null_positions(rng, n=5000, max_k=40, n_perm=250, chunk=64)
    assert pos.shape == (250, 40) and pos.dtype == np.int32
    assert (np.diff(np.sort(pos, axis=1), axis=1) > 0).all()   # no repeats within a row
    assert pos.min() >= 0 and pos.max() < 5000
    # rows differ from each other (a real permutation pool, not one row repeated)
    assert len({tuple(r) for r in pos[:50]}) == 50


def test_gsea_es_zero_has_empty_leading_edge_and_docstring_floor():
    scores = np.array([3.0, 2.0, 1.0, 0.0, 0.0, -1.0, -2.0])
    out = en.preranked_gsea(scores, [_rset('zero', [3, 4])], n_perm=20, min_size=1, max_size=10, seed=0)
    assert out['results'][0]['leading_edge'] == [] and out['results'][0]['n_leading_edge'] == 0
    assert '2/n_perm' in en.preranked_gsea.__doc__


def test_gsea_null_is_calibrated_across_mixed_set_sizes():
    """Random sets of several sizes must give ~uniform p (the null must be a
    uniform k-subset for every k, not a biased slice of one pool)."""
    rng = np.random.default_rng(5)
    n = 3000
    scores = rng.normal(size=n)
    sets = [_rset(f'r{i}', rng.choice(n, size=int(k), replace=False))
            for i, k in enumerate(rng.integers(15, 200, size=150))]
    out = en.preranked_gsea(scores, sets, n_perm=300, min_size=10, max_size=500, seed=1)
    p = np.array([r['pval'] for r in out['results']])
    frac05 = float((p < 0.05).mean())
    assert 0.0 <= frac05 <= 0.10, frac05         # sign-split null runs slightly conservative
    assert np.median(p) > 0.25
