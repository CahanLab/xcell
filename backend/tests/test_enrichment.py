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
