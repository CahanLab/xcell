"""Pure tests for enrichment figure assembly: matrix, collapse, network, layout."""
import numpy as np
import pytest

from xcell import enrichment_figures as ef


def _row(name, nes, padj, library='L'):
    return {'name': name, 'library': library, 'description': '', 'url': '', 'n_set': 10,
            'es': nes / 2, 'nes': nes, 'pval': padj / 2, 'padj': padj, 'leading_edge': [],
            'n_leading_edge': 0, 'curve': None}


def _gsea(rows):
    return {'kind': 'gsea', 'results': rows, 'n_perm': 100}


def _ora_row(name, k, K, padj):
    return {'name': name, 'library': 'L', 'description': '', 'url': '', 'n_set': K, 'n_overlap': k,
            'expected': 1.0, 'fold_enrichment': k / 1.0, 'odds_ratio': 2.0, 'pval': padj / 2, 'padj': padj,
            'genes': [], 'below_min_overlap': False}


def test_signed_logp_is_finite_and_signed():
    assert ef.signed_logp(0.0, 1.0) == 300.0
    assert ef.signed_logp(0.0, -1.0) == -300.0
    assert ef.signed_logp(0.01, -2.5) == pytest.approx(-2.0)
    assert ef.signed_logp(1.0, 1.0) == 0.0
    assert ef.row_value(_ora_row('x', 5, 10, 0.5), 'signed_logp') > 0      # fold enrichment 5 > 1 → positive
    assert ef.row_value(_ora_row('x', 0, 10, 0.5), 'signed_logp') <= 0


def test_assemble_matrix_thresholds_top_n_union_and_orders():
    a = _gsea([_row('S1', 2.5, 0.001), _row('S2', 2.0, 0.01), _row('S3', 1.5, 0.2), _row('S4', -2.2, 0.001)])
    b = _gsea([_row('S1', -1.0, 0.04), _row('S2', 0.5, 0.9), _row('S5', 2.8, 0.001), _row('S4', -1.9, 0.02)])
    m = ef.assemble_matrix({'a': a, 'b': b}, value='nes', padj_max=0.05, top_n=1, direction='both',
                           collapse_jaccard=None, row_order='peak', col_order='input', members=None)
    names = [r['name'] for r in m['rows']]
    # top 1 up and top 1 down per column: a→S1,S4; b→S5,S4 → union S1,S4,S5
    assert set(names) == {'S1', 'S4', 'S5'} and m['n_rows_total'] == 5
    assert [c['label'] for c in m['cols']] == ['a', 'b']
    vals = {r['name']: v for r, v in zip(m['rows'], m['values'])}
    assert vals['S1'] == [2.5, -1.0] and vals['S5'] == [0.0, 2.8]           # b:S5 kept; a lacks S5 → 0
    assert vals['S4'] == [-2.2, -1.9]
    # 'peak' order: rows grouped by the column of their largest |value| (a first), positive before negative
    assert names == ['S1', 'S4', 'S5']
    up = ef.assemble_matrix({'a': a, 'b': b}, value='nes', padj_max=0.05, top_n=5, direction='up',
                            collapse_jaccard=None, row_order='input', col_order='input', members=None)
    assert all(max(v) > 0 for v in up['values']) and 'S4' not in [r['name'] for r in up['rows']]
    assert m['value_label'] == 'NES'
    cl = ef.assemble_matrix({'a': a, 'b': b}, value='signed_logp', padj_max=0.05, top_n=5, direction='both',
                            collapse_jaccard=None, row_order='cluster', col_order='cluster', members=None)
    assert cl['value_label'].startswith('signed') and len(cl['rows']) == 4
    assert np.isfinite(np.asarray(cl['values'])).all()


def test_assemble_matrix_empty_gives_note_not_error():
    a = _gsea([_row('S1', 2.5, 0.5)])
    m = ef.assemble_matrix({'a': a}, value='nes', padj_max=0.05, top_n=5, direction='both',
                           collapse_jaccard=None, row_order='peak', col_order='input', members=None)
    assert m['rows'] == [] and m['values'] == [] and m['note']


def test_collapse_sets_by_jaccard_keeps_representative_and_lists_members():
    members = {'A': set('abcdefgh'), 'B': set('abcdefgX'), 'C': set('xyz'), 'D': set('abcd')}
    # A–B Jaccard = 7/9 = 0.78; A–D = 4/8 = 0.5 (exactly at threshold 0.5 → merged when >=)
    groups = ef.collapse_sets(['A', 'B', 'C', 'D'], members, jaccard=0.5)
    assert groups == {'A': ['B', 'D'], 'C': []}
    groups2 = ef.collapse_sets(['A', 'B', 'C', 'D'], members, jaccard=0.51)
    assert groups2 == {'A': ['B'], 'C': [], 'D': []}
    a = _gsea([_row('A', 2.5, 0.001), _row('B', 2.4, 0.001), _row('C', 2.0, 0.001), _row('D', 1.0, 0.001)])
    m = ef.assemble_matrix({'a': a}, value='nes', padj_max=0.05, top_n=10, direction='both',
                           collapse_jaccard=0.5, row_order='input', col_order='input', members=members)
    rows = {r['name']: r for r in m['rows']}
    assert set(rows) == {'A', 'C'} and rows['A']['members'] == ['B', 'D'] and m['n_collapsed'] == 2
    assert m['values'][[r['name'] for r in m['rows']].index('A')] == [2.5]   # representative keeps its own value


def test_network_nodes_edges_overlap_and_layouts():
    members = {'S1': set('abcdef'), 'S2': set('abcdeg'), 'S3': set('xyz')}
    a = _gsea([_row('S1', 2.5, 0.001), _row('S3', -2.0, 0.01)])
    b = _gsea([_row('S2', 2.2, 0.001)])
    net = ef.assemble_network({'a': a, 'b': b}, value='nes', padj_max=0.05, top_n=5, direction='both',
                              set_edge_jaccard=0.5, layout='force', seed=1, members=members)
    kinds = {n['id']: n['kind'] for n in net['nodes']}
    assert kinds == {'group:a': 'group', 'group:b': 'group', 'set:S1': 'set', 'set:S3': 'set', 'set:S2': 'set'}
    enr = [e for e in net['edges'] if e['kind'] == 'enrichment']
    ov = [e for e in net['edges'] if e['kind'] == 'overlap']
    assert {(e['source'], e['target']) for e in enr} == {('group:a', 'set:S1'), ('group:a', 'set:S3'), ('group:b', 'set:S2')}
    assert ov == [{'source': 'set:S1', 'target': 'set:S2', 'kind': 'overlap', 'value': pytest.approx(5 / 7), 'padj': None}]
    xy = np.array([[n['x'], n['y']] for n in net['nodes']])
    assert np.isfinite(xy).all() and len(net['bounds']) == 4
    net2 = ef.assemble_network({'a': a, 'b': b}, value='nes', padj_max=0.05, top_n=5, direction='both',
                               set_edge_jaccard=0.5, layout='force', seed=1, members=members)
    assert [n['x'] for n in net2['nodes']] == [n['x'] for n in net['nodes']]       # deterministic
    bip = ef.assemble_network({'a': a, 'b': b}, value='nes', padj_max=0.05, top_n=5, direction='both',
                              set_edge_jaccard=None, layout='bipartite', seed=0, members=members)
    gx = {n['x'] for n in bip['nodes'] if n['kind'] == 'group'}
    sx = {n['x'] for n in bip['nodes'] if n['kind'] == 'set'}
    assert len(gx) == 1 and len(sx) == 1 and gx != sx
    assert [n['size'] for n in net['nodes'] if n['kind'] == 'set'] and all(n['degree'] >= 1 for n in net['nodes'])


def test_force_layout_handles_isolated_nodes():
    xy = ef.force_layout(4, [(0, 1)], weights=[1.0], seed=0, iterations=50)
    assert xy.shape == (4, 2) and np.isfinite(xy).all()
