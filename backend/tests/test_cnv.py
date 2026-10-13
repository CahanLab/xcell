"""The pure CNV module: positions, genome order, windows, reference subtraction,
smoothing, clone calling, heatmap assembly.

The synthetic tumor: four groups of cells over six chromosomes of 150 genes.
``normal`` has no events; ``A`` gains chr2 (counts ×2); ``B`` loses chr4
(×0.5); ``C`` gains chr2 and loses chr6. A correct implementation recovers
the three clones from the window matrix alone.
"""
import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from xcell import cnv

N_CHROM, GENES_PER_CHROM = 6, 150
N_GENES = N_CHROM * GENES_PER_CHROM
GROUPS = {'normal': 120, 'A': 80, 'B': 80, 'C': 80}
EVENTS = {'A': {'chr2': 2.0}, 'B': {'chr4': 0.5}, 'C': {'chr2': 2.0, 'chr6': 0.5}}


def _genes():
    names, chroms, starts = [], [], []
    for c in range(1, N_CHROM + 1):
        for g in range(GENES_PER_CHROM):
            names.append(f'g{c}_{g:03d}')
            chroms.append(f'chr{c}')
            starts.append(1000 * (g + 1))
    return names, np.array(chroms), np.array(starts)


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    names, chroms, starts = _genes()
    lam = rng.lognormal(mean=0.3, sigma=0.8, size=N_GENES)
    labels = np.concatenate([[g] * n for g, n in GROUPS.items()])
    X = np.empty((labels.size, N_GENES), dtype=np.float32)
    for i, lab in enumerate(labels):
        scale = np.ones(N_GENES)
        for chrom, f in EVENTS.get(lab, {}).items():
            scale[chroms == chrom] = f
        # Per-cell depth varies 3x so the per-cell median centering has work to do.
        depth = rng.uniform(0.6, 1.8)
        X[i] = rng.poisson(lam * scale * depth)
    return X, names, chroms, starts, labels


def _lognorm(X):
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)


def _setup(window=30, step=5, include=None):
    X, names, chroms, starts, labels = _counts()
    pos = cnv.positions_from_columns(names, chroms, starts)
    order = cnv.gene_order(pos, names, include or [f'chr{i}' for i in range(1, N_CHROM + 1)])
    gene_cols = np.concatenate([o['indices'] for o in order])
    windows, chromosomes = cnv.plan_windows(order, window=window, step=step)
    L = _lognorm(X)
    return X, L, names, labels, gene_cols, windows, chromosomes


# ---------- positions ----------

def test_positions_from_columns_skips_missing_and_casts_start():
    pos = cnv.positions_from_columns(['a', 'b', 'c'], ['chr1', None, 'chr2'], ['100', np.nan, 5.0])
    assert pos == {'a': ('chr1', 100, 100), 'c': ('chr2', 5, 5)}


def test_match_positions_by_id_version_symbol_and_case():
    table = {'ENSG1': ('chr1', 10, 20), 'TP53': ('chr17', 7, 8), 'ACTB': ('chr7', 1, 2)}
    pos, how = cnv.match_positions(['ENSG1.14', 'TP53', 'Actb', 'nothing'], table)
    assert pos == {'ENSG1.14': ('chr1', 10, 20), 'TP53': ('chr17', 7, 8), 'Actb': ('chr7', 1, 2)}
    assert how == {'ensembl': 1, 'symbol': 1, 'symbol_upper': 1, 'unmatched': 1}


def test_gene_order_is_karyotype_order_and_excludes_unlisted():
    names = ['x', 'y', 'z', 'w', 'm']
    pos = {'x': ('chr10', 5, 5), 'y': ('chr2', 50, 50), 'z': ('chr2', 10, 10), 'w': ('chrX', 1, 1), 'm': ('chrM', 1, 1)}
    order = cnv.gene_order(pos, names, cnv.chromosome_list('human', include_xy=False))
    assert [o['chromosome'] for o in order] == ['chr2', 'chr10']
    assert list(order[0]['indices']) == [2, 1]           # z before y by start
    order_xy = cnv.gene_order(pos, names, cnv.chromosome_list('human', include_xy=True))
    assert [o['chromosome'] for o in order_xy] == ['chr2', 'chr10', 'chrX']


def test_chromosome_list_accepts_bare_numbers_in_positions():
    pos = cnv.positions_from_columns(['a', 'b'], ['1', 'X'], [1, 2])
    assert pos['a'][0] == 'chr1' and pos['b'][0] == 'chrX'


# ---------- windows ----------

def test_plan_windows_counts_and_short_chromosome():
    order = [{'chromosome': 'chr1', 'indices': np.arange(0, 100), 'starts': np.arange(100)},
             {'chromosome': 'chr2', 'indices': np.arange(100, 112), 'starts': np.arange(12)}]
    windows, chromosomes = cnv.plan_windows(order, window=30, step=10)
    n1 = (100 - 30) // 10 + 1
    assert [c['name'] for c in chromosomes] == ['chr1', 'chr2']
    assert chromosomes[0] == {'name': 'chr1', 'start': 0, 'size': n1}
    assert chromosomes[1] == {'name': 'chr2', 'start': n1, 'size': 1}   # fewer genes than the window: one window
    assert windows[-1]['n_genes'] == 12
    # Window gene indices address columns of the filtered matrix, in order.
    assert list(windows[0]['gene_idx']) == list(range(30))
    assert list(windows[1]['gene_idx']) == list(range(10, 40))


def test_plan_windows_covers_the_chromosome_tail():
    # 95 genes, window 30, step 10: the regular windows end at gene 90, so a
    # final window over genes 65..95 keeps the last five in the picture.
    order = [{'chromosome': 'chr1', 'indices': np.arange(95), 'starts': np.arange(95)}]
    windows, chromosomes = cnv.plan_windows(order, window=30, step=10)
    assert chromosomes[0]['size'] == (95 - 30) // 10 + 2
    assert list(windows[-1]['gene_idx']) == list(range(65, 95))
    assert windows[-1]['end'] == 94


def test_plan_windows_window_must_exceed_step():
    order = [{'chromosome': 'chr1', 'indices': np.arange(50), 'starts': np.arange(50)}]
    with pytest.raises(ValueError):
        cnv.plan_windows(order, window=5, step=10)


# ---------- reference ----------

def test_reference_subtract_bounds_rule():
    X = np.array([[1.0, 5.0, 2.0]])
    refs = np.array([[0.0, 4.0, 1.0], [2.0, 6.0, 3.0]])   # two reference groups
    out = cnv.reference_subtract(X, refs)
    # 1 within [0,2] → 0; 5 within [4,6] → 0; 2 within [1,3] → 0
    assert np.allclose(out, 0)
    out = cnv.reference_subtract(np.array([[3.0, 3.0, 0.0]]), refs)
    assert np.allclose(out, [[1.0, -1.0, -1.0]])         # above the top / below the bottom bound
    one = cnv.reference_subtract(np.array([[3.0, 3.0, 0.0]]), refs[:1])
    assert np.allclose(one, [[3.0, -1.0, -1.0]])


# ---------- inference ----------

def test_infer_chunked_equals_whole_and_centering():
    X, L, names, labels, gene_cols, windows, chromosomes = _setup()
    ref = labels == 'normal'
    ref_means = L[ref][:, gene_cols].mean(axis=0, keepdims=True)
    get_rows = lambda idx: L[idx][:, gene_cols]
    whole = cnv.infer(get_rows, np.arange(L.shape[0]), windows, ref_means, lfc_clip=3.0, center_cells=True, chunk=10_000)
    parts = cnv.infer(get_rows, np.arange(L.shape[0]), windows, ref_means, lfc_clip=3.0, center_cells=True, chunk=37)
    assert whole.dtype == np.float32 and whole.shape == (L.shape[0], len(windows))
    assert np.allclose(whole, parts, atol=1e-6)
    assert np.allclose(np.median(whole, axis=1), 0, atol=1e-6)


def test_infer_reports_progress():
    X, L, names, labels, gene_cols, windows, chromosomes = _setup()
    ref_means = L[:, gene_cols].mean(axis=0, keepdims=True)
    seen = []
    cnv.infer(lambda idx: L[idx][:, gene_cols], np.arange(L.shape[0]), windows, ref_means, lfc_clip=3.0,
              center_cells=False, chunk=100, report=lambda f, m: seen.append(f))
    assert seen and seen == sorted(seen) and 0 < seen[0] <= seen[-1] <= 1


def test_planted_events_show_in_windows():
    X, L, names, labels, gene_cols, windows, chromosomes = _setup()
    ref = labels == 'normal'
    ref_means = L[ref][:, gene_cols].mean(axis=0, keepdims=True)
    M = cnv.infer(lambda idx: L[idx][:, gene_cols], np.arange(L.shape[0]), windows, ref_means,
                  lfc_clip=3.0, center_cells=True)
    chrom_of = np.array([w['chromosome'] for w in windows])
    chr2, chr4, chr6 = (chrom_of == 'chr2'), (chrom_of == 'chr4'), (chrom_of == 'chr6')
    a = labels == 'A'; b = labels == 'B'; c = labels == 'C'; n = labels == 'normal'
    assert M[a][:, chr2].mean() > 0.2 and M[c][:, chr2].mean() > 0.2
    assert M[b][:, chr4].mean() < -0.2 and M[c][:, chr6].mean() < -0.2
    assert abs(M[n].mean()) < 0.05
    # No event there: well below the planted gain, if not zero at these low counts.
    assert np.abs(M[a][:, chr4]).mean() < 0.5 * M[a][:, chr2].mean()
    score = cnv.cnv_score(M)
    assert score[n].mean() < score[a].mean() < score[c].mean()


def test_noise_threshold_zeroes_small_values():
    M = np.array([[0.1, -0.1, 2.0], [0.05, 0.0, -2.0]], dtype=np.float32)
    out = cnv.apply_noise_threshold(M, 1.5)
    sd = M.std()
    expected = np.where(np.abs(M) < 1.5 * sd, 0, M)
    assert np.array_equal(out, expected)
    assert np.array_equal(cnv.apply_noise_threshold(M, 0), M)


def test_chromosome_means():
    chromosomes = [{'name': 'chr1', 'start': 0, 'size': 2}, {'name': 'chr2', 'start': 2, 'size': 1}]
    M = np.array([[1.0, 3.0, 5.0], [np.nan, np.nan, np.nan]])
    out = cnv.chromosome_means(M, chromosomes)
    assert np.allclose(out[0], [2.0, 5.0]) and np.isnan(out[1]).all()


# ---------- clones ----------

def _matrix():
    X, L, names, labels, gene_cols, windows, chromosomes = _setup()
    ref_means = L[labels == 'normal'][:, gene_cols].mean(axis=0, keepdims=True)
    M = cnv.infer(lambda idx: L[idx][:, gene_cols], np.arange(L.shape[0]), windows, ref_means,
                  lfc_clip=3.0, center_cells=True)
    return cnv.apply_noise_threshold(M, 1.5), labels, chromosomes


@pytest.mark.parametrize('method', ['leiden', 'hierarchical'])
def test_cluster_clones_recovers_planted_clones(method):
    from sklearn.metrics import adjusted_rand_score
    M, labels, chromosomes = _matrix()
    out = cnv.cluster_clones(M, method=method, resolution=0.5, n_clusters=4, n_pcs=10, seed=0)
    assert out.shape == (M.shape[0],) and out.min() == 0
    assert adjusted_rand_score(labels, out) > 0.9
    # Ranked: label 0 has the lowest mean score, i.e. the normal cells.
    score = cnv.cnv_score(M)
    means = [score[out == k].mean() for k in range(out.max() + 1)]
    assert means == sorted(means)
    assert (labels[out == 0] == 'normal').mean() > 0.9


def test_cluster_clones_is_deterministic():
    M, labels, chromosomes = _matrix()
    a = cnv.cluster_clones(M, method='leiden', resolution=0.5, n_clusters=4, n_pcs=10, seed=0)
    b = cnv.cluster_clones(M, method='leiden', resolution=0.5, n_clusters=4, n_pcs=10, seed=0)
    assert np.array_equal(a, b)


def test_cluster_clones_refuses_hierarchical_on_huge_input():
    M = np.zeros((cnv.HIERARCHICAL_MAX_CELLS + 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='leiden'):
        cnv.cluster_clones(M, method='hierarchical', resolution=1.0, n_clusters=2, n_pcs=2, seed=0)


def test_cluster_clones_with_very_few_windows():
    # Fewer windows than PCs (a huge window, or one window per chromosome):
    # PCA must not ask for more components than the matrix can give.
    rng = np.random.default_rng(0)
    M = np.vstack([rng.normal(0, 0.05, (40, 2)), rng.normal(1, 0.05, (40, 2))]).astype(np.float32)
    out = cnv.cluster_clones(M, method='leiden', resolution=0.3, n_clusters=2, n_pcs=30, seed=0)
    assert out.shape == (80,)
    one = cnv.cluster_clones(M[:, :1], method='hierarchical', resolution=1.0, n_clusters=2, n_pcs=30, seed=0)
    assert set(one) == {0, 1}


def test_cluster_clones_unknown_method():
    with pytest.raises(ValueError):
        cnv.cluster_clones(np.zeros((5, 2)), method='kmeans', resolution=1.0, n_clusters=2, n_pcs=2, seed=0)


def test_clone_summary():
    M, labels, chromosomes = _matrix()
    clones = cnv.cluster_clones(M, method='hierarchical', resolution=1.0, n_clusters=4, n_pcs=10, seed=0)
    score = cnv.cnv_score(M)
    cm = cnv.chromosome_means(M, chromosomes)
    rows = cnv.clone_summary(clones, score, cm, [c['name'] for c in chromosomes],
                             ref_mask=labels == 'normal', group_labels=labels)
    assert [r['clone'] for r in rows] == list(range(4))
    assert sum(r['n_cells'] for r in rows) == labels.size
    assert rows[0]['reference_fraction'] > 0.9 and rows[-1]['reference_fraction'] < 0.1
    assert rows[0]['top_group'] == 'normal'
    assert set(rows[0]['chromosome_means']) == {c['name'] for c in chromosomes}
    for r in rows:
        for v in r['chromosome_means'].values():
            assert v is None or np.isfinite(v)


# ---------- heatmap ----------

def test_heatmap_data_bins_rows_within_groups_and_orders_by_score():
    M, labels, chromosomes = _matrix()
    out = cnv.heatmap_data(M, groups=labels, group_order='cnv_score', row_mode='cells', max_rows=40,
                           vmax=None, chromosomes=chromosomes)
    assert len(out['matrix']) <= 40 and len(out['matrix']) == len(out['row_groups']) == len(out['row_labels'])
    # Every group present, quietest first, rows contiguous per group.
    seen = list(dict.fromkeys(out['row_groups']))
    assert seen[0] == 'normal' and set(seen) == set(GROUPS)
    assert out['column_groups'] == chromosomes
    assert out['n_cells'] == labels.size and out['vmax'] > 0
    assert all(np.isfinite(v) for row in out['matrix'] for v in row)
    grouped = cnv.heatmap_data(M, groups=labels, group_order='category', row_mode='groups', max_rows=40,
                               vmax=1.0, chromosomes=chromosomes)
    assert grouped['row_labels'] == sorted(GROUPS) and grouped['vmax'] == 1.0
    assert grouped['row_groups'] == sorted(GROUPS)


def test_heatmap_data_without_groups_and_with_nan_rows():
    M, labels, chromosomes = _matrix()
    M = M.copy(); M[:5] = np.nan
    out = cnv.heatmap_data(M, groups=None, group_order='cnv_score', row_mode='cells', max_rows=30,
                           vmax=None, chromosomes=chromosomes)
    assert out['n_cells'] == labels.size - 5
    assert all(g is None for g in out['row_groups'])
    assert len(out['matrix']) <= 30


def test_heatmap_data_every_group_gets_a_row():
    M, labels, chromosomes = _matrix()
    out = cnv.heatmap_data(M, groups=labels, group_order='category', row_mode='cells', max_rows=4,
                           vmax=None, chromosomes=chromosomes)
    assert sorted(set(out['row_groups'])) == sorted(GROUPS)


# ---------- shipped table ----------

TABLE = Path(cnv.__file__).parent / 'data' / 'gene_positions_mouse.tsv.gz'


@pytest.mark.skipif(not TABLE.exists(), reason='mouse position table not built')
def test_shipped_mouse_table_has_actb_on_chr5():
    table = cnv.load_positions('mouse')
    assert table['Actb'][0] == 'chr5'
    assert table['ENSMUSG00000029580'][0] == 'chr5'
    with gzip.open(TABLE, 'rt') as fh:
        assert fh.readline().rstrip('\n').split('\t') == ['gene_id', 'symbol', 'chromosome', 'start', 'end']


def test_load_positions_unknown_species():
    with pytest.raises(ValueError):
        cnv.load_positions('zebrafish')
