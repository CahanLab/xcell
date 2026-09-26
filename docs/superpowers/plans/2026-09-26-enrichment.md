# Gene-set Enrichment (ORA + preranked GSEA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap (hypergeometric) enrichment and preranked GSEA of the user's gene lists / contrasts against cached gene-set libraries, with results stored in `uns`, an analysis-record entry, a results modal with a running-score plot, and "add to gene sets".

**Architecture:** A pure `enrichment.py` (numpy/scipy) does the statistics. The adaptor builds the universe from the gene mask, resolves libraries via `gene_set_sources.find_library`, builds the GSEA ranking (scanpy DE on a snapshot, PCA loadings, or client scores), and persists JSON results under `uns['xcell_enrichment']`. Thin routes; a task for GSEA. One global `EnrichmentModal` with Overlap/GSEA tabs, pure helpers in `lib/enrichment.ts`.

**Tech Stack:** Python 3 / numpy / scipy.stats / scanpy (already deps); React + inline SVG (no chart lib); vitest; pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-enrichment-design.md`

## Global Constraints

- No new Python or npm dependencies (no gseapy). scipy + numpy only.
- Only cached libraries are used; an uncached one is `ValueError` → 400 with the hint "fetch it from the Gene set library first".
- `enrichment.py` never imports the adaptor and never mutates live state.
- Everything crossing the API is JSON-safe: no `inf`, no `NaN` (use `None` or corrected values).
- Results persist as `uns['xcell_enrichment'][key] = json.dumps(result)`; keys never overwrite (numeric suffix).
- Request models type `gene_subset` as `str | list[str] | GeneSubsetSpec | None`.
- Frontend styling inline; palette panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`/`#888`.
- Work happens in the worktree `/private/tmp/claude-501/xcell-wt` (branch `feat/enrichment`); run pytest with `cd /private/tmp/claude-501/xcell-wt/backend && "/Users/pcahan/Dropbox (Personal)/Code/xcell/.pixi/envs/dev/bin/python" -m pytest ...`; run vitest/tsc from `/private/tmp/claude-501/xcell-wt/frontend` after clonefile-copying `node_modules`.

## Review Focus

1. A query or library symbol that differs from the dataset only by case (`COL1A1` vs `Col1a1`) must count as present — test in Task 1 (`resolve_sets`) and Task 3 (adaptor query resolution).
2. A gene set whose members all have score exactly 0 (weight sum 0) must yield ES 0, NES 0, p 1, not a division error — test in Task 2.
3. A set larger than the ranked universe after NaN drops (or `k == N`) must not divide by zero in the miss term — test in Task 2 (`k == N` → ES 1).
4. Running the same key twice must produce `key`, `key_2`, never silently overwrite — test in Task 3.
5. A `gene_subset` sent as a `{columns, operation}` dict must reach the adaptor (the 422 trap) — test in Task 5.

---

### Task 1: `enrichment.py` — BH, set resolution, overlap test

**Files:**
- Create: `backend/xcell/enrichment.py`
- Test: `backend/tests/test_enrichment.py`

**Interfaces:**
- Produces:
  - `bh_adjust(p: np.ndarray) -> np.ndarray`
  - `ResolvedSet = dict` with keys `name, library, description, url, n_input, indices (np.ndarray[int])`
  - `resolve_sets(sets: list[dict], universe: list[str], *, min_size: int, max_size: int, directional: str = 'union') -> tuple[list[ResolvedSet], dict]` (second item: `{'n_input': int, 'n_dropped_size': int}`)
  - `resolve_symbols(symbols: list[str], universe: list[str]) -> tuple[list[int], list[str]]` (indices into universe, unresolved symbols)
  - `overlap_enrichment(query_indices: list[int], resolved_sets: list[ResolvedSet], n_universe: int, *, min_overlap: int = 2) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
"""Pure tests for xcell.enrichment: BH, symbol resolution, overlap test."""
import numpy as np
import pytest
from scipy.stats import fisher_exact, hypergeom

from xcell import enrichment as en


def test_bh_adjust_matches_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = en.bh_adjust(p)
    # sorted: 0.01,0.03,0.04,0.20 -> 0.04, 0.06, 0.0533->monotone 0.0533, 0.20
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /private/tmp/claude-501/xcell-wt/backend && "/Users/pcahan/Dropbox (Personal)/Code/xcell/.pixi/envs/dev/bin/python" -m pytest tests/test_enrichment.py -q`
Expected: ImportError / ModuleNotFoundError for `xcell.enrichment`.

- [ ] **Step 3: Implement**

```python
"""Gene-set enrichment statistics: overlap (hypergeometric) and preranked GSEA.

Pure: arrays and plain dicts in, dicts out. Never imports the adaptor.

Why hand-rolled rather than gseapy: it is not a dependency of the default
environment, the preranked algorithm is a few dozen numpy lines, and the
vectorised null (one permutation pool shared across set sizes, the fgsea
trick) is faster than gseapy's per-set loop for whole-library runs.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import hypergeom

ResolvedSet = dict[str, Any]
Report = Callable[[float, str], None] | None


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg; NaN stays NaN and is excluded from the count."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals, kind='stable')
    ranked = vals[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0.0, 1.0)
    out[ok] = adj
    return out


def _symbol_lookup(universe: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    exact = {g: i for i, g in enumerate(universe)}
    upper: dict[str, int] = {}
    for i, g in enumerate(universe):
        upper.setdefault(g.upper(), i)
    return exact, upper


def resolve_symbols(symbols: list[str], universe: list[str]) -> tuple[list[int], list[str]]:
    """Exact match first, then case-insensitive (COL1A1 -> Col1a1); de-duplicated."""
    exact, upper = _symbol_lookup(universe)
    return _resolve(symbols, exact, upper)


def _resolve(symbols: Any, exact: dict[str, int], upper: dict[str, int]) -> tuple[list[int], list[str]]:
    idx: list[int] = []
    missing: list[str] = []
    seen: set[int] = set()
    for raw in symbols if isinstance(symbols, (list, tuple)) else []:
        g = str(raw).strip()
        if not g:
            continue
        i = exact.get(g)
        if i is None:
            i = upper.get(g.upper())
        if i is None:
            missing.append(g)
            continue
        if i not in seen:
            seen.add(i)
            idx.append(i)
    return idx, missing


def resolve_sets(sets: list[dict[str, Any]], universe: list[str], *, min_size: int,
                 max_size: int, directional: str = 'union') -> tuple[list[ResolvedSet], dict[str, int]]:
    """Map library sets onto the universe and drop those outside [min_size, max_size].

    ``directional='union'`` merges ``genes`` and ``genes_down`` (overlap tests
    ignore sign); ``'split'`` emits ``<name>`` and ``<name> (down)`` because a
    ranked test cares which way a member should move.
    """
    if directional not in ('union', 'split'):
        raise ValueError("directional must be 'union' or 'split'")
    exact, upper = _symbol_lookup(universe)
    out: list[ResolvedSet] = []
    n_input = 0
    n_dropped = 0

    def emit(name: str, members: Any, src: dict[str, Any]) -> None:
        nonlocal n_dropped
        idx, _ = _resolve(members, exact, upper)
        if not (min_size <= len(idx) <= max_size):
            n_dropped += 1
            return
        out.append({
            'name': name,
            'library': str(src.get('library') or ''),
            'description': str(src.get('description') or ''),
            'url': str(src.get('url') or ''),
            'n_input': len(members) if isinstance(members, (list, tuple)) else 0,
            'indices': np.asarray(idx, dtype=np.int64),
        })

    for s in sets:
        if not isinstance(s, dict):
            continue
        n_input += 1
        name = str(s.get('name') or '')
        up = s.get('genes') or []
        down = s.get('genes_down') or []
        if directional == 'union':
            emit(name, list(up) + list(down), s)
        else:
            emit(name, list(up), s)
            if down:
                emit(f'{name} (down)', list(down), s)
    return out, {'n_input': n_input, 'n_dropped_size': n_dropped}


def _odds_ratio(k: int, n_query: int, n_set: int, n_universe: int) -> float:
    # 2x2: in-query&in-set, in-query&not, not-query&in-set, not-query&not.
    a, b = k, n_query - k
    c, d = n_set - k, n_universe - n_query - (n_set - k)
    if min(a, b, c, d) == 0:  # Haldane correction keeps the ratio finite
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float((a * d) / (b * c))


def overlap_enrichment(query_indices: list[int], resolved_sets: list[ResolvedSet], n_universe: int,
                       *, min_overlap: int = 2) -> list[dict[str, Any]]:
    """Hypergeometric over-representation of ``query_indices`` in each set."""
    q = np.unique(np.asarray(query_indices, dtype=np.int64))
    n_query = int(q.size)
    records: list[dict[str, Any]] = []
    pvals: list[float] = []
    for s in resolved_sets:
        members = s['indices']
        overlap = np.intersect1d(q, members, assume_unique=True)
        k, K = int(overlap.size), int(members.size)
        p = float(hypergeom.sf(k - 1, n_universe, K, n_query)) if k > 0 else 1.0
        expected = n_query * K / n_universe if n_universe else 0.0
        records.append({
            'name': s['name'], 'library': s['library'], 'description': s['description'],
            'url': s['url'], 'n_set': K, 'n_overlap': k,
            'expected': float(expected),
            'fold_enrichment': float(k / expected) if expected > 0 else 0.0,
            'odds_ratio': _odds_ratio(k, n_query, K, n_universe),
            'pval': min(max(p, 0.0), 1.0),
            'genes': [int(i) for i in overlap],
            'below_min_overlap': k < min_overlap,
        })
        pvals.append(records[-1]['pval'])
    adj = bh_adjust(np.asarray(pvals)) if records else np.array([])
    for r, a in zip(records, adj):
        r['padj'] = float(a)
    records.sort(key=lambda r: (r['pval'], r['name']))
    return records
```

- [ ] **Step 4: Run tests** — expected PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /private/tmp/claude-501/xcell-wt && git add backend/xcell/enrichment.py backend/tests/test_enrichment.py && git commit -m "feat(enrichment): BH, symbol resolution and hypergeometric overlap test"
```

---

### Task 2: `enrichment.py` — preranked GSEA

**Files:**
- Modify: `backend/xcell/enrichment.py`
- Test: `backend/tests/test_enrichment.py`

**Interfaces:**
- Produces: `preranked_gsea(scores: np.ndarray, resolved_sets: list[ResolvedSet], *, n_perm=1000, min_size=15, max_size=500, weight=1.0, seed=0, curve_top_n=50, report: Report = None) -> dict` returning `{n_ranked, n_perm, n_sets_tested, order: np.ndarray (ranked universe indices, descending score), results: [...]}`. Each result: `{name, library, description, url, n_set, es, nes, pval, padj, leading_edge: list[int] (universe indices), n_leading_edge, curve: list[[int, float]] | None}`.
- Also exposes `_es_from_positions(pos, w, n_ranked) -> (es, peak, pre, post)` for the naive-loop equivalence test.

- [ ] **Step 1: Write the failing tests** (append to `test_enrichment.py`)

```python
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
    assert by['top']['es'] > 0.9 and by['top']['nes'] > 1 and by['top']['pval'] == pytest.approx(1 / 201)
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
```

- [ ] **Step 2: Run to verify failure** — expected AttributeError on `_es_from_positions` / `preranked_gsea`.

- [ ] **Step 3: Implement** (append to `enrichment.py`)

```python
def _es_from_positions(pos: np.ndarray, w: np.ndarray, n_ranked: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enrichment score for m hit-position vectors at once.

    ``pos`` (m, k) sorted 0-based rank positions; ``w`` (m, k) the weights at
    those positions. The running sum rises at a hit and falls linearly between
    hits, so its extrema can only be the post-hit values (maxima) or the
    pre-hit values (minima) — no need to walk all n_ranked positions.
    Returns (es, peak_hit_index, pre, post).
    """
    m, k = pos.shape
    n_miss = n_ranked - k
    wsum = w.sum(axis=1, keepdims=True)
    ok = wsum[:, 0] > 0
    p_hit = np.zeros_like(w, dtype=float)
    if ok.any():
        p_hit[ok] = np.cumsum(w[ok], axis=1) / wsum[ok]
    if n_miss > 0:
        p_miss = (pos - np.arange(k)) / n_miss
    else:
        p_miss = np.zeros(pos.shape, dtype=float)
    post = p_hit - p_miss
    pre = np.concatenate([np.zeros((m, 1)), p_hit[:, :-1]], axis=1) - p_miss
    rows = np.arange(m)
    i_max = post.argmax(axis=1)
    i_min = pre.argmin(axis=1)
    vmax, vmin = post[rows, i_max], pre[rows, i_min]
    positive = vmax >= -vmin
    es = np.where(positive, vmax, vmin)
    peak = np.where(positive, i_max, i_min)
    es = np.where(ok, es, 0.0)
    return es, peak, pre, post


def preranked_gsea(scores: np.ndarray, resolved_sets: list[ResolvedSet], *, n_perm: int = 1000,
                   min_size: int = 15, max_size: int = 500, weight: float = 1.0, seed: int = 0,
                   curve_top_n: int = 50, report: Report = None) -> dict[str, Any]:
    """Preranked GSEA (Subramanian 2005) with a gene-permutation null.

    The null for size k is the first k entries of each of n_perm random
    permutations of the ranked list — a uniform random k-subset — so one
    permutation pool serves every set size and each size class costs one
    vectorised ES evaluation. p-values are therefore floored at 1/(n_perm+1).
    """
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    keep = np.flatnonzero(finite)
    order = keep[np.argsort(-scores[keep], kind='stable')]   # universe idx, best first
    n = int(order.size)
    if n < 2:
        raise ValueError('Fewer than 2 genes have a finite ranking score')
    rank_of = np.full(scores.size, -1, dtype=np.int64)
    rank_of[order] = np.arange(n)
    ranked_w = np.abs(scores[order]) ** float(weight) if weight != 0 else np.ones(n)

    sets: list[tuple[ResolvedSet, np.ndarray]] = []
    for s in resolved_sets:
        pos = rank_of[np.asarray(s['indices'], dtype=np.int64)]
        pos = np.sort(pos[pos >= 0])
        if min_size <= pos.size <= max_size:
            sets.append((s, pos))
    if not sets:
        raise ValueError(f'No gene set has between {min_size} and {max_size} ranked members')

    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((int(n_perm), n)), axis=1)
    null_cache: dict[int, np.ndarray] = {}

    def null_for(k: int) -> np.ndarray:
        if k not in null_cache:
            pos = np.sort(perms[:, :k], axis=1)
            null_cache[k], _, _, _ = _es_from_positions(pos, ranked_w[pos], n)
        return null_cache[k]

    sizes = sorted({pos.size for _, pos in sets})
    results: list[dict[str, Any]] = []
    for i, (s, pos) in enumerate(sets):
        k = int(pos.size)
        es_arr, peak, pre, post = _es_from_positions(pos[None, :], ranked_w[pos][None, :], n)
        es, j = float(es_arr[0]), int(peak[0])
        null = null_for(k)
        if es > 0:
            same = null[null > 0]
        elif es < 0:
            same = null[null < 0]
        else:
            same = np.array([])
        if es == 0 or same.size == 0:
            nes, p = 0.0, 1.0
        else:
            nes = es / float(np.mean(np.abs(same)))
            p = (1 + int((np.abs(same) >= abs(es)).sum())) / (1 + same.size)
        lead = pos[: j + 1] if es >= 0 else pos[j:]
        results.append({
            'name': s['name'], 'library': s['library'], 'description': s['description'],
            'url': s['url'], 'n_set': k, 'es': es, 'nes': float(nes), 'pval': float(p),
            'leading_edge': [int(order[q]) for q in lead], 'n_leading_edge': int(lead.size),
            '_pre': pre[0], '_post': post[0], '_pos': pos,
        })
        if report is not None:
            report((i + 1) / len(sets), f'Testing gene sets ({i + 1}/{len(sets)}, {len(sizes)} size classes)')

    adj = bh_adjust(np.asarray([r['pval'] for r in results]))
    for r, a in zip(results, adj):
        r['padj'] = float(a)
    results.sort(key=lambda r: (r['pval'], -abs(r['nes']), r['name']))
    for rank, r in enumerate(results):
        pre, post, pos = r.pop('_pre'), r.pop('_post'), r.pop('_pos')
        if rank < curve_top_n:
            curve: list[list[float]] = [[0, 0.0]]
            for q in range(pos.size):
                curve.append([int(pos[q]), float(pre[q])])
                curve.append([int(pos[q]), float(post[q])])
            curve.append([n - 1, 0.0])
            r['curve'] = curve
        else:
            r['curve'] = None
    return {'n_ranked': n, 'n_perm': int(n_perm), 'n_sets_tested': len(results),
            'order': order, 'results': results}
```

Note for the `'all'` edge test: with `k == n`, `n_miss == 0` and `p_miss` is all zeros, so post-hit values climb to exactly 1.0.

- [ ] **Step 4: Run tests** — expected PASS (9 tests). If `rand` NES bound flakes, widen to `< 1.8`; it is a sanity bound, not the contract.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/enrichment.py backend/tests/test_enrichment.py && git commit -m "feat(enrichment): vectorised preranked GSEA with gene-permutation null"
```

---

### Task 3: Adaptor — libraries, universe, overlap run, storage

**Files:**
- Modify: `backend/xcell/adaptor.py` (add a section after `gene_set_overlap`, ~line 5946)
- Test: `backend/tests/test_enrichment_adaptor.py`

**Interfaces:**
- Consumes: Task 1 functions; `gene_set_sources.find_library`, `self._resolve_gene_mask`, `self._sanitize_subset_name`, `self._log_action`.
- Produces on `DataAdaptor`:
  - `ENRICHMENT_UNS_KEY = 'xcell_enrichment'`
  - `_enrichment_sets(libraries, sets) -> list[dict]` (each `{name, genes, genes_down, library, description, url}`)
  - `_enrichment_universe(gene_subset) -> (universe: list[str], subset_type: str, meta: dict)`
  - `_store_enrichment(key_hint: str, result: dict) -> str` (returns the final key)
  - `run_overlap_enrichment(genes, *, name=None, libraries=None, sets=None, gene_subset=None, min_set_size=5, max_set_size=500, min_overlap=2, key=None) -> dict`
  - `get_enrichment_results() -> list[dict]` (summaries: `key, kind, label, n_sets_tested, n_significant, created_at`), `get_enrichment_result(key) -> dict` (KeyError), `delete_enrichment_result(key) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
"""Adaptor tests for overlap enrichment: universe, resolution, storage."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Sox9', 'Acan', 'Actb', 'Gapdh', 'Ptprc', 'Cd3e', 'Cd19',
         'Hoxd13', 'Meis1'] + [f'G{i}' for i in range(20)]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({
        'source': 'msigdb', 'id': 'toy', 'name': 'Toy library', 'species': 'mouse', 'version': '1',
        'n_sets': 3, 'sets': [
            {'name': 'COLLAGEN', 'genes': ['COL1A1', 'COL1A2', 'COL3A1', 'G0', 'G1'], 'url': 'u'},
            {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']},
            {'name': 'TINY', 'genes': ['Sox9']},
        ]})
    yield
    gss.set_cache_dir(None)


def _adata():
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(30, len(GENES))).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = GENES
    ad.var['panel'] = [g.startswith('Col') or g.startswith('G') for g in GENES]
    ad.obs['grp'] = pd.Categorical(['a'] * 15 + ['b'] * 15)
    return ad


def _adaptor():
    return DataAdaptor('x.h5ad', adata=_adata())


def test_overlap_run_resolves_case_and_stores_json():
    a = _adaptor()
    res = a.run_overlap_enrichment(['col1a1', 'Col1a2', 'Col3a1', 'Nope'], name='my set',
                                   libraries=[{'source': 'msigdb', 'id': 'toy', 'species': 'mouse'}],
                                   min_set_size=2, min_overlap=1)
    assert res['kind'] == 'ora' and res['key'] == 'ora_my_set'
    assert res['query'] == {'name': 'my set', 'n_input': 4, 'n_in_universe': 3, 'genes_missing': ['Nope']}
    assert res['universe_size'] == len(GENES) and res['n_sets_input'] == 3 and res['n_sets_tested'] == 2
    top = res['results'][0]
    assert top['name'] == 'COLLAGEN' and top['library'] == 'Toy library'
    assert top['genes'] == ['Col1a1', 'Col1a2', 'Col3a1'] and top['n_overlap'] == 3
    stored = json.loads(a.adata.uns['xcell_enrichment']['ora_my_set'])
    assert stored['results'][0]['name'] == 'COLLAGEN'
    assert a.get_enrichment_result('ora_my_set')['key'] == 'ora_my_set'
    assert [s['key'] for s in a.get_enrichment_results()] == ['ora_my_set']
    assert a._analysis_record[-1]['action'] == 'enrichment_ora' if hasattr(a, '_analysis_record') else True


def test_overlap_universe_honours_gene_mask_and_subset_column():
    a = _adaptor()
    a.set_gene_mask(keep_columns=['panel'], hide_columns=[])
    res = a.run_overlap_enrichment(['Col1a1', 'Col1a2', 'Ptprc'],
                                   libraries=[{'source': 'msigdb', 'id': 'toy'}], min_set_size=2, min_overlap=1)
    assert res['universe_size'] == 23            # 3 Col + 20 G
    assert res['query']['n_in_universe'] == 2 and res['query']['genes_missing'] == ['Ptprc']
    assert [r['name'] for r in res['results']] == ['COLLAGEN', 'IMMUNE']   # IMMUNE keeps G2,G3
    b = _adaptor()
    res2 = b.run_overlap_enrichment(['Col1a1', 'Col1a2'], gene_subset='panel',
                                    libraries=[{'source': 'msigdb', 'id': 'toy'}], min_set_size=2, min_overlap=1)
    assert res2['universe_size'] == 23 and res2['gene_subset_type'] == 'column:panel'


def test_overlap_keys_never_overwrite_and_delete():
    a = _adaptor()
    lib = [{'source': 'msigdb', 'id': 'toy'}]
    k1 = a.run_overlap_enrichment(['Col1a1', 'Col1a2'], name='x', libraries=lib, min_set_size=2, min_overlap=1)['key']
    k2 = a.run_overlap_enrichment(['Col1a1', 'Col1a2'], name='x', libraries=lib, min_set_size=2, min_overlap=1)['key']
    assert (k1, k2) == ('ora_x', 'ora_x_2')
    assert a.delete_enrichment_result('ora_x') == {'deleted': 'ora_x'}
    assert [s['key'] for s in a.get_enrichment_results()] == ['ora_x_2']
    with pytest.raises(KeyError):
        a.get_enrichment_result('ora_x')


def test_overlap_inline_sets_and_errors():
    a = _adaptor()
    res = a.run_overlap_enrichment(['Col1a1', 'Col1a2'], sets=[{'name': 'mine', 'genes': ['Col1a1', 'Col1a2', 'Sox9']}],
                                   min_set_size=2, min_overlap=1)
    assert res['results'][0]['library'] == 'My gene sets'
    with pytest.raises(ValueError, match='not cached'):
        a.run_overlap_enrichment(['Col1a1', 'Col1a2'], libraries=[{'source': 'msigdb', 'id': 'missing'}])
    with pytest.raises(ValueError, match='at least 2'):
        a.run_overlap_enrichment(['Col1a1', 'Nope'], sets=[{'name': 'mine', 'genes': ['Col1a1', 'Col1a2']}], min_set_size=1)
    with pytest.raises(ValueError, match='No gene sets'):
        a.run_overlap_enrichment(['Col1a1', 'Col1a2'])
```

- [ ] **Step 2: Run to verify failure** — AttributeError `run_overlap_enrichment`.

- [ ] **Step 3: Implement** (insert after `gene_set_overlap`, before `guess_species`)

```python
    # ------------------------------------------------------------------
    # Gene-set enrichment (overlap + preranked GSEA)
    # ------------------------------------------------------------------
    ENRICHMENT_UNS_KEY = 'xcell_enrichment'

    def _enrichment_sets(self, libraries: list[dict[str, Any]] | None,
                         sets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Cached libraries plus inline sets, each tagged with a library label."""
        from xcell import gene_set_sources as gss  # noqa: PLC0415
        out: list[dict[str, Any]] = []
        for ref in libraries or []:
            source = str(ref.get('source') or '')
            lib_id = str(ref.get('id') or '')
            lib = gss.find_library(source, lib_id, ref.get('species'))
            if lib is None:
                raise ValueError(
                    f"Library '{lib_id}' from {source or '?'} is not cached; "
                    "fetch it from the Gene set library first")
            label = str(lib.get('name') or lib_id)
            for s in lib.get('sets') or []:
                if isinstance(s, dict):
                    out.append({
                        'name': s.get('name', ''), 'genes': list(s.get('genes') or []),
                        'genes_down': list(s.get('genes_down') or []), 'library': label,
                        'description': s.get('description', ''), 'url': s.get('url', ''),
                    })
        for s in sets or []:
            if isinstance(s, dict):
                out.append({
                    'name': s.get('name', ''), 'genes': list(s.get('genes') or []),
                    'genes_down': list(s.get('genes_down') or s.get('genesDown') or []),
                    'library': 'My gene sets', 'description': '', 'url': '',
                })
        if not out:
            raise ValueError('No gene sets to test: choose at least one library or gene set')
        return out

    def _enrichment_universe(self, gene_subset: Any) -> tuple[list[str], str, dict[str, Any]]:
        mask, subset_type, meta = self._resolve_gene_mask(gene_subset)
        universe = [str(g) for g in self.adata.var_names[mask]]
        return universe, subset_type, meta

    def _enrichment_store(self) -> dict[str, str]:
        raw = self.adata.uns.get(self.ENRICHMENT_UNS_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def _store_enrichment(self, key_hint: str, result: dict[str, Any]) -> str:
        """Persist as JSON under a never-colliding key; returns the key used."""
        import json  # noqa: PLC0415
        base = self._sanitize_subset_name(key_hint)
        store = self._enrichment_store()
        key, n = base, 1
        while key in store:
            n += 1
            key = f'{base}_{n}'
        result['key'] = key
        store[key] = json.dumps(result)
        self.adata.uns[self.ENRICHMENT_UNS_KEY] = store
        return key

    def get_enrichment_results(self) -> list[dict[str, Any]]:
        import json  # noqa: PLC0415
        out = []
        for key, raw in self._enrichment_store().items():
            try:
                r = json.loads(raw)
            except (TypeError, ValueError):
                continue
            out.append({k: r.get(k) for k in ('key', 'kind', 'label', 'n_sets_tested', 'n_significant', 'created_at')})
        out.sort(key=lambda s: s.get('created_at') or '', reverse=True)
        return out

    def get_enrichment_result(self, key: str) -> dict[str, Any]:
        import json  # noqa: PLC0415
        store = self._enrichment_store()
        if key not in store:
            raise KeyError(f"No enrichment result named '{key}'")
        return json.loads(store[key])

    def delete_enrichment_result(self, key: str) -> dict[str, Any]:
        store = self._enrichment_store()
        if key not in store:
            raise KeyError(f"No enrichment result named '{key}'")
        del store[key]
        self.adata.uns[self.ENRICHMENT_UNS_KEY] = store
        self._log_action('enrichment_delete', {'key': key}, {'deleted': key})
        return {'deleted': key}

    def run_overlap_enrichment(self, genes: list[str], *, name: str | None = None,
                               libraries: list[dict[str, Any]] | None = None,
                               sets: list[dict[str, Any]] | None = None,
                               gene_subset: Any = None, min_set_size: int = 5,
                               max_set_size: int = 500, min_overlap: int = 2,
                               key: str | None = None) -> dict[str, Any]:
        """Hypergeometric over-representation of ``genes`` in each library set."""
        from datetime import datetime, timezone  # noqa: PLC0415
        from xcell import enrichment as en  # noqa: PLC0415
        if min_set_size < 1 or max_set_size < min_set_size:
            raise ValueError('Set size range must satisfy 1 <= min <= max')
        raw_sets = self._enrichment_sets(libraries, sets)
        universe, subset_type, meta = self._enrichment_universe(gene_subset)
        q_idx, missing = en.resolve_symbols(list(genes or []), universe)
        if len(q_idx) < 2:
            raise ValueError(
                f'Need at least 2 query genes present in the universe; {len(q_idx)} of '
                f'{len(genes or [])} resolved')
        resolved, rmeta = en.resolve_sets(raw_sets, universe, min_size=min_set_size,
                                          max_size=max_set_size, directional='union')
        if not resolved:
            raise ValueError(f'No gene set has between {min_set_size} and {max_set_size} members in the universe')
        records = en.overlap_enrichment(q_idx, resolved, len(universe), min_overlap=min_overlap)
        for r in records:
            r['genes'] = [universe[i] for i in r['genes']]
        label = name or 'gene list'
        params = {
            'name': label, 'n_query': len(genes or []), 'libraries': list(libraries or []),
            'n_inline_sets': len(sets or []), 'gene_subset': gene_subset,
            'min_set_size': min_set_size, 'max_set_size': max_set_size, 'min_overlap': min_overlap,
        }
        result = {
            'kind': 'ora', 'label': f'Overlap: {label}',
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'query': {'name': label, 'n_input': len(genes or []), 'n_in_universe': len(q_idx),
                      'genes_missing': missing[:self.MAX_REPORTED_OVERLAP_MISSING]},
            'universe_size': len(universe), 'gene_subset_type': subset_type,
            'n_sets_input': rmeta['n_input'], 'n_sets_tested': len(records),
            'n_significant': sum(1 for r in records if r['padj'] <= 0.05),
            'results': records, 'params': params,
        }
        stored_key = self._store_enrichment(key or f'ora_{label}', result)
        self._log_action('enrichment_ora', params, {
            'key': stored_key, 'n_sets_tested': len(records), 'n_significant': result['n_significant']})
        return result
```

(`self.MAX_REPORTED_OVERLAP_MISSING` already exists for `gene_set_overlap`; if the name differs, use the existing constant.) The `_analysis_record` assertion in the first test should be adjusted to however `_log_action` exposes the record — check `grep -n "def _log_action" -A 12 backend/xcell/adaptor.py` and assert on the real structure (e.g. `a.get_analysis_record()['actions'][-1]['action']`).

- [ ] **Step 4: Run tests** — expected PASS (4 tests). Also run `tests/test_gene_set_overlap*.py` if present to confirm nothing moved.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_enrichment_adaptor.py && git commit -m "feat(enrichment): adaptor overlap run, universe from gene mask, JSON results in uns"
```

---

### Task 4: Adaptor — `prepare_gsea` with diffexp / PCA / scores rankings

**Files:**
- Modify: `backend/xcell/adaptor.py` (same section)
- Test: `backend/tests/test_enrichment_adaptor.py`

**Interfaces:**
- Consumes: Task 2 `preranked_gsea`; `self._subset_mask(name)` (KeyError if unknown); `varm['PCs']` / `varm['PCs_<subset>']`.
- Produces: `prepare_gsea(ranking: dict, *, libraries=None, sets=None, gene_subset=None, n_perm=1000, min_set_size=15, max_set_size=500, weight=1.0, seed=0, key=None) -> (compute_fn, apply_fn)`; `apply_fn(result)` returns `{key, kind:'gsea', label, ranking: {kind, label, n_ranked, genes, scores}, universe_size, gene_subset_type, n_sets_input, n_sets_tested, n_significant, n_perm, results, params, created_at}` where each result's `leading_edge` holds gene names.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _adata_de():
    """Group a over-expresses Col*, group b over-expresses Ptprc/Cd3e/Cd19."""
    rng = np.random.default_rng(0)
    n = 60
    lam = np.full((n, len(GENES)), 1.0)
    grp = np.array(['a'] * 30 + ['b'] * 30)
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        lam[grp == 'a', GENES.index(g)] = 8.0
    for g in ('Ptprc', 'Cd3e', 'Cd19'):
        lam[grp == 'b', GENES.index(g)] = 8.0
    X = csr_matrix(rng.poisson(lam).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(grp)
    ad.obs['subset_half'] = [True] * 40 + [False] * 20
    ad.uns['xcell_cell_subsets'] = {'half': {'n_cells': 40, 'created_at': 't', 'origin': 'test'}}
    pcs = np.zeros((len(GENES), 2))
    pcs[GENES.index('Hoxd13'), 0] = 0.9
    pcs[GENES.index('Meis1'), 0] = -0.9
    ad.varm['PCs'] = pcs
    ad.obsm['X_pca'] = np.zeros((n, 2))
    return ad


def _run(a, ranking, **kw):
    compute_fn, apply_fn = a.prepare_gsea(ranking, libraries=[{'source': 'msigdb', 'id': 'toy'}],
                                         min_set_size=2, n_perm=100, **kw)
    return apply_fn(compute_fn(lambda f, m: None))


def test_gsea_diffexp_ranking_vs_rest_and_vs_group():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest'})
    assert res['kind'] == 'gsea' and res['key'] == 'gsea_grp_a_vs_rest'
    assert res['ranking']['genes'][:3] and set(res['ranking']['genes'][:3]) == {'Col1a1', 'Col1a2', 'Col3a1'}
    by = {r['name']: r for r in res['results']}
    assert by['COLLAGEN']['nes'] > 0 and by['IMMUNE']['nes'] < 0
    assert set(by['COLLAGEN']['leading_edge']) >= {'Col1a1', 'Col1a2', 'Col3a1'}
    assert json.loads(a.adata.uns['xcell_enrichment']['gsea_grp_a_vs_rest'])['kind'] == 'gsea'
    res2 = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'b', 'reference': 'a', 'metric': 'log2fc'})
    assert res2['key'] == 'gsea_grp_b_vs_a'
    assert {r['name']: r for r in res2['results']}['IMMUNE']['nes'] > 0


def test_gsea_diffexp_within_named_subset_and_validation():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest', 'cell_subset': 'half'})
    assert res['params']['ranking']['cell_subset'] == 'half' and res['ranking']['n_ranked'] == len(GENES)
    for bad, msg in [
        ({'kind': 'diffexp', 'obs_column': 'nope', 'group': 'a'}, 'nope'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'zzz'}, 'zzz'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'a'}, 'reference'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'method': 'logreg'}, 'method'),
        ({'kind': 'pca', 'component': 5}, 'component'),
        ({'kind': 'scores', 'genes': ['Col1a1'], 'scores': [1.0, 2.0]}, 'same length'),
        ({'kind': 'nope'}, 'kind'),
    ]:
        with pytest.raises(ValueError, match=msg):
            a.prepare_gsea(bad, libraries=[{'source': 'msigdb', 'id': 'toy'}], min_set_size=2)
    with pytest.raises(KeyError):
        a.prepare_gsea({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'cell_subset': 'ghost'},
                       libraries=[{'source': 'msigdb', 'id': 'toy'}], min_set_size=2)


def test_gsea_pca_and_scores_rankings():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'pca', 'component': 0})
    assert res['key'] == 'gsea_pca1' and res['ranking']['genes'][0] == 'Hoxd13' and res['ranking']['genes'][-1] == 'Meis1'
    res2 = _run(a, {'kind': 'scores', 'genes': ['COL1A1', 'col1a2', 'Col3a1', 'Ptprc'], 'scores': [3, 2, 1, -1]},
                key='custom')
    assert res2['key'] == 'custom' and res2['ranking']['n_ranked'] == 4 and res2['ranking']['genes'][0] == 'Col1a1'
    assert res2['ranking']['label'] == 'custom scores'
```

- [ ] **Step 2: Run to verify failure** — AttributeError `prepare_gsea`.

- [ ] **Step 3: Implement** (append to the enrichment section)

```python
    def _gsea_ranking_snapshot(self, ranking: dict[str, Any], universe: list[str],
                               universe_mask: np.ndarray) -> tuple[dict[str, Any], Callable[[], np.ndarray], str]:
        """Validate a ranking spec now; return (clean spec, builder, key hint).

        The builder runs inside the background task and returns one score per
        universe gene (NaN = unranked). Everything it needs is snapshotted here
        so a later mutation of the live AnnData cannot leak in.
        """
        kind = str(ranking.get('kind') or '')
        cell_subset = ranking.get('cell_subset') or None
        cell_mask = self._subset_mask(cell_subset) if cell_subset else None   # KeyError if unknown

        if kind == 'diffexp':
            col = str(ranking.get('obs_column') or '')
            if col not in self.adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in .obs")
            values = self.adata.obs[col].astype(str).values
            if cell_mask is not None:
                values = np.where(cell_mask, values, '\x00excluded')
            group = str(ranking.get('group') or '')
            reference = str(ranking.get('reference') or 'rest')
            method = str(ranking.get('method') or 'wilcoxon')
            metric = str(ranking.get('metric') or 'score')
            present = set(values[values != '\x00excluded'])
            if group not in present:
                raise ValueError(f"Group '{group}' not found in column '{col}'" + (f" within subset '{cell_subset}'" if cell_subset else ''))
            if reference != 'rest' and reference not in present:
                raise ValueError(f"Reference group '{reference}' not found in column '{col}'")
            if reference == group:
                raise ValueError('reference must differ from group')
            if method not in ('wilcoxon', 't-test'):
                raise ValueError("method must be 'wilcoxon' or 't-test'")
            if metric not in ('score', 'log2fc'):
                raise ValueError("metric must be 'score' or 'log2fc'")
            in_group = values == group
            in_ref = (values != group) & (values != '\x00excluded') if reference == 'rest' else values == reference
            if in_group.sum() < 2 or in_ref.sum() < 2:
                raise ValueError('Each side of the contrast needs at least 2 cells')
            cells = np.flatnonzero(in_group | in_ref)
            labels = np.where(in_group[cells], 'group', 'reference')
            X = self.adata.X[cells][:, universe_mask]
            X = X.copy() if hasattr(X, 'copy') else np.array(X)

            def build() -> np.ndarray:
                import anndata as _ad  # noqa: PLC0415
                import scanpy as sc  # noqa: PLC0415
                tmp = _ad.AnnData(X=X)
                tmp.var_names = universe
                tmp.obs['g'] = pd.Categorical(labels, categories=['group', 'reference'])
                sc.tl.rank_genes_groups(tmp, groupby='g', groups=['group'], reference='reference',
                                        method=method, use_raw=False, key_added='r')
                r = tmp.uns['r']
                names = [str(g) for g in r['names']['group']]
                vals = np.asarray(r['scores' if metric == 'score' else 'logfoldchanges']['group'], dtype=float)
                pos = {g: i for i, g in enumerate(universe)}
                out = np.full(len(universe), np.nan)
                for g, v in zip(names, vals):
                    out[pos[g]] = v
                return out

            clean = {'kind': 'diffexp', 'obs_column': col, 'group': group, 'reference': reference,
                     'method': method, 'metric': metric, 'cell_subset': cell_subset}
            label = f'{col}: {group} vs {reference}' + (f' [{cell_subset}]' if cell_subset else '')
            return clean, build, f'gsea_{col}_{group}_vs_{reference}'

        if kind == 'pca':
            comp = ranking.get('component')
            pcs_key = f'PCs_{cell_subset}' if cell_subset else 'PCs'
            if pcs_key not in self.adata.varm:
                raise ValueError(f"No PCA loadings ('{pcs_key}' missing from .varm); run PCA first")
            pcs = np.asarray(self.adata.varm[pcs_key], dtype=float)
            if not isinstance(comp, int) or comp < 0 or comp >= pcs.shape[1]:
                raise ValueError(f'component must be an integer in [0, {pcs.shape[1] - 1}]')
            loading = pcs[universe_mask, comp].copy()
            loading[loading == 0] = np.nan   # genes outside the PCA's gene mask
            clean = {'kind': 'pca', 'component': int(comp), 'cell_subset': cell_subset}
            return clean, (lambda: loading), f'gsea_pca{comp + 1}' + (f'_{cell_subset}' if cell_subset else '')

        if kind == 'scores':
            from xcell import enrichment as en  # noqa: PLC0415
            genes = list(ranking.get('genes') or [])
            scores = list(ranking.get('scores') or [])
            if len(genes) != len(scores) or not genes:
                raise ValueError('genes and scores must be non-empty lists of the same length')
            idx, _missing = en.resolve_symbols(genes, universe)
            arr = np.full(len(universe), np.nan)
            # resolve_symbols de-duplicates, so walk the originals to pair scores
            exact, upper = en._symbol_lookup(universe)
            for g, v in zip(genes, scores):
                i = exact.get(str(g).strip())
                if i is None:
                    i = upper.get(str(g).strip().upper())
                if i is not None and np.isnan(arr[i]):
                    arr[i] = float(v)
            if np.isfinite(arr).sum() < 2:
                raise ValueError('Fewer than 2 of the scored genes are in the universe')
            clean = {'kind': 'scores', 'n_genes': len(genes)}
            return clean, (lambda: arr), 'gsea_scores'

        raise ValueError("ranking kind must be 'diffexp', 'pca' or 'scores'")

    def prepare_gsea(self, ranking: dict[str, Any], *, libraries: list[dict[str, Any]] | None = None,
                     sets: list[dict[str, Any]] | None = None, gene_subset: Any = None,
                     n_perm: int = 1000, min_set_size: int = 15, max_set_size: int = 500,
                     weight: float = 1.0, seed: int = 0, key: str | None = None):
        """Preranked GSEA as a background task: (compute_fn(report), apply_fn(result))."""
        from datetime import datetime, timezone  # noqa: PLC0415
        from xcell import enrichment as en  # noqa: PLC0415
        if n_perm < 10:
            raise ValueError('n_perm must be at least 10')
        if min_set_size < 1 or max_set_size < min_set_size:
            raise ValueError('Set size range must satisfy 1 <= min <= max')
        raw_sets = self._enrichment_sets(libraries, sets)
        mask, subset_type, _meta = self._resolve_gene_mask(gene_subset)
        universe = [str(g) for g in self.adata.var_names[mask]]
        clean_ranking, build, key_hint = self._gsea_ranking_snapshot(dict(ranking), universe, mask)
        resolved, rmeta = en.resolve_sets(raw_sets, universe, min_size=min_set_size,
                                          max_size=max_set_size, directional='split')
        if not resolved:
            raise ValueError(f'No gene set has between {min_set_size} and {max_set_size} members in the universe')
        params = {
            'ranking': clean_ranking, 'libraries': list(libraries or []), 'n_inline_sets': len(sets or []),
            'gene_subset': gene_subset, 'n_perm': int(n_perm), 'min_set_size': int(min_set_size),
            'max_set_size': int(max_set_size), 'weight': float(weight), 'seed': int(seed),
        }
        label = {'diffexp': lambda r: f"{r['obs_column']}: {r['group']} vs {r['reference']}",
                 'pca': lambda r: f"PC{r['component'] + 1} loading",
                 'scores': lambda r: 'custom scores'}[clean_ranking['kind']](clean_ranking)
        if clean_ranking.get('cell_subset'):
            label += f" [{clean_ranking['cell_subset']}]"

        def compute_fn(report):
            report(0.02, 'Building the ranking…')
            scores = build()
            report(0.15, 'Running GSEA…')
            out = en.preranked_gsea(
                scores, resolved, n_perm=n_perm, min_size=min_set_size, max_size=max_set_size,
                weight=weight, seed=seed,
                report=lambda f, m: report(0.15 + 0.85 * f, m))
            out['scores'] = scores
            return out

        def apply_fn(out):
            order = np.asarray(out['order'])
            scores = np.asarray(out['scores'])
            for r in out['results']:
                r['leading_edge'] = [universe[i] for i in r['leading_edge']]
            result = {
                'kind': 'gsea', 'label': f'GSEA: {label}',
                'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'ranking': {'kind': clean_ranking['kind'], 'label': label, 'n_ranked': int(out['n_ranked']),
                            'genes': [universe[i] for i in order],
                            'scores': [float(scores[i]) for i in order]},
                'universe_size': len(universe), 'gene_subset_type': subset_type,
                'n_sets_input': rmeta['n_input'], 'n_sets_tested': int(out['n_sets_tested']),
                'n_significant': sum(1 for r in out['results'] if r['padj'] <= 0.05),
                'n_perm': int(out['n_perm']), 'results': out['results'], 'params': params,
            }
            stored_key = self._store_enrichment(key or key_hint, result)
            self._log_action('enrichment_gsea', params, {
                'key': stored_key, 'n_sets_tested': result['n_sets_tested'],
                'n_significant': result['n_significant']})
            return result

        return compute_fn, apply_fn
```

Add `from typing import Callable` to the adaptor's imports if not already present; `pd` is already imported.

- [ ] **Step 4: Run tests** — expected PASS (7 tests in the adaptor file).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_enrichment_adaptor.py && git commit -m "feat(enrichment): prepare_gsea with diffexp, PCA-loading and custom-score rankings"
```

---

### Task 5: Routes + config

**Files:**
- Modify: `backend/xcell/api/routes.py` (add after the `/gene_sets/species_guess` route, ~line 1130)
- Modify: `backend/xcell/config.yaml` (add an `enrichment:` section after `diff_exp:`)
- Test: `backend/tests/test_enrichment_routes.py`

**Interfaces:**
- Produces: `POST /api/enrichment/overlap` (200), `POST /api/enrichment/gsea` (202 `{task_id, status}`), `GET /api/enrichment/results` (`{results: [...]}`), `GET /api/enrichment/results/{key}`, `DELETE /api/enrichment/results/{key}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Route tests for /api/enrichment/*."""
import time

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Ptprc', 'Cd3e', 'Cd19'] + [f'G{i}' for i in range(14)]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    gss.save_library({'source': 'msigdb', 'id': 'toy', 'name': 'Toy', 'species': 'mouse', 'version': '1', 'n_sets': 2,
                      'sets': [{'name': 'COLLAGEN', 'genes': ['Col1a1', 'Col1a2', 'Col3a1', 'G0', 'G1']},
                               {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']}]})
    yield
    gss.set_cache_dir(None)


def _install():
    rng = np.random.default_rng(0)
    n = 60
    lam = np.full((n, len(GENES)), 1.0)
    grp = np.array(['a'] * 30 + ['b'] * 30)
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        lam[grp == 'a', GENES.index(g)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.var['hv'] = [True] * 10 + [False] * 10
    ad.var['panel'] = [True] * 20
    ad.obs['grp'] = pd.Categorical(grp)
    routes.set_adaptor(DataAdaptor('x.h5ad', adata=ad), slot='primary')


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f'/api/tasks/{task_id}').json()
        if s['status'] in ('completed', 'failed', 'cancelled'):
            return s
        time.sleep(0.05)
    raise AssertionError('task did not finish')


def test_overlap_route_accepts_dict_gene_subset_and_returns_result():
    _install()
    c = TestClient(app)
    body = {'genes': ['Col1a1', 'Col1a2', 'Col3a1'], 'name': 'q',
            'libraries': [{'source': 'msigdb', 'id': 'toy'}],
            'gene_subset': {'columns': ['hv', 'panel'], 'operation': 'intersection'},
            'min_set_size': 2, 'min_overlap': 1}
    r = c.post('/api/enrichment/overlap', json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['universe_size'] == 10 and d['results'][0]['name'] == 'COLLAGEN'
    assert d['gene_subset_type'].startswith('intersection')
    lst = c.get('/api/enrichment/results').json()['results']
    assert lst[0]['key'] == d['key'] and lst[0]['kind'] == 'ora'
    assert c.get(f"/api/enrichment/results/{d['key']}").json()['results'][0]['name'] == 'COLLAGEN'
    assert c.delete(f"/api/enrichment/results/{d['key']}").json() == {'deleted': d['key']}
    assert c.get(f"/api/enrichment/results/{d['key']}").status_code == 404


def test_overlap_route_error_mapping():
    _install()
    c = TestClient(app)
    r = c.post('/api/enrichment/overlap', json={'genes': ['Col1a1', 'Col1a2'], 'libraries': [{'source': 'msigdb', 'id': 'missing'}]})
    assert r.status_code == 400 and 'not cached' in r.json()['detail']
    r = c.post('/api/enrichment/overlap', json={'genes': ['Col1a1', 'Col1a2']})
    assert r.status_code == 400


def test_gsea_route_runs_as_task():
    _install()
    c = TestClient(app)
    body = {'ranking': {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest'},
            'libraries': [{'source': 'msigdb', 'id': 'toy'}], 'n_perm': 50, 'min_set_size': 2}
    r = c.post('/api/enrichment/gsea', json=body)
    assert r.status_code == 202, r.text
    s = _poll(c, r.json()['task_id'])
    assert s['status'] == 'completed', s
    res = s['result']
    assert res['kind'] == 'gsea' and res['results'] and res['ranking']['genes'][0] in ('Col1a1', 'Col1a2', 'Col3a1')
    bad = dict(body, ranking={'kind': 'diffexp', 'obs_column': 'grp', 'group': 'zzz'})
    assert c.post('/api/enrichment/gsea', json=bad).status_code == 400
    ghost = dict(body, ranking={'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'cell_subset': 'ghost'})
    assert c.post('/api/enrichment/gsea', json=ghost).status_code == 404
```

- [ ] **Step 2: Run to verify failure** — 404 on the routes.

- [ ] **Step 3: Implement** the routes

```python
# ---------------------------------------------------------------------------
# Gene-set enrichment
# ---------------------------------------------------------------------------

class EnrichmentLibraryRef(BaseModel):
    source: str
    id: str
    species: str | None = None


class EnrichmentInlineSet(BaseModel):
    name: str
    genes: list[str]
    genes_down: list[str] | None = None


class OverlapEnrichmentRequest(BaseModel):
    genes: list[str]
    name: str | None = None
    libraries: list[EnrichmentLibraryRef] | None = None
    sets: list[EnrichmentInlineSet] | None = None
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    min_set_size: int = 5
    max_set_size: int = 500
    min_overlap: int = 2
    key: str | None = None


def _gene_subset_arg(spec):
    return spec.model_dump() if isinstance(spec, GeneSubsetSpec) else spec


@router.post("/enrichment/overlap")
def overlap_enrichment(request: OverlapEnrichmentRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_overlap_enrichment(
            request.genes, name=request.name,
            libraries=[l.model_dump() for l in request.libraries] if request.libraries else None,
            sets=[s.model_dump() for s in request.sets] if request.sets else None,
            gene_subset=_gene_subset_arg(request.gene_subset),
            min_set_size=request.min_set_size, max_set_size=request.max_set_size,
            min_overlap=request.min_overlap, key=request.key)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class GseaRankingSpec(BaseModel):
    kind: str = 'diffexp'
    obs_column: str | None = None
    group: str | None = None
    reference: str = 'rest'
    method: str = 'wilcoxon'
    metric: str = 'score'
    cell_subset: str | None = None
    component: int | None = None
    genes: list[str] | None = None
    scores: list[float] | None = None


class GseaRequest(BaseModel):
    ranking: GseaRankingSpec
    libraries: list[EnrichmentLibraryRef] | None = None
    sets: list[EnrichmentInlineSet] | None = None
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    n_perm: int = 1000
    min_set_size: int = 15
    max_set_size: int = 500
    weight: float = 1.0
    seed: int = 0
    key: str | None = None


@router.post("/enrichment/gsea", status_code=202)
def run_gsea(request: GseaRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_gsea(
            request.ranking.model_dump(),
            libraries=[l.model_dump() for l in request.libraries] if request.libraries else None,
            sets=[s.model_dump() for s in request.sets] if request.sets else None,
            gene_subset=_gene_subset_arg(request.gene_subset),
            n_perm=request.n_perm, min_set_size=request.min_set_size,
            max_set_size=request.max_set_size, weight=request.weight, seed=request.seed,
            key=request.key)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    task_id = task_manager.submit(compute_fn, apply_fn)
    return {"task_id": task_id, "status": "running"}


@router.get("/enrichment/results")
def list_enrichment_results(dataset: str | None = Query(None)):
    return {"results": get_adaptor(dataset).get_enrichment_results()}


@router.get("/enrichment/results/{key}")
def get_enrichment_result(key: str, dataset: str | None = Query(None)):
    try:
        return get_adaptor(dataset).get_enrichment_result(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/enrichment/results/{key}")
def delete_enrichment_result(key: str, dataset: str | None = Query(None)):
    try:
        return get_adaptor(dataset).delete_enrichment_result(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

`GeneSubsetSpec` is defined at routes.py ~L2208, *after* this insertion point — either place the enrichment block after it or reference it via a forward-compatible spot (simplest: put the enrichment block right after the `GeneSubsetSpec` definition). Note that pydantic tries union members in order, so `str | list[str] | GeneSubsetSpec` accepts the dict form last; the test pins it.

Config:

```yaml
# ---------------------------------------------------------------------------
# Gene-set enrichment modal (overlap / ORA and preranked GSEA).
# ---------------------------------------------------------------------------
enrichment:
  min_set_size: 5                   # ORA: drop sets smaller than this in the universe
  max_set_size: 500
  min_overlap: 2                    # ORA rows below this are hidden by default
  gsea_min_set_size: 15
  gsea_max_set_size: 500
  n_perm: 1000                      # gene permutations; p floors at 1/(n_perm+1)
  weight: 1.0                       # running-sum exponent (0 = classic KS)
  padj_cutoff: 0.05                 # results table default filter
```

- [ ] **Step 4: Run** the three enrichment test files and then the full suite: `python -m pytest tests -q -x` (≈1400 tests). Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/api/routes.py backend/xcell/config.yaml backend/tests/test_enrichment_routes.py && git commit -m "feat(enrichment): /api/enrichment routes and config defaults"
```

---

### Task 6: Frontend pure helpers + store + category

**Files:**
- Create: `frontend/src/lib/enrichment.ts`, `frontend/src/lib/enrichment.test.ts`
- Modify: `frontend/src/store.ts` (types near L95-160, state near L840/L1093/L1401/L2165), `frontend/src/components/GenePanel.tsx:26-35` (CATEGORY_ORDER + icon), `frontend/src/components/MultiContourModal.tsx:32`, `frontend/src/components/HeatmapConfigModal.tsx:20` (CATEGORY_NAMES).
- Also: `frontend/src/hooks/useData.ts` (API helpers + types).

**Interfaces:**
- Produces in `lib/enrichment.ts`:

```ts
export interface OraRow { name: string; library: string; description: string; url: string; n_set: number; n_overlap: number; expected: number; fold_enrichment: number; odds_ratio: number; pval: number; padj: number; genes: string[]; below_min_overlap: boolean }
export interface GseaRow { name: string; library: string; description: string; url: string; n_set: number; es: number; nes: number; pval: number; padj: number; leading_edge: string[]; n_leading_edge: number; curve: [number, number][] | null }
export interface OraResult { key: string; kind: 'ora'; label: string; created_at: string; query: { name: string; n_input: number; n_in_universe: number; genes_missing: string[] }; universe_size: number; gene_subset_type: string; n_sets_input: number; n_sets_tested: number; n_significant: number; results: OraRow[]; params: Record<string, unknown> }
export interface GseaResult { key: string; kind: 'gsea'; label: string; created_at: string; ranking: { kind: string; label: string; n_ranked: number; genes: string[]; scores: number[] }; universe_size: number; gene_subset_type: string; n_sets_input: number; n_sets_tested: number; n_significant: number; n_perm: number; results: GseaRow[]; params: Record<string, unknown> }
export type EnrichmentResult = OraResult | GseaResult
export interface CachedLibrary { source: string; id: string; name: string; species: string; n_sets: number; version?: string | null }
export type EnrichmentSource = { kind: 'ora'; name?: string; genes?: string[] } | { kind: 'gsea' }

export function formatP(p: number, floor?: number): string
export function curvePath(curve: [number, number][], width: number, height: number, nRanked: number): { d: string; zeroY: number; yMax: number }
export function hitTicks(curve: [number, number][]): number[]
export function filterRows<T extends { name: string; library: string; padj: number }>(rows: T[], opts: { padjMax: number | null; query: string; hideBelowMinOverlap?: boolean }): T[]
export function resultsToGeneSets(result: EnrichmentResult, opts: { padjMax: number | null; topN: number }): { name: string; genes: string[] }[]
export function libraryGroups(cached: CachedLibrary[]): { source: string; libraries: CachedLibrary[] }[]
export function rowsToTsv(result: EnrichmentResult): string
export function metricStripBins(scores: number[], nBins: number): number[]   // mean score per bin, for the strip under the curve
```

- Store: `enrichmentSource: EnrichmentSource | null`, `setEnrichmentSource(src)`; `'enrichment'` in `GeneSetCategoryType`, `createDefaultCategories` entry `{type:'enrichment', name:'Enrichment', expanded:true, folders:[], geneSets:[]}`.
- `useData.ts`: `runOverlapEnrichment(body, slot?) => Promise<OraResult>`, `startGsea(body, slot?) => Promise<{task_id: string}>`, `fetchEnrichmentResults(slot?)`, `fetchEnrichmentResult(key, slot?)`, `deleteEnrichmentResult(key, slot?)`, `fetchCachedLibraries(slot?) => Promise<CachedLibrary[]>` (reads `GET /gene_set_sources` and returns `cached`).

- [ ] **Step 1: Write the failing tests** (`lib/enrichment.test.ts`)

```ts
import { describe, it, expect } from 'vitest'
import { formatP, curvePath, hitTicks, filterRows, resultsToGeneSets, libraryGroups, rowsToTsv, metricStripBins, type OraResult, type GseaResult } from './enrichment'

const ora = (over: Partial<OraResult> = {}): OraResult => ({
  key: 'ora_q', kind: 'ora', label: 'Overlap: q', created_at: 't',
  query: { name: 'q', n_input: 3, n_in_universe: 3, genes_missing: [] },
  universe_size: 100, gene_subset_type: 'all', n_sets_input: 2, n_sets_tested: 2, n_significant: 1,
  results: [
    { name: 'A', library: 'L', description: '', url: '', n_set: 10, n_overlap: 3, expected: 0.3, fold_enrichment: 10, odds_ratio: 20, pval: 1e-4, padj: 2e-4, genes: ['x', 'y', 'z'], below_min_overlap: false },
    { name: 'B', library: 'M', description: '', url: '', n_set: 10, n_overlap: 1, expected: 0.3, fold_enrichment: 3, odds_ratio: 3, pval: 0.3, padj: 0.3, genes: ['x'], below_min_overlap: true },
  ], params: {}, ...over,
})

const gsea = (): GseaResult => ({
  key: 'gsea_k', kind: 'gsea', label: 'GSEA: k', created_at: 't',
  ranking: { kind: 'diffexp', label: 'grp: a vs rest', n_ranked: 10, genes: ['g0','g1','g2','g3','g4','g5','g6','g7','g8','g9'], scores: [5,4,3,2,1,-1,-2,-3,-4,-5] },
  universe_size: 10, gene_subset_type: 'all', n_sets_input: 1, n_sets_tested: 2, n_significant: 1, n_perm: 100,
  results: [
    { name: 'TOP', library: 'L', description: '', url: '', n_set: 2, es: 0.8, nes: 1.9, pval: 1 / 101, padj: 0.02, leading_edge: ['g0', 'g1'], n_leading_edge: 2, curve: [[0, 0], [0, -0.0], [0, 0.5], [1, 0.5], [1, 1.0], [9, 0]] },
    { name: 'DOWN', library: 'L', description: '', url: '', n_set: 2, es: -0.7, nes: -1.5, pval: 0.2, padj: 0.2, leading_edge: ['g8', 'g9'], n_leading_edge: 2, curve: null },
  ], params: {},
})

describe('formatP', () => {
  it('formats large, small and floored values', () => {
    expect(formatP(0.032)).toBe('0.032')
    expect(formatP(1.234e-7)).toBe('1.2e-7')
    expect(formatP(1 / 101, 1 / 101)).toBe('<0.0099')
    expect(formatP(1)).toBe('1.0')
  })
})

describe('curvePath', () => {
  it('maps hits to x and running score to y around a zero line', () => {
    const { d, zeroY, yMax } = curvePath(gsea().results[0].curve!, 100, 50, 10)
    expect(yMax).toBe(1)
    expect(zeroY).toBe(25)
    expect(d.startsWith('M0,25')).toBe(true)
    expect(d).toContain('L100,25')      // last vertex at n-1 maps to full width
  })
  it('hitTicks lists one x per hit', () => {
    expect(hitTicks(gsea().results[0].curve!)).toEqual([0, 1])
  })
})

describe('filterRows', () => {
  it('applies padj, text and min-overlap filters', () => {
    const rows = ora().results
    expect(filterRows(rows, { padjMax: 0.05, query: '' }).map((r) => r.name)).toEqual(['A'])
    expect(filterRows(rows, { padjMax: null, query: 'm' }).map((r) => r.name)).toEqual(['B'])
    expect(filterRows(rows, { padjMax: null, query: '', hideBelowMinOverlap: true }).map((r) => r.name)).toEqual(['A'])
  })
})

describe('resultsToGeneSets', () => {
  it('uses overlap genes for ORA and leading edge for GSEA, capped and filtered', () => {
    expect(resultsToGeneSets(ora(), { padjMax: 0.05, topN: 50 })).toEqual([{ name: 'A (3/10)', genes: ['x', 'y', 'z'] }])
    expect(resultsToGeneSets(gsea(), { padjMax: null, topN: 1 })).toEqual([{ name: 'TOP (NES 1.9)', genes: ['g0', 'g1'] }])
  })
})

describe('libraryGroups + rowsToTsv + metricStripBins', () => {
  it('groups cached libraries by source keeping order', () => {
    const g = libraryGroups([
      { source: 'msigdb', id: 'a', name: 'A', species: 'mouse', n_sets: 1 },
      { source: 'go', id: 'b', name: 'B', species: 'mouse', n_sets: 2 },
      { source: 'msigdb', id: 'c', name: 'C', species: 'human', n_sets: 3 },
    ])
    expect(g.map((x) => x.source)).toEqual(['msigdb', 'go'])
    expect(g[0].libraries.map((l) => l.id)).toEqual(['a', 'c'])
  })
  it('writes a header and one row per result', () => {
    const tsv = rowsToTsv(ora())
    const lines = tsv.trim().split('\n')
    expect(lines[0].split('\t')).toEqual(['name', 'library', 'n_set', 'n_overlap', 'expected', 'fold_enrichment', 'odds_ratio', 'pval', 'padj', 'genes'])
    expect(lines).toHaveLength(3)
    expect(rowsToTsv(gsea()).split('\n')[0].split('\t')).toEqual(['name', 'library', 'n_set', 'es', 'nes', 'pval', 'padj', 'leading_edge'])
  })
  it('bins scores by mean', () => {
    expect(metricStripBins([4, 2, -2, -4], 2)).toEqual([3, -3])
    expect(metricStripBins([], 3)).toEqual([0, 0, 0])
  })
})
```

- [ ] **Step 2: Run** `cd /private/tmp/claude-501/xcell-wt/frontend && npx vitest run src/lib/enrichment.test.ts` — expected: module not found.

- [ ] **Step 3: Implement `lib/enrichment.ts`**

```ts
/** Pure helpers for the Enrichment modal: formatting, SVG geometry, filters,
 *  and turning result rows into Gene Panel sets. Kept free of React so the
 *  numerical/formatting contract is unit-testable. */

export interface OraRow { /* as in Interfaces above */ }
export interface GseaRow { /* as above */ }
export interface OraResult { /* as above */ }
export interface GseaResult { /* as above */ }
export type EnrichmentResult = OraResult | GseaResult
export interface CachedLibrary { source: string; id: string; name: string; species: string; n_sets: number; version?: string | null }
export type EnrichmentSource = { kind: 'ora'; name?: string; genes?: string[] } | { kind: 'gsea' }

export function formatP(p: number, floor?: number): string {
  if (floor !== undefined && p <= floor) return `<${floor.toPrecision(2)}`
  if (p >= 0.001) return p >= 1 ? '1.0' : p.toPrecision(2).replace(/0+$/, '').replace(/\.$/, '')
  return p.toExponential(1).replace('e-', 'e-').replace(/\.0e/, 'e')
}
```
(Write `formatP` so the test's expectations hold exactly: `0.032` → `'0.032'`; `1.234e-7` → `'1.2e-7'`; `1` → `'1.0'`; floored → `'<0.0099'`. Adjust the implementation, not the test.)

```ts
export function curvePath(curve: [number, number][], width: number, height: number, nRanked: number) {
  const yMax = Math.max(1e-9, ...curve.map(([, y]) => Math.abs(y)))
  const zeroY = height / 2
  const sx = (x: number) => (nRanked > 1 ? (x / (nRanked - 1)) * width : 0)
  const sy = (y: number) => zeroY - (y / yMax) * (height / 2)
  const d = curve.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${sx(x)},${sy(y)}`).join(' ')
  return { d, zeroY, yMax }
}

export function hitTicks(curve: [number, number][]): number[] {
  // Interior vertices come in (pre, post) pairs at the same x; one tick per pair.
  const xs: number[] = []
  for (let i = 1; i + 1 < curve.length; i += 2) xs.push(curve[i][0])
  return xs
}

export function filterRows<T extends { name: string; library: string; padj: number }>(
  rows: T[], opts: { padjMax: number | null; query: string; hideBelowMinOverlap?: boolean },
): T[] {
  const q = opts.query.trim().toLowerCase()
  return rows.filter((r) => {
    if (opts.padjMax !== null && r.padj > opts.padjMax) return false
    if (opts.hideBelowMinOverlap && (r as unknown as { below_min_overlap?: boolean }).below_min_overlap) return false
    if (q && !r.name.toLowerCase().includes(q) && !r.library.toLowerCase().includes(q)) return false
    return true
  })
}

export function resultsToGeneSets(result: EnrichmentResult, opts: { padjMax: number | null; topN: number }) {
  if (result.kind === 'ora') {
    return filterRows(result.results, { padjMax: opts.padjMax, query: '', hideBelowMinOverlap: true })
      .slice(0, opts.topN).filter((r) => r.genes.length > 0)
      .map((r) => ({ name: `${r.name} (${r.n_overlap}/${r.n_set})`, genes: r.genes }))
  }
  return filterRows(result.results, { padjMax: opts.padjMax, query: '' })
    .slice(0, opts.topN).filter((r) => r.leading_edge.length > 0)
    .map((r) => ({ name: `${r.name} (NES ${r.nes.toFixed(1)})`, genes: r.leading_edge }))
}

export function libraryGroups(cached: CachedLibrary[]) {
  const order: string[] = []
  const by = new Map<string, CachedLibrary[]>()
  for (const lib of cached) {
    if (!by.has(lib.source)) { by.set(lib.source, []); order.push(lib.source) }
    by.get(lib.source)!.push(lib)
  }
  return order.map((source) => ({ source, libraries: by.get(source)! }))
}

export function rowsToTsv(result: EnrichmentResult): string {
  if (result.kind === 'ora') {
    const head = ['name', 'library', 'n_set', 'n_overlap', 'expected', 'fold_enrichment', 'odds_ratio', 'pval', 'padj', 'genes']
    const rows = result.results.map((r) => [r.name, r.library, r.n_set, r.n_overlap, r.expected.toFixed(3), r.fold_enrichment.toFixed(3), r.odds_ratio.toFixed(3), r.pval.toExponential(3), r.padj.toExponential(3), r.genes.join(',')])
    return [head, ...rows].map((r) => r.join('\t')).join('\n') + '\n'
  }
  const head = ['name', 'library', 'n_set', 'es', 'nes', 'pval', 'padj', 'leading_edge']
  const rows = result.results.map((r) => [r.name, r.library, r.n_set, r.es.toFixed(4), r.nes.toFixed(3), r.pval.toExponential(3), r.padj.toExponential(3), r.leading_edge.join(',')])
  return [head, ...rows].map((r) => r.join('\t')).join('\n') + '\n'
}

export function metricStripBins(scores: number[], nBins: number): number[] {
  const out = new Array<number>(nBins).fill(0)
  if (scores.length === 0 || nBins <= 0) return out
  const per = scores.length / nBins
  for (let b = 0; b < nBins; b++) {
    const lo = Math.floor(b * per), hi = Math.max(lo + 1, Math.floor((b + 1) * per))
    const slice = scores.slice(lo, hi)
    out[b] = slice.reduce((s, v) => s + v, 0) / slice.length
  }
  return out
}
```

- [ ] **Step 4: Store + category + useData**

`store.ts`: add `'enrichment'` to `GeneSetCategoryType`; add the `enrichment` entry in `createDefaultCategories` (after `line_association`); add `enrichmentSource: EnrichmentSource | null` to the state interface, `setEnrichmentSource` to actions, `enrichmentSource: null` to initial state, and `setEnrichmentSource: (src) => set({ enrichmentSource: src })` next to `setDecomposeSource`. Import the type from `./lib/enrichment`.

`GenePanel.tsx:26`: append `'enrichment'` to `CATEGORY_ORDER`; add `enrichment: '🎯'` to `CATEGORY_ICONS`. `MultiContourModal.tsx:32`: append `'enrichment'`. `HeatmapConfigModal.tsx:20`: add `enrichment: 'Enrichment'`. Then `npx tsc --noEmit` will point at any remaining exhaustive record.

`useData.ts` (near `runMarkerGenes`):

```ts
import type { OraResult, GseaResult, CachedLibrary } from '../lib/enrichment'

export async function runOverlapEnrichment(body: Record<string, unknown>, slot?: DatasetSlot): Promise<OraResult> {
  return fetchJson<OraResult>(appendDataset(`${API_BASE}/enrichment/overlap`, slot), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}
export async function startGsea(body: Record<string, unknown>, slot?: DatasetSlot): Promise<{ task_id: string }> {
  return fetchJson(appendDataset(`${API_BASE}/enrichment/gsea`, slot), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}
export async function fetchEnrichmentResults(slot?: DatasetSlot): Promise<{ key: string; kind: 'ora' | 'gsea'; label: string; n_sets_tested: number; n_significant: number; created_at: string }[]> {
  return (await fetchJson<{ results: never[] }>(appendDataset(`${API_BASE}/enrichment/results`, slot))).results
}
export async function fetchEnrichmentResult(key: string, slot?: DatasetSlot): Promise<OraResult | GseaResult> {
  return fetchJson(appendDataset(`${API_BASE}/enrichment/results/${encodeURIComponent(key)}`, slot))
}
export async function deleteEnrichmentResult(key: string, slot?: DatasetSlot): Promise<void> {
  await fetchJson(appendDataset(`${API_BASE}/enrichment/results/${encodeURIComponent(key)}`, slot), { method: 'DELETE' })
}
export async function fetchCachedLibraries(): Promise<CachedLibrary[]> {
  const a = await fetchJson<{ cached?: CachedLibrary[] }>(`${API_BASE}/gene_set_sources`)
  return a.cached ?? []
}
```

- [ ] **Step 5: Run** `npx vitest run src/lib/enrichment.test.ts` (PASS) and `npx tsc --noEmit` (clean).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/enrichment.ts frontend/src/lib/enrichment.test.ts frontend/src/store.ts frontend/src/hooks/useData.ts frontend/src/components/GenePanel.tsx frontend/src/components/MultiContourModal.tsx frontend/src/components/HeatmapConfigModal.tsx && git commit -m "feat(enrichment): frontend helpers, store source, Enrichment gene-set category, API client"
```

---

### Task 7: `EnrichmentModal.tsx` + entry points

**Files:**
- Create: `frontend/src/components/EnrichmentModal.tsx`
- Modify: `frontend/src/App.tsx` (mount next to `<DecomposeGeneSetModal />`), `frontend/src/components/GenePanel.tsx` (per-set menu after "Decompose into programs…", toolbar menu after "Gene set library…"), `frontend/src/components/ScanpyModal.tsx` (a custom function `enrichment` next to `gene_nmf_meta` at ~L445 and the launcher button at ~L2881).

**Interfaces:**
- Consumes: everything from Task 6; `pollTask`, `useObsSummaries`, `appendDataset`, `cfgDefault`, `addFolderToCategory`, `addScanpyAction`, `setSelectedGene`, `cellSubsets`, `standaloneSvg` / `downloadText` from `lib/svgExport.ts`.

- [ ] **Step 1: Build the modal.** Structure (all inline styles, copy the `styles` object from `MarkerGenesModal.tsx` and widen `modal.width` to `900px`):

```tsx
export function EnrichmentModal() {
  const source = useStore((s) => s.enrichmentSource)
  const setSource = useStore((s) => s.setEnrichmentSource)
  const activeSlot = useStore((s) => s.activeSlot)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const cellSubsets = useStore((s) => s.cellSubsets)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const setSelectedGene = useStore((s) => s.setSelectedGene)
  const { summaries } = useObsSummaries()

  const [tab, setTab] = useState<'ora' | 'gsea'>('ora')
  // Libraries
  const [cached, setCached] = useState<CachedLibrary[]>([])
  const [chosenLibs, setChosenLibs] = useState<Set<string>>(new Set())   // `${source}/${id}`
  const [ownFolder, setOwnFolder] = useState<string>('')                  // `${category}/${folderId}` or '' = none
  const [booleanColumns, setBooleanColumns] = useState<{ name: string; n_true: number }[]>([])
  const [universeCol, setUniverseCol] = useState('')
  // ORA
  const [querySetId, setQuerySetId] = useState<string>('')   // '' = pasted
  const [pasted, setPasted] = useState('')
  const [minOverlap, setMinOverlap] = useState(cfgDefault(['enrichment', 'min_overlap'], 2))
  const [oraMin, setOraMin] = useState(cfgDefault(['enrichment', 'min_set_size'], 5))
  const [oraMax, setOraMax] = useState(cfgDefault(['enrichment', 'max_set_size'], 500))
  // GSEA
  const [rankKind, setRankKind] = useState<'diffexp' | 'pca'>('diffexp')
  const [obsColumn, setObsColumn] = useState(''); const [group, setGroup] = useState(''); const [reference, setReference] = useState('rest')
  const [method, setMethod] = useState<'wilcoxon' | 't-test'>('wilcoxon'); const [metric, setMetric] = useState<'score' | 'log2fc'>('score')
  const [cellSubset, setCellSubset] = useState(''); const [component, setComponent] = useState(1)
  const [nPerm, setNPerm] = useState(cfgDefault(['enrichment', 'n_perm'], 1000))
  const [gMin, setGMin] = useState(cfgDefault(['enrichment', 'gsea_min_set_size'], 15)); const [gMax, setGMax] = useState(cfgDefault(['enrichment', 'gsea_max_set_size'], 500))
  const [weight, setWeight] = useState(cfgDefault(['enrichment', 'weight'], 1)); const [seed, setSeed] = useState(0)
  // run / results
  const [phase, setPhase] = useState<'config' | 'running' | 'results'>('config')
  const [progress, setProgress] = useState({ frac: 0, message: '' })
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EnrichmentResult | null>(null)
  const [previous, setPrevious] = useState<{ key: string; label: string }[]>([])
  const [padjMax, setPadjMax] = useState<number | null>(cfgDefault(['enrichment', 'padj_cutoff'], 0.05))
  const [query, setQuery] = useState(''); const [hideBelow, setHideBelow] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  ...
  if (!source) return null   // AFTER all hooks (global modals never unmount)
```

Effects on open (`source` becomes non-null): reset `phase/result/error/saved/expanded`, set `tab` from `source.kind`, preselect `querySetId` when `source.kind === 'ora'` and `source.genes` given (store the genes in a ref and use them directly rather than searching by name), fetch `fetchCachedLibraries()`, `GET /api/var/boolean_columns` (as MarkerGenesModal does), and `fetchEnrichmentResults(activeSlot)`. Default `obsColumn` to the first categorical summary with 2–200 categories; default `group` to its first category whenever `obsColumn` changes.

Run handlers:

```tsx
const libraries = [...chosenLibs].map((k) => { const [source, id] = k.split('/'); const lib = cached.find((c) => c.source === source && c.id === id); return { source, id, species: lib?.species } })
const inlineSets = ownFolder ? folderSets(ownFolder).map((gs) => ({ name: gs.name, genes: gs.genes, genes_down: gs.genesDown ?? null })) : []
const geneSubset = universeCol || null

const runOra = async () => {
  const genes = querySetId ? (flatSets.find((g) => g.id === querySetId)?.genes ?? []) : pasted.split(/[\s,;]+/).filter(Boolean)
  const name = querySetId ? flatSets.find((g) => g.id === querySetId)?.name : 'pasted list'
  const body = { genes, name, libraries, sets: inlineSets, gene_subset: geneSubset, min_set_size: oraMin, max_set_size: oraMax, min_overlap: minOverlap }
  setPhase('running'); setError(null)
  try {
    const res = await runOverlapEnrichment(body, activeSlot)
    setResult(res); setPhase('results'); setSaved(false)
    addScanpyAction({ action: 'enrichment_ora', params: body, result: { key: res.key, n_sets_tested: res.n_sets_tested, n_significant: res.n_significant }, timestamp: new Date().toISOString() })
    setPrevious(await fetchEnrichmentResults(activeSlot))
  } catch (e) { setError((e as Error).message); setPhase('config') }
}

const runGsea = async () => {
  const ranking = rankKind === 'diffexp'
    ? { kind: 'diffexp', obs_column: obsColumn, group, reference, method, metric, cell_subset: cellSubset || null }
    : { kind: 'pca', component: component - 1, cell_subset: cellSubset || null }
  const body = { ranking, libraries, sets: inlineSets, gene_subset: geneSubset, n_perm: nPerm, min_set_size: gMin, max_set_size: gMax, weight, seed }
  setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Starting…' })
  try {
    const { task_id } = await startGsea(body, activeSlot)
    const task = await pollTask(task_id, activeSlot, (s) => setProgress({ frac: s.progress ?? 0, message: s.message ?? '' }))
    if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
    const res = task.result as unknown as GseaResult
    setResult(res); setPhase('results'); setSaved(false)
    addScanpyAction({ action: 'enrichment_gsea', params: body, result: { key: res.key, n_sets_tested: res.n_sets_tested, n_significant: res.n_significant }, timestamp: new Date().toISOString() })
    setPrevious(await fetchEnrichmentResults(activeSlot))
  } catch (e) { setError((e as Error).message); setPhase('config') }
}
```

Results rendering: summary line; filter row (`padj ≤` number input with an "all" checkbox, search box, ORA-only "hide < min overlap"); "Previous runs" select + × delete; table with columns `Set · Library · Size · Overlap|NES · p · padj · bar`. Bar: a `<div>` whose width is `Math.min(1, -log10(padj)/10) * 100%` (ORA, accent colour) or `|NES|/3` (GSEA, accent for positive, alert `#e94560` for negative). Row click toggles `expanded`; expanded row shows description + url link, gene chips (`onClick={() => setSelectedGene(g)}`), and for GSEA rows with `curve` a `<GseaCurve>`:

```tsx
function GseaCurve({ row, ranking }: { row: GseaRow; ranking: GseaResult['ranking'] }) {
  const W = 520, H = 140, STRIP = 14
  const { d, zeroY } = curvePath(row.curve!, W, H, ranking.n_ranked)
  const ticks = hitTicks(row.curve!)
  const bins = metricStripBins(ranking.scores, 100)
  const bmax = Math.max(1e-9, ...bins.map(Math.abs))
  return (
    <svg data-gsea-curve width={W} height={H + STRIP + 26} style={{ display: 'block' }}>
      <line x1={0} x2={W} y1={zeroY} y2={zeroY} stroke="#0f3460" />
      <path d={d} fill="none" stroke="#4ecdc4" strokeWidth={1.5} />
      {ticks.map((x, i) => <line key={i} x1={(x / (ranking.n_ranked - 1)) * W} x2={(x / (ranking.n_ranked - 1)) * W} y1={H + 1} y2={H + 9} stroke="#eee" />)}
      {bins.map((v, i) => <rect key={i} x={(i / 100) * W} y={H + 12} width={W / 100 + 0.5} height={STRIP} fill={v >= 0 ? `rgba(233,69,96,${Math.abs(v) / bmax})` : `rgba(78,205,196,${Math.abs(v) / bmax})`} />)}
      <text x={0} y={H + STRIP + 24} fill="#888" fontSize={10}>rank 1 (highest score)</text>
      <text x={W} y={H + STRIP + 24} fill="#888" fontSize={10} textAnchor="end">rank {ranking.n_ranked}</text>
      <text x={4} y={12} fill="#aaa" fontSize={10}>ES {row.es.toFixed(2)} · NES {row.nes.toFixed(2)} · padj {formatP(row.padj)}</text>
    </svg>
  )
}
```

Buttons: **Add to gene sets** → `addFolderToCategory('enrichment', result.key, resultsToGeneSets(result, { padjMax, topN: 50 }))`, disabled when empty, shows "Added" after. **Copy TSV** → `navigator.clipboard.writeText(rowsToTsv(result))`. **Download SVG** (only when a GSEA row is expanded) → `downloadText(`${result.key}_${row.name}.svg`, standaloneSvg(svgEl.outerHTML, { background: '#16213e' }), 'image/svg+xml')`. **Back** (to config) and **Close** (`setSource(null)`).

Escape closes; backdrop click closes; card `stopPropagation`. A note under the GSEA params: "p-values are empirical over gene permutations and floor at 1/(n_perm+1)."

- [ ] **Step 2: Entry points**

`App.tsx`: `<EnrichmentModal />` after `<DecomposeGeneSetModal />`.

`GenePanel.tsx` per-set menu (after "Decompose into programs…"):
```tsx
{
  label: 'Enrichment (overlap)…',
  onClick: () => setEnrichmentSource({ kind: 'ora', name: geneSet.name, genes: geneSet.genes }),
  disabled: geneSet.genes.length < 2,
  tooltip: 'Hypergeometric overlap with cached gene-set libraries',
},
```
Toolbar OverflowMenu (after "Gene set library…"):
```tsx
{ label: 'Enrichment analysis…', onClick: () => setEnrichmentSource({ kind: 'ora' }), tooltip: 'Overlap (ORA) or preranked GSEA against cached libraries' },
```
`ScanpyModal.tsx` catalogue (next to `gene_nmf_meta`):
```ts
enrichment: {
  label: 'Gene-set Enrichment',
  description: 'Overlap (hypergeometric) enrichment of a gene list, or preranked GSEA of a group-vs-rest / group-vs-group contrast or a PCA loading, against the gene-set libraries cached by the Gene set library (MSigDB, GO, Enrichr, OmniPath, MGI) and your own sets. Opens the Enrichment tool.',
  prerequisites: [], custom: true, params: [],
},
```
and the launcher branch next to `gene_nmf_meta`'s: `Open Enrichment tool…` → `setEnrichmentSource({ kind: 'gsea' }); setScanpyModalOpen(false)`. Check whether ScanpyModal's function list has a category/section map that also needs the new id (grep `gene_nmf_meta` in that file for every occurrence and mirror each).

- [ ] **Step 3: Typecheck + unit tests** — `npx tsc --noEmit` clean; `npx vitest run` all green.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/EnrichmentModal.tsx frontend/src/App.tsx frontend/src/components/GenePanel.tsx frontend/src/components/ScanpyModal.tsx && git commit -m "feat(enrichment): Enrichment modal (overlap + GSEA) with running-score plot and entry points"
```

---

### Task 8: Browser verification on the isolated stack + docs

**Files:**
- Modify: `CHANGELOG.md` (new entry at top), `README.md` (one paragraph under the analysis features list, if such a list exists), `CLAUDE.md` in the *main repo* is local-only — add a line to its analysis-module list (`enrichment.py`) at merge time.

- [ ] **Step 1: Stack.** Follow the isolated recipe: from the worktree, `cp -Rc "<main>/frontend/node_modules" frontend/node_modules` (done in Task 6 if not earlier); backend `cd backend && "<main>/.pixi/envs/default/bin/python" -m uvicorn xcell.main:app --port 8100` with `XDG_CACHE_HOME` left as-is so the real cached libraries (`~/.cache/xcell/gene_set_sources`, mouse Hallmark, GO, …) are visible; frontend `XCELL_BACKEND=http://127.0.0.1:8100 npx vite --port 5273 --strictPort`. Load the bundled toy data (`POST /api/load` with the toy path used by `xcell.main` on startup — check `grep -n toy backend/xcell/main.py`). If the toy genes are not mouse symbols, use the dev fallback: seed an inline "My gene sets" folder instead of a library, or point `config.yaml gene_set_sources.cache_dir` at a scratch cache with a synthetic library built from the toy's own genes (`PUT /api/gene_sets` for the query set; a small `sets` JSON via `gss.save_library`).

- [ ] **Step 2: Drive.** In the browser (Playwright MCP or CDP): open Genes ▸ menu ▸ "Enrichment analysis…"; tick a library; pick a query set; Run; assert the table has rows and `padj` sorted ascending; click a row → gene chips visible; "Add to gene sets" → Gene Panel shows an **Enrichment** category with a folder named after the key. Switch to GSEA; pick a Leiden column/group; Run; watch the progress bar; expand the top row → `svg[data-gsea-curve]` exists with a `path`; "Download SVG" produces a file. Open the analysis record and confirm `enrichment_ora` / `enrichment_gsea` entries. Patch `window.fetch` to capture the GSEA body and confirm `gene_subset` and `cell_subset` are what the controls show. Reload the page, reopen the modal, choose the run from "Previous runs" — the table returns.

- [ ] **Step 3: Fix what the browser finds**, each fix with a test where the failure was testable, then re-run `pytest tests -q` and `npx tsc --noEmit` / `npx vitest run`.

- [ ] **Step 4: Docs + commit.** CHANGELOG entry ("Gene-set enrichment: overlap (hypergeometric) and preranked GSEA against cached libraries…"), README paragraph. Delete `.playwright-mcp/` and screenshots from the worktree.

```bash
git add CHANGELOG.md README.md && git commit -m "docs: gene-set enrichment"
```

- [ ] **Step 5: Stop the isolated servers** (`kill` the uvicorn on :8100 and vite on :5273). Then the merge check: `curl -s localhost:8000/api/schema | head -c 200` and `/api/scanpy/history` — a real dataset with steps means **report the branch and ask before merging** (memory: check-live-dataset-before-merging); toy data → `git checkout main && git merge --no-ff feat/enrichment`.

---

## Self-review notes

- Spec coverage: statistics (T1–T2), adaptor/universe/storage/rankings (T3–T4), routes/config (T5), helpers/store/category/API client (T6), modal/plot/entry points/previous runs/add-to-sets/TSV/SVG (T7), browser + docs + merge gate (T8). "Copy TSV" and "Download SVG" live in T7. Species selection for a library is carried through `species` on the ref (T6 `libraries` builder).
- Types: `ResolvedSet` keys match between T1 and T2; `leading_edge` is indices out of `preranked_gsea` and names after `apply_fn`; `curve` is `[[x, y], …]` in Python and `[number, number][]` in TS; `below_min_overlap` present on ORA rows only and read via a cast in `filterRows`.
- Review Focus tests: (1) `test_resolve_symbols_exact_then_case_insensitive_dedup` + `test_overlap_run_resolves_case_and_stores_json`; (2) `test_gsea_edge_cases…` `'zero'`; (3) same test `'all'`; (4) `test_overlap_keys_never_overwrite_and_delete`; (5) `test_overlap_route_accepts_dict_gene_subset…`.
