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

    def emit(name: str, members: list[Any], src: dict[str, Any]) -> None:
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
            'n_input': len(members),
            'indices': np.asarray(idx, dtype=np.int64),
        })

    for s in sets:
        if not isinstance(s, dict):
            continue
        n_input += 1
        name = str(s.get('name') or '')
        up = list(s.get('genes') or [])
        down = list(s.get('genes_down') or [])
        if directional == 'union':
            emit(name, up + down, s)
        else:
            emit(name, up, s)
            if down:
                emit(f'{name} (down)', down, s)
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
    """Hypergeometric over-representation of ``query_indices`` in each set.

    Sets below ``min_overlap`` are still tested and still count toward the BH
    correction (dropping them first would bias it); they are only flagged.
    """
    q = np.unique(np.asarray(query_indices, dtype=np.int64))
    n_query = int(q.size)
    records: list[dict[str, Any]] = []
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
    adj = bh_adjust(np.asarray([r['pval'] for r in records])) if records else np.array([])
    for r, a in zip(records, adj):
        r['padj'] = float(a)
    records.sort(key=lambda r: (r['pval'], r['name']))
    return records


# --- preranked GSEA ------------------------------------------------------------

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
    p_hit = np.zeros(w.shape, dtype=float)
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
    keep = np.flatnonzero(np.isfinite(scores))
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
            null_cache[k] = _es_from_positions(pos, ranked_w[pos], n)[0]
        return null_cache[k]

    n_sizes = len({pos.size for _, pos in sets})
    results: list[dict[str, Any]] = []
    for i, (s, pos) in enumerate(sets):
        k = int(pos.size)
        es_arr, peak, pre, post = _es_from_positions(pos[None, :], ranked_w[pos][None, :], n)
        es, j = float(es_arr[0]), int(peak[0])
        null = null_for(k)
        same = null[null > 0] if es > 0 else null[null < 0] if es < 0 else np.array([])
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
            report((i + 1) / len(sets), f'Testing gene sets ({i + 1}/{len(sets)}, {n_sizes} size classes)')

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
