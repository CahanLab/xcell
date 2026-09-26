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
