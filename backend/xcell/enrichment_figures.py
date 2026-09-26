"""Assemble enrichment results across contrasts into figure data.

Pure: dicts and arrays in, JSON-safe dicts out. The heatmap follows the
PySingleCellNet convention (Cahan lab): cells above the padj threshold are 0,
rows are the union of each column's top-N positive and negative sets, and
NES (or a signed −log10 padj) is the value.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

LOGP_CAP = 300.0
VALUE_LABELS = {
    'nes': 'NES',
    'signed_logp': 'signed −log10 padj',
    'es': 'ES',
    'fold_enrichment': 'log2 fold enrichment',
}


def signed_logp(padj: float, sign: float, *, cap: float = LOGP_CAP) -> float:
    """−log10(padj) carrying the direction; capped so padj = 0 stays finite."""
    if padj is None or not math.isfinite(padj) or padj >= 1:
        return 0.0
    v = cap if padj <= 0 else min(cap, -math.log10(padj))
    return v if sign >= 0 else -v


def _sign_of(row: dict[str, Any]) -> float:
    if 'nes' in row:
        return 1.0 if row['nes'] >= 0 else -1.0
    return 1.0 if row.get('fold_enrichment', 1.0) > 1.0 else -1.0


def row_value(row: dict[str, Any], value: str) -> float:
    if value == 'signed_logp':
        return signed_logp(row.get('padj', 1.0), _sign_of(row))
    if value == 'nes':
        return float(row.get('nes', 0.0))
    if value == 'es':
        return float(row.get('es', 0.0))
    if value == 'fold_enrichment':
        fe = float(row.get('fold_enrichment', 0.0))
        return math.log2(fe) if fe > 0 else 0.0
    raise ValueError(f"Unknown value '{value}'; expected nes, signed_logp, es or fold_enrichment")


def _select(results_by_col: dict[str, dict[str, Any]], *, value: str, padj_max: float, top_n: int,
            direction: str):
    """Per column: {name: (value, padj)} after thresholding, plus the row union
    (each column's top-N positive and top-N negative) and per-set metadata."""
    per_col: dict[str, dict[str, tuple[float, float]]] = {}
    chosen: list[str] = []
    meta: dict[str, dict[str, Any]] = {}
    for col, res in results_by_col.items():
        vals: dict[str, tuple[float, float]] = {}
        for r in res.get('results') or []:
            v = row_value(r, value)
            padj = float(r.get('padj', 1.0))
            if padj > padj_max:
                v = 0.0
            vals[r['name']] = (v, padj)
            meta.setdefault(r['name'], {'library': r.get('library', ''), 'n_set': int(r.get('n_set', 0))})
        per_col[col] = vals
        sig = [(n, v) for n, (v, _) in vals.items() if v != 0.0]
        if direction in ('both', 'up'):
            for n, _ in sorted([x for x in sig if x[1] > 0], key=lambda x: -x[1])[:top_n]:
                if n not in chosen:
                    chosen.append(n)
        if direction in ('both', 'down'):
            for n, _ in sorted([x for x in sig if x[1] < 0], key=lambda x: x[1])[:top_n]:
                if n not in chosen:
                    chosen.append(n)
    return per_col, chosen, meta


def collapse_sets(names: list[str], members: dict[str, set[str]], *, jaccard: float) -> dict[str, list[str]]:
    """Greedy: walk ``names`` in priority order; a set joins the first earlier
    representative whose member Jaccard index is >= ``jaccard``."""
    reps: dict[str, list[str]] = {}
    for n in names:
        mine = members.get(n)
        home = None
        if mine:
            for rep in reps:
                other = members.get(rep)
                if not other:
                    continue
                union = len(mine | other)
                if union and len(mine & other) / union >= jaccard:
                    home = rep
                    break
        if home is None:
            reps[n] = []
        else:
            reps[home].append(n)
    return reps


def _order_rows(names: list[str], values: list[list[float]], *, row_order: str) -> list[int]:
    if row_order == 'input' or len(names) < 2:
        return list(range(len(names)))
    arr = np.asarray(values, dtype=float)
    if row_order == 'cluster':
        from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: PLC0415
        return [int(i) for i in leaves_list(linkage(arr, method='average', metric='euclidean'))]
    if row_order != 'peak':
        raise ValueError("row_order must be 'peak', 'cluster' or 'input'")
    # by the column of the largest |value|; positives before negatives within a column; then by |value|
    peak = np.abs(arr).argmax(axis=1)
    peakv = arr[np.arange(len(names)), peak]
    return sorted(range(len(names)), key=lambda i: (int(peak[i]), 0 if peakv[i] > 0 else 1, -abs(peakv[i])))


def _order_cols(cols: list[str], values: list[list[float]], *, col_order: str) -> list[int]:
    if col_order not in ('input', 'cluster'):
        raise ValueError("col_order must be 'input' or 'cluster'")
    if col_order != 'cluster' or len(cols) < 3 or not values:
        return list(range(len(cols)))
    from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: PLC0415
    arr = np.asarray(values, dtype=float).T
    return [int(i) for i in leaves_list(linkage(arr, method='average', metric='euclidean'))]


def assemble_matrix(results_by_col: dict[str, dict[str, Any]], *, value: str, padj_max: float, top_n: int,
                    direction: str, collapse_jaccard: float | None, row_order: str, col_order: str,
                    members: dict[str, set[str]] | None) -> dict[str, Any]:
    if direction not in ('both', 'up', 'down'):
        raise ValueError("direction must be 'both', 'up' or 'down'")
    per_col, chosen, meta = _select(results_by_col, value=value, padj_max=padj_max, top_n=top_n, direction=direction)
    cols = list(results_by_col)
    n_total = len(meta)
    note: str | None = None
    absorbed: dict[str, list[str]] = {}
    n_collapsed = 0
    if chosen and collapse_jaccard is not None:
        if members is None:
            note = 'Row collapse skipped: member genes unavailable'
        else:
            best = {n: max(abs(per_col[c].get(n, (0.0, 1.0))[0]) for c in cols) for n in chosen}
            reps = collapse_sets(sorted(chosen, key=lambda n: -best[n]), members, jaccard=collapse_jaccard)
            absorbed = reps
            chosen = [n for n in chosen if n in reps]
            n_collapsed = sum(len(v) for v in reps.values())
    if not chosen:
        return {'rows': [], 'cols': [{'label': c} for c in cols], 'values': [], 'padj': [],
                'value_label': VALUE_LABELS[value], 'n_rows_total': n_total, 'n_collapsed': 0,
                'note': note or f'No gene set passes padj ≤ {padj_max} in any column'}
    values = [[per_col[c].get(n, (0.0, 1.0))[0] for c in cols] for n in chosen]
    padj = [[per_col[c].get(n, (0.0, 1.0))[1] for c in cols] for n in chosen]
    ri = _order_rows(chosen, values, row_order=row_order)
    ci = _order_cols(cols, values, col_order=col_order)
    rows = [{'name': chosen[i], 'library': meta[chosen[i]]['library'], 'members': absorbed.get(chosen[i], []),
             'n_set': meta[chosen[i]]['n_set']} for i in ri]
    return {
        'rows': rows, 'cols': [{'label': cols[j]} for j in ci],
        'values': [[float(values[i][j]) for j in ci] for i in ri],
        'padj': [[float(padj[i][j]) for j in ci] for i in ri],
        'value_label': VALUE_LABELS[value], 'n_rows_total': n_total, 'n_collapsed': n_collapsed, 'note': note,
    }


def force_layout(n: int, edges: list[tuple[int, int]], *, weights: list[float] | None = None, seed: int = 0,
                 iterations: int = 200, heavy: set[int] | None = None) -> np.ndarray:
    """Fruchterman–Reingold on the unit square; deterministic for a seed.

    ``heavy`` nodes (the groups) move a third as far per step, so the picture
    reads as gene sets arranging themselves around the clusters.
    """
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.array([[0.5, 0.5]])
    pos = rng.random((n, 2))
    k = 1.0 / math.sqrt(n)
    w = np.asarray(weights if weights is not None else [1.0] * len(edges), dtype=float)
    src = np.asarray([e[0] for e in edges], dtype=int)
    dst = np.asarray([e[1] for e in edges], dtype=int)
    mass = np.ones(n)
    for h in heavy or ():
        mass[h] = 3.0
    t = 0.1
    dt = t / max(1, iterations)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        disp = ((k * k / dist)[:, :, None] * delta / dist[:, :, None]).sum(axis=1)
        if len(src):
            d = pos[src] - pos[dst]
            dl = np.linalg.norm(d, axis=1) + 1e-9
            att = (dl * dl / k * w)[:, None] * d / dl[:, None]
            np.add.at(disp, src, -att)
            np.add.at(disp, dst, att)
        length = np.linalg.norm(disp, axis=1) + 1e-9
        step = disp / length[:, None] * np.minimum(length, t)[:, None] / mass[:, None]
        pos = np.clip(pos + step, 0.0, 1.0)
        t = max(t - dt, 0.005)
    return pos


def bipartite_layout(groups: list[int], sets: list[int], edges: list[tuple[int, int]]) -> np.ndarray:
    """Groups on x=0, sets on x=1, sets ordered by the barycentre of their group neighbours."""
    n = len(groups) + len(sets)
    pos = np.zeros((n, 2))
    gy = {g: (i + 0.5) / max(1, len(groups)) for i, g in enumerate(groups)}
    for g, y in gy.items():
        pos[g] = (0.0, y)
    nbr: dict[int, list[float]] = {s: [] for s in sets}
    for a, b in edges:
        if a in gy and b in nbr:
            nbr[b].append(gy[a])
        elif b in gy and a in nbr:
            nbr[a].append(gy[b])
    order = sorted(sets, key=lambda s: (float(np.mean(nbr[s])) if nbr[s] else 0.5, s))
    for i, s in enumerate(order):
        pos[s] = (1.0, (i + 0.5) / max(1, len(sets)))
    return pos


def assemble_network(results_by_col: dict[str, dict[str, Any]], *, value: str, padj_max: float, top_n: int,
                     direction: str, set_edge_jaccard: float | None, layout: str, seed: int,
                     members: dict[str, set[str]] | None) -> dict[str, Any]:
    if direction not in ('both', 'up', 'down'):
        raise ValueError("direction must be 'both', 'up' or 'down'")
    if layout not in ('force', 'bipartite'):
        raise ValueError("layout must be 'force' or 'bipartite'")
    per_col, chosen, meta = _select(results_by_col, value=value, padj_max=padj_max, top_n=top_n, direction=direction)
    cols = list(results_by_col)
    note: str | None = None
    ids: list[str] = [f'group:{c}' for c in cols] + [f'set:{n}' for n in chosen]
    index = {i: k for k, i in enumerate(ids)}
    edges: list[dict[str, Any]] = []
    for c in cols:
        for n in chosen:
            v, p = per_col[c].get(n, (0.0, 1.0))
            if v != 0.0:
                edges.append({'source': f'group:{c}', 'target': f'set:{n}', 'kind': 'enrichment',
                              'value': float(v), 'padj': float(p)})
    if set_edge_jaccard is not None:
        if members is None:
            note = 'Overlap edges skipped: member genes unavailable'
        else:
            for i, a in enumerate(chosen):
                for b in chosen[i + 1:]:
                    ma, mb = members.get(a), members.get(b)
                    if not ma or not mb:
                        continue
                    j = len(ma & mb) / len(ma | mb)
                    if j >= set_edge_jaccard:
                        edges.append({'source': f'set:{a}', 'target': f'set:{b}', 'kind': 'overlap',
                                      'value': float(j), 'padj': None})
    if not chosen:
        note = note or f'No gene set passes padj ≤ {padj_max} in any column'
    pairs = [(index[e['source']], index[e['target']]) for e in edges]
    weights = [abs(e['value']) if e['kind'] == 'enrichment' else e['value'] for e in edges]
    if layout == 'bipartite':
        pos = bipartite_layout(list(range(len(cols))), list(range(len(cols), len(ids))), pairs)
    else:
        pos = force_layout(len(ids), pairs, weights=weights, seed=seed, heavy=set(range(len(cols))))
    degree = {i: 0 for i in ids}
    for e in edges:
        degree[e['source']] += 1
        degree[e['target']] += 1
    nodes = []
    for k, i in enumerate(ids):
        kind = 'group' if k < len(cols) else 'set'
        name = i.split(':', 1)[1]
        nodes.append({
            'id': i, 'kind': kind, 'label': name, 'x': float(pos[k, 0]), 'y': float(pos[k, 1]),
            'size': float(meta.get(name, {}).get('n_set', 0)) if kind == 'set' else 0.0,
            'degree': degree[i], 'library': meta.get(name, {}).get('library') if kind == 'set' else None,
        })
    bounds = ([float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max())]
              if len(ids) else [0.0, 0.0, 1.0, 1.0])
    return {'nodes': nodes, 'edges': edges, 'bounds': bounds, 'note': note}
