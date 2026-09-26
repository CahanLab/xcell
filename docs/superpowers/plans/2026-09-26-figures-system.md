# Figures System (Part B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Declarative, persisted, reproducible figures: a figure record (kind + inputs + params + provenance) stored in `uns`, rendered in the browser from a backend `figure_data` table, editable and exportable from a Figures tab, logged in the analysis record — with enrichment heatmap and enrichment network as the first two kinds.

**Architecture:** `enrichment_figures.py` (pure) assembles the heatmap matrix / network graph from per-contrast enrichment results. The adaptor owns the registry (`uns['xcell_figures']`, JSON strings, `xcell_figures_seq` id counter), validates inputs, resolves provenance step indices, logs `figure_create/update/delete`, and dispatches `figure_data`. Thin routes. Frontend: `lib/figures.ts` (types, PNG rasteriser), `lib/figureSchemas.ts` (param forms), `FiguresView.tsx` (gallery / renderer / params / export), two SVG renderers, store fields per dataset, Enrichment modal buttons.

**Tech Stack:** numpy/scipy; React + SVG; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-09-26-figures-and-batch-enrichment-design.md` (Part B)

## Global Constraints

- Figure records are JSON strings under `uns['xcell_figures'][id]`; ids `fig_<n>` from `uns['xcell_figures_seq']`, never reused.
- `figure_data` never mutates state; a `params_override` preview is not logged.
- Every `_log_action` name gets a `codegen.REGISTRY` entry in the same commit.
- Rendering is browser SVG (`data-figure-svg` root attribute); PNG = rasterised SVG. No matplotlib.
- Numbers crossing the API are finite (`signed_logp` floors at 300).
- Frontend inline styles, house palette. Worktree `/private/tmp/claude-501/xcell-wt`; test commands as in Plan 1.

## Review Focus

1. A figure whose input result was deleted must still list, and its data call must fail with a clear 409-style message rather than a stack trace — Task 2 test + Task 3 route test.
2. `padj = 0` (possible from ORA on tiny universes) must not make `signed_logp` infinite — Task 1 test.
3. Row collapse must never merge two sets whose Jaccard is exactly below threshold, and must keep the representative's own row values — Task 1 test.
4. The force layout must be deterministic for a seed and finite for a graph with isolated nodes — Task 1 test.
5. Editing params in the tab with an unsaved preview, then switching figures, must not write the preview into the other figure — Task 6 browser check.

---

### Task 1: `enrichment_figures.py` — matrix, collapse, network, layout

**Files:** Create `backend/xcell/enrichment_figures.py`; test `backend/tests/test_enrichment_figures.py`.

**Interfaces (produces):**
```python
signed_logp(padj: float, sign: float, *, cap: float = 300.0) -> float
row_value(row: dict, value: str) -> float          # 'nes' | 'signed_logp' | 'es' | 'fold_enrichment'
collapse_sets(names: list[str], members: dict[str, set[str]], *, jaccard: float) -> dict[str, list[str]]
    # names in priority order → {representative: [absorbed names...]} (representative first-come)
assemble_matrix(results_by_col: dict[str, dict], *, value: str, padj_max: float, top_n: int,
                direction: str, collapse_jaccard: float | None, row_order: str, col_order: str,
                members: dict[str, set[str]] | None) -> dict
    # results_by_col: {col_label: single enrichment result dict (kind ora|gsea)}
    # → {rows: [{name, library, members: [str], n_set}], cols: [{label}], values: [[float]],
    #    padj: [[float]], value_label: str, n_rows_total: int, n_collapsed: int, note: str | None}
assemble_network(results_by_col, *, value, padj_max, top_n, direction, set_edge_jaccard, layout, seed,
                 members) -> {nodes: [{id, kind, label, x, y, size, degree, library}], edges: [{source, target, kind, value, padj}], bounds: [xmin, ymin, xmax, ymax], note}
force_layout(n: int, edges: list[tuple[int, int]], *, weights: list[float] | None, seed: int, iterations: int = 200, heavy: set[int] | None) -> np.ndarray  # (n, 2)
bipartite_layout(groups: list[int], sets: list[int], edges) -> np.ndarray
```
`members` (name → gene set in universe) comes from the adaptor (it re-resolves the library sets); when `None`, collapse and overlap edges are skipped with a note.

- [ ] **Step 1: Failing tests**

```python
"""Pure tests for enrichment figure assembly: matrix, collapse, network, layout."""
import numpy as np
import pytest

from xcell import enrichment_figures as ef


def _row(name, nes, padj, library='L'):
    return {'name': name, 'library': library, 'description': '', 'url': '', 'n_set': 10,
            'es': nes / 2, 'nes': nes, 'pval': padj / 2, 'padj': padj, 'leading_edge': [], 'n_leading_edge': 0, 'curve': None}


def _gsea(rows):
    return {'kind': 'gsea', 'results': rows, 'n_perm': 100}


def _ora_row(name, k, K, padj):
    return {'name': name, 'library': 'L', 'description': '', 'url': '', 'n_set': K, 'n_overlap': k, 'expected': 1.0,
            'fold_enrichment': k / 1.0, 'odds_ratio': 2.0, 'pval': padj / 2, 'padj': padj, 'genes': [], 'below_min_overlap': False}


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
    assert all(np.isfinite(np.asarray(cl['values'])).all() for _ in [0])


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
```

- [ ] **Step 2: Run** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
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


def signed_logp(padj: float, sign: float, *, cap: float = LOGP_CAP) -> float:
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


VALUE_LABELS = {'nes': 'NES', 'signed_logp': 'signed −log10 padj', 'es': 'ES', 'fold_enrichment': 'log2 fold enrichment'}


def _select(results_by_col, *, value, padj_max, top_n, direction):
    """Per column: {name: (value, padj)} after thresholding, and the row union."""
    per_col: dict[str, dict[str, tuple[float, float]]] = {}
    chosen: list[str] = []
    meta: dict[str, dict[str, Any]] = {}
    for col, res in results_by_col.items():
        rows = res.get('results') or []
        vals: dict[str, tuple[float, float]] = {}
        for r in rows:
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
                inter = len(mine & other)
                union = len(mine | other)
                if union and inter / union >= jaccard:
                    home = rep
                    break
        if home is None:
            reps[n] = []
        else:
            reps[home].append(n)
    return reps


def _order_rows(names, values, *, row_order):
    if row_order == 'input' or len(names) < 2:
        return list(range(len(names)))
    arr = np.asarray(values, dtype=float)
    if row_order == 'cluster':
        from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: PLC0415
        return list(leaves_list(linkage(arr, method='average', metric='euclidean')))
    # 'peak': by the column of the largest |value|, positives first within a column, then by value
    peak = np.abs(arr).argmax(axis=1)
    peakv = arr[np.arange(len(names)), peak]
    return sorted(range(len(names)), key=lambda i: (peak[i], 0 if peakv[i] > 0 else 1, -abs(peakv[i])))


def _order_cols(cols, values, *, col_order):
    if col_order != 'cluster' or len(cols) < 3:
        return list(range(len(cols)))
    from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: PLC0415
    arr = np.asarray(values, dtype=float).T
    return list(leaves_list(linkage(arr, method='average', metric='euclidean')))


def assemble_matrix(results_by_col: dict[str, dict[str, Any]], *, value: str, padj_max: float, top_n: int,
                    direction: str, collapse_jaccard: float | None, row_order: str, col_order: str,
                    members: dict[str, set[str]] | None) -> dict[str, Any]:
    if direction not in ('both', 'up', 'down'):
        raise ValueError("direction must be 'both', 'up' or 'down'")
    per_col, chosen, meta = _select(results_by_col, value=value, padj_max=padj_max, top_n=top_n, direction=direction)
    cols = list(results_by_col)
    n_total = len(meta)
    note = None
    absorbed: dict[str, list[str]] = {}
    n_collapsed = 0
    if chosen and collapse_jaccard is not None:
        if members is None:
            note = 'Row collapse skipped: member genes unavailable'
        else:
            # priority: best |value| across columns
            best = {n: max(abs(per_col[c].get(n, (0.0, 1.0))[0]) for c in cols) for n in chosen}
            order = sorted(chosen, key=lambda n: -best[n])
            reps = collapse_sets(order, members, jaccard=collapse_jaccard)
            absorbed = reps
            chosen = [n for n in chosen if n in reps]
            n_collapsed = sum(len(v) for v in reps.values())
    values = [[per_col[c].get(n, (0.0, 1.0))[0] for c in cols] for n in chosen]
    padj = [[per_col[c].get(n, (0.0, 1.0))[1] for c in cols] for n in chosen]
    if not chosen:
        note = note or f'No gene set passes padj ≤ {padj_max} in any column'
        return {'rows': [], 'cols': [{'label': c} for c in cols], 'values': [], 'padj': [],
                'value_label': VALUE_LABELS[value], 'n_rows_total': n_total, 'n_collapsed': 0, 'note': note}
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
    """Fruchterman–Reingold on the unit square; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    if n == 0:
        return pos
    if n == 1:
        return np.array([[0.5, 0.5]])
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
        rep = (k * k / dist)[:, :, None] * delta / dist[:, :, None]
        disp = rep.sum(axis=1)
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
    order = sorted(sets, key=lambda s: (np.mean(nbr[s]) if nbr[s] else 0.5, s))
    for i, s in enumerate(order):
        pos[s] = (1.0, (i + 0.5) / max(1, len(sets)))
    return pos


def assemble_network(results_by_col: dict[str, dict[str, Any]], *, value: str, padj_max: float, top_n: int,
                     direction: str, set_edge_jaccard: float | None, layout: str, seed: int,
                     members: dict[str, set[str]] | None) -> dict[str, Any]:
    per_col, chosen, meta = _select(results_by_col, value=value, padj_max=padj_max, top_n=top_n, direction=direction)
    cols = list(results_by_col)
    note = None
    ids: list[str] = [f'group:{c}' for c in cols] + [f'set:{n}' for n in chosen]
    index = {i: k for k, i in enumerate(ids)}
    edges: list[dict[str, Any]] = []
    for c in cols:
        for n in chosen:
            v, p = per_col[c].get(n, (0.0, 1.0))
            if v != 0.0:
                edges.append({'source': f'group:{c}', 'target': f'set:{n}', 'kind': 'enrichment', 'value': float(v), 'padj': float(p)})
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
                        edges.append({'source': f'set:{a}', 'target': f'set:{b}', 'kind': 'overlap', 'value': float(j), 'padj': None})
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
        nodes.append({'id': i, 'kind': kind, 'label': name, 'x': float(pos[k, 0]), 'y': float(pos[k, 1]),
                      'size': float(meta.get(name, {}).get('n_set', 0)) if kind == 'set' else 0.0,
                      'degree': degree[i], 'library': meta.get(name, {}).get('library') if kind == 'set' else None})
    bounds = [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max())] if len(ids) else [0, 0, 1, 1]
    return {'nodes': nodes, 'edges': edges, 'bounds': bounds, 'note': note}
```

- [ ] **Step 4: Run** the tests; adjust ordering test expectations only if the `peak` rule is genuinely ambiguous (record a ruling). **Step 5: Commit** `feat(figures): enrichment figure assembly (matrix, collapse, network, layouts)`.

---

### Task 2: Adaptor figure registry, provenance, `figure_data`, attach

**Files:** `backend/xcell/adaptor.py` (new `figures` section after the enrichment section), `backend/xcell/analysis_record.py` (`Figure.figure_id`), `backend/xcell/codegen.py`, `backend/xcell/config.yaml` (`figures:`), test `backend/tests/test_figures_adaptor.py`.

**Interfaces (produces):**
```python
FIGURES_UNS_KEY = 'xcell_figures'; FIGURES_SEQ_KEY = 'xcell_figures_seq'
FIGURE_KINDS = ('enrichment_heatmap', 'enrichment_network', 'composition_barplot', 'expression_heatmap')
create_figure(kind, *, title=None, caption='', inputs, params=None) -> dict   # ValueError on bad kind/inputs
update_figure(id, *, title=None, caption=None, params=None) -> dict          # KeyError unknown id
delete_figure(id) -> {'deleted': id}
list_figures() -> [ {id, kind, title, created_at, updated_at, inputs, n_provenance_steps} ]   # newest first
get_figure(id) -> dict
figure_data(id, params_override=None) -> dict   # raises FigureInputMissing(ValueError subclass) when an input key is gone
attach_figure_to_record(id, png_b64, caption=None) -> {'record_figure_id', 'step_index'}
```
Defaults per kind from `config.yaml figures.<kind>` (spec values). `composition_barplot` / `expression_heatmap` data dispatch is implemented in Plan 3; in this plan `figure_data` raises `ValueError('kind not renderable yet')` for them and `create_figure` still accepts them.

- [ ] **Step 1: Failing tests** — cover: create validates kind (`ValueError` listing kinds) and inputs (`enrichment_keys` must exist; obs column for barplot), fills defaults, allocates `fig_1`, `fig_2`, and after deleting `fig_2` the next is `fig_3`; provenance: run a `prepare_gsea_batch` first, then create a heatmap on the collection → `provenance.steps` contains the index of the `enrichment_gsea_batch` step and `provenance.created_step` is the `figure_create` step's index; `update_figure` merges params, bumps `updated_at`, logs `figure_update` with only the changed fields; `figure_data` on the heatmap returns rows/cols matching `enrichment_figures.assemble_matrix` on the members, honours `params_override` without logging or persisting; `figure_data` on a network returns nodes; deleting the input collection then `figure_data` → `FigureInputMissing` with the key in the message while `list_figures` still lists it; `attach_figure_to_record` adds a record figure with `figure_id == id` linked to `created_step`; uns round-trip (`json.loads` of each entry). Use the batch fixture from `test_enrichment_batch.py` (copy `_adata`, LIB, cache fixture).

- [ ] **Step 2: Run RED. Step 3: Implement.** Key pieces:

```python
    class FigureInputMissing(ValueError):
        """A stored figure references a result that no longer exists (→ 409)."""

    def _figure_store(self) -> dict[str, str]: ...   # like _enrichment_store
    def _next_figure_id(self) -> str:
        n = int(self.adata.uns.get(self.FIGURES_SEQ_KEY, 0)) + 1
        self.adata.uns[self.FIGURES_SEQ_KEY] = n
        return f'fig_{n}'

    def _figure_defaults(self, kind) -> dict:   # from config: user_config.get_user_config().get('figures', {}).get(kind, {})
    def _validate_figure_inputs(self, kind, inputs) -> dict:
        # enrichment kinds: inputs['enrichment_keys'] non-empty list, each in the store; expand a collection's members lazily at data time
        # composition_barplot: column_a/column_b in obs; expression_heatmap: gene_sets non-empty
    def _figure_provenance_steps(self, kind, inputs) -> list[int]:
        keys = set(inputs.get('enrichment_keys') or [])
        out = []
        for step in self.analysis_record.steps:
            r = step.result or {}
            if r.get('key') in keys or keys & set((r.get('members') or {}).values()) or (
                    kind == 'composition_barplot' and step.params.get('key_added') in (inputs.get('column_a'), inputs.get('column_b'))):
                out.append(step.index)
        return out
```
`create_figure` builds the record, stores it, logs `figure_create` with `{kind, title, inputs, params}` and result `{id}`, then sets `provenance.created_step = len(self.analysis_record.steps) - 1` and re-stores (the step index is only known after logging). `figure_data` for enrichment kinds: load each key; a collection expands to its `member_results` labelled by group; a single result is labelled by `ranking.label` / `query.name`; `members` = re-resolve the library sets named in the first result's `params.libraries` + inline `sets` against the universe (`_enrichment_sets` + `en.resolve_sets(..., min_size=1, max_size=10**9, directional='split' if gsea else 'union')`, names → sets of gene names; a missing library → `members=None` with the note). Then `ef.assemble_matrix(...)` / `ef.assemble_network(...)` with `{**params, **(params_override or {})}` filtered to the kind's known params.

`attach_figure_to_record`: `fig = self.analysis_record.add_figure(png_b64, caption=caption or record['title'], step_index=record['provenance'].get('created_step'))`, set `fig.figure_id = id`. Add `figure_id: str | None = None` to `analysis_record.Figure` (+ `to_dict`/`from_dict`), and include it in `_record_payload`'s figure entries and `notebook_export.figure_payloads`' neighbour metadata (check how captions reach the notebook and add `figure_id` beside the caption).

Codegen entries: `figure_create` (XCELL, `_direct('create_figure', ('kind','title','caption','inputs','params'))` plus a second code line `xa.figure_data(<id>)` — write a custom `code=` that returns `[_xcall(...), f"{ADATA}_fig = {ADAPTOR}.figure_data({_lit(r['id'])})"]`; check how `_xcall`/`ADAPTOR` are named in codegen.py and mirror), `figure_update` (`_direct('update_figure', ('id','title','caption','params'))` — the logged params must include `id`), `figure_delete` (`_direct('delete_figure', ('id',))`).

Config:
```yaml
figures:
  enrichment_heatmap: {value: nes, padj_max: 0.05, top_n: 5, direction: both, collapse_jaccard: 0.5, row_order: peak, col_order: input, colormap: rdbu, vmax: null, show_values: false, label_max_chars: 40, cell_size: 14}
  enrichment_network: {value: nes, padj_max: 0.05, top_n: 5, direction: both, set_edge_jaccard: 0.3, layout: force, seed: 0, node_size_by: degree, edge_width_by: value, label_max_chars: 30, colormap: rdbu}
  composition_barplot: {order: category, share_of: null, normalize: true, min_cells: 0, show_values: false}
  expression_heatmap: {cell_ordering: category, gene_ordering: as_provided, aggregate_gene_sets: false, n_bins: 50}
```
(ORA collections override `value` to `signed_logp` at create time when the caller passes no `value`.)

- [ ] **Step 4:** tests green incl. `test_codegen.py`. **Step 5: Commit** `feat(figures): figure registry with provenance, figure_data, record attachment`.

---

### Task 3: Routes

**Files:** `backend/xcell/api/routes.py`; test `backend/tests/test_figures_routes.py`.

Routes: `GET /figures` → `{figures: [...]}`; `POST /figures` body `{kind, title?, caption?, inputs, params?}` → record; `GET /figures/{id}`; `PUT /figures/{id}` body `{title?, caption?, params?}`; `DELETE /figures/{id}`; `POST /figures/{id}/data` body `{params?}` → data (409 on `FigureInputMissing`, 400 on other ValueError); `POST /figures/{id}/attach` body `{png_b64, caption?}` (validate base64 like `/record/figure`). Tests: full CRUD; data for heatmap + network after a batch run via the route; 409 after deleting the input; attach returns ids and `GET /record` shows the figure with `figure_id`.

**Commit** `feat(figures): /api/figures routes`.

---

### Task 4: Frontend foundations — types, schemas, PNG export, store, API client

**Files:** Create `frontend/src/lib/figures.ts`, `frontend/src/lib/figureSchemas.ts`, tests `figures.test.ts`, `figureSchemas.test.ts`; modify `store.ts`, `hooks/useData.ts`.

- `lib/figures.ts`: `FigureKind`, `FigureRecord {id, kind, title, caption, created_at, updated_at, inputs, params, provenance: {steps: number[], created_step: number | null, source: string}}`, `FigureSummary`, `EnrichmentHeatmapData`, `EnrichmentNetworkData`; `defaultTitle(kind, inputs, labelsByKey)`; `svgToPngBlob(svg: SVGSVGElement, scale: number, background: string): Promise<Blob>` (serialise → `Image` → canvas → `toBlob`; reject if `toBlob` yields null); `blobToBase64(blob)`; `figureFilename(record, ext)`; `divergingColor(v, vmax, colormap)` → CSS colour (three small 5-stop ramps: `rdbu`, `prgn`, `roma`); `truncate(label, n)`.
- `lib/figureSchemas.ts`: `ParamField = {name, label, type: 'number'|'select'|'bool'|'nullable-number', min?, max?, step?, options?, help?}`; `FIGURE_SCHEMAS: Record<FigureKind, ParamField[]>` for the two enrichment kinds (all params in the spec); test asserts every schema field name is in the config defaults returned by a fixture copy of the YAML section (keep a literal copy of defaults in the test) and that `coerceParams(kind, raw)` rounds ints / nulls blanks.
- Store (per dataset): `figures: FigureSummary[]`, `activeFigureId: string | null`, `figuresVersion: number`; actions `setFigures`, `setActiveFigureId`, `refreshFigures()` (bumps version). Add `'figures'` to `CenterPanelView`, keep `'figure'` for the compositor.
- `useData.ts`: `fetchFigures`, `createFigure(body)`, `fetchFigure(id)`, `updateFigure(id, body)`, `deleteFigure(id)`, `fetchFigureData(id, paramsOverride?)` (surfaces 409 detail), `attachFigureToRecord(id, png_b64, caption?)`.

**Commit** `feat(figures): frontend types, schemas, PNG export, store, API client`.

---

### Task 5: FiguresView, renderers, tab, entry points

**Files:** Create `components/FiguresView.tsx`, `components/figures/EnrichmentHeatmapFigure.tsx`, `components/figures/EnrichmentNetworkFigure.tsx`, `components/figures/NewFigureModal.tsx`; modify `App.tsx` (tab "Figures" → `centerPanelView === 'figures'`, rename the compositor tab label to "Panels"), `EnrichmentModal.tsx` (enable the two buttons: create figure with `inputs: {enrichment_keys: [result.key]}` and default title, then `setActiveFigureId`, `refreshFigures`, `setCenterPanelView('figures')`, close the modal), `AnalysisRecordPanel.tsx` (figure entries with `figure_id` get an "open figure" link).

Renderers take `{data, params, title, width}` and expose the root `<svg data-figure-svg>` via a forwarded ref.
- Heatmap: left label column (truncate to `label_max_chars`, `(+n)` for collapsed members, tooltip lists them), top labels rotated −45°, cells `cell_size` px with `divergingColor(v, vmax ?? max|v|)`, optional value text (`show_values`), a legend bar (5 stops) with `value_label`, `note` rendered as text when rows are empty.
- Network: viewBox from `bounds` with 8% padding; overlap edges first (grey, dashed, width ∝ value), then enrichment edges (colour by sign via `divergingColor`, width ∝ |value| when `edge_width_by='value'`), group nodes as squares (accent fill, label bold), set nodes as circles (radius by `node_size_by`: `degree` → 4+2·deg, `n_set` → sqrt scale, `constant` → 6), labels truncated with a 2-px halo; legend.

`FiguresView`: three columns (rail 220 px, canvas flex, params 280 px). Rail: list from the store (refetched on `figuresVersion`), *New figure…* opens `NewFigureModal` (kind select limited to the two enrichment kinds in this plan; input = an enrichment result/collection select from `fetchEnrichmentResults`; title). Canvas: renderer at `width = min(container, 1400)`, wrapped in a div with `overflow: auto`. Params: `title`, `caption` inputs, the schema form, **Save** (PUT; disabled until dirty), **Revert**; preview fetch with the current form values debounced 300 ms; provenance box (steps from `record.provenance.steps` with labels fetched from `GET /record` once per open — show `#index action`). Toolbar: Export SVG, Export PNG (scale select 1–4), Attach to record (PNG 2× → attach; toast "attached to step N"), Duplicate (POST with same inputs/params, title + " copy"), Delete (confirm). Switching figures resets the form from the new record (Review Focus 5).

**Commit** `feat(figures): Figures tab with enrichment heatmap and network renderers`.

---

### Task 6: Browser verification + docs

Isolated stack; batch GSEA on the toy `cell_type` (300 perms, min set 3) → **Heatmap figure…** → Figures tab shows the figure with rows for TOY_* sets and two columns; edit `padj_max` to 0.5 → preview updates; Save → PUT seen; Export SVG → download; Export PNG → download named `<title>.png`; Attach to record → `GET /record` figure has `figure_id`; New figure… → network → renders nodes/edges; reload page → both figures listed and render; delete the batch result via the Enrichment modal → heatmap shows the 409 message, still listed. Check no console errors. Then CHANGELOG + README ("Figures" section). Stop servers, clean Playwright files, commit `docs: figures`.
