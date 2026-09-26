# Batch Enrichment (Part A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One run of GSEA (or marker-gene ORA) across every group of a categorical column, stored as per-group results plus a collection, reachable from Compare Cells and the Enrichment modal, ready to feed a figure.

**Architecture:** `prepare_gsea_batch` snapshots one throwaway AnnData and builds each group's ranking inside the task; per-group results reuse the single-run shape and key rule; a collection record (`kind: gsea_batch | ora_batch`) lists members. Routes and codegen mirror the single-run ones. The modal gains an "All groups" mode and a batch results view; MarkerGenes/DiffExp/Compare hand off into it.

**Tech Stack:** Python/numpy/scanpy; React; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-09-26-figures-and-batch-enrichment-design.md` (Part A)

## Global Constraints

- Per-group result keys: `gsea_<column>_<group>_vs_<reference>` / `ora_<column>_<group>_markers`; collection keys `gsea_<column>_batch` / `ora_<column>_batch`; all via `_store_enrichment` (suffix on collision, never overwrite).
- A group with < 2 cells on either side is skipped and reported in `skipped`; all skipped → `ValueError`.
- No new dependencies. JSON-safe outputs. Every new `_log_action` name gets a `codegen.REGISTRY` entry in the same commit (test_codegen enforces it).
- `gene_subset` keeps the full union type in request models.
- Worktree `/private/tmp/claude-501/xcell-wt` (branch `feat/figures`). Backend tests: `cd backend && "/Users/pcahan/Dropbox (Personal)/Code/xcell/.pixi/envs/dev/bin/python" -m pytest tests/<file> -q -p no:cacheprovider`. Frontend: `cd frontend && npx tsc --noEmit && npx vitest run`.

## Review Focus

1. A column with one group that has a single cell must still run for the others and name the skipped group — test in Task 1.
2. With exactly two groups and `reference='rest'`, the two contrasts are mirrors; both must be stored and both columns present — test in Task 1.
3. `get_enrichment_result(<collection key>)` must return `member_results` expanded so one GET feeds the figure — test in Task 1.
4. The ORA batch must store the marker lists it used (`markers`) so the input is self-contained after the marker modal's state is gone — test in Task 2.
5. The modal's batch mode must send `groups` as the checked subset when opened from Compare, and `null` (all) when opened from the Genes menu — verified in the browser in Task 5.

---

### Task 1: `prepare_gsea_batch` + collection listing/expansion

**Files:**
- Modify: `backend/xcell/adaptor.py` (enrichment section, after `prepare_gsea`)
- Test: `backend/tests/test_enrichment_batch.py`

**Interfaces:**
- Consumes: `_enrichment_sets`, `_resolve_gene_mask`, `en.resolve_sets`, `en.preranked_gsea`, `_store_enrichment`, `_subset_mask`.
- Produces: `prepare_gsea_batch(obs_column, *, groups=None, reference='rest', libraries=None, sets=None, gene_subset=None, method='wilcoxon', metric='score', cell_subset=None, n_perm=1000, min_set_size=15, max_set_size=500, weight=1.0, seed=0, key=None) -> (compute_fn, apply_fn)`; `apply_fn` returns the collection dict `{key, kind:'gsea_batch', label, created_at, obs_column, reference, groups, members: {group: key}, skipped: {group: reason}, n_perm, universe_size, gene_subset_type, n_sets_tested, n_significant, params}`. `get_enrichment_results()` summaries gain `n_groups` (None for single runs). `get_enrichment_result(key)` on a collection adds `member_results: {group: result}`.

- [ ] **Step 1: Failing tests**

```python
"""Batch enrichment across a column's groups: GSEA and marker-gene ORA."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Ptprc', 'Cd3e', 'Cd19', 'Hoxd13', 'Meis1'] + [f'G{i}' for i in range(24)]
LIB = [{'source': 'msigdb', 'id': 'toy', 'species': 'mouse'}]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({
        'source': 'msigdb', 'id': 'toy', 'name': 'Toy', 'species': 'mouse', 'version': '1', 'n_sets': 3,
        'sets': [{'name': 'COLLAGEN', 'genes': ['Col1a1', 'Col1a2', 'Col3a1', 'G0', 'G1']},
                 {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']},
                 {'name': 'HOX', 'genes': ['Hoxd13', 'Meis1', 'G4', 'G5', 'G6']}]})
    yield
    gss.set_cache_dir(None)


def _adata(n_per=(30, 30, 30, 1)):
    """Groups a (Col*), b (immune), c (Hox) and a one-cell group d."""
    rng = np.random.default_rng(0)
    labels = np.concatenate([[g] * n for g, n in zip('abcd', n_per)])
    n = labels.size
    lam = np.full((n, len(GENES)), 1.0)
    for g, hi in (('a', ('Col1a1', 'Col1a2', 'Col3a1')), ('b', ('Ptprc', 'Cd3e', 'Cd19')), ('c', ('Hoxd13', 'Meis1'))):
        for gene in hi:
            lam[labels == g, GENES.index(gene)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(labels)
    return ad


def _run(a, **kw):
    compute_fn, apply_fn = a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, n_perm=60, **kw)
    return apply_fn(compute_fn(lambda f, m: None))


def test_gsea_batch_stores_members_and_collection_and_skips_tiny_group():
    a = DataAdaptor('x.h5ad', adata=_adata())
    col = _run(a)
    assert col['kind'] == 'gsea_batch' and col['key'] == 'gsea_grp_batch'
    assert col['groups'] == ['a', 'b', 'c'] and col['skipped'] == {'d': 'fewer than 2 cells in group'}
    assert col['members'] == {'a': 'gsea_grp_a_vs_rest', 'b': 'gsea_grp_b_vs_rest', 'c': 'gsea_grp_c_vs_rest'}
    store = a.adata.uns['xcell_enrichment']
    assert set(col['members'].values()) <= set(store) and 'gsea_grp_batch' in store
    member = json.loads(store['gsea_grp_a_vs_rest'])
    assert member['kind'] == 'gsea' and {r['name']: r for r in member['results']}['COLLAGEN']['nes'] > 0
    assert a._action_history[-1]['action'] == 'enrichment_gsea_batch'
    assert a._action_history[-1]['result']['key'] == 'gsea_grp_batch'
    # listing + expansion
    summaries = {s['key']: s for s in a.get_enrichment_results()}
    assert summaries['gsea_grp_batch']['n_groups'] == 3 and summaries['gsea_grp_a_vs_rest']['n_groups'] is None
    full = a.get_enrichment_result('gsea_grp_batch')
    assert set(full['member_results']) == {'a', 'b', 'c'} and full['member_results']['b']['kind'] == 'gsea'


def test_gsea_batch_two_groups_are_mirrors_and_groups_subset():
    a = DataAdaptor('x.h5ad', adata=_adata((30, 30, 30, 5)))
    col = _run(a, groups=['a', 'b'])
    assert col['groups'] == ['a', 'b'] and col['skipped'] == {}
    ra = {r['name']: r for r in a.get_enrichment_result(col['members']['a'])['results']}
    rb = {r['name']: r for r in a.get_enrichment_result(col['members']['b'])['results']}
    assert ra['COLLAGEN']['nes'] > 0 and rb['COLLAGEN']['nes'] < 0
    # 'rest' for a means b,c,d (all cells not in a), so the mirror is only exact with groups=['a','b'] and reference='b'
    col2 = _run(a, groups=['a', 'b'], reference='b')
    assert col2['members'] == {'a': 'gsea_grp_a_vs_b'} and col2['groups'] == ['a']


def test_gsea_batch_validation():
    a = DataAdaptor('x.h5ad', adata=_adata((2, 1, 1, 1)))
    with pytest.raises(ValueError, match='every group'):
        a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, groups=['b', 'c'])
    with pytest.raises(ValueError, match='nope'):
        a.prepare_gsea_batch('nope', libraries=LIB, min_set_size=2)
    with pytest.raises(ValueError, match='Unknown group'):
        a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, groups=['zzz'])
```

- [ ] **Step 2: Run** — expected AttributeError `prepare_gsea_batch`.

- [ ] **Step 3: Implement** (append to the enrichment section)

```python
    def prepare_gsea_batch(self, obs_column: str, *, groups: list[str] | None = None,
                           reference: str = 'rest', libraries=None, sets=None, gene_subset=None,
                           method: str = 'wilcoxon', metric: str = 'score', cell_subset: str | None = None,
                           n_perm: int = 1000, min_set_size: int = 15, max_set_size: int = 500,
                           weight: float = 1.0, seed: int = 0, key: str | None = None):
        """GSEA for every group of ``obs_column`` (vs rest, or vs one group) in one task.

        One throwaway AnnData is snapshotted; each group's ranking is built
        from it inside the task. Per-group results are stored under the
        single-run keys so they are ordinary "previous runs"; a collection
        record lists them for the batch view and for figures.
        """
        from datetime import datetime, timezone  # noqa: PLC0415
        from xcell import enrichment as en  # noqa: PLC0415
        if not 10 <= n_perm <= self.GSEA_MAX_PERMUTATIONS:
            raise ValueError(f'n_perm must be between 10 and {self.GSEA_MAX_PERMUTATIONS}')
        if min_set_size < 1 or max_set_size < min_set_size:
            raise ValueError('Set size range must satisfy 1 <= min <= max')
        if not (weight >= 0):
            raise ValueError('weight must be >= 0')
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('seed must be a non-negative integer')
        if method not in ('wilcoxon', 't-test'):
            raise ValueError("method must be 'wilcoxon' or 't-test'")
        if metric not in ('score', 'log2fc'):
            raise ValueError("metric must be 'score' or 'log2fc'")
        if obs_column not in self.adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in .obs")
        raw_sets = self._enrichment_sets(libraries, sets)
        mask, subset_type, _meta = self._resolve_gene_mask(gene_subset)
        universe = [str(g) for g in self.adata.var_names[mask]]
        cell_mask = self._subset_mask(cell_subset) if cell_subset else np.ones(self.n_cells, dtype=bool)
        values = self.adata.obs[obs_column].astype(str).values
        present = sorted(set(values[cell_mask]))
        if groups is None:
            wanted = present
        else:
            unknown = [g for g in groups if g not in present]
            if unknown:
                raise ValueError(f"Unknown group(s) in '{obs_column}': {unknown}")
            wanted = [str(g) for g in groups]
        if reference != 'rest':
            if reference not in present:
                raise ValueError(f"Reference group '{reference}' not found in column '{obs_column}'")
            wanted = [g for g in wanted if g != reference]
        if not wanted:
            raise ValueError('No groups to test')
        # Skip rather than fail on a tiny cluster; fail only if nothing survives.
        runnable: list[str] = []
        skipped: dict[str, str] = {}
        for g in wanted:
            in_g = cell_mask & (values == g)
            in_ref = cell_mask & ((values != g) if reference == 'rest' else (values == reference))
            if in_g.sum() < 2:
                skipped[g] = 'fewer than 2 cells in group'
            elif in_ref.sum() < 2:
                skipped[g] = 'fewer than 2 reference cells'
            else:
                runnable.append(g)
        if not runnable:
            raise ValueError(f'No contrast has at least 2 cells on each side (every group skipped): {skipped}')
        resolved, rmeta = en.resolve_sets(raw_sets, universe, min_size=min_set_size,
                                          max_size=len(universe), directional='split')
        if not resolved:
            raise ValueError(f'No gene set has at least {min_set_size} members in the universe')
        cells = np.flatnonzero(cell_mask)
        X = self.adata.X[cells][:, mask]
        X = X.copy() if hasattr(X, 'copy') else np.array(X)
        labels = values[cells]
        params = {
            'obs_column': obs_column, 'groups': list(wanted), 'reference': reference, 'method': method,
            'metric': metric, 'cell_subset': cell_subset, 'libraries': list(libraries or []),
            'sets': list(sets or []), 'gene_subset': gene_subset, 'n_perm': int(n_perm),
            'min_set_size': int(min_set_size), 'max_set_size': int(max_set_size),
            'weight': float(weight), 'seed': int(seed),
        }

        def rank_for(group: str) -> np.ndarray:
            import anndata as _ad  # noqa: PLC0415
            import scanpy as sc  # noqa: PLC0415
            keep = (labels == group) | ((labels != group) if reference == 'rest' else (labels == reference))
            tmp = _ad.AnnData(X=X[keep])
            tmp.var_names = universe
            tmp.obs['g'] = pd.Categorical(np.where(labels[keep] == group, 'group', 'reference'),
                                          categories=['group', 'reference'])
            sc.tl.rank_genes_groups(tmp, groupby='g', groups=['group'], reference='reference',
                                    method=method, use_raw=False, key_added='r')
            r = tmp.uns['r']
            field = 'scores' if metric == 'score' else 'logfoldchanges'
            pos = {g: i for i, g in enumerate(universe)}
            out = np.full(len(universe), np.nan)
            for g, v in zip([str(x) for x in r['names']['group']], np.asarray(r[field]['group'], dtype=float)):
                out[pos[g]] = v
            return out

        def compute_fn(report):
            outs = {}
            for i, g in enumerate(runnable):
                report(i / len(runnable), f'{g} ({i + 1}/{len(runnable)}): ranking…')
                scores = rank_for(g)
                out = en.preranked_gsea(scores, resolved, n_perm=n_perm, min_size=min_set_size,
                                        max_size=max_set_size, weight=weight, seed=seed,
                                        report=lambda f, m, _i=i, _g=g: report((_i + 0.2 + 0.8 * f) / len(runnable), f'{_g}: {m}'))
                out['scores'] = scores
                outs[g] = out
            report(1.0, 'Storing…')
            return outs

        def apply_fn(outs):
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            members: dict[str, str] = {}
            n_sig = 0
            n_tested = 0
            for g, out in outs.items():
                order = np.asarray(out['order']); scores = np.asarray(out['scores'])
                for r in out['results']:
                    r['leading_edge'] = [universe[i] for i in r['leading_edge']]
                label = f'{obs_column}: {g} vs {reference}' + (f' [{cell_subset}]' if cell_subset else '')
                result = {
                    'kind': 'gsea', 'label': f'GSEA: {label}', 'created_at': now,
                    'ranking': {'kind': 'diffexp', 'label': label, 'n_ranked': int(out['n_ranked']),
                                'genes': [universe[i] for i in order], 'scores': [float(scores[i]) for i in order]},
                    'universe_size': len(universe), 'gene_subset_type': subset_type,
                    'n_sets_input': rmeta['n_input'], 'n_sets_tested': int(out['n_sets_tested']),
                    'n_significant': sum(1 for r in out['results'] if r['padj'] <= 0.05),
                    'n_perm': int(out['n_perm']), 'results': out['results'],
                    'params': {**params, 'ranking': {'kind': 'diffexp', 'obs_column': obs_column, 'group': g,
                                                     'reference': reference, 'method': method, 'metric': metric,
                                                     'cell_subset': cell_subset}},
                }
                members[g] = self._store_enrichment(f'gsea_{obs_column}_{g}_vs_{reference}', result)
                n_sig += result['n_significant']; n_tested = max(n_tested, result['n_sets_tested'])
            collection = {
                'kind': 'gsea_batch',
                'label': f'GSEA: {obs_column} ({len(members)} groups vs {reference})',
                'created_at': now, 'obs_column': obs_column, 'reference': reference,
                'groups': list(members), 'members': members, 'skipped': skipped,
                'n_perm': int(n_perm), 'universe_size': len(universe), 'gene_subset_type': subset_type,
                'n_sets_tested': n_tested, 'n_significant': n_sig, 'params': params,
            }
            stored = self._store_enrichment(key or f'gsea_{obs_column}_batch', collection)
            self._log_action('enrichment_gsea_batch', params, {
                'key': stored, 'members': members, 'skipped': skipped, 'n_significant': n_sig})
            return collection

        return compute_fn, apply_fn
```

Then extend `get_enrichment_results` (add `'n_groups': len(r['members']) if isinstance(r.get('members'), dict) else None`) and `get_enrichment_result`:

```python
        r = json.loads(store[key])
        members = r.get('members')
        if isinstance(members, dict):
            r['member_results'] = {g: json.loads(store[k]) for g, k in members.items() if k in store}
        return r
```

- [ ] **Step 4: Run** — 3 tests pass. Also run `tests/test_enrichment_adaptor.py` (unchanged behaviour).

- [ ] **Step 5: Commit** `feat(enrichment): batch GSEA across a column's groups with a stored collection`

---

### Task 2: `run_overlap_enrichment_batch` (marker genes → ORA per group)

**Files:** `backend/xcell/adaptor.py`; test `backend/tests/test_enrichment_batch.py`.

**Interfaces:** `run_overlap_enrichment_batch(obs_column, *, groups=None, top_n=100, min_in_group_fraction=None, max_out_group_fraction=None, min_fold_change=None, libraries=None, sets=None, gene_subset=None, min_set_size=5, max_set_size=500, min_overlap=2, key=None) -> dict` returning `{key, kind:'ora_batch', label, created_at, obs_column, groups, members, skipped, markers: {group: [genes]}, top_n, universe_size, gene_subset_type, n_sets_tested, n_significant, params}`.

- [ ] **Step 1: Failing test** (append)

```python
def test_ora_batch_runs_markers_then_overlap_and_stores_markers():
    a = DataAdaptor('x.h5ad', adata=_adata((30, 30, 30, 5)))
    col = a.run_overlap_enrichment_batch('grp', top_n=6, libraries=LIB, min_set_size=2, min_overlap=1)
    assert col['kind'] == 'ora_batch' and col['key'] == 'ora_grp_batch'
    assert set(col['groups']) == {'a', 'b', 'c', 'd'} and col['skipped'] == {}
    assert set(col['markers']['a']) >= {'Col1a1', 'Col1a2', 'Col3a1'} and len(col['markers']['a']) <= 6
    ra = a.get_enrichment_result(col['members']['a'])
    assert ra['kind'] == 'ora' and ra['results'][0]['name'] == 'COLLAGEN'
    assert ra['query']['name'] == 'a markers'
    acts = [h['action'] for h in a._action_history]
    assert acts[-1] == 'enrichment_ora_batch' and 'marker_genes' not in acts[-3:]
    full = a.get_enrichment_result('ora_grp_batch')
    assert full['member_results']['b']['results'][0]['name'] == 'IMMUNE'
```

Note: per-group `run_overlap_enrichment` logs `enrichment_ora` for each member; the assertion only requires no `marker_genes` step and the batch step last. If the per-member `enrichment_ora` steps clutter the record, suppress them: add a private `_log=True` kwarg to `run_overlap_enrichment` and pass `_log=False` from the batch (the batch step carries everything). Do that — the record should show one step for one user action.

- [ ] **Step 2: Run** — AttributeError.

- [ ] **Step 3: Implement**

```python
    def run_overlap_enrichment_batch(self, obs_column: str, *, groups=None, top_n: int = 100,
                                     min_in_group_fraction=None, max_out_group_fraction=None,
                                     min_fold_change=None, libraries=None, sets=None, gene_subset=None,
                                     min_set_size: int = 5, max_set_size: int = 500, min_overlap: int = 2,
                                     key: str | None = None) -> dict[str, Any]:
        """Marker genes (one-vs-rest, top N) for every group, then ORA of each list."""
        from datetime import datetime, timezone  # noqa: PLC0415
        raw_sets = self._enrichment_sets(libraries, sets)   # fail before the marker run
        markers = self.run_marker_genes(obs_column, groups=groups, top_n=top_n,
                                        min_in_group_fraction=min_in_group_fraction,
                                        max_out_group_fraction=max_out_group_fraction,
                                        min_fold_change=min_fold_change, gene_subset=gene_subset, _log=False)
        lists = {r['group']: [g['gene'] for g in r['genes']] for r in markers['results']}
        params = {
            'obs_column': obs_column, 'groups': list(lists), 'top_n': int(top_n),
            'min_in_group_fraction': min_in_group_fraction, 'max_out_group_fraction': max_out_group_fraction,
            'min_fold_change': min_fold_change, 'libraries': list(libraries or []), 'sets': list(sets or []),
            'gene_subset': gene_subset, 'min_set_size': int(min_set_size), 'max_set_size': int(max_set_size),
            'min_overlap': int(min_overlap),
        }
        members: dict[str, str] = {}
        skipped: dict[str, str] = {}
        n_sig = n_tested = 0
        universe_size = 0; subset_type = 'all'
        for g, genes in lists.items():
            if len(genes) < 2:
                skipped[g] = f'only {len(genes)} marker gene(s)'
                continue
            res = self.run_overlap_enrichment(genes, name=f'{g} markers', libraries=libraries, sets=sets,
                                              gene_subset=gene_subset, min_set_size=min_set_size,
                                              max_set_size=max_set_size, min_overlap=min_overlap,
                                              key=f'ora_{obs_column}_{g}_markers', _log=False)
            members[g] = res['key']; n_sig += res['n_significant']
            n_tested = max(n_tested, res['n_sets_tested']); universe_size = res['universe_size']
            subset_type = res['gene_subset_type']
        if not members:
            raise ValueError(f'No group had at least 2 marker genes: {skipped}')
        collection = {
            'kind': 'ora_batch', 'label': f'Overlap: {obs_column} markers ({len(members)} groups)',
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'obs_column': obs_column, 'groups': list(members), 'members': members, 'skipped': skipped,
            'markers': {g: lists[g] for g in members}, 'top_n': int(top_n),
            'universe_size': universe_size, 'gene_subset_type': subset_type,
            'n_sets_tested': n_tested, 'n_significant': n_sig, 'params': params,
        }
        stored = self._store_enrichment(key or f'ora_{obs_column}_batch', collection)
        self._log_action('enrichment_ora_batch', params, {
            'key': stored, 'members': members, 'skipped': skipped, 'n_significant': n_sig})
        return collection
```

Add `_log: bool = True` to `run_marker_genes` and `run_overlap_enrichment` (guard their `_log_action` calls). `raw_sets` is computed only to fail early; ignore the unused variable (name it `_`).

- [ ] **Step 4: Run** the batch + adaptor + marker tests. **Step 5: Commit** `feat(enrichment): marker-gene ORA batch`.

---

### Task 3: Routes + codegen + config

**Files:** `backend/xcell/api/routes.py` (after `/enrichment/gsea`), `backend/xcell/codegen.py`, `backend/tests/test_enrichment_routes.py`, `backend/tests/test_codegen.py` (only via the existing scans).

- [ ] **Step 1: Failing route tests** (append to `test_enrichment_routes.py`)

```python
def test_gsea_batch_route_and_collection_get():
    _install()
    c = TestClient(app)
    body = {'obs_column': 'grp', 'libraries': [{'source': 'msigdb', 'id': 'toy'}], 'n_perm': 40, 'min_set_size': 2}
    r = c.post('/api/enrichment/gsea_batch', json=body)
    assert r.status_code == 202, r.text
    s = _poll(c, r.json()['task_id'])
    assert s['status'] == 'completed', s
    col = s['result']
    assert col['kind'] == 'gsea_batch' and set(col['members']) == {'a', 'b'}
    full = c.get(f"/api/enrichment/results/{col['key']}").json()
    assert set(full['member_results']) == {'a', 'b'}
    assert [x for x in c.get('/api/enrichment/results').json()['results'] if x['key'] == col['key']][0]['n_groups'] == 2
    assert c.post('/api/enrichment/gsea_batch', json=dict(body, obs_column='nope')).status_code == 400


def test_ora_batch_route():
    _install()
    c = TestClient(app)
    r = c.post('/api/enrichment/ora_batch', json={'obs_column': 'grp', 'top_n': 5, 'min_overlap': 1, 'min_set_size': 2,
                                                  'libraries': [{'source': 'msigdb', 'id': 'toy'}],
                                                  'gene_subset': {'columns': ['panel'], 'operation': 'union'}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['kind'] == 'ora_batch' and 'a' in d['markers'] and d['members']['a'].startswith('ora_grp_a')
```

- [ ] **Step 2: Run** — 404s.

- [ ] **Step 3: Implement** routes:

```python
class GseaBatchRequest(BaseModel):
    obs_column: str
    groups: list[str] | None = None
    reference: str = 'rest'
    method: str = 'wilcoxon'
    metric: str = 'score'
    cell_subset: str | None = None
    libraries: list[EnrichmentLibraryRef] | None = None
    sets: list[EnrichmentInlineSet] | None = None
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    n_perm: int = 1000
    min_set_size: int = 15
    max_set_size: int = 500
    weight: float = 1.0
    seed: int = 0
    key: str | None = None


@router.post("/enrichment/gsea_batch", status_code=202)
def run_gsea_batch(request: GseaBatchRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_gsea_batch(
            request.obs_column, groups=request.groups, reference=request.reference,
            libraries=[lib.model_dump() for lib in request.libraries] if request.libraries else None,
            sets=[s.model_dump() for s in request.sets] if request.sets else None,
            gene_subset=_gene_subset_arg(request.gene_subset), method=request.method, metric=request.metric,
            cell_subset=request.cell_subset, n_perm=request.n_perm, min_set_size=request.min_set_size,
            max_set_size=request.max_set_size, weight=request.weight, seed=request.seed, key=request.key)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    task_id = task_manager.submit(compute_fn, apply_fn)
    return {"task_id": task_id, "status": "running"}


class OraBatchRequest(BaseModel):
    obs_column: str
    groups: list[str] | None = None
    top_n: int = 100
    min_in_group_fraction: float | None = None
    max_out_group_fraction: float | None = None
    min_fold_change: float | None = None
    libraries: list[EnrichmentLibraryRef] | None = None
    sets: list[EnrichmentInlineSet] | None = None
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    min_set_size: int = 5
    max_set_size: int = 500
    min_overlap: int = 2
    key: str | None = None


@router.post("/enrichment/ora_batch")
def run_ora_batch(request: OraBatchRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_overlap_enrichment_batch(
            request.obs_column, groups=request.groups, top_n=request.top_n,
            min_in_group_fraction=request.min_in_group_fraction,
            max_out_group_fraction=request.max_out_group_fraction, min_fold_change=request.min_fold_change,
            libraries=[lib.model_dump() for lib in request.libraries] if request.libraries else None,
            sets=[s.model_dump() for s in request.sets] if request.sets else None,
            gene_subset=_gene_subset_arg(request.gene_subset), min_set_size=request.min_set_size,
            max_set_size=request.max_set_size, min_overlap=request.min_overlap, key=request.key)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Codegen (next to the enrichment entries):

```python
    'enrichment_gsea_batch': ActionSpec(
        label='Preranked GSEA for every group', fidelity=XCELL, imports=XCELL_API,
        code=_two_phase('prepare_gsea_batch',
                        ('obs_column', 'groups', 'reference', 'method', 'metric', 'cell_subset',
                         'libraries', 'sets', 'gene_subset', 'n_perm', 'min_set_size', 'max_set_size',
                         'weight', 'seed')),
        summary=lambda p, r: (
            f"GSEA of every group in '{p.get('obs_column')}' vs {p.get('reference')} "
            f"({_n(len((r.get('members') or {})))} groups, {_n(p.get('n_perm'))} permutations): "
            f"{_n(r.get('n_significant'))} significant rows → `.uns['xcell_enrichment']['{r.get('key')}']`"
            + (f"; skipped {', '.join((r.get('skipped') or {}))}." if r.get('skipped') else '.')
        ),
    ),
    'enrichment_ora_batch': ActionSpec(
        label='Marker-gene overlap enrichment for every group', fidelity=XCELL, imports=XCELL_API,
        code=_direct('run_overlap_enrichment_batch',
                     ('obs_column', 'groups', 'top_n', 'min_in_group_fraction', 'max_out_group_fraction',
                      'min_fold_change', 'libraries', 'sets', 'gene_subset', 'min_set_size', 'max_set_size',
                      'min_overlap')),
        summary=lambda p, r: (
            f"Top {_n(p.get('top_n'))} markers of every group in '{p.get('obs_column')}' tested for overlap: "
            f"{_n(r.get('n_significant'))} significant rows → `.uns['xcell_enrichment']['{r.get('key')}']`."
        ),
    ),
```

`_two_phase` / `_direct` splat by name: `prepare_gsea_batch`'s first parameter is positional-or-keyword, so `obs_column=` works.

- [ ] **Step 4: Run** `tests/test_enrichment_routes.py tests/test_codegen.py tests/test_enrichment_batch.py`, then the full suite. **Step 5: Commit** `feat(enrichment): batch routes and codegen`.

---

### Task 4: Frontend — batch mode in the modal, batch results view, API client, types

**Files:** `frontend/src/lib/enrichment.ts` (+test), `frontend/src/hooks/useData.ts`, `frontend/src/components/EnrichmentModal.tsx`.

**Interfaces:**
- `lib/enrichment.ts`: `EnrichmentSource` becomes
  `{ kind: 'ora'; name?; genes?; markersOf?: { obsColumn: string; groups?: string[] } } | { kind: 'gsea'; obsColumn?: string; groups?: string[] | null; reference?: string }` (`groups: null` = all). New types `BatchCollection { key; kind: 'gsea_batch' | 'ora_batch'; label; created_at; obs_column; reference?; groups: string[]; members: Record<string,string>; skipped: Record<string,string>; markers?: Record<string,string[]>; n_sets_tested; n_significant; universe_size; gene_subset_type; n_perm?; top_n?; params; member_results?: Record<string, EnrichmentResult> }`, `AnyEnrichmentResult = EnrichmentResult | BatchCollection`, `EnrichmentSummary.n_groups?: number | null`. Helpers: `isBatch(r)`, `batchGroupsSorted(col)` (groups by `n_significant` of members desc), `batchToGeneSets(col, {padjMax, topN})` → `{folder, sets}[]` one folder per group.
- `useData.ts`: `startGseaBatch(body, slot?)`, `runOraBatch(body, slot?)`; `fetchEnrichmentResult` returns `AnyEnrichmentResult`.

- [ ] **Step 1: Failing vitest** (append to `enrichment.test.ts`): `isBatch` on both shapes; `batchGroupsSorted` ordering; `batchToGeneSets` returns one folder per group with names `<key>` and sets from `resultsToGeneSets` of each member; `filterRows` unchanged.

- [ ] **Step 2: Run RED. Step 3: Implement** helpers + API client.

- [ ] **Step 4: Modal.** In the GSEA tab: `group` select gets a first option `value="__all__"` labelled *All groups (one vs rest)*; when selected, hide the reference select. `runGsea`: if `group === '__all__'` (or `batchGroups` from the source is set), POST `startGseaBatch({obs_column, groups: batchGroups ?? null, reference: 'rest', method, metric, cell_subset, libraries, sets, gene_subset, n_perm, min_set_size, max_set_size, weight, seed})`, poll, `finishRun(collection, 'enrichment_gsea_batch', body)`. In the ORA tab: query select gets `value="__markers__"` *Marker genes of each group in…* which reveals a column select (categorical columns) and the four marker inputs (top N default `cfgDefault(['marker_genes','top_n'],100)`, three optional filters as text inputs like MarkerGenesModal); `runOra` posts `runOraBatch` in that mode. The open effect reads the new source fields: `source.kind === 'gsea' && source.obsColumn` → set `obsColumn`, `group = '__all__'`, `batchGroupsRef.current = source.groups ?? null`, and if `source.reference` is given set `group = source.groups[0]`, `reference = source.reference` (the diffexp hand-off); `source.kind === 'ora' && source.markersOf` → `querySetId = '__markers__'`, column + groups.
  Batch results view: when `isBatch(result)`: summary line (label, groups tested, skipped names, total significant), a chip row of groups (`group (n_sig)`, click selects; default = first of `batchGroupsSorted`), then the existing table for `result.member_results[selected]` (reuse `visibleRows` logic on the selected member; `pFloor` from the member), and footer: **Add to gene sets** (one folder per group via `batchToGeneSets`), **Copy TSV** (the selected member), and two placeholder-free buttons **Heatmap figure…** / **Network figure…** that are *disabled with tooltip "coming in the Figures update"* in this plan (Plan 2 wires them). Previous-runs list shows collections with a `(N groups)` suffix.

- [ ] **Step 5:** `npx tsc --noEmit && npx vitest run`. **Commit** `feat(enrichment): batch mode in the Enrichment modal`.

---

### Task 5: Compare Cells hand-off + browser verification

**Files:** `frontend/src/components/ScanpyModal.tsx` (compare section ~L2500 and `handleCompareRun`), `frontend/src/components/MarkerGenesModal.tsx` (footer), `frontend/src/components/DiffExpModal.tsx` (footer).

- [ ] **Step 1:** ScanpyModal compare section: under the "Will run…" hint add a second button *Enrichment for these groups…* (disabled below 2 checked) → `setEnrichmentSource({ kind: 'gsea', obsColumn: compareColumn, groups: [...compareChecked] }); setScanpyModalOpen(false)`. MarkerGenesModal footer: *Enrichment…* → `setEnrichmentSource({ kind: 'gsea', obsColumn: markerGenesColumn, groups: [...selectedGroups] }); setMarkerGenesModalOpen(false)`. DiffExpModal footer: *Enrichment…* → `setEnrichmentSource({ kind: 'gsea', obsColumn: <column>, groups: [group1Label], reference: group2Label })` — the diffexp modal only has labels and index lists; `comparison` lacks the column name, so store it: in ScanpyModal's 2-group branch call `setMarkerGenesColumn(compareColumn)` too (it is harmless) and read `markerGenesColumn` in DiffExpModal. Hide the button when the column is unknown.

- [ ] **Step 2:** tsc + vitest green. **Commit** `feat(enrichment): Compare Cells hands off to batch enrichment`.

- [ ] **Step 3: Browser.** Isolated stack as before (toy data, synthetic library in a scratch `XDG_CACHE_HOME`; seed gene sets). Drive: Analyze → Cells → Compare Cells → cell_type → check both groups → *Enrichment for these groups…* → modal opens in GSEA batch mode with `groups: ['Mesen','Primor']` in the POST body (capture via patched fetch) → run → chips for both groups → table for the selected group → Add to gene sets → two folders in Enrichment. Then Genes menu → Enrichment → GSEA → *All groups* → body has `groups: null`. ORA tab → *Marker genes of each group in cell_type* → run → collection. Previous runs shows `(2 groups)` entries. Check no console errors. Stop servers, delete Playwright leftovers.

- [ ] **Step 4:** Fix what the browser finds (with tests where testable). CHANGELOG entry under the enrichment bullet. **Commit** `docs: batch enrichment`.
