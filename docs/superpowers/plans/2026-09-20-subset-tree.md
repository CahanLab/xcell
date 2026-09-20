# Subset Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hierarchy of named cell subsets explicit in the AnnData — parent links, what each step wrote and with which parameters, the embeddings each subset owns and the shapes/territories drawn on them — surface it as a tree in the Cells panel, make PCA Loadings and PC subsets work on a subset, and add a delete button to `.obs` columns.

**Architecture:** `uns['xcell_cell_subsets']` stays a plain dict registry (scanpy-readable) and becomes the record: `parent` (inferred by containment at creation), `origin`, and a `derived` dict each scoped operation writes at run time. Decorations stay keyed by embedding name; the listing resolves them per subset. Shapes/lines persist in `uns['xcell_lines_json']` on every sync (like territories) and hydrate on load. Deleting a subset with its results cascades to its embeddings' decorations and its PC subsets.

**Tech Stack:** Python 3.11, numpy, pandas, AnnData 0.12, scanpy, FastAPI, pytest; React 18 + Zustand + inline styles, vitest, tsc.

**Spec:** `docs/superpowers/specs/2026-09-20-subset-tree-design.md`

## Global Constraints

- `backend/xcell/adaptor.py` is the only thing that owns AnnData. Routes are thin: resolve adaptor, call it, map exceptions (`ValueError → 400`, `KeyError → 404`, no data → 503; `except HTTPException: raise` before any catch-all).
- Every data-touching route takes `dataset: str | None = Query(None)` as its **last** parameter.
- Pydantic request models sit immediately above their route, named `<Thing>Request`, flat snake_case, defaults matching the adaptor signature; never narrower than the adaptor argument.
- Anything crossing the API is JSON-serializable: no `inf`, no `NaN`, no numpy types.
- Anything that mutates calls `self._log_action(action, params, result)`.
- A subset run must always write `<key>_<name>` (`_subset_output_name`).
- anndata writes empty lists into uns as `array([], dtype=float64)` — normalise registry values on read.
- Tests: `backend/tests/`, no `conftest.py`, module-level `_adata()` / `_adaptor()` factories, seeded RNG, `csr_matrix` + `float32`; assert on the AnnData state, not only the returned dict. Route tests `monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)`.
- Frontend styling is inline. Palette: panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`/`#888`. After a backend mutation that touches `.obs`, call both `refreshSchema()` and `refreshObsSummaries()`.
- Work in `/private/tmp/claude-501/xcell-wt` on `feat/subset-tree`. Backend tests: `cd <wt>/backend && "<main>/.pixi/envs/dev/bin/python" -m pytest tests/test_x.py -q`. Frontend: `cd <wt>/frontend && npx vitest run && npx tsc --noEmit`.
- Commit after each task. Merge into `main` with `--no-ff` at the end, after checking `:8000` holds nothing.

---

## File Structure

**Created:**
- `backend/tests/test_obs_column_delete.py` — delete an obs column (adaptor, route, codegen).
- `backend/tests/test_subset_tree.py` — parent/origin/derived record/tree listing/cascade/lines persistence/PC subsets on a subset.
- `frontend/src/lib/subsetTree.ts` + `.test.ts` — pure helpers: drop summary text, chip lists, tree ordering guard.

**Modified:**
- `backend/xcell/adaptor.py` — `delete_obs_column`; registry helpers (`_subset_registry` normalisation, `_subset_record_step`, `_subset_forget_key`, `_infer_parent`); `create_cell_subset(parent=, origin=)`; `_subset_derived_keys` merged; `_subset_summary` tree fields; `list_cell_subsets` ordering; `delete_cell_subset` cascade + reparent; lines hydrate/persist; `get_pca_loadings` / `create_pca_subset` / `list_pca_subsets` / `delete_pca_subset` subset-aware; `run_pca` clearing loop; `check_prerequisites`.
- `backend/xcell/api/routes.py` — `DELETE /obs/{column}`; `CellSubsetRequest.parent/origin`; `LineData` fields; `?cell_subset=` on PCA loadings / PC subsets.
- `backend/xcell/codegen.py` — `delete_obs_column` spec; `create_cell_subset` emits `parent=`; `create_pca_subset` emits `cell_subset=`.
- `frontend/src/lib/cellSubsets.ts` (+ test) — `CellSubsetInfo` tree fields; `derivedBadges` → chips.
- `frontend/src/hooks/useData.ts` — `deleteObsColumn`, `fetchLines`, `usePcaLoadings(cellSubset)`, `fetchPcaSubsets`/`createPcaSubset(cellSubset)`, `createCellSubset(origin)`.
- `frontend/src/store.ts` (+ `store.test.ts`) — `setDrawnLines`, `linesHydrated`, `forgetObsColumn`, `setSchema` embedding repair.
- `frontend/src/App.tsx` — lines hydrate + debounced sync effects.
- `frontend/src/components/CellPanel.tsx` — Delete column button + confirm; subset tree rows with chips; cascade confirm text; refine hand-off.
- `frontend/src/components/ScanpyModal.tsx` — PCA loadings / PC subsets use the active subset.
- `frontend/src/messages.ts` — new strings.
- `CHANGELOG.md`.

---

### Task 1: Backend — delete an `.obs` column

**Files:**
- Create: `backend/tests/test_obs_column_delete.py`
- Modify: `backend/xcell/adaptor.py` (replace `delete_annotation` ~line 3061)
- Modify: `backend/xcell/api/routes.py` (new route near `@router.get("/obs/{column}")` ~line 730; keep `DELETE /annotations/{name}` delegating)
- Modify: `backend/xcell/codegen.py` (new spec near `'create_annotation'` ~line 806)

**Interfaces:**
- Produces: `DataAdaptor.delete_obs_column(column: str) -> dict` returning `{'column': str, 'dropped_colors': bool, 'subset_removed': str | None}`; `delete_annotation(name)` kept as an alias. Route `DELETE /api/obs/{column}` → that dict. Action name `delete_obs_column`, params `{'column': ...}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Deleting an .obs column from the Cells panel.

A column is dropped with its scanpy colour list; if it was a named subset's
membership column the registry entry goes with it (a subset with no column
cannot be activated), and the step is recorded so the notebook replays it.
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.codegen import translate
from xcell.main import app


def _adata(n_cells=30, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.random((n_cells, n_genes)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    ad.obs['cell_type'] = pd.Categorical(['a', 'b', 'c'] * (n_cells // 3))
    ad.obs['score'] = rng.random(n_cells)
    ad.uns['cell_type_colors'] = ['#111111', '#222222', '#333333']
    return ad


def _adaptor():
    return DataAdaptor('x.h5ad', adata=_adata())


def test_deleting_drops_the_column_and_its_colours_and_records_the_step():
    a = _adaptor()
    out = a.delete_obs_column('cell_type')
    assert out == {'column': 'cell_type', 'dropped_colors': True, 'subset_removed': None}
    assert 'cell_type' not in a.adata.obs.columns
    assert 'cell_type_colors' not in a.adata.uns
    step = a.analysis_record.steps[-1]
    assert step.action == 'delete_obs_column' and step.params == {'column': 'cell_type'}


def test_deleting_a_numeric_column_reports_no_colours():
    a = _adaptor()
    assert a.delete_obs_column('score')['dropped_colors'] is False
    assert 'score' not in a.adata.obs.columns


def test_an_unknown_column_is_a_key_error():
    with pytest.raises(KeyError):
        _adaptor().delete_obs_column('nope')


def test_deleting_a_subsets_column_removes_its_registry_entry():
    a = _adaptor()
    a.create_cell_subset('chondro', list(range(0, 30, 2)))
    out = a.delete_obs_column('subset_chondro')
    assert out['subset_removed'] == 'chondro'
    assert 'chondro' not in a.adata.uns['xcell_cell_subsets']
    assert a.list_cell_subsets() == []


def test_delete_annotation_still_works_as_an_alias():
    a = _adaptor()
    a.delete_annotation('score')
    assert 'score' not in a.adata.obs.columns


def test_the_notebook_replays_the_deletion():
    a = _adaptor()
    a.delete_obs_column('score')
    code = translate(a.analysis_record).code
    assert "del adata.obs['score']" in code


def test_the_route_deletes_and_maps_a_missing_column_to_404(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    r = client.delete('/api/obs/cell_type')
    assert r.status_code == 200 and r.json()['column'] == 'cell_type'
    assert 'cell_type' not in a.adata.obs.columns
    assert client.delete('/api/obs/cell_type').status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_obs_column_delete.py -q`
Expected: FAIL — `AttributeError: 'DataAdaptor' object has no attribute 'delete_obs_column'`, 404/405 on the route. (Check `translate(...)` returns an object with `.code` by grepping `def translate` in `codegen.py`; adjust the test to the real return shape.)

- [ ] **Step 3: Implement**

In `adaptor.py`, replace `delete_annotation`:

```python
    def delete_obs_column(self, column: str) -> dict[str, Any]:
        """Drop an .obs column, its scanpy colour list, and — if it was a
        named subset's membership column — the subset's registry entry, since
        a subset with no column can never be activated again."""
        if column not in self.adata.obs.columns:
            raise KeyError(f"Column '{column}' not found in .obs")
        self.adata.obs.drop(columns=[column], inplace=True)
        dropped_colors = self.adata.uns.pop(f'{column}_colors', None) is not None
        subset_removed = None
        registry = self._subset_registry()
        for name, entry in registry.items():
            if entry.get('obs_key', SUBSET_OBS_PREFIX + name) == column:
                subset_removed = name
                break
        if subset_removed is not None:
            registry.pop(subset_removed)
            self.adata.uns[CELL_SUBSETS_UNS] = registry
        result = {'column': column, 'dropped_colors': dropped_colors,
                  'subset_removed': subset_removed}
        self._log_action('delete_obs_column', {'column': column}, result)
        return result

    def delete_annotation(self, name: str) -> None:
        """Older name for delete_obs_column; the /annotations route still calls it."""
        self.delete_obs_column(name)
```

Route (next to `GET /obs/{column}`):

```python
@router.delete("/obs/{column}")
def delete_obs_column(column: str, dataset: str | None = Query(None)):
    """Drop an .obs column (and its colours / subset entry)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.delete_obs_column(column)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

Codegen spec (next to `'create_annotation'`):

```python
    'delete_obs_column': ActionSpec(
        label='Delete column', fidelity=EXACT, imports=(),
        code=lambda s: [f"del {ADATA}.obs[{_lit(s.params.get('column'))}]"],
        summary=lambda p, r: f"Deleted `.obs['{p.get('column')}']`.",
    ),
```

(Check how other specs spell an empty `imports` — grep `imports=()` — and match.)

- [ ] **Step 4: Run tests — pass; run the full suite**

Run: `pytest tests/test_obs_column_delete.py tests/test_cell_subsets.py -q`

- [ ] **Step 5: Commit** — `feat(backend): delete an .obs column with its colours and subset entry`

---

### Task 2: Frontend — Delete button on `.obs` columns

**Files:**
- Modify: `frontend/src/hooks/useData.ts` (next to `deleteAnnotation` ~line 1652)
- Modify: `frontend/src/store.ts` (column actions ~line 2160; add `forgetObsColumn`)
- Modify: `frontend/src/store.test.ts` (new describe)
- Modify: `frontend/src/components/CellPanel.tsx` (`CategoryColumn` ~627–785, `ContinuousColumn` ~789–905, hidden rows ~1533–1555, `CellPanel` handlers)
- Modify: `frontend/src/messages.ts`

**Interfaces:**
- Produces: `deleteObsColumn(name: string, slot?: DatasetSlot): Promise<{column: string; subset_removed: string | null}>`; store action `forgetObsColumn(name: string)` clearing `colorBy` (if `colorBy.name === name`), `embeddingLabelColumn` (if equal), the `hiddenColumns` entry and `columnDisplayNames[name]`.

- [ ] **Step 1: Store test**

```ts
describe('forgetObsColumn', () => {
  it('clears everything that named the column', () => {
    const s = useStore.getState()
    s.setColorBy({ name: 'leiden', dtype: 'category', values: [] } as any)
    s.setEmbeddingLabelColumn('leiden')
    s.hideColumn('leiden')
    s.setColumnDisplayName('leiden', 'Clusters')
    useStore.getState().forgetObsColumn('leiden')
    const after = useStore.getState()
    expect(after.colorBy).toBeNull()
    expect(after.embeddingLabelColumn).toBeNull()
    expect(after.hiddenColumns.has('leiden')).toBe(false)
    expect(after.columnDisplayNames['leiden']).toBeUndefined()
  })
  it('leaves other columns alone', () => {
    const s = useStore.getState()
    s.setColorBy({ name: 'cell_type', dtype: 'category', values: [] } as any)
    useStore.getState().forgetObsColumn('leiden')
    expect(useStore.getState().colorBy?.name).toBe('cell_type')
  })
})
```

- [ ] **Step 2: Run — fails** (`forgetObsColumn is not a function`).

- [ ] **Step 3: Implement**

`useData.ts`:
```ts
export async function deleteObsColumn(name: string, slot?: DatasetSlot): Promise<{ column: string; subset_removed: string | null }> {
  return fetchJson(appendDataset(`${API_BASE}/obs/${encodeURIComponent(name)}`, slot), { method: 'DELETE' })
}
```

`store.ts` (type in the actions interface + implementation next to `showColumn`):
```ts
forgetObsColumn: (name) =>
  set(dsUpdateFn((state) => {
    const hidden = new Set(state.hiddenColumns); hidden.delete(name)
    const { [name]: _dropped, ...names } = state.columnDisplayNames
    return {
      hiddenColumns: hidden,
      columnDisplayNames: names,
      colorBy: state.colorBy?.name === name ? null : state.colorBy,
      embeddingLabelColumn: state.embeddingLabelColumn === name ? null : state.embeddingLabelColumn,
    }
  })),
```
(`embeddingLabelColumn` is global, not per-dataset — check where it lives (~line 1008) and clear it with a top-level `set` if so.)

`CellPanel.tsx`: add `onDelete: () => void` to both column props; render next to Hide inside the hovered `columnActions`:
```tsx
<button style={{ ...styles.columnActionButton, color: '#e94560' }}
        onClick={(e) => { e.stopPropagation(); onDelete() }}
        title="Delete this column from the dataset">Delete</button>
```
In `CellPanel`, `const [columnDeleteConfirm, setColumnDeleteConfirm] = useState<string | null>(null)`; `onDelete={() => setColumnDeleteConfirm(summary.name)}`. Directly under a column whose name matches, render the confirm row (reuse `styles.subsetConfirm`):
```tsx
{columnDeleteConfirm === summary.name && (
  <div style={{ ...styles.subsetConfirm, padding: '2px 16px 6px' }}>
    <span>{MESSAGES.obsColumns.deleteConfirm(getDisplayName(summary.name))}</span>
    <button style={{ ...styles.maskActionButton, color: '#e94560', border: '1px solid #e94560' }}
            onClick={() => handleDeleteColumn(summary.name)}>Delete</button>
    <button style={styles.maskActionButton} onClick={() => setColumnDeleteConfirm(null)}>Cancel</button>
  </div>
)}
```
Handler:
```ts
const handleDeleteColumn = useCallback(async (name: string) => {
  try {
    const out = await deleteObsColumn(name)
    forgetObsColumn(name)
    setColumnDeleteConfirm(null)
    await refreshSchema()
    refresh()               // obs summaries
    if (out.subset_removed) await refreshCellSubsets()
  } catch (err) { alert(`Failed to delete column: ${(err as Error).message}`) }
}, [forgetObsColumn, refresh])
```
Hidden rows get a `✕` button beside Show that calls `setColumnDeleteConfirm(summary.name)` and the same confirm row.

`messages.ts`: `obsColumns: { deleteConfirm: (name: string) => \`Delete "${name}" from the dataset? This cannot be undone here.\` }`.

- [ ] **Step 4: `npx vitest run && npx tsc --noEmit`** — pass.

- [ ] **Step 5: Commit** — `feat(cells): delete an .obs column from the Cells panel`

---

### Task 3: Registry as a tree — parent, origin, ordering, reparenting

**Files:**
- Create: `backend/tests/test_subset_tree.py`
- Modify: `backend/xcell/adaptor.py` (`_subset_registry` ~4716, `create_cell_subset` ~4758, `list_cell_subsets` ~4805, `delete_cell_subset` ~4824, `_subset_summary` ~4746)
- Modify: `backend/xcell/api/routes.py` (`CellSubsetRequest` ~2260)
- Modify: `backend/xcell/codegen.py` (`'create_cell_subset'` ~400)
- Modify: `backend/tests/test_cell_subsets.py` (listing test expects new fields — use `>=` on the dict or update)

**Interfaces:**
- Produces: `create_cell_subset(name, cell_indices, *, description=None, overwrite=False, parent: str | None = None, origin: dict | None = None)`; `_infer_parent(idx: np.ndarray, exclude: str | None = None) -> str | None`; summary gains `parent: str | None`, `children: list[str]`, `depth: int`, `origin: dict | None`. `list_cell_subsets()` orders depth-first (parents before children; siblings by `created_at`, then name). Deleting a subset reparents its children to its parent.

- [ ] **Step 1: Failing tests** (`test_subset_tree.py`, same `_adata`/`_adaptor`/`ACTIVE` factories as `test_cell_subsets.py`; 60 cells, 80 genes):

```python
def test_a_subset_inside_another_gets_it_as_parent():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)                 # 30 even cells
    out = a.create_cell_subset('prolif', [0, 2, 4, 6])
    assert out['parent'] == 'chondro' and out['depth'] == 1
    assert a.adata.uns['xcell_cell_subsets']['prolif']['parent'] == 'chondro'
    chondro = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert chondro['children'] == ['prolif'] and chondro['parent'] is None


def test_the_smallest_containing_subset_wins():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    assert a.create_cell_subset('small', [1, 2, 3])['parent'] == 'mid'


def test_a_selection_not_inside_any_subset_is_a_root():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.create_cell_subset('other', [1, 3, 5])['parent'] is None


def test_an_explicit_parent_is_honoured_and_validated():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    assert a.create_cell_subset('x', [1, 2], parent='big')['parent'] == 'big'
    with pytest.raises(ValueError):
        a.create_cell_subset('y', [45, 46], parent='mid')   # not inside mid
    with pytest.raises(ValueError):
        a.create_cell_subset('z', [1], parent='nope')


def test_origin_is_stored_and_returned():
    a = _adaptor()
    out = a.create_cell_subset('chondro', ACTIVE, origin={'kind': 'selection', 'embedding': 'X_umap'})
    assert out['origin'] == {'kind': 'selection', 'embedding': 'X_umap'}


def test_the_listing_is_depth_first_with_parents_before_children():
    a = _adaptor()
    a.create_cell_subset('b_root', [40, 41, 42])
    a.create_cell_subset('a_root', list(range(30)))
    a.create_cell_subset('kid2', [5, 6])
    a.create_cell_subset('kid1', [1, 2])
    a.create_cell_subset('grandkid', [1])
    names = [(s['name'], s['depth']) for s in a.list_cell_subsets()]
    assert names == [('b_root', 0), ('a_root', 0), ('kid2', 1), ('kid1', 1), ('grandkid', 2)]


def test_deleting_a_parent_reparents_its_children():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    a.create_cell_subset('small', [1, 2])
    a.delete_cell_subset('mid')
    small = next(s for s in a.list_cell_subsets() if s['name'] == 'small')
    assert small['parent'] == 'big' and small['depth'] == 1


def test_the_registry_survives_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.create_cell_subset('prolif', [0, 2])
    p = tmp_path / 'x.h5ad'
    a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    prolif = next(s for s in b.list_cell_subsets() if s['name'] == 'prolif')
    assert prolif['parent'] == 'chondro' and prolif['children'] == []


def test_the_route_accepts_parent_and_origin(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    client.post('/api/cell_subsets', json={'name': 'chondro', 'cell_indices': ACTIVE})
    r = client.post('/api/cell_subsets', json={
        'name': 'prolif', 'cell_indices': [0, 2], 'parent': 'chondro',
        'origin': {'kind': 'selection', 'embedding': 'X_umap'}})
    assert r.status_code == 200 and r.json()['parent'] == 'chondro'


def test_the_notebook_emits_the_parent():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.create_cell_subset('prolif', [0, 2])
    code = translate(a.analysis_record).code
    assert "create_cell_subset('prolif', SELECTIONS['step_" in code and "parent='chondro'" in code
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement**

Registry normalisation:
```python
    @staticmethod
    def _plain(value: Any) -> Any:
        """uns round-trips lists as arrays and dicts as h5 groups; give the
        registry back its JSON shape so callers can compare and serialise."""
        if isinstance(value, np.ndarray):
            return [DataAdaptor._plain(v) for v in value.tolist()]
        if isinstance(value, Mapping):
            return {str(k): DataAdaptor._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DataAdaptor._plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _subset_registry(self) -> dict[str, dict[str, Any]]:
        reg = self.adata.uns.get(CELL_SUBSETS_UNS)
        return self._plain(reg) if isinstance(reg, Mapping) else {}
```

Parent inference (new, below `_subset_mask`):
```python
    def _infer_parent(self, idx: np.ndarray, exclude: str | None = None) -> str | None:
        """The smallest registered subset containing every cell of `idx`; a
        subset with exactly these cells is a twin, not a parent. Ties go to
        the most recently created."""
        registry = self._subset_registry()
        best: tuple[int, str, str] | None = None
        for name, entry in registry.items():
            if name == exclude:
                continue
            key = entry.get('obs_key', SUBSET_OBS_PREFIX + name)
            if key not in self.adata.obs.columns:
                continue
            mask = self.adata.obs[key].values.astype(bool)
            n = int(mask.sum())
            if n <= len(idx) or not mask[idx].all():
                continue
            cand = (n, entry.get('created_at') or '', name)
            if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
        return best[2] if best else None
```

`create_cell_subset`: add `parent: str | None = None, origin: dict[str, Any] | None = None`; after the range checks:
```python
        if parent is not None:
            parent = self._sanitize_subset_name(parent)
            if parent not in registry:
                raise ValueError(f"No cell subset named '{parent}' to be the parent.")
            pmask = self.adata.obs[registry[parent].get('obs_key', SUBSET_OBS_PREFIX + parent)].values.astype(bool)
            if not pmask[idx].all():
                raise ValueError(f"Subset '{parent}' does not contain every cell of '{clean}', so it cannot be its parent.")
        else:
            parent = self._infer_parent(idx, exclude=clean)
        entry = {'obs_key': key, 'created_at': ..., 'description': ...}
        if parent:
            entry['parent'] = parent
        if origin:
            entry['origin'] = {str(k): v for k, v in origin.items() if v is not None}
        registry[clean] = entry
```
Log params gain `'parent': parent` when set.

Tree fields: a private `_subset_tree(registry) -> tuple[dict[str, list[str]], dict[str, int], list[str]]` returning children map, depth map, and depth-first order (roots and siblings sorted by `(created_at, name)`; a `parent` that names a missing subset is treated as none; depth is capped at `len(registry)` to survive a cycle in a hand-edited file). `_subset_summary(name, entry, tree)` adds `parent`, `children`, `depth`, `origin` (or `None`). `list_cell_subsets` uses the order.

`delete_cell_subset`: before popping, `for n, e in registry.items(): if e.get('parent') == clean: e['parent'] = entry.get('parent')` (or delete the key when None).

Route: `CellSubsetRequest` gains `parent: str | None = None`, `origin: dict[str, Any] | None = None` and passes them.

Codegen: `create_cell_subset` code adds `f", parent={_lit(step.params['parent'])}"` when `step.params.get('parent')`.

- [ ] **Step 4: Run `tests/test_subset_tree.py tests/test_cell_subsets.py` — pass**; fix the listing test in `test_cell_subsets.py` if it compares the whole dict.

- [ ] **Step 5: Commit** — `feat(subsets): the registry records the tree — parent by containment, origin, depth-first listing`

---

### Task 4: `derived` written at run time; merged detection; embeddings list

**Files:**
- Modify: `backend/xcell/adaptor.py` — `run_highly_variable_genes` (subset branch ~6205), `run_pca` (~6558), `run_neighbors` (~6955), `run_umap` (~7440), `run_leiden` (~7560), `_subset_derived_keys` (~4731), `_subset_summary`
- Modify: `backend/tests/test_subset_tree.py`, `backend/tests/test_cell_subsets.py` (listing test: `umap` becomes a list, `pca_subsets` appears)
- Modify: `frontend/src/lib/cellSubsets.ts` + `.test.ts` (`CellSubsetDerived.umap: string[]`, `pca_subsets: string[]`; `derivedBadges`)

**Interfaces:**
- Produces: `_subset_record_step(name: str, step: str, key: str, params: dict | None = None, extra: dict | None = None) -> None` — `step ∈ {'hvg','pca','graph'}` stores `derived[step] = {'key': key, 'params': params}`; `step ∈ {'umap','leiden'}` stores `derived[step][key] = params`; `step == 'pca_subsets'` stores `derived['pca_subsets'][key] = extra`. `_subset_forget_key(name: str, key: str) -> None` removes any record naming `key`. Summary `derived` = `{'hvg': str|None, 'pca': str|None, 'graph': str|None, 'umap': list[str], 'leiden': list[str], 'pca_subsets': list[str]}` (recorded ∪ detected, existing keys only); `steps` = the recorded dict; `embeddings` = `[pca] + pca_subsets + umap` that exist in obsm.

- [ ] **Step 1: Failing tests**

```python
def _chain(a, name):
    a.run_highly_variable_genes(n_top_genes=10, cell_subset=name)
    a.run_pca(n_comps=4, cell_subset=name)
    a.run_neighbors(n_neighbors=5, cell_subset=name)
    a.run_umap(min_dist=0.3, cell_subset=name)
    a.run_leiden(resolution=0.8, cell_subset=name)


def test_each_scoped_step_records_its_key_and_params_in_the_registry():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    d = a.adata.uns['xcell_cell_subsets']['chondro']['derived']
    assert d['hvg']['key'] == 'highly_variable__chondro' and d['hvg']['params']['n_top_genes'] == 10
    assert d['pca']['key'] == 'X_pca_chondro' and d['pca']['params']['n_comps'] == 4
    assert d['graph']['key'] == 'chondro_connectivities' and d['graph']['params']['n_neighbors'] == 5
    assert d['umap'] == {'X_umap_chondro': {'min_dist': 0.3, 'spread': 1.0, 'n_components': 2, 'graph_key': 'chondro_connectivities'}}
    assert d['leiden']['leiden_chondro']['resolution'] == 0.8


def test_a_second_umap_over_another_graph_is_recorded_beside_the_first():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.run_umap(cell_subset='chondro', graph_key='connectivities', key_added='X_umap_chondro_alt')
    s = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert s['derived']['umap'] == ['X_umap_chondro', 'X_umap_chondro_alt']
    assert s['embeddings'] == ['X_pca_chondro', 'X_umap_chondro', 'X_umap_chondro_alt']


def test_a_key_that_vanished_is_not_reported_even_if_recorded():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    del a.adata.obsm['X_umap_chondro']
    s = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert s['derived']['umap'] == [] and 'X_umap_chondro' not in s['embeddings']


def test_a_file_from_before_the_record_is_still_detected_by_name():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.adata.uns['xcell_cell_subsets']['chondro'].pop('derived')
    s = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert s['derived']['pca'] == 'X_pca_chondro' and s['derived']['leiden'] == ['leiden_chondro']
    assert s['steps'] == {}


def test_the_record_round_trips_through_h5ad(tmp_path):
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    p = tmp_path / 'x.h5ad'; a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    s = next(s for s in b.list_cell_subsets() if s['name'] == 'chondro')
    assert s['steps']['umap']['X_umap_chondro']['min_dist'] == 0.3
    json.dumps(s)   # nothing numpy left
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement**

```python
    def _subset_record_step(self, name, step, key, params=None, extra=None):
        registry = self._subset_registry()
        entry = registry.get(name)
        if entry is None:
            return
        derived = entry.setdefault('derived', {})
        clean_params = {k: (v.item() if isinstance(v, np.generic) else v)
                        for k, v in (params or {}).items()
                        if isinstance(v, (str, int, float, bool)) or v is None}
        if step in ('hvg', 'pca', 'graph'):
            derived[step] = {'key': key, 'params': clean_params}
        elif step in ('umap', 'leiden'):
            derived.setdefault(step, {})[key] = clean_params
        elif step == 'pca_subsets':
            derived.setdefault('pca_subsets', {})[key] = extra or {}
        registry[name] = entry
        self.adata.uns[CELL_SUBSETS_UNS] = registry

    def _subset_forget_key(self, name, key):
        registry = self._subset_registry()
        derived = (registry.get(name) or {}).get('derived') or {}
        for step in ('hvg', 'pca', 'graph'):
            if (derived.get(step) or {}).get('key') == key:
                derived.pop(step)
        for step in ('umap', 'leiden', 'pca_subsets'):
            (derived.get(step) or {}).pop(key, None)
        if name in registry:
            registry[name]['derived'] = derived
            self.adata.uns[CELL_SUBSETS_UNS] = registry
```

Call sites: HVG subset branch → `self._subset_record_step(subset_name, 'hvg', col, params)`; PCA → `('pca', pca_key, {'n_comps': n_comps, 'svd_solver': svd_solver, 'gene_subset_type': subset_type, 'n_genes_used': n_genes_used})`; neighbors → `('graph', f'{subset_name}_connectivities', {'n_neighbors', 'n_pcs', 'metric', 'use_rep': result['use_rep']})`; umap → `('umap', name, {'min_dist', 'spread', 'n_components', 'graph_key'})`; leiden → `('leiden', name, {'resolution', 'graph_key'})`. Find the exact HVG subset write by grepping `cell_subset` inside `run_highly_variable_genes` (it may go through `_hvg_mask_for_cells` with `name=subset_name`).

`_subset_derived_keys(name, entry)`:
```python
        derived = (entry or {}).get('derived') or {}
        rec = lambda step: (derived.get(step) or {}).get('key')
        hvg = rec('hvg') or f'highly_variable__{name}'
        pca = rec('pca') or f'X_pca_{name}'
        graph = rec('graph') or f'{name}_connectivities'
        umaps = list((derived.get('umap') or {}).keys())
        for k in self.adata.obsm:            # detection fallback
            if (k == f'X_umap_{name}' or k.startswith(f'X_umap_{name}_')) and k not in umaps:
                umaps.append(k)
        leidens = list((derived.get('leiden') or {}).keys())
        for c in self.adata.obs.columns:
            if (c == f'leiden_{name}' or c.startswith(f'leiden_{name}_')) and c not in leidens:
                leidens.append(c)
        pcs = list((derived.get('pca_subsets') or {}).keys())
        for k in self.adata.obsm:
            if k.startswith(f'X_pca_{name}_') and k not in pcs:
                pcs.append(k)
        return {
            'hvg': hvg if hvg in self.adata.var.columns else None,
            'pca': pca if pca in self.adata.obsm else None,
            'graph': graph if graph in self.adata.obsp else None,
            'umap': [k for k in umaps if k in self.adata.obsm],
            'leiden': [c for c in leidens if c in self.adata.obs.columns],
            'pca_subsets': [k for k in pcs if k in self.adata.obsm],
        }
```
Summary adds `'steps': derived_dict`, `'embeddings': [pca] (if any) + pca_subsets + umap`.

Frontend `cellSubsets.ts`: `umap: string[]`, `pca_subsets: string[]`; `derivedBadges` pushes each UMAP key and each PC subset key; tests updated.

- [ ] **Step 4: Run backend suite for the two files; `npx vitest run`; pass.**

- [ ] **Step 5: Commit** — `feat(subsets): scoped steps record what they wrote and with which parameters`

---

### Task 5: Shapes and lines persist in `uns['xcell_lines_json']`; decorations in the listing

**Files:**
- Modify: `backend/xcell/adaptor.py` — `__init__` (~816), `set_lines`/`get_lines` (~3562), `prepare_export_with_lines` (~3705–3724), `_subset_summary`
- Modify: `backend/xcell/api/routes.py` — `LineData` (~1755)
- Modify: `backend/tests/test_subset_tree.py`

**Interfaces:**
- Produces: `LINES_UNS = 'xcell_lines_json'`; `set_lines(lines: list[dict]) -> None` writes uns and memory; `get_lines() -> list[dict]`; `_restore_lines() -> list[dict]` (tolerates the old export shape with `embedding` instead of `embeddingName`); `_lines_on(embeddings: set[str]) -> list[dict]`; `_territories_on(embeddings: set[str]) -> list[str]`. Summary gains `decorations: {'lines': [names], 'territories': [types]}`. `LineData` gains `id: str | None = None`, `drawType: str = 'pencil'`, `closed: bool = False`, `visible: bool = True`, `strokeColor: str = '#4ecdc4'`, `strokeWidth: float = 2`, `fillColor: str | None = None`.

- [ ] **Step 1: Failing tests**

```python
LINE = {'id': 'line_1', 'name': 'ridge', 'embeddingName': 'X_umap_chondro', 'dimX': 0, 'dimY': 1,
        'points': [[0.0, 0.0], [1.0, 1.0]], 'smoothedPoints': None, 'drawType': 'pencil',
        'closed': False, 'visible': True, 'strokeColor': '#4ecdc4', 'strokeWidth': 2, 'fillColor': None}


def test_lines_are_written_to_uns_and_survive_a_reload(tmp_path):
    a = _adaptor()
    a.set_lines([LINE])
    assert json.loads(a.adata.uns['xcell_lines_json'])[0]['name'] == 'ridge'
    p = tmp_path / 'x.h5ad'; a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    assert b.get_lines() == [LINE]


def test_an_old_export_with_embedding_instead_of_embedding_name_still_loads():
    ad = _adata()
    ad.uns['xcell_lines_json'] = json.dumps([{'name': 'old', 'embedding': 'X_umap', 'points': [[0, 0], [1, 1]]}])
    a = DataAdaptor('x.h5ad', adata=ad)
    line = a.get_lines()[0]
    assert line['embeddingName'] == 'X_umap' and line['visible'] is True and line['drawType'] == 'pencil'


def test_the_listing_names_the_shapes_and_territories_on_a_subsets_embeddings():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.set_lines([LINE, {**LINE, 'id': 'line_2', 'name': 'elsewhere', 'embeddingName': 'X_umap'}])
    a.save_territories('zones', {'embedding': 'X_umap_chondro', 'sections': {
        'all': {'ring': [[0, 0], [1, 0], [1, 1]], 'cuts': [], 'anchors': []}}})
    s = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert s['decorations'] == {'lines': ['ridge'], 'territories': ['zones']}


def test_the_lines_route_round_trips_every_field(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    assert client.post('/api/lines', json={'lines': [LINE]}).status_code == 200
    assert client.get('/api/lines').json()['lines'] == [LINE]
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement**

```python
LINES_UNS = 'xcell_lines_json'
_LINE_DEFAULTS = {'dimX': 0, 'dimY': 1, 'smoothedPoints': None, 'drawType': 'pencil',
                  'closed': False, 'visible': True, 'strokeColor': '#4ecdc4',
                  'strokeWidth': 2, 'fillColor': None}

    def _restore_lines(self) -> list[dict[str, Any]]:
        raw = self.adata.uns.get(LINES_UNS)
        if not raw:
            return []
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            return []
        out = []
        for i, line in enumerate(stored if isinstance(stored, list) else []):
            if not isinstance(line, dict) or not line.get('points'):
                continue
            emb = line.get('embeddingName') or line.get('embedding')
            if not emb:
                continue
            fixed = {**_LINE_DEFAULTS, **{k: v for k, v in line.items() if k != 'embedding'},
                     'embeddingName': emb}
            fixed.setdefault('id', f'line_restored_{i}')
            fixed.setdefault('name', f'Line {i + 1}')
            out.append(fixed)
        return out

    def set_lines(self, lines):
        self._drawn_lines = [dict(l) for l in lines]
        self.adata.uns[LINES_UNS] = json.dumps(self._drawn_lines)
```
`__init__`: `self._drawn_lines = self._restore_lines()` (after `self.adata` is set). `prepare_export_with_lines`: replace the reduced `line_metadata` build with `adata_export.uns[LINES_UNS] = json.dumps(self._drawn_lines)` (the copy already carries it, but the explicit write keeps the projections block's precondition obvious). `_lines_on(keys)` filters `_drawn_lines` by `embeddingName in keys`; `_territories_on(keys)` filters `get_territories()` by `spec.get('embedding') in keys`. Summary: `'decorations': {'lines': [l['name'] for l in self._lines_on(set(embeddings))], 'territories': self._territories_on(set(embeddings))}`.

Route `LineData` — add the fields with the defaults above (`id: str | None = None`). `set_lines` route: `model_dump()` already includes them.

- [ ] **Step 4: Tests pass; also `tests/test_lines*.py` / any test touching `set_lines` (grep).**

- [ ] **Step 5: Commit** — `feat(lines): drawn shapes persist in uns and hydrate on load`

---

### Task 6: Frontend — hydrate lines on load, sync on change

**Files:**
- Modify: `frontend/src/store.ts` (`DatasetState`: `linesHydrated: boolean`; action `setDrawnLines(slot, lines)`), `frontend/src/store.test.ts`
- Modify: `frontend/src/hooks/useData.ts` (`fetchLines`, export `syncLinesToBackend` with full fields)
- Modify: `frontend/src/App.tsx` (two effects near the existing `handleExportH5ad`)

**Interfaces:**
- Produces: `fetchLines(slot?): Promise<DrawnLine[]>` (maps backend dicts → `DrawnLine` with `projections: []`); store `setDrawnLines(slot: DatasetSlot, lines: DrawnLine[])` sets `drawnLines` and `linesHydrated: true` for that slot (and the flat mirror when active); `syncLinesToBackend(lines, slot)` sends every `DrawnLine` field except `projections`.

- [ ] **Step 1: Store test**

```ts
describe('drawn lines hydrate per slot', () => {
  it('marks the slot hydrated and mirrors the active slot', () => {
    const s = useStore.getState()
    expect(s.datasets.primary.linesHydrated).toBe(false)
    s.setDrawnLines('primary', [{ id: 'l1', name: 'a', embeddingName: 'X_umap', dimX: 0, dimY: 1, points: [[0, 0], [1, 1]], smoothedPoints: null, visible: true, projections: [], drawType: 'pencil', closed: false, strokeColor: '#fff', strokeWidth: 2, fillColor: null }])
    const after = useStore.getState()
    expect(after.datasets.primary.linesHydrated).toBe(true)
    expect(after.drawnLines.map((l) => l.id)).toEqual(['l1'])
  })
  it('a fresh load resets hydration', () => {
    useStore.getState().setDrawnLines('primary', [])
    useStore.getState().loadDatasetIntoSlot('primary', PRIMARY)
    expect(useStore.getState().datasets.primary.linesHydrated).toBe(false)
  })
})
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement**

Store: add `linesHydrated: false` to `createDefaultDatasetState`, the `DatasetState` interface and `syncFlatFields` is not needed (it is per-dataset only). Action:
```ts
setDrawnLines: (slot, lines) =>
  set((state) => {
    const ds = state.datasets[slot]; if (!ds) return {}
    const datasets = { ...state.datasets, [slot]: { ...ds, drawnLines: lines, linesHydrated: true } }
    return slot === state.activeSlot ? { datasets, drawnLines: lines } : { datasets }
  }),
```
`useData.ts`:
```ts
export async function fetchLines(slot?: DatasetSlot): Promise<DrawnLine[]> {
  const data = await fetchJson<{ lines: Array<Omit<DrawnLine, 'projections'>> }>(appendDataset(`${API_BASE}/lines`, slot))
  return (data.lines || []).map((l) => ({ ...l, projections: [] }))
}
```
`syncLinesToBackend`: send `{ id, name, embeddingName, dimX, dimY, points, smoothedPoints, drawType, closed, visible, strokeColor, strokeWidth, fillColor }`; export it.

`App.tsx`:
```ts
// Hydrate drawn shapes for the active slot once per load.
useEffect(() => {
  if (!slotsResolved || !schema) return
  const slot = useStore.getState().activeSlot
  if (useStore.getState().datasets[slot]?.linesHydrated) return
  fetchLines(slot).then((lines) => useStore.getState().setDrawnLines(slot, lines))
    .catch((err) => console.warn('Could not load drawn shapes:', err))
}, [slotsResolved, activeSlot, schema?.filename, schema?.n_cells])

// Persist shapes on change — never before hydration, or an empty store would wipe the file's shapes.
useEffect(() => {
  const slot = useStore.getState().activeSlot
  if (!useStore.getState().datasets[slot]?.linesHydrated) return
  const t = setTimeout(() => { syncLinesToBackend(drawnLines, slot).catch((err) => console.warn('Could not save drawn shapes:', err)) }, 500)
  return () => clearTimeout(t)
}, [drawnLines])
```
(`schema` and `activeSlot` are already selected in App; check the names.)

- [ ] **Step 4: `npx vitest run && npx tsc --noEmit`.**

- [ ] **Step 5: Commit** — `feat(lines): the browser hydrates drawn shapes on load and saves them on change`

---

### Task 7: Cascade on delete-with-results; repair a vanished primary embedding

**Files:**
- Modify: `backend/xcell/adaptor.py` — `delete_cell_subset`
- Modify: `backend/tests/test_subset_tree.py`
- Modify: `frontend/src/store.ts` (`setSchema`), `frontend/src/store.test.ts`
- Modify: `frontend/src/components/CellPanel.tsx` (`handleDeleteSubset` re-hydrates lines)

**Interfaces:**
- Produces: `delete_cell_subset` result `{'name', 'dropped': [keys], 'dropped_lines': [names], 'dropped_territories': [types]}`; with `drop_derived` it also removes `X_pca_<name>_*` (+ `PCs_`, uns entries), territories and lines on any dropped embedding. Store `setSchema` re-picks `selectedEmbedding` (spatial > umap > pca > first) and nulls `embedding` when the selected one is gone.

- [ ] **Step 1: Failing tests**

```python
def test_dropping_results_also_drops_decorations_and_pc_subsets_on_its_embeddings():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.create_pca_subset([1], cell_subset='chondro')                   # Task 8 — write this test after Task 8 or mark xfail until then
    a.set_lines([LINE, {**LINE, 'id': 'l2', 'name': 'keep', 'embeddingName': 'X_umap'}])
    a.save_territories('zones', {'embedding': 'X_umap_chondro', 'sections': {
        'all': {'ring': [[0, 0], [1, 0], [1, 1]], 'cuts': [], 'anchors': []}}})
    a.save_territories('keepzones', {'embedding': 'X_umap', 'sections': {
        'all': {'ring': [[0, 0], [1, 0], [1, 1]], 'cuts': [], 'anchors': []}}})
    out = a.delete_cell_subset('chondro', drop_derived=True)
    assert out['dropped_lines'] == ['ridge'] and out['dropped_territories'] == ['zones']
    assert [l['name'] for l in a.get_lines()] == ['keep']
    assert set(a.get_territories()) == {'keepzones'}
    assert 'X_pca_chondro_noPC1' not in a.adata.obsm and 'PCs_chondro_noPC1' not in a.adata.varm


def test_deleting_without_results_keeps_decorations():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.set_lines([LINE])
    out = a.delete_cell_subset('chondro')
    assert out['dropped_lines'] == [] and len(a.get_lines()) == 1
```

Store test:
```ts
it('re-picks the primary embedding when the schema no longer lists it', () => {
  const s = useStore.getState()
  s.setSelectedEmbedding('X_umap')
  s.setSchema(makeSchema(['X_pca', 'X_spatial'], 4212))
  expect(useStore.getState().selectedEmbedding).toBe('X_spatial')
  expect(useStore.getState().embedding).toBeNull()
})
it('leaves a still-listed embedding alone', () => { … expect 'X_umap' kept })
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement** — in `delete_cell_subset` under `drop_derived`, after the existing drops: collect `dropped_embeddings = {k for k in dropped if k in ('X_pca_…', 'X_umap_…')}`, drop `X_pca_{clean}_*` keys (`obsm`, `varm[f'PCs_{suffix}']`, `uns[f'pca_{clean}']` handled by the earlier pop), then:
```python
            keep, gone = [], []
            for line in self._drawn_lines:
                (gone if line.get('embeddingName') in dropped_embeddings else keep).append(line)
            if gone:
                self.set_lines(keep)
            stored = self.get_territories()
            gone_t = [t for t, spec in stored.items() if spec.get('embedding') in dropped_embeddings]
            for t in gone_t:
                del stored[t]
            if gone_t:
                self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
```
Result gains `dropped_lines`, `dropped_territories` (empty lists otherwise). Extract a pure `_pick_embedding(names)` in the store (the same rule `loadDatasetIntoSlot` uses) and call it from `setSchema`:
```ts
setSchema: (schema) =>
  set(dsUpdateFn((state) => {
    if (state.selectedEmbedding && schema.embeddings.includes(state.selectedEmbedding)) return { schema }
    return { schema, selectedEmbedding: pickPreferredEmbedding(schema.embeddings), embedding: null }
  })),
```
`CellPanel.handleDeleteSubset`: after `refreshCellSubsets()`, `fetchLines(slot).then((lines) => setDrawnLines(slot, lines))`.

- [ ] **Step 4: Tests pass (backend + vitest + tsc).**

- [ ] **Step 5: Commit** — `feat(subsets): deleting results drops the shapes, territories and PC subsets on a subset's embeddings`

---

### Task 8: PCA Loadings and PC subsets on a subset (backend)

**Files:**
- Modify: `backend/xcell/adaptor.py` — `get_pca_loadings` (~6639), `create_pca_subset` (~6719), `list_pca_subsets` (~6800), `delete_pca_subset` (~6829), `run_pca` clearing loop (~6600–6626), `check_prerequisites` (`pca_with_loadings`)
- Modify: `backend/xcell/api/routes.py` — `GET /scanpy/pca_loadings`, `GET/POST /scanpy/pca_subsets`
- Modify: `backend/xcell/codegen.py` — `'create_pca_subset'` only tuple adds `'cell_subset'`
- Modify: `backend/tests/test_subset_tree.py`

**Interfaces:**
- Produces: `get_pca_loadings(top_n=10, cell_subset: str | None = None)`; `create_pca_subset(drop_pc_indices, suffix=None, cell_subset=None)` → keys `X_pca_<name>_<suffix>` / `PCs_<name>_<suffix>`, uns under `pca_<name>`, registry `pca_subsets`; `list_pca_subsets(cell_subset=None)` — dataset list excludes any key under a registered subset's prefix; `delete_pca_subset(obsm_key)` resolves the owner; `_pca_subset_owner(obsm_key) -> tuple[str | None, str]` (owner subset or None, suffix). Routes: `?cell_subset=` on loadings and list; `cell_subset` body field on create.

- [ ] **Step 1: Failing tests**

```python
def test_loadings_read_the_subsets_own_pca():
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    out = a.get_pca_loadings(top_n=3, cell_subset='chondro')
    assert out['n_pcs'] == 4 and out['cell_subset'] == 'chondro'
    with pytest.raises(ValueError):
        a.get_pca_loadings(cell_subset='chondro') if False else a.get_pca_loadings()   # dataset PCA absent


def test_a_pc_subset_on_a_subset_writes_under_its_prefix_and_is_recorded():
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    out = a.create_pca_subset([1, 3], cell_subset='chondro')
    assert out['obsm_key'] == 'X_pca_chondro_noPC1_3'
    assert a.adata.obsm['X_pca_chondro_noPC1_3'].shape == (60, 2)
    assert np.isnan(a.adata.obsm['X_pca_chondro_noPC1_3'][1]).all()      # outside the subset
    assert 'PCs_chondro_noPC1_3' in a.adata.varm
    assert a.adata.uns['pca_chondro']['subsets']['noPC1_3'] == {'dropped_pcs': [1, 3]}
    assert 'pca' not in a.adata.uns
    s = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert s['derived']['pca_subsets'] == ['X_pca_chondro_noPC1_3']
    assert 'X_pca_chondro_noPC1_3' in s['embeddings']


def test_the_datasets_pc_subset_list_does_not_show_subset_pcas():
    a = _adaptor(); _global_pipeline(a); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro'); a.create_pca_subset([1], cell_subset='chondro')
    a.create_pca_subset([2])
    assert [s['obsm_key'] for s in a.list_pca_subsets()] == ['X_pca_noPC2']
    assert [s['obsm_key'] for s in a.list_pca_subsets(cell_subset='chondro')] == ['X_pca_chondro_noPC1']


def test_deleting_a_subsets_pc_subset_resolves_the_owner():
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro'); a.create_pca_subset([1], cell_subset='chondro')
    a.delete_pca_subset('X_pca_chondro_noPC1')
    assert 'X_pca_chondro_noPC1' not in a.adata.obsm and 'PCs_chondro_noPC1' not in a.adata.varm
    assert 'noPC1' not in a.adata.uns['pca_chondro'].get('subsets', {})
    assert a.list_cell_subsets()[0]['derived']['pca_subsets'] == []
    with pytest.raises(ValueError):
        a.delete_pca_subset('X_pca_chondro')     # that is the subset's PCA itself


def test_a_dataset_pca_rerun_spares_subset_pc_subsets_and_a_subset_rerun_clears_its_own():
    a = _adaptor(); _global_pipeline(a); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro'); a.create_pca_subset([1], cell_subset='chondro')
    a.create_pca_subset([2])
    a.run_pca(n_comps=5)
    assert 'X_pca_chondro_noPC1' in a.adata.obsm and 'X_pca_noPC2' not in a.adata.obsm
    a.run_pca(n_comps=3, cell_subset='chondro')
    assert 'X_pca_chondro_noPC1' not in a.adata.obsm


def test_neighbors_on_a_subset_accepts_its_pc_subset():
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro'); a.create_pca_subset([1], cell_subset='chondro')
    out = a.run_neighbors(n_neighbors=5, use_rep='X_pca_chondro_noPC1', cell_subset='chondro')
    assert out['use_rep'] == 'X_pca_chondro_noPC1' and 'chondro_connectivities' in a.adata.obsp


def test_prerequisites_for_loadings_accept_the_subsets_keys():
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE); a.run_pca(n_comps=4, cell_subset='chondro')
    assert a.check_prerequisites('pca_loadings')['satisfied'] is False
    assert a.check_prerequisites('pca_loadings', cell_subset='chondro')['satisfied'] is True


def test_the_routes_take_the_subset(monkeypatch):
    a = _adaptor(); a.create_cell_subset('chondro', ACTIVE); a.run_pca(n_comps=4, cell_subset='chondro')
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    c = TestClient(app)
    assert c.get('/api/scanpy/pca_loadings?top_n=2&cell_subset=chondro').json()['n_pcs'] == 4
    r = c.post('/api/scanpy/pca_subsets', json={'drop_pc_indices': [1], 'cell_subset': 'chondro'})
    assert r.status_code == 200 and r.json()['obsm_key'] == 'X_pca_chondro_noPC1'
    assert [s['obsm_key'] for s in c.get('/api/scanpy/pca_subsets?cell_subset=chondro').json()['subsets']] == ['X_pca_chondro_noPC1']
    assert c.get('/api/scanpy/pca_subsets').json()['subsets'] == []
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement** — introduce `_pca_slots(cell_subset) -> tuple[str, str, str]` returning `('X_pca', 'PCs', 'pca')` or the `_<name>` forms (sanitising and checking the subset exists via `_subset_mask`); rewrite the four PCA-subset methods over those keys; `_pca_subset_owner(obsm_key)`: the longest registered subset name `s` with `obsm_key.startswith(f'X_pca_{s}_')` → `(s, rest)`; `obsm_key == f'X_pca_{s}'` → `ValueError("'X_pca_chondro' is the subset's own PCA; delete the subset instead")`. `run_pca` clearing: dataset run skips `suffix in names or any(suffix.startswith(s + '_') for s in names)`; subset run clears keys starting with `f'X_pca_{subset_name}_'` and forgets them in the registry. `check_prerequisites`: `pca_with_loadings` satisfied when (`'pca' in uns and 'PCs' in varm`) or (`cell_subset` and `f'pca_{cell_subset}' in uns and f'PCs_{cell_subset}' in varm`). `get_pca_loadings` returns `cell_subset` in its dict. Log params include `cell_subset` when set.

Routes: `get_pca_loadings(top_n, cell_subset: str | None = Query(None), dataset=...)`; `list_pca_subsets(cell_subset: str | None = Query(None), dataset=...)`; `PcaSubsetRequest` (find its real name) gains `cell_subset: str | None = None`. Map `KeyError → 404`.

- [ ] **Step 4: Run `tests/test_subset_tree.py tests/test_cell_subsets.py tests/test_pca*.py` — pass.**

- [ ] **Step 5: Commit** — `feat(pca): loadings and PC subsets on a named subset`

---

### Task 9: Frontend — PCA loadings and PC subsets follow the active subset

**Files:**
- Modify: `frontend/src/hooks/useData.ts` — `fetchPcaLoadings`, `usePcaLoadings`, `fetchPcaSubsets`, `createPcaSubset`
- Modify: `frontend/src/components/ScanpyModal.tsx` — call sites (~932, ~936, ~2175), PC-source picker (~2615)

**Interfaces:**
- Produces: `usePcaLoadings(topN, enabled, cellSubset?: string | null)`; `fetchPcaSubsets(slot?, cellSubset?)`; `createPcaSubset(drop, suffix, slot?, cellSubset?)`. The store's `pcaSubsets` holds whichever list was fetched last (the subset's when one is active).

- [ ] **Step 1: Implement** — thread `activeSubsetName` (already read in the modal) into the three calls; re-fetch subsets when it changes (`useEffect` deps). The picker's base option label says "(this subset's PCA)" when active (it already does for `subsetPca`); the fetched list are the subset's PC subsets. The loadings header shows `MESSAGES.pcaLoadings.subsetHeader(name)` ("Loadings of X_pca_<name>") when a subset is active — add the string.

- [ ] **Step 2: `npx tsc --noEmit`; browser check in Task 11.**

- [ ] **Step 3: Commit** — `feat(modal): PCA Loadings and PC subsets act on the active subset`

---

### Task 10: Cells panel — the tree

**Files:**
- Create: `frontend/src/lib/subsetTree.ts`, `frontend/src/lib/subsetTree.test.ts`
- Modify: `frontend/src/components/CellPanel.tsx` (Subsets section ~1563–1640; `TransferLabelsModal` wiring ~1906)
- Modify: `frontend/src/components/TransferLabelsModal.tsx` (accept `sourceColumnDefault?: string`)
- Modify: `frontend/src/hooks/useData.ts` (`createCellSubset(name, idx, {origin})`), `CellPanel.handleSaveSubset` and `ScanpyModal` inline save pass `origin: {kind: 'selection', embedding: selectedEmbedding}`
- Modify: `frontend/src/messages.ts`

**Interfaces:**
- Produces (`subsetTree.ts`):
  ```ts
  export type SubsetChip = { kind: 'step' | 'embedding' | 'leiden' | 'lines' | 'territories'; label: string; key?: string; embedding?: string }
  export function subsetChips(s: CellSubsetInfo): SubsetChip[]
  export function dropSummary(s: CellSubsetInfo): string   // "X_pca_chondro, X_umap_chondro, leiden_chondro, 2 shapes, 1 territory type"
  export function refineTarget(s: CellSubsetInfo, all: CellSubsetInfo[], obsColumns: string[]): string | null  // parent's first leiden, else 'leiden' if present, else null
  ```

- [ ] **Step 1: Tests** (`subsetTree.test.ts`):

```ts
const base = (over: Partial<CellSubsetInfo>): CellSubsetInfo => ({
  name: 'chondro', obs_key: 'subset_chondro', n_cells: 30, n_total: 60, created_at: null, description: null,
  parent: null, children: [], depth: 0, origin: null, steps: {},
  derived: { hvg: 'highly_variable__chondro', pca: 'X_pca_chondro', graph: 'chondro_connectivities', umap: ['X_umap_chondro'], leiden: ['leiden_chondro'], pca_subsets: [] },
  embeddings: ['X_pca_chondro', 'X_umap_chondro'], decorations: { lines: ['ridge', 'r2'], territories: ['zones'] }, ...over,
})
it('turns a subset into chips in a stable order', () => {
  expect(subsetChips(base({})).map((c) => c.label)).toEqual(['HVG', 'X_pca_chondro', 'kNN', 'X_umap_chondro', 'leiden_chondro', '2 shapes', '1 territory'])
})
it('summarises what a delete with results removes', () => {
  expect(dropSummary(base({}))).toBe('X_pca_chondro, X_umap_chondro, leiden_chondro, 2 shapes, 1 territory type')
  expect(dropSummary(base({ embeddings: [], decorations: { lines: [], territories: [] }, derived: { hvg: null, pca: null, graph: null, umap: [], leiden: [], pca_subsets: [] } }))).toBe('')
})
it('picks the parent leiden as the refine target, then the dataset leiden', () => {
  const parent = base({ name: 'p', derived: { ...base({}).derived, leiden: ['leiden_p'] } })
  expect(refineTarget(base({ parent: 'p' }), [parent], ['leiden'])).toBe('leiden_p')
  expect(refineTarget(base({}), [], ['leiden'])).toBe('leiden')
  expect(refineTarget(base({}), [], [])).toBeNull()
})
```

- [ ] **Step 2: Run — fail; implement the three helpers; pass.**

- [ ] **Step 3: Render** — replace the badges block. Row `style={{ ...styles.subsetRow, paddingLeft: 16 + s.depth * 12 }}`; a `└` glyph before the name when `depth > 0`. Chips: `kind === 'embedding'` → `onClick={() => setSelectedEmbedding(chip.key!)}` (title "Show this embedding"); `kind === 'leiden'` → `onClick={() => selectColorColumn(chip.key!)}` plus a tiny `⋯` opening a two-item `OverflowMenu` (`Refine <target> with this…` → `setTransferModalColumn(target); setTransferSource(chip.key)`; `Color by`); `kind === 'lines' | 'territories'` → `setSelectedEmbedding(chip.embedding!)`. Confirm row: `title` of "+ results" = `Also removes: ${dropSummary(s)}` and a second line under the confirm with that text. Pass `sourceColumnDefault={transferSource ?? undefined}` to `TransferLabelsModal` (add the prop; use it as the initial value of its source select).

- [ ] **Step 4: `npx vitest run && npx tsc --noEmit`.**

- [ ] **Step 5: Commit** — `feat(cells): subsets shown as a tree with embedding, cluster and decoration chips`

---

### Task 11: CHANGELOG, browser verification, merge

- [ ] **Step 1: CHANGELOG** — one `### Added` entry per feature (subset tree; shapes persist; delete column; PCA on a subset) and `### Fixed` for the PC-subset list bug and the vanished-embedding repair.
- [ ] **Step 2: Isolated stack** — backend `cd <wt>/backend && "<main>/.pixi/envs/default/bin/python" -m uvicorn xcell.main:app --reload --port 8100`; frontend `cd <wt>/frontend && XCELL_BACKEND=http://127.0.0.1:8100 npx vite --port 5273 --strictPort`. Load the toy dataset. Script via curl: create `chondro` (even cells), run the chain on it, create `prolif` inside it, run PCA on it. In the browser: the Subsets section shows the tree (indent), chips switch embedding / colour; draw a shape on `X_umap_chondro`, reload the tab, the shape is back; PCA Loadings with `chondro` active reads its PCA and creates `X_pca_chondro_noPC1`, which the Neighbors picker lists; delete `chondro` "+ results" — the confirm names the shape, the shape is gone, the embedding picker falls back; Delete on an obs column removes it and the colour-by clears. Check the browser console for failed requests.
- [ ] **Step 3: Full suites** — backend `pytest -q`, frontend `vitest run`, `tsc --noEmit`.
- [ ] **Step 4: Kill both servers; remove `.playwright-mcp/` and screenshots from the worktree.**
- [ ] **Step 5: Merge** — `curl -s :8000/api/schema` empty → `git checkout main && git merge --no-ff feat/subset-tree`; `git worktree remove`.
