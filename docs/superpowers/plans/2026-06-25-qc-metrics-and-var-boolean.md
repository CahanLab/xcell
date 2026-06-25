# QC metrics + boolean .var columns — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add (B) a pattern-based "add boolean `.var` column" function under Analyze → Genes, and (A) `sc.pp.calculate_qc_metrics` under Analyze → Preprocess with a `qc_vars` picker that selects those boolean columns.

**Architecture:** Two new adaptor methods + a shared `_match_var_names` helper + two `POST /scanpy/<key>` routes (TDD). Frontend adds two `SCANPY_FUNCTIONS` registry entries and one new param type `var_bool_multiselect` (chips from the already-fetched `booleanColumns`, stored as a comma string), plus post-run refresh wiring.

**Tech Stack:** Python (FastAPI, scanpy, numpy, pytest), React/TypeScript (Vite). No frontend test runner — verify via `tsc --noEmit`, `vite build`, Playwright smoke.

**Spec:** `docs/superpowers/specs/2026-06-25-qc-metrics-and-var-boolean-design.md`

**Conventions:** backend `cd backend && pixi run -e dev python -m pytest tests/ -q`; frontend `cd frontend && npx tsc --noEmit && npm run build`; commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; isolated smoke stack on :8001/:5180 (never the user's :8000/:5173).

## File map
- `backend/xcell/adaptor.py` — `_match_var_names`, `add_var_boolean_column`, `run_calculate_qc_metrics`; refactor `sum_counts_by_pattern`.
- `backend/xcell/api/routes.py` — two `POST /scanpy/*` routes + models.
- `backend/tests/test_var_boolean_qc.py` — new.
- `frontend/src/components/ScanpyModal.tsx` — `var_bool_multiselect` type + render + send; 2 registry entries; post-run wiring.

---

### Task 1: Backend — `_match_var_names` + `add_var_boolean_column`

**Files:** Modify `backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`; create `backend/tests/test_var_boolean_qc.py`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_var_boolean_qc.py`:

```python
"""Pattern-based boolean .var columns + calculate_qc_metrics."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    genes = ["mt-Nd1", "mt-Co1", "Actb", "Gapdh", "Rpl13", "Rps6"]
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.integers(0, 30, (12, 6)).astype(np.float32)))
    ad.var_names = genes
    return ad


def test_add_var_boolean_prefix():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    assert r["n_genes_matched"] == 2
    assert a.adata.var["mt"].dtype == bool
    assert a.adata.var["mt"].tolist() == [True, True, False, False, False, False]
    # appears in the boolean-columns list used by the qc_vars picker
    assert "mt" in {c["name"] for c in a.get_var_boolean_columns()}


def test_add_var_boolean_regex():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.add_var_boolean_column("ribo", "^Rp[ls]", match_mode="regex")
    assert r["n_genes_matched"] == 2
    assert a.adata.var["ribo"].tolist() == [False, False, False, False, True, True]


def test_add_var_boolean_rejects_empty_and_nomatch():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.add_var_boolean_column("x", "", match_mode="prefix")
    with pytest.raises(ValueError):
        a.add_var_boolean_column("x", "ZZZ", match_mode="prefix")


def test_add_var_boolean_rejects_nonbool_name_collision():
    ad = _adata()
    ad.var["mean_counts"] = np.arange(6, dtype=float)  # numeric column
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.add_var_boolean_column("mean_counts", "mt-", match_mode="prefix")


def test_route_add_var_boolean(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/add_var_boolean", json={
        "name": "mt", "pattern": "mt-", "match_mode": "prefix",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_genes_matched"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_boolean_qc.py -q`
Expected: FAIL (`add_var_boolean_column` missing).

- [ ] **Step 3: Refactor the matcher + implement the method**

In `backend/xcell/adaptor.py`, replace the matching block inside
`sum_counts_by_pattern` (the `if not pattern … mask = …` lines, currently
`adaptor.py:571-585`'s match portion) so it delegates to a shared helper. First
add the helper immediately above `sum_counts_by_pattern`:

```python
    def _match_var_names(self, pattern: str, match_mode: str = 'prefix') -> np.ndarray:
        """Boolean mask over var_names matching a prefix or regex."""
        import re
        if not pattern:
            raise ValueError("pattern must be non-empty")
        if match_mode not in ('prefix', 'regex'):
            raise ValueError("match_mode must be 'prefix' or 'regex'")
        names = self.adata.var_names
        if match_mode == 'prefix':
            return np.asarray(names.str.startswith(pattern))
        rx = re.compile(pattern)
        return np.array([bool(rx.search(str(n))) for n in names])
```

Then in `sum_counts_by_pattern`, replace:

```python
        import re
        import scipy.sparse as sp
        if not pattern:
            raise ValueError("pattern must be non-empty")
        if match_mode not in ('prefix', 'regex'):
            raise ValueError("match_mode must be 'prefix' or 'regex'")
        names = self.adata.var_names
        if match_mode == 'prefix':
            mask = np.asarray(names.str.startswith(pattern))
        else:
            rx = re.compile(pattern)
            mask = np.array([bool(rx.search(str(n))) for n in names])
        n_matched = int(mask.sum())
```

with:

```python
        import re
        import scipy.sparse as sp
        mask = self._match_var_names(pattern, match_mode)
        n_matched = int(mask.sum())
```

(`re` stays imported for the `obs_name` sanitization later in that method.)

Now add `add_var_boolean_column` after `sum_counts_by_pattern`:

```python
    def add_var_boolean_column(
        self, name: str, pattern: str, match_mode: str = 'prefix',
    ) -> dict[str, Any]:
        """Add a boolean .var column flagging genes whose names match a
        prefix/regex (e.g. mitochondrial, a species of origin).

        Raises ValueError on empty name/pattern, no match, or a name that
        collides with an existing non-boolean .var column.
        """
        if not name:
            raise ValueError("column name must be non-empty")
        if name in self.adata.var.columns and self.adata.var[name].dtype != bool:
            raise ValueError(f"'{name}' already exists and is not boolean")
        mask = self._match_var_names(pattern, match_mode)
        n_matched = int(mask.sum())
        if n_matched == 0:
            raise ValueError(f"no genes match {match_mode} '{pattern}'")
        self.adata.var[name] = mask.astype(bool)
        return {"name": name, "n_genes_matched": n_matched}
```

- [ ] **Step 4: Implement the route**

In `backend/xcell/api/routes.py`, near the other species routes (after
`scanpy_assign_species`) add:

```python
class AddVarBooleanRequest(BaseModel):
    name: str
    pattern: str
    match_mode: str = "prefix"


@router.post("/scanpy/add_var_boolean")
def scanpy_add_var_boolean(
    request: AddVarBooleanRequest, dataset: str | None = Query(None)
):
    """Add a boolean .var column from a gene-name prefix/regex."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.add_var_boolean_column(
            name=request.name, pattern=request.pattern, match_mode=request.match_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 5: Run tests + existing species tests**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_boolean_qc.py tests/test_species_counts.py -q`
Expected: PASS (new + the refactored `sum_counts_by_pattern` still green).

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_var_boolean_qc.py
git commit -m "$(printf 'backend: add_var_boolean_column (+ shared _match_var_names helper)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Backend — `run_calculate_qc_metrics`

**Files:** Modify `backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`; extend `backend/tests/test_var_boolean_qc.py`.

- [ ] **Step 1: Add failing tests**

Append to `backend/tests/test_var_boolean_qc.py`:

```python
def test_calculate_qc_metrics_adds_columns():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    r = a.run_calculate_qc_metrics(qc_vars=["mt"], log1p=True)
    obs = a.adata.obs.columns
    assert "total_counts" in obs
    assert "n_genes_by_counts" in obs
    assert "log1p_total_counts" in obs
    assert "pct_counts_mt" in obs
    assert "total_counts_mt" in obs
    assert r["qc_vars"] == ["mt"]


def test_calculate_qc_metrics_log1p_false():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.run_calculate_qc_metrics(qc_vars=None, log1p=False)
    assert "total_counts" in a.adata.obs.columns
    assert "log1p_total_counts" not in a.adata.obs.columns


def test_calculate_qc_metrics_rejects_non_boolean_qc_var():
    ad = _adata()
    ad.var["mean_counts"] = np.arange(6, dtype=float)
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.run_calculate_qc_metrics(qc_vars=["mean_counts"])


def test_calculate_qc_metrics_accepts_comma_string():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    r = a.run_calculate_qc_metrics(qc_vars="mt", log1p=True)
    assert "pct_counts_mt" in a.adata.obs.columns
    assert r["qc_vars"] == ["mt"]


def test_route_calculate_qc_metrics(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/calculate_qc_metrics", json={
        "qc_vars": "mt", "log1p": True,
    })
    assert resp.status_code == 200, resp.text
    assert "pct_counts_mt" in a.adata.obs.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_boolean_qc.py -k qc_metrics -q`
Expected: FAIL (`run_calculate_qc_metrics` missing).

- [ ] **Step 3: Implement the adaptor method**

In `backend/xcell/adaptor.py`, after `add_var_boolean_column` add:

```python
    def run_calculate_qc_metrics(
        self, qc_vars=None, percent_top=None, log1p: bool = True,
    ) -> dict[str, Any]:
        """Run sc.pp.calculate_qc_metrics(inplace=True).

        qc_vars: list or comma-separated string of boolean .var columns.
        percent_top: None (default; skips top-N columns) or list/comma-string
        of ints. log1p adds the log1p_* columns.
        """
        if isinstance(qc_vars, str):
            qc_vars = [c.strip() for c in qc_vars.split(',') if c.strip()]
        qc_vars = list(qc_vars or [])
        for v in qc_vars:
            if v not in self.adata.var.columns:
                raise ValueError(f"qc_var '{v}' not found in .var")
            if self.adata.var[v].dtype != bool:
                raise ValueError(f"qc_var '{v}' is not boolean")
        if isinstance(percent_top, str):
            percent_top = [int(x) for x in percent_top.split(',') if x.strip()] or None
        obs_before = set(self.adata.obs.columns)
        var_before = set(self.adata.var.columns)
        sc.pp.calculate_qc_metrics(
            self.adata, qc_vars=qc_vars, percent_top=percent_top,
            log1p=log1p, inplace=True,
        )
        return {
            "qc_vars": qc_vars,
            "n_obs_columns": len(set(self.adata.obs.columns) - obs_before),
            "n_var_columns": len(set(self.adata.var.columns) - var_before),
        }
```

- [ ] **Step 4: Implement the route**

In `backend/xcell/api/routes.py`, after `scanpy_add_var_boolean` add:

```python
class CalculateQcMetricsRequest(BaseModel):
    qc_vars: str | None = None      # comma-separated boolean .var columns
    percent_top: str | None = None  # comma-separated ints; blank -> None
    log1p: bool = True


@router.post("/scanpy/calculate_qc_metrics")
def scanpy_calculate_qc_metrics(
    request: CalculateQcMetricsRequest, dataset: str | None = Query(None)
):
    """sc.pp.calculate_qc_metrics with user-selected qc_vars."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_calculate_qc_metrics(
            qc_vars=request.qc_vars, percent_top=request.percent_top,
            log1p=request.log1p,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 5: Run the new tests + full suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_boolean_qc.py -q && pixi run -e dev python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_var_boolean_qc.py
git commit -m "$(printf 'backend: run_calculate_qc_metrics route (qc_vars from boolean .var columns)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Frontend — `var_bool_multiselect` param type

**Files:** Modify `frontend/src/components/ScanpyModal.tsx`

- [ ] **Step 1: Add the type to ParamDef**

In the `ParamDef.type` union (~:150) add `'var_bool_multiselect'`:

```ts
  type: 'number' | 'text' | 'select' | 'gene_subset' | 'textarea' | 'pc_source_select' | 'layer_select' | 'graph_select' | 'obs_column_select' | 'var_bool_multiselect'
```

- [ ] **Step 2: Render chips storing a comma-separated string in paramValues**

In the param render, add a branch BEFORE the `obs_column_select` branch
(search `param.type === 'obs_column_select' ?`) — insert this ternary arm just
above it:

```tsx
                      ) : param.type === 'var_bool_multiselect' ? (
                        booleanColumns.length === 0 ? (
                          <div style={{ fontSize: '11px', color: '#888' }}>
                            No boolean .var columns yet — add one via Genes → Add boolean column.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {booleanColumns.map((col) => {
                              const selected = String(paramValues[param.name] ?? '')
                                .split(',').map((s) => s.trim()).filter(Boolean)
                              const isSel = selected.includes(col.name)
                              return (
                                <button
                                  key={col.name}
                                  onClick={() => {
                                    const next = isSel
                                      ? selected.filter((c) => c !== col.name)
                                      : [...selected, col.name]
                                    handleParamChange(param.name, next.join(','))
                                  }}
                                  style={{
                                    padding: '4px 10px', fontSize: '11px',
                                    backgroundColor: isSel ? '#4ecdc4' : '#1a1a2e',
                                    color: isSel ? '#000' : '#aaa',
                                    border: `1px solid ${isSel ? '#4ecdc4' : '#333'}`,
                                    borderRadius: '12px', cursor: 'pointer',
                                  }}
                                >
                                  {col.name} <span style={{ opacity: 0.7 }}>({col.n_true})</span>
                                </button>
                              )
                            })}
                          </div>
                        )```

Insert it as an **anchored replacement**: find the existing opener
`) : param.type === 'obs_column_select' ? (` and prepend the new arm so the
chain reads `… ) : param.type === 'var_bool_multiselect' ? ( <chips block above> ) : param.type === 'obs_column_select' ? (`.
(`booleanColumns` items are `{ name, n_true, n_total }`.)

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "$(printf 'frontend: var_bool_multiselect param type (boolean .var column chips)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Frontend — registry entries + post-run wiring

**Files:** Modify `frontend/src/components/ScanpyModal.tsx`

- [ ] **Step 1: Add `calculate_qc_metrics` under Preprocess**

In `SCANPY_FUNCTIONS.preprocessing.functions` (the `preprocessing` category), add
a new function entry (place it after `highly_variable_genes` or at the end of the
preprocessing functions block):

```ts
      calculate_qc_metrics: {
        label: 'QC Metrics',
        description: 'Compute standard QC metrics (sc.pp.calculate_qc_metrics) into .obs/.var, including per-qc_var pct_counts (e.g. % mitochondrial).',
        prerequisites: [],
        params: [
          { name: 'qc_vars', label: 'QC vars (boolean .var columns)', type: 'var_bool_multiselect', default: null, description: 'Boolean .var columns to compute pct/total counts for (e.g. mt). Add columns via Genes → Add boolean column.' },
          { name: 'log1p', label: 'Add log1p columns', type: 'select', default: 'true', options: ['true', 'false'], description: 'Also write log1p_* QC columns' },
          { name: 'percent_top', label: 'Percent-top (optional)', type: 'text', default: null, description: 'Comma-separated ints (e.g. 50,100,200,500). Blank skips top-N columns.' },
        ],
      },
```

Note: the `log1p` select sends the string `'true'`/`'false'`. Confirm the run
handler coerces select booleans (search the run handler for how other
true/false `select` params like `scale`/`use_kneedle` are sent). If it sends the
raw string, the Pydantic `bool` field coerces `'true'`/`'false'` correctly
(FastAPI parses these). If issues arise, coerce in the handler.

- [ ] **Step 2: Add `add_var_boolean` under Genes**

In `SCANPY_FUNCTIONS.gene_analysis.functions`, add:

```ts
      add_var_boolean: {
        label: 'Add boolean column',
        description: 'Flag genes whose names match a prefix/regex as a boolean .var column (e.g. mt from ^mt-, or a species). Usable as a QC var.',
        prerequisites: [],
        params: [
          { name: 'name', label: 'Column name', type: 'text', default: 'mt', description: 'New boolean .var column name' },
          { name: 'pattern', label: 'Gene-name pattern', type: 'text', default: '^mt-', description: 'Prefix (e.g. GRCh38_) or regex (e.g. ^mt-)' },
          { name: 'match_mode', label: 'Match mode', type: 'select', default: 'regex', options: ['prefix', 'regex'], description: 'How to match gene names' },
        ],
      },
```

- [ ] **Step 3: Post-run refresh wiring**

In the post-run refresh list (the `['filter_genes', …, 'assign_species'].includes(selectedFunction)` array) add both keys:

```ts
      if ([... , 'embedding_from_obs', 'sum_counts_by_pattern', 'assign_species', 'add_var_boolean', 'calculate_qc_metrics'].includes(selectedFunction)) {
        await refreshSchema()
        refreshObsSummaries()
      }
```

(Keep the existing entries; just append `'add_var_boolean'` and
`'calculate_qc_metrics'`. `refreshSchema` updates `.var`/`.obs`; `booleanColumns`
re-fetches automatically because it depends on `scanpyActionHistory`, which every
run appends to — so a freshly added column shows in the qc_vars chips.)

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "$(printf 'frontend: Analyze entries for add boolean .var column + QC metrics\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: End-to-end verification + finish

**Files:** none.

- [ ] **Step 1: Full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all green (previous total + ~10 new).

- [ ] **Step 2: Frontend typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 3: Playwright smoke (isolated stack)**

Start backend :8001 (adE11 has `mt-`/`Rpl`/`Rps` genes) + vite :5180. Then via
Analyze:
- Genes → *Add boolean column*: name `mt2`, pattern `^mt-`, regex → Run; confirm
  success (n matched).
- Preprocess → *QC Metrics*: confirm the `qc_vars` chips include `mt2` (and the
  existing `mt`/`ribo`); check `mt2`, Run; confirm `pct_counts_mt2` appears in the
  Cells pane (Continuous tab).

Stop both servers; delete screenshots / `.playwright-mcp`.

- [ ] **Step 4: Mark spec implemented + commit**

Append an "Implemented (2026-06-25)" note near the top of the spec; commit.

- [ ] **Step 5: Finish the branch**

Use superpowers:finishing-a-development-branch.

---

## Self-review notes
- **Spec coverage:** add boolean `.var` column (Task 1, 4) ✓; calculate_qc_metrics + qc_vars picker (Task 2, 3, 4) ✓; shared matcher refactor (Task 1) ✓; testing (Tasks 1, 2, 5) ✓.
- **Naming consistency:** `add_var_boolean_column`/`/scanpy/add_var_boolean`/`AddVarBooleanRequest`; `run_calculate_qc_metrics`/`/scanpy/calculate_qc_metrics`/`CalculateQcMetricsRequest`; param type `var_bool_multiselect`; `_match_var_names`.
- **Reuse:** `booleanColumns` (already fetched, refetched on run) powers the qc_vars chips; `sum_counts_by_pattern` now shares `_match_var_names` (its tests must stay green — Task 1 Step 5).
- **Out of scope:** general multi-select type; percent_top UI beyond text; var-column editing/deletion.
