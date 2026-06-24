# Custom .obs embeddings + multi-species counts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add (1) a custom 2-D embedding built from two numeric `.obs` columns with optional per-axis log, and (2) multi-species (PDX) count summation by gene-name prefix/regex plus a species-assignment step — all as ScanpyModal registry functions.

**Architecture:** Three new adaptor methods + three `POST /scanpy/<key>` routes (TDD). The frontend adds three entries to the existing `SCANPY_FUNCTIONS` registry (one new *Multi-genome* category), an `obsDtype` filter option on the existing `obs_column_select` param type, and post-run refresh/auto-select wiring. No new modals.

**Tech Stack:** Python (FastAPI, pandas/anndata/numpy/scipy, pytest), React/TypeScript (Vite). No frontend test runner — frontend verified via `tsc --noEmit`, `vite build`, Playwright smoke.

**Spec:** `docs/superpowers/specs/2026-06-24-obs-embedding-and-multispecies-counts-design.md`

**Conventions:**
- Backend tests: `cd backend && pixi run -e dev python -m pytest tests/ -q`
- Frontend: `cd frontend && npx tsc --noEmit && npm run build`
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Isolated smoke stack (never the user's :8000/:5173): backend on :8001, vite `XCELL_BACKEND=http://127.0.0.1:8001 npx vite --port 5180 --strictPort`.

## File map
- `backend/xcell/adaptor.py` — `create_obs_embedding`, `sum_counts_by_pattern`, `assign_species`.
- `backend/xcell/api/routes.py` — three `POST /scanpy/*` routes + request models.
- `backend/tests/test_obs_embedding.py`, `backend/tests/test_species_counts.py` — new.
- `frontend/src/components/ScanpyModal.tsx` — `obsDtype` on ParamDef + numeric obs fetch + render branch; 3 registry entries + `multigenome` category; post-run wiring.

---

### Task 1: Backend — `create_obs_embedding`

**Files:**
- Modify `backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`
- Create `backend/tests/test_obs_embedding.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_obs_embedding.py`:

```python
"""Custom 2-D embedding from two numeric .obs columns."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((30, 4)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(4)]
    ad.obs["total_counts"] = rng.integers(1, 1000, 30).astype(float)
    ad.obs["n_genes"] = rng.integers(1, 500, 30).astype(float)
    ad.obs["score"] = rng.standard_normal(30)        # has negatives
    ad.obs["celltype"] = ["a", "b"] * 15             # categorical
    return ad


def test_create_obs_embedding_stores_2col_obsm():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.create_obs_embedding("total_counts", "n_genes")
    key = r["embedding_name"]
    assert key == "X_total_counts_vs_n_genes"
    assert a.adata.obsm[key].shape == (30, 2)
    assert np.allclose(a.adata.obsm[key][:, 0], a.adata.obs["total_counts"].to_numpy())


def test_create_obs_embedding_logs_requested_axis():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.create_obs_embedding("total_counts", "n_genes", log_axes="x")
    coords = a.adata.obsm[r["embedding_name"]]
    assert np.allclose(coords[:, 0], np.log1p(a.adata.obs["total_counts"].to_numpy()))
    assert np.allclose(coords[:, 1], a.adata.obs["n_genes"].to_numpy())  # y untouched


def test_create_obs_embedding_rejects_non_numeric():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.create_obs_embedding("total_counts", "celltype")


def test_create_obs_embedding_rejects_log_on_negative():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.create_obs_embedding("score", "n_genes", log_axes="x")


def test_create_obs_embedding_rejects_duplicate_name():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.create_obs_embedding("total_counts", "n_genes", name="X_mine")
    with pytest.raises(ValueError):
        a.create_obs_embedding("total_counts", "n_genes", name="X_mine")


def test_route_embedding_from_obs(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/embedding_from_obs", json={
        "col_x": "total_counts", "col_y": "n_genes", "log_axes": "both",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["embedding_name"] == "X_total_counts_vs_n_genes"


def test_route_embedding_from_obs_bad_column(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/embedding_from_obs", json={
        "col_x": "total_counts", "col_y": "celltype",
    })
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_obs_embedding.py -q`
Expected: FAIL (`DataAdaptor` has no attribute `create_obs_embedding`).

- [ ] **Step 3: Implement the adaptor method**

In `backend/xcell/adaptor.py`, after `get_embedding` (ends ~:522) add:

```python
    def create_obs_embedding(
        self, col_x: str, col_y: str, log_axes: str = 'none',
        name: str | None = None,
    ) -> dict[str, Any]:
        """Build a 2-D embedding (obsm['X_...']) from two numeric .obs columns.

        log_axes in {'none','x','y','both'} applies log1p to the chosen axis
        (requires non-negative values). Raises ValueError on bad input or a
        duplicate embedding name.
        """
        obs = self.adata.obs
        for c in (col_x, col_y):
            if c not in obs.columns:
                raise ValueError(f"obs column '{c}' not found")
            if not pd.api.types.is_numeric_dtype(obs[c].dtype):
                raise ValueError(f"obs column '{c}' is not numeric")
        if log_axes not in ('none', 'x', 'y', 'both'):
            raise ValueError(f"log_axes must be none/x/y/both, got {log_axes!r}")
        x = obs[col_x].to_numpy(dtype=float)
        y = obs[col_y].to_numpy(dtype=float)
        if log_axes in ('x', 'both'):
            if np.nanmin(x) < 0:
                raise ValueError(f"log requires non-negative values; '{col_x}' has negatives")
            x = np.log1p(x)
        if log_axes in ('y', 'both'):
            if np.nanmin(y) < 0:
                raise ValueError(f"log requires non-negative values; '{col_y}' has negatives")
            y = np.log1p(y)
        if name:
            key = name if name.startswith('X_') else f'X_{name}'
        else:
            key = f'X_{col_x}_vs_{col_y}'
        if key in self.adata.obsm:
            raise ValueError(f"embedding '{key}' already exists")
        self.adata.obsm[key] = np.column_stack([x, y]).astype(float)
        return {"embedding_name": key, "n_cells": self.n_cells}
```

- [ ] **Step 4: Implement the route**

In `backend/xcell/api/routes.py`, near the other scanpy routes (e.g. after the
`/scanpy/highly_variable_genes` block, ~:1755) add a request model + route:

```python
class EmbeddingFromObsRequest(BaseModel):
    col_x: str
    col_y: str
    log_axes: str = "none"
    name: str | None = None


@router.post("/scanpy/embedding_from_obs")
def scanpy_embedding_from_obs(
    request: EmbeddingFromObsRequest, dataset: str | None = Query(None)
):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.create_obs_embedding(
            col_x=request.col_x, col_y=request.col_y,
            log_axes=request.log_axes, name=request.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_obs_embedding.py -q`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_obs_embedding.py
git commit -m "$(printf 'backend: create_obs_embedding (2D obsm from two numeric .obs columns)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Backend — `sum_counts_by_pattern`

**Files:**
- Modify `backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`
- Create `backend/tests/test_species_counts.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_species_counts.py`:

```python
"""Per-species count summation by gene-name pattern + species assignment."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 4 human (GRCh38_) + 3 mouse (mm10___) genes; integer-ish counts in .X.
    genes = ["GRCh38_A1BG", "GRCh38_TP53", "GRCh38_EGFR", "GRCh38_MYC",
             "mm10___Xkr4", "mm10___Sox2", "mm10___Actb"]
    rng = np.random.default_rng(1)
    X = rng.integers(0, 20, size=(10, 7)).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = genes
    return ad, X


def test_sum_counts_prefix_sums_matching_genes():
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.sum_counts_by_pattern("GRCh38_", match_mode="prefix")
    assert r["n_genes_matched"] == 4
    assert r["obs_name"] == "GRCh38_counts"
    expected = X[:, :4].sum(axis=1)
    assert np.allclose(a.adata.obs["GRCh38_counts"].to_numpy(), expected)


def test_sum_counts_regex():
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.sum_counts_by_pattern("^mm10", match_mode="regex", obs_name="mouse")
    assert r["n_genes_matched"] == 3
    assert np.allclose(a.adata.obs["mouse"].to_numpy(), X[:, 4:].sum(axis=1))


def test_sum_counts_no_match_raises():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.sum_counts_by_pattern("ZZZ_", match_mode="prefix")


def test_assign_species_threshold():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    # Plant explicit count columns: cell0 human-pure, cell1 mouse-pure, cell2 mixed, cell3 empty.
    a.adata.obs["human_counts"] = np.array([100.0, 1, 50] + [10] * 7)
    a.adata.obs["mouse_counts"] = np.array([0.0, 100, 50] + [10] * 7)
    a.adata.obs.loc[a.adata.obs.index[3], "human_counts"] = 0.0
    a.adata.obs.loc[a.adata.obs.index[3], "mouse_counts"] = 0.0
    r = a.assign_species(["human_counts", "mouse_counts"], threshold=0.9)
    sp = a.adata.obs[r["obs_name"]].astype(str).to_numpy()
    assert sp[0] == "human"
    assert sp[1] == "mouse"
    assert sp[2] == "mixed"
    assert sp[3] == "unassigned"
    assert r["obs_name"] == "species"


def test_assign_species_requires_two_columns():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.adata.obs["human_counts"] = np.ones(10)
    with pytest.raises(ValueError):
        a.assign_species(["human_counts"])


def test_route_sum_counts(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/sum_counts_by_pattern", json={
        "pattern": "GRCh38_", "match_mode": "prefix",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_genes_matched"] == 4


def test_route_assign_species(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.adata.obs["human_counts"] = np.array([100.0] + [10] * 9)
    a.adata.obs["mouse_counts"] = np.array([0.0] + [10] * 9)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/assign_species", json={
        "count_columns": "human_counts, mouse_counts", "threshold": 0.9,
    })
    assert resp.status_code == 200, resp.text
    assert "counts" in resp.json()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_species_counts.py -q`
Expected: FAIL (`sum_counts_by_pattern` missing).

- [ ] **Step 3: Implement `sum_counts_by_pattern`**

In `backend/xcell/adaptor.py`, after `create_obs_embedding` add:

```python
    def sum_counts_by_pattern(
        self, pattern: str, match_mode: str = 'prefix',
        obs_name: str | None = None, layer: str = 'counts',
    ) -> dict[str, Any]:
        """Sum counts of genes whose names match a prefix/regex into .obs.

        match_mode in {'prefix','regex'}. Reads layers[layer] (default
        'counts') if present, else .X. Raises ValueError on empty pattern or
        no match.
        """
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
        if n_matched == 0:
            raise ValueError(f"no genes match {match_mode} '{pattern}'")
        src = layer if layer else 'counts'
        M = self.adata.layers[src] if (src != 'X' and src in self.adata.layers) else self.adata.X
        sub = M[:, mask]
        sums = (np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub)
                else np.asarray(sub).sum(axis=1).ravel())
        if not obs_name:
            base = re.sub(r'[^0-9A-Za-z]+', '', pattern) or 'species'
            obs_name = f"{base}_counts"
        self.adata.obs[obs_name] = sums.astype(float)
        return {"obs_name": obs_name, "n_genes_matched": n_matched}
```

- [ ] **Step 4: Implement `assign_species`**

Immediately after, add:

```python
    def assign_species(
        self, count_columns, labels=None, obs_name: str = 'species',
        threshold: float = 0.9,
    ) -> dict[str, Any]:
        """Assign each cell a species from per-species count columns.

        For each cell, the argmax-fraction species is assigned iff its fraction
        >= threshold, else 'mixed'; zero-total cells are 'unassigned'.
        """
        if isinstance(count_columns, str):
            count_columns = [c.strip() for c in count_columns.split(',') if c.strip()]
        count_columns = list(count_columns)
        if len(count_columns) < 2:
            raise ValueError("assign_species needs at least 2 count columns")
        obs = self.adata.obs
        for c in count_columns:
            if c not in obs.columns:
                raise ValueError(f"obs column '{c}' not found")
            if not pd.api.types.is_numeric_dtype(obs[c].dtype):
                raise ValueError(f"obs column '{c}' is not numeric")
        if labels is None:
            labels = [c[:-7] if c.endswith('_counts') else c for c in count_columns]
        elif isinstance(labels, str):
            labels = [s.strip() for s in labels.split(',') if s.strip()]
        labels = list(labels)
        if len(labels) != len(count_columns):
            raise ValueError("labels must match count_columns length")
        if not (0 < threshold <= 1):
            raise ValueError("threshold must be in (0, 1]")
        mat = np.column_stack([obs[c].to_numpy(dtype=float) for c in count_columns])
        total = mat.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(total[:, None] > 0, mat / total[:, None], 0.0)
        argmax = frac.argmax(axis=1)
        labels_arr = np.array(labels, dtype=object)
        assigned = np.where(frac[np.arange(len(total)), argmax] >= threshold,
                            labels_arr[argmax], 'mixed').astype(object)
        assigned[total <= 0] = 'unassigned'
        cats = [c for c in (list(dict.fromkeys(labels)) + ['mixed', 'unassigned'])
                if c in set(assigned.tolist())]
        self.adata.obs[obs_name] = pd.Categorical(assigned, categories=cats)
        counts = {c: int((assigned == c).sum()) for c in cats}
        return {"obs_name": obs_name, "counts": counts}
```

- [ ] **Step 5: Implement the routes**

In `backend/xcell/api/routes.py`, after the `embedding_from_obs` route add:

```python
class SumCountsRequest(BaseModel):
    pattern: str
    match_mode: str = "prefix"
    obs_name: str | None = None
    layer: str = "counts"


@router.post("/scanpy/sum_counts_by_pattern")
def scanpy_sum_counts_by_pattern(
    request: SumCountsRequest, dataset: str | None = Query(None)
):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.sum_counts_by_pattern(
            pattern=request.pattern, match_mode=request.match_mode,
            obs_name=request.obs_name, layer=request.layer,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class AssignSpeciesRequest(BaseModel):
    count_columns: str  # comma-separated obs column names
    labels: str | None = None  # comma-separated, optional
    obs_name: str = "species"
    threshold: float = 0.9


@router.post("/scanpy/assign_species")
def scanpy_assign_species(
    request: AssignSpeciesRequest, dataset: str | None = Query(None)
):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.assign_species(
            count_columns=request.count_columns, labels=request.labels,
            obs_name=request.obs_name, threshold=request.threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_species_counts.py -q`
Expected: PASS (7 tests).

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_species_counts.py
git commit -m "$(printf 'backend: sum_counts_by_pattern + assign_species (multi-species/PDX)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Frontend — `obsDtype` filter on `obs_column_select`

**Files:** Modify `frontend/src/components/ScanpyModal.tsx`

- [ ] **Step 1: Add `obsDtype` to ParamDef**

In `ScanpyModal.tsx` change the `ParamDef` interface (~:147) to add the field:

```ts
interface ParamDef {
  name: string
  label: string
  type: 'number' | 'text' | 'select' | 'gene_subset' | 'textarea' | 'pc_source_select' | 'layer_select' | 'graph_select' | 'obs_column_select'
  default: string | number | null
  description: string
  options?: string[]
  obsDtype?: 'numeric' | 'category'
  visibleWhen?: { param: string; value: string }
}
```

- [ ] **Step 2: Add numeric obs state + fetch both dtypes**

Find the `availableObsColumns` state declaration (search `setAvailableObsColumns`) and add next to it:

```ts
  const [availableNumericObsColumns, setAvailableNumericObsColumns] = useState<string[]>([])
```

Then change the obs-columns effect (~:754) so it loads both category and numeric lists:

```ts
  useEffect(() => {
    const needsObsColumns = (functionDef?.params || []).some((p) => p.type === 'obs_column_select')
    if (!needsObsColumns) return
    fetch(appendDataset(`${API_BASE}/obs/summaries`))
      .then((res) => res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`)))
      .then((data) => {
        const rows = (data.summaries || data || []) as { name: string; dtype?: string }[]
        setAvailableObsColumns(rows.filter((s) => s.dtype === 'category').map((s) => s.name))
        setAvailableNumericObsColumns(rows.filter((s) => s.dtype === 'numeric').map((s) => s.name))
      })
      .catch(() => { setAvailableObsColumns([]); setAvailableNumericObsColumns([]) })
  }, [selectedFunction, functionDef, activeSlot, scanpyActionHistory])
```

- [ ] **Step 3: Render the right list + placeholder per `obsDtype`**

Replace the `obs_column_select` render branch (~:2049) with:

```tsx
                      ) : param.type === 'obs_column_select' ? (
                        <select
                          style={styles.paramInput}
                          value={paramValues[param.name] ?? ''}
                          onChange={(e) => handleParamChange(param.name, e.target.value)}
                        >
                          <option value="">
                            {param.obsDtype === 'numeric' ? '— select column —' : '— treat as one tissue —'}
                          </option>
                          {(param.obsDtype === 'numeric' ? availableNumericObsColumns : availableObsColumns).map((c) => (
                            <option key={c} value={c}>{c}</option>
                          ))}
                        </select>
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "$(printf 'frontend: obs_column_select gains an obsDtype (numeric|category) filter\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Frontend — registry entries + post-run wiring

**Files:** Modify `frontend/src/components/ScanpyModal.tsx`

- [ ] **Step 1: Add `embedding_from_obs` under Cell Analysis**

In `SCANPY_FUNCTIONS.cell_analysis.functions` (the `cell_analysis` category, ~:246), add a new function entry (place it after the existing functions in that block):

```ts
      embedding_from_obs: {
        label: 'Embedding from .obs',
        description: 'Build a 2-D embedding from two numeric .obs columns (e.g. a barnyard plot of two species count columns). Optionally log each axis.',
        prerequisites: [],
        params: [
          { name: 'col_x', label: 'X column', type: 'obs_column_select', obsDtype: 'numeric', default: null, description: 'Numeric .obs column for the X axis' },
          { name: 'col_y', label: 'Y column', type: 'obs_column_select', obsDtype: 'numeric', default: null, description: 'Numeric .obs column for the Y axis' },
          { name: 'log_axes', label: 'Log axes', type: 'select', default: 'none', options: ['none', 'x', 'y', 'both'], description: 'Apply log1p to the chosen axis/axes' },
          { name: 'name', label: 'Name (optional)', type: 'text', default: null, description: 'Embedding name; default X_<x>_vs_<y>' },
        ],
      },
```

- [ ] **Step 2: Add the Multi-genome category**

Add a new top-level category to `SCANPY_FUNCTIONS` (after `spatial_analysis`, ~:374 block ends):

```ts
  multigenome: {
    label: 'Multi-genome',
    functions: {
      sum_counts_by_pattern: {
        label: 'Sum species counts',
        description: 'Sum counts of genes whose names match a prefix or regex into a new .obs column (e.g. GRCh38_ for human, mm10_ for mouse in a PDX).',
        prerequisites: [],
        params: [
          { name: 'pattern', label: 'Gene-name pattern', type: 'text', default: 'GRCh38_', description: 'Prefix (e.g. GRCh38_) or regex (e.g. ^mm10)' },
          { name: 'match_mode', label: 'Match mode', type: 'select', default: 'prefix', options: ['prefix', 'regex'], description: 'How to match gene names' },
          { name: 'obs_name', label: 'Output column (optional)', type: 'text', default: null, description: 'Default derived from the pattern (e.g. GRCh38_counts)' },
          { name: 'layer', label: 'Source matrix', type: 'layer_select', default: 'counts', description: 'Counts layer to sum (falls back to .X)' },
        ],
      },
      assign_species: {
        label: 'Assign species',
        description: 'Classify each cell by which species dominates its counts. Cells below the purity threshold become "mixed"; zero-count cells "unassigned".',
        prerequisites: [],
        params: [
          { name: 'count_columns', label: 'Count columns', type: 'text', default: 'GRCh38_counts, mm10_counts', description: 'Comma-separated .obs count columns (>= 2)' },
          { name: 'labels', label: 'Labels (optional)', type: 'text', default: null, description: 'Comma-separated species labels; default from column names' },
          { name: 'obs_name', label: 'Output column', type: 'text', default: 'species', description: 'New categorical .obs column' },
          { name: 'threshold', label: 'Purity threshold', type: 'number', default: 0.9, description: 'Min fraction to call a species (else "mixed")' },
        ],
      },
    },
  },
```

Note: `layer_select` fetches layers only for a hardcoded function list (~:770). Add `'sum_counts_by_pattern'` to that `needsLayers` list so its layer dropdown populates:

```ts
    const needsLayers = ['smooth', 'gene_pca', 'gene_neighbors', 'build_gene_graph', 'sum_counts_by_pattern'].includes(selectedFunction)
```

- [ ] **Step 3: Wire post-run refresh + embedding auto-select**

In the post-run block (~:1260), add the three keys to the schema/obs refresh list:

```ts
      if (['filter_genes', 'exclude_genes', 'filter_cells', 'pca', 'umap', 'leiden', 'cluster_genes', 'spatial_autocorr', 'highly_variable_genes', 'contourize', 'embedding_from_obs', 'sum_counts_by_pattern', 'assign_species'].includes(selectedFunction)) {
        await refreshSchema()
        refreshObsSummaries()
      }
```

And in the embedding auto-select list (~:1281) add `embedding_from_obs`:

```ts
      if (['umap', 'pca', 'filter_cells', 'embedding_from_obs'].includes(selectedFunction)) {
        setEmbedding(null)
```

(Read the few lines after :1283 to mirror exactly how umap/pca auto-select the new embedding — if it reads `data.embedding_name` and calls a store setter, the obs embedding returns the same `embedding_name` field, so it works unchanged.)

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "$(printf 'frontend: Analyze registry entries for obs embedding + multi-species counts\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all green (previous total + 14 new).

- [ ] **Step 2: Frontend typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 3: Playwright smoke (isolated stack)**

Start backend :8001 + vite :5180. Use a dataset with numeric `.obs` columns and
species-prefixed genes. The user's adE11 dataset has numeric obs (e.g.
`total_counts`, `n_genes_by_counts`) for Feature 1; for Feature 2 use a dataset
with `GRCh38_`/`mm10_` genes, or synthesize one
(`backend/xcell/data/` toy) — if none is available, verify Feature 2 via the
backend route with curl and verify only Feature 1 in the browser.

Browser (Analyze → ScanpyModal):
- *Cell Analysis → Embedding from .obs*: pick two numeric columns, log = both,
  Run → confirm the new embedding is created, appears in the Embedding dropdown,
  and is auto-selected; lasso a region to confirm selection works.
- *Multi-genome → Sum species counts* (if a multi-species dataset is loaded):
  pattern `GRCh38_`, Run → a `GRCh38_counts` column appears in the Cells pane;
  repeat for `mm10_`; *Assign species* with both columns → a `species`
  categorical column appears.

Stop both servers; delete screenshots / `.playwright-mcp`.

- [ ] **Step 4: Append an implemented note to the spec**

Add an "Implemented (2026-06-24)" line near the top of
`docs/superpowers/specs/2026-06-24-obs-embedding-and-multispecies-counts-design.md`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "$(printf 'obs embedding + multi-species counts: verified end-to-end\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-review notes
- **Spec coverage:** custom obs embedding + log (Task 1, 4) ✓; species summation (Task 2, 4) ✓; species assignment (Task 2, 4) ✓; numeric obs picker (Task 3) ✓; registry integration + post-run refresh/auto-select (Task 4) ✓; testing (Tasks 1, 2, 5) ✓.
- **Naming consistency:** `create_obs_embedding`/`embedding_from_obs`/`EmbeddingFromObsRequest`; `sum_counts_by_pattern`/`SumCountsRequest`; `assign_species`/`AssignSpeciesRequest`; param `obsDtype`; category key `multigenome`. Used identically across tasks.
- **Prereqs:** `check_prerequisites` defaults unknown actions to satisfied (`prereqs.get(action, [])`), so the three new functions need no prereq changes.
- **Routes are `/scanpy/<key>`** to match the registry dispatch (`POST /api/scanpy/<selectedFunction>`); the generic handler reports `embedding_name` and supports sync responses (these are sync).
- **Out of scope (untouched):** live embeddings; a multi-select param type (comma-text used); doublet modeling.
