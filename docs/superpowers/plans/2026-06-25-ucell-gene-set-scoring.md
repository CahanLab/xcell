# UCell Gene-Set Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add UCell-style directional (up/down) gene-set scoring — a per-cell rank-based AUC — with a persisted "Score with UCell" action (writes `.obs` columns) and an interactive non-persisted color-by-set method.

**Architecture:** Backend computes per-cell capped ranks once and caches them transiently on the adaptor (keyed by layer+maxRank, invalidated on adata reassignment; never written to adata). The exact UCell `u_stat` AUC + `max(u_p − w_neg·u_n, 0)` combination runs vectorized over the cached sparse rank matrix. Two routes consume it: a batch route that writes `UCell_<name>` `.obs` columns, and an interactive route returning per-cell values. Frontend extends `GeneSet` with an optional `genesDown`, teaches the importer up/down + `-` suffix, and adds the GenePanel action + interactive method.

**Tech Stack:** Python (FastAPI, anndata, scipy.sparse, scipy.stats.rankdata), pytest; React/TypeScript, Zustand, Vite; Playwright for smoke.

**Spec:** `docs/superpowers/specs/2026-06-25-ucell-gene-set-scoring-design.md`

**Test commands:**
- Backend: `cd backend && pixi run -e dev python -m pytest tests/ -q`
- Frontend: `cd frontend && npx tsc --noEmit && npm run build`

---

## Phase 1 — Backend core (rank cache + u_stat + scoring), TDD

### Task 1: Per-cell capped rank cache (`_ucell_ranks`)

**Files:**
- Modify: `backend/xcell/adaptor.py` (add method; init cache attrs in `__init__` near line 272–282; clear cache wherever `self._normalized_adata = None` is reset)
- Test: `backend/tests/test_ucell_scoring.py` (create)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_ucell_scoring.py`:
```python
"""UCell-style directional gene-set scoring."""
import numpy as np
import anndata
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 2 cells x 4 genes, hand-chosen so descending ranks are exact.
    # Cell0: A=10,B=5,C=1,D=0 -> ranks A=1,B=2,C=3,D=4
    # Cell1: A=0,B=1,C=5,D=10 -> ranks D=1,C=2,B=3,A=4
    X = np.array([[10, 5, 1, 0],
                  [0, 1, 5, 10]], dtype=np.float32)
    ad = anndata.AnnData(X=sp.csr_matrix(X))
    ad.var_names = ["A", "B", "C", "D"]
    return ad


def test_ucell_ranks_caps_to_n_genes_and_caches():
    a = DataAdaptor("x.h5ad", adata=_adata())
    ranks = a._ucell_ranks("X", 1500)   # maxRank auto-capped to n_genes=4
    assert sp.issparse(ranks)
    assert ranks.shape == (2, 4)
    # rank>=maxRank(4) dropped to 0; cell0 keeps A=1,B=2,C=3, drops D
    dense = ranks.toarray()
    assert dense[0, 0] == 1 and dense[0, 1] == 2 and dense[0, 2] == 3
    assert dense[0, 3] == 0   # D rank 4 == maxRank -> dropped
    # identical call returns the SAME cached object
    assert a._ucell_ranks("X", 1500) is ranks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py::test_ucell_ranks_caps_to_n_genes_and_caches -q`
Expected: FAIL — `AttributeError: 'DataAdaptor' object has no attribute '_ucell_ranks'`

- [ ] **Step 3: Initialize cache attributes in `__init__`**

In `backend/xcell/adaptor.py`, in `__init__` right after the line `self._multicontour_cache: dict[str, dict[str, Any]] = {}` (around line 282), add:
```python
        # UCell rank matrices, keyed by (resolved_layer, max_rank). Transient:
        # validated against id(self.adata); never written into adata.
        self._ucell_rank_cache: dict[tuple[str, int], Any] = {}
        self._ucell_rank_cache_adata_id: int | None = None
```

- [ ] **Step 4: Implement `_ucell_ranks`**

Add this method to the adaptor (place it just above `_aggregate_gene_set_scores`, ~line 1161):
```python
    def _ucell_ranks(self, layer: str | None, max_rank: int):
        """Per-cell capped descending gene ranks as a sparse CSC matrix.

        Ranks each cell's genes by expression descending (rank 1 = highest),
        average ties, then caps at ``max_rank`` (also capped to n_genes, per
        UCell). Ranks >= max_rank are dropped to 0 (sparse), meaning "treat as
        max_rank" when scoring. Result shape (n_cells, n_genes), cached on the
        adaptor keyed by (resolved layer, max_rank) and invalidated whenever
        ``self.adata`` is reassigned. Source layer 'counts' falls back to X.
        """
        from scipy.stats import rankdata
        import scipy.sparse as sp

        if layer in (None, 'counts'):
            resolved = 'counts' if 'counts' in self.adata.layers else 'X'
        else:
            resolved = layer
        n_cells, n_genes = self.adata.shape
        eff_max_rank = int(min(max_rank, n_genes))
        key = (resolved, eff_max_rank)

        if self._ucell_rank_cache_adata_id != id(self.adata):
            self._ucell_rank_cache = {}
            self._ucell_rank_cache_adata_id = id(self.adata)
        if key in self._ucell_rank_cache:
            return self._ucell_rank_cache[key]

        if resolved == 'X':
            M = self.adata.X
        elif resolved in self.adata.layers:
            M = self.adata.layers[resolved]
        else:
            raise ValueError(f"Layer '{resolved}' not found for UCell ranking")

        chunk = 2000
        blocks = []
        for start in range(0, n_cells, chunk):
            block = M[start:start + chunk]
            dense = block.toarray() if sp.issparse(block) else np.asarray(block)
            dense = dense.astype(np.float64, copy=False)
            r = rankdata(-dense, method='average', axis=1)
            r[r >= eff_max_rank] = 0.0
            blocks.append(sp.csr_matrix(r.astype(np.float32)))
        ranks = sp.vstack(blocks).tocsc() if blocks else sp.csc_matrix((0, n_genes))
        self._ucell_rank_cache[key] = ranks
        return ranks
```

Also clear the cache wherever the normalized cache is invalidated. At EACH line that currently reads `self._normalized_adata = None` (lines ~787, 810, 952, 3690, 3743, 3816, 3856, 3907, 3944), add immediately after it:
```python
        self._ucell_rank_cache = {}
        self._ucell_rank_cache_adata_id = None
```
(The `id(self.adata)` guard already makes stale caches self-correct; this just frees memory eagerly.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py::test_ucell_ranks_caps_to_n_genes_and_caches -q`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add backend/xcell/adaptor.py backend/tests/test_ucell_scoring.py
git commit -m "backend: UCell per-cell rank cache (_ucell_ranks)"
```

---

### Task 2: u_stat scoring + up/down combination (`_ucell_score_one`, `ucell_score_values`)

**Files:**
- Modify: `backend/xcell/adaptor.py` (add two methods after `_ucell_ranks`)
- Test: `backend/tests/test_ucell_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_ucell_scoring.py`:
```python
def test_ucell_up_only_scores():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B"], [], layer="X", max_rank=1500, w_neg=1.0)
    # Cell0: ranks A=1,B=2 -> u=1.0 ; Cell1: ranks A=4,B=3 -> u=0.2
    assert np.allclose(r["values"], [1.0, 0.2])
    assert r["n_up_used"] == 2 and r["n_down_used"] == 0


def test_ucell_up_and_down_subtracts_and_clips():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B"], ["D"], layer="X", max_rank=1500, w_neg=1.0)
    # u_p=[1.0,0.2]; u_n(D)=[0,1]; max(u_p-u_n,0)=[1.0, 0.0]
    assert np.allclose(r["values"], [1.0, 0.0])
    assert r["n_down_used"] == 1


def test_ucell_down_only_is_zero():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values([], ["D"], layer="X", max_rank=1500, w_neg=1.0)
    assert np.allclose(r["values"], [0.0, 0.0])
    assert r["n_up_used"] == 0


def test_ucell_skips_missing_genes():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B", "ZZZ"], [], layer="X", max_rank=1500)
    assert r["n_up_used"] == 2          # ZZZ filtered out
    assert np.allclose(r["values"], [1.0, 0.2])
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -q`
Expected: FAIL — `AttributeError: ... 'ucell_score_values'`

- [ ] **Step 3: Implement the two methods**

Add directly after `_ucell_ranks`:
```python
    def _ucell_score_one(self, up_idx, down_idx, ranks, max_rank, w_neg):
        """Per-cell UCell score for one signature given the cached rank matrix.

        ``ranks`` is the sparse CSC from ``_ucell_ranks`` (0 means rank>=max_rank).
        Returns a float64 array of shape (n_cells,).
        """
        n_cells = ranks.shape[0]

        def u_stat(idx):
            n = len(idx)
            if n == 0:
                return np.zeros(n_cells, dtype=np.float64)
            sub = ranks[:, idx]
            sum_stored = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
            nnz = sub.getnnz(axis=1).astype(np.float64)
            # stored entries hold the true rank (<max_rank); missing -> max_rank
            rank_sum = n * max_rank - max_rank * nnz + sum_stored
            rank_sum_min = n * (n + 1) / 2.0
            denom = n * max_rank - rank_sum_min
            if denom <= 0:
                return np.ones(n_cells, dtype=np.float64)
            return 1.0 - (rank_sum - rank_sum_min) / denom

        u_p = u_stat(up_idx)
        u_n = u_stat(down_idx) if down_idx else 0.0
        return np.maximum(u_p - w_neg * u_n, 0.0)

    def ucell_score_values(
        self, up: list[str], down: list[str] | None = None,
        layer: str = 'counts', max_rank: int = 1500, w_neg: float = 1.0,
    ) -> dict[str, Any]:
        """Compute (non-persisted) per-cell UCell scores for one signature.

        Filters up/down to genes present in .var (missing skipped). A signature
        with no usable up-genes scores 0 everywhere (UCell property). Returns
        values + min/max + counts of genes used.
        """
        down = down or []
        var_index = self.adata.var.index
        up_g = [g for g in up if g in var_index]
        down_g = [g for g in down if g in var_index]
        n_genes = self.adata.shape[1]
        eff_max_rank = int(min(max(max_rank, len(up_g), len(down_g), 1), n_genes))
        ranks = self._ucell_ranks(layer, eff_max_rank)
        up_idx = [var_index.get_loc(g) for g in up_g]
        down_idx = [var_index.get_loc(g) for g in down_g]
        if not up_idx:
            scores = np.zeros(self.adata.shape[0], dtype=np.float64)
        else:
            scores = self._ucell_score_one(up_idx, down_idx, ranks, eff_max_rank, w_neg)
        return {
            "values": [float(v) for v in scores],
            "min": float(scores.min()) if scores.size else 0.0,
            "max": float(scores.max()) if scores.size else 0.0,
            "n_up_used": len(up_idx),
            "n_down_used": len(down_idx),
            "max_rank": eff_max_rank,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**
```bash
git add backend/xcell/adaptor.py backend/tests/test_ucell_scoring.py
git commit -m "backend: UCell u_stat + up/down combination (ucell_score_values)"
```

---

### Task 3: Batch scoring that writes `.obs` columns (`score_gene_sets_ucell`)

**Files:**
- Modify: `backend/xcell/adaptor.py` (add method + a small obs-name helper)
- Test: `backend/tests/test_ucell_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append:
```python
def test_score_gene_sets_writes_obs_columns():
    a = DataAdaptor("x.h5ad", adata=_adata())
    out = a.score_gene_sets_ucell(
        [{"name": "Sig A", "up": ["A", "B"], "down": ["D"]},
         {"name": "Sig C", "up": ["C", "D"]}],
        layer="X", max_rank=1500, w_neg=1.0,
    )
    cols = {r["name"]: r for r in out["results"]}
    assert "UCell_Sig_A" in a.adata.obs.columns
    assert cols["Sig A"]["obs_column"] == "UCell_Sig_A"
    assert np.allclose(a.adata.obs["UCell_Sig_A"].to_numpy(), [1.0, 0.0])
    assert cols["Sig A"]["n_up_used"] == 2 and cols["Sig A"]["n_down_used"] == 1


def test_score_gene_sets_skips_down_only_set():
    a = DataAdaptor("x.h5ad", adata=_adata())
    out = a.score_gene_sets_ucell([{"name": "DownOnly", "up": [], "down": ["D"]}])
    r = out["results"][0]
    assert r.get("skipped")
    assert "UCell_DownOnly" not in a.adata.obs.columns


def test_score_gene_sets_obs_name_collision_suffixes():
    ad = _adata()
    ad.obs["UCell_Dup"] = [9.0, 9.0]
    a = DataAdaptor("x.h5ad", adata=ad)
    out = a.score_gene_sets_ucell([{"name": "Dup", "up": ["A"]}])
    assert out["results"][0]["obs_column"] == "UCell_Dup_1"
    assert "UCell_Dup_1" in a.adata.obs.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -q`
Expected: FAIL — `AttributeError: ... 'score_gene_sets_ucell'`

- [ ] **Step 3: Implement**

Add after `ucell_score_values`:
```python
    @staticmethod
    def _sanitize_obs_name(name: str) -> str:
        import re
        base = re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_') or 'set'
        return f"UCell_{base}"

    def _unique_obs_name(self, base: str) -> str:
        if base not in self.adata.obs.columns:
            return base
        i = 1
        while f"{base}_{i}" in self.adata.obs.columns:
            i += 1
        return f"{base}_{i}"

    def score_gene_sets_ucell(
        self, sets: list[dict[str, Any]], layer: str = 'counts',
        max_rank: int = 1500, w_neg: float = 1.0,
    ) -> dict[str, Any]:
        """Score directional gene sets with UCell and write .obs columns.

        Each set is {name, up:[...], down:[...]}. All sets share one rank matrix
        (one eff_max_rank for comparability). Sets with no usable up-genes are
        skipped (would score 0). Writes obs[UCell_<name>] (collision-safe) and
        returns per-set metadata.
        """
        var_index = self.adata.var.index
        n_genes = self.adata.shape[1]
        prepared = []
        for s in sets:
            name = s.get("name") or "set"
            up = [g for g in (s.get("up") or []) if g in var_index]
            down = [g for g in (s.get("down") or []) if g in var_index]
            prepared.append((name, up, down))
        longest = max([len(u) for _, u, _ in prepared]
                      + [len(d) for _, _, d in prepared] + [1])
        eff_max_rank = int(min(max(max_rank, longest), n_genes))
        ranks = self._ucell_ranks(layer, eff_max_rank)

        results = []
        for name, up, down in prepared:
            if not up:
                results.append({"name": name, "skipped": "no up-genes present in dataset"})
                continue
            up_idx = [var_index.get_loc(g) for g in up]
            down_idx = [var_index.get_loc(g) for g in down]
            scores = self._ucell_score_one(up_idx, down_idx, ranks, eff_max_rank, w_neg)
            col = self._unique_obs_name(self._sanitize_obs_name(name))
            self.adata.obs[col] = scores.astype(np.float64)
            results.append({
                "name": name, "obs_column": col,
                "min": float(scores.min()), "max": float(scores.max()),
                "n_up_used": len(up_idx), "n_down_used": len(down_idx),
            })
        return {"results": results, "max_rank": eff_max_rank, "layer": layer}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**
```bash
git add backend/xcell/adaptor.py backend/tests/test_ucell_scoring.py
git commit -m "backend: score_gene_sets_ucell writes UCell_<name> obs columns"
```

---

## Phase 2 — Backend routes, TDD

### Task 4: Routes `/scanpy/score_genes_ucell` and `/expression/ucell`

**Files:**
- Modify: `backend/xcell/api/routes.py` (add two routes near the other `/scanpy/*` and `/expression/*` routes; reuse the `get_adaptor(dataset)` + `dataset: str | None = Query(None)` pattern)
- Test: `backend/tests/test_ucell_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append:
```python
def test_route_score_genes_ucell(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/score_genes_ucell", json={
        "sets": [{"name": "Sig A", "up": ["A", "B"], "down": ["D"]}],
        "layer": "X", "max_rank": 1500, "w_neg": 1.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["obs_column"] == "UCell_Sig_A"
    assert "UCell_Sig_A" in a.adata.obs.columns


def test_route_expression_ucell(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/expression/ucell", json={
        "up": ["A", "B"], "down": ["D"], "layer": "X",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert np.allclose(body["values"], [1.0, 0.0])
    # interactive endpoint must NOT persist a column
    assert not any(c.startswith("UCell_") for c in a.adata.obs.columns)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -k route -q`
Expected: FAIL — 404 Not Found for both routes

- [ ] **Step 3: Implement the routes**

In `backend/xcell/api/routes.py`, add a request model + route next to `/scanpy/calculate_qc_metrics` (~line 1850):
```python
class ScoreGenesUcellRequest(BaseModel):
    sets: list[dict[str, Any]]
    layer: str = 'counts'
    max_rank: int = 1500
    w_neg: float = 1.0


@router.post("/scanpy/score_genes_ucell")
def scanpy_score_genes_ucell(request: ScoreGenesUcellRequest, dataset: str | None = Query(None)):
    """Score directional gene sets with UCell; writes UCell_<name> .obs columns."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.score_gene_sets_ucell(
            sets=request.sets, layer=request.layer,
            max_rank=request.max_rank, w_neg=request.w_neg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Then add the interactive route next to `/expression/multi` (find it via `grep -n '"/expression/multi"' backend/xcell/api/routes.py`):
```python
class UcellExpressionRequest(BaseModel):
    up: list[str]
    down: list[str] = []
    layer: str = 'counts'
    max_rank: int = 1500
    w_neg: float = 1.0


@router.post("/expression/ucell")
def expression_ucell(request: UcellExpressionRequest, dataset: str | None = Query(None)):
    """Non-persisted per-cell UCell score for one directional set (interactive coloring)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.ucell_score_values(
            up=request.up, down=request.down, layer=request.layer,
            max_rank=request.max_rank, w_neg=request.w_neg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

If `Any` is not already imported in routes.py, confirm with `grep -n "from typing import" backend/xcell/api/routes.py` and add `Any` to that import.

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_ucell_scoring.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Full backend regression**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: PASS (all prior tests + 10 new)

- [ ] **Step 6: Commit**
```bash
git add backend/xcell/api/routes.py backend/tests/test_ucell_scoring.py
git commit -m "backend: routes for UCell batch scoring + interactive coloring"
```

---

## Phase 3 — Frontend data model + importer

### Task 5: Extend `GeneSet` with `genesDown` and thread it through store actions

**Files:**
- Modify: `frontend/src/store.ts` (interface line 77–82; `addGeneSetToCategory` ~1227; `addFolderToCategory` ~1241; their type signatures ~752–753)

- [ ] **Step 1: Extend the interface and action signatures**

In `frontend/src/store.ts`, change the `GeneSet` interface (lines 77–82) to:
```ts
export interface GeneSet {
  id: string
  name: string
  genes: string[]          // UP / positive list
  genesDown?: string[]     // DOWN / negative list (UCell only; optional)
  pinned?: boolean
}
```

Update the action type signatures (lines ~752–753) to accept an optional down-list:
```ts
  addGeneSetToCategory: (categoryType: GeneSetCategoryType, name: string, genes: string[], genesDown?: string[]) => void
  addFolderToCategory: (categoryType: GeneSetCategoryType, folderName: string, geneSets: { name: string; genes: string[]; genesDown?: string[] }[]) => void
```

- [ ] **Step 2: Thread `genesDown` through the action bodies**

`addGeneSetToCategory` body (line 1227–1239) — change the signature and the pushed object:
```ts
    addGeneSetToCategory: (categoryType, name, genes, genesDown) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            geneSets: [
              ...state.geneSetCategories[categoryType].geneSets,
              { id: generateGeneSetId(), name, genes, genesDown },
            ],
          },
        },
      })),
```

`addFolderToCategory` body — change the mapped object (line 1254–1258):
```ts
                geneSets: geneSets.map((gs) => ({
                  id: generateGeneSetId(),
                  name: gs.name,
                  genes: gs.genes,
                  genesDown: gs.genesDown,
                })),
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS (no type errors)

- [ ] **Step 4: Commit**
```bash
git add frontend/src/store.ts
git commit -m "frontend: GeneSet gains optional genesDown (directional sets)"
```

---

### Task 6: Teach the importer up/down JSON + `-`/`+` suffix convention

**Files:**
- Modify: `frontend/src/components/ImportModal.tsx` (`ParsedGeneList` line 91–94; `parseGMT` 101–115; `parseCSV` 117–161; `parseJSON` 163–196; `handleImport` 321–333)

- [ ] **Step 1: Extend `ParsedGeneList` and add a suffix splitter**

Change `ParsedGeneList` (lines 91–94):
```ts
interface ParsedGeneList {
  name: string
  genes: string[]          // up / positive
  genesDown?: string[]     // down / negative
}
```

Add a helper above `parseGMT` (after line 99):
```ts
// Split a flat token list by the UCell suffix convention: trailing '-' -> down,
// trailing '+' (or none) -> up. Returns cleaned symbols.
function splitSuffixDirection(tokens: string[]): { up: string[]; down: string[] } {
  const up: string[] = []
  const down: string[] = []
  for (const raw of tokens) {
    const t = raw.trim()
    if (!t) continue
    if (t.endsWith('-')) down.push(t.slice(0, -1))
    else if (t.endsWith('+')) up.push(t.slice(0, -1))
    else up.push(t)
  }
  return { up, down }
}
```

- [ ] **Step 2: Apply the suffix split in `parseGMT` and `parseCSV`**

In `parseGMT`, replace the body of the loop that builds `genes`/pushes (lines 107–112) with:
```ts
    const name = parts[0].trim()
    const { up, down } = splitSuffixDirection(parts.slice(2))
    if (name && up.length + down.length > 0) {
      lists.push({ name, genes: up, genesDown: down.length ? down : undefined })
    }
```

In `parseCSV`, for the single-column branch (lines 132–140) replace `const genes = ...; return genes.length > 0 ? [{ name, genes }] : []` with:
```ts
    const tokens = lines.slice(startIdx)
      .map((l) => l.trim().replace(/^["']|["']$/g, ''))
      .filter(Boolean)
    const name = hasHeader ? headerParts[0] : filename.replace(/\.\w+$/, '')
    const { up, down } = splitSuffixDirection(tokens)
    return up.length + down.length > 0
      ? [{ name, genes: up, genesDown: down.length ? down : undefined }]
      : []
```
And in the multi-column branch, after collecting `genes` for a column (line 156–158) replace the push with:
```ts
    if (genes.length > 0) {
      const { up, down } = splitSuffixDirection(genes)
      lists.push({ name: names[col], genes: up, genesDown: down.length ? down : undefined })
    }
```

- [ ] **Step 3: Extend `parseJSON` for up/down objects + suffix**

Replace `parseJSON` (lines 163–196) with:
```ts
function parseJSON(text: string): ParsedGeneList[] {
  const data = JSON.parse(text)
  // Accepted shapes:
  //   1. legacy array:           [{name, genes:[...]}]
  //   2. geneset-builder export: {sets:[{name, genes:[...]}]}
  //   3. .gsb.json:              genes may be [{symbol, from}]
  //   4. directional:            {name, up:[...], down:[...]} (synonyms below)
  //   Plus the '-'/'+' suffix convention inside any flat list.
  let rawSets: unknown[] | null = null
  if (Array.isArray(data)) rawSets = data
  else if (data && typeof data === 'object' && Array.isArray((data as { sets?: unknown }).sets)) {
    rawSets = (data as { sets: unknown[] }).sets
  }
  if (!rawSets) return []

  const toSymbols = (val: unknown): string[] => {
    if (!Array.isArray(val)) return []
    return val
      .map((g) => {
        if (typeof g === 'string') return g
        if (g && typeof g === 'object' && typeof (g as { symbol?: unknown }).symbol === 'string') {
          return (g as { symbol: string }).symbol
        }
        return ''
      })
      .filter((s) => s.length > 0)
  }

  const lists: ParsedGeneList[] = []
  for (const item of rawSets) {
    if (!item || typeof item !== 'object') continue
    const obj = item as Record<string, unknown>
    if (typeof obj.name !== 'string') continue
    const upRaw = toSymbols(obj.up ?? obj.positive ?? obj.genesUp ?? obj.genes)
    const downRaw = toSymbols(obj.down ?? obj.negative ?? obj.genesDown)
    // honor suffix convention inside the up list (e.g. ["CD8A","CCR7-"])
    const splitUp = splitSuffixDirection(upRaw)
    const up = splitUp.up
    const down = [...splitUp.down, ...downRaw.map((g) => g.replace(/[-+]$/, ''))]
    if (up.length + down.length > 0) {
      lists.push({ name: obj.name, genes: up, genesDown: down.length ? down : undefined })
    }
  }
  return lists
}
```

- [ ] **Step 4: Carry `genesDown` through `handleImport`**

In `handleImport` (lines 321–333), pass the down-list:
```ts
      if (pf.geneLists.length === 1) {
        const gl = pf.geneLists[0]
        addGeneSetToCategory('manual', gl.name, gl.genes, gl.genesDown)
      } else {
        addFolderToCategory('manual', folderName, pf.geneLists)
      }
```
(`addFolderToCategory` already receives the full `ParsedGeneList[]`, so `genesDown` rides along after Task 5.)

- [ ] **Step 5: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add frontend/src/components/ImportModal.tsx
git commit -m "frontend: import directional gene sets (up/down JSON + -/+ suffix)"
```

---

## Phase 4 — Frontend GenePanel "Score with UCell" action

### Task 7: UCell scoring modal component

**Files:**
- Create: `frontend/src/components/UcellScoreModal.tsx`
- Reference (read for layer list + refreshSchema): `frontend/src/components/ScanpyModal.tsx` (`layer_select` uses `GET /api/scanpy/layers`), `frontend/src/hooks/useData.ts` (`refreshSchema`)

- [ ] **Step 1: Create the modal**

Create `frontend/src/components/UcellScoreModal.tsx`:
```tsx
import { useEffect, useState } from 'react'
import { appendDataset } from '../hooks/useData'

interface UcellTarget {
  sets: { name: string; up: string[]; down: string[] }[]
}

interface Props {
  target: UcellTarget | null
  onClose: () => void
  onScored: (msg: string) => void   // caller refreshes schema + toasts
}

export function UcellScoreModal({ target, onClose, onScored }: Props) {
  const [layers, setLayers] = useState<string[]>(['X'])
  const [layer, setLayer] = useState('counts')
  const [maxRank, setMaxRank] = useState(1500)
  const [wNeg, setWNeg] = useState(1.0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!target) return
    fetch(appendDataset('/api/scanpy/layers'))
      .then((r) => r.json())
      .then((d) => {
        const names: string[] = (d.layers ?? []).map((l: { name: string }) => l.name)
        setLayers(names.length ? names : ['X'])
        setLayer(names.includes('counts') ? 'counts' : 'X')
      })
      .catch(() => setLayers(['X']))
  }, [target])

  if (!target) return null

  const run = async () => {
    setBusy(true); setError(null)
    try {
      const resp = await fetch(appendDataset('/api/scanpy/score_genes_ucell'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sets: target.sets, layer, max_rank: maxRank, w_neg: wNeg }),
      })
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Scoring failed')
      const data = await resp.json()
      const written = data.results.filter((r: { obs_column?: string }) => r.obs_column)
      const skipped = data.results.filter((r: { skipped?: string }) => r.skipped)
      const cols = written.map((r: { obs_column: string }) => r.obs_column).join(', ')
      let msg = written.length ? `Wrote ${written.length} UCell column(s): ${cols}` : 'No columns written'
      if (skipped.length) msg += ` — skipped ${skipped.length} (no up-genes)`
      onScored(msg)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onClose}>
      <div style={{ background: '#1e1e1e', color: '#eee', padding: 20, borderRadius: 8,
        minWidth: 360, fontSize: 13 }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Score with UCell</h3>
        <p style={{ color: '#aaa' }}>
          {target.sets.length} set{target.sets.length > 1 ? 's' : ''} →
          writes <code>UCell_&lt;name&gt;</code> obs column(s).
        </p>
        <label style={{ display: 'block', margin: '8px 0' }}>Source layer
          <select value={layer} onChange={(e) => setLayer(e.target.value)}
            style={{ width: '100%', marginTop: 4 }}>
            {layers.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
        <label style={{ display: 'block', margin: '8px 0' }}>maxRank
          <input type="number" value={maxRank} min={1}
            onChange={(e) => setMaxRank(Math.max(1, parseInt(e.target.value) || 1500))}
            style={{ width: '100%', marginTop: 4 }} />
        </label>
        <button onClick={() => setShowAdvanced((v) => !v)}
          style={{ background: 'none', color: '#4ecdc4', border: 'none', cursor: 'pointer', padding: 0 }}>
          {showAdvanced ? '▼' : '▶'} Advanced
        </button>
        {showAdvanced && (
          <label style={{ display: 'block', margin: '8px 0' }}>w_neg (down-set weight)
            <input type="number" step="0.1" value={wNeg}
              onChange={(e) => setWNeg(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', marginTop: 4 }} />
          </label>
        )}
        {error && <div style={{ color: '#ff6b6b', margin: '8px 0' }}>{error}</div>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 12 }}>
          <button onClick={onClose} disabled={busy}>Cancel</button>
          <button onClick={run} disabled={busy}
            style={{ background: '#4ecdc4', color: '#000', border: 'none', padding: '6px 14px',
              borderRadius: 4, cursor: 'pointer' }}>
            {busy ? 'Scoring…' : 'Score'}
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**
```bash
git add frontend/src/components/UcellScoreModal.tsx
git commit -m "frontend: UcellScoreModal (layer/maxRank/w_neg form -> batch score)"
```

---

### Task 8: Wire the modal into GenePanel (menu item + refreshSchema)

**Files:**
- Modify: `frontend/src/components/GenePanel.tsx` (OverflowMenu items ~1010–1040; add modal state + render; import the modal and `refreshSchema`)

- [ ] **Step 1: Add menu item + state**

In `GenePanel.tsx`:
1. Add the import near the other component imports:
```ts
import { UcellScoreModal } from './UcellScoreModal'
```
2. The gene-set row component needs to open the modal. Add a prop `onScoreUcell` to the gene-set row component (the one rendering the OverflowMenu at ~1010) alongside the existing `onColorBySet` prop, and add a new menu item to the `items` array (after the "Select cells…" item, line 1038):
```ts
              {
                label: 'Score with UCell…',
                onClick: () => onScoreUcell(geneSet),
                disabled: geneSet.genes.length === 0,
                tooltip: geneSet.genes.length === 0 ? 'Need at least one up-gene' : undefined,
              },
```

- [ ] **Step 2: Hold modal state in the GenePanel container and render it**

In the top-level `GenePanel` component, add state and a handler, and pass `onScoreUcell` down to each gene-set row (mirror how `onColorBySet` is threaded). Use the existing `refreshSchema` from `useDataActions()` and the store's toast/notice mechanism (search GenePanel for an existing success/toast call to reuse; if none, use `console.info` + `window.alert` is NOT acceptable — reuse the existing notification used after other scanpy ops, e.g. grep `setStatusMessage` / `toast` in the file). Add:
```ts
  const [ucellTarget, setUcellTarget] = useState<{ sets: { name: string; up: string[]; down: string[] }[] } | null>(null)
  const handleScoreUcell = (gs: GeneSet) => {
    setUcellTarget({ sets: [{ name: gs.name, up: gs.genes, down: gs.genesDown ?? [] }] })
  }
```
Render near the other modals returned by GenePanel:
```tsx
      <UcellScoreModal
        target={ucellTarget}
        onClose={() => setUcellTarget(null)}
        onScored={(msg) => { refreshSchema(); /* reuse existing toast: */ showNotice(msg) }}
      />
```
Replace `showNotice(msg)` with whatever the file already uses to surface post-op messages (confirm by grep). Ensure `refreshSchema` is destructured from `useDataActions()` in this component (grep to see if already present; add if not).

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add frontend/src/components/GenePanel.tsx
git commit -m "frontend: GenePanel 'Score with UCell…' action wired to modal + refreshSchema"
```

---

## Phase 5 — Frontend interactive UCell color-by-set method

### Task 9: `geneSetScoringMethod` preference + DisplaySettings toggle

**Files:**
- Modify: `frontend/src/store.ts` (`DisplayPreferences` ~415–424; defaults ~485–487; persistence merge ~524–528)
- Modify: `frontend/src/components/DisplaySettings.tsx` (near the geneSet norm/aggregation controls ~465–518)

- [ ] **Step 1: Add the preference field + default**

In `DisplayPreferences` (after `geneSetAggregation`, line 424) add:
```ts
  geneSetScoringMethod: 'mean' | 'ucell'   // 'mean' = current per-gene-norm+aggregate path
```
In the defaults object (after `geneSetAggregation: 'mean'`, line 487) add:
```ts
    geneSetScoringMethod: 'mean',
```
In the persistence merge block (near lines 524–528, where `gsn`/`gsc`/`gsa` are read) add an analogous read so the choice survives reloads:
```ts
  const gsm = (raw as Record<string, unknown>).geneSetScoringMethod
  if (gsm === 'mean' || gsm === 'ucell') out.geneSetScoringMethod = gsm
```
(Match the exact variable/source names used by the surrounding code; confirm by reading lines 515–530.)

- [ ] **Step 2: Add the toggle to DisplaySettings**

In `DisplaySettings.tsx`, just above the per-gene-norm control (~line 460), add a method selector:
```tsx
                <label style={labelStyle}>Gene-set scoring
                  <select
                    value={displayPreferences.geneSetScoringMethod}
                    onChange={(e) => setDisplayPreferences({ geneSetScoringMethod: e.target.value as 'mean' | 'ucell' })}
                  >
                    <option value="mean">Mean / per-gene norm</option>
                    <option value="ucell">UCell (rank AUC, directional)</option>
                  </select>
                </label>
```
(Reuse the existing `labelStyle`/wrapper used by the adjacent controls; the per-gene-norm + aggregation controls below remain — they apply only when method is 'mean'. Optionally wrap them in `{displayPreferences.geneSetScoringMethod === 'mean' && (...)}` so they hide under UCell.)

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add frontend/src/store.ts frontend/src/components/DisplaySettings.tsx
git commit -m "frontend: geneSetScoringMethod preference (mean | ucell) + toggle"
```

---

### Task 10: Route color-by-set through `/expression/ucell` when method is 'ucell'

**Files:**
- Modify: `frontend/src/hooks/useData.ts` (the `colorByGenes` callback — find via `grep -n 'colorByGenes' frontend/src/hooks/useData.ts`; mirrors the `/expression/multi` block around lines 195/365)
- Modify: `frontend/src/components/GenePanel.tsx` (`onColorBySet` call site line 998 — pass the down-list)

- [ ] **Step 1: Extend `onColorBySet` to carry the down-list**

In `GenePanel.tsx` line 998, change:
```ts
            onClick={(e) => { e.stopPropagation(); onColorBySet(geneSet.genes, geneSet.name) }}
```
to also pass `genesDown`:
```ts
            onClick={(e) => { e.stopPropagation(); onColorBySet(geneSet.genes, geneSet.name, geneSet.genesDown) }}
```
Update the `onColorBySet` prop type (search GenePanel for its declaration) to `(genes: string[], name: string, genesDown?: string[]) => void`, and update the container's implementation that calls `colorByGenes` to forward `genesDown`.

- [ ] **Step 2: Branch `colorByGenes` on the scoring method**

In `useData.ts` `colorByGenes`, accept an optional `genesDown` param and, when `displayPreferences.geneSetScoringMethod === 'ucell'`, POST `/expression/ucell` instead of `/expression/multi`. Read the current `colorByGenes` body first; the UCell branch mirrors the existing multi-gene state updates:
```ts
  const colorByGenes = useCallback(
    async (genes: string[], _opts?: unknown, geneSetName?: string, genesDown?: string[]) => {
      const prefs = useStore.getState().displayPreferences
      if (prefs.geneSetScoringMethod === 'ucell') {
        const layerArg = /* same displayLayer resolution used by the multi path */ undefined
        const data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/ucell`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ up: genes, down: genesDown ?? [], layer: layerArg ?? 'counts' }),
        })
        setExpressionData(data)
        setSelectedGenes(genes)
        setSelectedGeneSetName(geneSetName ?? null)
        setColorMode('expression')
        setSelectedColorColumn(null)
        return
      }
      /* ...existing /expression/multi path unchanged... */
    },
    [/* existing deps */],
  )
```
Keep the existing mean path exactly as-is in the `else`. Match the real param list/signature of the current `colorByGenes` (the second positional arg may differ — preserve it; only add the trailing `genesDown`).

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add frontend/src/hooks/useData.ts frontend/src/components/GenePanel.tsx
git commit -m "frontend: color-by-set uses /expression/ucell when method=ucell (directional)"
```

---

## Phase 6 — Frontend directional display

### Task 11: Show up/down counts and style down-genes in the gene-set row

**Files:**
- Modify: `frontend/src/components/GenePanel.tsx` (the count label ~976–985; the expanded gene-chip list ~1044–1067)

- [ ] **Step 1: Show up/down counts**

In the gene-set header where the gene count renders (the block computing `const total = geneSet.genes.length` ~line 976), append a down-count badge when present:
```tsx
                {geneSet.genesDown && geneSet.genesDown.length > 0 && (
                  <span style={{ color: '#ff9e64', marginLeft: 4 }}
                    title="down / negative genes">
                    {geneSet.genes.length}↑ {geneSet.genesDown.length}↓
                  </span>
                )}
```
(If a down-list exists, prefer showing `N↑ M↓`; otherwise keep the existing plain count.)

- [ ] **Step 2: Render down-genes in the expanded list, styled distinctly**

In the expanded chips block (`expanded && geneSet.genes.length > 0`, ~line 1044), after the `.map` over `geneSet.genes`, also map over `geneSet.genesDown` (when present), rendering each with a distinct style (e.g. a leading `↓` and the `#ff9e64` color) and the same drag behavior. Keep it minimal — a second `.map` block guarded by `geneSet.genesDown?.length`.

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add frontend/src/components/GenePanel.tsx
git commit -m "frontend: show up/down counts + style down-genes in gene-set row"
```

---

## Phase 7 — Verification

### Task 12: Full backend regression + frontend build

- [ ] **Step 1: Backend**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: PASS (all, including the 10 UCell tests)

- [ ] **Step 2: Frontend**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: PASS

---

### Task 13: Playwright smoke on the isolated stack

**Pattern (from prior sessions):** start an isolated backend on :8001 and a vite on :5180 — do NOT touch the user's :8000/:5173. Use the rich real dataset:
`/Users/pcahan/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/scRNAseq/Onggo/soupX_PC/adE11_soupx_corrected_C30_111925.h5ad` (mouse symbols `mt-`, `Rpl`, `Rps`).

- [ ] **Step 1: Launch isolated backend + frontend**

```bash
cd backend && XCELL_PORT=8001 pixi run -e dev python -m xcell.main &   # confirm the actual launch cmd in main.py
cd frontend && XCELL_BACKEND=http://127.0.0.1:8001 npx vite --port 5180 --strictPort &
```
(Confirm the backend launch invocation by reading `backend/xcell/main.py`; match how prior smoke runs started it.)

- [ ] **Step 2: Drive via Playwright (DOM eval)**

Using the Playwright MCP tools:
1. Navigate to `http://127.0.0.1:5180`, load the adE11 dataset.
2. Build a small directional set in the manual category (or import one) — e.g. up `["Epcam","Krt8","Krt18"]`, down `["Ptprc"]` (adjust to symbols present).
3. Open the gene-set OverflowMenu → "Score with UCell…", accept defaults (layer=counts, maxRank=1500), Score.
4. Assert a toast/notice naming a `UCell_*` column appears, and that the column is selectable for coloring.
5. Color cells by the `UCell_*` obs column; screenshot.
6. Flip Display settings → Gene-set scoring → UCell, click the 🎨 color-by-set on the directional set; confirm coloring renders without error; screenshot.

- [ ] **Step 3: Verify the two screenshots show a continuous score gradient (not an error/blank).**

- [ ] **Step 4: Clean up**

```bash
# kill the :8001 backend and :5180 vite; remove any *.png written during smoke
```

- [ ] **Step 5: Final commit (smoke is verification-only; no code change expected). If smoke surfaced a fix, commit it with a descriptive message.**

---

## Self-Review notes (coverage vs spec)

- Algorithm (u_stat AUC + max(u_p − w_neg·u_n, 0), maxRank cap to n_genes, descending avg-tie ranks): Tasks 1–2, tested with hand-computed values.
- Directional data model (`genesDown`): Task 5. Loader (up/down JSON + `-`/`+` suffix across JSON/GMT/CSV): Task 6.
- Transient rank cache keyed by (layer, maxRank), invalidated on adata reassignment, never exported: Task 1.
- Persisted batch → `UCell_<name>` obs columns + collision suffix + down-only skip: Task 3; route Task 4.
- Interactive non-persisted endpoint: Task 4 (route), Tasks 9–10 (UI method + routing).
- GenePanel action + modal (maxRank, layer=counts default, w_neg advanced) + refreshSchema: Tasks 7–8.
- Directional display: Task 11.
- Missing-gene skip, signature>maxRank auto-raise, down-only→skip/zero: covered in Tasks 2–3 + tests.
- Verification (pytest, tsc/build, Playwright smoke): Tasks 12–13.
