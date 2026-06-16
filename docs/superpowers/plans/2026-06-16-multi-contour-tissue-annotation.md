# Multi-contour Tissue Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Analyze → Spatial → Multi-contour workflow that contourizes several gene sets, binarizes each to "high/not-high", and fuses them into one `tissue` `.obs` annotation, resolving multi-high spots via spatial-then-`X_pca`-profile kNN.

**Architecture:** New pure module `backend/xcell/multicontour.py` (scoring reuse + binarize + assign + conflict resolution), thin two-phase adaptor methods (`prepare_multicontour` cancellable / `finalize_multicontour` synchronous) with a server-side score cache, two API routes, and a dedicated `MultiContourModal` frontend. The contourize scoring core is extracted into a shared function so single- and multi-contour share one code path.

**Tech Stack:** Python (numpy, scipy, anndata, pandas, FastAPI, pytest), React/TypeScript (Zustand store, fetch + pollTask).

Spec: `docs/superpowers/specs/2026-06-16-multi-contour-tissue-annotation-design.md`

---

## File Structure

- Create `backend/xcell/multicontour.py` — pure functions: `score_module`, `suggest_grid_res`, `suggest_smooth_sigma`, `auto_cutoff`, `binarize_modules`, `assign_tissue`, `resolve_conflicts_knn`.
- Create `backend/tests/test_multicontour.py` — unit tests.
- Modify `backend/xcell/adaptor.py` — extract `_contour_score_field()` shared core from `run_contourize`/`prepare_contourize`; add `prepare_multicontour()` + `finalize_multicontour()` + `self._multicontour_cache`.
- Modify `backend/xcell/api/routes.py` — `POST /scanpy/multicontour/prepare` (202, cancellable) and `POST /scanpy/multicontour/finalize`; Pydantic models; register `multicontour` in the prerequisites map.
- Create `frontend/src/components/MultiContourModal.tsx` — the modal.
- Modify `frontend/src/store.ts` — modal open/close state + gene-set access.
- Modify the Analyze/Spatial entry point (wherever `ScanpyModal` "Contourize" is surfaced) to add a "Multi-contour" launcher.
- Modify `frontend/src/components/ScanpyModal.tsx` — richer contourize param tooltips.
- Modify `README.md`, `CHANGELOG.md`.

---

## Task 1: Extract shared contour scoring core (refactor, behavior-preserving)

**Files:**
- Modify: `backend/xcell/adaptor.py` (the `compute_fn` in `prepare_contourize` ~5846-5903 and the body of `run_contourize` ~5963-6047)
- Test: `backend/tests/test_contour_score.py`

The core computation (per-gene log/clip/normalize → average → grid interpolate → smooth → sample at spots, returning the continuous per-spot score and `vmax`) is duplicated. Extract it into one module-level function so multi-contour reuses it.

- [ ] **Step 1: Write the characterization test**

```python
# backend/tests/test_contour_score.py
import numpy as np
import anndata
from scipy.sparse import csr_matrix
from xcell.adaptor import _contour_score_field


def _toy_adata():
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = csr_matrix(rng.integers(0, 5, size=(n, 3)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = ["g0", "g1", "g2"]
    ad.obsm["spatial"] = coords
    return ad


def test_score_field_shape_and_range():
    ad = _toy_adata()
    expr = {g: np.asarray(ad[:, g].X.todense()).ravel() for g in ["g0", "g1"]}
    score, vmax = _contour_score_field(
        coords=ad.obsm["spatial"], gene_expr=expr,
        log_transform=True, clip_percentiles=(1, 99),
        grid_res=50, smooth_sigma=2.0,
    )
    assert score.shape == (ad.n_obs,)
    assert np.isfinite(score).all()
    assert vmax >= score.max() - 1e-9
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_contour_score.py -v`
Expected: FAIL — `ImportError: cannot import name '_contour_score_field'`.

- [ ] **Step 3: Add the shared function**

Add at module scope in `adaptor.py` (near the contourize methods):

```python
def _contour_score_field(coords, gene_expr, log_transform, clip_percentiles,
                         grid_res, smooth_sigma):
    """Continuous per-spot contour score (the value before banding).

    coords: (n,2) array. gene_expr: dict gene -> (n,) expression vector.
    Returns (score: (n,) float in [0, vmax], vmax: float).
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    x, y = coords[:, 0], coords[:, 1]
    normed = []
    for vals in gene_expr.values():
        v = np.asarray(vals, dtype=float).copy()
        if log_transform:
            v = np.log1p(v)
        lo, hi = np.percentile(v, clip_percentiles)
        clipped = np.clip(v, lo, hi)
        normed.append((clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped))
    summary = np.mean(np.column_stack(normed), axis=1)

    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), summary, (Xi, Yi), method='cubic', fill_value=0.0)
    Zi_s = gaussian_filter(Zi, sigma=smooth_sigma, mode='nearest')
    vmax = float(np.nanmax(Zi_s))

    pts = np.vstack((Xi.ravel(), Yi.ravel())).T
    cell_vals = griddata(pts, Zi_s.ravel(), (x, y), method='nearest')
    return np.asarray(cell_vals, dtype=float), vmax
```

- [ ] **Step 4: Refactor `run_contourize` and `prepare_contourize` to call it**

In both, replace the inline steps 1–6 with:
```python
gene_expr = {g: _get_array(self.adata[:, g].X) for g in genes}  # prepare_* uses its snapshot dict
cell_vals, vmax = _contour_score_field(
    coords, gene_expr, log_transform, clip_percentiles, grid_res, smooth_sigma)
N = contour_levels
thresholds = np.linspace(0, vmax, N + 2)[1:-1]
```
Keep steps 7–8 (threshold assignment + categorical) unchanged. In `prepare_contourize`, build `gene_expr` from the existing `gene_expression_snap` dict and `coords_snap`.

- [ ] **Step 5: Run tests, verify pass**

Run: `cd backend && pixi run -e dev pytest tests/test_contour_score.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_contour_score.py
git commit -m "Refactor: extract shared _contour_score_field for contourize reuse"
```

---

## Task 2: `multicontour.py` — scoring + binarize + auto cutoff

**Files:**
- Create: `backend/xcell/multicontour.py`
- Test: `backend/tests/test_multicontour.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_multicontour.py
import numpy as np
import anndata
from scipy.sparse import csr_matrix
import xcell.multicontour as mc


def make_adata():
    # 5x5 grid of spots; 2 gene sets that are high in disjoint corners.
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    # genes 0,1 = "setA" high where i<=1 ; genes 2,3 = "setB" high where i>=3
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1:
            X[k, 0] = X[k, 1] = 10.0
        if i >= 3:
            X[k, 2] = X[k, 3] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()  # simple 2D profile
    return ad


def test_score_module_returns_bands_and_cutoff():
    ad = make_adata()
    res = mc.score_module(ad, ["a0", "a1"], contour_levels=3,
                          log_transform=True, clip_percentiles=(1, 99),
                          grid_res=40, smooth_sigma=2.0)
    assert res["score"].shape == (ad.n_obs,)
    assert res["bands"].shape == (ad.n_obs,)
    # auto cutoff = top band (max threshold)
    assert res["auto_cutoff"] == res["thresholds"][-1]
    assert len(res["histogram"]) == len(res["band_values"])


def test_binarize_uses_cutoff():
    ad = make_adata()
    res = mc.score_module(ad, ["a0", "a1"], contour_levels=3, log_transform=True,
                          clip_percentiles=(1, 99), grid_res=40, smooth_sigma=2.0)
    high = mc.binarize(res["bands"], res["auto_cutoff"])
    assert high.dtype == bool
    assert high.sum() > 0          # some spots high
    assert high.sum() < ad.n_obs   # not all
```

- [ ] **Step 2: Run, verify fail**

Run: `cd backend && pixi run -e dev pytest tests/test_multicontour.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xcell.multicontour'`.

- [ ] **Step 3: Implement `score_module`, `binarize`, `auto_cutoff`**

```python
# backend/xcell/multicontour.py
"""Multi-contour tissue annotation: score gene-set modules, binarize, and
fuse into one tissue label, resolving multi-high spots via spatial+PCA kNN."""
from __future__ import annotations

import numpy as np

from xcell.adaptor import _contour_score_field


def _expr_dict(adata, genes):
    out = {}
    for g in genes:
        x = adata[:, g].X
        out[g] = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    return out


def score_module(adata, genes, contour_levels=3, log_transform=True,
                 clip_percentiles=(1, 99), grid_res=200, smooth_sigma=2.0):
    """Continuous score, threshold bands, auto cutoff, and band histogram."""
    coords = np.asarray(adata.obsm[_spatial_key(adata)])
    score, vmax = _contour_score_field(
        coords, _expr_dict(adata, genes), log_transform, clip_percentiles,
        grid_res, smooth_sigma)
    thresholds = np.linspace(0, vmax, contour_levels + 2)[1:-1]
    band_values = np.unique(np.concatenate(([0.0], thresholds)))
    bands = np.zeros(adata.n_obs, dtype=float)
    for t in sorted(thresholds):
        bands[score >= t] = t
    histogram = [int(np.sum(bands == bv)) for bv in band_values]
    auto = float(thresholds[-1]) if len(thresholds) else 0.0
    return {
        "score": score, "bands": bands, "thresholds": thresholds,
        "band_values": band_values, "histogram": histogram, "auto_cutoff": auto,
    }


def binarize(bands, cutoff):
    """Boolean 'high' mask: band >= cutoff."""
    return np.asarray(bands) >= cutoff


def auto_cutoff(thresholds):
    """Default high cutoff = top band."""
    return float(thresholds[-1]) if len(thresholds) else 0.0


def _spatial_key(adata):
    for k in ("spatial", "X_spatial"):
        if k in adata.obsm:
            return k
    raise ValueError("No spatial coordinates found")
```

- [ ] **Step 4: Run, verify pass**

Run: `cd backend && pixi run -e dev pytest tests/test_multicontour.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/multicontour.py backend/tests/test_multicontour.py
git commit -m "multicontour: score_module + binarize + auto_cutoff"
```

---

## Task 3: `assign_tissue` + `resolve_conflicts_knn`

**Files:**
- Modify: `backend/xcell/multicontour.py`
- Test: `backend/tests/test_multicontour.py`

- [ ] **Step 1: Write failing tests**

```python
def test_assign_single_high_and_unassigned():
    # 3 spots, 2 modules. spot0 high in A only, spot1 high in B only, spot2 none.
    highs = {"A": np.array([True, False, False]),
             "B": np.array([False, True, False])}
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=5, spatial_conn=None, pca=None, coords=None)
    assert list(labels) == ["A", "B", "unassigned"]
    assert list(status) == ["single", "single", "unassigned"]


def test_conflict_resolved_by_neighbor_majority():
    # 4 spots in a line: 0=A,1=A,2=conflict(A&B),3=B. PCA puts 2 nearest to 1 (A).
    highs = {"A": np.array([True, True, True, False]),
             "B": np.array([False, False, True, True])}
    coords = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    pca = coords.copy()
    # spatial graph: each adjacent to neighbors +-1 (dense bool adj)
    conn = np.array([
        [0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=float)
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=2, spatial_conn=conn, pca=pca, coords=coords)
    assert labels[2] == "A"       # neighbor 1 is unambiguous A and closest in PCA
    assert status[2] == "resolved"


def test_conflict_unresolved_when_no_unambiguous_neighbors():
    highs = {"A": np.array([True, True]), "B": np.array([True, True])}  # both conflict
    coords = np.array([[0.,0.],[1.,0.]])
    conn = np.array([[0,1],[1,0]], dtype=float)
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=2, spatial_conn=conn, pca=coords.copy(), coords=coords)
    assert list(labels) == ["unassigned", "unassigned"]
    assert list(status) == ["unassigned", "unassigned"]
```

- [ ] **Step 2: Run, verify fail**

Run: `cd backend && pixi run -e dev pytest tests/test_multicontour.py -v`
Expected: FAIL — `AttributeError: module 'xcell.multicontour' has no attribute 'assign_tissue'`.

- [ ] **Step 3: Implement assignment + resolution**

```python
def assign_tissue(highs, adata, profile_k, spatial_conn, pca, coords):
    """Return (labels: object array, status: object array).

    highs: dict module_name -> bool mask (n,). Names define label space.
    Resolution for multi-high spots: among the spot's spatial-graph neighbors,
    keep unambiguous (exactly-one-high) ones, rank by Euclidean distance in
    ``pca``, take nearest ``profile_k``, assign majority label. Tie/empty ->
    'unassigned'. Single pass (uses original unambiguous labels only).
    """
    names = list(highs.keys())
    H = np.column_stack([highs[m] for m in names])  # (n, M) bool
    n = H.shape[0]
    counts = H.sum(axis=1)

    labels = np.array(["unassigned"] * n, dtype=object)
    status = np.array(["unassigned"] * n, dtype=object)

    single = counts == 1
    single_idx_of = np.argmax(H, axis=1)
    labels[single] = np.array(names, dtype=object)[single_idx_of[single]]
    status[single] = "single"

    conflict_ids = np.where(counts >= 2)[0]
    if conflict_ids.size:
        neigh = _neighbor_lists(spatial_conn, coords, n)
        for s in conflict_ids:
            cands = [j for j in neigh[s] if single[j]]
            if not cands:
                continue
            d = np.linalg.norm(pca[cands] - pca[s], axis=1)
            nearest = [cands[i] for i in np.argsort(d)[:profile_k]]
            votes = {}
            for j in nearest:
                votes[labels[j]] = votes.get(labels[j], 0) + 1
            top = max(votes.values())
            winners = [lab for lab, c in votes.items() if c == top]
            if len(winners) == 1:
                labels[s] = winners[0]
                status[s] = "resolved"
            # tie -> leave unassigned
    return labels, status


def _neighbor_lists(spatial_conn, coords, n, k_fallback=15):
    """Adjacency as a list-of-lists. Uses spatial_conn if given, else builds a
    coordinate kNN graph."""
    if spatial_conn is not None:
        conn = spatial_conn
        if hasattr(conn, "tocsr"):
            conn = conn.tocsr()
            return [list(conn.indices[conn.indptr[i]:conn.indptr[i + 1]]) for i in range(n)]
        conn = np.asarray(conn)
        return [list(np.where(conn[i] != 0)[0]) for i in range(n)]
    # fallback: coordinate kNN
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    k = min(k_fallback + 1, n)
    _, idx = tree.query(coords, k=k)
    return [[j for j in row if j != i] for i, row in enumerate(np.atleast_2d(idx))]
```

- [ ] **Step 4: Run, verify pass**

Run: `cd backend && pixi run -e dev pytest tests/test_multicontour.py -v`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/multicontour.py backend/tests/test_multicontour.py
git commit -m "multicontour: assign_tissue + spatial/PCA conflict resolution"
```

---

## Task 4: Data-aware default helpers

**Files:**
- Modify: `backend/xcell/multicontour.py`
- Test: `backend/tests/test_multicontour.py`

- [ ] **Step 1: Write failing tests**

```python
def test_suggest_grid_res_scales_with_spots():
    small = mc.suggest_grid_res(n_spots=500)
    big = mc.suggest_grid_res(n_spots=344021)
    assert 50 <= small <= 600 and 50 <= big <= 600
    assert big >= small


def test_suggest_smooth_sigma_positive():
    coords = np.array([[float(i), float(j)] for i in range(10) for j in range(10)])
    sigma = mc.suggest_smooth_sigma(coords, grid_res=100)
    assert sigma > 0
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_multicontour.py -v` → FAIL (no `suggest_grid_res`).

- [ ] **Step 3: Implement**

```python
def suggest_grid_res(n_spots):
    """Grid resolution ~ sqrt(n_spots), clamped to [50, 600]."""
    import math
    return int(min(600, max(50, round(math.sqrt(max(1, n_spots))))))


def suggest_smooth_sigma(coords, grid_res):
    """~2 grid-pixels at the median spot spacing; clamped to [1, 6]."""
    from scipy.spatial import cKDTree
    coords = np.asarray(coords)
    if coords.shape[0] < 2:
        return 2.0
    d, _ = cKDTree(coords).query(coords, k=2)
    median_spacing = float(np.median(d[:, 1]))
    extent = float(max(coords[:, 0].ptp(), coords[:, 1].ptp())) or 1.0
    px = extent / grid_res
    sigma = 2.0 * (median_spacing / px) if px > 0 else 2.0
    return float(min(6.0, max(1.0, sigma)))
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_multicontour.py -v` → PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/multicontour.py backend/tests/test_multicontour.py
git commit -m "multicontour: data-aware grid_res/smooth_sigma suggestions"
```

---

## Task 5: Adaptor two-phase methods + score cache

**Files:**
- Modify: `backend/xcell/adaptor.py` (add methods; init `self._multicontour_cache = {}` in `__init__` near other state ~line 162)
- Test: `backend/tests/test_multicontour_adaptor.py`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_multicontour_adaptor.py
import numpy as np, anndata
from scipy.sparse import csr_matrix
from xcell.adaptor import DataAdaptor


def _adata():
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1: X[k, 0] = X[k, 1] = 10.0
        if i >= 3: X[k, 2] = X[k, 3] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()
    return ad


def test_prepare_requires_pca():
    ad = _adata(); del ad.obsm["X_pca"]
    a = DataAdaptor("x.h5ad", adata=ad)
    try:
        a.prepare_multicontour({"A": ["a0", "a1"]}, contour_levels=3)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "PCA" in str(e)


def test_prepare_then_finalize_writes_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    prep = a.prepare_multicontour(
        {"A": ["a0", "a1"], "B": ["b0", "b1"]}, contour_levels=3,
        grid_res=40, smooth_sigma=2.0)
    token = prep["token"]
    cutoffs = {m["name"]: m["auto_cutoff"] for m in prep["modules"]}
    res = a.finalize_multicontour(token=token, cutoffs=cutoffs, profile_k=5,
                                  out_name="tissue", save_qc=True, params=prep["params"])
    assert res["annotation_key"] == "tissue"
    assert "tissue" in a.adata.obs
    cats = set(a.adata.obs["tissue"].cat.categories)
    assert {"A", "B", "unassigned"}.issubset(cats)
    assert "tissue_status" in a.adata.obs  # save_qc
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_multicontour_adaptor.py -v` → FAIL (no `prepare_multicontour`).

- [ ] **Step 3: Implement methods**

```python
# in adaptor.py, import at top of file alongside other imports is fine, but to
# avoid a circular import (multicontour imports from adaptor), import lazily.

def prepare_multicontour(self, gene_sets, contour_levels=3, log_transform=True,
                         clip_percentiles=(1, 99), grid_res=None, smooth_sigma=None):
    """Phase 1: score each module; cache scores; return review payload (no write)."""
    import uuid
    from xcell import multicontour as mc

    if "X_pca" not in self.adata.obsm:
        raise ValueError("Multi-contour requires X_pca — run PCA first.")
    if self._get_spatial_key() is None:
        raise ValueError("No spatial coordinates found")
    if len(gene_sets) < 2:
        raise ValueError("Select at least 2 gene sets")
    for name, genes in gene_sets.items():
        missing = [g for g in genes if g not in self.adata.var_names]
        if missing:
            raise ValueError(f"Gene set '{name}' has genes not in data: {missing}")

    if grid_res is None:
        grid_res = mc.suggest_grid_res(self.adata.n_obs)
    if smooth_sigma is None:
        smooth_sigma = mc.suggest_smooth_sigma(
            self.adata.obsm[self._get_spatial_key()], grid_res)

    modules, scores = [], {}
    for name, genes in gene_sets.items():
        r = mc.score_module(self.adata, genes, contour_levels, log_transform,
                            tuple(clip_percentiles), grid_res, smooth_sigma)
        scores[name] = {"bands": r["bands"], "thresholds": r["thresholds"]}
        modules.append({
            "name": name, "n_genes": len(genes),
            "thresholds": [float(t) for t in r["thresholds"]],
            "band_values": [float(v) for v in r["band_values"]],
            "histogram": r["histogram"], "auto_cutoff": r["auto_cutoff"],
        })

    token = uuid.uuid4().hex
    params = {"gene_sets": gene_sets, "contour_levels": contour_levels,
              "log_transform": log_transform, "clip_percentiles": list(clip_percentiles),
              "grid_res": grid_res, "smooth_sigma": smooth_sigma}
    self._multicontour_cache[token] = {"scores": scores, "params": params}
    return {"token": token, "modules": modules, "params": params}


def finalize_multicontour(self, token, cutoffs, profile_k=15, out_name="tissue",
                          save_qc=False, params=None):
    """Phase 2: binarize per module, assign, resolve conflicts, write column(s)."""
    from xcell import multicontour as mc

    cached = self._multicontour_cache.get(token)
    if cached is None:
        if params is None:
            raise ValueError("Score cache expired; resubmit with params to recompute")
        # recompute scores deterministically from params
        scores = {}
        for name, genes in params["gene_sets"].items():
            r = mc.score_module(self.adata, genes, params["contour_levels"],
                                params["log_transform"], tuple(params["clip_percentiles"]),
                                params["grid_res"], params["smooth_sigma"])
            scores[name] = {"bands": r["bands"], "thresholds": r["thresholds"]}
    else:
        scores = cached["scores"]
        params = cached["params"]

    highs = {name: mc.binarize(scores[name]["bands"], cutoffs[name]) for name in scores}

    spatial_conn = self.adata.obsp.get("spatial_connectivities")
    coords = np.asarray(self.adata.obsm[self._get_spatial_key()])
    pca = np.asarray(self.adata.obsm["X_pca"])
    labels, status = mc.assign_tissue(highs, self.adata, profile_k, spatial_conn, pca, coords)

    categories = list(scores.keys()) + ["unassigned"]
    self.adata.obs[out_name] = pd.Categorical(labels, categories=categories, ordered=False)
    if save_qc:
        self.adata.obs[f"{out_name}_status"] = pd.Categorical(
            status, categories=["single", "resolved", "unassigned"], ordered=False)
        for name in scores:
            self.adata.obs[f"{name}_high"] = pd.Categorical(
                np.where(highs[name], "high", "low"), categories=["low", "high"])

    counts = {c: int(np.sum(labels == c)) for c in categories}
    result = {"status": "completed", "annotation_key": out_name,
              "categories": categories, "counts": counts,
              "n_resolved": int(np.sum(status == "resolved"))}
    self._log_action("multicontour", {
        "gene_sets": {k: v for k, v in params["gene_sets"].items()},
        "cutoffs": cutoffs, "profile_k": profile_k, "out_name": out_name,
        "params": params}, result)
    self._multicontour_cache.pop(token, None)
    return result
```

Also add `self._multicontour_cache: dict = {}` in `__init__`.

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_multicontour_adaptor.py -v` → PASS (2 tests).

- [ ] **Step 5: Run full backend suite** — `cd backend && pixi run -e dev pytest -q` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_multicontour_adaptor.py
git commit -m "adaptor: prepare/finalize multicontour with score cache"
```

---

## Task 6: API routes

**Files:**
- Modify: `backend/xcell/api/routes.py` (add models + 2 routes near contourize ~2398; add `'multicontour'` to the prerequisites map)
- Test: `backend/tests/test_multicontour_routes.py` (FastAPI TestClient)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_multicontour_routes.py
from fastapi.testclient import TestClient
import numpy as np, anndata
from scipy.sparse import csr_matrix
from xcell.main import app
from xcell.api import routes


def _install_adaptor():
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1: X[k, 0] = X[k, 1] = 10.0
        if i >= 3: X[k, 2] = X[k, 3] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]; ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()
    from xcell.adaptor import DataAdaptor
    routes._adaptors = {"default": DataAdaptor("x.h5ad", adata=ad)}  # adjust to real registry


def test_prepare_and_finalize_roundtrip():
    _install_adaptor()
    c = TestClient(app)
    pr = c.post("/api/scanpy/multicontour/prepare", json={
        "gene_sets": {"A": ["a0", "a1"], "B": ["b0", "b1"]},
        "contour_levels": 3, "grid_res": 40, "smooth_sigma": 2.0})
    assert pr.status_code == 200
    body = pr.json()
    cutoffs = {m["name"]: m["auto_cutoff"] for m in body["modules"]}
    fr = c.post("/api/scanpy/multicontour/finalize", json={
        "token": body["token"], "cutoffs": cutoffs, "profile_k": 5,
        "out_name": "tissue", "save_qc": False, "params": body["params"]})
    assert fr.status_code == 200
    assert fr.json()["annotation_key"] == "tissue"
```

> Note: match `_install_adaptor` to the real adaptor registry used by `get_adaptor` (inspect `routes.py` top for the global). If a fixture file already drives the app, reuse it.

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_multicontour_routes.py -v` → FAIL (404 on the new routes).

- [ ] **Step 3: Implement routes + models**

```python
class MultiContourPrepareRequest(BaseModel):
    gene_sets: dict[str, list[str]]
    contour_levels: int = 3
    log_transform: bool = True
    grid_res: int | None = None
    smooth_sigma: float | None = None

class MultiContourFinalizeRequest(BaseModel):
    token: str
    cutoffs: dict[str, float]
    profile_k: int = 15
    out_name: str = "tissue"
    save_qc: bool = False
    params: dict | None = None


@router.post("/scanpy/multicontour/prepare", status_code=202)
def multicontour_prepare(request: MultiContourPrepareRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        def compute_fn():
            return adaptor.prepare_multicontour(
                gene_sets=request.gene_sets, contour_levels=request.contour_levels,
                log_transform=request.log_transform, grid_res=request.grid_res,
                smooth_sigma=request.smooth_sigma)
        def apply_fn(result):
            return result  # already cached server-side in compute
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/multicontour/finalize")
def multicontour_finalize(request: MultiContourFinalizeRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.finalize_multicontour(
            token=request.token, cutoffs=request.cutoffs, profile_k=request.profile_k,
            out_name=request.out_name, save_qc=request.save_qc, params=request.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

> The cache lives on the adaptor instance, so prepare's compute_fn must run on the same adaptor `apply_fn` writes to — `task_manager.submit` runs both against captured `adaptor`, so the cache persists. The test calls prepare synchronously via TestClient + pollTask; if the route returns a task_id, the route test must poll. To keep the route test simple, you MAY expose prepare synchronously (return the payload directly instead of a task) — decide based on dataset size. **Decision: make prepare cancellable (task_id) to match contourize; update the route test to poll `/scanpy/task/{id}` like the frontend does.** Inspect the existing task polling route name in `routes.py` and use it.

- [ ] **Step 4: Add `'multicontour'` prereq** — in adaptor's prereq map (~2733), add `'multicontour': ['has_spatial']` (PCA checked inside prepare with a clearer message).

- [ ] **Step 5: Run, verify pass** — adjust test to poll if prepare is async; `pytest tests/test_multicontour_routes.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_multicontour_routes.py
git commit -m "routes: /scanpy/multicontour prepare + finalize"
```

---

## Task 7: Frontend — MultiContourModal

**Files:**
- Create: `frontend/src/components/MultiContourModal.tsx`
- Modify: `frontend/src/store.ts` (add `isMultiContourModalOpen`, `setMultiContourModalOpen`)
- Modify: the toolbar/menu that opens Spatial analysis to add a "Multi-contour" entry (find where ScanpyModal's spatial section / Analyze menu lives).

Follow `ScanpyModal.tsx` for: `API_BASE`, `appendDataset`, `pollTask`, `cancelTask`, toast/`addScanpyAction`, `refreshObsSummaries`, `setColorBy`.

- [ ] **Step 1: Add store state**

In `store.ts`, alongside `isScanpyModalOpen`:
```ts
isMultiContourModalOpen: boolean
setMultiContourModalOpen: (open: boolean) => void
```
and in the store creator: `isMultiContourModalOpen: false, setMultiContourModalOpen: (open) => set({ isMultiContourModalOpen: open }),`

- [ ] **Step 2: Build the modal component**

Structure (three views driven by local state `phase: 'select' | 'review' | 'done'`):
1. **select** — list available gene sets (read from the gene-set store/categories the same way ScanpyModal reads modules; reuse the gene-set source used elsewhere). Checkboxes for multi-select. Inputs: `contour_levels` (default 3), `grid_res` (placeholder = suggested), `smooth_sigma` (placeholder = suggested), each with a tooltip (see Task 8 text). Button **Compute** → POST `/scanpy/multicontour/prepare`, then `pollTask(task_id)` → store `modules` + `token` + `params`, go to **review**.
2. **review** — for each module: name, a tiny bar histogram of `histogram` across `band_values`, and a slider/select to choose the high cutoff among `thresholds` (default `auto_cutoff`); live count of high spots = sum of histogram bins at/above cutoff. Inputs: `profile_k` (default 15), `out_name` (default "tissue"), `save_qc` checkbox. Button **Finalize** → POST `/scanpy/multicontour/finalize` with `{token, cutoffs, profile_k, out_name, save_qc, params}`.
3. **done** — show `counts` summary; buttons: "Color by <out_name>" (calls `setColorBy(out_name)` + close) and "Run again".

Wire `refreshObsSummaries()` and `addScanpyAction()` after finalize, mirroring ScanpyModal's post-run block (~1190–1228).

- [ ] **Step 3: Mount the modal** — render `<MultiContourModal />` wherever `<ScanpyModal />` is mounted; add the menu entry that calls `setMultiContourModalOpen(true)`.

- [ ] **Step 4: Verify build + typecheck**

Run: `pixi run build` (or `cd frontend && npm run build`)
Expected: build succeeds, no TS errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MultiContourModal.tsx frontend/src/store.ts frontend/src/<menu-file>
git commit -m "frontend: Multi-contour modal (select -> review/bin -> finalize)"
```

---

## Task 8: Contourize parameter guidance (tooltips + defaults)

**Files:**
- Modify: `frontend/src/components/ScanpyModal.tsx` (~399–409 contourize param descriptions)

- [ ] **Step 1: Update descriptions** to direction-of-effect text:
  - `contour_levels`: "Number of expression bands. Fewer = coarser zones; more = finer gradient. For tissue calling, 2–3 is usually enough."
  - `smooth_sigma`: "Gaussian smoothing in grid pixels. Higher = smoother, larger merged zones; too high blurs distinct regions together. Its real-world radius scales with grid resolution."
  - `grid_res`: "Interpolation grid size per axis. Higher = finer spatial detail but slower; should grow with cell count (≈√N is a good start)."
  - `log_transform`: "log1p before contouring — recommended for raw counts; turn off for already-normalized data."

- [ ] **Step 2: Verify build** — `pixi run build` → success.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "Contourize: clearer parameter guidance tooltips"
```

---

## Task 9: Docs

**Files:**
- Modify: `README.md` (§12 Spatial Contouring — add a Multi-contour subsection)
- Modify: `CHANGELOG.md` ([Unreleased] → Added)

- [ ] **Step 1: README** — after §12, add a "Multi-contour tissue annotation" subsection: select gene sets → Compute → review/bin each module's high cutoff → Finalize → a single `tissue` column; note `X_pca` is required and how multi-high spots are resolved.

- [ ] **Step 2: CHANGELOG** — add an Added entry describing the workflow, the `unassigned` background, conflict resolution, and the parameter-guidance improvements.

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "Docs: multi-contour tissue annotation"
```

---

## Final verification

- [ ] `cd backend && pixi run -e dev pytest -q` → all green.
- [ ] `pixi run build` → frontend builds clean.
- [ ] Manual smoke (optional): launch app, load a spatial dataset with `X_pca`, run Multi-contour with 2 gene sets, confirm a `tissue` column appears and colors sensibly.
- [ ] Merge to `main`, push (requires `pcahan1` auth).
```
