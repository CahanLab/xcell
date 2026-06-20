# Gene Co-expression Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `method='auto'` to the genes-panel "Cluster genes" path that produces purer co-expression modules without the user specifying K, using a robust metric plus a merge/split/prune refinement pass.

**Architecture:** A new dependency-free module `backend/xcell/gene_coexpression.py` holds pure-NumPy primitives (metric standardization → correlation/distance; module eigengene; module coherence; silhouette-cut base clustering; split / merge / prune) and a top-level `auto_coexpression_modules`. `adaptor.cluster_gene_set` gains an `auto` branch that builds the existing `(n_genes, n_cells)` profile matrix and delegates to it. The route and React modal expose the new parameters; the return contract (`list[list[str]]`, trailing unassigned bucket) is unchanged.

**Tech Stack:** Python (NumPy, SciPy `linkage`/`fcluster`/`squareform`/`rankdata`, sklearn `silhouette_score`, `adjusted_rand_score` in tests), FastAPI/Pydantic, React/TypeScript.

**Spec:** `docs/superpowers/specs/2026-06-20-gene-coexpression-clustering-design.md`

**Conventions:**
- Backend tests: `cd backend && pixi run -e dev python -m pytest tests/ -q`
- Frontend check: `cd frontend && npx tsc --noEmit && npm run build`
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- All internal primitives operate on **integer gene row-indices** into the
  `(n_genes, n_cells)` matrix; names are mapped only at the very end.

---

### Task 1: Metric standardization → correlation / distance

**Files:**
- Create: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing test (and the shared planted-data helper used by later tasks)**

Create `backend/tests/test_gene_coexpression.py`:

```python
"""Tests for robust gene co-expression module detection."""
import numpy as np
import pytest

from xcell import gene_coexpression as gc


# ---- shared synthetic helpers (reused by later tasks) ---------------------

def _profiles_from_factors(rng, factors, noise=0.15):
    """Stack genes generated from a list of (factor_vector, n_genes) specs.

    Returns (X_genes, labels): X_genes is (total_genes, n_cells), labels is an
    int ground-truth module id per gene row (noise factor -> its own ids).
    """
    rows, labels = [], []
    for mod_id, (f, n) in enumerate(factors):
        f = (f - f.mean()) / (f.std() + 1e-9)
        for _ in range(n):
            g = f + rng.standard_normal(f.shape[0]) * noise
            rows.append(g)
            labels.append(mod_id)
    return np.asarray(rows, dtype=float), np.asarray(labels)


def test_standardize_rows_are_unit_norm_pearson():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 200))
    Z = gc._standardize_profiles(X, metric="pearson")
    norms = np.linalg.norm(Z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_corr_matrix_recovers_perfect_correlation():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(200)
    X = np.vstack([base, 2 * base + 3, -base])  # corr: 1, 1, -1 with base
    C = gc.corr_matrix(X, metric="pearson")
    assert C.shape == (3, 3)
    assert np.allclose(np.diag(C), 1.0, atol=1e-8)
    assert C[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert C[0, 2] == pytest.approx(-1.0, abs=1e-6)
    assert np.allclose(C, C.T)


def test_distance_matrix_is_one_minus_corr_clipped():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(200)
    X = np.vstack([base, base, -base])
    D = gc.distance_matrix(X, metric="pearson")
    assert np.all(D >= -1e-9) and np.all(D <= 2 + 1e-9)
    assert np.allclose(np.diag(D), 0.0, atol=1e-8)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert D[0, 2] == pytest.approx(2.0, abs=1e-6)


def test_bicor_is_robust_to_outliers_where_pearson_breaks():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(300)
    a = base + rng.standard_normal(300) * 0.1
    b = base + rng.standard_normal(300) * 0.1
    # Corrupt a few cells with large opposite-sign spikes.
    for i in (5, 50, 150, 250):
        a[i] += 30
        b[i] -= 30
    X = np.vstack([a, b])
    pear = gc.corr_matrix(X, metric="pearson")[0, 1]
    bic = gc.corr_matrix(X, metric="bicor")[0, 1]
    assert bic > pear            # bicor less damaged by the outliers
    assert bic > 0.7             # and still recognizes the co-expression


def test_spearman_handles_monotone_nonlinear():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(300)
    X = np.vstack([base, np.exp(base)])  # monotone but nonlinear
    sp = gc.corr_matrix(X, metric="spearman")[0, 1]
    assert sp > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: FAIL with `AttributeError: module 'xcell.gene_coexpression' has no attribute ...` (or `ModuleNotFoundError`).

- [ ] **Step 3: Write the minimal implementation**

Create `backend/xcell/gene_coexpression.py`:

```python
"""Robust gene co-expression module detection.

Pure NumPy / SciPy / sklearn. Operates on a (n_genes, n_cells) expression
matrix (one row per gene). All internal helpers work on integer gene
row-indices; gene names are mapped back only at the top-level boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

_METRICS = ("bicor", "pearson", "spearman")


def _center_unit(X: np.ndarray) -> np.ndarray:
    """Center each row and scale to unit L2 norm (rows of all-equal -> zeros)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return Xc / norm


def _bicor_standardize(X: np.ndarray, c: float = 9.0) -> np.ndarray:
    """Biweight-midcorrelation standardization (WGCNA bicor).

    Rows whose median absolute deviation is 0 fall back to Pearson
    standardization (WGCNA's pearsonFallback).
    """
    med = np.median(X, axis=1, keepdims=True)
    d = X - med
    mad = np.median(np.abs(d), axis=1, keepdims=True)
    zero_mad = mad[:, 0] == 0
    mad_safe = np.where(mad == 0, 1.0, mad)
    u = d / (c * mad_safe)
    w = (1.0 - u ** 2) ** 2
    w[np.abs(u) >= 1.0] = 0.0
    xw = d * w
    norm = np.linalg.norm(xw, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Z = xw / norm
    if zero_mad.any():
        Z[zero_mad] = _center_unit(X[zero_mad])
    return Z


def _standardize_profiles(X: np.ndarray, metric: str) -> np.ndarray:
    """Transform rows so that ``Z @ Z.T`` is the requested correlation matrix."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric!r}; expected one of {_METRICS}")
    X = np.asarray(X, dtype=float)
    if metric == "pearson":
        return _center_unit(X)
    if metric == "spearman":
        ranks = np.vstack([rankdata(row) for row in X])
        return _center_unit(ranks)
    return _bicor_standardize(X)


def corr_matrix(X: np.ndarray, metric: str = "bicor") -> np.ndarray:
    """Gene-by-gene correlation matrix under the chosen robust metric."""
    Z = _standardize_profiles(X, metric)
    C = Z @ Z.T
    np.clip(C, -1.0, 1.0, out=C)
    C = 0.5 * (C + C.T)  # enforce exact symmetry
    np.fill_diagonal(C, 1.0)
    return C


def distance_matrix(X: np.ndarray, metric: str = "bicor") -> np.ndarray:
    """Signed correlation distance ``1 - corr`` in ``[0, 2]`` (zero diagonal)."""
    D = 1.0 - corr_matrix(X, metric)
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: robust metric standardization (bicor/pearson/spearman)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Module eigengene and coherence (PVE)

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
def test_coherence_high_for_single_factor_module():
    rng = np.random.default_rng(10)
    X, _ = _profiles_from_factors(rng, [(rng.standard_normal(200), 10)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    assert gc._module_coherence(Z) > 0.85


def test_coherence_low_for_two_factor_glued_module():
    rng = np.random.default_rng(11)
    f1 = rng.standard_normal(200)
    f2 = rng.standard_normal(200)  # independent of f1
    X, _ = _profiles_from_factors(rng, [(f1, 6), (f2, 6)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    assert gc._module_coherence(Z) < 0.7


def test_single_gene_module_is_perfectly_coherent():
    rng = np.random.default_rng(12)
    Z = gc._standardize_profiles(rng.standard_normal((1, 200)), "pearson")
    assert gc._module_coherence(Z) == 1.0


def test_eigengene_tracks_underlying_factor():
    rng = np.random.default_rng(13)
    f = rng.standard_normal(200)
    X, _ = _profiles_from_factors(rng, [(f, 8)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    eg = gc._module_eigengene(Z)
    assert eg.shape == (200,)
    fc = (f - f.mean()) / np.linalg.norm(f - f.mean())
    assert abs(float(eg @ fc)) > 0.9
    assert np.linalg.norm(eg) == pytest.approx(1.0, abs=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k "coherence or eigengene" -q`
Expected: FAIL (`module ... has no attribute '_module_coherence'`).

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/gene_coexpression.py`:

```python
def _module_coherence(profiles: np.ndarray) -> float:
    """Fraction of variance the top PC (eigengene) explains for a module.

    ``profiles`` are standardized rows (``profiles @ profiles.T`` is the
    correlation matrix with unit diagonal, so its trace is the gene count).
    Returns 1.0 for a single-gene module.
    """
    g = profiles.shape[0]
    if g <= 1:
        return 1.0
    C = profiles @ profiles.T
    w = np.linalg.eigvalsh(C)  # ascending
    return float(w[-1] / g)


def _module_eigengene(profiles: np.ndarray) -> np.ndarray:
    """Module eigengene: top PC over cells, sign-aligned, centered, unit norm."""
    g = profiles.shape[0]
    if g == 1:
        eg = profiles[0].astype(float).copy()
    else:
        C = profiles @ profiles.T
        _, V = np.linalg.eigh(C)
        eg = profiles.T @ V[:, -1]  # length n_cells
    mean_prof = profiles.mean(axis=0)
    if float(eg @ mean_prof) < 0:
        eg = -eg
    eg = eg - eg.mean()
    norm = np.linalg.norm(eg)
    if norm > 0:
        eg = eg / norm
    return eg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: module eigengene and coherence (PVE)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Silhouette-cut base clustering (auto K)

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
from sklearn.metrics import adjusted_rand_score


def test_base_cut_recovers_separated_modules():
    rng = np.random.default_rng(20)
    X, labels = _profiles_from_factors(
        rng,
        [(rng.standard_normal(200), 12),
         (rng.standard_normal(200), 10),
         (rng.standard_normal(200), 8)],
        noise=0.12,
    )
    D = gc.distance_matrix(X, metric="pearson")
    found = gc._auto_cut_hierarchical(D)
    assert len(set(found)) >= 2
    assert adjusted_rand_score(labels, found) > 0.7


def test_base_cut_trivial_for_two_genes():
    rng = np.random.default_rng(21)
    D = gc.distance_matrix(rng.standard_normal((2, 50)), metric="pearson")
    found = gc._auto_cut_hierarchical(D)
    assert found.shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k base_cut -q`
Expected: FAIL (`no attribute '_auto_cut_hierarchical'`).

- [ ] **Step 3: Write the implementation**

Add the imports at the top of `backend/xcell/gene_coexpression.py` (below the existing `from scipy.stats import rankdata`):

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
```

Append to `backend/xcell/gene_coexpression.py`:

```python
def _auto_cut_hierarchical(D: np.ndarray, k_max: int = 20) -> np.ndarray:
    """Average-linkage clustering, K chosen by max silhouette over the distance.

    Returns an integer label per gene. Always yields a partition (every gene
    assigned); the refinement pass corrects under-/over-segmentation.
    """
    g = D.shape[0]
    if g <= 2:
        return np.zeros(g, dtype=int)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    k_hi = min(k_max, g - 1)
    best_labels = np.zeros(g, dtype=int)
    best_score = -np.inf
    for k in range(2, k_hi + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(D, labels, metric="precomputed")
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: silhouette-cut average-linkage base clustering\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Split impure modules

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
def test_split_breaks_glued_two_factor_module():
    rng = np.random.default_rng(30)
    f1 = rng.standard_normal(200)
    f2 = rng.standard_normal(200)
    X, labels = _profiles_from_factors(rng, [(f1, 8), (f2, 8)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    glued = [np.arange(16)]  # both factors as one module
    out = gc.split_impure_modules(
        glued, Z, purity_threshold=0.7, min_genes=3, max_split_depth=2
    )
    assert len(out) == 2
    # each child is predominantly one ground-truth factor
    for child in out:
        vals = labels[child]
        major = np.bincount(vals).max()
        assert major / len(vals) >= 0.8


def test_split_leaves_coherent_module_intact():
    rng = np.random.default_rng(31)
    X, _ = _profiles_from_factors(rng, [(rng.standard_normal(200), 12)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    out = gc.split_impure_modules(
        [np.arange(12)], Z, purity_threshold=0.7, min_genes=3, max_split_depth=2
    )
    assert len(out) == 1
    assert sorted(out[0].tolist()) == list(range(12))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k split -q`
Expected: FAIL (`no attribute 'split_impure_modules'`).

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/gene_coexpression.py`:

```python
def _two_way_split(idx: np.ndarray, Z: np.ndarray, min_genes: int):
    """Average-linkage 2-cut of one module; None if either side < min_genes."""
    sub = Z[idx]
    C = sub @ sub.T
    np.clip(C, -1.0, 1.0, out=C)
    D = 1.0 - C
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    link = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(link, t=2, criterion="maxclust")
    a = idx[labels == 1]
    b = idx[labels == 2]
    if len(a) < min_genes or len(b) < min_genes:
        return None
    return a, b


def _split_recursive(idx, Z, purity_threshold, min_genes, depth):
    profiles = Z[idx]
    parent_pve = _module_coherence(profiles)
    if depth <= 0 or len(idx) < 2 * min_genes or parent_pve >= purity_threshold:
        return [idx]
    pair = _two_way_split(idx, Z, min_genes)
    if pair is None:
        return [idx]
    a, b = pair
    if _module_coherence(Z[a]) > parent_pve and _module_coherence(Z[b]) > parent_pve:
        return (
            _split_recursive(a, Z, purity_threshold, min_genes, depth - 1)
            + _split_recursive(b, Z, purity_threshold, min_genes, depth - 1)
        )
    return [idx]


def split_impure_modules(modules, Z, *, purity_threshold, min_genes, max_split_depth):
    """Recursively split modules whose eigengene PVE < purity_threshold."""
    out = []
    for m in modules:
        out.extend(
            _split_recursive(
                np.asarray(m), Z, purity_threshold, min_genes, max_split_depth
            )
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: split impure modules by eigengene PVE\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: Merge similar modules

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
def test_merge_combines_near_duplicate_modules():
    rng = np.random.default_rng(40)
    f = rng.standard_normal(200)
    X, _ = _profiles_from_factors(rng, [(f, 6), (f, 6)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    out = gc.merge_similar_modules(
        [np.arange(6), np.arange(6, 12)], Z, merge_threshold=0.8
    )
    assert len(out) == 1
    assert sorted(out[0].tolist()) == list(range(12))


def test_merge_keeps_distinct_modules_apart():
    rng = np.random.default_rng(41)
    X, _ = _profiles_from_factors(
        rng, [(rng.standard_normal(200), 6), (rng.standard_normal(200), 6)], noise=0.1
    )
    Z = gc._standardize_profiles(X, "pearson")
    out = gc.merge_similar_modules(
        [np.arange(6), np.arange(6, 12)], Z, merge_threshold=0.8
    )
    assert len(out) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k merge -q`
Expected: FAIL (`no attribute 'merge_similar_modules'`).

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/gene_coexpression.py`:

```python
def merge_similar_modules(modules, Z, *, merge_threshold):
    """Iteratively merge the closest module pair whose eigengenes correlate
    at or above merge_threshold, recomputing eigengenes after each merge."""
    modules = [np.asarray(m) for m in modules]
    while len(modules) >= 2:
        egs = [_module_eigengene(Z[m]) for m in modules]
        best_c, bi, bj = -np.inf, -1, -1
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                c = float(egs[i] @ egs[j])
                if c > best_c:
                    best_c, bi, bj = c, i, j
        if best_c < merge_threshold:
            break
        merged = np.concatenate([modules[bi], modules[bj]])
        modules = [m for k, m in enumerate(modules) if k not in (bi, bj)]
        modules.append(merged)
    return modules
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: merge near-duplicate modules by eigengene correlation\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 6: Prune small modules (reassign or set aside)

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
def test_prune_sets_aside_uncorrelated_small_module():
    rng = np.random.default_rng(50)
    big = (rng.standard_normal(200), 10)
    X, _ = _profiles_from_factors(rng, [big], noise=0.1)
    # two extra pure-noise genes, uncorrelated with the big module
    noise = rng.standard_normal((2, 200))
    Z = gc._standardize_profiles(np.vstack([X, noise]), "pearson")
    modules = [np.arange(10), np.array([10, 11])]
    kept, unassigned = gc.prune_small_modules(
        modules, Z, min_genes=5, reassign_floor=0.5
    )
    assert len(kept) == 1
    assert sorted(unassigned) == [10, 11]


def test_prune_reassigns_correlated_small_module_genes():
    rng = np.random.default_rng(51)
    f = rng.standard_normal(200)
    X, _ = _profiles_from_factors(rng, [(f, 10)], noise=0.1)
    extra = (f + rng.standard_normal((2, 200)) * 0.1)  # correlated with big mod
    Z = gc._standardize_profiles(np.vstack([X, extra]), "pearson")
    modules = [np.arange(10), np.array([10, 11])]
    kept, unassigned = gc.prune_small_modules(
        modules, Z, min_genes=5, reassign_floor=0.5
    )
    assert len(kept) == 1
    assert unassigned == []
    assert sorted(kept[0].tolist()) == list(range(12))


def test_prune_extra_orphans_routed_too():
    rng = np.random.default_rng(52)
    X, _ = _profiles_from_factors(rng, [(rng.standard_normal(200), 10)], noise=0.1)
    orphan = rng.standard_normal((1, 200))
    Z = gc._standardize_profiles(np.vstack([X, orphan]), "pearson")
    kept, unassigned = gc.prune_small_modules(
        [np.arange(10)], Z, min_genes=5, reassign_floor=0.5, extra_orphans=[10]
    )
    assert unassigned == [10]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k prune -q`
Expected: FAIL (`no attribute 'prune_small_modules'`).

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/gene_coexpression.py`:

```python
def _unit(vec: np.ndarray) -> np.ndarray:
    v = vec - vec.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def prune_small_modules(modules, Z, *, min_genes, reassign_floor, extra_orphans=()):
    """Drop modules smaller than min_genes; reassign their genes (and any
    extra_orphans, e.g. base-clustering noise / zero-variance genes) to the
    nearest surviving module whose eigengene correlation >= reassign_floor,
    else collect them into the returned unassigned list.

    Returns (kept_modules, unassigned_gene_indices).
    """
    modules = [np.asarray(m) for m in modules]
    keep = [m for m in modules if len(m) >= min_genes]
    orphans = [int(i) for m in modules if len(m) < min_genes for i in m]
    orphans += [int(i) for i in extra_orphans]
    if not keep:
        return [], sorted(set(orphans))
    egs = [_module_eigengene(Z[m]) for m in keep]
    unassigned = []
    for gi in orphans:
        z = _unit(Z[gi])
        cors = [float(z @ eg) for eg in egs]
        best = int(np.argmax(cors))
        if cors[best] >= reassign_floor:
            keep[best] = np.append(keep[best], gi)
        else:
            unassigned.append(gi)
    return keep, sorted(set(unassigned))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (18 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: prune small modules with eigengene reassignment\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 7: Top-level `auto_coexpression_modules`

**Files:**
- Modify: `backend/xcell/gene_coexpression.py`
- Test: `backend/tests/test_gene_coexpression.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gene_coexpression.py`:

```python
def _names(n):
    return [f"g{i}" for i in range(n)]


def test_auto_recovers_modules_and_partitions_all_genes():
    rng = np.random.default_rng(60)
    X, labels = _profiles_from_factors(
        rng,
        [(rng.standard_normal(220), 12),
         (rng.standard_normal(220), 10),
         (rng.standard_normal(220), 8)],
        noise=0.12,
    )
    n_struct = X.shape[0]
    noise = rng.standard_normal((8, 220))      # 8 uncorrelated noise genes
    tiny_f = rng.standard_normal(220)
    tiny, _ = _profiles_from_factors(rng, [(tiny_f, 2)], noise=0.1)  # 2-gene mod
    Xall = np.vstack([X, noise, tiny])
    names = _names(Xall.shape[0])

    out = gc.auto_coexpression_modules(
        Xall, names, metric="pearson", min_genes=5,
        merge_threshold=0.8, purity_threshold=0.6, max_split_depth=2,
    )

    # every input gene appears exactly once across all returned groups
    flat = [g for grp in out for g in grp]
    assert sorted(flat) == sorted(names)
    assert len(flat) == len(set(flat))

    # the three planted modules are recovered: build a per-gene predicted label
    # over the structured genes and check agreement.
    pred = {}
    for ci, grp in enumerate(out):
        for g in grp:
            pred[g] = ci
    struct_names = names[:n_struct]
    pred_labels = [pred[g] for g in struct_names]
    assert adjusted_rand_score(labels.tolist(), pred_labels) > 0.8


def test_auto_merges_near_duplicate_planted_modules():
    rng = np.random.default_rng(61)
    f = rng.standard_normal(220)
    X, _ = _profiles_from_factors(rng, [(f, 8), (f, 8)], noise=0.12)
    names = _names(16)
    out = gc.auto_coexpression_modules(
        X, names, metric="pearson", min_genes=4,
        merge_threshold=0.8, purity_threshold=0.95, max_split_depth=2,
    )
    # all 16 co-regulated genes end up in a single module (no unassigned)
    main = [grp for grp in out if len(grp) >= 4]
    assert len(main) == 1
    assert len(main[0]) == 16


def test_auto_zero_variance_genes_go_unassigned_not_error():
    rng = np.random.default_rng(62)
    X, _ = _profiles_from_factors(rng, [(rng.standard_normal(200), 8)], noise=0.1)
    flat = np.zeros((1, 200))  # constant gene
    names = _names(9)
    out = gc.auto_coexpression_modules(np.vstack([X, flat]), names, metric="pearson")
    allg = [g for grp in out for g in grp]
    assert "g8" in allg                       # present, not dropped
    assert sorted(allg) == sorted(names)


def test_auto_bicor_robust_to_outliers():
    rng = np.random.default_rng(63)
    facs = [(rng.standard_normal(300), 10), (rng.standard_normal(300), 10)]
    X, labels = _profiles_from_factors(rng, facs, noise=0.1)
    # inject outlier cells into a handful of genes
    for r in (0, 1, 11, 12):
        for ccell in (10, 90, 180, 260):
            X[r, ccell] += 25 * (1 if r < 10 else -1)
    names = _names(X.shape[0])
    out = gc.auto_coexpression_modules(
        X, names, metric="bicor", min_genes=4,
        merge_threshold=0.8, purity_threshold=0.6,
    )
    pred = {g: ci for ci, grp in enumerate(out) for g in grp}
    pred_labels = [pred[n] for n in names]
    assert adjusted_rand_score(labels.tolist(), pred_labels) >= 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -k auto -q`
Expected: FAIL (`no attribute 'auto_coexpression_modules'`).

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/gene_coexpression.py`:

```python
def auto_coexpression_modules(
    X_genes,
    gene_names,
    *,
    metric: str = "bicor",
    min_genes: int = 5,
    merge_threshold: float = 0.8,
    purity_threshold: float = 0.5,
    max_split_depth: int = 2,
    reassign_floor: float = 0.5,
):
    """Detect co-expression modules from a (n_genes, n_cells) matrix.

    Pipeline: robust metric -> silhouette-cut base clustering -> split impure
    -> merge near-duplicates -> prune small (reassign or set aside). Returns a
    list of gene-name lists, ordered by size descending, with a trailing
    "unassigned" group last when any genes are left over.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric!r}; expected one of {_METRICS}")
    gene_names = list(gene_names)
    X_genes = np.asarray(X_genes, dtype=float)
    n_genes = X_genes.shape[0]

    # 1. set aside zero-variance genes (cannot be co-expressed meaningfully)
    var = X_genes.var(axis=1)
    valid_idx = np.where(var > 1e-12)[0]
    zero_var_idx = np.where(var <= 1e-12)[0].tolist()

    if len(valid_idx) < max(min_genes, 2):
        # too few usable genes to form a module: everything is one group
        return [gene_names]

    Z = np.zeros_like(X_genes)
    Z[valid_idx] = _standardize_profiles(X_genes[valid_idx], metric)

    # 2. base clustering on the valid genes
    D = distance_matrix(X_genes[valid_idx], metric)
    base_labels = _auto_cut_hierarchical(D)
    modules = [valid_idx[base_labels == lab] for lab in sorted(set(base_labels))]

    # 3. refinement: split -> merge -> prune
    modules = split_impure_modules(
        modules, Z, purity_threshold=purity_threshold,
        min_genes=min_genes, max_split_depth=max_split_depth,
    )
    modules = merge_similar_modules(modules, Z, merge_threshold=merge_threshold)
    modules, unassigned = prune_small_modules(
        modules, Z, min_genes=min_genes, reassign_floor=reassign_floor,
        extra_orphans=zero_var_idx,
    )

    # 4. order by size desc, map to names, append trailing unassigned bucket
    modules.sort(key=lambda m: len(m), reverse=True)
    result = [[gene_names[i] for i in sorted(m.tolist())] for m in modules]
    if unassigned:
        result.append([gene_names[i] for i in sorted(unassigned)])
    return result
```

- [ ] **Step 4: Run the full module test file**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_gene_coexpression.py -q`
Expected: PASS (22 tests). If `test_auto_bicor_robust_to_outliers` is flaky, widen its seed search once; the assertion threshold (0.7) is intentionally permissive.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_coexpression.py backend/tests/test_gene_coexpression.py
git commit -m "$(printf 'gene_coexpression: top-level auto_coexpression_modules pipeline\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 8: Wire `method='auto'` into `cluster_gene_set`

**Files:**
- Modify: `backend/xcell/adaptor.py` (`cluster_gene_set`, ~`:5443`)
- Test: `backend/tests/test_cluster_gene_set_auto.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cluster_gene_set_auto.py`:

```python
"""cluster_gene_set(method='auto') co-expression module path."""
import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata_with_modules(seed=0):
    rng = np.random.default_rng(seed)
    n_cells = 200
    f1 = rng.standard_normal(n_cells)
    f2 = rng.standard_normal(n_cells)
    cols, names = [], []
    for i in range(8):
        cols.append(f1 + rng.standard_normal(n_cells) * 0.15); names.append(f"A{i}")
    for i in range(8):
        cols.append(f2 + rng.standard_normal(n_cells) * 0.15); names.append(f"B{i}")
    for i in range(4):
        cols.append(rng.standard_normal(n_cells)); names.append(f"N{i}")
    X = np.vstack(cols).T            # (cells, genes)
    X = X - X.min()                  # keep non-negative, count-like
    ad = anndata.AnnData(X=csr_matrix(X.astype(np.float32)))
    ad.var_names = names
    # A 'raw' layer so purity tests read expression directly, bypassing the
    # default normalize_total+log1p path. On a 20-gene synthetic matrix
    # normalize_total injects a compositional confound that does not exist on
    # real (thousands-of-genes) data; the raw layer isolates the algorithm.
    ad.layers["raw"] = ad.X.copy()
    return ad, names


def test_auto_returns_valid_partition():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    # default (normalized) path: partition property is normalization-agnostic.
    clusters = a.cluster_gene_set(names, method="auto", metric="pearson", min_genes=4)
    flat = [g for c in clusters for g in c]
    assert sorted(flat) == sorted(names)         # every gene exactly once
    assert len(flat) == len(set(flat))


def test_auto_groups_coexpressed_genes_together():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    clusters = a.cluster_gene_set(
        names, method="auto", metric="pearson", min_genes=4, layer="raw"
    )
    # find the cluster containing A0; it should hold most of the A-genes and
    # exclude the B-module genes.
    a_cluster = next(c for c in clusters if "A0" in c)
    a_hits = sum(1 for g in a_cluster if g.startswith("A"))
    b_in_a = sum(1 for g in a_cluster if g.startswith("B"))
    assert a_hits >= 6                            # >=6 of 8 A-genes co-cluster
    assert b_in_a == 0                            # no B-genes leak into A


def test_auto_does_not_require_k():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    # no k passed at all
    clusters = a.cluster_gene_set(names, method="auto", layer="raw")
    assert len(clusters) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_cluster_gene_set_auto.py -q`
Expected: FAIL — `cluster_gene_set` raises `Unknown method: 'auto'`.

- [ ] **Step 3: Implement the `auto` branch**

In `backend/xcell/adaptor.py`, update `cluster_gene_set`'s signature to add the
new keyword params (place them after `use_gene_mask`):

```python
    def cluster_gene_set(
        self,
        gene_names: list[str],
        method: str,
        k: int | None = None,
        cell_indices: list[int] | None = None,
        eps: float = 0.3,
        min_samples: int = 3,
        layer: str | None = None,
        use_gene_mask: bool = False,
        metric: str = 'bicor',
        min_genes: int = 5,
        merge_threshold: float = 0.8,
        purity_threshold: float = 0.5,
        max_split_depth: int = 2,
    ) -> list[list[str]]:
```

Change the method-validation block to accept `'auto'` and set its min gene
requirement (replace the existing `if method not in (...)` / `min_required`
logic):

```python
        if method not in ('hierarchical', 'kmeans', 'dbscan', 'auto'):
            raise ValueError(f"Unknown method: {method}")
        if method in ('hierarchical', 'kmeans'):
            if k is None or k < 2:
                raise ValueError(f"{method} requires k >= 2, got {k!r}")
        if method == 'dbscan':
            if not (0.0 < eps <= 2.0):
                raise ValueError(f"dbscan eps must be in (0, 2], got {eps!r}")
            if min_samples < 2:
                raise ValueError(
                    f"dbscan min_samples must be >= 2, got {min_samples!r}"
                )
```

Find the existing minimum-gene-count guard:

```python
        min_required = k if method != 'dbscan' else min_samples
```

and replace it with:

```python
        if method == 'auto':
            min_required = 2
        elif method == 'dbscan':
            min_required = min_samples
        else:
            min_required = k
```

Then add the `auto` branch to the method dispatch. It must run **before** the
`X = X.T` transpose path is consumed — i.e., right after `X_genes` is built and
before the `if method == 'hierarchical':` block, insert:

```python
        if method == 'auto':
            from .gene_coexpression import auto_coexpression_modules
            return auto_coexpression_modules(
                X_genes, found_genes,
                metric=metric, min_genes=min_genes,
                merge_threshold=merge_threshold,
                purity_threshold=purity_threshold,
                max_split_depth=max_split_depth,
            )
```

(`X_genes` and `found_genes` already exist at that point in the function;
`auto` returns directly, bypassing the label-partition tail used by the other
methods.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_cluster_gene_set_auto.py tests/test_cluster_gene_set_mask.py -q`
Expected: PASS (new auto tests + existing mask tests still green).

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_cluster_gene_set_auto.py
git commit -m "$(printf 'adaptor: cluster_gene_set gains auto co-expression module method\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 9: Expose new params on the route

**Files:**
- Modify: `backend/xcell/api/routes.py` (`ClusterGeneSetRequest` ~`:1082`, `cluster_gene_set_route` ~`:2712`)
- Test: `backend/tests/test_cluster_gene_set_route_auto.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_cluster_gene_set_route_auto.py`:

```python
"""POST /cluster_gene_set accepts and forwards the auto-method params."""
import numpy as np
import anndata
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes


def _adata():
    rng = np.random.default_rng(0)
    n_cells = 200
    f1, f2 = rng.standard_normal(n_cells), rng.standard_normal(n_cells)
    cols, names = [], []
    for i in range(8):
        cols.append(f1 + rng.standard_normal(n_cells) * 0.15); names.append(f"A{i}")
    for i in range(8):
        cols.append(f2 + rng.standard_normal(n_cells) * 0.15); names.append(f"B{i}")
    X = np.vstack(cols).T
    X = X - X.min()
    ad = anndata.AnnData(X=csr_matrix(X.astype(np.float32)))
    ad.var_names = names
    ad.layers["raw"] = ad.X.copy()
    return ad, names


def test_route_auto_method(monkeypatch):
    from xcell.adaptor import DataAdaptor
    ad, names = _adata()
    adaptor = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: adaptor)
    client = TestClient(app)
    resp = client.post("/api/cluster_gene_set", json={
        "gene_names": names,
        "method": "auto",
        "cell_context": "all",
        "metric": "pearson",
        "min_genes": 4,
        "merge_threshold": 0.8,
        "purity_threshold": 0.5,
        "max_split_depth": 2,
        "layer": "raw",
    })
    assert resp.status_code == 200, resp.text
    clusters = resp.json()["clusters"]
    flat = [g for c in clusters for g in c]
    assert sorted(flat) == sorted(names)
```

Note: confirm the route prefix. If `app` mounts the router under `/api`, the
path is `/api/cluster_gene_set` (as above). If not, drop the `/api` prefix.
Check with: `grep -n "include_router" backend/xcell/main.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_cluster_gene_set_route_auto.py -q`
Expected: FAIL — `auto` params not forwarded (extra fields ignored → method runs but with bicor defaults; or a 422 if the model rejects). Most likely the test passes the params but they aren't forwarded, so adjust only if it errors. The real failure to drive is the missing forwarding below.

- [ ] **Step 3: Add fields to the request model and forward them**

In `backend/xcell/api/routes.py`, extend `ClusterGeneSetRequest` (after
`use_gene_mask`):

```python
    # method='auto' (co-expression modules) knobs; ignored by other methods.
    metric: str = "bicor"            # 'bicor' | 'pearson' | 'spearman'
    min_genes: int = 5               # min genes per surviving module
    merge_threshold: float = 0.8     # eigengene-corr above which modules merge
    purity_threshold: float = 0.5    # eigengene PVE below which a module splits
    max_split_depth: int = 2         # recursion cap on splitting
```

In `cluster_gene_set_route`, forward them to the adaptor call:

```python
        clusters = adaptor.cluster_gene_set(
            gene_names=req.gene_names,
            method=req.method,
            k=req.k,
            cell_indices=cell_indices,
            eps=req.eps,
            min_samples=req.min_samples,
            layer=req.layer,
            use_gene_mask=req.use_gene_mask,
            metric=req.metric,
            min_genes=req.min_genes,
            merge_threshold=req.merge_threshold,
            purity_threshold=req.purity_threshold,
            max_split_depth=req.max_split_depth,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_cluster_gene_set_route_auto.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all pass (previous 92 + the new tests).

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_cluster_gene_set_route_auto.py
git commit -m "$(printf 'routes: forward auto co-expression params to cluster_gene_set\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 10: Frontend — "Auto" method in the modal

**Files:**
- Modify: `frontend/src/hooks/useData.ts` (`runClusterGeneSet`, ~`:1832`)
- Modify: `frontend/src/components/ClusterGeneSetModal.tsx`

- [ ] **Step 1: Extend the API helper types and payload**

In `frontend/src/hooks/useData.ts`, change the `runClusterGeneSet` `params`
type `method` union and add the auto fields:

```typescript
    method: 'hierarchical' | 'kmeans' | 'dbscan' | 'auto'
```

and add (after `useGeneMask?: boolean`):

```typescript
    /** Auto only: robust correlation metric. */
    metric?: 'bicor' | 'pearson' | 'spearman'
    /** Auto only: minimum genes per surviving module. */
    minGenes?: number
    /** Auto only: eigengene-correlation above which modules merge. */
    mergeThreshold?: number
    /** Auto only: eigengene PVE below which a module is split. */
    purityThreshold?: number
```

and in the `JSON.stringify({ ... })` body add:

```typescript
      metric: params.metric,
      min_genes: params.minGenes,
      merge_threshold: params.mergeThreshold,
      purity_threshold: params.purityThreshold,
```

- [ ] **Step 2: Add the Auto method + controls to the modal**

In `frontend/src/components/ClusterGeneSetModal.tsx`:

Change the `Method` type and default state:

```typescript
type Method = 'auto' | 'hierarchical' | 'kmeans' | 'dbscan'
```

```typescript
  const [method, setMethod] = useState<Method>('auto')
```

Add new state next to the other knobs (after `minSamples`):

```typescript
  const [metric, setMetric] = useState<'bicor' | 'pearson' | 'spearman'>('bicor')
  const [minGenes, setMinGenes] = useState(5)
  const [mergeThreshold, setMergeThreshold] = useState(0.8)
  const [purityThreshold, setPurityThreshold] = useState(0.5)
```

Reset them in the open-modal effect (where `setMethod('hierarchical')` is —
change it to `'auto'` and add the resets):

```typescript
    setMethod('auto')
    setK(3)
    setEps(0.3)
    setMinSamples(3)
    setMetric('bicor')
    setMinGenes(5)
    setMergeThreshold(0.8)
    setPurityThreshold(0.5)
```

Update `isFormValid` so `auto` is always valid (it needs no K):

```typescript
  const isFormValid =
    (cellContext !== 'annotation' ||
      (annotationColumn !== '' && annotationValues.size > 0)) &&
    (
      method === 'auto'
        ? true
        : method === 'dbscan'
          ? eps > 0 && eps <= 2 && minSamples >= 2 && minSamples <= maxMinSamples
          : k >= 2 && k <= maxK
    )
```

Add the Auto option as the first entry in the method `<select>`:

```tsx
            <option value="auto">Auto — co-expression modules (recommended)</option>
            <option value="hierarchical">Hierarchical (Ward linkage)</option>
            <option value="kmeans">K-means</option>
            <option value="dbscan">DBSCAN (density-based)</option>
```

Render auto controls. Replace the existing `{method === 'dbscan' ? (...) : (
<K input> )}` ternary with a three-way branch — `auto` first:

```tsx
        {method === 'auto' ? (
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Similarity metric
            </label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as 'bicor' | 'pearson' | 'spearman')}
              style={{
                width: '100%', padding: '6px 8px', fontSize: '12px',
                backgroundColor: '#0f3460', color: '#eee',
                border: '1px solid #1a1a2e', borderRadius: '4px', marginBottom: '10px',
              }}
            >
              <option value="bicor">Biweight midcorrelation (robust, default)</option>
              <option value="pearson">Pearson</option>
              <option value="spearman">Spearman (rank)</option>
            </select>
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Min genes per module: <span style={{ color: '#eee' }}>{minGenes}</span>
            </label>
            <input
              type="range" min={2} max={20} step={1} value={minGenes}
              onChange={(e) => setMinGenes(parseInt(e.target.value))}
              style={{ width: '100%', marginBottom: '10px' }}
            />
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Merge similarity: <span style={{ color: '#eee' }}>{mergeThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range" min={0.5} max={0.95} step={0.05} value={mergeThreshold}
              onChange={(e) => setMergeThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', marginBottom: '10px' }}
            />
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Split sensitivity (purity): <span style={{ color: '#eee' }}>{purityThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range" min={0.3} max={0.9} step={0.05} value={purityThreshold}
              onChange={(e) => setPurityThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
              No need to pick a cluster count — modules are discovered, then merged,
              split, and pruned (modules under "min genes" are set aside as "unassigned").
            </div>
          </div>
        ) : method === 'dbscan' ? (
```

(keep the existing dbscan `<>...</>` block, then the existing `) : (` K-input
block, then the closing `)}` — the result is `auto ? ... : dbscan ? ... : K`.)

Finally, include the new fields in the `payload` built in `handleRun` (add after
`useGeneMask`):

```typescript
        ...(method === 'auto'
          ? { metric, minGenes, mergeThreshold, purityThreshold }
          : {}),
```

- [ ] **Step 3: Typecheck and build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: no type errors; build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useData.ts frontend/src/components/ClusterGeneSetModal.tsx
git commit -m "$(printf 'frontend: Auto co-expression method in Cluster genes modal\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 11: End-to-end smoke test + final verification

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 2: Frontend typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 3: Playwright smoke on an isolated stack**

Start an isolated backend + frontend (do NOT touch the user's :8000/:5173):

```bash
cd backend && XCELL_DATA_PATH="$(pwd)/xcell/data/toy_spatial_ligrec.h5ad" \
  pixi run -e dev uvicorn xcell.main:app --host 127.0.0.1 --port 8001 &
cd frontend && XCELL_BACKEND=http://127.0.0.1:8001 npx vite --port 5180 --strictPort &
```

Drive with Playwright MCP at `http://127.0.0.1:5180`:
- Open a gene set in the Genes panel, choose its "…" → Cluster genes.
- Confirm "Auto — co-expression modules" is the default method and there is no
  "Number of clusters (K)" box; the metric dropdown + sliders are shown.
- Click Run; confirm a new `… sub-clusters (…)` folder of gene-set modules
  appears under gene_clusters.

Then stop both servers and delete any screenshot PNGs you created.

- [ ] **Step 4: Update the spec's status line**

Append a short "Implemented (2026-06-20)" note to
`docs/superpowers/specs/2026-06-20-gene-coexpression-clustering-design.md`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "$(printf 'gene-coexpression clustering: verified end-to-end\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-review notes
- **Spec coverage:** robust metric (Task 1) ✓; auto-K base (Task 3) ✓; split
  (Task 4) ✓; merge (Task 5) ✓; prune/min-size (Task 6) ✓; top-level pipeline
  (Task 7) ✓; adaptor wiring + no-K (Task 8) ✓; route params (Task 9) ✓; UI
  default + controls (Task 10) ✓; tests/build/smoke (Task 11) ✓.
- **Return contract** unchanged (`list[list[str]]` + trailing unassigned), so the
  existing modal folder-creation code is untouched.
- **Naming consistency:** `auto_coexpression_modules`, `corr_matrix`,
  `distance_matrix`, `_standardize_profiles`, `_module_eigengene`,
  `_module_coherence`, `_auto_cut_hierarchical`, `split_impure_modules`,
  `merge_similar_modules`, `prune_small_modules` — used identically across tasks.
- **Out of scope** (untouched): `run_cluster_genes` (.var Leiden path); existing
  hierarchical/kmeans/dbscan behavior.
