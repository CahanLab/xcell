# PySingleCellNet — upstream issues found while integrating with xcell

**Audience:** a session working on
[CahanLab/PySingleCellNet](https://github.com/CahanLab/PySingleCellNet). This
document is self-contained; you do not need the conversation it came from.

**Found:** 2026-08-06, while building xcell's Cell Typing panel
(`backend/xcell/pyscn.py` + `PyscnModal.tsx` in CahanLab/xcell).
**Against:** PyPI `pySingleCellNet` **0.1.5**; `master` at `09f8f10cd250`
(2026-02-25). Re-check line numbers before editing — they are from 0.1.5.

All findings are in `src/pySingleCellNet/tools/classifier.py` unless noted.
Every one was hit or confirmed by running the code, not by reading alone.

> **Scope note.** xcell already works around #1–#4 on its own side, so nothing
> here is blocking for xcell. They are listed because they will bite anyone
> else using the library directly, and because the workarounds belong upstream.

---

## Summary

| # | Issue | Severity | Fix size |
|---|---|---|---|
| 1 | Gene symbols containing `_` cannot round-trip through the pair encoding | **Hard failure**, opaque error | Medium (format change) |
| 2 | `nrand > 0` breaks `classify_anndata` | Hard failure | Small |
| 3 | Genes missing from the query are zero-filled silently | **Silently wrong answers** | Small |
| 4 | Whole query densified before being subset to the classifier's genes | Memory blow-up | Small |
| 5 | `categorize_classification` needs a graph `train_classifier` doesn't save | Feature not portable | Medium |
| 6 | No `save_classifier` / `load_classifier` | Ergonomics | Small |
| 7 | `__version__` declared in `__all__` but never bound | Cosmetic | Trivial |

Suggested order: **3** (highest consequence, smallest fix) → **4** → **2** →
**7** → **6** → **1** (needs #6 for a migration path) → **5**.

---

## 1. Gene symbols containing `_` cannot round-trip

Top-scoring pairs are encoded as the string `f"{geneA}_{geneB}"` and decoded by
splitting on `_`:

```python
# _query_transform, classifier.py:63-67
for g in genePairs:
    sp = g.split("_")
    genes1.append(sp[0])
    genes2.append(sp[1])
expTemp = expMat.loc[:, np.unique(np.concatenate([genes1, genes2]))]
```

A symbol that itself contains an underscore decodes into fragments that are not
genes. `Mesen_1` paired with `Ubiq_3` becomes `"Mesen_1_Ubiq_3"`, which splits
to `Mesen` and `1`.

**Reproduction** — this is not hypothetical: the toy dataset bundled with xcell
(`backend/xcell/data/toy_spatial.h5ad`) names all 76 of its genes this way, and
training on it fails.

```python
import numpy as np, pandas as pd, anndata, scanpy as sc, pySingleCellNet as cn
from scipy.sparse import csr_matrix

rng = np.random.default_rng(0)
ad = anndata.AnnData(X=csr_matrix(rng.poisson(3, (120, 200)).astype(np.float32)))
ad.var_names = [f"mod_{i}" for i in range(200)]          # <-- underscores
ad.obs["celltype"] = pd.Categorical(["a"] * 60 + ["b"] * 60)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad); sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=150, flavor="seurat_v3", layer="counts")
cn.tl.train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)
```

**Observed:** `KeyError: "None of [Index([...])] are in the [columns]"`, where
the index lists decoded *fragments* (`'1'`, `'10'`, `'mod'`, …) rather than
genes. Nothing in the message points at the cause.

**Suggested fix.** Stop making the pair identity a joined string. Either keep
`topPairs` as an `(n_pairs, 2)` array / list of tuples and join only for display
labels, or use a delimiter that cannot occur in a gene symbol. Note this changes
the on-disk `topPairs` format, so it needs a version stamp and a migration path
— see #6. A cheap interim step is to validate in `train_classifier` and raise a
clear error naming the offending symbols.

**What xcell does meanwhile:** excludes underscore-containing symbols before
training and reports how many; refuses with an explanation when too few remain;
and rejects a classifier whose `tpGeneArray` contains them at load time rather
than letting it fail mid-prediction.

---

## 2. `nrand > 0` breaks `classify_anndata`

Already flagged in a source comment at `classifier.py:485`:

```python
# there is an issue in that making the random profiles here will break later
# addition of results to original annData object
def _rf_classPredict(rfObj, expQuery, numRand=50):
    if numRand > 0:
        randDat = _randomize(expQuery, num=numRand)
        expQuery = pd.concat([expQuery, randDat])       # <-- extra rows
```

`classify_anndata` then does `adata.obsm['SCN_score'] = classRes`
(`classifier.py:452`) with a `classRes` that has `n_cells + nrand` rows, which
cannot be assigned.

The default is `nrand=0`, so it stays latent — but the parameter is public and
documented in the docstring as "Number of random permutations for the null
distribution", which reads like something a user should be able to set.

**Suggested fix.** Either raise when `nrand > 0` in `classify_anndata`, or keep
the null distribution separate from the per-cell result (return it, or store it
in `.uns`) instead of concatenating it into the same frame.

**What xcell does meanwhile:** pins `nrand=0` and does not expose it.

---

## 3. Genes missing from the query are zero-filled silently

```python
# _scn_predict, classifier.py:481
expValTrans = _query_transform(
    expDat.reindex(labels=rf_tsp['tpGeneArray'], axis='columns', fill_value=0),
    rf_tsp['topPairs'],
)
```

Any classifier gene absent from the query becomes a column of zeros. Every pair
feature involving it then collapses to a constant `False` (`0 > 0`), so the
forest votes on degenerate input — but the output looks exactly like a healthy
run. A query sharing 10% of the classifier's genes returns scores as confident
as one sharing 100%.

This is the highest-consequence item here, because there is no signal at all: no
warning, no exception, no field in the result. The most likely real-world cause
is a species/symbol-convention mismatch (`ACTB` vs `Actb`), which produces ~0%
overlap and still "works".

**Suggested fix.** Compute the overlap in `_scn_predict` and, at minimum, warn.
Better: return it (or attach it to `adata.uns['SCN_gene_overlap']`) so it is
inspectable after the fact, and raise below some floor. Even a one-line
`warnings.warn(f"{n_missing}/{n_required} classifier genes absent from the
query; scores may be unreliable")` would remove the whole failure mode.

**What xcell does meanwhile:** computes coverage before every run and shows it
banded ok / warn / error with the missing symbols listed, and detects the pure
`ACTB`/`Actb` case specifically so it can offer a case-insensitive match instead
of reporting 0% overlap.

---

## 4. The whole query is densified before being subset

```python
# _scn_predict, classifier.py:480-481
expDat = pd.DataFrame(data=aDat.X.toarray(),                 # n_cells x n_genes, dense, float64
                      index=aDat.obs.index.values,
                      columns=aDat.var.index.values)
expValTrans = _query_transform(
    expDat.reindex(labels=rf_tsp['tpGeneArray'], axis='columns', fill_value=0), ...)
```

The dense intermediate is sized by the *dataset*, though only the classifier's
genes (typically a few hundred) are ever used. For a 100k x 30k query that is
~24 GB allocated to reach a matrix of ~100k x 300 (~0.2 GB).

**Suggested fix.** Subset columns before densifying — take the intersection of
`aDat.var_names` with `rf_tsp['tpGeneArray']`, slice the sparse matrix, then
`.toarray()`, then `reindex` to add the genuinely-missing genes as zeros.
Chunking over cells on top of that bounds peak memory regardless of query size.

**What xcell does meanwhile:** hands PySCN a query AnnData already narrowed to
exactly `tpGeneArray` in order (zero-filling absent genes itself), so the
internal `reindex` is a no-op reorder, and chunks prediction over cells.

**Not an issue:** the `isinstance(aDat.X, np.ndarray)` guard at
`classifier.py:476-478` that wraps a dense `.X` in
`anndata._core.views.ArrayView` — I checked, and `ArrayView` does provide
`.toarray()`, so line 480 works for both dense and sparse `.X`. Noting it here
only so nobody re-investigates it.

---

## 5. `categorize_classification` needs a graph `train_classifier` doesn't save

`tl.categorize_classification` splits cells that clear more than one class
threshold into **Intermediate** (the classes are within `k` edges in a
cell-type relatedness graph) and **Hybrid** (they are not). The graph is a
required argument:

```python
# categorize.py:73-74
if graph is None:
    raise ValueError("A valid iGraph 'graph' must be provided. None was given.")
```

It is built from the *reference* data by `tl.paga_connectivities_to_igraph`.
But `train_classifier` returns only:

```python
{'tpGeneArray', 'topPairs', 'classifier', 'diffExpGenes', 'ctColors', 'argList'}
```

So the graph does not travel with the classifier. Anyone who receives a trained
`.pkl` — the normal way these are shared — cannot reproduce the
Intermediate/Hybrid distinction at all, only Singular / None / Rand.

**Suggested fix.** Have `train_classifier` build the PAGA graph on the training
data and stash it in the returned dict (e.g. `'ctGraph'`), then let
`categorize_classification` default to `graph=clf['ctGraph']`. That makes the
full categorization portable to any consumer, which is what the feature is for.

**What xcell does meanwhile:** reports those cells as **Ambiguous** — declining
to guess a distinction it has no information to make — and documents the gap in
its UI and `docs/ROADMAP.md`.

---

## 6. No `save_classifier` / `load_classifier`

There is no persistence helper anywhere in the package (checked `utils/misc.py`,
`utils/adataTools.py`, `tools/*`). The quickstart notebook trains and uses `clf`
in-process; users who want to keep one hand-roll `pickle.dump`.

That is workable but leaves no place to record a format version — which #1's fix
will need — and no single point to validate a loaded object. A `load_classifier`
that checks for the required keys and a fitted estimator would also turn today's
failure mode (an `AttributeError` or `KeyError` from deep inside `_scn_predict`)
into one clear message at load time.

**Suggested fix.** `ut.save_classifier(clf, path)` / `ut.load_classifier(path)`
writing a dict with a `'format_version'` key, validating on load, and migrating
older payloads.

**What xcell does meanwhile:** `xcell/pyscn.py::load_classifier` validates the
required keys, checks for a fitted estimator with `predict_proba`, and turns a
`ModuleNotFoundError` during unpickling into an install hint. Reusable as a
starting point.

---

## 7. `__version__` declared in `__all__` but never bound

```python
# src/pySingleCellNet/__init__.py
__all__ = ["__version__", "pl", "ut", "tl"]     # but __version__ is never imported
```

`setuptools_scm` writes `src/pySingleCellNet/_version.py`, but `__init__.py`
never imports from it, so `pySingleCellNet.__version__` raises `AttributeError`
and `getattr(cn, '__version__', ...)` silently falls through to the default.

```python
>>> import pySingleCellNet as cn; cn.__version__
AttributeError: module 'pySingleCellNet' has no attribute '__version__'
>>> import importlib.metadata; importlib.metadata.version('pySingleCellNet')
'0.1.5'
```

**Suggested fix.** `from ._version import version as __version__` in
`__init__.py`, with a `try/except ImportError` fallback to
`importlib.metadata.version(__name__)` for editable/source checkouts.

---

## Cross-cutting suggestion

Issues #1, #3 and #5 share a shape: the library accepts input it cannot serve
correctly and proceeds anyway. A short validation pass at the two entry points
— `train_classifier` on the training data, `classify_anndata` on the query and
classifier — would catch all three at the boundary, where the error can name
the actual problem, instead of deep in pandas or not at all.

## Related

- xcell's adapter: `backend/xcell/pyscn.py`, tests in
  `backend/tests/test_pyscn.py` (43 tests; the ones exercising PySCN skip when
  it is not installed). Some may be liftable as upstream regression tests —
  notably the underscore-gene and gene-overlap cases.
- xcell's own roadmap entry for this integration: `docs/ROADMAP.md`, "Cell
  typing (PySingleCellNet)".
