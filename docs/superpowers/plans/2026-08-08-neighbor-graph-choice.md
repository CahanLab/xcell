# Neighbor-Graph Choice for UMAP and Leiden — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user choose which `obsp` connectivity graph UMAP and Leiden run on, and name the output after that graph so results from different graphs coexist.

**Architecture:** `run_umap` and `run_leiden` gain `graph_key`. Leiden passes it straight to `sc.tl.leiden(obsp=...)`. UMAP has no such parameter, so the adaptor synthesizes a `uns['neighbors']`-shaped entry for the chosen graph, passes it as `neighbors_key`, and deletes it in a `finally`. Output names derive from the graph key through one shared helper that the frontend also reads via `/api/scanpy/neighbor_graphs`.

**Tech Stack:** Python 3 / FastAPI / scanpy 1.11.5 / squidpy / anndata / scipy.sparse; React 18 + TypeScript + Zustand; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-08-08-neighbor-graph-choice-for-umap-leiden-design.md`

## Global Constraints

- Branch `feat/graph-choice` off `main`; merge back with `--no-ff`.
- TDD: write the failing test, run it, watch it fail, then implement.
- Run backend tests from `backend/`: `pixi run -e dev pytest`.
- Frontend typecheck: `cd frontend && npx tsc --noEmit`. There is no lint step.
- `backend/tests/` has **no `conftest.py`** — each test file is self-contained, with module-level `_adata()` / `_adaptor()` factories, seeded `np.random.default_rng(0)`, `csr_matrix` + `float32`.
- `DataAdaptor("x.h5ad", adata=ad)` skips disk I/O; the path is a dummy.
- Heavy imports (`squidpy`, `scipy.stats`) stay function-local. `scanpy as sc`, `numpy as np`, `pandas as pd` are already module-level in `adaptor.py`.
- Error mapping in routes: `ValueError → 400`, `KeyError → 404`. `except HTTPException: raise` goes before any catch-all.
- Anything that mutates AnnData calls `self._log_action(action, params, result)`.
- Comments explain *why*, not what.

**Two deliberate refinements over the spec**, both discovered while verifying against the real scanpy:

1. **`key_added` blank means "derive the default", not an error.** The spec called for a `ValueError`. The field is prefilled in the UI, so clearing it expresses "give me the default" — erroring there is friction with no upside.
2. **`params['use_rep'] = 'X'` when there is no PCA.** `sc.tl.umap` calls `_choose_representation` unconditionally, and when `X_pca` is absent and `n_vars > 50` scanpy **silently runs `sc.pp.pca` and writes `X_pca` + `uns['pca']` into the live object**. That would put an embedding into the AnnData that no recorded step created, so the Analysis Record's exported notebook would not reproduce the state. Verified both ways: with `use_rep='X'`, nothing leaks. This is the headline case — Visium + Spatial Neighbors, no PCA run — so it is not an edge case.

---

### Task 1: Naming helpers and `suffix` on the graph list

One rule for deriving names from a graph key, in one place, so the backend default and the frontend prefill cannot drift.

**Files:**
- Modify: `backend/xcell/adaptor.py` — add two module-level helpers near `_contour_score_field` (~line 246, just above `class DataAdaptor`); extend `list_neighbor_graphs` (~line 5348).
- Test: `backend/tests/test_graph_choice.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `_graph_suffix(graph_key: str | None) -> str`
  - `_default_output_name(base: str, graph_key: str | None) -> str`
  - `list_neighbor_graphs()` entries gain `'suffix': str`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_graph_choice.py`:

```python
"""Choosing which obsp connectivity graph UMAP and Leiden run on.

The point of these tests is that picking a non-default graph must not disturb
anything else: the default path stays byte-identical, the temporary neighbors
entry never survives the call, and scanpy is never given the chance to compute
a PCA nobody asked for.
"""

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor, _default_output_name, _graph_suffix


def _adata(n_genes=8):
    """A small grid of cells with spatial coords and two gene blocks."""
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = rng.poisson(1.0, (n, n_genes)).astype(np.float32)
    # Left half expresses the first block, right half the second, so a graph
    # built on either expression or space has real structure to cluster.
    X[coords[:, 0] <= 2, : n_genes // 2] += 10.0
    X[coords[:, 0] > 2, n_genes // 2:] += 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obsm["spatial"] = coords
    return ad


def _adaptor(**kw):
    return DataAdaptor("x.h5ad", adata=_adata(**kw))


def test_graph_suffix_rules():
    assert _graph_suffix(None) == ""
    assert _graph_suffix("") == ""
    # The default graph contributes no suffix, so X_umap keeps its name.
    assert _graph_suffix("connectivities") == ""
    assert _graph_suffix("spatial_connectivities") == "spatial"
    assert _graph_suffix("expr_plus_space_connectivities") == "expr_plus_space"
    # A key that does not follow the convention is used whole rather than dropped.
    assert _graph_suffix("weird") == "weird"


def test_default_output_name():
    assert _default_output_name("X_umap", None) == "X_umap"
    assert _default_output_name("X_umap", "connectivities") == "X_umap"
    assert _default_output_name("X_umap", "spatial_connectivities") == "X_umap_spatial"
    assert _default_output_name("leiden", "spatial_connectivities") == "leiden_spatial"


def test_list_neighbor_graphs_reports_suffix():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)

    graphs = {g["key"]: g for g in a.list_neighbor_graphs()}
    assert graphs["connectivities"]["suffix"] == ""
    assert graphs["spatial_connectivities"]["suffix"] == "spatial"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q
```

Expected: `ImportError: cannot import name '_default_output_name'`.

If `run_spatial_neighbors` turns out to be a `prepare_*` pair rather than a direct method, check its real signature with `rg -n "def run_spatial_neighbors|def prepare_spatial_neighbors" backend/xcell/adaptor.py` and drive it the way `backend/tests/` already does — grep for `spatial_neighbors` in the existing tests and copy that call.

- [ ] **Step 3: Implement the helpers**

In `backend/xcell/adaptor.py`, immediately above `class DataAdaptor:`:

```python
_CONN_SUFFIX = '_connectivities'


def _graph_suffix(graph_key: str | None) -> str:
    """Short name for a connectivity graph, used to name what it produces.

    'connectivities' is the default graph, so it contributes no suffix and
    UMAP/Leiden keep writing X_umap / leiden — nothing changes for a user who
    never touches the picker. Every other graph is named after its prefix so
    results from different graphs sit side by side instead of overwriting.
    """
    if not graph_key or graph_key == 'connectivities':
        return ''
    if graph_key.endswith(_CONN_SUFFIX):
        return graph_key[: -len(_CONN_SUFFIX)]
    return graph_key


def _default_output_name(base: str, graph_key: str | None) -> str:
    """'X_umap' + spatial_connectivities -> 'X_umap_spatial'."""
    suffix = _graph_suffix(graph_key)
    return f'{base}_{suffix}' if suffix else base
```

- [ ] **Step 4: Add `suffix` to the graph list**

In `list_neighbor_graphs` (~line 5373), extend the appended dict:

```python
                graphs.append({
                    'key': key,
                    'label': label,
                    'n_edges': nnz,
                    'suffix': _graph_suffix(key),
                })
```

- [ ] **Step 5: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_graph_choice.py
git commit -m "feat(graphs): derive output names from the connectivity graph key"
```

---

### Task 2: UMAP on any graph

**Files:**
- Modify: `backend/xcell/adaptor.py` — add `_GRAPH_META_KEY`, `_umap_neighbors_meta`, `_umap_call` beside the Task 1 helpers; rewrite `run_umap` (~line 5657); add `_require_graph` as a `DataAdaptor` method next to `run_umap`.
- Test: `backend/tests/test_graph_choice.py`

**Interfaces:**
- Consumes: `_graph_suffix`, `_default_output_name` (Task 1).
- Produces:
  - `_GRAPH_META_KEY: str = '_xcell_graph'`
  - `_umap_neighbors_meta(adata, graph_key: str) -> dict`
  - `DataAdaptor._require_graph(self, graph_key: str) -> None` (raises `ValueError`)
  - `run_umap(min_dist=0.5, spread=1.0, n_components=2, graph_key=None, key_added=None, active_cell_indices=None) -> dict` with `embedding_name` and `graph_key` in the result.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_graph_choice.py`:

```python
# --- UMAP -----------------------------------------------------------------

def _spatial_only():
    """Visium-shaped: spatial graph, no PCA, no expression kNN.

    n_vars is above scanpy's N_PCS (50) on purpose — that is the branch where
    _choose_representation would otherwise compute a PCA behind our back.
    """
    a = DataAdaptor("x.h5ad", adata=_adata(n_genes=60))
    a.run_spatial_neighbors(n_neighs=6)
    return a


def test_umap_on_spatial_graph_writes_its_own_embedding():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_umap()                       # the default one
    a.run_spatial_neighbors(n_neighs=6)

    res = a.run_umap(graph_key="spatial_connectivities")

    assert res["embedding_name"] == "X_umap_spatial"
    assert res["graph_key"] == "spatial_connectivities"
    assert a.adata.obsm["X_umap_spatial"].shape == (a.n_cells, 2)
    # The expression UMAP must survive untouched — coexisting is the point.
    assert "X_umap" in a.adata.obsm
    assert not np.allclose(a.adata.obsm["X_umap"], a.adata.obsm["X_umap_spatial"])


def test_umap_leaves_no_temporary_neighbors_entry():
    from xcell.adaptor import _GRAPH_META_KEY

    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)
    a.run_umap(graph_key="spatial_connectivities")

    assert _GRAPH_META_KEY not in a.adata.uns


def test_temporary_entry_is_removed_even_when_umap_raises(monkeypatch):
    from xcell import adaptor as adaptor_mod
    from xcell.adaptor import _GRAPH_META_KEY

    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)

    def boom(*args, **kwargs):
        raise RuntimeError("umap exploded")

    monkeypatch.setattr(adaptor_mod.sc.tl, "umap", boom)
    with pytest.raises(RuntimeError):
        a.run_umap(graph_key="spatial_connectivities")

    # A fabricated neighbors entry left behind would be silently picked up by
    # the next tool that reads uns.
    assert _GRAPH_META_KEY not in a.adata.uns


def test_umap_on_a_graph_with_no_uns_entry_at_all():
    """combine_neighbor_graphs writes only the matrix.

    Constructing a minimal uns entry is not enough either — NeighborsView
    requires distances_key to be present, though the matrix it names is never
    read.
    """
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)
    a.combine_neighbor_graphs(
        sources=[{"key": "connectivities", "weight": 1.0},
                 {"key": "spatial_connectivities", "weight": 1.0}],
        target_key="blend_connectivities",
    )
    assert "blend_connectivities" in a.adata.obsp

    res = a.run_umap(graph_key="blend_connectivities")

    assert res["embedding_name"] == "X_umap_blend"
    assert not np.isnan(a.adata.obsm["X_umap_blend"]).any()


def test_umap_on_spatial_graph_without_pca_does_not_invent_one():
    """scanpy's _choose_representation runs sc.pp.pca when X_pca is missing.

    That would put an embedding into the object that no recorded step created,
    so the exported notebook would not reproduce the state. use_rep='X' takes
    the branch that just reads .X.
    """
    a = _spatial_only()
    a.run_umap(graph_key="spatial_connectivities")

    assert "X_umap_spatial" in a.adata.obsm
    assert "X_pca" not in a.adata.obsm
    assert "pca" not in a.adata.uns


def test_umap_keeps_a_real_uns_entrys_recorded_params():
    from xcell.adaptor import _umap_neighbors_meta

    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6)
    meta = _umap_neighbors_meta(a.adata, "spatial_connectivities")

    assert meta["connectivities_key"] == "spatial_connectivities"
    assert meta["distances_key"] == "spatial_distances"
    # scanpy reads params['method'] unguarded; squidpy never writes it.
    assert meta["params"]["method"] == "umap"
    # squidpy's own record survives, so the exported notebook is not invented.
    assert meta["params"]["n_neighbors"] == 6


def test_umap_default_path_is_unchanged():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_umap(min_dist=0.4, spread=1.2)

    assert res["embedding_name"] == "X_umap"
    assert "X_umap" in a.adata.obsm
    step = a._action_history[-1]
    # Recording graph_key/key_added on the default path would change every
    # existing export; it is only recorded when it differs from the default.
    assert step["params"] == {"min_dist": 0.4, "spread": 1.2, "n_components": 2}


def test_umap_custom_name_on_the_default_graph_is_recorded():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_umap(key_added="X_umap_v2")

    assert res["embedding_name"] == "X_umap_v2"
    assert a._action_history[-1]["params"]["key_added"] == "X_umap_v2"


def test_umap_blank_name_falls_back_to_the_derived_default():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)

    assert a.run_umap(graph_key="spatial_connectivities",
                      key_added="   ")["embedding_name"] == "X_umap_spatial"


def test_umap_rejects_an_unknown_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    with pytest.raises(ValueError) as e:
        a.run_umap(graph_key="nope_connectivities")
    assert "nope_connectivities" in str(e.value)
    assert "connectivities" in str(e.value)   # lists what is available


def test_umap_rejects_a_wrong_shaped_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.adata.obsp["bad_connectivities"] = csr_matrix((a.n_cells, a.n_cells - 1))
    with pytest.raises(ValueError) as e:
        a.run_umap(graph_key="bad_connectivities")
    assert "shape" in str(e.value)


def test_umap_on_a_selection_slices_the_chosen_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)
    active = list(range(20))

    res = a.run_umap(graph_key="spatial_connectivities", active_cell_indices=active)

    E = a.adata.obsm[res["embedding_name"]]
    assert not np.isnan(E[active]).any()
    assert np.isnan(E[20:]).all()
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q
```

Expected: the new tests fail — `TypeError: run_umap() got an unexpected keyword argument 'graph_key'` and `ImportError` for `_umap_neighbors_meta` / `_GRAPH_META_KEY`. Tasks 1's three tests still pass.

- [ ] **Step 3: Add the metadata synthesis**

In `backend/xcell/adaptor.py`, beside the Task 1 helpers. Add `from collections.abc import Mapping` to the module imports:

```python
# Where the synthesized neighbors entry lives while sc.tl.umap reads it. Also
# emitted by codegen, so the exported notebook uses the same name.
_GRAPH_META_KEY = '_xcell_graph'


def _umap_neighbors_meta(adata, graph_key: str) -> dict[str, Any]:
    """A ``uns['neighbors']``-shaped entry that ``sc.tl.umap`` will accept.

    scanpy reads three fields without guarding them, and the graphs xcell
    produces are each missing one:

    - squidpy's ``uns['spatial_neighbors']`` has no ``params['method']``, and
      ``tl.umap`` evaluates it to decide whether to warn -> ``KeyError``.
    - ``combine_neighbor_graphs`` writes only the obsp matrix, no ``uns`` entry
      at all, and ``NeighborsView`` requires ``distances_key`` to be present.
      The matrix it names is never read — UMAP uses connectivities only — so a
      key pointing at nothing is fine.

    Start from any real entry that already points at this graph, so squidpy's
    recorded ``n_neighbors`` reaches the exported notebook instead of an
    invented one, then fill the gaps.
    """
    source: Mapping = {}
    for entry in adata.uns.values():
        if isinstance(entry, Mapping) and entry.get('connectivities_key') == graph_key:
            source = entry
            break

    params = dict(source.get('params', {}))
    params['method'] = 'umap'

    # Without a representation scanpy falls through to _get_pca_or_small_x,
    # which *computes and stores* a PCA when X_pca is missing and n_vars > 50.
    # That would leave an embedding in the object that no recorded step made,
    # so the exported notebook would not reproduce it. Reading .X costs nothing
    # here: for method='umap' the matrix is not what drives the layout.
    if 'use_rep' not in params and 'X_pca' not in adata.obsm:
        params['use_rep'] = 'X'

    suffix = _graph_suffix(graph_key)
    return {
        'connectivities_key': graph_key,
        'distances_key': (source.get('distances_key')
                          or (f'{suffix}_distances' if suffix else 'distances')),
        'params': params,
    }


def _umap_call(adata, *, min_dist, spread, n_components, meta, key_added):
    """``sc.tl.umap``, with a synthesized neighbors entry when one is needed."""
    kwargs = {'min_dist': min_dist, 'spread': spread, 'n_components': n_components}
    # key_added='X_umap' would move the params to uns['X_umap']; omitting it
    # keeps scanpy's own uns['umap'], exactly as before this parameter existed.
    if key_added != 'X_umap':
        kwargs['key_added'] = key_added

    if meta is None:
        sc.tl.umap(adata, **kwargs)
        return

    adata.uns[_GRAPH_META_KEY] = meta
    try:
        sc.tl.umap(adata, neighbors_key=_GRAPH_META_KEY, **kwargs)
    finally:
        # A failed run must not leave a fabricated neighbors entry behind for
        # the next tool that reads uns.
        adata.uns.pop(_GRAPH_META_KEY, None)
```

- [ ] **Step 4: Add `_require_graph` to `DataAdaptor`**

Directly above `run_umap`:

```python
    def _require_graph(self, graph_key: str) -> None:
        """Validate an obsp connectivity graph before any work starts."""
        if graph_key not in self.adata.obsp:
            available = [k for k in self.adata.obsp
                         if k == 'connectivities' or k.endswith(_CONN_SUFFIX)]
            raise ValueError(
                f"Graph '{graph_key}' not found in obsp. "
                f"Available: {available if available else 'none'}"
            )
        n = self.n_cells
        shape = tuple(self.adata.obsp[graph_key].shape)
        if shape != (n, n):
            raise ValueError(
                f"Graph '{graph_key}' has shape {shape}, expected ({n}, {n})."
            )
```

- [ ] **Step 5: Rewrite `run_umap`**

Replace the whole method (~lines 5657–5714):

```python
    def run_umap(
        self,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        graph_key: str | None = None,
        key_added: str | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compute a UMAP embedding.

        Args:
            min_dist: Minimum distance between points.
            spread: Spread of the embedding.
            n_components: Number of dimensions.
            graph_key: obsp connectivity graph to embed. None uses whatever
                ``uns['neighbors']`` points at — the expression kNN, and the
                behaviour before this parameter existed. Any other graph
                (spatial, or a Combine Neighbors result) reaches scanpy through
                a synthesized entry; see :func:`_umap_neighbors_meta`.
            key_added: obsm key for the result. None or blank derives it from
                the graph: ``X_umap`` for the default, ``X_umap_<prefix>``
                otherwise, so embeddings from different graphs coexist.
            active_cell_indices: If provided, compute UMAP on these cells only;
                inactive cells get NaN coordinates.

        Returns:
            Dict with status, embedding_name, n_components, graph_key.
        """
        if graph_key:
            # A named graph carries its own prerequisite. check_prerequisites
            # looks for obsp['connectivities'], which a spatial-only dataset
            # does not have — and that is the case this parameter exists for.
            self._require_graph(graph_key)
        else:
            prereq = self.check_prerequisites('umap')
            if not prereq['satisfied']:
                raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        name = (key_added or '').strip() or _default_output_name('X_umap', graph_key)
        meta = _umap_neighbors_meta(self.adata, graph_key) if graph_key else None

        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            import anndata as ad
            adata_sub = ad.AnnData(
                obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            if 'X_pca' in self.adata.obsm:
                adata_sub.obsm['X_pca'] = self.adata.obsm['X_pca'][cell_indices]

            # Slice the graph this run actually uses. Hardcoding
            # 'connectivities' here sliced the wrong matrix for any other
            # choice, and nothing at all on a spatial-only dataset.
            if meta is not None:
                wanted = (meta['connectivities_key'], meta['distances_key'])
            else:
                wanted = ('connectivities', 'distances')
            for key in wanted:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if meta is None and 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sub_meta = dict(meta) if meta is not None else None
            if sub_meta is not None and sub_meta['params'].get('use_rep') == 'X':
                # adata_sub deliberately has no .X — only the obsm block — so
                # 'X' is not a readable representation here. X_pca is the only
                # thing a subset run can offer, and _validate_cell_indices
                # callers all have it.
                sub_meta['params'] = {**sub_meta['params'], 'use_rep': 'X_pca'}

            _umap_call(adata_sub, min_dist=min_dist, spread=spread,
                       n_components=n_components, meta=sub_meta, key_added='X_umap')

            full_umap = np.full((self.n_cells, n_components), np.nan)
            full_umap[cell_indices] = adata_sub.obsm['X_umap']
            self.adata.obsm[name] = full_umap
        else:
            _umap_call(self.adata, min_dist=min_dist, spread=spread,
                       n_components=n_components, meta=meta, key_added=name)

        result = {
            'status': 'completed',
            'embedding_name': name,
            'n_components': n_components,
            'graph_key': graph_key,
        }
        # Only record what differs from the default, so existing recordings and
        # their exported notebooks stay byte-identical.
        params: dict[str, Any] = {
            'min_dist': min_dist, 'spread': spread, 'n_components': n_components,
        }
        if graph_key:
            params['graph_key'] = graph_key
        if name != 'X_umap':
            params['key_added'] = name
        self._log_action('umap', params, result, subset=cell_indices)
        return result
```

- [ ] **Step 6: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q
```

Expected: all pass. If `test_umap_on_a_selection_slices_the_chosen_graph` fails because a spatial subset graph is too sparse for UMAP to embed 20 cells, raise `n_neighs` in that test's `run_spatial_neighbors` call — do **not** loosen the assertion.

- [ ] **Step 7: Run the whole backend suite**

```bash
cd backend && pixi run -e dev pytest -q
```

Expected: no regressions. `run_umap`'s signature gained keyword-only-ish params with defaults, so existing callers are unaffected.

- [ ] **Step 8: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_graph_choice.py
git commit -m "feat(umap): embed any obsp connectivity graph, not just the default"
```

---

### Task 3: Leiden on any graph

**Files:**
- Modify: `backend/xcell/adaptor.py` — `run_leiden` (~line 5716)
- Test: `backend/tests/test_graph_choice.py`

**Interfaces:**
- Consumes: `_default_output_name`, `DataAdaptor._require_graph`.
- Produces: `run_leiden(resolution=1.0, key_added='leiden', graph_key=None, active_cell_indices=None) -> dict` with `graph_key` in the result.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_graph_choice.py`:

```python
# --- Leiden ---------------------------------------------------------------

def test_leiden_on_spatial_graph_writes_its_own_column():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_leiden(resolution=0.5)
    a.run_spatial_neighbors(n_neighs=6)

    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities")

    assert res["key_added"] == "leiden_spatial"
    assert res["graph_key"] == "spatial_connectivities"
    assert "leiden_spatial" in a.adata.obs
    assert "leiden" in a.adata.obs           # the expression clustering survives
    assert res["n_clusters"] >= 2


def test_leiden_on_spatial_graph_without_pca_or_expression_knn():
    a = _spatial_only()
    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities")
    assert res["n_clusters"] >= 2
    assert "connectivities" not in a.adata.obsp


def test_leiden_explicit_name_wins_over_the_derived_one():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6)
    res = a.run_leiden(resolution=0.5, key_added="domains",
                       graph_key="spatial_connectivities")
    assert res["key_added"] == "domains"
    assert "domains" in a.adata.obs


def test_leiden_default_path_is_unchanged():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_leiden(resolution=0.5)

    assert res["key_added"] == "leiden"
    assert a._action_history[-1]["params"] == {"resolution": 0.5, "key_added": "leiden"}


def test_leiden_rejects_an_unknown_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    with pytest.raises(ValueError):
        a.run_leiden(graph_key="nope_connectivities")


def test_leiden_on_a_selection_slices_the_chosen_graph():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6)
    active = list(range(20))
    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities",
                       active_cell_indices=active)
    labels = a.adata.obs[res["key_added"]]
    assert set(labels.iloc[20:]) == {"unassigned"}
    assert "unassigned" not in set(labels.iloc[:20])
```

**Note on the derived default:** `run_leiden`'s `key_added` already defaults to the string `'leiden'`, not `None`, so "was it explicitly set?" cannot be read from the value alone. The implementation below derives the name only when `key_added` is still exactly `'leiden'`, which is what makes `test_leiden_explicit_name_wins_over_the_derived_one` meaningful. Do not change the parameter's default to `None` — the routes and the recorded params both rely on it.

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q -k leiden
```

Expected: `TypeError: run_leiden() got an unexpected keyword argument 'graph_key'`.

- [ ] **Step 3: Rewrite `run_leiden`**

Replace the method (~lines 5716–5778):

```python
    def run_leiden(
        self,
        resolution: float = 1.0,
        key_added: str = 'leiden',
        graph_key: str | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run Leiden clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters).
            key_added: obs column for the labels. Left at its default, a
                non-default graph derives its own name (``leiden_spatial``) so
                clusterings from different graphs coexist.
            graph_key: obsp connectivity graph to cluster. None uses
                ``obsp['connectivities']`` via ``uns['neighbors']``, as before.
            active_cell_indices: If provided, cluster only these cells;
                inactive cells are labeled 'unassigned'.

        Returns:
            Dict with status, key_added, n_clusters, resolution, graph_key.
        """
        if graph_key:
            self._require_graph(graph_key)
        else:
            prereq = self.check_prerequisites('leiden')
            if not prereq['satisfied']:
                raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # key_added defaults to a string rather than None, so an explicit
        # 'leiden' is indistinguishable from an unset one — treat the default
        # value as unset and let the graph name the column.
        name = (_default_output_name('leiden', graph_key)
                if key_added == 'leiden' else key_added)
        # sc.tl.leiden takes the adjacency directly; unlike UMAP it needs no
        # synthesized uns entry. obsp and neighbors_key are mutually exclusive,
        # so only one is ever passed.
        graph_kwargs = {'obsp': graph_key} if graph_key else {}

        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            import anndata as ad
            adata_sub = ad.AnnData(
                obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))

            wanted = ((graph_key,) if graph_key else ('connectivities', 'distances'))
            for key in wanted:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if not graph_key and 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sc.tl.leiden(adata_sub, resolution=resolution, key_added=name,
                         **graph_kwargs)

            sub_categories = list(adata_sub.obs[name].cat.categories)
            all_categories = sub_categories + ['unassigned']
            full_labels = ['unassigned'] * self.n_cells
            for i, idx in enumerate(cell_indices):
                full_labels[idx] = str(adata_sub.obs[name].iloc[i])
            self.adata.obs[name] = pd.Categorical(
                full_labels, categories=all_categories,
            )
            n_clusters = len(sub_categories)
        else:
            sc.tl.leiden(self.adata, resolution=resolution, key_added=name,
                         **graph_kwargs)
            n_clusters = len(self.adata.obs[name].cat.categories)

        result = {
            'status': 'completed',
            'key_added': name,
            'n_clusters': n_clusters,
            'resolution': resolution,
            'graph_key': graph_key,
        }
        params: dict[str, Any] = {'resolution': resolution, 'key_added': name}
        if graph_key:
            params['graph_key'] = graph_key
        self._log_action('leiden', params, result, subset=cell_indices)
        return result
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice.py -q
```

Expected: all pass. If Leiden warns about its `flavor` default, leave it — the default path must keep calling scanpy exactly as it did before.

- [ ] **Step 5: Run the whole backend suite**

```bash
cd backend && pixi run -e dev pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_graph_choice.py
git commit -m "feat(leiden): cluster any obsp connectivity graph"
```

---

### Task 4: Routes

**Files:**
- Modify: `backend/xcell/api/routes.py` — `UmapRequest` (~line 1678), `LeidenRequest` (~line 1685), `run_umap` route (~line 2216), `run_leiden` route (follows it)
- Test: `backend/tests/test_graph_choice_routes.py` (create)

**Interfaces:**
- Consumes: `run_umap` / `run_leiden` from Tasks 2–3.
- Produces: `POST /api/scanpy/umap` accepting `graph_key`, `key_added`; `POST /api/scanpy/leiden` accepting `graph_key`; `GET /api/scanpy/neighbor_graphs` entries carrying `suffix`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_graph_choice_routes.py`:

```python
"""Routes for choosing the connectivity graph UMAP and Leiden run on."""

import numpy as np
import anndata
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

client = TestClient(app)


def _adata():
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = rng.poisson(1.0, (n, 8)).astype(np.float32)
    X[coords[:, 0] <= 2, :4] += 10.0
    X[coords[:, 0] > 2, 4:] += 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.obsm["spatial"] = coords
    return ad


def _install():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6)
    routes.set_adaptor(a, slot="primary")
    return a


def test_neighbor_graphs_lists_a_suffix():
    _install()
    body = client.get("/api/scanpy/neighbor_graphs").json()
    graphs = {g["key"]: g for g in body["graphs"]}
    assert graphs["spatial_connectivities"]["suffix"] == "spatial"
    assert graphs["connectivities"]["suffix"] == ""


def test_umap_route_accepts_a_graph_key():
    a = _install()
    r = client.post("/api/scanpy/umap", json={"graph_key": "spatial_connectivities"})
    assert r.status_code == 200, r.text
    assert r.json()["embedding_name"] == "X_umap_spatial"
    assert "X_umap_spatial" in a.adata.obsm


def test_umap_route_accepts_an_explicit_name():
    _install()
    r = client.post("/api/scanpy/umap",
                    json={"graph_key": "spatial_connectivities", "key_added": "X_domains"})
    assert r.json()["embedding_name"] == "X_domains"


def test_umap_route_rejects_an_unknown_graph_with_400():
    _install()
    r = client.post("/api/scanpy/umap", json={"graph_key": "nope_connectivities"})
    assert r.status_code == 400
    assert "nope_connectivities" in r.json()["detail"]


def test_leiden_route_accepts_a_graph_key():
    a = _install()
    r = client.post("/api/scanpy/leiden",
                    json={"resolution": 0.5, "graph_key": "spatial_connectivities"})
    assert r.status_code == 200, r.text
    assert r.json()["key_added"] == "leiden_spatial"
    assert "leiden_spatial" in a.adata.obs


def test_umap_route_default_body_still_works():
    _install()
    r = client.post("/api/scanpy/umap", json={})
    assert r.status_code == 200
    assert r.json()["embedding_name"] == "X_umap"
```

Before running, confirm the import path and the installer helper by looking at an existing route test — `rg -n "from xcell.main import app|set_adaptor" backend/tests/*.py | head`. If the suite uses `monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)` instead of `set_adaptor`, follow that; the lambda **must** accept `dataset=None`.

- [ ] **Step 2: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice_routes.py -q
```

Expected: `graph_key` is silently dropped by pydantic, so `test_umap_route_accepts_a_graph_key` fails on `embedding_name == 'X_umap'`.

- [ ] **Step 3: Extend the request models**

```python
class UmapRequest(BaseModel):
    min_dist: float = 0.5
    spread: float = 1.0
    n_components: int = 2
    graph_key: str | None = None
    key_added: str | None = None
    active_cell_indices: list[int] | None = None


class LeidenRequest(BaseModel):
    resolution: float = 1.0
    key_added: str = 'leiden'
    graph_key: str | None = None
    active_cell_indices: list[int] | None = None
```

- [ ] **Step 4: Pass them through**

In the `/scanpy/umap` route body:

```python
        return adaptor.run_umap(
            min_dist=request.min_dist,
            spread=request.spread,
            n_components=request.n_components,
            graph_key=request.graph_key,
            key_added=request.key_added,
            active_cell_indices=request.active_cell_indices,
        )
```

And in `/scanpy/leiden`, add `graph_key=request.graph_key,` to the existing call.

- [ ] **Step 5: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_graph_choice_routes.py -q
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_graph_choice_routes.py
git commit -m "feat(api): graph_key on the umap and leiden routes"
```

---

### Task 5: Codegen

The exported notebook runs standalone, so `neighbors_key='_xcell_graph'` cannot be emitted alone — the entry does not exist there. Emit the synthesis too, and fidelity stays `exact`.

**Files:**
- Modify: `backend/xcell/codegen.py` — `umap` and `leiden` entries in `REGISTRY` (~lines 351–366)
- Test: `backend/tests/test_codegen.py`

**Interfaces:**
- Consumes: recorded params from Tasks 2–3 (`graph_key`, `key_added`).
- Produces: no new names; `REGISTRY['umap'].code` becomes a named builder `_code_umap`, `REGISTRY['leiden'].code` becomes `_code_leiden`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_codegen.py`:

```python
# --- graph choice ---------------------------------------------------------

def test_umap_on_the_default_graph_emits_one_unchanged_line():
    step = _step('umap', {'min_dist': 0.5, 'spread': 1.0, 'n_components': 2})
    t = translate(step)
    assert t.fidelity == EXACT
    assert t.code == [
        'sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2)'
    ]


def test_umap_on_a_named_graph_emits_the_neighbors_entry_it_needs():
    step = _step('umap', {
        'min_dist': 0.5, 'spread': 1.0, 'n_components': 2,
        'graph_key': 'spatial_connectivities', 'key_added': 'X_umap_spatial',
    })
    t = translate(step)
    assert t.fidelity == EXACT
    code = '\n'.join(t.code)
    # Emitting only the call would fail in the notebook: the entry it names
    # does not exist there.
    assert "adata.uns['_xcell_graph']" in code
    assert "'connectivities_key': 'spatial_connectivities'" in code
    assert "'method': 'umap'" in code
    assert "'distances_key'" in code
    assert "neighbors_key='_xcell_graph'" in code
    assert "key_added='X_umap_spatial'" in code


def test_umap_with_only_a_custom_name_stays_a_one_liner():
    step = _step('umap', {'min_dist': 0.5, 'spread': 1.0,
                          'n_components': 2, 'key_added': 'X_umap_v2'})
    t = translate(step)
    assert t.code == [
        "sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2, "
        "key_added='X_umap_v2')"
    ]


def test_leiden_on_a_named_graph_passes_obsp():
    step = _step('leiden', {'resolution': 0.5, 'key_added': 'leiden_spatial',
                            'graph_key': 'spatial_connectivities'})
    t = translate(step)
    assert t.fidelity == EXACT
    assert t.code == [
        "sc.tl.leiden(adata, resolution=0.5, key_added='leiden_spatial', "
        "obsp='spatial_connectivities')"
    ]


def test_leiden_on_the_default_graph_is_unchanged():
    step = _step('leiden', {'resolution': 0.5, 'key_added': 'leiden'})
    assert translate(step).code == [
        "sc.tl.leiden(adata, resolution=0.5, key_added='leiden')"
    ]
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_codegen.py -q -k "graph or umap or leiden"
```

Expected: the two named-graph tests fail — `graph_key` is not in the `only` tuple, so it is dropped.

- [ ] **Step 3: Implement the builders**

In `backend/xcell/codegen.py`, above `REGISTRY`. Import the shared constant so the emitted key cannot drift from the adaptor's:

```python
from xcell.adaptor import _GRAPH_META_KEY, _graph_suffix
```

If that import turns out to be circular (`adaptor` importing `codegen`), check with `rg -n "^from|^import" backend/xcell/codegen.py backend/xcell/adaptor.py | rg "codegen|adaptor"`. If it is circular, redefine `_GRAPH_META_KEY = '_xcell_graph'` in `codegen.py` with a comment naming `adaptor.py` as the source of truth, and add a test asserting the two are equal.

```python
def _code_umap(step: Step) -> list[str]:
    """UMAP, plus the neighbors entry a non-default graph needs.

    The adaptor builds this entry, hands it to scanpy, and deletes it. A
    notebook has to build it too, or `neighbors_key` names nothing — so the
    emitted lines are what really ran, and the tier stays `exact`.
    """
    p = step.params
    graph_key = p.get('graph_key')
    if not graph_key:
        return [_call('sc.tl.umap', p, ('min_dist', 'spread', 'n_components',
                                        'key_added'))]

    suffix = _graph_suffix(graph_key)
    meta = {
        'connectivities_key': graph_key,
        'distances_key': f'{suffix}_distances' if suffix else 'distances',
        'params': {'method': 'umap'},
    }
    call = _call('sc.tl.umap', p,
                 ('min_dist', 'spread', 'n_components', 'key_added'))
    return [
        f'{ADATA}.uns[{_lit(_GRAPH_META_KEY)}] = {_lit(meta)}',
        call[:-1] + f', neighbors_key={_lit(_GRAPH_META_KEY)})',
    ]


def _code_leiden(step: Step) -> list[str]:
    p = step.params
    call = _call('sc.tl.leiden', p, ('resolution', 'key_added'))
    graph_key = p.get('graph_key')
    if not graph_key:
        return [call]
    return [call[:-1] + f', obsp={_lit(graph_key)})']
```

Then point the registry entries at them and mention the graph in the prose:

```python
    'umap': ActionSpec(
        label='UMAP', fidelity=EXACT, imports=SCANPY,
        code=_code_umap,
        summary=lambda p, r: (
            f"UMAP embedding in {_n(p.get('n_components', 2))} dimensions "
            f"(min_dist {p.get('min_dist')}, spread {p.get('spread')})"
            + (f" over `{p['graph_key']}`" if p.get('graph_key') else '')
            + f" → `.obsm['{r.get('embedding_name', 'X_umap')}']`."
        ),
    ),
    'leiden': ActionSpec(
        label='Leiden clustering', fidelity=EXACT, imports=SCANPY,
        code=_code_leiden,
        summary=lambda p, r: (
            f"Leiden clustering at resolution {p.get('resolution')}"
            + (f" over `{p['graph_key']}`" if p.get('graph_key') else '')
            + f" → {_n(r.get('n_clusters'))} clusters in "
            f"`.obs['{p.get('key_added', 'leiden')}']`."
        ),
    ),
```

`_lit` renders the nested dict as a Python literal already — check its dict branch handles nesting before relying on it; the test asserting `'connectivities_key': 'spatial_connectivities'` appears in the output is what catches it if not.

- [ ] **Step 4: Run the codegen tests**

```bash
cd backend && pixi run -e dev pytest tests/test_codegen.py -q
```

Expected: all pass, including the pre-existing `test_every_logged_action_has_a_registry_entry`.

- [ ] **Step 5: Verify the emitted code actually runs**

This is the point of the fidelity claim, so check it rather than assume it:

```bash
cd backend && pixi run -e dev python -c "
import warnings; warnings.filterwarnings('ignore')
import numpy as np, scanpy as sc, anndata as ad, squidpy as sq
from scipy.sparse import csr_matrix
rng = np.random.default_rng(0)
adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0,(200,60)).astype('float32')))
adata.var_names=[f'g{i}' for i in range(60)]
adata.obsm['spatial']=rng.random((200,2))*100
sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
# --- paste the two emitted lines here, verbatim ---
adata.uns['_xcell_graph'] = {'connectivities_key': 'spatial_connectivities', 'distances_key': 'spatial_distances', 'params': {'method': 'umap'}}
sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2, key_added='X_umap_spatial', neighbors_key='_xcell_graph')
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_spatial', obsp='spatial_connectivities')
print('ran clean:', adata.obsm['X_umap_spatial'].shape, adata.obs['leiden_spatial'].nunique(), 'clusters')
print('no phantom pca:', 'X_pca' not in adata.obsm)
"
```

Expected: `ran clean: (200, 2) N clusters` and `no phantom pca: True`.

Note the emitted `params` has no `use_rep`, so this notebook path relies on `n_vars` (60) being above scanpy's `N_PCS` **and** hits `_get_pca_or_small_x`. If `no phantom pca` prints `False`, add `'use_rep': 'X'` to the emitted `meta` in `_code_umap` when the recorded step has no PCA — and add a codegen test asserting it.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/codegen.py backend/tests/test_codegen.py
git commit -m "feat(record): export UMAP/Leiden runs that used a non-default graph"
```

---

### Task 6: The picker in the UI

**Files:**
- Modify: `frontend/src/components/ScanpyModal.tsx`
  - `ParamDef` type (~line 150) — add `emptyLabel`
  - `umap` and `leiden` param lists (~lines 299–317)
  - `needsGraphs` allowlist (~line 883)
  - `graph_select` renderer (~line 2169)
  - result-message handling (~line 1348) so the banner reports the real output name

**Interfaces:**
- Consumes: `/api/scanpy/neighbor_graphs` entries with `suffix` (Task 1/4); `graph_key` + `key_added` on both routes (Task 4).
- Produces: no exported names.

- [ ] **Step 1: Add `emptyLabel` to the param type and the renderer**

In the `ParamDef` interface, add:

```ts
  emptyLabel?: string   // graph_select: label for the "" option. Absent => required.
```

In the `graph_select` branch (~line 2169), replace the `— pick a graph —` option so it honours the flag:

```tsx
                              <option value="">
                                {param.emptyLabel ?? '— pick a graph —'}
                              </option>
```

Smooth keeps the old text because it passes no `emptyLabel`; UMAP and Leiden pass one, because `''` is a real choice there.

- [ ] **Step 2: Add the params**

```ts
      umap: {
        label: 'UMAP',
        description: 'Compute UMAP embedding',
        prerequisites: ['neighbors'],
        params: [
          { name: 'graph_key', label: 'kNN graph', type: 'graph_select', default: '', emptyLabel: 'Expression neighbors (default)', description: 'Which cell-cell graph to embed. Spatial or a Combine Neighbors result gives a different map of the same cells.' },
          { name: 'key_added', label: 'Embedding name', type: 'text', default: '', description: 'obsm key for the result. Blank uses the name derived from the graph, so embeddings from different graphs coexist.' },
          { name: 'min_dist', label: 'Min distance', type: 'number', default: 0.5, description: 'Minimum distance between points' },
          { name: 'spread', label: 'Spread', type: 'number', default: 1.0, description: 'Spread of embedding' },
          { name: 'n_components', label: 'Dimensions', type: 'number', default: 2, description: 'Number of dimensions' },
        ],
      },
      leiden: {
        label: 'Leiden',
        description: 'Leiden clustering algorithm',
        prerequisites: ['neighbors'],
        params: [
          { name: 'graph_key', label: 'kNN graph', type: 'graph_select', default: '', emptyLabel: 'Expression neighbors (default)', description: 'Which cell-cell graph to cluster. Clustering the spatial graph finds spatial domains rather than cell types.' },
          { name: 'resolution', label: 'Resolution', type: 'number', default: 0.5, description: 'Higher = more clusters' },
          { name: 'key_added', label: 'Column name', type: 'text', default: 'leiden', description: 'Name for cluster labels' },
        ],
      },
```

**`prerequisites: ['neighbors']` stays as-is.** It gates the Run button on the expression kNN, which a spatial-only dataset lacks — so on such a dataset the tab would refuse to run despite the backend now allowing it. Fixing the gate is Step 5; leave the array alone here so the two changes stay reviewable apart.

- [ ] **Step 3: Populate the dropdown on those tabs**

```ts
    const needsGraphs = ['smooth', 'umap', 'leiden'].includes(selectedFunction)
```

- [ ] **Step 4: Prefill the name from the chosen graph**

Add near the other `useEffect`s (after the graph-loading effect, ~line 897). It re-prefills only while the field still holds the value this effect last wrote, so a name the user typed is never clobbered:

```tsx
  // Name the result after the graph it came from, unless the user has typed
  // their own. autoNameRef holds what we last filled in; if the field still
  // matches, it is ours to replace.
  const autoNameRef = useRef<string | null>(null)
  useEffect(() => {
    if (selectedFunction !== 'umap' && selectedFunction !== 'leiden') return
    const base = selectedFunction === 'umap' ? 'X_umap' : 'leiden'
    const field = selectedFunction === 'umap' ? 'key_added' : 'key_added'
    const graphKey = (paramValues.graph_key as string) || ''
    const suffix = availableGraphs.find((g) => g.key === graphKey)?.suffix ?? ''
    const derived = suffix ? `${base}_${suffix}` : (selectedFunction === 'umap' ? '' : base)

    const current = (paramValues[field] as string) ?? ''
    if (current === '' || current === autoNameRef.current) {
      autoNameRef.current = derived
      if (current !== derived) handleParamChange(field, derived)
    }
  }, [selectedFunction, paramValues.graph_key, availableGraphs])
```

UMAP's derived default for the *default* graph is the empty string, not `'X_umap'` — blank already means "derive", and showing a placeholder the backend would produce anyway keeps the field honest. Leiden's is `'leiden'`, matching its existing default.

Add `suffix: string` to the `NeighborGraphInfo` type (~line 784), and `useRef` to the React import.

- [ ] **Step 5: Let a spatial-only dataset run**

The Run button is gated on `prerequisites`, which for both tabs is `['neighbors']`. When a graph is picked explicitly, that prerequisite is the wrong question. Find where prerequisites are evaluated (`rg -n "prerequisites" frontend/src/components/ScanpyModal.tsx`) and treat them as satisfied when `paramValues.graph_key` is a non-empty string, for `umap` and `leiden` only.

- [ ] **Step 6: Report the real output name**

The result banner assumes `X_umap`. Make it read the response: use `data.embedding_name` for UMAP and `data.key_added` for Leiden, both of which the adaptor now returns. Check the existing message-building block near line 1348 and follow its shape.

- [ ] **Step 7: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "feat(ui): pick the kNN graph for UMAP and Leiden"
```

---

### Task 7: Browser verification and docs

Both recent features in this repo shipped bugs that passed pytest and died in a real run. This task is not optional.

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Start both servers**

```bash
pixi run backend     # :8000
pixi run dev         # :5173
```

Note: `uvicorn --reload` spawns a child. To stop it later use `pkill -f "uvicorn xcell.main:app"` — `lsof -ti :8000 | xargs kill` leaves a reloader that re-binds and serves stale data.

Editing any backend file restarts the server and resets the loaded dataset, so batch backend fixes and re-run the setup once.

- [ ] **Step 2: Drive the spatial path**

Load a spatial dataset (the bundled toy has coordinates). Then:

1. Analyze → Spatial → Spatial Neighbors.
2. Analyze → Cells → Leiden. The kNN graph dropdown should list "Spatial neighbors". Pick it; the column name should become `leiden_spatial`. Run.
3. Confirm `leiden_spatial` appears in the Cell Manager column list and colours the plot in spatial domains — contiguous patches, not the salt-and-pepper of an expression clustering.
4. Analyze → Cells → UMAP with the same graph. Confirm `X_umap_spatial` appears in the embedding picker **and that `X_umap` is still there and still different**.

- [ ] **Step 3: Confirm no phantom PCA**

The failure this feature's design turns on. On a dataset where you have **not** run PCA, run UMAP on the spatial graph, then check the schema:

```bash
curl -s localhost:8000/api/schema | python3 -m json.tool | grep -A5 embeddings
```

Expected: `X_umap_spatial` present, `X_pca` **absent**. If `X_pca` appears, `use_rep='X'` is not reaching `_choose_representation` — go back to `_umap_neighbors_meta`.

- [ ] **Step 4: Export the record and run it**

File → Analysis Record → export the notebook. Execute it. Both the UMAP and Leiden cells must run clean and produce the same column and embedding names. This is what makes the `exact` fidelity claim true.

- [ ] **Step 5: Stop the servers**

```bash
pkill -f "uvicorn xcell.main:app"; pkill -f "vite"
```

- [ ] **Step 6: Update the docs**

`README.md` — in whatever section covers UMAP/Leiden, note that both take a kNN graph and that clustering the spatial graph finds spatial domains.

`CHANGELOG.md` — a new entry following the existing format, covering: the graph picker on UMAP and Leiden; outputs named after the graph so they coexist; the two scanpy metadata gaps that had to be repaired; and the silent-PCA side effect that `use_rep='X'` avoids.

- [ ] **Step 7: Commit and merge**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: kNN graph choice for UMAP and Leiden"
git checkout main
git merge --no-ff feat/graph-choice
```

---

## Self-Review

**Spec coverage.** Metadata synthesis → Task 2. Leiden via `obsp` → Task 3. Suffix naming in one place → Task 1 (+ consumed in 2, 3, 6). Prerequisite skip → Task 2 backend, Task 6 step 5 frontend. Subset paths → Tasks 2 and 3. Routes → Task 4. Codegen → Task 5. Frontend incl. `emptyLabel` and name prefill → Task 6. Tests → in every task. Browser verification → Task 7.

**Deviations from the spec**, both stated at the top and both discovered by running the real library: blank `key_added` derives rather than errors; `use_rep='X'` guards against scanpy computing an unrecorded PCA. The `use_rep` finding also added Task 7 Step 3, which the spec did not anticipate.

**Names used consistently:** `_graph_suffix`, `_default_output_name`, `_GRAPH_META_KEY`, `_umap_neighbors_meta`, `_umap_call`, `_require_graph`, `_CONN_SUFFIX` — each defined in Task 1 or 2 and referenced with the same spelling afterwards. `suffix` is the field name on both the backend graph entry and the frontend `NeighborGraphInfo`.

**Known soft spots**, each with an in-plan instruction rather than a silent assumption: `run_spatial_neighbors`'s real signature (Task 1 Step 2), the route-test installer idiom (Task 4 Step 1), a possible `codegen → adaptor` import cycle (Task 5 Step 3), and whether the emitted notebook needs `use_rep` too (Task 5 Step 5).
