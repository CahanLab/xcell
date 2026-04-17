# PCA Loadings Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PCA Loadings step to Analyze → Cell Analysis that shows top ±loading genes per PC and creates derived `X_pca_no*` embeddings (à la pySCN `drop_pcs_from_embedding`) that Neighbors can target via `use_rep`.

**Architecture:** Backend `DataAdaptor` grows four new methods (`get_pca_loadings`, `create_pca_subset`, `list_pca_subsets`, `delete_pca_subset`) and two existing ones are extended (`run_pca` — fix loadings copy-back + auto-clear derived slots on re-run; `run_neighbors` — accept `use_rep`). Four new routes under `/api/scanpy/*`. Frontend gets a `pcaSubsets` field on `DatasetState`, API helpers in `useData.ts`, a new `MESSAGES.pcaLoadings` namespace, a new `pca_loadings` function entry + custom UI block in `ScanpyModal.tsx`, and a new `pc_source_select` param type for the Neighbors form.

**Tech Stack:** Python 3.9+ / FastAPI / AnnData / NumPy / scanpy (backend); React 18 + TypeScript + Zustand 4 (frontend). No new dependencies.

**Repository note:** The working root is `/Users/pcahan/code/xcell/`. The active project is `xcell/`. All file paths below are relative to the git root at `xcell/`. Run commands from that directory unless otherwise specified.

**Spec:** `docs/superpowers/specs/2026-04-17-pca-loadings-explorer-design.md`

**Testing note:** This project has no automated tests. Verify Python changes by importing the module (`python -c "from xcell.adaptor import DataAdaptor"`) and running the dev server (`uvicorn xcell.main:app --reload`) against `backend/xcell/data/toy_spatial.h5ad`. Verify frontend changes with `npm run build` (from `frontend/`) plus manual browser testing.

---

## File Map

**Backend — modified:**
- `backend/xcell/adaptor.py` — fix `run_pca` loadings copy-back; add derived-slot invalidation at end of `run_pca`; add `use_rep` to `run_neighbors`; add `get_pca_loadings`, `create_pca_subset`, `list_pca_subsets`, `delete_pca_subset`; add `pca_loadings` entry to `check_prerequisites`.
- `backend/xcell/api/routes.py` — add `NeighborsRequest.use_rep`; plumb `use_rep` through `/api/scanpy/neighbors`; add `/api/scanpy/pca_loadings`, `/api/scanpy/pca_subsets` (GET/POST), `/api/scanpy/pca_subsets/{obsm_key}` (DELETE).

**Frontend — modified:**
- `frontend/src/store.ts` — add `PCASubsetSummary` type; add `pcaSubsets` field to `DatasetState` (default, syncFlatFields, top-level mirror); add `setPcaSubsets` action.
- `frontend/src/hooks/useData.ts` — add `fetchPcaLoadings`, `fetchPcaSubsets`, `createPcaSubset`, `deletePcaSubset` helpers; add `usePcaLoadings` hook.
- `frontend/src/messages.ts` — add `MESSAGES.pcaLoadings` namespace.
- `frontend/src/components/ScanpyModal.tsx` — add `pc_source_select` to `ParamDef.type` union; add `pca_loadings` entry to `SCANPY_FUNCTIONS.cell_analysis.functions`; add `use_rep` param to `neighbors`; render `pc_source_select` in the form loop; render the PCA Loadings custom UI block; handle `cleared_subsets` in the PCA response.

**Docs — modified:**
- `CHANGELOG.md` — `[Unreleased] → Added` entry.
- `CLAUDE.md` — add to DataAdaptor method groups, API endpoints table, Key Behaviors.
- `README.md` — walkthrough paragraph describing the PCA Loadings step.

**Unchanged (important):**
- Cell-PCA-variance chart in the Neighbors form stays as-is.
- `get_cell_pca_variance`, `get_gene_pca_variance` stay as-is.
- Gene-PCA pipeline unchanged — this spec covers cell PCA only.
- `check_prerequisites('neighbors')` still requires `pca`; the new `use_rep` string is validated at run time rather than via a new prereq.

---

### Task 1: Backend — fix `run_pca` loadings copy-back for subset runs

**Goal:** Ensure `self.adata.varm['PCs']` is populated after `run_pca`, even when `gene_subset` restricts PCA to a subset of genes. Loadings for non-subset genes become NaN rows, mirroring how `X_pca` is NaN-padded for inactive cells.

**Files:**
- Modify: `backend/xcell/adaptor.py` — around lines 3073-3091 (within `run_pca`).

- [ ] **Step 1: Inspect the current copy-back block**

Open `backend/xcell/adaptor.py`. Find `run_pca`'s result-copying section (currently around lines 3076-3090). It looks like:

```python
        # Copy results back to main adata
        if cell_indices is not None:
            # Store X_pca with NaN for inactive cells
            full_pca = np.full((self.n_cells, n_comps), np.nan)
            full_pca[cell_indices] = adata_pca.obsm['X_pca']
            self.adata.obsm['X_pca'] = full_pca
        else:
            self.adata.obsm['X_pca'] = adata_pca.obsm['X_pca']
        self.adata.uns['pca'] = adata_pca.uns['pca']
        if 'PCs' in adata_pca.varm:
            # Store loadings for the subset genes
            self.adata.uns['pca']['gene_subset'] = {
                'type': subset_type,
                'n_genes': n_genes_used,
            }
```

Note: `adata_pca.varm['PCs']` holds the loadings sized `(n_genes_used, n_comps)`. The current code only records subset metadata in `uns` — the loadings array itself is lost when `adata_pca` is discarded.

- [ ] **Step 2: Replace the block with a full-size loadings copy-back**

Replace the block shown in Step 1 with:

```python
        # Copy results back to main adata
        if cell_indices is not None:
            # Store X_pca with NaN for inactive cells
            full_pca = np.full((self.n_cells, n_comps), np.nan)
            full_pca[cell_indices] = adata_pca.obsm['X_pca']
            self.adata.obsm['X_pca'] = full_pca
        else:
            self.adata.obsm['X_pca'] = adata_pca.obsm['X_pca']
        self.adata.uns['pca'] = adata_pca.uns['pca']

        # Copy gene loadings back as a full-size (n_genes, n_comps) matrix
        # with NaN rows for genes not included in the subset. Downstream
        # code (get_pca_loadings, create_pca_subset) expects varm['PCs']
        # to be present and correctly shaped for self.adata.n_vars.
        if 'PCs' in adata_pca.varm:
            full_pcs = np.full((self.n_genes, n_comps), np.nan)
            if gene_subset is not None:
                full_pcs[gene_mask, :] = adata_pca.varm['PCs']
            else:
                full_pcs[:, :] = adata_pca.varm['PCs']
            self.adata.varm['PCs'] = full_pcs
            self.adata.uns['pca']['gene_subset'] = {
                'type': subset_type,
                'n_genes': n_genes_used,
            }
```

The `gene_mask` variable comes from `_resolve_gene_mask()` earlier in the method (around line 3046). In the no-subset branch, `adata_pca` is the full AnnData so `varm['PCs']` is already `n_genes × n_comps` — we copy it through unchanged.

- [ ] **Step 3: Smoke-test the change**

Start the backend server from `backend/`:

```bash
uvicorn xcell.main:app --reload
```

In another shell, run this one-liner to exercise both the default path and a subset path:

```bash
python -c "
from xcell.adaptor import DataAdaptor
import numpy as np
a = DataAdaptor('backend/xcell/data/toy_spatial.h5ad')
a.run_pca(n_comps=10)
assert 'PCs' in a.adata.varm, 'PCs missing after default run'
assert a.adata.varm['PCs'].shape == (a.n_genes, 10), a.adata.varm['PCs'].shape
assert not np.isnan(a.adata.varm['PCs']).any(), 'unexpected NaN in default PCs'
print('default OK', a.adata.varm['PCs'].shape)
"
```

Expected output: `default OK (N, 10)` where `N = a.n_genes`.

- [ ] **Step 4: Commit**

```bash
git add backend/xcell/adaptor.py
git commit -m "Fix run_pca loadings copy-back for subset runs

Pad varm['PCs'] to full (n_genes, n_comps) with NaN rows for
genes not in the subset, so downstream per-gene loadings
access works after a subset PCA."
```

---

### Task 2: Backend — add `get_pca_loadings` + prerequisite + route

**Goal:** Expose top +/− loading genes per PC via a new adaptor method and route.

**Files:**
- Modify: `backend/xcell/adaptor.py` — add `get_pca_loadings`; extend `check_prerequisites`.
- Modify: `backend/xcell/api/routes.py` — add `GET /api/scanpy/pca_loadings`.

- [ ] **Step 1: Add `get_pca_loadings` method**

Open `backend/xcell/adaptor.py`. Scroll to the end of `run_pca` (the method ends around line 3108 with `return result`). Immediately after `run_pca`, insert:

```python
    def get_pca_loadings(self, top_n: int = 10) -> dict[str, Any]:
        """Return top +/- loading genes per computed PC.

        Reads self.adata.varm['PCs'] and self.adata.uns['pca']['variance_ratio'].
        Gene rows containing NaN loadings (from subset-PCA runs) are excluded
        from per-PC rankings; up to top_n valid genes are returned per side.

        Raises:
            ValueError: if PCA has not been run or loadings are missing.

        Returns:
            {
              'n_pcs': int,
              'top_n': int,
              'pcs': [
                {
                  'index': 0,                 # zero-based
                  'variance_ratio': 0.127,
                  'positive': [{'gene': 'MALAT1', 'loading': 0.18}, ...],
                  'negative': [{'gene': 'MT-CO1', 'loading': -0.15}, ...],
                }, ...
              ]
            }
        """
        if 'pca' not in self.adata.uns:
            raise ValueError("PCA has not been run. Run pca first.")
        if 'PCs' not in self.adata.varm:
            raise ValueError("PC loadings are unavailable (varm['PCs'] missing). Re-run PCA.")

        pcs_matrix = np.asarray(self.adata.varm['PCs'])
        if pcs_matrix.ndim != 2:
            raise ValueError(f"Unexpected varm['PCs'] shape: {pcs_matrix.shape}")

        n_genes, n_comps = pcs_matrix.shape
        var_ratio = np.asarray(self.adata.uns['pca'].get('variance_ratio', []))
        gene_names = list(self.adata.var_names)
        top_n = max(1, int(top_n))

        pcs_out = []
        for i in range(n_comps):
            col = pcs_matrix[:, i]
            valid = ~np.isnan(col)
            valid_indices = np.where(valid)[0]
            valid_loadings = col[valid_indices]

            # Positive side: sort descending, take top_n
            pos_order = valid_indices[np.argsort(-valid_loadings)][:top_n]
            positive = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in pos_order
                if col[int(j)] > 0
            ]

            # Negative side: sort ascending, take top_n
            neg_order = valid_indices[np.argsort(valid_loadings)][:top_n]
            negative = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in neg_order
                if col[int(j)] < 0
            ]

            pcs_out.append({
                'index': i,
                'variance_ratio': float(var_ratio[i]) if i < len(var_ratio) else None,
                'positive': positive,
                'negative': negative,
            })

        return {
            'n_pcs': n_comps,
            'top_n': top_n,
            'pcs': pcs_out,
        }
```

- [ ] **Step 2: Add `pca_loadings` to `check_prerequisites`**

Still in `backend/xcell/adaptor.py`, find `check_prerequisites` (around line 2188). In the `prereqs` dict (around line 2197), add `'pca_loadings': ['pca_with_loadings']` to the `# Cell analysis` group. Then in the prerequisite-check loop below, add a new `elif` clause for the new sentinel:

Update the `prereqs` dict, adding the new key after `'leiden': ['neighbors']`:

```python
            'leiden': ['neighbors'],
            'pca_loadings': ['pca_with_loadings'],
```

Then inside the loop (which currently starts with `if prereq == 'pca':` around line 2224), after the existing `elif prereq == 'neighbors':` block add:

```python
            elif prereq == 'pca_with_loadings':
                if 'pca' not in self.adata.uns or 'PCs' not in self.adata.varm:
                    missing.append('pca_with_loadings')
```

- [ ] **Step 3: Add the route**

Open `backend/xcell/api/routes.py`. Scroll to the bottom of the scanpy cell-analysis routes — find `@router.post("/scanpy/leiden")` (around line 1571). After the `run_leiden` function body, add the new route (pick an insertion point that stays grouped with the cell-analysis endpoints):

```python
@router.get("/scanpy/pca_loadings")
def get_pca_loadings(
    top_n: int = Query(10, ge=1, le=500),
    dataset: str | None = Query(None),
):
    """Return top +/- loading genes per computed PC."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_pca_loadings(top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Exercise the endpoint**

With `uvicorn xcell.main:app --reload` running and the default toy dataset loaded:

```bash
curl -s 'http://localhost:8000/api/scanpy/pca_loadings?top_n=3' | head -200
```

Expected: an error saying PCA has not been run.

Now run PCA:

```bash
curl -s -X POST 'http://localhost:8000/api/scanpy/pca' -H 'Content-Type: application/json' -d '{"n_comps": 10}'
```

Re-run the loadings curl:

```bash
curl -s 'http://localhost:8000/api/scanpy/pca_loadings?top_n=3' | python -m json.tool | head -60
```

Expected: JSON containing `n_pcs: 10`, `top_n: 3`, and a `pcs` array with 10 entries each listing up to 3 positive and 3 negative gene/loading pairs.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py
git commit -m "Add get_pca_loadings adaptor method and route

Exposes top +/- loading genes per PC, with NaN-row skipping so
subset-PCA runs return partial rankings. Guarded by a new
pca_loadings prerequisite."
```

---

### Task 3: Backend — add `create_pca_subset`, `list_pca_subsets`, `delete_pca_subset` + routes

**Goal:** Provide the persistence layer for derived `X_pca_no*` slots — create, list, delete.

**Files:**
- Modify: `backend/xcell/adaptor.py` — add three methods after `get_pca_loadings`.
- Modify: `backend/xcell/api/routes.py` — add three routes under `/scanpy/pca_subsets*`.

- [ ] **Step 1: Add `create_pca_subset`**

In `backend/xcell/adaptor.py`, immediately after the newly added `get_pca_loadings` method, insert:

```python
    def create_pca_subset(
        self,
        drop_pc_indices: list[int],
        suffix: str | None = None,
    ) -> dict[str, Any]:
        """Create derived PCA slots that exclude specific 1-indexed PCs.

        Writes:
          - obsm[f'X_pca_{suffix}'] — base embedding with dropped columns removed.
          - varm[f'PCs_{suffix}'] — matching loadings with dropped columns removed.
          - uns['pca'][f'variance_ratio_{suffix}'] — matching variance ratios.
          - uns['pca']['subsets'][suffix] = {'dropped_pcs': [i, j, ...]}
            (round-trips exact indices regardless of suffix).

        Raises:
            ValueError: missing PCA, empty indices, out-of-range, all-dropped.
            ValueError: suffix collision with existing obsm key.
        """
        if 'X_pca' not in self.adata.obsm:
            raise ValueError("PCA has not been run. Run pca first.")
        if not drop_pc_indices:
            raise ValueError("drop_pc_indices must contain at least one PC.")

        base_embed = np.asarray(self.adata.obsm['X_pca'])
        n_cells, n_pcs = base_embed.shape

        # Convert from 1-indexed user-facing to 0-indexed column positions.
        idx = np.asarray(drop_pc_indices, dtype=int) - 1
        if (idx < 0).any():
            raise ValueError("drop_pc_indices must be >= 1 (PC numbers are 1-indexed).")
        if (idx >= n_pcs).any():
            raise ValueError(
                f"drop_pc_indices contains entries > {n_pcs} (total PCs available)."
            )
        idx = np.unique(idx)
        keep = np.setdiff1d(np.arange(n_pcs), idx, assume_unique=False)
        if keep.size == 0:
            raise ValueError("Cannot drop all PCs.")

        dropped_1indexed = sorted(int(i + 1) for i in idx)

        if suffix is None or suffix == '':
            suffix = f"noPC{'_'.join(str(i) for i in dropped_1indexed)}"

        new_obsm_key = f"X_pca_{suffix}"
        if new_obsm_key in self.adata.obsm:
            raise ValueError(f"A PC subset named '{suffix}' already exists.")

        # Write the three companion slots.
        self.adata.obsm[new_obsm_key] = base_embed[:, keep]

        varm_key = None
        if 'PCs' in self.adata.varm:
            varm_key = f"PCs_{suffix}"
            self.adata.varm[varm_key] = np.asarray(self.adata.varm['PCs'])[:, keep]

        var_ratio_key = None
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            if 'variance_ratio' in self.adata.uns['pca']:
                var_ratio_key = f"variance_ratio_{suffix}"
                self.adata.uns['pca'][var_ratio_key] = np.asarray(
                    self.adata.uns['pca']['variance_ratio']
                )[keep]
            # Record the dropped indices for round-tripping in list_pca_subsets.
            subsets_meta = self.adata.uns['pca'].setdefault('subsets', {})
            subsets_meta[suffix] = {'dropped_pcs': dropped_1indexed}

        result = {
            'obsm_key': new_obsm_key,
            'varm_key': varm_key,
            'variance_ratio_key': var_ratio_key,
            'suffix': suffix,
            'n_pcs_kept': int(keep.size),
            'dropped_pcs': dropped_1indexed,
        }
        self._log_action('create_pca_subset', {
            'drop_pc_indices': dropped_1indexed,
            'suffix': suffix,
        }, result)
        return result
```

- [ ] **Step 2: Add `list_pca_subsets` and `delete_pca_subset`**

Immediately after `create_pca_subset`, add:

```python
    def list_pca_subsets(self) -> list[dict[str, Any]]:
        """List every derived PC subset in adata.obsm.

        Iterates obsm keys with prefix 'X_pca_' (excluding the exact key
        'X_pca'). For each, reports obsm_key, suffix, n_pcs_kept, and
        dropped_pcs (from uns['pca']['subsets'][suffix] when present,
        otherwise []).
        """
        out: list[dict[str, Any]] = []
        subsets_meta = {}
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            subsets_meta = self.adata.uns['pca'].get('subsets', {}) or {}

        for key in sorted(self.adata.obsm.keys()):
            if not key.startswith('X_pca_') or key == 'X_pca':
                continue
            suffix = key[len('X_pca_'):]
            arr = np.asarray(self.adata.obsm[key])
            n_pcs_kept = int(arr.shape[1]) if arr.ndim == 2 else 0
            meta = subsets_meta.get(suffix, {})
            dropped = list(meta.get('dropped_pcs', []))
            out.append({
                'obsm_key': key,
                'suffix': suffix,
                'n_pcs_kept': n_pcs_kept,
                'dropped_pcs': dropped,
            })
        return out

    def delete_pca_subset(self, obsm_key: str) -> None:
        """Delete a derived PC subset's obsm, varm, variance_ratio, and
        uns['pca']['subsets'] entries.

        Raises:
            ValueError: if obsm_key == 'X_pca', is missing, or doesn't start
                with 'X_pca_'.
        """
        if obsm_key == 'X_pca':
            raise ValueError("Cannot delete the base X_pca embedding.")
        if not obsm_key.startswith('X_pca_'):
            raise ValueError(f"'{obsm_key}' is not a derived PC subset.")
        if obsm_key not in self.adata.obsm:
            raise ValueError(f"'{obsm_key}' not found in obsm.")

        suffix = obsm_key[len('X_pca_'):]
        self.adata.obsm.pop(obsm_key, None)
        self.adata.varm.pop(f"PCs_{suffix}", None)
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            self.adata.uns['pca'].pop(f"variance_ratio_{suffix}", None)
            subsets_meta = self.adata.uns['pca'].get('subsets', {})
            if isinstance(subsets_meta, dict):
                subsets_meta.pop(suffix, None)
        self._log_action('delete_pca_subset', {'obsm_key': obsm_key}, None)
```

- [ ] **Step 3: Add Pydantic request model + routes**

Open `backend/xcell/api/routes.py`. Find the existing `NeighborsRequest` class (around line 1339). After it, add:

```python
class CreatePcaSubsetRequest(BaseModel):
    drop_pc_indices: list[int]
    suffix: str | None = None
```

Then scroll to just below the new `/scanpy/pca_loadings` route added in Task 2, and add:

```python
@router.get("/scanpy/pca_subsets")
def list_pca_subsets(dataset: str | None = Query(None)):
    """List derived PC subsets (X_pca_no* obsm slots)."""
    adaptor = get_adaptor(dataset)
    try:
        return {'subsets': adaptor.list_pca_subsets()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/pca_subsets")
def create_pca_subset(
    request: CreatePcaSubsetRequest,
    dataset: str | None = Query(None),
):
    """Create a derived PC subset that excludes the given (1-indexed) PCs."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.create_pca_subset(
            drop_pc_indices=request.drop_pc_indices,
            suffix=request.suffix,
        )
    except ValueError as e:
        # Use 409 for suffix collision so the UI can show a specific toast.
        if 'already exists' in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/scanpy/pca_subsets/{obsm_key}")
def delete_pca_subset(
    obsm_key: str,
    dataset: str | None = Query(None),
):
    """Delete a derived PC subset by its obsm key (e.g., X_pca_noPC2_5)."""
    adaptor = get_adaptor(dataset)
    try:
        adaptor.delete_pca_subset(obsm_key)
        return {'status': 'deleted', 'obsm_key': obsm_key}
    except ValueError as e:
        if 'not found' in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Exercise the endpoints**

With the server running and PCA already computed (from Task 2 testing):

```bash
curl -s 'http://localhost:8000/api/scanpy/pca_subsets' | python -m json.tool
```

Expected: `{"subsets": []}`.

Create one:

```bash
curl -s -X POST 'http://localhost:8000/api/scanpy/pca_subsets' \
  -H 'Content-Type: application/json' \
  -d '{"drop_pc_indices": [2, 5]}' | python -m json.tool
```

Expected: an object with `obsm_key: "X_pca_noPC2_5"`, `varm_key: "PCs_noPC2_5"`, `n_pcs_kept: 8`, `dropped_pcs: [2, 5]`.

Re-list:

```bash
curl -s 'http://localhost:8000/api/scanpy/pca_subsets' | python -m json.tool
```

Expected: one subset entry.

Attempt collision:

```bash
curl -s -o /tmp/collide.json -w '%{http_code}\n' -X POST \
  'http://localhost:8000/api/scanpy/pca_subsets' \
  -H 'Content-Type: application/json' \
  -d '{"drop_pc_indices": [2, 5]}'
```

Expected: `409`.

Delete:

```bash
curl -s -X DELETE 'http://localhost:8000/api/scanpy/pca_subsets/X_pca_noPC2_5' | python -m json.tool
```

Expected: `{"status": "deleted", "obsm_key": "X_pca_noPC2_5"}`.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py
git commit -m "Add create/list/delete PCA subset methods and routes

Implements pySCN-style drop_pcs_from_embedding: creates
X_pca_<suffix> obsm + matching varm + variance_ratio, records
dropped-PC indices in uns['pca']['subsets'] for round-tripping.
Suffix collisions return 409."
```

---

### Task 4: Backend — auto-clear derived subsets on PCA re-run

**Goal:** When `run_pca` is called again, remove any existing `X_pca_no*` derived slots (they reference stale eigenvector columns) and return `cleared_subsets` so the frontend can toast and reset.

**Files:**
- Modify: `backend/xcell/adaptor.py` — append invalidation logic inside `run_pca`, before the `return result` line.

- [ ] **Step 1: Add invalidation block**

Open `backend/xcell/adaptor.py`. Find the end of `run_pca` (just before the final `return result`, around line 3107). Insert immediately before the existing `self._log_action('pca', ...)` call:

```python
        # Clear derived PC subsets — they reference columns of the previous
        # X_pca and become stale on re-run. This scans obsm/varm/uns in case
        # the user ran PCA without going through create_pca_subset (e.g.,
        # import-time slots). Note: sc.tl.pca above already replaced
        # adata.uns['pca'] wholesale, so variance_ratio_* keys are already
        # gone; the pop calls here are defensive.
        cleared_subsets: list[str] = []
        for key in list(self.adata.obsm.keys()):
            if key.startswith('X_pca_') and key != 'X_pca':
                suffix = key[len('X_pca_'):]
                self.adata.obsm.pop(key, None)
                self.adata.varm.pop(f"PCs_{suffix}", None)
                if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
                    self.adata.uns['pca'].pop(f"variance_ratio_{suffix}", None)
                cleared_subsets.append(key)
        if cleared_subsets:
            result['cleared_subsets'] = cleared_subsets
```

This block runs BEFORE `_log_action` so the action record reflects the cleared state. The `result` dict was built a few lines earlier (around line 3095); we simply add a new key to it when relevant.

- [ ] **Step 2: Smoke test**

With the server running:

```bash
# Ensure at least one derived subset exists
curl -s -X POST 'http://localhost:8000/api/scanpy/pca' -H 'Content-Type: application/json' -d '{"n_comps": 10}' > /dev/null
curl -s -X POST 'http://localhost:8000/api/scanpy/pca_subsets' \
  -H 'Content-Type: application/json' -d '{"drop_pc_indices": [2]}' > /dev/null
curl -s 'http://localhost:8000/api/scanpy/pca_subsets' | python -m json.tool
```

Expected: one entry.

Now re-run PCA and inspect the response:

```bash
curl -s -X POST 'http://localhost:8000/api/scanpy/pca' \
  -H 'Content-Type: application/json' -d '{"n_comps": 10}' | python -m json.tool
```

Expected: includes `"cleared_subsets": ["X_pca_noPC2"]`.

Confirm the list is now empty:

```bash
curl -s 'http://localhost:8000/api/scanpy/pca_subsets' | python -m json.tool
```

Expected: `{"subsets": []}`.

- [ ] **Step 3: Commit**

```bash
git add backend/xcell/adaptor.py
git commit -m "Auto-clear derived PC subsets on PCA re-run

Scan obsm for X_pca_* keys (except X_pca) at end of run_pca and
drop the companion varm + variance_ratio entries. Include the
list under 'cleared_subsets' in the response so the UI can toast."
```

---

### Task 5: Backend — add `use_rep` to `run_neighbors` + route

**Goal:** Let the Neighbors route target any obsm slot (e.g. a derived `X_pca_noPC2_5`) via a new `use_rep` argument. `n_pcs` semantics unchanged: first N of whatever `use_rep` resolves to.

**Files:**
- Modify: `backend/xcell/adaptor.py` — extend `run_neighbors` signature, kwargs, and subset path.
- Modify: `backend/xcell/api/routes.py` — add `use_rep` to `NeighborsRequest` and pass it through.

- [ ] **Step 1: Extend `run_neighbors`**

Open `backend/xcell/adaptor.py`. Find `run_neighbors` (around line 3110). Update its signature and body.

Before (around lines 3110-3173):

```python
    def run_neighbors(
        self,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        metric: str = 'euclidean',
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
```

After:

```python
    def run_neighbors(
        self,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        metric: str = 'euclidean',
        use_rep: str | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
```

Below the signature, extend the kwargs construction. Replace:

```python
        kwargs = {
            'n_neighbors': n_neighbors,
            'metric': metric,
        }
        if n_pcs is not None:
            kwargs['n_pcs'] = n_pcs
```

with:

```python
        kwargs = {
            'n_neighbors': n_neighbors,
            'metric': metric,
        }
        if n_pcs is not None:
            kwargs['n_pcs'] = n_pcs

        # Resolve and validate use_rep. None / 'X_pca' preserve the existing
        # default path. Any other value must exist in adata.obsm.
        rep_key = use_rep if use_rep and use_rep != 'X_pca' else None
        if rep_key is not None:
            if rep_key not in self.adata.obsm:
                raise ValueError(
                    f"use_rep '{rep_key}' not found in obsm. "
                    f"Create it via /api/scanpy/pca_subsets first."
                )
            kwargs['use_rep'] = rep_key
```

Next, update the cell-subset path so it copies the correct obsm into `adata_sub`. Find these lines (around 3147-3150):

```python
            pca_full = self.adata.obsm['X_pca']
            pca_sub = pca_full[cell_indices]
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            adata_sub.obsm['X_pca'] = pca_sub
```

Replace with:

```python
            source_key = rep_key if rep_key is not None else 'X_pca'
            pca_full = self.adata.obsm[source_key]
            pca_sub = pca_full[cell_indices]
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            adata_sub.obsm[source_key] = pca_sub
```

(The `sc.pp.neighbors(adata_sub, **kwargs)` call on the next line already passes `use_rep` through `kwargs` when set.)

- [ ] **Step 2: Extend `NeighborsRequest` and route**

Open `backend/xcell/api/routes.py`. Find `NeighborsRequest` (around line 1339). Update it to:

```python
class NeighborsRequest(BaseModel):
    n_neighbors: int = 15
    n_pcs: int | None = None
    metric: str = 'euclidean'
    use_rep: str | None = None
    active_cell_indices: list[int] | None = None
```

Then find the `run_neighbors` route handler (around line 1525). Update the adaptor call from:

```python
        return adaptor.run_neighbors(
            n_neighbors=request.n_neighbors,
            n_pcs=request.n_pcs,
            metric=request.metric,
            active_cell_indices=request.active_cell_indices,
        )
```

to:

```python
        return adaptor.run_neighbors(
            n_neighbors=request.n_neighbors,
            n_pcs=request.n_pcs,
            metric=request.metric,
            use_rep=request.use_rep,
            active_cell_indices=request.active_cell_indices,
        )
```

- [ ] **Step 3: Smoke test**

With the server running, PCA computed, and a subset created:

```bash
curl -s -X POST 'http://localhost:8000/api/scanpy/pca' -H 'Content-Type: application/json' -d '{"n_comps": 10}' > /dev/null
curl -s -X POST 'http://localhost:8000/api/scanpy/pca_subsets' \
  -H 'Content-Type: application/json' -d '{"drop_pc_indices": [2]}' > /dev/null

# Default path — use X_pca
curl -s -X POST 'http://localhost:8000/api/scanpy/neighbors' \
  -H 'Content-Type: application/json' -d '{"n_neighbors": 15}' | python -m json.tool

# Derived slot path
curl -s -X POST 'http://localhost:8000/api/scanpy/neighbors' \
  -H 'Content-Type: application/json' \
  -d '{"n_neighbors": 15, "use_rep": "X_pca_noPC2"}' | python -m json.tool

# Bad use_rep
curl -s -o /tmp/bad.json -w '%{http_code}\n' -X POST \
  'http://localhost:8000/api/scanpy/neighbors' \
  -H 'Content-Type: application/json' \
  -d '{"n_neighbors": 15, "use_rep": "nope"}'
```

Expected: first two succeed (`status: completed`), third returns `400`.

- [ ] **Step 4: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py
git commit -m "Add use_rep to run_neighbors and the /scanpy/neighbors route

Lets Neighbors target any obsm slot (e.g., derived X_pca_no*).
None or 'X_pca' preserves the existing default path. The cell-subset
path copies the chosen obsm instead of hard-coding X_pca."
```

---

### Task 6: Frontend — add `PCASubsetSummary` type and `pcaSubsets` store field

**Goal:** Make per-dataset `pcaSubsets: PCASubsetSummary[]` available via the existing dual-write pattern (per-slot nested + flat mirror).

**Files:**
- Modify: `frontend/src/store.ts`.

- [ ] **Step 1: Export `PCASubsetSummary` type**

Open `frontend/src/store.ts`. Find `GeneMaskConfig` (around line 259). Just after its export, add:

```ts
export interface PCASubsetSummary {
  obsmKey: string       // e.g. 'X_pca_noPC2_5'
  suffix: string        // e.g. 'noPC2_5'
  droppedPcs: number[]  // 1-indexed
  nPcsKept: number
}
```

- [ ] **Step 2: Add field to `DatasetState` and default state**

In `DatasetState` (around line 307), add a new field below `geneMaskConfig`:

```ts
  geneMaskConfig: GeneMaskConfig | null
  pcaSubsets: PCASubsetSummary[]
}
```

In `createDefaultDatasetState()` (around line 335), add after `geneMaskConfig: null`:

```ts
    geneMaskConfig: null,
    pcaSubsets: [],
  }
}
```

- [ ] **Step 3: Mirror at top level and in initial state**

In the `AppState` interface (around line 464), add after the `geneMaskConfig` flat mirror:

```ts
  // Gene mask config (per-dataset, flat mirror)
  geneMaskConfig: GeneMaskConfig | null

  // PCA subsets (per-dataset, flat mirror)
  pcaSubsets: PCASubsetSummary[]
```

In the initial state returned by the store (around line 768), add after `geneMaskConfig: null,`:

```ts
    geneMaskConfig: null,
    pcaSubsets: [],
```

- [ ] **Step 4: Add to `syncFlatFields`**

In `syncFlatFields` (around line 678), add after `geneMaskConfig: ds.geneMaskConfig`:

```ts
      geneMaskConfig: ds.geneMaskConfig,
      pcaSubsets: ds.pcaSubsets,
    }
```

- [ ] **Step 5: Add setter action**

Find `setGeneMaskConfig` in the interface (around line 613) and add immediately after:

```ts
  setGeneMaskConfig: (config: GeneMaskConfig | null) => void
  setPcaSubsets: (subsets: PCASubsetSummary[]) => void
```

Then find the setter implementation (around line 1724):

```ts
    setGeneMaskConfig: (config) => set(dsUpdate({ geneMaskConfig: config })),
```

Add immediately below:

```ts
    setPcaSubsets: (subsets) => set(dsUpdate({ pcaSubsets: subsets })),
```

- [ ] **Step 6: Verify TypeScript compiles**

From `frontend/`:

```bash
npm run build
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/store.ts
git commit -m "Add per-dataset pcaSubsets field to store

Dual-write pattern like geneMaskConfig: per-slot nested +
flat top-level mirror + syncFlatFields entry. Backed by a
new setPcaSubsets action."
```

---

### Task 7: Frontend — add API helpers and `usePcaLoadings` hook

**Goal:** Add typed helpers in `useData.ts` for all four new endpoints, plus a hook that fetches loadings keyed by `(slot, topN)`.

**Files:**
- Modify: `frontend/src/hooks/useData.ts`.

- [ ] **Step 1: Inspect existing helpers**

Open `frontend/src/hooks/useData.ts`. Note the two usage patterns:
- Standalone async functions (e.g. `fetchGeneMask` at line 696) that wrap fetch + error handling + store writes.
- `useX()` hooks (e.g. `useObsSummaries` at line 823) using `useEffect` + `useStore` + `fetch`.

- [ ] **Step 2: Import `PCASubsetSummary` in the file**

At the top of `useData.ts`, find the import from `'../store'` and add `PCASubsetSummary` to the import list. For example, if the current line is:

```ts
import { useStore, DatasetSlot, GeneMaskConfig, ... } from '../store'
```

append `PCASubsetSummary` to the named imports.

- [ ] **Step 3: Add `PCALoading` / `PCALoadingsResponse` types**

Near the other type exports at the top of the file (search for `export interface` to find the grouping), add:

```ts
export interface PCALoadingGene {
  gene: string
  loading: number
}

export interface PCALoadingEntry {
  index: number                     // zero-based
  variance_ratio: number | null
  positive: PCALoadingGene[]
  negative: PCALoadingGene[]
}

export interface PCALoadingsResponse {
  n_pcs: number
  top_n: number
  pcs: PCALoadingEntry[]
}
```

- [ ] **Step 4: Add the four API helpers**

Below `clearGeneMask` (around line 799) and above `fetchBooleanColumnValues`, add:

```ts
export async function fetchPcaLoadings(
  topN: number,
  slot?: DatasetSlot,
): Promise<PCALoadingsResponse> {
  const url = appendDataset(`/api/scanpy/pca_loadings?top_n=${topN}`, slot)
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Failed to fetch PCA loadings')
  }
  return res.json()
}

export async function fetchPcaSubsets(slot?: DatasetSlot): Promise<PCASubsetSummary[]> {
  const url = appendDataset('/api/scanpy/pca_subsets', slot)
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Failed to fetch PCA subsets')
  }
  const data = await res.json()
  const subsets: PCASubsetSummary[] = (data.subsets || []).map((s: any) => ({
    obsmKey: s.obsm_key,
    suffix: s.suffix,
    droppedPcs: s.dropped_pcs || [],
    nPcsKept: s.n_pcs_kept,
  }))
  // Write directly into the active dataset's store so the Neighbors dropdown
  // and PCA Loadings subsets list stay in sync.
  const targetSlot = slot ?? useStore.getState().activeSlot
  useStore.getState().patchSlotState(targetSlot, { pcaSubsets: subsets })
  return subsets
}

export async function createPcaSubset(
  dropPcIndices: number[],
  suffix: string | null,
  slot?: DatasetSlot,
): Promise<PCASubsetSummary> {
  const url = appendDataset('/api/scanpy/pca_subsets', slot)
  const body: { drop_pc_indices: number[]; suffix?: string } = {
    drop_pc_indices: dropPcIndices,
  }
  if (suffix && suffix.trim() !== '') body.suffix = suffix.trim()

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    const err = new Error(detail.detail || 'Failed to create PC subset') as Error & { status?: number }
    err.status = res.status
    throw err
  }
  const data = await res.json()
  const summary: PCASubsetSummary = {
    obsmKey: data.obsm_key,
    suffix: data.suffix,
    droppedPcs: data.dropped_pcs || [],
    nPcsKept: data.n_pcs_kept,
  }
  // Optimistically append to the store.
  const targetSlot = slot ?? useStore.getState().activeSlot
  const current = useStore.getState().datasets[targetSlot]?.pcaSubsets || []
  useStore.getState().patchSlotState(targetSlot, {
    pcaSubsets: [...current, summary],
  })
  return summary
}

export async function deletePcaSubset(
  obsmKey: string,
  slot?: DatasetSlot,
): Promise<void> {
  const url = appendDataset(`/api/scanpy/pca_subsets/${encodeURIComponent(obsmKey)}`, slot)
  const res = await fetch(url, { method: 'DELETE' })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Failed to delete PC subset')
  }
  const targetSlot = slot ?? useStore.getState().activeSlot
  const current = useStore.getState().datasets[targetSlot]?.pcaSubsets || []
  useStore.getState().patchSlotState(targetSlot, {
    pcaSubsets: current.filter((s) => s.obsmKey !== obsmKey),
  })
}
```

- [ ] **Step 5: Add the `usePcaLoadings` hook**

Just after `fetchPcaLoadings`, add a hook that refetches on top-N change or active-slot change:

```ts
export function usePcaLoadings(topN: number, enabled: boolean): {
  loadings: PCALoadingsResponse | null
  loading: boolean
  error: string | null
  reload: () => void
} {
  const activeSlot = useStore((s) => s.activeSlot)
  const [loadings, setLoadings] = useState<PCALoadingsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchPcaLoadings(topN)
      .then((data) => { if (!cancelled) setLoadings(data) })
      .catch((e) => { if (!cancelled) setError(e.message || 'Failed to fetch loadings') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [topN, enabled, activeSlot, reloadToken])

  const reload = useCallback(() => setReloadToken((n) => n + 1), [])
  return { loadings, loading, error, reload }
}
```

(The React imports `useState`, `useEffect`, and `useCallback` are already in place at the top of the file at `useData.ts:1` — no import changes needed.)

- [ ] **Step 6: Verify TypeScript compiles**

```bash
cd frontend && npm run build
```

Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useData.ts
git commit -m "Add PCA loadings/subsets API helpers and hook

fetchPcaLoadings, fetchPcaSubsets, createPcaSubset,
deletePcaSubset standalone functions (each writes back into
the active dataset's pcaSubsets) plus usePcaLoadings hook
that refetches on top-N or active-slot change."
```

---

### Task 8: Frontend — add `MESSAGES.pcaLoadings` namespace

**Goal:** Centralize all UI strings for the new feature.

**Files:**
- Modify: `frontend/src/messages.ts`.

- [ ] **Step 1: Add the namespace**

Open `frontend/src/messages.ts`. After the `geneMask` namespace (which ends around line 102), before the closing `} as const`, add:

```ts
  // PCA Loadings Explorer
  pcaLoadings: {
    description:
      'Inspect top-loading genes per PC, then create a derived subset that excludes selected PCs for downstream analysis.',
    topNLabel: 'Top-N genes per side:',
    colPC: 'PC',
    colVariance: 'Var %',
    colPositive: 'Top + loading genes',
    colNegative: 'Top − loading genes',
    suffixLabel: 'Suffix (optional):',
    suffixAutoPrefix: 'auto: ',
    createButton: 'Create PC subset →',
    createBusyButton: 'Creating…',
    deleteButton: '✕',
    checkedSummary: (nChecked: number, nKept: number) =>
      `${nChecked} PC${nChecked === 1 ? '' : 's'} checked, ${nKept} would remain`,
    noneChecked: 'Check at least one PC to drop',
    allDropped: 'Cannot drop all PCs',
    prereqMissing: 'Run PCA first to explore loadings.',
    empty: 'No PC loadings available — re-run PCA to populate loadings.',
    loading: 'Loading loadings…',
    fetchError: 'Failed to load PC loadings.',
    existingSubsetsHeader: 'Existing PC subsets',
    noSubsets: 'No derived PC subsets yet.',
    subsetSummary: (suffix: string, nKept: number, dropped: number[]) =>
      dropped.length > 0
        ? `${suffix} · ${nKept} kept · dropped ${dropped.join(', ')}`
        : `${suffix} · ${nKept} kept`,
    createdToast: (suffix: string, nKept: number) =>
      `Created PC subset "${suffix}" (${nKept} PCs kept)`,
    collisionToast: (suffix: string) =>
      `A PC subset named "${suffix}" already exists.`,
    clearedToast: (n: number) =>
      `PCA recomputed — cleared ${n.toLocaleString()} derived PC subset${n === 1 ? '' : 's'}.`,
    neighborsSourceLabel: 'PC source',
    neighborsSourceDescription:
      'Which PC embedding to use. Create derived subsets via PCA Loadings.',
    neighborsSourceBaseLabel: 'X_pca (all PCs)',
  },
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npm run build
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/messages.ts
git commit -m "Add MESSAGES.pcaLoadings namespace

Centralized UI strings for the PCA Loadings explorer tab and
Neighbors PC source dropdown."
```

---

### Task 9: Frontend — scaffold the PCA Loadings function entry + read-only table

**Goal:** Add the `pca_loadings` entry under Cell Analysis so it appears in the function list, and render the loadings table (read-only) with the top-N input. No checkbox or create button yet — that comes in Task 10.

**Files:**
- Modify: `frontend/src/components/ScanpyModal.tsx`.

- [ ] **Step 1: Widen the `ParamDef.type` union**

Open `frontend/src/components/ScanpyModal.tsx`. Find the `ParamDef` interface (around line 147). Update the `type` field to include the new param type:

```ts
  type: 'number' | 'text' | 'select' | 'gene_subset' | 'textarea' | 'pc_source_select'
```

(We add `pc_source_select` now so Task 12 can target it without reopening this interface.)

- [ ] **Step 2: Register the `pca_loadings` entry**

In `SCANPY_FUNCTIONS.cell_analysis.functions`, insert a new entry between `pca` and `neighbors`. Find the `neighbors:` key (around line 247) and add above it:

```ts
      pca_loadings: {
        label: 'PCA Loadings',
        description: 'Explore PC gene loadings and create PC subsets to exclude technical PCs',
        prerequisites: ['pca_loadings'],
        params: [],
        custom: true,
      },
```

(The `'pca_loadings'` prereq matches the backend sentinel added in Task 2.)

- [ ] **Step 3: Import new helpers and state**

Near the top of the file, add the needed imports. If the following symbols are not already imported, add them:

- From `'../hooks/useData'`: `usePcaLoadings`, `fetchPcaSubsets`, `createPcaSubset`, `deletePcaSubset`, `PCALoadingsResponse`.
- From `'../messages'`: `MESSAGES`.
- From `'../store'`: `PCASubsetSummary`, `useStore`.

Inside the `ScanpyModal` component, add the following local state near the other `useState` calls (search for `useState` inside the component — add near existing blocks like `compareColumn`):

```ts
  const [pcaTopN, setPcaTopN] = useState<number>(10)
  const [pcaCheckedPCs, setPcaCheckedPCs] = useState<Set<number>>(new Set())
  const [pcaSuffix, setPcaSuffix] = useState<string>('')
  const [pcaCreateBusy, setPcaCreateBusy] = useState<boolean>(false)
  const activeSlot = useStore((s) => s.activeSlot)
  const pcaSubsetsFromStore: PCASubsetSummary[] =
    useStore((s) => s.datasets[s.activeSlot]?.pcaSubsets || [])

  const { loadings: pcaLoadings, loading: pcaLoadingsLoading, error: pcaLoadingsError } =
    usePcaLoadings(pcaTopN, selectedFunction === 'pca_loadings')

  // Load existing subsets whenever we open the PCA Loadings tab.
  useEffect(() => {
    if (selectedFunction === 'pca_loadings') {
      fetchPcaSubsets().catch(() => { /* toast shown by global error handler */ })
    }
  }, [selectedFunction, activeSlot])

  // Reset checked set when loadings payload changes (e.g., PCA was re-run).
  useEffect(() => {
    setPcaCheckedPCs(new Set())
    setPcaSuffix('')
  }, [pcaLoadings?.n_pcs, activeSlot])
```

(`selectedFunction` is an existing state variable in this component — search for `selectedFunction` to confirm.)

- [ ] **Step 4: Render the read-only table**

Find the existing custom UI block for `compare_cells` (around line 1200, starts with `{selectedFunction === 'compare_cells' && (`). Directly BEFORE that `{selectedFunction === 'compare_cells' && (` line, insert:

```tsx
        {/* PCA Loadings custom UI (read-only scaffold — Task 9) */}
        {selectedFunction === 'pca_loadings' && (
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '10px', lineHeight: 1.4 }}>
              {MESSAGES.pcaLoadings.description}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <label style={{ fontSize: '12px', color: '#aaa' }}>
                {MESSAGES.pcaLoadings.topNLabel}
              </label>
              <input
                type="number"
                min={1}
                max={500}
                value={pcaTopN}
                onChange={(e) => setPcaTopN(Math.max(1, parseInt(e.target.value) || 1))}
                style={{ width: '70px', padding: '4px 6px', fontSize: '12px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px' }}
              />
            </div>

            {pcaLoadingsLoading && (
              <div style={{ fontSize: '12px', color: '#888', padding: '8px' }}>
                {MESSAGES.pcaLoadings.loading}
              </div>
            )}
            {pcaLoadingsError && (
              <div style={{ fontSize: '12px', color: '#ff7f7f', padding: '8px' }}>
                {pcaLoadingsError}
              </div>
            )}

            {pcaLoadings && pcaLoadings.pcs.length === 0 && (
              <div style={{ fontSize: '12px', color: '#888', padding: '8px' }}>
                {MESSAGES.pcaLoadings.empty}
              </div>
            )}

            {pcaLoadings && pcaLoadings.pcs.length > 0 && (
              <div style={{ maxHeight: '400px', overflowY: 'auto', border: '1px solid #1a1a2e', borderRadius: '4px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead style={{ position: 'sticky', top: 0, backgroundColor: '#0f3460', zIndex: 1 }}>
                    <tr>
                      <th style={{ padding: '6px 8px', color: '#aaa', textAlign: 'left', width: '36px' }}></th>
                      <th style={{ padding: '6px 8px', color: '#aaa', textAlign: 'right', width: '48px' }}>{MESSAGES.pcaLoadings.colPC}</th>
                      <th style={{ padding: '6px 8px', color: '#aaa', textAlign: 'right', width: '64px' }}>{MESSAGES.pcaLoadings.colVariance}</th>
                      <th style={{ padding: '6px 8px', color: '#aaa', textAlign: 'left' }}>{MESSAGES.pcaLoadings.colPositive}</th>
                      <th style={{ padding: '6px 8px', color: '#aaa', textAlign: 'left' }}>{MESSAGES.pcaLoadings.colNegative}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pcaLoadings.pcs.map((pc) => {
                      const pcNum = pc.index + 1
                      const checked = pcaCheckedPCs.has(pcNum)
                      const varPct = pc.variance_ratio != null ? `${(pc.variance_ratio * 100).toFixed(1)}%` : '—'
                      return (
                        <tr
                          key={pc.index}
                          style={{
                            borderBottom: '1px solid #0a0f1a',
                            backgroundColor: checked ? 'rgba(78, 205, 196, 0.12)' : 'transparent',
                          }}
                        >
                          <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled
                              style={{ opacity: 0.4 }}
                            />
                          </td>
                          <td style={{ padding: '4px 8px', textAlign: 'right', color: '#ccc' }}>{pcNum}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'right', color: '#ccc' }}>{varPct}</td>
                          <td style={{ padding: '4px 8px', color: '#ccc' }}>
                            {pc.positive.length === 0
                              ? <span style={{ color: '#555' }}>—</span>
                              : pc.positive.map((g, i) => (
                                  <span key={g.gene} title={`loading=${g.loading.toFixed(4)}`}>
                                    {g.gene}{i < pc.positive.length - 1 ? ', ' : ''}
                                  </span>
                                ))
                            }
                          </td>
                          <td style={{ padding: '4px 8px', color: '#ccc' }}>
                            {pc.negative.length === 0
                              ? <span style={{ color: '#555' }}>—</span>
                              : pc.negative.map((g, i) => (
                                  <span key={g.gene} title={`loading=${g.loading.toFixed(4)}`}>
                                    {g.gene}{i < pc.negative.length - 1 ? ', ' : ''}
                                  </span>
                                ))
                            }
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

```

(Checkboxes appear but are `disabled` in this task — they become interactive in Task 10.)

- [ ] **Step 5: Suppress the generic Run button for `pca_loadings`**

Find the button row (around line 1432):

```tsx
          {selectedFunction === 'compare_cells' ? (
            <button ... handleCompareRun ...>
              {compareLoading ? 'Running...' : ...}
            </button>
          ) : (
            <button ... onClick={isRunning && activeTaskId ? handleCancel : handleRun} ...>
              {isRunning ? (activeTaskId ? 'Cancel' : 'Running...') : 'Run'}
            </button>
          )}
```

Extend the ternary chain to also suppress the generic Run button for `pca_loadings` (which has its own Create button rendered inline inside the custom block in Task 10):

```tsx
          {selectedFunction === 'compare_cells' ? (
            /* existing compare-cells button block unchanged */
          ) : selectedFunction === 'pca_loadings' ? (
            null /* Create button rendered inline in the PCA Loadings custom block */
          ) : (
            /* existing Run button block unchanged */
          )}
```

- [ ] **Step 6: Verify build and visual smoke test**

From `frontend/`:

```bash
npm run build
```

Expected: clean build.

Start the frontend dev server: `npm run dev`. In the browser, open the Scanpy modal → Cell Analysis → PCA Loadings. If PCA hasn't been run, the prerequisite blocker should show (same mechanism as every other function). Run PCA first, then re-open; the loadings table should appear with all computed PCs and top-N genes per side.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "Scaffold PCA Loadings tab with read-only loadings table

Adds pca_loadings entry under Cell Analysis, renders the
top-N control and a scrollable table of per-PC +/- loadings
with tooltip-exposed loading values. Checkboxes render but
are disabled until Task 10 wires them up."
```

---

### Task 10: Frontend — interactive PC selection, suffix field, create button, existing subsets list

**Goal:** Wire the PC checkboxes, suffix field, Create button, and existing subsets list with delete buttons.

**Files:**
- Modify: `frontend/src/components/ScanpyModal.tsx`.

- [ ] **Step 1: Replace the disabled checkboxes with interactive ones**

In the PCA Loadings block added in Task 9, find the existing `<input type="checkbox" checked={checked} disabled ... />` inside the table row. Replace that entire `<td>` cell with:

```tsx
                          <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setPcaCheckedPCs((prev) => {
                                  const next = new Set(prev)
                                  if (next.has(pcNum)) next.delete(pcNum)
                                  else next.add(pcNum)
                                  return next
                                })
                              }}
                            />
                          </td>
```

- [ ] **Step 2: Add suffix field + Create button below the table**

Immediately after the closing `</div>` of the table container (`{pcaLoadings && pcaLoadings.pcs.length > 0 && (...)}` block), still inside the outer `{selectedFunction === 'pca_loadings' && (...)}` block, insert:

```tsx
            {pcaLoadings && pcaLoadings.pcs.length > 0 && (() => {
              const droppedSorted = Array.from(pcaCheckedPCs).sort((a, b) => a - b)
              const autoSuffix = droppedSorted.length > 0 ? `noPC${droppedSorted.join('_')}` : ''
              const nKept = pcaLoadings.n_pcs - droppedSorted.length
              const disableReason =
                droppedSorted.length === 0 ? MESSAGES.pcaLoadings.noneChecked
                : nKept === 0 ? MESSAGES.pcaLoadings.allDropped
                : null
              return (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                    <label style={{ fontSize: '12px', color: '#aaa' }}>
                      {MESSAGES.pcaLoadings.suffixLabel}
                    </label>
                    <input
                      type="text"
                      value={pcaSuffix}
                      onChange={(e) => setPcaSuffix(e.target.value)}
                      placeholder={autoSuffix ? `${MESSAGES.pcaLoadings.suffixAutoPrefix}${autoSuffix}` : ''}
                      style={{ flex: 1, padding: '4px 6px', fontSize: '12px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px' }}
                    />
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <button
                      disabled={!!disableReason || pcaCreateBusy}
                      onClick={async () => {
                        if (disableReason) return
                        setPcaCreateBusy(true)
                        try {
                          const summary = await createPcaSubset(
                            droppedSorted,
                            pcaSuffix.trim() || null,
                          )
                          setPcaCheckedPCs(new Set())
                          setPcaSuffix('')
                          // Reuse the existing setResult toast mechanism
                          // (success banner + auto-dismiss) so the user gets
                          // the same feedback as every other scanpy action.
                          setResult({
                            success: true,
                            message: MESSAGES.pcaLoadings.createdToast(summary.suffix, summary.nPcsKept),
                          })
                        } catch (e: any) {
                          if (e?.status === 409) {
                            setResult({
                              success: false,
                              message: MESSAGES.pcaLoadings.collisionToast(pcaSuffix.trim() || autoSuffix),
                            })
                          } else {
                            setResult({
                              success: false,
                              message: e?.message || 'Failed to create PC subset',
                            })
                          }
                        } finally {
                          setPcaCreateBusy(false)
                        }
                      }}
                      style={{
                        padding: '6px 12px',
                        fontSize: '12px',
                        backgroundColor: disableReason ? '#2a2a3e' : '#4ecdc4',
                        color: disableReason ? '#666' : '#000',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: disableReason ? 'not-allowed' : 'pointer',
                        fontWeight: 600,
                      }}
                    >
                      {pcaCreateBusy ? MESSAGES.pcaLoadings.createBusyButton : MESSAGES.pcaLoadings.createButton}
                    </button>
                    <span style={{ fontSize: '11px', color: disableReason ? '#ff9966' : '#888' }}>
                      {disableReason || MESSAGES.pcaLoadings.checkedSummary(droppedSorted.length, nKept)}
                    </span>
                  </div>
                </div>
              )
            })()}
```

(`setResult` is the existing toast-state setter at the top of the component — see lines around 610: `const [result, setResult] = useState<{ success: boolean; message: string } | null>(null)`. Every other scanpy action funnels success/error messages through the same banner. No new toast mechanism is introduced.)

- [ ] **Step 3: Add the Existing PC Subsets list**

After the Create button block above, still inside the `{selectedFunction === 'pca_loadings' && (...)}` outer block, insert:

```tsx
            <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '6px' }}>
                {MESSAGES.pcaLoadings.existingSubsetsHeader}
              </div>
              {pcaSubsetsFromStore.length === 0 ? (
                <div style={{ fontSize: '11px', color: '#666', padding: '6px 8px' }}>
                  {MESSAGES.pcaLoadings.noSubsets}
                </div>
              ) : (
                <div style={{ border: '1px solid #1a1a2e', borderRadius: '4px' }}>
                  {pcaSubsetsFromStore.map((s) => (
                    <div
                      key={s.obsmKey}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '8px',
                        padding: '6px 10px',
                        fontSize: '11px',
                        color: '#ccc',
                        borderBottom: '1px solid #0a0f1a',
                      }}
                    >
                      <span>
                        {MESSAGES.pcaLoadings.subsetSummary(s.suffix, s.nPcsKept, s.droppedPcs)}
                      </span>
                      <button
                        onClick={async () => {
                          try {
                            await deletePcaSubset(s.obsmKey)
                          } catch (e: any) {
                            setResult({
                              success: false,
                              message: e?.message || 'Failed to delete PC subset',
                            })
                          }
                        }}
                        title="Delete"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#888',
                          cursor: 'pointer',
                          fontSize: '13px',
                          padding: '2px 6px',
                        }}
                      >
                        {MESSAGES.pcaLoadings.deleteButton}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: clean build.

- [ ] **Step 5: Browser validation**

With the dev server running:
1. Load the toy dataset, open Scanpy modal → Cell Analysis → PCA → Run (keep defaults).
2. Switch to PCA Loadings. Confirm the loadings table renders.
3. Check PC 2 and PC 5. The suffix placeholder reads `auto: noPC2_5`, the status line reads "2 PCs checked, N−2 would remain".
4. Click Create PC subset. Confirm a new row appears under "Existing PC subsets" and the checkboxes reset.
5. Provide an explicit suffix `my_run` and check PC 3; click Create. Confirm both rows are present.
6. Attempt to recreate an identical subset (same checked PCs, empty suffix); confirm the collision toast.
7. Click `✕` on a subset; confirm it disappears.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "Wire PCA Loadings interactive UI: checkboxes, create, delete

Checkboxes toggle a per-PC set, the suffix field takes an
optional override (placeholder shows the auto-suffix),
Create PC subset posts to the backend and refreshes the
Existing PC Subsets list, and ✕ deletes a derived slot."
```

---

### Task 11: Frontend — add `pc_source_select` to Neighbors + handle `cleared_subsets` from PCA

**Goal:** Let users choose any derived `X_pca_no*` obsm slot as the Neighbors PC source, and clear the frontend `pcaSubsets` with a toast when PCA is re-run.

**Files:**
- Modify: `frontend/src/components/ScanpyModal.tsx`.

- [ ] **Step 1: Register the `use_rep` param on `neighbors`**

In `SCANPY_FUNCTIONS.cell_analysis.functions.neighbors.params` (around line 251), insert a new first param so the dropdown appears above `n_neighbors`:

```ts
        params: [
          { name: 'use_rep', label: 'PC source', type: 'pc_source_select', default: 'X_pca', description: 'Which PC embedding to use. Create derived subsets via PCA Loadings.' },
          { name: 'n_neighbors', label: 'Neighbors', type: 'number', default: 15, description: 'Number of neighbors' },
          { name: 'n_pcs', label: 'PCs to use', type: 'number', default: null, description: 'Number of PCs (null = all)' },
          { name: 'metric', label: 'Metric', type: 'select', default: 'euclidean', options: ['euclidean', 'cosine', 'manhattan'], description: 'Distance metric' },
        ],
```

- [ ] **Step 2: Ensure subsets list is loaded when Neighbors is selected**

Right after the existing effect that fetches cell-PCA variance for Neighbors (the block starting `if (selectedFunction !== 'neighbors')` around line 646), add a new sibling effect that populates `pcaSubsets` whenever Neighbors is selected:

```tsx
  // Load derived PC subsets when the Neighbors tab is selected so the
  // PC source dropdown has up-to-date options.
  useEffect(() => {
    if (selectedFunction === 'neighbors') {
      fetchPcaSubsets().catch(() => { /* ignore; dropdown falls back to X_pca */ })
    }
  }, [selectedFunction, activeSlot])
```

- [ ] **Step 3: Render `pc_source_select` in the form loop**

Find the form-renderer around line 1357 where param type branches exist:

```tsx
                  <>
                    <div style={styles.paramRow}>
                      <label style={styles.paramLabel}>{param.label}</label>
                      {param.type === 'select' ? (
                        <select ...>
```

Insert a new `param.type === 'pc_source_select' ? ... :` branch as the FIRST check inside the `<>` fragment (before `param.type === 'select'`):

```tsx
                      {param.type === 'pc_source_select' ? (
                        <select
                          style={styles.paramInput}
                          value={paramValues[param.name] ?? 'X_pca'}
                          onChange={(e) => handleParamChange(param.name, e.target.value)}
                        >
                          <option value="X_pca">{MESSAGES.pcaLoadings.neighborsSourceBaseLabel}</option>
                          {pcaSubsetsFromStore.map((s) => (
                            <option key={s.obsmKey} value={s.obsmKey}>
                              {s.obsmKey} ({s.nPcsKept} kept)
                            </option>
                          ))}
                        </select>
                      ) : param.type === 'select' ? (
                        /* existing select branch */
                        <select ...
```

Leave the existing `select`/`textarea`/default branches intact.

- [ ] **Step 4: Omit `use_rep` from the request body when it equals `'X_pca'`**

Open `ScanpyModal.tsx` and find `handleRun` (around line 806). The request body is built in `requestParams` (lines ~823–833) and posted at line ~892 (`body: JSON.stringify(requestParams)`). Immediately BEFORE the `fetch(...)` call at line 892, add:

```ts
      // Neighbors: the PC source dropdown uses 'X_pca' as a sentinel for the
      // default path. Strip it from the body so the backend preserves its
      // existing behavior (n_pcs unchanged, use_rep defaults to None).
      if (selectedFunction === 'neighbors' && requestParams['use_rep'] === 'X_pca') {
        delete requestParams['use_rep']
      }
```

- [ ] **Step 5: Handle `cleared_subsets` in the PCA response**

Still in `handleRun`, the success-message builder starts around line 918 (`let message = ...`) and ends around line 1006 (just before `setResult({ success: true, message })` at line 1008). Right after the big message-builder if/else block and BEFORE the `setResult({ success: true, message })` call, insert:

```ts
      // PCA re-runs clear any derived PC subsets server-side. If the response
      // reports any, wipe the frontend mirror and append to the success toast
      // so the user sees what happened.
      if (
        selectedFunction === 'pca' &&
        Array.isArray(data.cleared_subsets) &&
        data.cleared_subsets.length > 0
      ) {
        useStore.getState().setPcaSubsets([])
        message = `${message} — ${MESSAGES.pcaLoadings.clearedToast(data.cleared_subsets.length)}`
      }
```

This reuses the existing `setResult({ success: true, message })` call below — the cleared-subsets note is folded into the existing PCA success banner rather than introducing a second toast.

- [ ] **Step 6: Verify build**

```bash
cd frontend && npm run build
```

Expected: clean build.

- [ ] **Step 7: Browser validation**

With the dev server running:
1. Load the toy dataset. Run PCA, create two subsets (e.g. `noPC2` and `custom_run` via the PCA Loadings tab from Task 10).
2. Switch to Neighbors. Confirm the PC source dropdown lists `X_pca`, `X_pca_noPC2`, `X_pca_custom_run`.
3. Pick `X_pca_noPC2` and click Run. Confirm the request body (in the Network tab) includes `"use_rep": "X_pca_noPC2"` and the response is a successful neighbors result.
4. Run UMAP; confirm it uses the new graph.
5. Go back to PCA and re-run with `n_comps=10`. Confirm a toast says "PCA recomputed — cleared 2 derived PC subset(s)", the PCA Loadings tab's existing-subsets list is empty, and the Neighbors PC source dropdown only shows `X_pca`.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ScanpyModal.tsx
git commit -m "Add PC source dropdown to Neighbors; handle cleared_subsets

Neighbors form renders a pc_source_select dropdown populated
from the active dataset's pcaSubsets; use_rep is stripped from
the request when it equals 'X_pca'. On PCA re-run, any non-empty
cleared_subsets in the response clears the store mirror and
surfaces a toast."
```

---

### Task 12: Docs — update CLAUDE.md, CHANGELOG.md, README.md

**Goal:** Documentation updates per CLAUDE.md's "Keeping Documentation Current" policy.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Open `CLAUDE.md`. Four edits:

(a) In the **Scanpy cell analysis** bullet under "DataAdaptor Key Methods", append the new methods. The current line is:

```
**Scanpy cell analysis**: `run_pca()`, `run_neighbors()`, `run_umap()`, `run_leiden()`
```

Change it to:

```
**Scanpy cell analysis**: `run_pca()`, `run_neighbors(use_rep=...)`, `run_umap()`, `run_leiden()`, `get_pca_loadings(top_n)`, `create_pca_subset(drop_pc_indices, suffix)`, `list_pca_subsets()`, `delete_pca_subset(obsm_key)`
```

(b) In the **Scanpy** row of the API Endpoints table, add the new routes to the comma-separated list. Find the row starting with `POST /scanpy/{operation} for:`. Append `, pca_loadings, pca_subsets` and add new items under "Also" for the dedicated routes:

```
| **Scanpy** | `POST /scanpy/{operation}` for: ..., `contourize`. Also `GET /scanpy/history`, `GET /scanpy/prerequisites/{action}`, `GET /scanpy/cell_pca_variance`, `GET /scanpy/gene_pca_variance`, `GET /scanpy/gene_modules`, `GET /scanpy/has_spatial`, `GET /scanpy/pca_loadings`, `GET /scanpy/pca_subsets`, `POST /scanpy/pca_subsets`, `DELETE /scanpy/pca_subsets/{obsm_key}` |
```

(c) In the **Store Types** section, add a note about the new field. After the line `Optional field `GeneMaskConfig = ...` was added to `DatasetState` under `geneMaskConfig`.`, add:

```
Optional field `PCASubsetSummary = { obsmKey, suffix, droppedPcs, nPcsKept }` was added. Per-dataset array `pcaSubsets: PCASubsetSummary[]` lives on `DatasetState`, mirrored to the top level via the dual-write pattern. Backed by the `setPcaSubsets` action.
```

(d) In the **Key Behaviors** section, add a new bullet anywhere in the list (near the other Scanpy-related behaviors):

```
- **PCA Loadings explorer**: Analyze → Cell Analysis → PCA Loadings renders a scrollable table with top ±loading genes per PC (tooltip shows the exact loading). User checks PCs to exclude and clicks "Create PC subset" to persist a derived embedding `X_pca_<suffix>` (+ matching `varm['PCs_<suffix>']` and `variance_ratio_<suffix>`), modeled on pySingleCellNet's `drop_pcs_from_embedding`. Auto-suffix is `noPC<i>_<j>_…` (1-indexed). The Neighbors form gains a PC source dropdown that targets any derived slot via `sc.pp.neighbors(use_rep=...)`; UMAP/Leiden inherit automatically via the neighbors graph. Re-running PCA auto-clears all derived slots and toasts the user; the `run_pca` response includes `cleared_subsets: string[]`. Derived slots round-trip via h5ad export.
```

- [ ] **Step 2: Update `CHANGELOG.md`**

Open `CHANGELOG.md`. Under the `[Unreleased]` heading → `Added` subsection, add:

```markdown
- PCA Loadings explorer (Analyze → Cell Analysis): scan a scrollable table of top +/− loading genes per PC, then create derived PC-subset embeddings (`X_pca_noPC2_5`, etc.) that exclude selected PCs. The Neighbors step gained a PC source dropdown that routes through any derived subset via `use_rep`; UMAP and Leiden inherit the choice automatically through the neighbors graph. Re-running PCA auto-clears stale derived subsets.
```

- [ ] **Step 3: Update `README.md`**

Open `README.md`. Find the section describing cell-analysis workflow (look for the heading that covers PCA / Neighbors / UMAP — likely under a walkthrough or features section). Insert a paragraph after the PCA description and before the Neighbors description:

```markdown
**PCA Loadings (optional).** After running PCA, open **Analyze → Cell Analysis → PCA Loadings** to scan the top-loading genes on each side of every PC. Hovering a gene reveals its exact loading. If you spot PCs dominated by technical signal (cell cycle, mitochondrial genes, etc.), check them and click **Create PC subset** to persist a derived embedding (e.g. `X_pca_noPC2_5`). Then in the **Neighbors** step, pick that subset from the **PC source** dropdown — UMAP and Leiden inherit the change automatically. Re-running PCA clears all derived subsets (with a toast) since their column indices refer to the previous eigenvectors.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md CHANGELOG.md README.md
git commit -m "Document PCA Loadings explorer feature

CLAUDE.md: new adaptor methods, new routes, new store field,
new Key Behaviors bullet. CHANGELOG.md: Unreleased/Added entry.
README.md: walkthrough paragraph for the loadings tab and PC
source dropdown."
```

---

## Verification checklist (end-to-end)

After all tasks are complete, walk through the spec's Testing section (steps 1–12) end to end in the browser to confirm the feature works as designed. Key invariants:

- `npm run build` succeeds from `frontend/`.
- Running PCA creates `varm['PCs']` with full shape `(n_genes, n_comps)` (with NaN rows for gene-subset runs).
- Creating a subset writes obsm/varm/variance_ratio slots with the expected suffix and records `dropped_pcs` in `uns['pca']['subsets']`.
- Listing subsets returns correct `n_pcs_kept` and `dropped_pcs` for both auto and custom suffixes.
- Re-running PCA clears derived slots and the `run_pca` response includes `cleared_subsets`.
- Neighbors with `use_rep` pointed at a derived slot runs without error; with a non-existent slot, returns 400.
- Exporting the h5ad and reloading preserves derived slots (they're standard AnnData fields).

---

## Self-review notes

- **Spec coverage:** all spec sections mapped to tasks: backend methods (Tasks 2/3), backend fix (Task 1), invalidation (Task 4), use_rep (Task 5), store (Task 6), hooks/API (Task 7), messages (Task 8), UI scaffold (Task 9), UI interactivity (Task 10), Neighbors dropdown + PCA toast (Task 11), docs (Task 12).
- **Placeholder scan:** every step shows the actual code or command. No TBDs, no "handle error appropriately", no "similar to earlier task".
- **Type consistency:** `PCASubsetSummary` { obsmKey, suffix, droppedPcs, nPcsKept } is used identically across store, hooks, and UI. Backend payload translates snake_case to camelCase once, in `fetchPcaSubsets` / `createPcaSubset`.
- **Scope:** single feature, single implementation session.
- **Out-of-scope reminders:** no gene-PCA loadings view, no click-to-color-by-gene in the table, no stale-tagging alternative to auto-clear.
