# PCA Loadings Explorer — Design

**Date:** 2026-04-17
**Author:** Patrick Cahan (brainstormed with Claude)
**Status:** Draft — awaiting user review before implementation plan

## Summary

Add a **PCA Loadings** step to Analyze → Cell Analysis (positioned between PCA and Neighbors) that (1) renders a scrollable table of all computed PCs with their top +/− loading genes, letting the user identify PCs dominated by technical signal (cell cycle, mito, ribo, etc.), and (2) lets the user check a subset of PCs and click **Create PC subset** to persist a derived embedding in `adata.obsm` (e.g. `X_pca_noPC2_5`) alongside matching `varm` loadings and `variance_ratio` — a direct port of [pySingleCellNet's `drop_pcs_from_embedding`](https://github.com/CahanLab/PySingleCellNet/blob/master/src/pySingleCellNet/utils/adataTools.py).

The Neighbors form gains a **PC source** dropdown listing `X_pca` and any derived `X_pca_no*` slots; selecting one routes the neighbors graph (and therefore UMAP and Leiden) through the chosen subset via `sc.pp.neighbors(use_rep=...)`.

## Goals

- Give users a scannable view of per-PC biology (top-N genes on each side of every PC).
- Let users exclude specific PCs from downstream cell analysis without re-running PCA, matching the pySCN workflow.
- Persist derived PC subsets as standard AnnData slots so they round-trip through h5ad export and can be reused across Neighbors runs.
- Keep composition with UMAP and Leiden automatic — those steps read the neighbors graph, so they inherit the PC selection without extra plumbing.

## Non-Goals

- Gene-PCA loadings view (this spec covers cell PCA only).
- Seurat-style `PCHeatmap` (top-loading genes × top-scoring cells heatmap).
- Clicking a gene name in the loadings table to color cells by its expression (v1.5 polish).
- Stale-tagging of derived subsets on PCA re-run — we auto-clear them (see Invalidation).
- Persisting user-selected PC-checkbox state across modal close/reopen (local React state only).
- Allowing downstream steps other than Neighbors to accept `use_rep` (UMAP and Leiden already compose through the neighbors graph; gene-PCA is a separate pipeline out of scope).

## User-Facing Behavior

### Entry point

Analyze → Cell Analysis gets a new function entry **PCA Loadings** (between `PCA` and `Neighbors`). Its prerequisite is `pca`; until PCA has run, the entry is disabled with a standard prerequisite tooltip.

### Modal layout (PCA Loadings tab)

```
┌──────────────────────────────────────────────────────────────────────┐
│  PCA Loadings                                                        │
│  Inspect top-loading genes per PC, then create a derived subset that │
│  excludes selected PCs for downstream analysis.                      │
│                                                                      │
│  Top-N genes per side: [ 10 ]                                        │
│                                                                      │
│  ┌────┬────┬────────┬─────────────────────────┬────────────────────┐ │
│  │ ✓  │ PC │ Var %  │ Top + loading genes     │ Top − loading gene │ │
│  ├────┼────┼────────┼─────────────────────────┼────────────────────┤ │
│  │ ☐  │ 1  │ 12.7%  │ MALAT1, ACTB, GAPDH…    │ MT-CO1, RPS4, …    │ │
│  │ ☐  │ 2  │  8.1%  │ TOP2A, MKI67, CCNB1…    │ FOS, JUN, EGR1…    │ │
│  │ ☐  │ 3  │  5.9%  │ HBB, HBA1, HBA2, …      │ LYZ, S100A9, …     │ │
│  │ …  │    │        │                         │                    │ │
│  └────┴────┴────────┴─────────────────────────┴────────────────────┘ │
│                                                                      │
│  Suffix (optional):  [                  ]  auto: noPC2_3             │
│                                                                      │
│  [  Create PC subset  ]     2 PCs checked, 48 would remain           │
│                                                                      │
│  ── Existing PC subsets ───────────────────────────────────────────  │
│  • X_pca_noPC2_5     · 48 kept · dropped 2, 5     [✕]                │
│  • X_pca_custom_run  · 47 kept · dropped 2, 5, 9  [✕]                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Loadings table behavior

- Sticky header, scrollable body ~360–420 px tall.
- Gene names rendered as `<span title="loading=0.183">MALAT1</span>` — exact loading value appears on hover without widening cells.
- Row highlighted teal when the checkbox is active.
- Top-N input triggers a refetch of `/api/scanpy/pca_loadings?top_n=N` (cached per-(slot, top_n) in `useData.ts`).
- NaN loadings (which occur when PCA was run on a gene subset — e.g. HVG — so non-subset genes have NaN rows in `varm['PCs']`) are skipped when ranking. The table caption notes "N genes loaded" when a subset PCA was used.

### Creating a PC subset

- The **Suffix** text input is optional. If left blank, the backend generates `noPC<i>_<j>_…` per the pySCN pattern (e.g. `noPC2_5` when PCs 2 and 5 are checked). The live-computed auto-suffix appears as placeholder text.
- Clicking **Create PC subset** calls `POST /api/scanpy/pca_subsets` with `{drop_pc_indices: [2, 5], suffix?: "my_run"}`. On success, the new slot appears in the list below and in the Neighbors form's **PC source** dropdown.
- Disabled states: 0 PCs checked; all PCs checked (would drop everything); suffix collision with an existing slot.
- Errors render as toasts: out-of-range PC indices, suffix collision, no PCA available, no PCs remaining.

### Existing subsets list

A small list below the Create button shows every derived slot. Each row: `<suffix> · <n_kept> kept · dropped <i>, <j>, …` with a `✕` delete button. Clicking `✕` calls `DELETE /api/scanpy/pca_subsets/{obsm_key}`. Empty-state text: "No derived PC subsets yet."

### Neighbors form changes

The Neighbors form gains one new parameter before `n_pcs`:

```
PC source: [ X_pca ▾ ]
           ├ X_pca
           ├ X_pca_noPC2_5
           └ X_pca_custom_run
```

- Default is `X_pca`.
- Dropdown is populated from `pcaSubsets` in the active dataset's store state.
- List refreshes when the Neighbors tab is opened (cheap — a single GET).
- The existing `n_pcs` field is unchanged: "use first N of whatever `use_rep` resolves to." For a derived slot with 48 PCs, `n_pcs=20` takes the first 20 of those 48.
- On submission, the request includes `use_rep` when the selection is not `X_pca`; `sc.pp.neighbors` receives `use_rep` and behaves identically to passing a custom representation.

### Invalidation on PCA re-run

Re-running PCA in the same session overwrites `X_pca`, `varm['PCs']`, and `uns['pca']`, which makes every existing derived slot semantically invalid (column indices no longer correspond to the same eigenvectors).

- On re-run, `run_pca` scans `self.adata.obsm` for keys matching `X_pca_*` (excluding the exact key `X_pca`), deletes each along with its companion `varm['PCs_*']` and `uns['pca']['variance_ratio_*']` entries, and includes `cleared_subsets: [obsm_key, …]` in the response.
- Frontend shows an info toast: `"PCA recomputed — cleared N derived PC subset(s)"` and empties `pcaSubsets` in the active dataset's state.

## Architecture

### Data flow

```
PCA  ──►  .obsm['X_pca'], .varm['PCs'], .uns['pca']['variance_ratio']
           │
           ▼
PCA Loadings (new UI) ──►  .obsm['X_pca_noPC2_5'],
                           .varm['PCs_noPC2_5'],
                           .uns['pca']['variance_ratio_noPC2_5']
           │
           ▼
Neighbors (use_rep = chosen slot)  ──►  .obsp['connectivities']
                                        .obsp['distances']
           │
           ▼
UMAP / Leiden  ──►  .obsm['X_umap'], .obs['leiden']
```

### Backend (`backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`)

No new modules — changes live alongside existing PCA code in `DataAdaptor` and its API routes.

#### New `DataAdaptor` methods

```python
def get_pca_loadings(self, top_n: int = 10) -> dict:
    """Return top ± loading genes per computed PC.

    Raises ValueError if 'pca' not in adata.uns or 'PCs' not in adata.varm.

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
```

Reads `self.adata.varm['PCs']` (shape `n_genes × n_comps`) and `self.adata.uns['pca']['variance_ratio']`. Uses `self.adata.var_names` for gene labels. Rows with NaN values (from subset-PCA runs) are excluded from ranking; returns up to `top_n` valid genes per side per PC.

```python
def create_pca_subset(
    self,
    drop_pc_indices: list[int],   # 1-indexed, user-facing
    suffix: str | None = None,    # optional override for slot suffix
) -> dict:
    """Port of pySCN drop_pcs_from_embedding.

    Creates .obsm[f'X_pca_{suffix}'], .varm[f'PCs_{suffix}'],
    .uns['pca'][f'variance_ratio_{suffix}'].

    Validation:
      - 'pca' must exist and 'X_pca' must be in obsm.
      - drop_pc_indices must be non-empty.
      - All indices must be in [1, n_pcs].
      - Cannot drop all PCs.
      - Generated or provided suffix must not collide with an existing obsm key.

    Side effects:
      - Writes obsm[f'X_pca_{suffix}'] with dropped columns removed.
      - Writes varm[f'PCs_{suffix}'] with matching dropped columns removed.
      - Writes uns['pca'][f'variance_ratio_{suffix}'] likewise.
      - Writes uns['pca'].setdefault('subsets', {})[suffix] =
          {'dropped_pcs': [i, j, ...]} so list_pca_subsets can round-trip
          the exact indices regardless of custom suffix naming.

    Returns:
      {
        'obsm_key': 'X_pca_noPC2_5',
        'varm_key': 'PCs_noPC2_5',
        'variance_ratio_key': 'variance_ratio_noPC2_5',
        'suffix': 'noPC2_5',
        'n_pcs_kept': 48,
        'dropped_pcs': [2, 5],
      }
    """
```

```python
def list_pca_subsets(self) -> list[dict]:
    """Return metadata for every derived PCA subset in adata.obsm.

    Iterates obsm keys starting with 'X_pca_' (excluding the exact key 'X_pca').
    For each match, reports obsm_key, suffix, n_pcs_kept (column count),
    and dropped_pcs — looked up from
    self.adata.uns['pca']['subsets'][suffix]['dropped_pcs'] when present
    (always written by create_pca_subset), otherwise an empty list for
    externally-authored obsm keys that happen to match the prefix.
    """

def delete_pca_subset(self, obsm_key: str) -> None:
    """Delete a derived PC subset's obsm, varm, variance_ratio, and
    uns['pca']['subsets'] entries.

    Raises if obsm_key == 'X_pca' or doesn't exist or doesn't start with 'X_pca_'.
    """
```

#### Required fix in existing `run_pca`

At `adaptor.py:3083-3090`, when `gene_subset` is used, `sc.tl.pca` stores loadings in `adata_pca.varm['PCs']` (subset-sized), but only `obsm['X_pca']` and `uns['pca']` are copied back to `self.adata`. The loadings array is lost.

Fix: after `sc.tl.pca`, build a full-size `(n_genes, n_comps)` array filled with NaN and copy loadings for the subset-selected genes, analogous to how `X_pca` is padded with NaN for inactive cells at `adaptor.py:3079`. Write this to `self.adata.varm['PCs']`.

Without this fix, `get_pca_loadings` cannot show gene loadings after a subset-PCA run.

#### Required changes to existing `run_pca` and `run_neighbors`

**`run_pca`** — add at the end, before returning:

```python
cleared = []
for key in list(self.adata.obsm.keys()):
    if key.startswith('X_pca_') and key != 'X_pca':
        suffix = key[len('X_pca_'):]
        self.adata.obsm.pop(key, None)
        self.adata.varm.pop(f'PCs_{suffix}', None)
        if 'pca' in self.adata.uns:
            self.adata.uns['pca'].pop(f'variance_ratio_{suffix}', None)
        cleared.append(key)
result['cleared_subsets'] = cleared
```

**`run_neighbors`** — add new parameter:

```python
def run_neighbors(
    self,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    metric: str = 'euclidean',
    use_rep: str | None = None,         # NEW
    active_cell_indices: list[int] | None = None,
) -> dict:
    ...
    kwargs = {'n_neighbors': n_neighbors, 'metric': metric}
    if n_pcs is not None:
        kwargs['n_pcs'] = n_pcs
    if use_rep is not None:
        kwargs['use_rep'] = use_rep
    ...
```

The existing cell-subset code path (building `adata_sub` with `X_pca` copied in) must also honor `use_rep`: when `use_rep` is set, copy `adata.obsm[use_rep]` instead of `X_pca` into the subset `adata_sub`.

#### Prerequisites

Add a new entry to `check_prerequisites`:

```python
'pca_loadings': {
    'satisfied': 'pca' in self.adata.uns and 'PCs' in self.adata.varm,
    'missing': [] if satisfied else ['pca (with loadings)'],
}
```

So the UI can disable the PCA Loadings function entry with the standard prerequisite tooltip when PCA hasn't run (or was run before the loadings-copy-back fix was applied).

#### New API routes (`api/routes.py`)

All under the existing `/api/scanpy/*` grouping; all accept `?dataset=<slot>`.

| Method | Path | Payload | Returns |
|---|---|---|---|
| GET | `/api/scanpy/pca_loadings` | `?top_n=10` | `get_pca_loadings` result |
| GET | `/api/scanpy/pca_subsets` | — | `list_pca_subsets` result |
| POST | `/api/scanpy/pca_subsets` | `{drop_pc_indices: [int], suffix?: str}` | `create_pca_subset` result |
| DELETE | `/api/scanpy/pca_subsets/{obsm_key}` | — | `{status: 'deleted', obsm_key}` |

The existing `POST /api/scanpy/neighbors` route gains an optional `use_rep: str` field in its JSON body, passed through to `DataAdaptor.run_neighbors`.

The existing `POST /api/scanpy/pca` response now includes `cleared_subsets: string[]`.

### Frontend (`frontend/src/`)

#### Store (`store.ts`)

Add to `DatasetState`:

```ts
pcaSubsets: Array<{
  obsmKey: string      // e.g. 'X_pca_noPC2_5'
  suffix: string       // e.g. 'noPC2_5'
  droppedPcs: number[] // 1-indexed
  nPcsKept: number
}>
```

Default `[]`. Populated/cleared by:
- Opening the PCA Loadings tab (fetch + write).
- Creating a subset (append after POST succeeds).
- Deleting a subset (filter after DELETE succeeds).
- Re-running PCA with a non-empty `cleared_subsets` response (set to `[]`).

No new top-level store fields. Top-N input and PC-checkbox set live as local React state in the modal.

#### Hooks / API (`hooks/useData.ts`)

- `usePcaLoadings(topN)` — hook wrapping `/api/scanpy/pca_loadings?top_n=N`. Caches per `(slot, topN)` with dataset-slot invalidation on dataset switch.
- `usePcaSubsets()` — hook wrapping `/api/scanpy/pca_subsets`. Writes result into active dataset's `pcaSubsets`.
- `createPcaSubset(dropIndices, suffix?, slot?)` — standalone function. Optimistically appends to store on success.
- `deletePcaSubset(obsmKey, slot?)` — standalone function.

All include `?dataset=<slot>` via the existing `appendDataset` helper.

#### ScanpyModal (`components/ScanpyModal.tsx`)

Add a new function entry under `cell_analysis.functions`:

```ts
pca_loadings: {
  label: 'PCA Loadings',
  description: 'Explore PC gene loadings and create PC subsets to exclude technical PCs',
  prerequisites: ['pca'],
  params: [],
  custom: true,
}
```

Positioned between `pca` and `neighbors` in the function list iteration order.

Add a custom UI block rendered when `selectedFunction === 'pca_loadings'` (analogous to the existing `compare_cells` custom block at `ScanpyModal.tsx:1200`):

1. Top-N number input (default 10).
2. `PCALoadingsTable` — a scrollable, sticky-header table. Each row: checkbox, PC number, variance percent, comma-separated top + genes (`<span title="loading=...">GENE</span>`), comma-separated top − genes.
3. Suffix text input (optional) with live-computed placeholder based on the checked set (`noPC2_3` etc.).
4. Create button with a status line to its right: `N PCs checked, M would remain` or disabled state explanations.
5. Existing subsets list (reads from `pcaSubsets`), with a `✕` delete per row.

**Neighbors form change**: extend the form renderer to handle a new param type `pc_source_select`, which renders a `<select>` populated from the active dataset's `pcaSubsets` plus the literal `X_pca`. Add this param to the `neighbors` function definition before `n_pcs`:

```ts
{
  name: 'use_rep',
  label: 'PC source',
  type: 'pc_source_select',
  default: 'X_pca',
  description: 'Which PC embedding to use. Create derived subsets via PCA Loadings.',
},
```

When `use_rep === 'X_pca'`, the field is omitted from the request body (preserving the existing default path).

**PCA invalidation handling**: after any PCA submission, inspect the response for a non-empty `cleared_subsets`, and if present, clear `pcaSubsets` in the active dataset state and show an info toast via the existing toast mechanism.

#### Messages (`frontend/src/messages.ts`)

```ts
pcaLoadingsPrereq: 'Run PCA first to explore loadings.',
pcaLoadingsEmpty: 'No PC loadings available — re-run PCA to populate loadings.',
pcaSubsetCreated: (suffix: string, nKept: number) =>
  `Created PC subset "${suffix}" (${nKept} PCs kept)`,
pcaSubsetCollision: (suffix: string) =>
  `A PC subset named "${suffix}" already exists`,
pcaSubsetsCleared: (n: number) =>
  `PCA recomputed — cleared ${n} derived PC subset${n === 1 ? '' : 's'}`,
pcaSubsetNoneChecked: 'Check at least one PC to drop',
pcaSubsetAllDropped: 'Cannot drop all PCs',
```

## Error Handling

| Case | Backend response | Frontend behavior |
|---|---|---|
| PCA not run when opening loadings tab | 400 with `missing: ['pca']` | Tab disabled with prereq tooltip; empty state if somehow reached |
| `varm['PCs']` missing (pre-fix adata) | 400 with `reason: 'no_loadings'` | Empty state message, prompt to re-run PCA |
| `drop_pc_indices` empty | 400 | Button disabled client-side; defense-in-depth toast on server error |
| PC index out of range | 400 | Toast |
| Suffix collision | 409 | Toast `pcaSubsetCollision` |
| All PCs dropped | 400 | Button disabled client-side; defense-in-depth toast |
| Delete non-existent obsm_key | 404 | Toast + refresh subsets list |
| Neighbors called with `use_rep` pointing at a slot that no longer exists (e.g. PCA was re-run in another browser) | 400 | Toast + refresh subsets list + reset PC source to `X_pca` |

## Testing

No automated tests exist in the project (per CLAUDE.md). Manual validation:

1. Load `toy_spatial.h5ad` (bundled). Run PCA with defaults.
2. Open PCA Loadings. Confirm table renders with 50 rows, top-10 genes per side, variance percents.
3. Change top-N to 5 and to 25; confirm refetch and re-render.
4. Check PC 2 and PC 5; confirm placeholder reads `noPC2_5` and status line reads "2 PCs checked, 48 would remain".
5. Click Create PC subset. Confirm subset appears in the list, and appears in Neighbors' PC source dropdown.
6. Provide a custom suffix (`my_run`), confirm it's used and appears in the list.
7. Run Neighbors with PC source = `X_pca_noPC2_5`, then UMAP, then Leiden. Confirm they succeed.
8. Delete the subset; confirm it disappears from both the list and the Neighbors dropdown.
9. Re-run PCA; confirm a toast appears saying "cleared N derived PC subsets" and the list empties.
10. Run PCA with a `gene_subset` (HVG). Confirm loadings table still renders (non-HVG genes skipped in rankings), varm has full-size `PCs` with NaN for non-HVG genes.
11. Export h5ad, reload, confirm derived subsets survive round-trip (if present at export time) and are visible in the new session's subsets list.
12. Verify TypeScript builds cleanly: `npm run build` from `frontend/`.

## Documentation

- `CLAUDE.md`: new rows in the DataAdaptor grouped methods list (under Scanpy cell analysis); new rows in the API Endpoints table; new Key Behaviors bullet for "PCA Loadings explorer / PC subsets".
- `CHANGELOG.md`: `Added — PCA Loadings explorer in Analyze → Cell Analysis: inspect top +/− loading genes per PC, create derived PC-subset embeddings that exclude selected PCs, and route Neighbors through them.`
- `README.md`: short walkthrough step describing the loadings table, PC-dropping workflow, and how it feeds Neighbors / UMAP / Leiden.
