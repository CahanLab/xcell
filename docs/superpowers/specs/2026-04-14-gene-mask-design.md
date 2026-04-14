# Gene Mask — Design

**Date:** 2026-04-14
**Author:** Patrick Cahan (brainstormed with Claude)
**Status:** Draft — awaiting user review before implementation plan

## Summary

Add a per-dataset gene mask feature that lets users hide genes based on boolean `.var` columns (e.g. `highly_variable`, `spatially_variable`, `mt`). Masked genes disappear from the Gene Panel (browse, search, gene set expansions) and are excluded when computing aggregated gene set scores for display coloring. The feature is a **view filter** — the underlying data, gene sets, and all analysis operations remain untouched.

## Goals

- Let users quickly scope the Gene Panel to a relevant gene universe (HVG, spatially variable, etc.) while hiding noise (mitochondrial, ribosomal, blacklisted).
- Support multiple simultaneous boolean columns with per-column Keep/Hide semantics.
- Reflect the mask in all Gene Panel surfaces: browse, search, gene set displays, and gene set score aggregation.
- Keep analysis operations untouched so existing `gene_subset` dropdowns work unchanged.

## Non-Goals

- Persisting the mask across sessions (no localStorage, no h5ad embedding).
- Applying the mask to Diff Exp, Marker Genes, Gene PCA, Gene Neighbors, Spatial Autocorr, Build Gene Graph, Find Similar Genes, Cluster Genes, or Line Association — they retain their existing per-call `gene_subset` dropdowns.
- Applying the mask to the heatmap (heatmap is an explicit-gene-list operation; users drive it directly).
- Creating a UI for authoring new `.var` boolean columns.
- Exporting or saving mask presets.
- An "invert mask" global toggle.

## User-Facing Behavior

### Entry point

The Gene Panel header gains a `⋯` overflow menu (reusing the existing `OverflowMenu` component). The menu contains one item: **"Gene mask…"**, which opens the `GeneMaskModal`.

When a mask is active, a muted badge in the Gene Panel header shows `"1,987 / 32,000"` (visible / total).

### Gene Mask modal

Layout:

```
┌────────────────────────────────────────┐
│  Gene Mask                          ✕  │
├────────────────────────────────────────┤
│  Filter genes by .var boolean columns. │
│                                        │
│  highly_variable      (2,000 / 32,000) │
│    ( ) Off  (•) Keep  ( ) Hide         │
│                                        │
│  spatially_variable     (450 / 32,000) │
│    (•) Off  ( ) Keep  ( ) Hide         │
│                                        │
│  mt                     (13 / 32,000)  │
│    ( ) Off  ( ) Keep  (•) Hide         │
│                                        │
│  Combine Keep columns:                 │
│    (•) Match ANY   ( ) Match ALL       │
│                                        │
│  Preview: 1,987 of 32,000 visible      │
│                                        │
│  [ Clear ]            [ Cancel ] [Apply]│
└────────────────────────────────────────┘
```

- On open, the modal fetches `/api/var/boolean_columns` (existing) and a new compact endpoint that returns the packed bool values per column so the preview count can update client-side without round-trips.
- Each column has a three-state radio: Off / Keep / Hide.
- The "Combine Keep columns" radio controls `keep_combine_mode`: `or` (default) or `and`. Hide columns are always combined with OR.
- The preview count recomputes locally as radios change.
- `Clear` is disabled when no mask is currently active on the server. It sends `DELETE /api/gene_mask`.
- `Apply` sends `POST /api/gene_mask`. If the server rejects with "0 visible genes", the modal shows an inline error and stays open.
- `Cancel` closes without applying.

### Combination semantics

```
visible = keep_mask AND NOT hide_mask

keep_mask = all True  if no Keep columns
          = OR(columns) if keep_combine_mode == 'or'
          = AND(columns) if keep_combine_mode == 'and'

hide_mask = all False if no Hide columns
          = OR(columns) otherwise
```

### Masking side effects

1. **Currently colored single gene is now masked:** `colorMode` resets to `'none'`, `expressionData` cleared, and a toast shows `"<gene> is masked; coloring cleared"`.
2. **Currently colored bivariate genes contain a masked gene:** same treatment as single — clear bivariate state and toast.
3. **Gene set contains masked genes:** in the expanded set view, masked genes are filtered from the displayed list and a small muted suffix shows `"(3 hidden)"` next to the `"12 genes"` count. The underlying `set.genes` array is untouched.
4. **Gene set coloring with masked genes present:** `/api/expression/multi` automatically filters masked genes from the incoming list before scoring, so the displayed color score excludes them. Response includes `n_masked_excluded` (optional, for UI feedback).
5. **Gene set where all genes are masked:** the existing empty-gene-list path in `get_multi_gene_expression` returns zeros. A toast shows `"All genes in this set are masked — score is zero"`.
6. **Adding a masked gene to a set (drag, import, search-add):** allowed. The set stores it; it just won't display or score until the mask is lifted.
7. **Dataset switch (multi-dataset mode):** each dataset has its own mask. `setActiveSlot` already syncs per-slot state to flat fields, so the Gene Panel swaps automatically.

## Architecture

### Mask state lives on the backend `DataAdaptor`, per-dataset

The adaptor already fits the per-dataset requirement (one adaptor per slot). Storing mask state there means:

- One source of truth per dataset.
- Multi-dataset slot isolation falls out for free.
- Session-only lifetime falls out for free (adaptors reset on page reload).
- No need to pass mask parameters on every request.

### Data flow

```
User toggles modal radios
        │
        ▼
POST /api/gene_mask  (keep_columns, hide_columns, keep_combine_mode)
        │
        ▼
DataAdaptor.set_gene_mask()
  ├── validate column names
  ├── _compute_visible_mask() → np.ndarray[bool]
  ├── store _gene_mask_config + _visible_gene_mask
  └── return {active, keep_columns, hide_columns, keep_combine_mode,
              n_visible, n_total, visible_gene_names}
        │
        ▼
Frontend store updates datasets[slot].geneMaskConfig
        │
        ▼
Reaction helpers run:
  - Check coloredGene/bivariateGenes against visible set → clear if masked
  - Invalidate browse page (useGeneBrowse re-fetches via useEffect dep)
  - Re-fetch schema for updated n_genes_visible
  - GenePanel re-renders with filtered gene sets
```

## Backend Changes

### `DataAdaptor` (adaptor.py)

New instance state, initialized in `__init__`:

```python
self._gene_mask_config: dict | None = None
self._visible_gene_mask: np.ndarray | None = None  # bool array, shape (n_genes,)
```

New methods:

```python
def get_gene_mask(self) -> dict:
    """Return current mask state.

    Returns:
        {
            'active': bool,
            'keep_columns': list[str],
            'hide_columns': list[str],
            'keep_combine_mode': 'or' | 'and',
            'n_visible': int,
            'n_total': int,
            'visible_gene_names': list[str],  # only when active
        }
    """

def set_gene_mask(
    self,
    keep_columns: list[str],
    hide_columns: list[str],
    keep_combine_mode: str = 'or',
) -> dict:
    """Apply a mask.

    - Validates all column names exist in .var and are bool-like.
    - Empty keep + empty hide clears the mask (equivalent to clear_gene_mask).
    - Computes the visible mask and caches it.
    - Raises ValueError if the mask would leave 0 visible genes.

    Returns: same shape as get_gene_mask().
    """

def clear_gene_mask(self) -> dict:
    """Clear mask state. Returns get_gene_mask() post-clear."""

def _compute_visible_mask(
    self,
    keep_columns: list[str],
    hide_columns: list[str],
    keep_combine_mode: str,
) -> np.ndarray:
    """Internal: build bool mask from config. Reuses column-to-bool conversion
    already used by _resolve_gene_mask()."""

def get_visible_gene_names(self) -> list[str]:
    """Return gene names where _visible_gene_mask is True, or all gene names
    if mask is inactive."""

def _filter_to_visible(
    self,
    genes: list[str],
) -> tuple[list[str], int]:
    """Split a gene list: (visible_genes, n_excluded).
    Used by get_multi_gene_expression."""
```

Modified methods:

- `get_multi_gene_expression(genes, ...)`:
  - At the top, call `_filter_to_visible(genes)`. Operate on the visible subset.
  - Add `n_masked_excluded` to the response dict.
  - The existing empty-gene-list branch handles the "all masked" case.
- `get_schema()`:
  - When the mask is active, include `n_genes_visible` in the schema response alongside existing `n_genes`.
- `run_filter_genes()`, `run_exclude_genes()`:
  - After the adata mutation, if `_gene_mask_config is not None`:
    - If all referenced columns still exist: re-call `set_gene_mask(existing_config)` to regenerate the cached mask against the new gene axis.
    - If any referenced column no longer exists: `clear_gene_mask()` and set a flag `mask_cleared=True` in the operation response.
- `swap_var_index()`:
  - Same as above — re-regenerate the mask so cached bool array matches new `adata.var.index`.

### Routes (routes.py)

New:

```python
GET  /api/gene_mask                     → adaptor.get_gene_mask()
POST /api/gene_mask                     → adaptor.set_gene_mask(body)
DELETE /api/gene_mask                   → adaptor.clear_gene_mask()
GET  /api/var/boolean_column_values     → returns per-column positional True-indices
                                          for client-side preview computation
```

Response shape for `GET /api/var/boolean_column_values`:

```json
{
  "n_genes": 32000,
  "columns": {
    "highly_variable": [2, 17, 83, ...],
    "spatially_variable": [17, 202, ...],
    "mt": [99, 183, ...]
  }
}
```

Indices are positions in the current `adata.var.index` (0..n_genes-1). Column order matches `get_var_boolean_columns()` (i.e. insertion order in `.var`). With this representation, the modal can compute the preview count locally by building index sets, intersecting/unioning per the radios, and taking `|keep_set - hide_set|`.

All accept `?dataset=` query param.

Modified:

- `GET /api/genes/browse` → paginates over visible gene list; `total` reflects visible count.
- `GET /api/genes/search` → restricts matches to visible genes.

Left alone:

- `GET /api/genes` → still returns the full gene list. This endpoint is not called by the frontend today, but it remains a full-universe escape hatch for anything that needs it.

### What's explicitly NOT modified in the backend

- `get_gene_names()` stays as-is (returns full list). Used by heatmap, export, analysis operations, gene identifier swap, `GET /api/genes` — all need the full gene universe.
- `_resolve_gene_mask()` (analysis gene_subset resolver) is untouched.
- Single-gene `/expression/{gene}` endpoint is untouched — a user may still look up a masked gene by exact name if needed.
- All `/scanpy/*` endpoints are untouched.
- Heatmap endpoint is untouched.
- Line association endpoints are untouched.
- Diff exp and marker gene endpoints are untouched.

## Frontend Changes

### Store (store.ts)

Add to `DatasetState`:

```ts
interface GeneMaskConfig {
  active: boolean
  keepColumns: string[]
  hideColumns: string[]
  keepCombineMode: 'or' | 'and'
  nVisible: number
  nTotal: number
  visibleGeneNames: string[] | null  // null when inactive
}

geneMaskConfig: GeneMaskConfig | null
```

Follows the dual-write pattern: nested under `datasets[slot]` + flat top-level field synced via `syncFlatFields()`. `loadDatasetIntoSlot` initializes it to `null`.

New top-level (not per-dataset) state:

```ts
geneMaskModalOpen: boolean
setGeneMaskModalOpen: (open: boolean) => void
```

No derived store field. Components that need a `Set<string>` for fast membership checks build it locally with `useMemo(() => new Set(geneMaskConfig?.visibleGeneNames ?? []), [geneMaskConfig])`. When `geneMaskConfig` is null or `visibleGeneNames` is null, the component short-circuits and renders the full unfiltered list.

### API helpers (hooks/useData.ts)

Three standalone async functions, each accepting optional `slot?: DatasetSlot`:

```ts
async function fetchGeneMask(slot?: DatasetSlot): Promise<void>
async function applyGeneMask(config: GeneMaskConfig, slot?: DatasetSlot): Promise<void>
async function clearGeneMask(slot?: DatasetSlot): Promise<void>
async function fetchBooleanColumnValues(slot?: DatasetSlot): Promise<Record<string, number[]>>
```

Each mutation helper (`applyGeneMask`, `clearGeneMask`):

1. Calls the backend endpoint.
2. Updates `datasets[slot].geneMaskConfig` via `patchSlotState`.
3. Syncs flat fields if the target slot is active.
4. Runs reaction logic:
   - If `colorMode === 'expression'` and `coloredGene` is not in new visible set → clear coloring + push toast.
   - Same check for bivariate mode.
5. Triggers schema re-fetch (updated `n_genes_visible`).
6. The existing `useGeneBrowse` hook depends on `geneMaskConfig` via `useEffect` so it re-fetches the current page automatically.

### New component: `components/GeneMaskModal.tsx`

- Opens when `geneMaskModalOpen === true`.
- On mount, fetches `/api/var/boolean_columns` and `/api/var/boolean_column_values`.
- Renders one three-state radio per column, the Keep combine radio, the preview count, and the Clear/Cancel/Apply buttons.
- Preview count computed locally by combining the fetched True-index lists according to the selected radios.
- On `Apply`, calls `applyGeneMask(config)`. On error (0 visible), shows inline error, stays open.
- On `Clear`, calls `clearGeneMask()`, then closes.
- Shares visual conventions with `ClusterGeneSetModal` for consistency.

### Modified component: `components/GenePanel.tsx`

1. **Header:**
   - New `⋯` overflow menu (uses existing `OverflowMenu`) with `"Gene mask…"` item that sets `geneMaskModalOpen = true`.
   - Mask-active badge showing `"1,987 / 32,000"` in muted text.

2. **Gene set rendering (expanded view):**
   - Read `visibleGeneNameSet` from the store.
   - If non-null, filter `set.genes` through it when rendering.
   - Show `"12 genes (3 hidden)"` when `hiddenCount > 0`.

3. **Gene search / browse rendering:**
   - No changes — these hooks already fetch from the backend, which now returns only visible genes.

4. **Gene set scoring display:**
   - No changes — `/api/expression/multi` handles the filter on the backend.

### New UI strings (messages.ts)

```ts
geneMaskCleared: (gene: string) => `${gene} is masked; coloring cleared.`
geneMaskNoneVisible: 'Mask would leave zero visible genes. Adjust your selection.'
geneMaskNoBoolColumns: 'No boolean .var columns found. Run Highly Variable Genes or Spatial Autocorrelation first.'
geneMaskAllMaskedInSet: 'All genes in this set are masked — score is zero.'
geneMaskClearedAfterFilter: 'Gene mask was cleared because referenced columns were removed.'
```

## Edge Case Table

| Case | Behavior |
|------|----------|
| User colors by masked gene (shouldn't happen — not shown in search) | Backend single-gene endpoint works; frontend can't easily surface it (search hides). Acceptable. |
| Active single-gene coloring, then mask applied hiding it | Clear `colorMode`, `expressionData`, `coloredGene`. Toast. |
| Active bivariate coloring, either gene now masked | Clear bivariate state. Toast. |
| Gene set display with some masked genes | Filter out masked; show `(N hidden)` suffix. |
| Gene set coloring with some masked genes | Backend auto-filters before scoring. No visual change beyond updated score. |
| Gene set where all genes masked | Backend returns zeros. Toast. |
| Apply mask that leaves 0 visible | Backend `ValueError`. Modal shows inline error, stays open. |
| Dataset switch | Automatic — per-slot state, existing sync machinery. |
| Scanpy `run_filter_genes` / `run_exclude_genes` drops columns referenced by active mask | Backend clears mask, response flag, frontend toasts. |
| Scanpy `run_filter_genes` / `run_exclude_genes` keeps all referenced columns | Backend regenerates `_visible_gene_mask` against new gene axis. |
| Gene identifier swap (`swap_var_index`) | Backend regenerates `_visible_gene_mask` (gene order may change). Mask columns preserved across swap. |
| New boolean column added via Scanpy (e.g. HVG, spatial_autocorr) | Appears in modal next time it opens. No live refresh needed. |
| Drag masked gene into set | Allowed. Set stores it; just hidden from display until mask lifted. |

## Testing Strategy

Manual verification (no automated tests in the project yet):

1. **Setup:** Load `toy_spatial.h5ad` (or similar with several `.var` boolean columns).
2. **Run HVG** to create a `highly_variable` column.
3. **Open Gene Mask modal** from Gene Panel `⋯` → `Gene mask…`.
4. **Toggle `highly_variable` → Keep**, apply. Verify:
   - Gene Panel header shows `"N / total"` badge.
   - Browse paginates over HVG genes only.
   - Search returns only HVG matches.
   - A gene set containing non-HVG genes shows `"(N hidden)"` suffix.
5. **Color by a gene set** whose members include both HVG and non-HVG. Verify the score differs from the unmasked score (non-HVG excluded).
6. **Color by a single HVG gene**, then apply a mask that hides it. Verify coloring clears and toast appears.
7. **Apply a mask with conflicting columns** (Keep on a column with no True values). Verify "0 visible" error is inline, modal stays open.
8. **Switch datasets** in dual-dataset mode. Verify each dataset remembers its own mask.
9. **Run `filter_genes`** while a mask is active. Verify mask regenerates or clears (depending on whether referenced columns survive).
10. **Swap gene identifier** while a mask is active. Verify mask still applies correctly.
11. **Reload the page.** Verify mask is cleared (session-only).
12. **Export h5ad.** Verify exported file has no mask artifacts.

## Documentation Updates

During implementation:

- **CLAUDE.md** — Add to API endpoints table, Store Types (`GeneMaskConfig`), Components table (`GeneMaskModal`), and Key Behaviors (Gene mask section).
- **CHANGELOG.md** — Entry under `## [Unreleased]` → `### Added`: "Gene Mask: hide or keep genes in the Gene Panel based on `.var` boolean columns (e.g. HVG, spatially variable, mitochondrial). Opens from the Gene Panel `⋯` menu. Applies to browse, search, gene set display, and gene set scoring. Per-dataset, session-only."
- **README.md** — Add a short "Gene Mask" subsection under the walkthrough, describing the modal and the three-state toggles.

## Open Questions

None at this stage. All design decisions were resolved during brainstorming.
