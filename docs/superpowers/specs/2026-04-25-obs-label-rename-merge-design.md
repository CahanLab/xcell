# Design: Selectable contour annotations + rename/merge category labels

Date: 2026-04-25

## Problem

1. **Bug**: Cells in a Contourize-generated annotation cannot be selected from the Cells panel. Clicking a category value or its checkbox does nothing.
2. **Missing capability**: Users cannot rename individual category values within a categorical `.obs` column (contour level labels, leiden clusters, user annotations, etc.).
3. **Missing capability**: Users cannot merge two or more category values into a single new label (e.g. merge `'a'` and `'b'` from a `cell parts` annotation into a new combined cluster).

## Root cause of the bug

`run_contourize` writes a `pd.Categorical` whose categories are floats (the threshold values).

- `GET /api/obs/summary/{col}` stringifies categories: `"0.0"`, `"0.123"`, …
- `GET /api/obs/{col}` returns raw float categories: `[0.0, 0.123, …]`.
- Frontend computes `categories.indexOf(categoryValue)` where `categoryValue` is the stringified value from the summary endpoint. `indexOf("0.0")` against `[0.0, …]` returns `-1` (strict equality). Selection silently does nothing.

The same defect breaks `ScanpyModal`'s "Compare Cells" picker for any float-valued categorical column.

## Fix and feature design

### 1. Bug fix — wire-format consistency

`DataAdaptor.get_obs_column` stringifies category values to match the summary endpoint:

```python
result["categories"] = [str(v) for v in series.cat.categories.tolist()]
```

In-memory `.obs` is unchanged. The frontend `indexOf` lookup against the stringified summary value now succeeds, fixing both Cells-panel selection and ScanpyModal Compare-Cells for contour columns.

### 2. Rename a single label

Goal: Let the user rename one category value within any categorical `.obs` column (contour, leiden, user annotation, original metadata).

Backend:
- `DataAdaptor.rename_obs_label(column, old_label, new_label) -> dict`
  - Validates column exists and is categorical or string dtype.
  - Validates `old_label` exists; for categorical columns matches against stringified categories. For string columns matches values directly.
  - Rejects empty `new_label` (after trim).
  - Rejects collision: if `new_label` already exists in the column, return a 409-style error suggesting "Use merge instead". Avoids surprising data loss.
  - Implements via `cat.rename_categories({old: new})` for categorical, or string replace for string columns.
  - Returns `{column, old_label, new_label, n_cells_renamed}`.
- Route: `POST /api/obs/{column}/rename_label` body `{old_label: str, new_label: str}`.

Frontend:
- In `CellPanel.tsx CategoryColumn`, make each label name in the expanded category list double-click editable (mirrors the existing column-rename pattern). On commit:
  - POST to the rename endpoint.
  - On success, refresh `obs/summaries` and `obs/{col}` (via the existing data refresh path).
  - If `selectedCategorySource` matched the renamed label, update its `value` so the teal highlight stays accurate.
  - On collision error, show a toast ("A label with that name already exists. Use Merge labels… instead.").

### 3. Merge multiple labels into a new label

Goal: Let the user pick 2+ labels in a column and replace them with a single new label.

Backend:
- `DataAdaptor.merge_obs_labels(column, labels, new_label) -> dict`
  - Validates column exists and is categorical or string dtype.
  - Validates `labels` is a list of length ≥ 2 and all members exist (against stringified categories for categorical).
  - Trims `new_label`; rejects empty.
  - For categorical columns:
    - If `new_label` matches a category not in `labels`, that's a "merge into existing" — replace each label in `labels` with `new_label` via `rename_categories`, then `remove_unused_categories`.
    - If `new_label` matches a category in `labels`, treat that as the survivor; merge the others into it.
    - Otherwise, rename one of `labels` to `new_label`, then merge the rest into it.
  - For string columns: vectorized replace.
  - Returns `{column, merged_labels, new_label, n_cells_merged}`.
- Route: `POST /api/obs/{column}/merge_labels` body `{labels: string[], new_label: str}`.

Frontend:
- New small modal `MergeLabelsModal.tsx`:
  - Triggered from a `⋯` overflow on the column header (added next to the existing Hide/Color row).
  - Lists all category values with checkboxes (Select all / Clear all helpers).
  - "New label name" text input (defaults to `<first>+<second>` joined for convenience, but freely editable).
  - "Merge" button enabled when ≥2 boxes are checked and the name is non-empty.
  - On confirm: POST, on success refresh summaries, close modal. On error: inline error message.

### Out of scope (explicit non-goals for v1)

- Renaming the column itself (the user confirmed this is not what they meant).
- Undo. (Matches existing patterns for delete-annotation and label-cells.)
- Splitting a label.
- Cross-column merges.
- Persisting these edits separately from h5ad — they round-trip via the normal h5ad export.

## Files to touch

- `xcell/backend/xcell/adaptor.py` — fix `get_obs_column`; add `rename_obs_label`, `merge_obs_labels`.
- `xcell/backend/xcell/api/routes.py` — `POST /obs/{column}/rename_label`, `POST /obs/{column}/merge_labels`.
- `xcell/frontend/src/hooks/useData.ts` — two helpers (`renameObsLabel`, `mergeObsLabels`) accepting optional `slot`.
- `xcell/frontend/src/components/CellPanel.tsx` — inline label rename, column overflow with "Merge labels…", refresh after success, selectedCategorySource remap.
- `xcell/frontend/src/components/MergeLabelsModal.tsx` — new modal.
- `xcell/CHANGELOG.md`, `xcell/README.md`, `xcell/CLAUDE.md` — doc updates.

## Verification

- Manual: load a spatial dataset, run Contourize, confirm clicking a contour level in the Cells panel selects cells. Rename one level, confirm summary refreshes. Merge two levels, confirm result. Repeat on a leiden clustering and on a user annotation.
- Build: `npm run build` in `xcell/frontend` must pass.
- No automated tests exist for this area.
