# Multi-Line Combined Association Analysis

## Problem

Spatial transcriptomics datasets often contain multiple tissue sections or replicates. Users need to test gene-expression association along a shared biological axis (e.g., cortex→medulla) by drawing one line per section, assigning cells to each line, and pooling the cells for a single, higher-powered spline regression.

## User Flow

1. Draw a line on each section/replicate representing the same biological axis.
2. For each line, select cells (via lasso tool or clicking a category value in the Cell Panel) and click `+` in the Lines panel to associate them.
3. In the Lines panel, check the lines to include (checkboxes appear only on lines with projections).
4. Click "Find Associated Genes" in the action bar that appears.
5. The Line Tools modal opens in multi-line mode: shows the selected lines with per-line direction toggles, analysis parameters, and a run button.
6. Results appear in the existing Line Association results modal.

## Design

### 1. Lines Panel — Line Selection Checkboxes

**File:** `xcell/frontend/src/components/ShapeManager.tsx`

- Add a checkbox to each line row where `projections.length > 0`.
- Store checked line IDs in local component state (`checkedLineIds: Set<string>`).
- When 1+ lines are checked, render a compact action bar below the line list:
  - Text: "{N} lines selected ({M} cells)"
  - "Find Associated Genes" button
- Clicking the button opens the Line Tools modal in multi-line mode, passing the checked line IDs.
- Checkboxes are independent of the existing `activeLineId` selection (which highlights a single line for editing).

### 2. Line Tools Modal — Multi-Line Mode

**File:** `xcell/frontend/src/components/ShapeManager.tsx` (existing `LineToolsModal`)

Currently the modal accepts a single `line` prop. Add support for a `lines` prop (array) for multi-line mode.

**Multi-line mode layout:**

- **Header:** "Line Association: {N} lines ({M} cells)"
- **Line list:** Each checked line shown as a row with:
  - Line name
  - Cell count (from projections)
  - Reverse direction toggle button (arrow icon, toggles between → and ←). Local state: `reversals: Record<string, boolean>`, all default `false`.
- **Gene Association section:** Same controls as current single-line mode:
  - Gene subset selector (boolean .var columns)
  - Test variable (position / distance)
  - Spline knots, FDR threshold, max genes/module
  - "Find Associated Genes" button
- **Omitted sections:** Smoothing, Appearance, Projections, Projection Embedding (these are per-line concerns handled via the single-line gear button).

**Single-line mode:** Unchanged. Opened via gear button on any line row. Retains all existing sections including Gene Association.

### 3. Backend — Shared Regression Helper

**File:** `xcell/backend/xcell/adaptor.py`

Extract the core spline regression logic from `test_line_association` into a private method:

```python
def _run_spline_association(
    self,
    test_values: np.ndarray,      # pooled normalized positions or distances
    cell_indices: np.ndarray,      # pooled cell indices into adata
    gene_mask: np.ndarray,         # boolean mask over genes
    n_spline_knots: int = 5,
    fdr_threshold: float = 0.05,
    top_n: int = 50,
) -> dict:
```

This method handles:
- Expression matrix extraction for `cell_indices` × `gene_mask`
- B-spline basis construction with quantile-based knots
- OLS regression (all genes simultaneously)
- F-test vs intercept-only null
- FDR correction (Benjamini-Hochberg)
- Effect sizes (R², amplitude, direction)
- Profile evaluation at 50 positions
- Hierarchical clustering into modules
- Diagnostics

Returns the same result dict shape as the current `test_line_association`.

Refactor `test_line_association` to:
1. Project cells onto the single line → get positions/distances
2. Select test values
3. Call `_run_spline_association`
4. Add line-level metadata to result

### 4. Backend — Multi-Line Association Method

**File:** `xcell/backend/xcell/adaptor.py`

```python
def test_multi_line_association(
    self,
    lines: list[dict],  # [{name, cell_indices, reversed}, ...]
    gene_subset: str | list[str] | dict | None = None,
    test_variable: str = 'position',
    n_spline_knots: int = 5,
    min_cells: int = 20,
    fdr_threshold: float = 0.05,
    top_n: int = 50,
) -> dict:
```

Logic:
1. For each entry in `lines`:
   - Look up line geometry by name in `self._drawn_lines`
   - Get embedding coordinates
   - Project the entry's `cell_indices` onto the line → positions, distances
   - If `reversed=True`: `positions = 1.0 - positions`
   - Append to pooled arrays: `all_test_values`, `all_cell_indices`
2. Select test values (position or distance) from pooled arrays
3. Validate `min_cells` against pooled count
4. Resolve `gene_mask` from `gene_subset`
5. Call `_run_spline_association(all_test_values, all_cell_indices, gene_mask, ...)`
6. Add metadata to result: `n_lines`, `lines_used` (list of line names), combined `line_name` string

### 5. API — New Endpoint

**File:** `xcell/backend/xcell/api/routes.py`

New request model:

```python
class MultiLineEntry(BaseModel):
    name: str
    cell_indices: list[int]
    reversed: bool = False

class MultiLineAssociationRequest(BaseModel):
    lines: list[MultiLineEntry]
    gene_subset: str | list[str] | None = None
    test_variable: str = 'position'
    n_spline_knots: int = 5
    min_cells: int = 20
    fdr_threshold: float = 0.05
    top_n: int = 50
```

Endpoint: `POST /lines/multi-association`

Response model: Extend `LineAssociationResponse` with optional fields:

```python
n_lines: int = 1
lines_used: list[str] = []
```

Existing `POST /lines/association` endpoint remains unchanged.

### 6. Frontend — Multi-Line Analysis Call

**File:** `xcell/frontend/src/hooks/useData.ts`

Add `runMultiLineAssociation` function:

```typescript
interface MultiLineEntry {
  name: string
  cellIndices: number[]
  reversed: boolean
}

interface MultiLineAssociationParams {
  lines: MultiLineEntry[]
  geneSubset?: string | string[] | { columns: string[]; operation: string } | null
  testVariable?: 'position' | 'distance'
  nSplineKnots?: number
  minCells?: number
  fdrThreshold?: number
  topN?: number
}
```

Posts to `/api/lines/multi-association`. The `useLineAssociation` hook gains a `runMultiLineAssociation` method alongside the existing `runAssociation`.

### 7. Store — Response Type Extension

**File:** `xcell/frontend/src/store.ts`

Add optional fields to `LineAssociationResult`:

```typescript
n_lines?: number
lines_used?: string[]
```

### 8. Results Modal — Multi-Line Display

**File:** `xcell/frontend/src/components/LineAssociationModal.tsx`

- When `n_lines > 1`: header shows "Line Association: {lines_used joined by ' + '}"
- Summary stats bar adds a "Lines" item showing the count
- All other behavior (modules, heatmap, filters, gene set export) unchanged — the result shape is identical

### 9. Cell Panel — Selection Highlight

**File:** `xcell/frontend/src/components/CellPanel.tsx`

When a user clicks a category value to select those cells, highlight the clicked row with a visual indicator:

- Apply a left border accent (e.g., 3px solid teal) or subtle background tint to the category value row whose cells are currently in `selectedCellIndices`.
- Compare `selectedCellIndices` against each category's cell indices to determine which row is active.
- The highlight clears when selection changes (lasso, different category click, clear selection).

## Scope Exclusions

- No automatic alignment of line directions (user manually toggles reversal per line).
- No weighted pooling across lines (all cells contribute equally regardless of which line they belong to).
- No per-line results breakdown in the results modal (results are for the pooled analysis only).
- No new backend endpoint for metadata-based cell selection — users select cells via existing Cell Panel → category click → `+` workflow.
