# Select Cells by Expression — Design

**Date:** 2026-04-14
**Status:** Draft — awaiting user review
**Scope:** Feature 1 of 2 from brainstorming session. Feature 2 (mask genes by `.var` boolean columns) is a separate spec.

## Goal

Let users select cells in the scatter plot based on the expression of a gene or a gene set score, using an interactive histogram with draggable thresholds. Results can either update the current cell selection or be saved as an annotation with two labels (`high` / `low`), optionally handed off to the differential expression modal in one click.

## Motivating workflow

A common scRNA-seq use case: the user lassoes a spatial region, then wants to split those cells into two groups based on expression of a marker gene or a signature score, and run differential expression between the groups. Today this requires manual juggling of annotations across multiple modals. This feature collapses the flow into: lasso → open `Select cells…` on a gene → pick a threshold → Label cells → Open Diff Exp.

## UX

### Entry points

A new modal, `SelectByExpressionModal`, opened from the `⋯` overflow menu on:

1. **Gene set rows** (all categories). New item `Select cells…` added to the existing `OverflowMenu` in `GenePanel.tsx` (~line 937). Disabled with a tooltip when the set has 0 genes.
2. **Individual gene rows** in browse/search results (`GenePanel.tsx` ~line 614–636). A new `⋯` button is added with a single `Select cells…` item.
3. **Individual gene rows inside expanded gene sets** (`GenePanel.tsx` ~line 962–989). Same treatment.

Modal title varies by source:
- Single gene: `Select cells by CD3E`
- Gene set: `Select cells by T cell markers (mean)` — parenthetical shows the current `geneSetScoringMethod` from display preferences.

### Modal layout

```
╔═══════════════════════════════════════════╗
║ Select cells by CD3E                      ║
║ ─────────────────────────────────────────║
║                                           ║
║  Threshold mode:  [Above] [Below] [Between]
║                                           ║
║  ┌─────────────────────────────────────┐  ║
║  │         ▂▆█▆▄▂                      │  ║
║  │      ▁▄██████▆▄▂▁                   │  ║  ← histogram (SVG)
║  │   ▁▃██████████▆▄▂▁                  │  ║  ← draggable cutoff line(s)
║  │________________|___________________│  ║
║  │   0           1.8                 9 │  ║
║  └─────────────────────────────────────┘  ║
║  Threshold: [ 1.80 ]   Matching: 1,247/8,432 cells
║                                           ║
║  Action:                                  ║
║   ◉ Update selection                      ║
║       ○ Replace  ● Add  ○ Intersect       ║
║   ○ Label cells                           ║
║       Annotation name: [ CD3E_above    ]  ║
║       ▸ More options                      ║
║                                           ║
║  [ Cancel ]              [ Apply ]        ║
╚═══════════════════════════════════════════╝
```

**Components:**

- **Mode selector** — segmented control with `Above` / `Below` / `Between`. Switching preserves the current cutoff value.
- **Histogram** — SVG, 60 linear bins. Y-axis auto-scales. One draggable vertical line for `Above`/`Below`, two for `Between`. Drag handles show hover tooltips with the numeric value.
- **Numeric threshold input(s)** — bidirectionally synced with the handles. Clamped to `[min, max]` of observed values. For `Between`, two inputs (`lo` and `hi`) shown side by side.
- **Live match counter** — `Matching: N / total` updates on every drag.
- **Action radio group** (mutually exclusive):
  - `Update selection` — reveals sub-radio `Replace` / `Add` / `Intersect`. Default sub-action is `Replace`. `Add` and `Intersect` are disabled with a tooltip when `selectedCellIndices.length === 0`.
  - `Label cells` — reveals an annotation name text input, default `<gene>_<mode>` (e.g. `CD3E_above`, `T_cell_markers_between`), and a `▸ More options` disclosure.
- **More options disclosure** (collapsed by default):
  - `High label` input (default `high`)
  - `Low label` input (default `low`)
  - `Context:` radio — `Current selection` (default when non-empty) / `All cells`
- **Buttons** — `Cancel` / `Apply`.

### On open

- If the scatter plot is not already coloring by this exact gene/gene set, the modal calls the parent's `onColorByGene(gene)` or `onColorBySet(genes, name)` handler (matching the existing signatures in `GenePanel.tsx`), which sets `colorMode='expression'` and populates `expressionData` through the existing flow.
- Histogram values are read from `expressionData.values` — no separate fetch. If `expressionData` already matches the source, nothing is refetched.
- Default threshold values:
  - `Above`: median of non-zero values (or 25th percentile if that's 0).
  - `Below`: same.
  - `Between`: 25th and 75th percentile.

### On Apply — Update selection

1. Compute `matching = indices of cells where expression satisfies threshold`.
2. Combine with `selectedCellIndices` based on sub-action:
   - `Replace`: `matching`
   - `Add`: union
   - `Intersect`: intersection
3. Call `store.setSelectedCellIndices(final)`.
4. Close the modal.

### On Apply — Label cells

1. Determine `context = selectedCellIndices` (if `Current selection` chosen and non-empty) or `all cell indices`.
2. Compute `matching = matching indices ∩ context`.
3. `low = context \ matching`.
4. If annotation name collides with an existing column, show inline error `"An annotation named '<name>' already exists — choose a different name"` and block Apply. Do not clobber.
5. Sequentially call `createAnnotation(name)`, `addLabelToAnnotation(name, highLabel)`, `addLabelToAnnotation(name, lowLabel)`, `labelCells(name, highLabel, matching)`, `labelCells(name, lowLabel, low)`.
6. On success, replace the modal footer with:
   `"Labeled 1,247 cells high, 3,218 cells low. [Open Diff Exp ▸]   [Close]"`
7. Clicking `Open Diff Exp ▸`:
   - Sets `store.pendingDiffExpGroups = { column: name, group1: highLabel, group2: lowLabel }`.
   - Closes this modal.
   - Opens `DiffExpModal` (via existing `setDiffExpModalOpen` store action).
   - `DiffExpModal` reads `pendingDiffExpGroups` in a mount effect, pre-fills its column/group1/group2 selectors, and calls `setPendingDiffExpGroups(null)` to consume the handoff.

### Edge cases

| Case | Handling |
|------|----------|
| Zero variance (all values equal) | Histogram renders a single bar; threshold handles snap to the value; inline notice `"All cells have the same value (X.XX) — nothing to threshold on"`; Apply disabled. |
| Gene not found / fetch error | Standard error toast via existing mechanism; modal closes. |
| Empty context on Label cells | Apply disabled with tooltip `"Current selection is empty — pick 'All cells' or make a selection first"`. |
| Between mode, crossed handles | Auto-swap to maintain `lo ≤ hi`. |
| Dataset switch while modal is open | Modal closes via effect watching `activeSlot`. |
| Display prefs change while modal is open (scoring method, expression transform) | Modal keeps the `expressionData` it opened with; user must reopen to pick up new settings. Documented as a known simplification. |
| Annotation name collision | Blocked with inline error; user must rename. |
| Gene set is empty | Entry menu item disabled with tooltip. |

## Implementation

### Architecture

Pure frontend thresholding. No new backend endpoints. Reuses:
- `GET /api/expression/{gene}` — single gene values.
- `POST /api/expression/multi` — gene set scored values.
- `POST /api/annotations` / `.../labels` / `.../label-cells` — existing annotation APIs.

### New files

- **`frontend/src/components/SelectByExpressionModal.tsx`**
  - Props: `{ source: ModalSource, onClose: () => void }` where `ModalSource = { type: 'gene', gene: string } | { type: 'geneSet', name: string, genes: string[] }`.
  - Local state: `mode`, `lo`, `hi`, `action`, `subAction`, `labelConfig` (annotation name, high/low labels, context), `moreOptionsOpen`, `applyStatus` (`idle` / `running` / `success` / `error`), `successResult`.
  - Memoized histogram computation from `expressionData.values`.
  - Effects:
    - On mount: trigger coloring if not already coloring by this source.
    - On `activeSlot` change: close.
    - On `applyStatus === 'success'` (label path): render success footer.

### Modified files

**`frontend/src/components/GenePanel.tsx`**
- Gene set row `OverflowMenu` (~line 937): append `{ label: 'Select cells…', onClick, disabled: geneSet.genes.length === 0 }`.
- Gene rows in search/browse results (~line 614–636): add a new `<OverflowMenu>` with single item `Select cells…`.
- Gene rows inside expanded gene sets (~line 962–989): same.
- Local state `selectByExpressionSource: ModalSource | null` at the `GenePanel` component level. Renders `<SelectByExpressionModal>` when non-null.
- New props threaded to sub-components where the gene rows live: `onOpenSelectByExpression(source)`.

**`frontend/src/store.ts`**
- New top-level (not per-dataset) field: `pendingDiffExpGroups: { column: string, group1: string, group2: string } | null`. Initial `null`.
- New action `setPendingDiffExpGroups(p: typeof pendingDiffExpGroups)`.
- Rationale: top-level because it's a transient UI handoff, not data-bound. Per-dataset is unnecessary since the modal workflow targets the active dataset anyway.

**`frontend/src/components/DiffExpModal.tsx`**
- `DiffExpModal` is always mounted and shows/hides based on `isDiffExpModalOpen`. Add a `useEffect` keyed on `isDiffExpModalOpen` that runs when the modal opens: if `pendingDiffExpGroups` is non-null, set the column selector to `column`, set group1/group2 selectors to the given labels, then call `setPendingDiffExpGroups(null)` to consume the handoff.
- No change to existing behavior when `pendingDiffExpGroups` is null.

**`frontend/src/messages.ts`**
- Add strings for:
  - Modal title variants (single gene vs gene set)
  - Zero-variance notice
  - Empty context tooltip
  - Annotation name collision error
  - Success footer template
  - Match counter template

### Not modified

- **`frontend/src/App.tsx`** — modal lives inside `GenePanel`.
- **Backend** — no changes. All required endpoints exist.

### Data flow

```
[user clicks ⋯ → Select cells…]
        │
        ▼
[GenePanel sets selectByExpressionSource]
        │
        ▼
[SelectByExpressionModal mounts]
        │
        ├── if not already coloring by this source → call onColorByGene / onColorByGeneSet
        │      (populates expressionData through existing flow)
        │
        ▼
[useMemo computes {binEdges, counts, min, max} from expressionData.values]
        │
        ▼
[User drags handles / edits numeric inputs → local threshold state updates]
        │
        ▼
[useMemo recomputes matching indices and match count]
        │
        ▼
[User clicks Apply]
        │
        ├── Update selection path:
        │      store.setSelectedCellIndices(final)
        │      onClose()
        │
        └── Label cells path:
               createAnnotation(name)
               addLabelToAnnotation(name, highLabel)
               addLabelToAnnotation(name, lowLabel)
               labelCells(name, highLabel, matching)
               labelCells(name, lowLabel, low)
               → show success footer
               → optional: setPendingDiffExpGroups(...); setDiffExpModalOpen(true); onClose()
```

### Histogram computation

Pure JS in a `useMemo` keyed on `expressionData.values`:

```ts
function computeHistogram(values: Float32Array | number[], nBins = 60) {
  let min = Infinity, max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  if (min === max) return { binEdges: [min], counts: [values.length], min, max, zeroVariance: true }
  const width = (max - min) / nBins
  const counts = new Array(nBins).fill(0)
  for (const v of values) {
    const idx = Math.min(nBins - 1, Math.floor((v - min) / width))
    counts[idx]++
  }
  const binEdges = Array.from({ length: nBins + 1 }, (_, i) => min + i * width)
  return { binEdges, counts, min, max, zeroVariance: false }
}
```

Complexity: O(N), a few ms even at 1M cells.

### Threshold → matching indices

```ts
function matchingIndices(
  values: Float32Array | number[],
  mode: 'above' | 'below' | 'between',
  lo: number,
  hi: number,
): number[] {
  const out: number[] = []
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (mode === 'above' && v >= lo) out.push(i)
    else if (mode === 'below' && v <= lo) out.push(i)
    else if (mode === 'between' && v >= lo && v <= hi) out.push(i)
  }
  return out
}
```

Recomputed in a `useMemo` on drag. For 1M cells, ~10–20 ms — noticeable but acceptable. Can later be optimized with a pre-sorted index if needed.

### Testing

Manual browser testing (no automated test infrastructure exists yet):

1. Load a dataset. Color by an obs column. Open `⋯ → Select cells…` on a gene in browse results. Confirm the plot switches to expression coloring.
2. Drag the histogram cutoff in `Above` mode. Confirm match counter updates live.
3. Click Apply with `Update selection → Replace`. Confirm the scatter plot shows the new selection.
4. Make a lasso selection. Open the modal on a gene set. Use `Update selection → Add`. Confirm the result is the union.
5. With an existing selection, use `Update selection → Intersect`. Confirm the result is the intersection.
6. With an existing lasso selection, use `Label cells` (default name). Confirm the success footer appears with counts. Click `Open Diff Exp ▸`. Confirm `DiffExpModal` opens with the new annotation column and `high`/`low` as group1/group2.
7. Try to label with a name that collides with an existing column. Confirm the error blocks Apply.
8. Open on a gene with zero variance (e.g. a constant or absent gene). Confirm the zero-variance notice and disabled Apply.
9. Switch datasets while the modal is open. Confirm it closes.
10. `npm run build` from `xcell/frontend/` passes cleanly.

## Documentation updates (required per CLAUDE.md)

- **`xcell/CLAUDE.md`**
  - Add `SelectByExpressionModal.tsx` to the Components table.
  - Add a Key Behaviors entry: "Select cells by expression" — summary of modal workflow, entry points, and diff-exp handoff.
  - Add `pendingDiffExpGroups` to Store Types.
- **`xcell/CHANGELOG.md`**
  - Under `[Unreleased] → Added`: "Select cells by expression threshold from gene and gene set overflow menus, with interactive histogram cutoff and optional labeling with direct handoff to differential expression."
- **`xcell/README.md`**
  - Add a short walkthrough subsection (under the analysis workflows section) showing the gene → threshold → label → diff exp flow.

## Out of scope

- Feature 2 — mask genes in the Gene Panel by `.var` boolean columns (`highly_variable`, spatially variable, etc.). Separate spec to follow.
- Bivariate thresholding (two genes together).
- Multi-cutoff (partitioning into more than 2 groups).
- Backend changes beyond existing endpoints.
- Automated tests.

## Open questions

None — all design decisions settled in brainstorming.
