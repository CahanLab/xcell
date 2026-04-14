# Select Cells by Expression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users threshold on a gene's expression or a gene set score via an interactive histogram inside a new modal, with outputs to either update the cell selection or label cells into `high`/`low` groups with direct handoff to differential expression.

**Architecture:** Pure frontend thresholding. A new `SelectByExpressionModal` component reads expression values from the existing `expressionData` store field (re-used from the scatter plot's expression-coloring path). Histogram bins and matching indices are computed in JS. "Label cells" calls existing `createAnnotation` / `addLabelToAnnotation` / `labelCells` APIs. Diff-exp handoff reuses existing `setComparisonGroup1` / `setComparisonGroup2` / `setDiffExpModalOpen` store actions. No backend changes.

**Tech Stack:** React 18 + TypeScript + Zustand 4 (frontend only). SVG for the histogram. No new dependencies.

**Repository note:** The working root is `/Users/pcahan/code/xcell/`. The active project is `xcell/`. All file paths below are relative to the git root at `xcell/`. Run commands from that directory.

**Spec:** `docs/superpowers/specs/2026-04-14-select-cells-by-expression-design.md`

---

## File Map

**New files:**
- `frontend/src/components/SelectByExpressionModal.tsx` — the modal component. Self-contained. Reads source from store, renders null when source is null.

**Modified files:**
- `frontend/src/store.ts` — add `SelectByExpressionSource` type, `selectByExpressionSource` field, `setSelectByExpressionSource` action.
- `frontend/src/components/GenePanel.tsx` — add `⋯` menus on gene rows (browse/search + inside expanded gene sets) and add `Select cells…` item to the existing gene-set-row `OverflowMenu`.
- `frontend/src/App.tsx` — render `<SelectByExpressionModal />` once near the other always-mounted modals.
- `frontend/src/messages.ts` — add UI strings for the modal.
- `frontend/CHANGELOG.md` *(project root `CHANGELOG.md`)* — `[Unreleased] → Added` entry.
- `CLAUDE.md` — add to Components table and Key Behaviors.
- `README.md` — add walkthrough subsection.

**Unchanged (important):**
- Backend — no new routes.
- `frontend/src/components/DiffExpModal.tsx` — already reads `comparison.*` from the store; pre-populated groups are picked up automatically.
- `frontend/src/hooks/useData.ts` — all needed APIs (`createAnnotation`, `addLabelToAnnotation`, `labelCells`, `colorByGene`, `colorByGenes` via `useDataActions`) already exist.

---

### Task 1: Add store field for modal source

**Goal:** Create the store plumbing for opening/closing the modal. The modal will read `selectByExpressionSource` and render nothing when it is `null`. Callers (gene rows and gene set rows) will call `setSelectByExpressionSource(...)` to open it.

**Files:**
- Modify: `frontend/src/store.ts`

- [ ] **Step 1: Add the source type and field to `AppState`**

Open `frontend/src/store.ts`. Find the section around line 135 where `ComparisonState` is defined. Below the existing type definitions (`ComparisonState`, `DatasetSlot`, etc.) but above the `DatasetState` interface, add:

```ts
// Source for the "Select cells by expression" modal.
// `null` means the modal is closed. Mirrors the ClusterGeneSetModal pattern.
export type SelectByExpressionSource =
  | { type: 'gene'; gene: string }
  | { type: 'geneSet'; name: string; genes: string[] }
```

Find the top-level `AppState` interface (around line 353). Find the existing field `clusterModalSourceSet` (search for it — it was added in an earlier feature). Immediately after its line, add:

```ts
  selectByExpressionSource: SelectByExpressionSource | null
```

Find the actions section of `AppState` (further down). Search for `setClusterModalSourceSet` (the matching setter). Immediately after its line, add:

```ts
  setSelectByExpressionSource: (src: SelectByExpressionSource | null) => void
```

- [ ] **Step 2: Add the field's initial value and the setter implementation**

In the same file, find the `create<AppState>(...)` body. Search for `clusterModalSourceSet: null,` in the initial state object. Add immediately after it:

```ts
    selectByExpressionSource: null,
```

Find the setter implementation for `setClusterModalSourceSet` (search: `setClusterModalSourceSet: (src) => set(`). Add immediately after its closing line:

```ts
    setSelectByExpressionSource: (src) => set({ selectByExpressionSource: src }),
```

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors. No behavior change yet.

- [ ] **Step 4: Commit**

```
git add frontend/src/store.ts
git commit -m "feat: add selectByExpressionSource store field

Plumbs the source for the upcoming Select-cells-by-expression modal.
Null when closed. Follows the clusterModalSourceSet pattern."
```

---

### Task 2: Scaffold the modal shell and mount it in App.tsx

**Goal:** Create `SelectByExpressionModal.tsx` as a minimal shell: modal chrome, a title derived from the source, a `Cancel` button that calls `setSelectByExpressionSource(null)`. Mount it once at the `App` level.

**Files:**
- Create: `frontend/src/components/SelectByExpressionModal.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create the modal shell file**

Write `frontend/src/components/SelectByExpressionModal.tsx`:

```tsx
import { useEffect } from 'react'
import { useStore } from '../store'

export default function SelectByExpressionModal() {
  const source = useStore((s) => s.selectByExpressionSource)
  const setSource = useStore((s) => s.setSelectByExpressionSource)

  // Close on Escape
  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSource(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [source, setSource])

  if (!source) return null

  const title =
    source.type === 'gene'
      ? `Select cells by ${source.gene}`
      : `Select cells by ${source.name}`

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={() => setSource(null)}
    >
      <div
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '6px',
          padding: '16px 20px',
          width: '520px',
          maxWidth: '95vw',
          color: '#ccc',
          fontSize: '12px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            fontSize: '14px',
            fontWeight: 600,
            marginBottom: '12px',
            color: '#e94560',
          }}
        >
          {title}
        </div>

        <div style={{ color: '#888', padding: '24px 0' }}>
          (Modal under construction)
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button
            style={{
              padding: '6px 14px',
              backgroundColor: '#0f3460',
              color: '#ccc',
              border: '1px solid #0f3460',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '11px',
            }}
            onClick={() => setSource(null)}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Mount the modal in App.tsx**

Open `frontend/src/App.tsx`. Find the import for `ClusterGeneSetModal` (near line 14):

```tsx
import ClusterGeneSetModal from './components/ClusterGeneSetModal'
```

Add immediately after it:

```tsx
import SelectByExpressionModal from './components/SelectByExpressionModal'
```

Find the render site for `<ClusterGeneSetModal />` (near line 1694):

```tsx
      <ClusterGeneSetModal />
```

Add immediately after it:

```tsx
      <SelectByExpressionModal />
```

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors. The modal is mounted but invisible (nothing sets the source yet).

- [ ] **Step 4: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx frontend/src/App.tsx
git commit -m "feat: scaffold SelectByExpressionModal shell

Empty modal with title and Cancel button. Reads source from the store
and renders null when closed. Mounted in App alongside other modals."
```

---

### Task 3: Wire entry points from the Gene Panel

**Goal:** Add a `Select cells…` item to the existing gene-set-row `OverflowMenu`, and add new `⋯` overflow menus to individual gene rows (both in search/browse results and inside expanded gene sets). Clicking any of them sets `selectByExpressionSource`, opening the shell from Task 2.

**Files:**
- Modify: `frontend/src/components/GenePanel.tsx`

- [ ] **Step 1: Add the store setter to the gene set row component**

Open `frontend/src/components/GenePanel.tsx`. Search for the gene set row component that contains the `OverflowMenu` — search for the exact line `label: 'Cluster genes…',` (around line 944). Scroll up to find the enclosing function (search upwards for `function GeneSetRow` or similar — there is a component that renders an individual gene set). Near the top of that component, where store access already happens (look for existing `useStore` calls), add:

```tsx
  const setSelectByExpressionSource = useStore((s) => s.setSelectByExpressionSource)
```

If there is no existing `useStore` import yet in this sub-component, note that the file already imports `useStore` from `'../store'` at the top — just add the hook call.

- [ ] **Step 2: Add `Select cells…` to the gene set row `OverflowMenu`**

Find the `items` array for the gene set row's `<OverflowMenu items={[...]}>` (contains `label: 'Cluster genes…',`). After the `Cluster genes…` item and before the closing `]`, add:

```tsx
              {
                label: 'Select cells…',
                onClick: () =>
                  setSelectByExpressionSource({
                    type: 'geneSet',
                    name: geneSet.name,
                    genes: geneSet.genes,
                  }),
                disabled: geneSet.genes.length === 0,
                tooltip: geneSet.genes.length === 0 ? 'Gene set is empty' : undefined,
              },
```

- [ ] **Step 3: Add a `⋯` menu to gene rows in search/browse results**

Find the gene row rendering in the `GeneSearch` component (around line 607–636). Locate this block:

```tsx
              <div
                key={gene}
                draggable
                onDragStart={(e) => handleDragStart(e, gene)}
                ...
              >
                <input
                  type="checkbox"
                  ...
                />
                <span
                  style={{ flex: 1, cursor: 'pointer' }}
                  onClick={() => onColorByGene(gene)}
                  title="Click to color by expression"
                >
                  {gene}
                </span>
              </div>
```

Replace the trailing `</span>` and the closing `</div>` region to add an `<OverflowMenu>`:

```tsx
                <span
                  style={{ flex: 1, cursor: 'pointer' }}
                  onClick={() => onColorByGene(gene)}
                  title="Click to color by expression"
                >
                  {gene}
                </span>
                <OverflowMenu
                  items={[
                    {
                      label: 'Select cells…',
                      onClick: () => setSelectByExpressionSource({ type: 'gene', gene }),
                    },
                  ]}
                />
              </div>
```

Now `setSelectByExpressionSource` needs to be available inside `GeneSearch`. Near the top of the `GeneSearch` function body (search: `function GeneSearch(`), add:

```tsx
  const setSelectByExpressionSource = useStore((s) => s.setSelectByExpressionSource)
```

The `useStore` import is already present at the top of the file. `OverflowMenu` is defined in the same file and is in scope — no import needed.

- [ ] **Step 4: Add a `⋯` menu to gene rows inside expanded gene sets**

Still in `GenePanel.tsx`, find the gene rendering inside the expanded gene set body (around line 962–989). The block currently looks like:

```tsx
              <span style={styles.geneName} onClick={() => onColorByGene(gene)} title="Click to color by expression">
                {gene}
              </span>
              <button
                style={{ ...styles.iconButton, fontSize: '10px' }}
                onClick={() => handleRemoveGene(gene)}
                title="Remove from set"
              >
                ✕
              </button>
```

Replace that region with:

```tsx
              <span style={styles.geneName} onClick={() => onColorByGene(gene)} title="Click to color by expression">
                {gene}
              </span>
              <OverflowMenu
                items={[
                  {
                    label: 'Select cells…',
                    onClick: () => setSelectByExpressionSource({ type: 'gene', gene }),
                  },
                ]}
              />
              <button
                style={{ ...styles.iconButton, fontSize: '10px' }}
                onClick={() => handleRemoveGene(gene)}
                title="Remove from set"
              >
                ✕
              </button>
```

The enclosing component already reads `setSelectByExpressionSource` (added in Step 1), so no additional hook call needed here.

- [ ] **Step 5: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 6: Manual smoke test**

Start the backend and frontend dev servers (from `backend/`: `uvicorn xcell.main:app --reload` ; from `frontend/`: `npm run dev`). Open the browser at `http://localhost:5173`. Click a gene in the browse/search list's `⋯` menu and pick `Select cells…`. Confirm the empty modal shell opens. Cancel it. Do the same from a gene set row's `⋯` menu and from a gene row inside an expanded gene set.

- [ ] **Step 7: Commit**

```
git add frontend/src/components/GenePanel.tsx
git commit -m "feat: wire Select cells entry points in Gene Panel

Adds 'Select cells…' menu item on gene set rows, and new ⋯ overflow
menus on individual gene rows (both in search/browse results and
inside expanded gene sets). Opens the empty modal shell."
```

---

### Task 4: Auto-color on open and histogram utility functions

**Goal:** (a) When the modal opens, trigger expression coloring for the source if the plot isn't already coloring by exactly this source, so the histogram and scatter plot are coordinated. (b) Add pure utility functions for histogram computation and matching indices. These are used by the next tasks.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add histogram and matching-index utility functions**

At the top of `frontend/src/components/SelectByExpressionModal.tsx`, above the `export default function SelectByExpressionModal`, add:

```tsx
// ---------------------------------------------------------------------------
// Pure helpers — histogram binning and threshold-to-indices.
// Exported for future unit tests; not used outside this file currently.
// ---------------------------------------------------------------------------

export type ThresholdMode = 'above' | 'below' | 'between'

export interface Histogram {
  binEdges: number[]
  counts: number[]
  min: number
  max: number
  zeroVariance: boolean
}

export function computeHistogram(values: Float32Array | number[], nBins = 60): Histogram {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!isFinite(min) || !isFinite(max)) {
    return { binEdges: [0, 1], counts: new Array(nBins).fill(0), min: 0, max: 1, zeroVariance: true }
  }
  if (min === max) {
    return { binEdges: [min, min], counts: [values.length], min, max, zeroVariance: true }
  }
  const width = (max - min) / nBins
  const counts = new Array(nBins).fill(0)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    const idx = Math.min(nBins - 1, Math.floor((v - min) / width))
    counts[idx]++
  }
  const binEdges: number[] = new Array(nBins + 1)
  for (let i = 0; i <= nBins; i++) binEdges[i] = min + i * width
  return { binEdges, counts, min, max, zeroVariance: false }
}

export function matchingIndices(
  values: Float32Array | number[],
  mode: ThresholdMode,
  lo: number,
  hi: number
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

// Pick a sensible default threshold for a given mode from observed values.
// For Above/Below: median of non-zero values, falling back to the 25th percentile
// and then to the midpoint of [min, max]. For Between: 25th and 75th percentile.
export function defaultThresholds(
  values: Float32Array | number[],
  mode: ThresholdMode,
  min: number,
  max: number
): { lo: number; hi: number } {
  if (values.length === 0 || min === max) {
    return { lo: min, hi: max }
  }
  const sorted = Array.from(values).sort((a, b) => a - b)
  const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)))]
  if (mode === 'between') {
    return { lo: percentile(0.25), hi: percentile(0.75) }
  }
  // above/below
  const nonZero = sorted.filter((v) => v > 0)
  if (nonZero.length > 0) {
    const median = nonZero[Math.floor(nonZero.length / 2)]
    return { lo: median, hi: median }
  }
  const mid = (min + max) / 2
  return { lo: mid, hi: mid }
}
```

- [ ] **Step 2: Wire auto-color on open**

Update `SelectByExpressionModal`'s component body to also trigger coloring and read `expressionData`. Replace the existing component body with:

```tsx
import { useEffect, useMemo } from 'react'
import { useStore } from '../store'
import { useDataActions } from '../hooks/useData'

// (keep the utility functions above)

export default function SelectByExpressionModal() {
  const source = useStore((s) => s.selectByExpressionSource)
  const setSource = useStore((s) => s.setSelectByExpressionSource)
  const expressionData = useStore((s) => s.expressionData)
  const selectedGenes = useStore((s) => s.selectedGenes)
  const selectedGeneSetName = useStore((s) => s.selectedGeneSetName)
  const colorMode = useStore((s) => s.colorMode)
  const { colorByGene, colorByGenes } = useDataActions()

  // Close on Escape
  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSource(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [source, setSource])

  // Auto-color the plot when the modal opens, if it isn't already coloring by this source.
  useEffect(() => {
    if (!source) return
    if (source.type === 'gene') {
      const alreadyColoring =
        colorMode === 'expression' &&
        selectedGenes.length === 1 &&
        selectedGenes[0] === source.gene &&
        selectedGeneSetName === null
      if (!alreadyColoring) {
        colorByGene(source.gene)
      }
    } else {
      const alreadyColoring =
        colorMode === 'expression' &&
        selectedGeneSetName === source.name &&
        selectedGenes.length === source.genes.length
      if (!alreadyColoring) {
        colorByGenes(source.genes, undefined, source.name)
      }
    }
    // Deliberately only run when `source` identity changes — we don't want
    // to re-fetch on every store tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Histogram is memoized on expressionData.values identity.
  const histogram = useMemo(() => {
    if (!expressionData) return null
    return computeHistogram(expressionData.values)
  }, [expressionData])

  if (!source) return null

  const title =
    source.type === 'gene'
      ? `Select cells by ${source.gene}`
      : `Select cells by ${source.name}`

  const body = !expressionData
    ? <div style={{ color: '#888', padding: '24px 0' }}>Loading expression values…</div>
    : histogram?.zeroVariance
      ? (
        <div style={{ color: '#e94560', padding: '24px 0' }}>
          All cells have the same value ({histogram.min.toFixed(2)}) — nothing to threshold on.
        </div>
      )
      : <div style={{ color: '#888', padding: '24px 0' }}>
          Histogram ready. ({histogram?.counts.length} bins, range {histogram?.min.toFixed(2)}–{histogram?.max.toFixed(2)})
        </div>

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={() => setSource(null)}
    >
      <div
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '6px',
          padding: '16px 20px',
          width: '520px',
          maxWidth: '95vw',
          color: '#ccc',
          fontSize: '12px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#e94560' }}>
          {title}
        </div>
        {body}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button
            style={{
              padding: '6px 14px',
              backgroundColor: '#0f3460',
              color: '#ccc',
              border: '1px solid #0f3460',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '11px',
            }}
            onClick={() => setSource(null)}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
```

Note: make sure the `computeHistogram` / `matchingIndices` / `defaultThresholds` / type definitions from Step 1 remain in the file above the component. The `import` lines at the top must include `useEffect, useMemo` from `'react'`, `useStore` from `'../store'`, and `useDataActions` from `'../hooks/useData'`.

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 4: Manual smoke test**

With dev servers running, open any gene from the Gene Panel's search list via `⋯ → Select cells…`. Expected behavior: the scatter plot switches to expression coloring for that gene, and the modal shows either "Histogram ready. (60 bins, range …)" or the zero-variance notice. Cancel and repeat for a gene set row.

- [ ] **Step 5: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: auto-color on modal open and histogram utilities

SelectByExpressionModal now triggers colorByGene/colorByGenes on open
when not already coloring by the requested source, and memoizes a
histogram of the expression values. No UI for thresholding yet."
```

---

### Task 5: Histogram SVG rendering (static)

**Goal:** Replace the placeholder "Histogram ready." text with an actual SVG bar chart showing the binned counts. No interaction yet — no drag, no mode selector.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add a sub-component `HistogramChart`**

In `frontend/src/components/SelectByExpressionModal.tsx`, above `export default function SelectByExpressionModal`, add:

```tsx
const CHART_WIDTH = 460
const CHART_HEIGHT = 140
const CHART_PADDING = { top: 6, right: 6, bottom: 22, left: 6 }

function HistogramChart({ histogram }: { histogram: Histogram }) {
  const innerW = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
  const innerH = CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom
  const maxCount = Math.max(...histogram.counts, 1)
  const nBins = histogram.counts.length
  const barW = innerW / nBins

  return (
    <svg width={CHART_WIDTH} height={CHART_HEIGHT} style={{ display: 'block' }}>
      {/* Background */}
      <rect
        x={CHART_PADDING.left}
        y={CHART_PADDING.top}
        width={innerW}
        height={innerH}
        fill="#0f1625"
      />
      {/* Bars */}
      {histogram.counts.map((count, i) => {
        const h = (count / maxCount) * innerH
        return (
          <rect
            key={i}
            x={CHART_PADDING.left + i * barW}
            y={CHART_PADDING.top + innerH - h}
            width={Math.max(1, barW - 1)}
            height={h}
            fill="#4ecdc4"
          />
        )
      })}
      {/* X axis labels: min and max */}
      <text
        x={CHART_PADDING.left}
        y={CHART_HEIGHT - 6}
        fill="#888"
        fontSize="10"
      >
        {histogram.min.toFixed(2)}
      </text>
      <text
        x={CHART_PADDING.left + innerW}
        y={CHART_HEIGHT - 6}
        fill="#888"
        fontSize="10"
        textAnchor="end"
      >
        {histogram.max.toFixed(2)}
      </text>
    </svg>
  )
}
```

- [ ] **Step 2: Render the chart in the modal body**

In the component body, replace the placeholder body expression (`const body = ...`) with:

```tsx
  let body: React.ReactNode
  if (!expressionData || !histogram) {
    body = <div style={{ color: '#888', padding: '24px 0' }}>Loading expression values…</div>
  } else if (histogram.zeroVariance) {
    body = (
      <div style={{ color: '#e94560', padding: '24px 0' }}>
        All cells have the same value ({histogram.min.toFixed(2)}) — nothing to threshold on.
      </div>
    )
  } else {
    body = (
      <div style={{ padding: '8px 0' }}>
        <HistogramChart histogram={histogram} />
      </div>
    )
  }
```

Add the `React` type import at the top if it's not already there: change `import { useEffect, useMemo } from 'react'` to `import React, { useEffect, useMemo } from 'react'` (or add a separate `import type * as React from 'react'` if the project style prefers that — the default import form works because `React.ReactNode` is used).

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 4: Manual smoke test**

Open the modal on a gene that has variation. Confirm the histogram bars render and the x-axis shows the min and max values.

- [ ] **Step 5: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: render histogram bars in SelectByExpressionModal

Static SVG bar chart of expression value bins, no interaction yet."
```

---

### Task 6: Mode selector + threshold state + numeric inputs + match counter

**Goal:** Add the `Above` / `Below` / `Between` mode selector and the `lo` / `hi` threshold state. Show numeric input(s) that reflect the current thresholds, and a live match counter `Matching: N / total cells`. No drag interaction yet — users can only adjust thresholds via the numeric inputs.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add mode and threshold state**

In the `SelectByExpressionModal` component body, after the existing `const histogram = useMemo(...)` line, add:

```tsx
  const [mode, setMode] = useState<ThresholdMode>('above')
  const [lo, setLo] = useState<number>(0)
  const [hi, setHi] = useState<number>(0)

  // When the histogram becomes available for a new source, reset mode/lo/hi
  // to sensible defaults based on the actual value distribution.
  useEffect(() => {
    if (!histogram || histogram.zeroVariance) return
    const defaults = defaultThresholds(
      expressionData?.values ?? [],
      mode,
      histogram.min,
      histogram.max
    )
    setLo(defaults.lo)
    setHi(defaults.hi)
    // Only when the histogram identity changes (i.e. new expressionData for a new source)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [histogram])
```

Also update the `import` at the top to include `useState`:

```tsx
import React, { useEffect, useMemo, useState } from 'react'
```

- [ ] **Step 2: Compute matching count as a memo**

Immediately after the mode/threshold state block, add:

```tsx
  const matchCount = useMemo(() => {
    if (!expressionData || !histogram || histogram.zeroVariance) return 0
    // Fast count without building the indices array.
    const values = expressionData.values
    let count = 0
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (mode === 'above' && v >= lo) count++
      else if (mode === 'below' && v <= lo) count++
      else if (mode === 'between' && v >= lo && v <= hi) count++
    }
    return count
  }, [expressionData, histogram, mode, lo, hi])
```

- [ ] **Step 3: Render the mode selector, numeric inputs, and match counter**

Replace the `HistogramChart` render site. Change the `body` expression's `else` branch:

```tsx
  } else {
    body = (
      <div style={{ padding: '8px 0' }}>
        {/* Mode selector */}
        <div style={{ display: 'flex', gap: '4px', marginBottom: '8px', alignItems: 'center' }}>
          <span style={{ color: '#888', marginRight: '4px' }}>Threshold mode:</span>
          {(['above', 'below', 'between'] as ThresholdMode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m)
                const d = defaultThresholds(expressionData.values, m, histogram.min, histogram.max)
                setLo(d.lo)
                setHi(d.hi)
              }}
              style={{
                padding: '4px 10px',
                backgroundColor: mode === m ? '#4ecdc4' : '#0f3460',
                color: mode === m ? '#16213e' : '#ccc',
                border: '1px solid #0f3460',
                borderRadius: '3px',
                cursor: 'pointer',
                fontSize: '11px',
                textTransform: 'capitalize',
              }}
            >
              {m}
            </button>
          ))}
        </div>

        <HistogramChart histogram={histogram} />

        {/* Numeric inputs + match counter */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' }}>
          {mode === 'between' ? (
            <>
              <label style={{ color: '#888' }}>
                Lo:{' '}
                <input
                  type="number"
                  value={lo}
                  step="0.01"
                  onChange={(e) => setLo(Number(e.target.value))}
                  style={{
                    width: '70px',
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
              <label style={{ color: '#888' }}>
                Hi:{' '}
                <input
                  type="number"
                  value={hi}
                  step="0.01"
                  onChange={(e) => setHi(Number(e.target.value))}
                  style={{
                    width: '70px',
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
            </>
          ) : (
            <label style={{ color: '#888' }}>
              Threshold:{' '}
              <input
                type="number"
                value={lo}
                step="0.01"
                onChange={(e) => setLo(Number(e.target.value))}
                style={{
                  width: '70px',
                  backgroundColor: '#0f1625',
                  border: '1px solid #0f3460',
                  color: '#ccc',
                  padding: '3px 6px',
                  fontSize: '11px',
                }}
              />
            </label>
          )}
          <span style={{ marginLeft: 'auto', color: '#4ecdc4' }}>
            Matching: {matchCount.toLocaleString()} / {expressionData.values.length.toLocaleString()} cells
          </span>
        </div>
      </div>
    )
  }
```

- [ ] **Step 4: Auto-swap handles for Between mode**

Below the `matchCount` memo, add a small effect to keep `lo <= hi` when in Between mode:

```tsx
  useEffect(() => {
    if (mode === 'between' && lo > hi) {
      setLo(hi)
      setHi(lo)
    }
  }, [mode, lo, hi])
```

- [ ] **Step 5: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 6: Manual smoke test**

Open the modal on a gene with variation. Confirm:
- Three mode buttons are shown; clicking switches the UI between one and two inputs.
- Editing a numeric input updates the match counter in real time.
- In Between mode, typing a `Lo` value greater than `Hi` causes them to swap.

- [ ] **Step 7: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: add mode selector, numeric threshold inputs, match counter

Above/Below/Between mode toggle, bidirectional numeric inputs, and a
live match counter. No drag interaction yet."
```

---

### Task 7: Interactive drag handles on the histogram

**Goal:** Make the histogram show vertical cutoff line(s) that the user can drag with the mouse. Dragging updates `lo` / `hi` state, which flows back to both the numeric inputs and the match counter.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Extend `HistogramChart` to render and drag cutoff lines**

Replace the existing `HistogramChart` definition with:

```tsx
function HistogramChart({
  histogram,
  mode,
  lo,
  hi,
  onChangeLo,
  onChangeHi,
}: {
  histogram: Histogram
  mode: ThresholdMode
  lo: number
  hi: number
  onChangeLo: (v: number) => void
  onChangeHi: (v: number) => void
}) {
  const innerW = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
  const innerH = CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom
  const maxCount = Math.max(...histogram.counts, 1)
  const nBins = histogram.counts.length
  const barW = innerW / nBins
  const svgRef = React.useRef<SVGSVGElement | null>(null)

  const valueToX = (v: number) => {
    const clamped = Math.max(histogram.min, Math.min(histogram.max, v))
    const frac = (clamped - histogram.min) / (histogram.max - histogram.min || 1)
    return CHART_PADDING.left + frac * innerW
  }

  const xToValue = (x: number) => {
    const localX = Math.max(0, Math.min(innerW, x - CHART_PADDING.left))
    const frac = localX / innerW
    return histogram.min + frac * (histogram.max - histogram.min)
  }

  const startDrag = (which: 'lo' | 'hi') => (e: React.MouseEvent<SVGRectElement>) => {
    e.preventDefault()
    const svg = svgRef.current
    if (!svg) return
    const onMove = (ev: MouseEvent) => {
      const rect = svg.getBoundingClientRect()
      const x = ev.clientX - rect.left
      const v = xToValue(x)
      if (which === 'lo') onChangeLo(v)
      else onChangeHi(v)
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }

  const loX = valueToX(lo)
  const hiX = valueToX(hi)

  return (
    <svg ref={svgRef} width={CHART_WIDTH} height={CHART_HEIGHT} style={{ display: 'block' }}>
      {/* Background */}
      <rect
        x={CHART_PADDING.left}
        y={CHART_PADDING.top}
        width={innerW}
        height={innerH}
        fill="#0f1625"
      />
      {/* Bars */}
      {histogram.counts.map((count, i) => {
        const h = (count / maxCount) * innerH
        return (
          <rect
            key={i}
            x={CHART_PADDING.left + i * barW}
            y={CHART_PADDING.top + innerH - h}
            width={Math.max(1, barW - 1)}
            height={h}
            fill="#4ecdc4"
          />
        )
      })}

      {/* Cutoff line(s) */}
      {mode === 'above' && (
        <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />
      )}
      {mode === 'below' && (
        <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />
      )}
      {mode === 'between' && (
        <>
          <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />
          <CutoffLine x={hiX} innerH={innerH} onMouseDown={startDrag('hi')} />
        </>
      )}

      {/* X axis labels */}
      <text x={CHART_PADDING.left} y={CHART_HEIGHT - 6} fill="#888" fontSize="10">
        {histogram.min.toFixed(2)}
      </text>
      <text
        x={CHART_PADDING.left + innerW}
        y={CHART_HEIGHT - 6}
        fill="#888"
        fontSize="10"
        textAnchor="end"
      >
        {histogram.max.toFixed(2)}
      </text>
    </svg>
  )
}

function CutoffLine({
  x,
  innerH,
  onMouseDown,
}: {
  x: number
  innerH: number
  onMouseDown: (e: React.MouseEvent<SVGRectElement>) => void
}) {
  return (
    <g>
      {/* Visible line */}
      <line
        x1={x}
        x2={x}
        y1={CHART_PADDING.top}
        y2={CHART_PADDING.top + innerH}
        stroke="#e94560"
        strokeWidth={2}
      />
      {/* Wide invisible hit-target for drag */}
      <rect
        x={x - 6}
        y={CHART_PADDING.top}
        width={12}
        height={innerH}
        fill="transparent"
        style={{ cursor: 'ew-resize' }}
        onMouseDown={onMouseDown}
      />
    </g>
  )
}
```

- [ ] **Step 2: Pass mode and threshold state to `HistogramChart`**

Find the `<HistogramChart histogram={histogram} />` call in the body expression and replace with:

```tsx
        <HistogramChart
          histogram={histogram}
          mode={mode}
          lo={lo}
          hi={hi}
          onChangeLo={setLo}
          onChangeHi={setHi}
        />
```

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 4: Manual smoke test**

Open the modal on a gene. Drag the red cutoff line horizontally. Confirm the match counter and numeric input update in real time. Switch to Between mode and confirm two draggable lines appear.

- [ ] **Step 5: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: drag handles for histogram cutoff lines

Click and drag red cutoff line(s) to adjust threshold in real time.
Between mode shows two handles."
```

---

### Task 8: Update-selection action path with Apply

**Goal:** Add the `Update selection` action and sub-action (`Replace` / `Add` / `Intersect`) radio groups. On Apply, compute `matchingIndices` and call `setSelectedCellIndices`. `Add` and `Intersect` are disabled when there is no existing selection.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add action state and store reads**

In the component body, after the existing `useState` calls, add:

```tsx
  type Action = 'updateSelection' | 'labelCells'
  type SubAction = 'replace' | 'add' | 'intersect'

  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const setSelectedCellIndices = useStore((s) => s.setSelectedCellIndices)

  const [action, setAction] = useState<Action>('updateSelection')
  const [subAction, setSubAction] = useState<SubAction>('replace')
```

Verify that `setSelectedCellIndices` is an action on the store. Search `frontend/src/store.ts` for `setSelectedCellIndices:` to confirm — it is an existing top-level action.

- [ ] **Step 2: Add the action UI below the numeric inputs**

In the body's `else` branch (the non-zero-variance case), add below the numeric inputs `<div>` and before the closing `</div>` of the body wrapper:

```tsx
        {/* Action selector */}
        <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid #0f3460' }}>
          <div style={{ color: '#888', marginBottom: '6px' }}>Action:</div>
          <label style={{ display: 'block', marginBottom: '4px', color: '#ccc' }}>
            <input
              type="radio"
              checked={action === 'updateSelection'}
              onChange={() => setAction('updateSelection')}
            />{' '}
            Update selection
          </label>
          {action === 'updateSelection' && (
            <div style={{ paddingLeft: '22px', display: 'flex', gap: '12px' }}>
              {(['replace', 'add', 'intersect'] as SubAction[]).map((sa) => {
                const disabled = sa !== 'replace' && selectedCellIndices.length === 0
                return (
                  <label
                    key={sa}
                    style={{ color: disabled ? '#555' : '#ccc', textTransform: 'capitalize' }}
                    title={disabled ? 'No existing selection' : undefined}
                  >
                    <input
                      type="radio"
                      checked={subAction === sa}
                      disabled={disabled}
                      onChange={() => setSubAction(sa)}
                    />{' '}
                    {sa}
                  </label>
                )
              })}
            </div>
          )}
        </div>
```

Also, when the existing selection goes empty, force-reset `subAction` to `replace`:

```tsx
  useEffect(() => {
    if (selectedCellIndices.length === 0 && subAction !== 'replace') {
      setSubAction('replace')
    }
  }, [selectedCellIndices, subAction])
```

- [ ] **Step 3: Add an Apply button and wire the Update-selection path**

Replace the footer button row with:

```tsx
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '14px' }}>
          <button
            style={{
              padding: '6px 14px',
              backgroundColor: '#0f3460',
              color: '#ccc',
              border: '1px solid #0f3460',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '11px',
            }}
            onClick={() => setSource(null)}
          >
            Cancel
          </button>
          <button
            style={{
              padding: '6px 14px',
              backgroundColor: '#4ecdc4',
              color: '#16213e',
              border: '1px solid #4ecdc4',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 600,
            }}
            onClick={handleApply}
          >
            Apply
          </button>
        </div>
```

And add a `handleApply` callback in the component body (above the return):

```tsx
  const handleApply = () => {
    if (!expressionData) return
    const matching = matchingIndices(expressionData.values, mode, lo, hi)

    if (action === 'updateSelection') {
      let final: number[]
      if (subAction === 'replace') {
        final = matching
      } else if (subAction === 'add') {
        const existing = new Set(selectedCellIndices)
        for (const i of matching) existing.add(i)
        final = Array.from(existing)
      } else {
        // intersect
        const matchingSet = new Set(matching)
        final = selectedCellIndices.filter((i) => matchingSet.has(i))
      }
      setSelectedCellIndices(final)
      setSource(null)
      return
    }

    // Label cells branch — implemented in Task 9.
  }
```

- [ ] **Step 4: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 5: Manual smoke test**

1. Open the modal on a gene. Pick Above, drag the cutoff, click Apply. Confirm the scatter plot shows the selection (cells above the threshold).
2. Lasso some cells. Open the modal. Pick Add. Apply. Confirm the selection expands (union).
3. Lasso a different region. Open the modal. Pick Intersect. Apply. Confirm the selection shrinks to the intersection.
4. With no prior selection, confirm Add and Intersect are disabled and greyed.

- [ ] **Step 6: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: Update selection action with Replace/Add/Intersect

Apply button computes matching indices and updates selectedCellIndices.
Add/Intersect disabled when no existing selection."
```

---

### Task 9: Label-cells action path (default UI, no More options)

**Goal:** Add the `Label cells` radio with a default annotation-name input. On Apply, call `createAnnotation`, `addLabelToAnnotation` twice, `labelCells` twice. Default labels are `high` and `low`. Context is the current selection if non-empty, otherwise all cells. No More-options disclosure yet (added in Task 10). No collision detection yet (added in Task 11). No success footer yet (added in Task 12).

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add label-cells state and an imports block**

At the top of `frontend/src/components/SelectByExpressionModal.tsx`, update the `useData` import:

```tsx
import { createAnnotation, addLabelToAnnotation, labelCells, useDataActions } from '../hooks/useData'
```

In the component body, after the existing `const [subAction, setSubAction] = useState<SubAction>('replace')` line, add:

```tsx
  const defaultAnnotationName = useMemo(() => {
    if (!source) return ''
    const base = source.type === 'gene' ? source.gene : source.name
    return `${base}_${mode}`.replace(/\s+/g, '_')
  }, [source, mode])

  const [annotationName, setAnnotationName] = useState('')
  const [applyStatus, setApplyStatus] = useState<'idle' | 'running' | 'error'>('idle')
  const [applyError, setApplyError] = useState<string | null>(null)

  // Keep annotationName in sync with the default until the user has edited it.
  const [userEditedName, setUserEditedName] = useState(false)
  useEffect(() => {
    if (!userEditedName) setAnnotationName(defaultAnnotationName)
  }, [defaultAnnotationName, userEditedName])
```

- [ ] **Step 2: Add the Label cells UI inside the action selector**

Below the `Update selection` radio block (still inside the action selector div), add:

```tsx
          <label style={{ display: 'block', marginTop: '8px', marginBottom: '4px', color: '#ccc' }}>
            <input
              type="radio"
              checked={action === 'labelCells'}
              onChange={() => setAction('labelCells')}
            />{' '}
            Label cells
          </label>
          {action === 'labelCells' && (
            <div style={{ paddingLeft: '22px' }}>
              <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                Annotation name:
                <input
                  type="text"
                  value={annotationName}
                  onChange={(e) => {
                    setUserEditedName(true)
                    setAnnotationName(e.target.value)
                  }}
                  style={{
                    flex: 1,
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
            </div>
          )}
```

- [ ] **Step 3: Extend `handleApply` with the Label cells branch**

Replace the `// Label cells branch — implemented in Task 9.` placeholder with:

```tsx
    // Label cells branch
    const context = selectedCellIndices.length > 0 ? selectedCellIndices : null
    let high: number[]
    let low: number[]
    if (context === null) {
      // All cells
      high = matching
      const matchingSet = new Set(matching)
      const total = expressionData.values.length
      low = []
      for (let i = 0; i < total; i++) {
        if (!matchingSet.has(i)) low.push(i)
      }
    } else {
      const matchingSet = new Set(matching)
      high = context.filter((i) => matchingSet.has(i))
      low = context.filter((i) => !matchingSet.has(i))
    }

    const highLabel = 'high'
    const lowLabel = 'low'
    const name = annotationName.trim()
    if (!name) {
      setApplyError('Please enter an annotation name.')
      return
    }

    setApplyStatus('running')
    setApplyError(null)
    try {
      await createAnnotation(name)
      await addLabelToAnnotation(name, highLabel)
      await addLabelToAnnotation(name, lowLabel)
      if (high.length > 0) await labelCells(name, highLabel, high)
      if (low.length > 0) await labelCells(name, lowLabel, low)
      setApplyStatus('idle')
      setSource(null)
    } catch (err) {
      setApplyStatus('error')
      setApplyError((err as Error).message || 'Failed to label cells')
    }
```

Mark `handleApply` as `async`:

```tsx
  const handleApply = async () => {
```

- [ ] **Step 4: Show the error message if apply fails**

Directly above the footer buttons (`<div style={{ display: 'flex', justifyContent: 'flex-end' ...`), add:

```tsx
        {applyError && (
          <div style={{ color: '#e94560', fontSize: '11px', marginTop: '8px' }}>
            {applyError}
          </div>
        )}
```

Disable the Apply button while running:

```tsx
          <button
            disabled={applyStatus === 'running'}
            style={{
              padding: '6px 14px',
              backgroundColor: '#4ecdc4',
              color: '#16213e',
              border: '1px solid #4ecdc4',
              borderRadius: '3px',
              cursor: applyStatus === 'running' ? 'wait' : 'pointer',
              fontSize: '11px',
              fontWeight: 600,
              opacity: applyStatus === 'running' ? 0.6 : 1,
            }}
            onClick={handleApply}
          >
            {applyStatus === 'running' ? 'Labeling…' : 'Apply'}
          </button>
```

- [ ] **Step 5: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 6: Manual smoke test**

1. Lasso a region. Open the modal on a gene. Pick `Label cells`. Leave the default name. Click Apply. Verify that a new annotation column appears in the Cell Panel, with `high` and `low` labels, and cell counts that match the region.
2. With no selection, pick Label cells on a gene, Apply. Verify the annotation labels all cells into high/low.

- [ ] **Step 7: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: Label cells action with default annotation name

Creates an annotation column and assigns cells to high/low labels
based on the threshold. Context auto-picks current selection or all."
```

---

### Task 10: Label cells — More options disclosure

**Goal:** Add the `▸ More options` disclosure under the annotation name input. When expanded, shows text inputs for high/low label names and a radio for context (`Current selection` / `All cells`). Defaults preserved from Task 9.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add state for the disclosed fields**

In the component body, below the existing `const [userEditedName, setUserEditedName] = useState(false)`, add:

```tsx
  const [moreOptionsOpen, setMoreOptionsOpen] = useState(false)
  const [highLabel, setHighLabel] = useState('high')
  const [lowLabel, setLowLabel] = useState('low')
  type LabelContext = 'selection' | 'all'
  const [labelContext, setLabelContext] = useState<LabelContext>('selection')

  // When the existing selection becomes empty, force context to 'all'.
  useEffect(() => {
    if (selectedCellIndices.length === 0 && labelContext === 'selection') {
      setLabelContext('all')
    }
  }, [selectedCellIndices, labelContext])
```

- [ ] **Step 2: Render the disclosure UI**

In the Label cells action block (below the annotation name input), add:

```tsx
              <div style={{ marginTop: '6px' }}>
                <button
                  onClick={() => setMoreOptionsOpen((o) => !o)}
                  style={{
                    backgroundColor: 'transparent',
                    border: 'none',
                    color: '#4ecdc4',
                    cursor: 'pointer',
                    fontSize: '11px',
                    padding: 0,
                  }}
                >
                  {moreOptionsOpen ? '▾ More options' : '▸ More options'}
                </button>
                {moreOptionsOpen && (
                  <div
                    style={{
                      marginTop: '6px',
                      padding: '8px',
                      backgroundColor: '#0f1625',
                      borderRadius: '3px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                    }}
                  >
                    <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                      High label:
                      <input
                        type="text"
                        value={highLabel}
                        onChange={(e) => setHighLabel(e.target.value)}
                        style={{
                          flex: 1,
                          backgroundColor: '#16213e',
                          border: '1px solid #0f3460',
                          color: '#ccc',
                          padding: '3px 6px',
                          fontSize: '11px',
                        }}
                      />
                    </label>
                    <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                      Low label:
                      <input
                        type="text"
                        value={lowLabel}
                        onChange={(e) => setLowLabel(e.target.value)}
                        style={{
                          flex: 1,
                          backgroundColor: '#16213e',
                          border: '1px solid #0f3460',
                          color: '#ccc',
                          padding: '3px 6px',
                          fontSize: '11px',
                        }}
                      />
                    </label>
                    <div style={{ color: '#888' }}>Context:</div>
                    <label
                      style={{
                        color: selectedCellIndices.length === 0 ? '#555' : '#ccc',
                      }}
                      title={selectedCellIndices.length === 0 ? 'No existing selection' : undefined}
                    >
                      <input
                        type="radio"
                        checked={labelContext === 'selection'}
                        disabled={selectedCellIndices.length === 0}
                        onChange={() => setLabelContext('selection')}
                      />{' '}
                      Current selection ({selectedCellIndices.length.toLocaleString()} cells)
                    </label>
                    <label style={{ color: '#ccc' }}>
                      <input
                        type="radio"
                        checked={labelContext === 'all'}
                        onChange={() => setLabelContext('all')}
                      />{' '}
                      All cells
                    </label>
                  </div>
                )}
              </div>
```

- [ ] **Step 3: Update `handleApply` to respect the disclosed values**

Replace the Label cells branch's hard-coded `'high'` / `'low'` / context logic with the configurable versions. Change this block:

```tsx
    // Label cells branch
    const context = selectedCellIndices.length > 0 ? selectedCellIndices : null
    let high: number[]
    let low: number[]
    if (context === null) {
      ...
    } else {
      ...
    }

    const highLabel = 'high'
    const lowLabel = 'low'
```

to:

```tsx
    // Label cells branch
    const contextIndices =
      labelContext === 'selection' ? selectedCellIndices : null // null => all cells
    let high: number[]
    let low: number[]
    if (contextIndices === null) {
      high = matching
      const matchingSet = new Set(matching)
      const total = expressionData.values.length
      low = []
      for (let i = 0; i < total; i++) {
        if (!matchingSet.has(i)) low.push(i)
      }
    } else {
      if (contextIndices.length === 0) {
        setApplyError('Current selection is empty — choose All cells or make a selection.')
        return
      }
      const matchingSet = new Set(matching)
      high = contextIndices.filter((i) => matchingSet.has(i))
      low = contextIndices.filter((i) => !matchingSet.has(i))
    }

    const highLabelTrimmed = highLabel.trim() || 'high'
    const lowLabelTrimmed = lowLabel.trim() || 'low'
```

Then rename the two `highLabel` / `lowLabel` references in the downstream API calls to use the `*Trimmed` variables:

```tsx
      await addLabelToAnnotation(name, highLabelTrimmed)
      await addLabelToAnnotation(name, lowLabelTrimmed)
      if (high.length > 0) await labelCells(name, highLabelTrimmed, high)
      if (low.length > 0) await labelCells(name, lowLabelTrimmed, low)
```

- [ ] **Step 4: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 5: Manual smoke test**

Open the modal. Pick Label cells. Click `▸ More options`. Change `high` to `positive` and `low` to `negative`. Apply. Verify the new annotation uses the custom labels. Try selecting `All cells` with a lasso active — confirm both radios work.

- [ ] **Step 6: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: More options disclosure for Label cells

Collapsible section with custom high/low label names and context
radio (current selection vs all cells)."
```

---

### Task 11: Annotation name collision detection

**Goal:** Block Apply when the user-entered annotation name already exists in the dataset's obs columns, to avoid clobbering. Show an inline error next to the name input.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Read existing obs column names**

In the component body, add a hook that reads the current dataset's obs summaries. The existing `useObsSummaries` hook in `frontend/src/hooks/useData.ts` returns `{ summaries: ObsSummary[] }`.

Update the imports:

```tsx
import {
  createAnnotation,
  addLabelToAnnotation,
  labelCells,
  useDataActions,
  useObsSummaries,
} from '../hooks/useData'
```

In the component body, below the `useDataActions` call, add:

```tsx
  const { summaries } = useObsSummaries()
  const existingColumnNames = useMemo(
    () => new Set(summaries.map((s) => s.name)),
    [summaries]
  )
```

- [ ] **Step 2: Compute and show a collision error**

Below the `annotationName` state, add:

```tsx
  const nameCollision =
    action === 'labelCells' &&
    annotationName.trim().length > 0 &&
    existingColumnNames.has(annotationName.trim())
```

In the Label cells UI, below the annotation name `<label>` and before the `More options` button, insert:

```tsx
              {nameCollision && (
                <div style={{ color: '#e94560', fontSize: '11px', marginTop: '4px' }}>
                  An annotation named "{annotationName.trim()}" already exists — choose a different name.
                </div>
              )}
```

- [ ] **Step 3: Block Apply when colliding**

Update the Apply button's `disabled` prop:

```tsx
            disabled={applyStatus === 'running' || (action === 'labelCells' && nameCollision)}
```

And update the opacity/cursor conditionally in the same style block:

```tsx
              cursor: applyStatus === 'running' || nameCollision ? 'not-allowed' : 'pointer',
              ...
              opacity: applyStatus === 'running' || (action === 'labelCells' && nameCollision) ? 0.6 : 1,
```

- [ ] **Step 4: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 5: Manual smoke test**

1. Open the modal. Switch to Label cells. Type an annotation name that already exists (e.g. the default `leiden` if present, or the name of the annotation you created in Task 9's test). Confirm the inline error appears and Apply is disabled.
2. Change the name to a fresh one. Confirm the error clears and Apply becomes enabled.

- [ ] **Step 6: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: detect annotation name collisions in Label cells

Reads obs column names from the store and blocks Apply if the user's
proposed annotation name already exists. Inline error message."
```

---

### Task 12: Success footer with Diff Exp handoff

**Goal:** After Label cells succeeds, instead of closing the modal, show a success footer with counts and an `Open Diff Exp ▸` button. Clicking it calls `setComparisonGroup1` / `setComparisonGroup2` / `setDiffExpModalOpen(true)` and closes this modal.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add success state**

Replace the `applyStatus` state with:

```tsx
  type ApplyStatus =
    | { kind: 'idle' }
    | { kind: 'running' }
    | { kind: 'error'; message: string }
    | { kind: 'success'; highCount: number; lowCount: number; annotationName: string; highLabel: string; lowLabel: string; highIndices: number[]; lowIndices: number[] }

  const [applyStatus, setApplyStatus] = useState<ApplyStatus>({ kind: 'idle' })
```

Remove the separate `applyError` state and its `useState` line — it is now folded into `applyStatus.kind === 'error'`.

Update all existing references:

- Everywhere you see `applyStatus === 'running'`, change to `applyStatus.kind === 'running'`.
- Everywhere you see `applyStatus === 'error'`, change to `applyStatus.kind === 'error'`.
- The apply-error render block:

```tsx
        {applyStatus.kind === 'error' && (
          <div style={{ color: '#e94560', fontSize: '11px', marginTop: '8px' }}>
            {applyStatus.message}
          </div>
        )}
```

- In `handleApply`, replace `setApplyError(...)` calls with `setApplyStatus({ kind: 'error', message: ... })`.
- Replace `setApplyStatus('running')` with `setApplyStatus({ kind: 'running' })`.
- Replace `setApplyStatus('idle')` with `setApplyStatus({ kind: 'idle' })`.

- [ ] **Step 2: Hook setters for the Diff Exp handoff**

Above the `const handleApply = ...` line, read the setters:

```tsx
  const setComparisonGroup1 = useStore((s) => s.setComparisonGroup1)
  const setComparisonGroup2 = useStore((s) => s.setComparisonGroup2)
  const setDiffExpModalOpen = useStore((s) => s.setDiffExpModalOpen)
```

- [ ] **Step 3: Emit success state after Label cells**

In `handleApply`'s Label cells success branch, replace:

```tsx
      setApplyStatus({ kind: 'idle' })
      setSource(null)
```

with:

```tsx
      setApplyStatus({
        kind: 'success',
        highCount: high.length,
        lowCount: low.length,
        annotationName: name,
        highLabel: highLabelTrimmed,
        lowLabel: lowLabelTrimmed,
        highIndices: high,
        lowIndices: low,
      })
```

Remove the `setSource(null)` in the success branch — the modal now stays open and shows the footer instead.

- [ ] **Step 4: Render the success footer**

Replace the footer button row (`<div style={{ display: 'flex', justifyContent: 'flex-end' ...`) with a conditional render:

```tsx
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '14px', alignItems: 'center' }}>
          {applyStatus.kind === 'success' ? (
            <>
              <span style={{ marginRight: 'auto', color: '#4ecdc4', fontSize: '11px' }}>
                Labeled {applyStatus.highCount.toLocaleString()} cells {applyStatus.highLabel},{' '}
                {applyStatus.lowCount.toLocaleString()} cells {applyStatus.lowLabel}.
              </span>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#4ecdc4',
                  color: '#16213e',
                  border: '1px solid #4ecdc4',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                }}
                onClick={() => {
                  setComparisonGroup1(applyStatus.highIndices, applyStatus.highLabel)
                  setComparisonGroup2(applyStatus.lowIndices, applyStatus.lowLabel)
                  setDiffExpModalOpen(true)
                  setSource(null)
                }}
              >
                Open Diff Exp ▸
              </button>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#0f3460',
                  color: '#ccc',
                  border: '1px solid #0f3460',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => setSource(null)}
              >
                Close
              </button>
            </>
          ) : (
            <>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#0f3460',
                  color: '#ccc',
                  border: '1px solid #0f3460',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => setSource(null)}
              >
                Cancel
              </button>
              <button
                disabled={applyStatus.kind === 'running' || (action === 'labelCells' && nameCollision)}
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#4ecdc4',
                  color: '#16213e',
                  border: '1px solid #4ecdc4',
                  borderRadius: '3px',
                  cursor:
                    applyStatus.kind === 'running' || nameCollision ? 'not-allowed' : 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                  opacity:
                    applyStatus.kind === 'running' ||
                    (action === 'labelCells' && nameCollision)
                      ? 0.6
                      : 1,
                }}
                onClick={handleApply}
              >
                {applyStatus.kind === 'running' ? 'Labeling…' : 'Apply'}
              </button>
            </>
          )}
        </div>
```

- [ ] **Step 5: Refresh obs summaries so the new annotation shows up**

After the `setApplyStatus({ kind: 'success', ... })` call, also trigger a refresh so the `useObsSummaries` hook re-fetches and the new annotation column appears in the Cell Panel immediately. The store exposes `refreshObsSummaries` (defined near line 1694 of `store.ts`) which bumps the per-dataset `obsSummariesVersion`.

In the component body, read the action:

```tsx
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
```

After the `setApplyStatus({ kind: 'success', ... })` line in `handleApply`, add:

```tsx
      refreshObsSummaries()
```

- [ ] **Step 6: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 7: Manual smoke test**

1. Lasso a region. Open the modal on a gene. Pick Label cells, Apply. Confirm the modal now shows the success footer with counts and `Open Diff Exp ▸`. Click it and confirm `DiffExpModal` opens with the new annotation pre-filled as group1/group2 with `high` and `low`.
2. Repeat with custom labels (e.g. `positive`/`negative` via More options). Confirm the success footer and Diff Exp handoff use those labels.

- [ ] **Step 8: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: success footer with Open Diff Exp handoff

After Label cells succeeds, the modal shows counts and a button that
pre-fills DiffExpModal via setComparisonGroup1/2, then closes."
```

---

### Task 13: Edge cases — zero variance, empty context, dataset switch

**Goal:** Handle a few remaining edge cases that are described in the spec but not yet wired: (a) zero variance blocks Apply with a clear notice (the notice already renders; now ensure Apply is disabled), (b) dataset switch while the modal is open closes the modal, (c) the Label cells context-is-empty case disables Apply with a tooltip rather than showing an error post-click.

**Files:**
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Close the modal on dataset switch**

Read the active slot and close on change. In the component body:

```tsx
  const activeSlot = useStore((s) => s.activeSlot)
  const [openedInSlot, setOpenedInSlot] = useState<string | null>(null)

  useEffect(() => {
    if (!source) {
      setOpenedInSlot(null)
      return
    }
    if (openedInSlot === null) {
      setOpenedInSlot(activeSlot)
      return
    }
    if (openedInSlot !== activeSlot) {
      setSource(null)
    }
  }, [source, activeSlot, openedInSlot, setSource])
```

- [ ] **Step 2: Block Apply on zero variance**

Update the Apply button's `disabled` prop:

```tsx
                disabled={
                  applyStatus.kind === 'running' ||
                  (action === 'labelCells' && nameCollision) ||
                  !!histogram?.zeroVariance ||
                  (action === 'labelCells' &&
                    labelContext === 'selection' &&
                    selectedCellIndices.length === 0)
                }
```

Update the button's `title` attribute for a tooltip hint:

```tsx
                title={
                  histogram?.zeroVariance
                    ? 'Cannot threshold: all cells have the same value'
                    : action === 'labelCells' && nameCollision
                    ? 'Annotation name already exists'
                    : action === 'labelCells' &&
                      labelContext === 'selection' &&
                      selectedCellIndices.length === 0
                    ? 'Current selection is empty'
                    : undefined
                }
```

(Add the `title` prop alongside the existing `style`/`onClick` on the Apply button.)

Update the opacity accordingly — any disabled condition should dim it:

```tsx
                  opacity:
                    applyStatus.kind === 'running' ||
                    (action === 'labelCells' && nameCollision) ||
                    histogram?.zeroVariance ||
                    (action === 'labelCells' &&
                      labelContext === 'selection' &&
                      selectedCellIndices.length === 0)
                      ? 0.6
                      : 1,
```

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 4: Manual smoke test**

1. Load two datasets (Primary and Secondary). Open the modal on a gene in Primary. Switch to Secondary. Confirm the modal closes.
2. Find or add a zero-variance gene (e.g. a gene with no expression in any cell), or load a tiny dataset. Open the modal. Confirm the zero-variance notice renders and the Apply button is disabled with a tooltip.
3. With no selection, pick Label cells, leave context on `Current selection` (it should auto-switch to `All cells` via the existing effect from Task 10 — if it doesn't for some reason, the Apply button should still be disabled with the tooltip).

- [ ] **Step 5: Commit**

```
git add frontend/src/components/SelectByExpressionModal.tsx
git commit -m "feat: edge cases for SelectByExpressionModal

Close on dataset switch; disable Apply on zero-variance, name
collision, and empty-context-for-label."
```

---

### Task 14: Centralize UI strings in messages.ts

**Goal:** Move the user-facing strings introduced by the modal into `frontend/src/messages.ts` to match project convention.

**Files:**
- Modify: `frontend/src/messages.ts`
- Modify: `frontend/src/components/SelectByExpressionModal.tsx`

- [ ] **Step 1: Add the strings**

Open `frontend/src/messages.ts`. Find the export structure (it should be a single object or multiple named exports — follow the existing pattern). Add a new section:

```ts
export const selectByExpression = {
  titleGene: (gene: string) => `Select cells by ${gene}`,
  titleGeneSet: (name: string) => `Select cells by ${name}`,
  loading: 'Loading expression values…',
  zeroVariance: (value: number) =>
    `All cells have the same value (${value.toFixed(2)}) — nothing to threshold on.`,
  matchCounter: (match: number, total: number) =>
    `Matching: ${match.toLocaleString()} / ${total.toLocaleString()} cells`,
  annotationCollision: (name: string) =>
    `An annotation named "${name}" already exists — choose a different name.`,
  emptyContextTooltip: 'Current selection is empty',
  zeroVarianceTooltip: 'Cannot threshold: all cells have the same value',
  collisionTooltip: 'Annotation name already exists',
  emptyNameError: 'Please enter an annotation name.',
  labelingButton: 'Labeling…',
  applyButton: 'Apply',
  cancelButton: 'Cancel',
  closeButton: 'Close',
  openDiffExpButton: 'Open Diff Exp ▸',
  successFooter: (highCount: number, highLabel: string, lowCount: number, lowLabel: string) =>
    `Labeled ${highCount.toLocaleString()} cells ${highLabel}, ${lowCount.toLocaleString()} cells ${lowLabel}.`,
}
```

If the existing file is organized differently (e.g. a single `messages` object), slot these in as a sub-key named `selectByExpression`.

- [ ] **Step 2: Use the strings in the modal**

At the top of `frontend/src/components/SelectByExpressionModal.tsx`, add:

```tsx
import { selectByExpression as msg } from '../messages'
```

Replace each hard-coded string with its message-table equivalent:

- `title` construction → `source.type === 'gene' ? msg.titleGene(source.gene) : msg.titleGeneSet(source.name)`
- "Loading expression values…" → `msg.loading`
- "All cells have the same value (…)" → `msg.zeroVariance(histogram.min)`
- "Matching: N / total cells" → `msg.matchCounter(matchCount, expressionData.values.length)`
- "An annotation named … already exists …" → `msg.annotationCollision(annotationName.trim())`
- "Please enter an annotation name." → `msg.emptyNameError`
- "Cannot threshold: all cells have the same value" → `msg.zeroVarianceTooltip`
- "Annotation name already exists" → `msg.collisionTooltip`
- "Current selection is empty" → `msg.emptyContextTooltip`
- "Labeling…" / "Apply" / "Cancel" / "Close" / "Open Diff Exp ▸" → `msg.labelingButton` / `msg.applyButton` / `msg.cancelButton` / `msg.closeButton` / `msg.openDiffExpButton`
- Success footer text → `msg.successFooter(applyStatus.highCount, applyStatus.highLabel, applyStatus.lowCount, applyStatus.lowLabel)`

- [ ] **Step 3: Verify the build**

Run from `frontend/`:
```
npm run build
```
Expected: TypeScript compiles without errors.

- [ ] **Step 4: Commit**

```
git add frontend/src/messages.ts frontend/src/components/SelectByExpressionModal.tsx
git commit -m "refactor: centralize SelectByExpressionModal strings in messages.ts"
```

---

### Task 15: Documentation (CLAUDE.md, CHANGELOG.md, README.md) and final verification

**Goal:** Per the project's documentation policy, update CLAUDE.md's Components table and Key Behaviors section, add a CHANGELOG entry under `[Unreleased]`, add a README walkthrough subsection, and confirm the build is clean.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md Components table**

Open `CLAUDE.md`. Find the Components table (search for `ClusterGeneSetModal.tsx` row). Add a new row immediately after it:

```markdown
| `SelectByExpressionModal.tsx` | Threshold-based cell selection from a gene or gene set score. Interactive histogram with Above/Below/Between modes, Update-selection (Replace/Add/Intersect) or Label-cells actions, and direct handoff to DiffExpModal. Opened from the `⋯` overflow menu on gene rows and gene set rows in GenePanel. |
```

- [ ] **Step 2: Update CLAUDE.md Key Behaviors**

Scroll to the Key Behaviors section. Add a new bullet (in a sensible spot — near the existing cluster-gene-set behavior is natural):

```markdown
- **Select cells by expression**: From the `⋯` overflow menu on any gene row (search/browse results and inside expanded gene sets) or gene set row, pick `Select cells…` to open `SelectByExpressionModal`. The modal auto-colors the scatter plot by the source, shows an interactive histogram of expression values with draggable cutoff line(s), and supports Above/Below/Between threshold modes. Two action paths: `Update selection` (Replace/Add/Intersect) updates the current cell selection in-place; `Label cells` creates a new annotation column with `high`/`low` labels (labels and context customizable via More options), blocks on annotation-name collisions, and on success shows a footer with `Open Diff Exp ▸` that pre-fills `DiffExpModal`'s group1/group2 via `setComparisonGroup1`/`setComparisonGroup2` for immediate differential expression. Threshold compute and histogram binning are pure frontend; all annotation and diff-exp APIs are existing endpoints.
```

- [ ] **Step 3: Update CHANGELOG.md**

Open `CHANGELOG.md`. Find the `[Unreleased]` section. Under `### Added`, add:

```markdown
- Select cells by expression threshold from gene and gene set `⋯` overflow menus. Interactive histogram with Above / Below / Between threshold modes; Update-selection action supports Replace / Add / Intersect; Label-cells action creates an annotation column with customizable high/low labels and context, and hands off directly to differential expression for the newly labeled groups.
```

If no `[Unreleased]` section exists yet, add one at the top:

```markdown
## [Unreleased]

### Added

- Select cells by expression threshold from gene and gene set `⋯` overflow menus. Interactive histogram with Above / Below / Between threshold modes; Update-selection action supports Replace / Add / Intersect; Label-cells action creates an annotation column with customizable high/low labels and context, and hands off directly to differential expression for the newly labeled groups.
```

- [ ] **Step 4: Update README.md walkthrough**

Open `README.md`. Find the analysis walkthrough section (search for headings related to differential expression, cell selection, or gene sets). Add a new subsection:

```markdown
### Selecting cells by expression threshold

You can select cells based on a gene's expression or a gene set score without needing to eyeball the scatter plot:

1. In the Gene Panel, click the `⋯` menu on any gene row or gene set row and choose `Select cells…`.
2. The modal opens and the scatter plot switches to expression coloring for that source. An interactive histogram of the values is shown.
3. Pick a threshold mode (`Above`, `Below`, or `Between`) and drag the red cutoff line(s). The match counter updates live.
4. Choose an action:
   - **Update selection** replaces, adds to, or intersects with your current lasso selection.
   - **Label cells** creates a new annotation column with `high`/`low` labels for the cells in the chosen context (current selection or all cells). On success, click `Open Diff Exp ▸` to immediately run differential expression between the two groups.

Typical workflow for "find DEGs by expression state in a region": lasso a region → `⋯ → Select cells…` on a gene → drag the threshold → Label cells → Open Diff Exp.
```

- [ ] **Step 5: Final build check**

Run from `frontend/`:
```
npm run build
```
Expected: clean TypeScript compilation, no errors or warnings beyond the pre-existing baseline.

- [ ] **Step 6: Final manual regression test**

With dev servers running, walk through the full feature once more:

1. Load a dataset. Lasso a region on the scatter plot.
2. Open `⋯ → Select cells…` on a gene in the browse list.
3. Drag the threshold, click `Label cells`, expand `More options`, change labels to `pos`/`neg`, click Apply.
4. Click `Open Diff Exp ▸`. Confirm the diff exp modal opens with the new annotation and the correct group labels.
5. Cancel out. Open the modal again on a gene set row. Apply `Update selection → Intersect`. Confirm the selection shrinks.
6. Switch to a second dataset (if available). Confirm the modal closes gracefully.

- [ ] **Step 7: Commit docs**

```
git add CLAUDE.md CHANGELOG.md README.md
git commit -m "docs: document Select cells by expression feature

CLAUDE.md Components table + Key Behaviors entry, CHANGELOG entry
under [Unreleased] → Added, README walkthrough subsection."
```

---

## Self-Review Summary

**Spec coverage (each requirement → task):**

| Spec requirement | Task(s) |
|---|---|
| Entry point: gene set row `⋯` `Select cells…` | Task 3 |
| Entry point: gene row `⋯` in browse/search | Task 3 |
| Entry point: gene row `⋯` inside expanded gene sets | Task 3 |
| Modal shell + store pattern mount | Tasks 1, 2 |
| Auto-color plot on open | Task 4 |
| Histogram computation | Task 4 |
| Histogram SVG rendering | Task 5 |
| Mode selector Above/Below/Between | Task 6 |
| Numeric threshold inputs + live match counter | Task 6 |
| Between mode crossed-handle auto-swap | Task 6 |
| Interactive drag handles on histogram | Task 7 |
| Update selection: Replace | Task 8 |
| Update selection: Add / Intersect (disabled states) | Task 8 |
| Label cells default UI (name only) | Task 9 |
| Label cells More options (labels, context) | Task 10 |
| Annotation name collision check | Task 11 |
| Success footer + counts | Task 12 |
| Open Diff Exp handoff via existing setComparisonGroup1/2 | Task 12 |
| Refresh obs summaries after labeling | Task 12 |
| Zero-variance handling (disable apply, notice) | Tasks 4 (notice), 13 (disable) |
| Dataset switch closes modal | Task 13 |
| Empty context on Label cells (disabled Apply) | Task 13 |
| Messages centralization | Task 14 |
| CLAUDE.md / CHANGELOG.md / README.md updates | Task 15 |

**Placeholder scan:** No "TBD", "TODO", or "implement later" strings. Every step with a code change includes the actual code.

**Type consistency:** `ThresholdMode`, `SelectByExpressionSource`, `Action`, `SubAction`, `ApplyStatus`, `Histogram`, `LabelContext` — all defined exactly once in the tasks that introduce them, referenced consistently after. Function names: `computeHistogram`, `matchingIndices`, `defaultThresholds`, `handleApply`. Store setters referenced: `setSelectByExpressionSource`, `setSelectedCellIndices`, `setComparisonGroup1`, `setComparisonGroup2`, `setDiffExpModalOpen`, `refreshObsSummaries` — all verified against the existing store shape.

**Not in scope of this plan (per spec):** Feature 2 (mask genes in the Gene Panel by `.var` boolean columns). Bivariate thresholding. Multi-cutoff. Backend changes. Automated tests.
