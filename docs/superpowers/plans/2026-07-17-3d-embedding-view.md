# 3D Embedding View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users view an existing ≥3-column `.obsm` embedding as an interactive 3D scatter (orbit/zoom/pan, hover/click, all coloring modes, category labels, and projection-based lasso selection), rendered smoothly on the GPU.

**Architecture:** Reuse the existing deck.gl GPU pipeline. A new `ScatterPlot3D.tsx` renders a deck.gl `OrbitView` with a `size: 3` position buffer and opaque, depth-tested points. The current per-cell coloring logic is extracted from `ScatterPlot.tsx` into a shared `useCellColor` hook so 2D and 3D never diverge. A new `EmbeddingPlot` wrapper picks 2D vs 3D. 3D lasso selection projects each cell to screen and runs point-in-polygon (extracted as a pure, unit-tested function).

**Tech Stack:** React 18 + TypeScript, Zustand store, deck.gl v9 (`@deck.gl/core`, `@deck.gl/layers`, `@deck.gl/react`), FastAPI + AnnData backend, pytest (backend), Playwright E2E, vitest (new — frontend unit).

## Global Constraints

- **View-only.** No 3D compute (no `n_components=3` UMAP/t-SNE). Only view existing ≥3-column `.obsm` embeddings.
- **`coordinates` stays `[number, number][]`** on `EmbeddingData`. The third dimension rides alongside as `z?: number[]`. No existing 2D code path may change shape.
- **`dim_z=None` must be byte-identical** to today's `get_embedding` response.
- **Opaque points + depth test in 3D** (no per-frame depth sorting for transparency).
- **Lasso captures everything inside the on-screen outline** (projection-based, no depth limit).
- **These 2D-only tools stay hidden in 3D:** shape drawing, line/trajectory drawing, embedding rotate/"quilt" transforms, SVG line overlays.
- **New dependency:** this plan introduces `vitest` as the frontend's first unit-test runner (Task 4). It is Vite-native and isolated; if the maintainer prefers, its single test can instead be validated only through the Task 8 E2E.
- **Toolchain:** backend tests run under the `dev` pixi environment (`pixi run -e dev pytest ...`); frontend builds via `npm run build` (runs `tsc` then `vite build`) inside `frontend/`. Dev servers: `pixi run backend` (:8000), `pixi run dev` (:5173). **Kill :8000 and :5173 after browser verification — do not leave them running.**

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/xcell/adaptor.py` | `get_embedding` gains `dim_z`; add `_clamp_one_dim` |
| `backend/xcell/api/routes.py` | `get_embedding` route gains `dim_z` query param |
| `backend/tests/test_embedding_dims.py` | new `dim_z` tests |
| `frontend/src/lib/cellColors.ts` | **NEW** — all color helpers + `useCellColor` hook (extracted) |
| `frontend/src/lib/lasso3d.ts` | **NEW** — `pointsInLassoScreen` pure function |
| `frontend/src/lib/lasso3d.test.ts` | **NEW** — vitest unit test |
| `frontend/src/components/ScatterPlot.tsx` | consume `useCellColor`; import helpers from `cellColors.ts` (behavior unchanged) |
| `frontend/src/components/ScatterPlot3D.tsx` | **NEW** — OrbitView, 3D layer, orbit/hover/click, lasso, labels |
| `frontend/src/components/EmbeddingPlot.tsx` | **NEW** — thin wrapper: render `ScatterPlot3D` when 3D-able else `ScatterPlot` |
| `frontend/src/store.ts` | `EmbeddingData.z`/`dim_z`; `embeddingDims[].z`; `viewMode` + setter |
| `frontend/src/hooks/useData.ts` | `useEmbedding` sends `dim_z` when in 3D |
| `frontend/src/App.tsx` | `DimensionPicker` Z dropdown + 2D/3D toggle; 3 render sites use `EmbeddingPlot`; imports helpers from `cellColors.ts` |
| `frontend/src/utils/exportFigure.ts`, `components/FigurePanel.tsx`, `components/DisplaySettings.tsx` | import color helpers from `cellColors.ts` |

---

## Task 1: Backend — `get_embedding` gains `dim_z`

**Files:**
- Modify: `backend/xcell/adaptor.py` (`get_embedding` ~line 540; add `_clamp_one_dim` near `_clamp_dims` ~line 514)
- Modify: `backend/xcell/api/routes.py` (`get_embedding` route ~line 374)
- Test: `backend/tests/test_embedding_dims.py`

**Interfaces:**
- Produces: `adaptor.get_embedding(name, dim_x=0, dim_y=1, dim_z=None)` → dict. When `dim_z is not None`, dict additionally has `"z": list[float]` (length n_cells) and `"dim_z": int`. When `dim_z is None`, dict is unchanged (`name`, `coordinates`, `dim_x`, `dim_y`).
- Produces: REST `GET /api/embedding/{name}?dim_x=&dim_y=&dim_z=` — `dim_z` optional int.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_embedding_dims.py` (reuses the existing `_a()` fixture whose `X_pca` is 4-D):

```python
def test_get_embedding_with_dim_z():
    a = _a(); pca = a.adata.obsm['X_pca']
    d = a.get_embedding('X_pca', dim_x=0, dim_y=1, dim_z=2)
    assert (d['dim_x'], d['dim_y'], d['dim_z']) == (0, 1, 2)
    assert len(d['z']) == pca.shape[0]
    np.testing.assert_allclose(d['z'], pca[:, 2])
    # x/y unchanged alongside z
    np.testing.assert_allclose([c[0] for c in d['coordinates']], pca[:, 0])


def test_get_embedding_dim_z_none_is_unchanged():
    a = _a()
    d = a.get_embedding('X_pca', dim_x=0, dim_y=1)  # dim_z defaults to None
    assert 'z' not in d and 'dim_z' not in d
    assert set(d.keys()) == {'name', 'coordinates', 'dim_x', 'dim_y'}


def test_get_embedding_dim_z_out_of_range_clamps():
    a = _a()
    d = a.get_embedding('X_pca', dim_x=0, dim_y=1, dim_z=99)
    assert d['dim_z'] == 0  # clamps into range
    assert 'z' in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/pcahan/Dropbox (Personal)/Code/xcell" && pixi run -e dev pytest backend/tests/test_embedding_dims.py -v`
Expected: the three new tests FAIL (`get_embedding() got an unexpected keyword argument 'dim_z'`).

- [ ] **Step 3: Add `_clamp_one_dim` to `adaptor.py`**

Immediately after `_clamp_dims` (ends ~line 524), add:

```python
    def _clamp_one_dim(self, name: str, dim: int) -> int:
        """Clamp a single .obsm column index into [0, ncols-1] (fallback 0)."""
        arr = self.adata.obsm[name]
        ncols = arr.shape[1] if getattr(arr, 'ndim', 1) == 2 else 1
        d = int(dim)
        return d if 0 <= d < ncols else 0
```

- [ ] **Step 4: Extend `get_embedding` in `adaptor.py`**

Change the signature and return (current body at ~540–566):

```python
    def get_embedding(self, name: str, dim_x: int = 0, dim_y: int = 1,
                      dim_z: int | None = None) -> dict[str, Any]:
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {list(self.adata.obsm.keys())}")
        dx, dy = self._clamp_dims(name, dim_x, dim_y)
        coords_2d = self.adata.obsm[name][:, [dx, dy]]
        result = {
            "name": name,
            "coordinates": coords_2d.tolist(),
            "dim_x": dx,
            "dim_y": dy,
        }
        if dim_z is not None:
            dz = self._clamp_one_dim(name, dim_z)
            result["z"] = np.asarray(self.adata.obsm[name][:, dz], dtype=float).tolist()
            result["dim_z"] = dz
        return result
```

(Keep the existing docstring; append a line noting `dim_z` adds `z`/`dim_z`.)

- [ ] **Step 5: Extend the REST route in `routes.py`**

Change `get_embedding` (~374):

```python
def get_embedding(
    name: str,
    dim_x: int = Query(0),
    dim_y: int = Query(1),
    dim_z: int | None = Query(None),
    dataset: str | None = Query(None),
):
    """Get embedding coordinates by name, viewing two (or three, via dim_z) .obsm columns."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_embedding(name, dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd "/Users/pcahan/Dropbox (Personal)/Code/xcell" && pixi run -e dev pytest backend/tests/test_embedding_dims.py -v`
Expected: all tests PASS (new + pre-existing).

- [ ] **Step 7: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_embedding_dims.py
git commit -m "feat(backend): get_embedding gains optional dim_z (third .obsm column)"
```

---

## Task 2: Extract shared coloring into `cellColors.ts`

Behavior-preserving refactor. Move the color helpers and the `getColor` memo out of `ScatterPlot.tsx` so `ScatterPlot3D` can reuse them.

**Files:**
- Create: `frontend/src/lib/cellColors.ts`
- Modify: `frontend/src/components/ScatterPlot.tsx` (remove moved symbols; import them back; replace `getColor` memo with `useCellColor(...)`)
- Modify importers: `frontend/src/App.tsx:5`, `frontend/src/utils/exportFigure.ts:2`, `frontend/src/components/FigurePanel.tsx:7`, `frontend/src/components/DisplaySettings.tsx:3`

**Interfaces:**
- Produces (all exported from `cellColors.ts`): `hexToRgb`, `resolveCategoryPalette`, `getColorFromScale`, `getBivariateColor`, `BIVARIATE_COLORMAPS`, and
  ```ts
  export interface CellColorParams {
    displayPreferences: DisplayPreferences
    activeCellMask: (boolean[] | Uint8Array | null)
    colorMode: ColorMode
    colorBy: ObsColumnData | null
    expressionData: ExpressionData | null
    bivariateData: BivariateData | null
    highlightLayers: HighlightLayer[]
    selectedSet: Set<number>
  }
  export function useCellColor(p: CellColorParams): (d: { index: number }) => [number, number, number, number]
  ```
  (Import the exact types — `DisplayPreferences`, `ColorMode`, `ObsColumnData`, `ExpressionData`, `BivariateData`, `HighlightLayer` — from `../store`.)

- [ ] **Step 1: Create `cellColors.ts` and move the pure helpers**

Cut these symbols **verbatim** from `ScatterPlot.tsx` into a new `frontend/src/lib/cellColors.ts` and prefix each with `export` (some already are):
- `hexToRgb` (~line 85), `resolveCategoryPalette` (~100), `getColorFromScale` (~249), `getBivariateColor` (~292)
- `BIVARIATE_COLORMAPS` (exported; find its declaration in `ScatterPlot.tsx`)
- module constants `DEFAULT_COLOR` (~113), `SELECTED_COLOR` (~114) — export them
- helper `layerWeightFn` (~309) — export it

Bring along any imports those functions need (e.g. `ColorScale`, `HighlightLayer` types from `../store`, and any palette constants they reference). Fix relative import depth (`../store`, not `./store`).

- [ ] **Step 2: Add the `useCellColor` hook to `cellColors.ts`**

```ts
import { useMemo } from 'react'
// ...types imported from '../store'

export function useCellColor(p: CellColorParams) {
  return useMemo(() => {
    // PASTE the entire body of the former getColor useMemo (ScatterPlot.tsx ~670–801)
    // verbatim, replacing each captured variable with the p.* field:
    //   displayPreferences -> p.displayPreferences
    //   activeCellMask     -> p.activeCellMask
    //   colorMode          -> p.colorMode
    //   bivariateData      -> p.bivariateData
    //   expressionData     -> p.expressionData
    //   colorBy            -> p.colorBy
    //   selectedSet        -> p.selectedSet
    //   highlightLayers    -> p.highlightLayers
    return baseColorFn
  }, [p.colorBy, p.expressionData, p.bivariateData, p.highlightLayers,
      p.colorMode, p.selectedSet, p.displayPreferences, p.activeCellMask])
}
```

- [ ] **Step 3: Update `ScatterPlot.tsx` to consume the hook**

- Delete the moved symbols and the whole `const getColor = useMemo(...)` block (~669–802).
- Add import: `import { hexToRgb, resolveCategoryPalette, getColorFromScale, getBivariateColor, BIVARIATE_COLORMAPS, useCellColor } from '../lib/cellColors'` (only the names still referenced in this file).
- Replace the deleted memo with:
```ts
const getColor = useCellColor({
  displayPreferences, activeCellMask, colorMode, colorBy,
  expressionData, bivariateData, highlightLayers, selectedSet,
})
```
- If `ScatterPlot.tsx` still re-exports helpers other files import, keep a re-export line (`export { hexToRgb, resolveCategoryPalette, getBivariateColor, getColorFromScale, BIVARIATE_COLORMAPS } from '../lib/cellColors'`) OR update the importers in Step 4. Prefer Step 4 (no re-export).

- [ ] **Step 4: Repoint the four external importers**

- `App.tsx:5` → `import EmbeddingPlot from './components/EmbeddingPlot'` is added in Task 5; for now change the helper import to `import { BIVARIATE_COLORMAPS, getBivariateColor, resolveCategoryPalette } from './lib/cellColors'` and keep `import ScatterPlot from './components/ScatterPlot'`.
- `utils/exportFigure.ts:2` → `import { hexToRgb } from '../lib/cellColors'`
- `components/FigurePanel.tsx:7` → `import { getColorFromScale, getBivariateColor, resolveCategoryPalette, hexToRgb } from '../lib/cellColors'`
- `components/DisplaySettings.tsx:3` → `import { getBivariateColor } from '../lib/cellColors'`

- [ ] **Step 5: Typecheck/build**

Run: `cd frontend && npm run build`
Expected: build succeeds, no TS errors, no unresolved imports.

- [ ] **Step 6: Verify 2D is visually unchanged**

Start servers (`pixi run backend` and `pixi run dev` in background), open `http://localhost:5173`, load a dataset, and confirm the embedding renders and coloring (categorical, then a gene) looks identical to before. Screenshot for the record. Kill :8000 and :5173.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/cellColors.ts frontend/src/components/ScatterPlot.tsx frontend/src/App.tsx frontend/src/utils/exportFigure.ts frontend/src/components/FigurePanel.tsx frontend/src/components/DisplaySettings.tsx
git commit -m "refactor(frontend): extract cell coloring into shared useCellColor hook"
```

---

## Task 3: Frontend data model — `z`, `dim_z`, `viewMode`

**Files:**
- Modify: `frontend/src/store.ts` (`EmbeddingData` ~15; `embeddingDims` type ~704; `setEmbeddingDims` ~903/2008; add `viewMode`/`setViewMode`)
- Modify: `frontend/src/hooks/useData.ts` (`useEmbedding` ~112–133)

**Interfaces:**
- Produces: `EmbeddingData.z?: number[]`, `EmbeddingData.dim_z?: number`.
- Produces: `embeddingDims: Record<string, { x: number; y: number; z?: number }>`; `setEmbeddingDims(name, x, y, z?)`.
- Produces: `viewMode: '2d' | '3d'`; `setViewMode(m: '2d' | '3d')`.

- [ ] **Step 1: Extend `EmbeddingData` (store.ts ~15)**

```ts
export interface EmbeddingData {
  name: string
  coordinates: [number, number][]
  dim_x?: number
  dim_y?: number
  z?: number[]      // third .obsm column, present only when viewing in 3D
  dim_z?: number
}
```

- [ ] **Step 2: Extend `embeddingDims` + setter + add `viewMode`**

- Type (~704): `embeddingDims: Record<string, { x: number; y: number; z?: number }>`
- Setter type (~903): `setEmbeddingDims: (embeddingName: string, x: number, y: number, z?: number) => void`
- Setter impl (~2008):
```ts
setEmbeddingDims: (embeddingName, x, y, z) =>
  set((state) => ({ embeddingDims: { ...state.embeddingDims, [embeddingName]: { x, y, z } } })),
```
- Add state field (near other UI state) `viewMode: '2d' as '2d' | '3d'`, and in the actions:
```ts
setViewMode: (m: '2d' | '3d') => set({ viewMode: m }),
```
Add `viewMode: '2d' | '3d'` and `setViewMode: (m: '2d' | '3d') => void` to the store's TS interface.

(Existing `setEmbeddingDims(name, x, y)` calls remain valid — `z` is optional/undefined.)

- [ ] **Step 3: Send `dim_z` from `useEmbedding` (useData.ts ~112)**

```ts
export function useEmbedding() {
  const { selectedEmbedding, embedding, setEmbedding, setLoading, setError } = useStore()
  const dims = useStore((s) => (selectedEmbedding ? s.embeddingDims[selectedEmbedding] : undefined))
  const viewMode = useStore((s) => s.viewMode)
  const dimX = dims?.x ?? 0
  const dimY = dims?.y ?? 1
  const dimZ = viewMode === '3d' ? (dims?.z ?? null) : null

  useEffect(() => {
    if (!selectedEmbedding) return
    const loadedZ = embedding?.dim_z ?? null
    if (embedding?.name === selectedEmbedding
        && (embedding?.dim_x ?? 0) === dimX
        && (embedding?.dim_y ?? 1) === dimY
        && loadedZ === dimZ) return

    setLoading(true)
    const zq = dimZ != null ? `&dim_z=${dimZ}` : ''
    fetchJson<EmbeddingData>(appendDataset(`${API_BASE}/embedding/${selectedEmbedding}?dim_x=${dimX}&dim_y=${dimY}${zq}`))
      .then(setEmbedding)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [selectedEmbedding, embedding, dimX, dimY, dimZ, setEmbedding, setLoading, setError])

  return embedding
}
```

- [ ] **Step 4: Build**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 5: Verify 2D unaffected**

With servers running, load a dataset — confirm the embedding still fetches and renders (viewMode defaults `'2d'`, so no `dim_z` is sent; behavior identical). Kill servers.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/store.ts frontend/src/hooks/useData.ts
git commit -m "feat(frontend): embedding z/dim_z + viewMode state; useEmbedding requests dim_z in 3D"
```

---

## Task 4: `lasso3d.ts` — pure projection lasso + unit test

**Files:**
- Create: `frontend/src/lib/lasso3d.ts`
- Create: `frontend/src/lib/lasso3d.test.ts`
- Modify: `frontend/package.json` (add `vitest`, `test` script)

**Interfaces:**
- Produces:
  ```ts
  export interface ProjectingViewport { project(coord: number[]): number[] }  // deck.gl viewport is compatible
  export function pointsInLassoScreen(
    coordinates: [number, number][],
    z: number[],
    polygonScreen: [number, number][],
    viewport: ProjectingViewport,
  ): number[]   // indices whose projected [sx,sy] fall inside polygonScreen
  ```

- [ ] **Step 1: Add vitest to the frontend**

In `frontend/package.json`, add to `devDependencies`: `"vitest": "^2.0.0"`, and to `scripts`: `"test": "vitest run"`. Then:

Run: `cd frontend && npm install`
Expected: installs vitest.

- [ ] **Step 2: Write the failing test**

`frontend/src/lib/lasso3d.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { pointsInLassoScreen } from './lasso3d'

// Stub viewport: identity projection using x,y (ignores z) so expectations are obvious.
const identityViewport = { project: (c: number[]) => [c[0], c[1]] }

describe('pointsInLassoScreen', () => {
  it('selects points whose projection is inside the polygon', () => {
    const coords: [number, number][] = [[1, 1], [5, 5], [9, 9]]
    const z = [0, 0, 0]
    const square: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]]
    expect(pointsInLassoScreen(coords, z, square, identityViewport)).toEqual([0])
  })

  it('uses the projected coordinate (z affects projection via viewport)', () => {
    // viewport that shifts x by z: a point at x=1,z=10 projects to x=11 (outside)
    const vp = { project: (c: number[]) => [c[0] + c[2], c[1]] }
    const coords: [number, number][] = [[1, 1], [1, 1]]
    const z = [0, 10]
    const square: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]]
    expect(pointsInLassoScreen(coords, z, square, vp)).toEqual([0])
  })

  it('returns empty when polygon has < 3 vertices', () => {
    const coords: [number, number][] = [[1, 1]]
    expect(pointsInLassoScreen(coords, [0], [[0, 0], [1, 1]], identityViewport)).toEqual([])
  })
})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL (`pointsInLassoScreen` not found).

- [ ] **Step 4: Implement `lasso3d.ts`**

```ts
export interface ProjectingViewport {
  project(coord: number[]): number[]
}

/** Ray-casting point-in-polygon on screen-space coordinates. */
function inPolygon(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect = (yi > py) !== (yj > py)
      && px < ((xj - xi) * (py - yi)) / ((yj - yi) || 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

/**
 * Indices of cells whose 3D position projects to a screen point inside the
 * lasso polygon. Depth-agnostic by design: everything under the outline is
 * captured (rotate + re-lasso to refine).
 */
export function pointsInLassoScreen(
  coordinates: [number, number][],
  z: number[],
  polygonScreen: [number, number][],
  viewport: ProjectingViewport,
): number[] {
  if (polygonScreen.length < 3) return []
  const out: number[] = []
  for (let i = 0; i < coordinates.length; i++) {
    const c = coordinates[i]
    const [sx, sy] = viewport.project([c[0], c[1], z[i] ?? 0])
    if (inPolygon(sx, sy, polygonScreen)) out.push(i)
  }
  return out
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/lasso3d.ts frontend/src/lib/lasso3d.test.ts frontend/package.json frontend/package-lock.json
git commit -m "feat(frontend): pointsInLassoScreen pure fn + vitest (first frontend unit test)"
```

---

## Task 5: `ScatterPlot3D` + `EmbeddingPlot` wrapper + UI toggle

Build the 3D component (render + orbit + hover/click, no lasso yet), the wrapper that selects 2D vs 3D, and the `DimensionPicker` Z dropdown + 2D/3D toggle. This is the smallest chunk that makes 3D reachable and E2E-testable.

**Files:**
- Create: `frontend/src/components/ScatterPlot3D.tsx`
- Create: `frontend/src/components/EmbeddingPlot.tsx`
- Modify: `frontend/src/App.tsx` (`DimensionPicker` ~668; 3 `<ScatterPlot>` sites → `<EmbeddingPlot>`)

**Interfaces:**
- Consumes: `useCellColor` (Task 2), `EmbeddingData.z`/`dim_z` + `viewMode` (Task 3).
- Produces: `<ScatterPlot3D {...scatterProps} />` — accepts the same props object the three `<ScatterPlot>` sites already pass (it ignores the 2D-only callbacks `onLineDrawn`, `onTransformEmbedding`, `onTransformEmbeddingSubset`). `<EmbeddingPlot {...scatterProps} />` renders `ScatterPlot3D` when `viewMode==='3d' && embedding.z` is present, else `ScatterPlot`.

- [ ] **Step 1: Create `ScatterPlot3D.tsx`**

```tsx
import { useEffect, useMemo, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrbitView } from '@deck.gl/core'
import { useStore } from '../store'
import { useCellColor } from '../lib/cellColors'
import type { EmbeddingData, ObsColumnData, ExpressionData, BivariateData, HighlightLayer, ColorMode } from '../store'

interface Props {
  slot: 'primary' | 'secondary' | 'comparison'
  embedding: EmbeddingData
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  bivariateData: BivariateData | null
  highlightLayers: HighlightLayer[]
  colorMode: ColorMode
  interactionMode: string
  selectedCellIndices: number[]
  onSelectionComplete: (indices: number[]) => void
  // 2D-only props accepted but ignored:
  onLineDrawn?: unknown
  onTransformEmbedding?: unknown
  onTransformEmbeddingSubset?: unknown
}

export default function ScatterPlot3D(props: Props) {
  const { embedding, colorBy, expressionData, bivariateData, highlightLayers, colorMode, interactionMode } = props
  const displayPreferences = useStore((s) => s.displayPreferences)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const selectedSet = useMemo(() => new Set(props.selectedCellIndices), [props.selectedCellIndices])

  const getColor = useCellColor({
    displayPreferences, activeCellMask, colorMode, colorBy,
    expressionData, bivariateData, highlightLayers, selectedSet,
  })

  // size:3 interleaved position buffer (x, y from coordinates; z alongside)
  const positionsBuf = useMemo(() => {
    const n = embedding.coordinates.length
    const buf = new Float32Array(n * 3)
    const z = embedding.z ?? []
    for (let i = 0; i < n; i++) {
      buf[i * 3] = embedding.coordinates[i][0]
      buf[i * 3 + 1] = embedding.coordinates[i][1]
      buf[i * 3 + 2] = z[i] ?? 0
    }
    return buf
  }, [embedding])

  // 3D bounds -> initial camera
  const bounds = useMemo(() => {
    const n = embedding.coordinates.length
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
    const z = embedding.z ?? []
    for (let i = 0; i < n; i++) {
      const x = embedding.coordinates[i][0], y = embedding.coordinates[i][1], zz = z[i] ?? 0
      if (x < minX) minX = x; if (x > maxX) maxX = x
      if (y < minY) minY = y; if (y > maxY) maxY = y
      if (zz < minZ) minZ = zz; if (zz > maxZ) maxZ = zz
    }
    return { minX, maxX, minY, maxY, minZ, maxZ }
  }, [embedding])

  const [viewState, setViewState] = useState<Record<string, unknown> | null>(null)
  const lastEmbeddingRef = useRef<string | null>(null)
  useEffect(() => {
    if (lastEmbeddingRef.current === embedding.name && viewState !== null) return
    lastEmbeddingRef.current = embedding.name
    const cx = (bounds.minX + bounds.maxX) / 2
    const cy = (bounds.minY + bounds.maxY) / 2
    const cz = (bounds.minZ + bounds.maxZ) / 2
    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) || 1
    setViewState({
      target: [cx, cy, cz],
      zoom: Math.log2(600 / span) - 1,
      rotationX: 30,
      rotationOrbit: 30,
      minZoom: -10, maxZoom: 10,
    })
  }, [embedding.name, bounds, viewState])

  // getRadius mirrors the 2D view (base/selected/masked). Read the 2D
  // baseRadius/selectedRadius/maskedRadius derivation (ScatterPlot.tsx, search
  // `const getRadius`) and reuse the same displayPreferences.pointSize-derived
  // values so 2D and 3D points match in size.
  const getRadius = (index: number): number => {
    if (activeCellMask !== null && !activeCellMask[index]) return (displayPreferences.pointSize ?? 3) * 0.6
    if (selectedSet.has(index)) return (displayPreferences.pointSize ?? 3) * 1.6
    return displayPreferences.pointSize ?? 3
  }

  const layer = useMemo(() => new ScatterplotLayer({
    id: `scatter3d-${props.slot}`,
    // Binary-attribute fast path — SAME shape the 2D layer uses (attributes
    // nested under data), only size:3 instead of size:2.
    data: {
      length: embedding.coordinates.length,
      attributes: { getPosition: { value: positionsBuf, size: 3 } },
    },
    getFillColor: (_o: unknown, info: { index: number }) => getColor({ index: info.index }),
    getRadius: (_o: unknown, info: { index: number }) => getRadius(info.index),
    // Robust default: constant screen-space size like 2D. (Optional depth cue:
    // switch to radiusUnits:'common' + a data-scale-relative radiusScale later;
    // 'pixels' is chosen first because embedding coordinate scales vary wildly.)
    radiusUnits: 'pixels',
    radiusMinPixels: 1,
    radiusMaxPixels: 20,
    billboard: true,            // discs always face the camera -> stay circular
    opacity: 1,                 // opaque; depth-tested (no per-frame sorting)
    pickable: true,
    parameters: { depthTest: true, depthMask: true },
    updateTriggers: {
      getFillColor: [colorBy, expressionData, bivariateData, highlightLayers, colorMode, displayPreferences, activeCellMask, selectedSet],
      getRadius: [displayPreferences.pointSize, activeCellMask, selectedSet],
    },
  }), [positionsBuf, getColor, embedding, displayPreferences, props.slot, colorBy, expressionData, bivariateData, highlightLayers, colorMode, activeCellMask, selectedSet])

  const view = useMemo(() => new OrbitView({ id: 'main', orbitAxis: 'Y', fov: 50 }), [])
  const orbitEnabled = interactionMode === 'pan'  // orbit only in navigate mode; lasso mode (Task 6) disables it

  const [hover, setHover] = useState<{ x: number; y: number; index: number } | null>(null)

  if (!viewState) return null
  return (
    <div style={{ position: 'absolute', inset: 0, backgroundColor: displayPreferences.backgroundColor }}>
      <DeckGL
        views={view}
        viewState={viewState}
        controller={orbitEnabled}
        onViewStateChange={(e: { viewState: Record<string, unknown> }) => setViewState(e.viewState)}
        layers={[layer]}
        onHover={(info: { index: number; x: number; y: number }) =>
          setHover(info.index >= 0 ? { x: info.x, y: info.y, index: info.index } : null)}
        style={{ position: 'absolute', width: '100%', height: '100%', background: 'transparent' }}
      />
      {hover && (
        <div style={{ position: 'absolute', left: hover.x + 8, top: hover.y + 8, pointerEvents: 'none',
          background: '#000a', color: '#fff', fontSize: 11, padding: '2px 6px', borderRadius: 4 }}>
          {`cell ${hover.index}`}
        </div>
      )}
    </div>
  )
}
```

> Notes for the implementer: The 2D `ScatterPlot` has no deck-level click-to-inspect handler (`handleClick` there drives polygon/line tools). So for parity, 3D provides a **hover tooltip** (above). If the app has a single-cell inspect action in the store, wire the deck `onClick` to it here; otherwise hover-only is correct parity. Read the 2D `ScatterplotLayer` (`ScatterPlot.tsx`, search `id: 'scatterplot'`) and its `getRadius` to confirm the radius values you mirror.

- [ ] **Step 2: Create `EmbeddingPlot.tsx`**

```tsx
import { useStore } from '../store'
import ScatterPlot from './ScatterPlot'
import ScatterPlot3D from './ScatterPlot3D'
import type { ComponentProps } from 'react'

type Props = ComponentProps<typeof ScatterPlot>

export default function EmbeddingPlot(props: Props) {
  const viewMode = useStore((s) => s.viewMode)
  const use3D = viewMode === '3d' && !!props.embedding?.z && props.embedding.z.length > 0
  return use3D ? <ScatterPlot3D {...(props as any)} /> : <ScatterPlot {...props} />
}
```

- [ ] **Step 3: Point the three render sites at `EmbeddingPlot`**

In `App.tsx`, add `import EmbeddingPlot from './components/EmbeddingPlot'`, and change each of the three `<ScatterPlot ... />` usages (~1902, ~1989, ~2068) to `<EmbeddingPlot ... />` (props unchanged). Keep the `import ScatterPlot` line only if still used elsewhere; otherwise remove it.

- [ ] **Step 4: Add Z dropdown + 2D/3D toggle to `DimensionPicker` (App.tsx ~668)**

Replace the `DimensionPicker` return with a version that adds the toggle and Z select (visible only in 3D). Read `viewMode`/`setViewMode` from the store; default Z to `2` when entering 3D:

```tsx
function DimensionPicker() {
  const selectedEmbedding = useStore((s) => s.selectedEmbedding)
  const embeddingDims = useStore((s) => s.embeddingDims)
  const setEmbeddingDims = useStore((s) => s.setEmbeddingDims)
  const schema = useStore((s) => s.schema)
  const viewMode = useStore((s) => s.viewMode)
  const setViewMode = useStore((s) => s.setViewMode)
  if (!selectedEmbedding || !schema) return null
  const ncols = schema.embedding_dims?.[selectedEmbedding] ?? 2
  if (ncols <= 2) {
    // Ensure we never get stuck in 3D on a 2-col embedding
    if (viewMode === '3d') setViewMode('2d')
    return null
  }
  const cur = embeddingDims[selectedEmbedding] ?? { x: 0, y: 1 }
  const names = schema.score_matrices?.[selectedEmbedding]
  const isPca = /pca/i.test(selectedEmbedding)
  const label = (i: number) => names?.[i] ?? (isPca ? `PC${i + 1}` : `dim ${i + 1}`)
  const opts = Array.from({ length: ncols }, (_, i) => i)
  const sel: React.CSSProperties = { padding: '3px 6px', fontSize: '11px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px', maxWidth: 150 }
  const curZ = cur.z ?? 2
  const toggle3D = () => {
    if (viewMode === '3d') { setViewMode('2d') }
    else { setEmbeddingDims(selectedEmbedding, cur.x, cur.y, curZ); setViewMode('3d') }
  }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
      <span style={{ fontSize: '11px', color: '#888' }}>Axes:</span>
      <select style={sel} value={cur.x} onChange={(e) => setEmbeddingDims(selectedEmbedding, Number(e.target.value), cur.y, viewMode === '3d' ? curZ : undefined)}>
        {opts.map((i) => <option key={i} value={i}>{label(i)}</option>)}
      </select>
      <span style={{ fontSize: '11px', color: '#888' }}>×</span>
      <select style={sel} value={cur.y} onChange={(e) => setEmbeddingDims(selectedEmbedding, cur.x, Number(e.target.value), viewMode === '3d' ? curZ : undefined)}>
        {opts.map((i) => <option key={i} value={i}>{label(i)}</option>)}
      </select>
      {viewMode === '3d' && (
        <>
          <span style={{ fontSize: '11px', color: '#888' }}>×</span>
          <select style={sel} value={curZ} onChange={(e) => setEmbeddingDims(selectedEmbedding, cur.x, cur.y, Number(e.target.value))}>
            {opts.map((i) => <option key={i} value={i}>{label(i)}</option>)}
          </select>
        </>
      )}
      <button style={{ ...sel, cursor: 'pointer', backgroundColor: viewMode === '3d' ? '#2a6f97' : '#0f3460' }} onClick={toggle3D} title="Toggle 3D view">
        {viewMode === '3d' ? '3D ✓' : '3D'}
      </button>
    </div>
  )
}
```

- [ ] **Step 5: Build**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 6: E2E — render + orbit + hover**

Start `pixi run backend` and `pixi run dev` (background). With Playwright: open `http://localhost:5173`, load a dataset that has a ≥3-column embedding, select `X_pca`. Confirm:
1. The `3D` toggle button is visible (and absent/hidden for a 2-column embedding like `X_umap` if 2-D).
2. Click `3D` → a Z dropdown appears and the plot re-renders as a 3D cloud (canvas present, **no console errors** — check `browser_console_messages`).
3. Drag on the canvas → the cloud rotates (take before/after screenshots).
4. Hover a point → the `cell N` tooltip appears at the cursor.

Kill :8000 and :5173.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/ScatterPlot3D.tsx frontend/src/components/EmbeddingPlot.tsx frontend/src/App.tsx
git commit -m "feat(frontend): 3D embedding view (OrbitView render, orbit, hover/click) + 2D/3D toggle"
```

---

## Task 6: 3D lasso selection

**Files:**
- Modify: `frontend/src/components/ScatterPlot3D.tsx` (add lasso overlay + selection)

**Interfaces:**
- Consumes: `pointsInLassoScreen` (Task 4), `props.onSelectionComplete` (already passed by all three sites).

- [ ] **Step 1: Add lasso state + SVG overlay**

In `ScatterPlot3D`, add a screen-space lasso captured while `interactionMode` indicates selection (match the 2D view's selection mode value — grep `interactionMode` / `selectionTool` in `ScatterPlot.tsx` to use the same predicate; below assumes a `lassoActive` boolean derived from those props/store):
- Track `const [lassoPts, setLassoPts] = useState<[number, number][]>([])` (screen coords relative to the canvas).
- On `pointerdown`/`pointermove` over the deck canvas while `lassoActive`, append points; render an SVG `<polyline>`/`<polygon>` overlay (absolute-positioned over the canvas) so the user sees the outline.
- While `lassoActive`, pass `controller={false}` to `DeckGL` so the drag draws instead of orbiting (extend `orbitEnabled` to `interactionMode === 'pan' && !lassoActive`).

- [ ] **Step 2: On lasso end, project + select**

On `pointerup`:
```tsx
import { pointsInLassoScreen } from '../lib/lasso3d'
import { OrbitView } from '@deck.gl/core'
// ...
const finishLasso = () => {
  if (lassoPts.length >= 3) {
    const { width, height } = canvasRef.current!.getBoundingClientRect()
    const viewport = new OrbitView({ id: 'main', orbitAxis: 'Y', fov: 50 })
      .makeViewport({ width, height, viewState: viewState as any })
    const idx = pointsInLassoScreen(embedding.coordinates, embedding.z ?? [], lassoPts, viewport)
    props.onSelectionComplete(idx)
  }
  setLassoPts([])
}
```
(`view.makeViewport({width,height,viewState})` yields a viewport whose `.project([x,y,z])` returns pixel coords matching the on-screen lasso. Verify the lasso points are in the same pixel space — both relative to the canvas top-left.)

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 4: E2E — lasso selects**

Servers up, open app, load ≥3-col embedding, toggle 3D, switch to the selection/lasso tool, drag a loop around a visible sub-region. Confirm the selected-cell count propagates (e.g. `CellPanel` shows the count, points recolor to the selected color). Rotate, re-lasso a smaller region, confirm the selection updates. Kill servers.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ScatterPlot3D.tsx
git commit -m "feat(frontend): 3D projection lasso selection (rotate + re-lasso to refine)"
```

---

## Task 7: Category labels in 3D

**Files:**
- Modify: `frontend/src/components/ScatterPlot3D.tsx`

**Interfaces:**
- Consumes: same category-label condition the 2D view uses (color column === embedding label column; `showCategoryLabels`). Grep `categoryLabelData` / `showCategoryLabels` / `embeddingLabelColumn` in `ScatterPlot.tsx` (~1368–1397) to mirror the predicate and centroid math.

- [ ] **Step 1: Compute 3D centroids + project each frame**

- When labels are active and `colorBy` is the embedding label column, compute each category's 3D centroid (mean of `coordinates[i]` + `z[i]` over cells in that category) in a `useMemo`.
- On `onAfterRender` (or whenever `viewState` changes), project each centroid via the current viewport (`view.makeViewport({width,height,viewState}).project([cx,cy,cz])`) to a pixel position, and render DOM `<div>` labels absolutely positioned over the canvas. Skip labels whose projected point is behind the camera (deck.gl `project` returns a z; drop if outside `[0,1]` depth or off-canvas).

- [ ] **Step 2: Build + E2E**

Run: `cd frontend && npm run build` (expected: succeeds). Then servers up, color by a categorical label column with labels enabled, toggle 3D, confirm category labels appear at cluster centers and track as you rotate. Confirm the 2D-only tools (draw shape, draw line, rotate/quilt) are **not** shown in 3D mode. Kill servers.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ScatterPlot3D.tsx
git commit -m "feat(frontend): projected category labels in 3D embedding view"
```

---

## Task 8: Full-flow E2E regression + cleanup

**Files:** none (verification only) — fix-forward into the relevant component if a defect is found.

- [ ] **Step 1: Backend suite**

Run: `cd "/Users/pcahan/Dropbox (Personal)/Code/xcell" && pixi run -e dev pytest backend/tests/ -q`
Expected: all pass.

- [ ] **Step 2: Frontend unit + build**

Run: `cd frontend && npm test && npm run build`
Expected: vitest passes; build succeeds.

- [ ] **Step 3: Playwright full flow**

Servers up. Verify end-to-end with a ≥3-column embedding:
1. 2-column embedding (e.g. a 2-D `X_umap`) → `3D` toggle is hidden/disabled.
2. `X_pca` → toggle 3D → 3D cloud renders, **zero console errors**.
3. Orbit (drag), zoom (wheel), pan (shift-drag) all respond smoothly.
4. Change Z dropdown to another column → cloud updates.
5. Color by category, then a gene → colors match the 2D view; category labels track rotation.
6. Lasso a region → selection count matches expectation and syncs to `CellPanel`.
7. Toggle back to 2D → the 2D view and all its tools return unchanged.

Capture screenshots of 2D and 3D of the same embedding for the record. Kill :8000 and :5173.

- [ ] **Step 4: Verify with the `verify` skill**

Invoke the repo's `verify` skill (or `/verify`) to drive the 3D flow once more and confirm behavior, per project convention.

- [ ] **Step 5: Final commit (if any fixes were made)**

```bash
git add -A && git commit -m "test: 3D embedding view end-to-end verification + fixes"
```

---

## Self-Review notes (author)

- **Spec coverage:** §3 backend → Task 1; §2 shared coloring → Task 2; §3.3/3.4 data model → Task 3; §5.3 lasso pure fn → Task 4; §4 render + §5.1/5.2 orbit/hover/click + §6 toggle/Z picker → Task 5; §5.3 lasso wired → Task 6; §5.4 category labels + §6 hidden 2D tools → Task 7; §7 tests → Tasks 1/4/8. All spec sections mapped.
- **Type consistency:** `useCellColor(CellColorParams)`, `pointsInLassoScreen(coordinates, z, polygonScreen, viewport)`, `EmbeddingData.z`/`dim_z`, `embeddingDims[].z`, `viewMode`/`setViewMode`, `setEmbeddingDims(name, x, y, z?)` used identically across tasks.
- **Known soft spots (validate during execution, not placeholders):** exact `ScatterplotLayer` radius/opacity/tooltip/click props and the `interactionMode`/selection-tool predicate must be read from `ScatterPlot.tsx` and mirrored (called out in Tasks 5–7); deck.gl v9 `makeViewport`/`project` pixel-space must be confirmed to match the SVG lasso's canvas-relative coordinates (Task 6 Step 2).
