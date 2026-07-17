# 3D Embedding View — Design

**Date:** 2026-07-17
**Status:** Approved (design)
**Scope:** View existing ≥3-dimensional `.obsm` embeddings in an interactive 3D
scatter, with orbit navigation, hover/click inspection, all existing coloring
modes, category labels, and 3D lasso selection. View-only (no 3D compute).

---

## 1. Motivation & goals

Users can already view any two columns of a multi-dimensional `.obsm` embedding
(PCA, gene-set scores, a 3D UMAP they computed elsewhere). This adds a **3D
mode**: pick three columns (X/Y/Z) and explore the cloud with an orbiting
camera.

The overriding non-functional requirement is **smooth interaction** (60fps
orbit/zoom on realistic datasets — hundreds of thousands of cells). This is
achievable because the app already renders on the GPU via deck.gl; 3D reuses the
same binary-buffer fast path.

**In scope**
- Render an existing embedding's three chosen columns as a 3D `OrbitView`.
- Orbit / zoom / pan; hover + click to inspect a cell.
- Every existing coloring mode (categorical, continuous/numeric, gene
  expression, bivariate, highlight overlays, cell masking) and category labels.
- 3D lasso selection that captures every cell whose on-screen projection falls
  inside the outline (rotate → re-lasso to refine), synced to the rest of the app.

**Out of scope (YAGNI)**
- Computing new 3D embeddings (n_components=3 UMAP/t-SNE) — view-only.
- Depth-band / occlusion-aware selection — the lasso projects through the whole
  cloud by design.
- Bringing 2D editing tools (shape drawing, line/trajectory drawing, embedding
  rotate/"quilt" transforms) into 3D. These stay exclusively 2D.

---

## 2. Key decisions

1. **Separate component, not a branch inside `ScatterPlot.tsx`.**
   `ScatterPlot.tsx` (~75 KB) is saturated with 2D-only machinery. 3D lives in a
   new `ScatterPlot3D.tsx` with its own view/layer/interactions. The parent
   chooses which to render.

2. **Extract shared coloring into a hook.** Today `ScatterPlot.tsx` builds a
   `getColor(d)` memo over (`colorBy`, `expressionData`, `bivariateData`,
   `highlightLayers`, `colorMode`, `selectedSet`, `displayPreferences`,
   `activeCellMask`). Extract this into `frontend/src/lib/cellColors.ts` as
   `useCellColor(...)` so 2D and 3D share one source of truth and coloring can
   never drift between them. This is the single targeted refactor.

3. **`coordinates` stays `[number, number][]`.** The third dimension rides
   alongside as `z?: number[]`, so no existing 2D code path changes. Only the 3D
   path reads `z`.

4. **Opaque points + depth test.** Avoids the one genuine 3D perf trap
   (per-frame depth sorting required for correct alpha blending) and reads better
   visually (you don't see through the whole cloud).

5. **Lasso captures everything inside the outline** (projection-based, no depth
   limit). Predictable; refine by rotating and re-lassoing.

---

## 3. Backend + data model

### 3.1 `adaptor.get_embedding` (`backend/xcell/adaptor.py`)

Add an optional `dim_z`:

```python
def get_embedding(self, name, dim_x=0, dim_y=1, dim_z=None) -> dict[str, Any]:
    ...
    dx, dy = self._clamp_dims(name, dim_x, dim_y)
    coords_2d = self.adata.obsm[name][:, [dx, dy]]
    result = {
        "name": name,
        "coordinates": coords_2d.tolist(),
        "dim_x": dx, "dim_y": dy,
    }
    if dim_z is not None:
        dz = self._clamp_one_dim(name, dim_z)   # clamp to n_cols-1, fallback 2→0
        result["z"] = self.adata.obsm[name][:, dz].tolist()
        result["dim_z"] = dz
    return result
```

- `dim_z=None` (default) → response is byte-identical to today.
- Introduce a small `_clamp_one_dim(name, d)` helper (or reuse/extend
  `_clamp_dims`) that clamps a single index into `[0, n_cols-1]`.
- **No new metadata endpoint needed.** `get_schema` already returns
  `embedding_dims: dict[str, int]` (each embedding's column count, from
  `adaptor.py:507`). The frontend uses `schema.embedding_dims[name]` to decide
  whether 3D is available (≥ 3) and to populate the Z dropdown's options.

### 3.2 REST route (`backend/xcell/api/routes.py`)

```python
def get_embedding(name, dim_x=Query(0), dim_y=Query(1),
                  dim_z: int | None = Query(None), dataset=Query(None)):
    ...
    return adaptor.get_embedding(name, dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)
```

### 3.3 Frontend store (`frontend/src/store.ts`)

```ts
export interface EmbeddingData {
  name: string
  coordinates: [number, number][]   // unchanged — X, Y
  dim_x?: number
  dim_y?: number
  z?: number[]                      // NEW — third column, present only in 3D
  dim_z?: number                    // NEW
}
```

- `embeddingDims[name]` gains optional `z?: number`.
- `viewMode: '2d' | '3d'` added to the store (default `'2d'`), with a setter.

### 3.4 Fetch hook (`frontend/src/hooks/useData.ts`)

`useEmbedding()` includes `dim_z` in the URL and the "already loaded?" guard when
`viewMode === '3d'` and a `z` dim is selected. Switching `viewMode` or the Z
column triggers a refetch exactly like changing `dim_x`/`dim_y` does today.

---

## 4. Rendering (`frontend/src/components/ScatterPlot3D.tsx`)

- **View:** `new OrbitView({ id: 'main', orbitAxis: 'Y', fov: 50 })` replacing
  `OrthographicView`.
- **viewState:** `{ target, zoom, rotationX, rotationOrbit }`. On embedding
  *identity* change, initialize `target` to the cloud centroid and `zoom` to fit
  the 3D bounding box (mirrors the existing 2D bounds-fit logic, extended to Z).
  In-place data updates keep the same viewState (same rule as 2D).
- **Layer:** reuse `ScatterplotLayer` with
  - `getPosition: { value: positionsBuf, size: 3 }` — interleaved X (from
    `coordinates[i][0]`), Y (`coordinates[i][1]`), Z (`z[i]`) in a `Float32Array`.
  - `getFillColor` via the shared `useCellColor` accessor.
  - `billboard: true` (discs always face the camera → stay circular at any angle).
  - `radiusUnits: 'common'` (nearer points render larger → perspective depth cue).
  - Opaque fill (alpha 255) + `parameters: { depthTest: true, depthMask: true }`.
- **Perf note:** same GPU binary-attribute path as 2D; expected 60fps at
  hundreds of thousands of points. No per-frame JS over the point set during
  orbit.

---

## 5. Interaction

### 5.1 Navigation
`OrbitController` (drag = rotate, scroll = zoom, shift-drag = pan). Enabled only
in navigate mode; disabled while the lasso tool is active so the drag draws the
lasso — mirrors the 2D `controller={interactionMode === 'pan'}` pattern.

### 5.2 Hover / click
deck.gl GPU picking (`pickable: true`, `onHover`, `onClick`) → the same tooltip
and cell-inspect callbacks the 2D view already receives as props.

### 5.3 Lasso selection (projection-based)
On lasso completion (mouseup):
1. Build an `OrbitViewport` from the current `viewState` + canvas width/height.
2. Project every cell `[x, y, z] → [sx, sy]` via `viewport.project`.
3. Point-in-polygon test each projected point against the screen-space lasso.
4. Collect matching indices → call the existing `onSelectionComplete(indices)`
   prop (same one 2D uses), so selection flows to `CellPanel` etc. unchanged.

The projection + point-in-polygon step is extracted as a **pure, WebGL-free
function**:

```ts
// frontend/src/lib/lasso3d.ts
export function pointsInLassoScreen(
  coordinates: [number, number][],
  z: number[],
  polygonScreen: [number, number][],
  viewport: { project(p: number[]): number[] },
): number[]
```

so it is unit-testable with a stub viewport. Projecting a few hundred k points in
JS on a one-shot mouseup is tens of ms — acceptable; noted as the first place to
optimize (e.g. a GPU pass) if a dataset ever makes it feel slow.

### 5.4 Category labels
When the active color column is the embedding label column (same condition as 2D),
compute each category's **3D centroid** once, then project centroids to screen on
`onAfterRender` (and on viewState change) to position the DOM label overlay.

---

## 6. UI / entry point

- A **2D / 3D toggle** in the embedding toolbar, adjacent to the column picker.
  Disabled (with tooltip) when the selected embedding has < 3 columns.
- The column picker gains a **Z dropdown**, shown only in 3D. Switching to 3D
  with no Z chosen defaults Z to column index 2 (or the last column if only 3
  exist).
- **Parent switch:** `GenePanel`/`App` renders `<ScatterPlot3D>` when
  `viewMode === '3d'`, else `<ScatterPlot>`. Both receive the same selection,
  hover, and cell-inspect props.
- **Hidden in 3D** (restored on return to 2D): shape drawing, line/trajectory
  drawing, embedding rotate/quilt, SVG line overlays. **Active in 3D:** coloring
  controls, legend, masking, snapshots.
- **Snapshots:** reuse the existing canvas-capture path; a 3D snapshot captures
  the current viewpoint.

---

## 7. Testing & verification

- **Backend unit tests** (`backend/tests/`):
  - `get_embedding(..., dim_z=k)` returns the correct third column and echoes
    `dim_z`; `z` length == n_cells.
  - `dim_z=None` output is unchanged from current behavior.
  - `dim_z` out of range clamps gracefully.
- **Frontend unit test** (`lasso3d.ts`): `pointsInLassoScreen` with a stub
  viewport and a known polygon returns exactly the expected indices (points
  inside vs. outside, incl. a point projecting onto the polygon edge).
- **Playwright E2E**: load a dataset with a ≥3-column embedding → toggle 3D →
  assert canvas renders with no console errors → drag to orbit (viewState
  changes) → draw a lasso → assert the resulting selection count propagates to
  `CellPanel`. Also assert the 3D toggle is disabled for a 2-column embedding.

---

## 8. Component / file summary

| File | Change |
|------|--------|
| `backend/xcell/adaptor.py` | `get_embedding` gains `dim_z`; add `_clamp_one_dim` (schema already reports `embedding_dims`) |
| `backend/xcell/api/routes.py` | `get_embedding` route gains `dim_z` query param |
| `backend/tests/…` | new tests for `dim_z` behavior |
| `frontend/src/store.ts` | `EmbeddingData.z`/`dim_z`; `embeddingDims[].z`; `viewMode` state |
| `frontend/src/hooks/useData.ts` | `useEmbedding` sends `dim_z` in 3D |
| `frontend/src/lib/cellColors.ts` | NEW — `useCellColor` extracted from `ScatterPlot.tsx` |
| `frontend/src/lib/lasso3d.ts` | NEW — `pointsInLassoScreen` pure function |
| `frontend/src/components/ScatterPlot.tsx` | consume `useCellColor` (behavior unchanged) |
| `frontend/src/components/ScatterPlot3D.tsx` | NEW — OrbitView, 3D layer, 3D interactions |
| `frontend/src/App.tsx` / `GenePanel.tsx` | 2D/3D toggle, Z dropdown, render switch |

---

## 9. Risks & mitigations

- **Coloring-refactor ripple.** Extracting `useCellColor` touches a hot path in
  `ScatterPlot.tsx`. Mitigation: extract as a behavior-preserving move first,
  verify 2D is visually identical (Playwright) before building 3D on top.
- **Lasso projection cost on very large datasets.** Mitigation: pure function is
  isolated and swappable for a GPU pass later; it runs once per lasso, not per
  frame.
- **Transparency expectations.** Users used to semi-transparent 2D points may
  notice opaque 3D points. Mitigation: this is intentional (perf + depth); can
  revisit with a depth-sorted transparent mode later if requested.
