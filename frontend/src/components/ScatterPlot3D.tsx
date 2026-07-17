import { useEffect, useMemo, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrbitView, OrbitViewState, PickingInfo } from '@deck.gl/core'
import {
  useStore,
  EmbeddingData,
  ObsColumnData,
  ExpressionData,
  BivariateExpressionData,
  HighlightLayer,
  ColorMode,
  InteractionMode,
  DatasetSlot,
} from '../store'
import { useCellColor } from '../lib/cellColors'
import { pointsInLassoScreen } from '../lib/lasso3d'

// 3D sibling of ScatterPlot: renders the embedding as an orbitable point cloud
// using deck.gl's OrbitView. Deliberately mirrors ScatterPlot's ScatterplotLayer
// (binary getPosition attribute, radius derivation, controller-enable predicate,
// shared useCellColor coloring) so 2D and 3D render identically apart from the
// extra Z axis and camera. No lasso/draw/adjust here — those stay 2D-only; the
// 3D inspect affordance is a hover tooltip (Task 6 adds 3D lasso).
interface Props {
  slot?: DatasetSlot
  embedding: EmbeddingData
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  bivariateData: BivariateExpressionData | null
  highlightLayers: HighlightLayer[]
  colorMode: ColorMode
  interactionMode: InteractionMode
  selectedCellIndices: number[]
  onSelectionComplete: (indices: number[], additive: boolean) => void
  // 2D-only callbacks — accepted (so the same props object flows through
  // EmbeddingPlot) but unused here:
  onLineDrawn?: unknown
  onTransformEmbedding?: unknown
  onTransformEmbeddingSubset?: unknown
}

export default function ScatterPlot3D({
  slot,
  embedding,
  colorBy,
  expressionData,
  bivariateData,
  highlightLayers,
  colorMode,
  interactionMode,
  selectedCellIndices,
  onSelectionComplete,
}: Props) {
  // Read display prefs / mask from the per-slot dataset when a slot is given,
  // else the flat top-level fields — same selector pattern as ScatterPlot.
  const displayPreferences = useStore((state) =>
    slot ? state.datasets[slot].displayPreferences : state.displayPreferences
  )
  const activeCellMask = useStore((state) =>
    slot ? state.datasets[slot].activeCellMask : state.activeCellMask
  )
  // When false, masked (inactive) cells are hidden entirely (2D drops them from
  // the render set; here we hide them via getRadius → 0, see below).
  const showMaskedCells = useStore((state) =>
    slot ? state.datasets[slot].showMaskedCells : state.showMaskedCells
  )
  // Which column the user chose to label at cluster centroids (global toggle,
  // same store field the 2D ScatterPlot reads). Labels show only when THIS
  // plot is colored by that column.
  const embeddingLabelColumn = useStore((state) => state.embeddingLabelColumn)

  const selectedSet = useMemo(() => new Set(selectedCellIndices), [selectedCellIndices])

  // Exact same argument object ScatterPlot passes to useCellColor.
  const getColor = useCellColor({
    displayPreferences, activeCellMask, colorMode, colorBy,
    expressionData, bivariateData, highlightLayers, selectedSet,
  })

  // Interleaved size:3 position buffer (x, y from `coordinates`; z alongside).
  // Built in original cell order, so the buffer index === cell index and the
  // index-keyed getColor/getRadius accessors work directly (unlike 2D, which
  // reorders `data` and looks cells back up).
  const positionsBuf = useMemo(() => {
    const coords = embedding.coordinates
    const n = coords.length
    const buf = new Float32Array(n * 3)
    const z = embedding.z ?? []
    for (let i = 0; i < n; i++) {
      buf[i * 3] = coords[i][0]
      buf[i * 3 + 1] = coords[i][1]
      buf[i * 3 + 2] = z[i] ?? 0
    }
    return buf
  }, [embedding])

  // 3D bounds → initial camera. Derived from embedding only so the camera does
  // NOT reset on color/mask/selection changes (mirrors ScatterPlot's bounds memo).
  const bounds = useMemo(() => {
    const coords = embedding.coordinates
    const z = embedding.z ?? []
    if (coords.length === 0) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1, minZ: 0, maxZ: 1 }
    }
    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity
    let minZ = Infinity, maxZ = -Infinity
    for (let i = 0; i < coords.length; i++) {
      const x = coords[i][0], y = coords[i][1], zz = z[i] ?? 0
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (zz < minZ) minZ = zz
      if (zz > maxZ) maxZ = zz
    }
    return { minX, maxX, minY, maxY, minZ, maxZ }
  }, [embedding])

  // Initialize the camera when the embedding *identity* changes (first render or
  // the user switches embeddings) — NOT on in-place data/color updates. Same
  // lastEmbeddingRef guard ScatterPlot uses so ops don't snap the view back.
  const [viewState, setViewState] = useState<OrbitViewState | null>(null)
  const lastEmbeddingRef = useRef<string | null>(null)
  useEffect(() => {
    if (lastEmbeddingRef.current === embedding.name && viewState !== null) return
    lastEmbeddingRef.current = embedding.name
    const cx = (bounds.minX + bounds.maxX) / 2
    const cy = (bounds.minY + bounds.maxY) / 2
    const cz = (bounds.minZ + bounds.maxZ) / 2
    const span = Math.max(
      bounds.maxX - bounds.minX,
      bounds.maxY - bounds.minY,
      bounds.maxZ - bounds.minZ,
    ) || 1
    setViewState({
      target: [cx, cy, cz],
      zoom: Math.log2(600 / span) - 1,
      rotationX: 30,      // tilt down ~30° for a legible default 3D angle
      rotationOrbit: 30,  // spin ~30° around the orbit (Y) axis
      minZoom: -10,
      maxZoom: 10,
    })
  }, [embedding.name, bounds, viewState])

  // Force a re-render right after viewState first becomes non-null. Until then
  // the component returns null (no container), so containerRef.current is still
  // null when projectedLabels is first computed in the render body — the ref
  // only attaches once the container commits. Keying on `viewState === null`
  // fires this the moment the container exists (true -> false, once), painting
  // labels on entry instead of waiting for the first orbit/hover. It does not
  // re-fire during orbiting (viewState stays non-null).
  const [, forceTick] = useState(0)
  useEffect(() => {
    forceTick((t) => t + 1)
  }, [viewState === null])

  // Radius derivation mirrors ScatterPlot exactly so 2D and 3D point sizes match.
  const baseRadius = displayPreferences.pointSize
  const selectedRadius = baseRadius + 2
  const maskedRadius = Math.max(1, baseRadius / 3) // 1/3 size for masked cells
  const getRadius = (index: number): number => {
    if (activeCellMask !== null && !activeCellMask[index]) {
      // Hide masked cells entirely when the user turned them off (radius 0 =
      // invisible and effectively unpickable). Keeps the position buffer intact
      // so buffer index === cell index for coloring / picking / lasso.
      return showMaskedCells === false ? 0 : maskedRadius
    }
    if (selectedSet.has(index)) return selectedRadius
    return baseRadius
  }

  // Categorical text labels overlaid at cluster centroids. Mirrors the 2D
  // ScatterPlot predicate exactly: active only when this plot is colored (in
  // metadata mode) by the categorical column the user chose to label.
  const showCategoryLabels =
    colorMode === 'metadata' &&
    !!colorBy &&
    colorBy.dtype === 'category' &&
    colorBy.name === embeddingLabelColumn

  // Each category's 3D centroid (mean of x/y from `coordinates` + z), grouped
  // by category the same way 2D does. Memoized on [colorBy, embedding] so it
  // recomputes only when the coloring column or embedding changes — NOT on
  // every orbit (projection to screen happens per-render below, cheaply).
  const categoryCentroids3D = useMemo(() => {
    if (!showCategoryLabels || !colorBy?.categories) {
      return [] as { label: string; x: number; y: number; z: number }[]
    }
    const coords = embedding.coordinates
    const zArr = embedding.z ?? []
    const values = colorBy.values
    const cats = colorBy.categories
    // Running sums per category name → mean centroid in one pass.
    const sums: Record<string, { x: number; y: number; z: number; n: number }> = {}
    const n = Math.min(coords.length, values.length)
    for (let i = 0; i < n; i++) {
      const v = values[i]
      if (v === null || v === undefined) continue
      // Categorical .obs values arrive as numeric codes indexing `categories`.
      const name = typeof v === 'number' ? (cats[v] ?? String(v)) : String(v)
      if (name === 'unassigned' || name === 'nan' || name === 'NaN') continue
      const s = (sums[name] ||= { x: 0, y: 0, z: 0, n: 0 })
      s.x += coords[i][0]
      s.y += coords[i][1]
      s.z += zArr[i] ?? 0
      s.n += 1
    }
    return Object.keys(sums)
      .filter((k) => sums[k].n > 0)
      .map((k) => {
        const s = sums[k]
        return { label: k, x: s.x / s.n, y: s.y / s.n, z: s.z / s.n }
      })
  }, [showCategoryLabels, colorBy, embedding])

  // Y is the orbit axis (points spin around vertical); perspective projection.
  const view = useMemo(() => new OrbitView({ id: 'main', orbitAxis: 'Y', fovy: 50 }), [])

  // Lasso selection — ANY lasso interaction mode drives a freehand loop in 3D.
  // The 2D polygon sub-tool is a click affordance that simply doesn't apply here,
  // so we don't gate on selectionTool: treating every lasso-mode drag as freehand
  // avoids a dead state when the user last picked the polygon sub-tool.
  const lassoActive = interactionMode === 'lasso'
  // Orbit whenever NOT actively lassoing. Any non-lasso interactionMode (pan,
  // draw, adjust, quilt, …) still orbits, so the 3D view can never freeze — 2D-only
  // tools are hidden in 3D anyway, but this keeps the camera robust regardless.
  const orbitEnabled = !lassoActive

  const [hover, setHover] = useState<{ x: number; y: number; index: number } | null>(null)

  // Freehand lasso polygon in CANVAS-RELATIVE pixels (top-left origin) — the SAME
  // space deck.gl's viewport.project returns, so the point-in-polygon test lines
  // up exactly with what's on screen. containerRef wraps the deck canvas; its
  // getBoundingClientRect gives both the rect (for clientX/Y → canvas px) and the
  // width/height fed to makeViewport.
  const containerRef = useRef<HTMLDivElement>(null)
  const [lassoPts, setLassoPts] = useState<[number, number][]>([])
  const isLassoing = useRef(false)

  const toCanvasPoint = (e: React.PointerEvent): [number, number] | null => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return null
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  const handleLassoDown = (e: React.PointerEvent) => {
    if (!lassoActive) return
    const p = toCanvasPoint(e)
    if (!p) return
    isLassoing.current = true
    setLassoPts([p])
    // Capture so a drag that leaves the canvas still streams move/up events here.
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }

  const handleLassoMove = (e: React.PointerEvent) => {
    if (!lassoActive || !isLassoing.current) return
    const p = toCanvasPoint(e)
    if (!p) return
    setLassoPts((prev) => [...prev, p])
  }

  const finishLasso = (e: React.PointerEvent) => {
    if (!isLassoing.current) return
    isLassoing.current = false
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    const pts = lassoPts
    // Need ≥3 vertices AND a live camera to build a viewport.
    if (pts.length >= 3 && viewState && containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect()
      // Build a viewport from the CURRENT camera using the SAME view instance deck
      // renders with (fovy/orbitAxis identical), so project() output pixel space ===
      // the lasso polygon pixel space (both canvas-relative, top-left origin).
      const viewport = view.makeViewport({ width, height, viewState: viewState as never })
      if (viewport) {
        const idx = pointsInLassoScreen(embedding.coordinates, embedding.z ?? [], pts, viewport)
        // Shift = add to existing selection (mirrors 2D).
        onSelectionComplete(idx, e.shiftKey)
      }
    }
    setLassoPts([])
  }

  // Constructed inline each render (like ScatterPlot's `layers`); deck.gl
  // reconciles via `id` + updateTriggers, and fresh closures keep getColor/
  // getRadius current.
  const layer = new ScatterplotLayer({
    id: `scatterplot3d-${slot ?? 'single'}`,
    // Binary fast path — SAME shape as the 2D layer (attributes nested under
    // data), only size:3. Also sidesteps the deck.gl 9.2.6 per-cell accessor
    // stride bug documented in ScatterPlot.tsx.
    data: {
      length: embedding.coordinates.length,
      attributes: {
        getPosition: { value: positionsBuf, size: 3 },
      },
    },
    // With binary data, accessors are called as (undefined, { index, ... }); the
    // buffer is in cell order so info.index is the cell index directly.
    getFillColor: (_obj: unknown, info: { index: number }) => getColor({ index: info.index }),
    getRadius: (_obj: unknown, info: { index: number }) => getRadius(info.index),
    // Constant screen-space size like 2D — robust across wildly varying
    // embedding coordinate scales (PCA vs UMAP). billboard keeps discs circular
    // and camera-facing regardless of orbit angle.
    radiusUnits: 'pixels',
    radiusMinPixels: 1,
    radiusMaxPixels: 20,
    billboard: true,
    pickable: true,
    // Opaque, depth-tested so nearer points occlude farther ones (no per-frame
    // CPU sorting). deck.gl v9 / luma.gl v9 parameter names: depthWriteEnabled +
    // depthCompare (the old depthTest/depthMask keys are ignored at runtime).
    parameters: {
      depthWriteEnabled: true,
      depthCompare: 'less-equal',
    },
    updateTriggers: {
      // Mirror useCellColor's deps so deck.gl rebuilds the GPU color buffer
      // whenever the color function changes (same set ScatterPlot keys on).
      getFillColor: [colorBy, expressionData, bivariateData, highlightLayers, colorMode, selectedSet, displayPreferences, activeCellMask],
      getRadius: [selectedCellIndices, displayPreferences.pointSize, activeCellMask, showMaskedCells],
    },
  })

  if (!viewState) return null

  // Project each category centroid to screen for the label overlay. Computed
  // every render (not memoized): orbiting fires setViewState → re-render, so
  // recomputing here makes the labels track the camera each frame. Uses the
  // SAME `view` instance deck renders with, so pixel space === the canvas.
  const projectedLabels: { label: string; x: number; y: number }[] = []
  if (categoryCentroids3D.length > 0 && containerRef.current) {
    const { width, height } = containerRef.current.getBoundingClientRect()
    const viewport = view.makeViewport({ width, height, viewState: viewState as never })
    if (viewport) {
      for (const c of categoryCentroids3D) {
        const [sx, sy, sz] = viewport.project([c.x, c.y, c.z])
        // Cull behind-camera / clipped: deck's project returns a depth `sz` that
        // falls outside [0,1] when the point is behind the near plane. Also cull
        // non-finite projections and anything off the canvas.
        if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue
        if (sz < 0 || sz > 1) continue
        if (sx < 0 || sx > width || sy < 0 || sy > height) continue
        projectedLabels.push({ label: c.label, x: sx, y: sy })
      }
    }
  }

  return (
    <div
      ref={containerRef}
      style={{ position: 'absolute', inset: 0, backgroundColor: displayPreferences.backgroundColor }}
      onPointerDown={handleLassoDown}
      onPointerMove={handleLassoMove}
      onPointerUp={finishLasso}
    >
      <DeckGL
        views={view}
        viewState={viewState}
        controller={orbitEnabled}
        onViewStateChange={({ viewState: next }) => setViewState(next as OrbitViewState)}
        layers={[layer]}
        onHover={(info: PickingInfo) =>
          setHover(info.index >= 0 ? { x: info.x, y: info.y, index: info.index } : null)}
        style={{ position: 'absolute', width: '100%', height: '100%', background: 'transparent' }}
        getCursor={() => (lassoActive ? 'crosshair' : orbitEnabled ? 'grab' : 'default')}
      />

      {/* Freehand lasso outline — SVG over the canvas, canvas-relative px, so it
          traces exactly where the pointer went. Same styling as the 2D lasso. */}
      {lassoActive && lassoPts.length > 1 && (
        <svg
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <polygon
            points={lassoPts.map((p) => `${p[0]},${p[1]}`).join(' ')}
            fill="rgba(233, 69, 96, 0.2)"
            stroke="#e94560"
            strokeWidth={2}
            strokeDasharray="5,5"
          />
        </svg>
      )}
      {/* Categorical labels at cluster centroids, projected from 3D each frame
          so they track the orbit. White bold text with a dark halo (text-shadow
          mirrors the 2D SVG stroke halo) for legibility over any background. */}
      {projectedLabels.map((c) => (
        <div
          key={c.label}
          style={{
            position: 'absolute',
            left: c.x,
            top: c.y,
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            fontSize: 13,
            fontWeight: 700,
            color: '#fff',
            textShadow:
              '-1.5px -1.5px 0 #000, 1.5px -1.5px 0 #000, -1.5px 1.5px 0 #000, 1.5px 1.5px 0 #000, 0 0 3px #000',
            whiteSpace: 'nowrap',
            zIndex: 15,
          }}
        >
          {c.label}
        </div>
      ))}
      {hover && (
        <div
          style={{
            position: 'absolute',
            left: hover.x + 8,
            top: hover.y + 8,
            pointerEvents: 'none',
            background: 'rgba(0,0,0,0.8)',
            color: '#fff',
            fontSize: 11,
            padding: '2px 6px',
            borderRadius: 4,
            whiteSpace: 'nowrap',
            zIndex: 20,
          }}
        >
          {`cell ${hover.index}`}
        </div>
      )}
    </div>
  )
}
