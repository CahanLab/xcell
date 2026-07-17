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

  // Radius derivation mirrors ScatterPlot exactly so 2D and 3D point sizes match.
  const baseRadius = displayPreferences.pointSize
  const selectedRadius = baseRadius + 2
  const maskedRadius = Math.max(1, baseRadius / 3) // 1/3 size for masked cells
  const getRadius = (index: number): number => {
    if (activeCellMask !== null && !activeCellMask[index]) return maskedRadius
    if (selectedSet.has(index)) return selectedRadius
    return baseRadius
  }

  // Y is the orbit axis (points spin around vertical); perspective projection.
  const view = useMemo(() => new OrbitView({ id: 'main', orbitAxis: 'Y', fovy: 50 }), [])

  // Lasso selection — mirrors the 2D freehand lasso predicate: interaction mode
  // 'lasso' with the freehand ('lasso') selection tool active (polygon tool is a
  // 2D-only click affordance, not implemented here).
  const selectionTool = useStore((state) => state.selectionTool)
  const lassoActive = interactionMode === 'lasso' && selectionTool === 'lasso'
  // Orbit only in navigate ('pan') mode — same predicate the 2D controller uses.
  // Extend with !lassoActive so a lasso drag never orbits (pan ≠ lasso already,
  // so this is defensive; keeps orbit ON in pan mode).
  const orbitEnabled = interactionMode === 'pan' && !lassoActive

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
      getRadius: [selectedCellIndices, displayPreferences.pointSize, activeCellMask],
    },
  })

  if (!viewState) return null

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
