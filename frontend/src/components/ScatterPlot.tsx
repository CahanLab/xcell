import { useMemo, useState, useRef, useCallback, useEffect } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView, OrthographicViewState } from '@deck.gl/core'
import { useStore, EmbeddingData, ObsColumnData, ExpressionData, ColorMode, InteractionMode, ColorScale } from '../store'

interface ScatterPlotProps {
  embedding: EmbeddingData
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  colorMode: ColorMode
  interactionMode: InteractionMode
  selectedCellIndices: number[]
  onSelectionComplete: (indices: number[]) => void
}

// Color palette for categorical data (similar to d3 category10)
const CATEGORY_COLORS: [number, number, number][] = [
  [31, 119, 180],
  [255, 127, 14],
  [44, 160, 44],
  [214, 39, 40],
  [148, 103, 189],
  [140, 86, 75],
  [227, 119, 194],
  [127, 127, 127],
  [188, 189, 34],
  [23, 190, 207],
]

const DEFAULT_COLOR: [number, number, number, number] = [100, 149, 237, 200]
const SELECTED_COLOR: [number, number, number, number] = [255, 255, 0, 255]

// Linear interpolation helper
function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t)
}

// Interpolate between color stops
function interpolateStops(
  t: number,
  stops: { pos: number; color: [number, number, number] }[]
): [number, number, number] {
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].pos && t <= stops[i + 1].pos) {
      const localT = (t - stops[i].pos) / (stops[i + 1].pos - stops[i].pos)
      return [
        lerp(stops[i].color[0], stops[i + 1].color[0], localT),
        lerp(stops[i].color[1], stops[i + 1].color[1], localT),
        lerp(stops[i].color[2], stops[i + 1].color[2], localT),
      ]
    }
  }
  return stops[stops.length - 1].color
}

// Color scale definitions
const COLOR_SCALES: Record<ColorScale, { pos: number; color: [number, number, number] }[]> = {
  viridis: [
    { pos: 0, color: [68, 1, 84] },
    { pos: 0.25, color: [59, 82, 139] },
    { pos: 0.5, color: [33, 145, 140] },
    { pos: 0.75, color: [94, 201, 98] },
    { pos: 1, color: [253, 231, 37] },
  ],
  plasma: [
    { pos: 0, color: [13, 8, 135] },
    { pos: 0.25, color: [126, 3, 168] },
    { pos: 0.5, color: [204, 71, 120] },
    { pos: 0.75, color: [248, 149, 64] },
    { pos: 1, color: [240, 249, 33] },
  ],
  magma: [
    { pos: 0, color: [0, 0, 4] },
    { pos: 0.25, color: [81, 18, 124] },
    { pos: 0.5, color: [183, 55, 121] },
    { pos: 0.75, color: [252, 137, 97] },
    { pos: 1, color: [252, 253, 191] },
  ],
  inferno: [
    { pos: 0, color: [0, 0, 4] },
    { pos: 0.2, color: [66, 10, 104] },
    { pos: 0.4, color: [147, 38, 103] },
    { pos: 0.6, color: [221, 81, 58] },
    { pos: 0.8, color: [252, 165, 10] },
    { pos: 1, color: [252, 255, 164] },
  ],
  cividis: [
    { pos: 0, color: [0, 32, 81] },
    { pos: 0.33, color: [82, 95, 110] },
    { pos: 0.66, color: [152, 136, 62] },
    { pos: 1, color: [253, 234, 69] },
  ],
  coolwarm: [
    { pos: 0, color: [59, 76, 192] },
    { pos: 0.25, color: [112, 146, 208] },
    { pos: 0.5, color: [197, 197, 197] },
    { pos: 0.75, color: [230, 128, 103] },
    { pos: 1, color: [180, 4, 38] },
  ],
  blues: [
    { pos: 0, color: [247, 251, 255] },
    { pos: 0.5, color: [107, 174, 214] },
    { pos: 1, color: [8, 48, 107] },
  ],
  reds: [
    { pos: 0, color: [255, 245, 240] },
    { pos: 0.5, color: [251, 106, 74] },
    { pos: 1, color: [103, 0, 13] },
  ],
}

function getColorFromScale(t: number, scale: ColorScale): [number, number, number] {
  return interpolateStops(t, COLOR_SCALES[scale])
}

// For continuous metadata (not expression)
function interpolateColor(t: number): [number, number, number] {
  const r = Math.round(255 * t)
  const b = Math.round(255 * (1 - t))
  return [r, 50, b]
}

// Point-in-polygon using ray casting algorithm
function pointInPolygon(x: number, y: number, polygon: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1]
    const xj = polygon[j][0], yj = polygon[j][1]
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside
    }
  }
  return inside
}

export default function ScatterPlot({
  embedding,
  colorBy,
  expressionData,
  colorMode,
  interactionMode,
  selectedCellIndices,
  onSelectionComplete,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [lassoPoints, setLassoPoints] = useState<[number, number][]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [viewState, setViewState] = useState<OrthographicViewState | null>(null)

  // Get display preferences and cell masking state from store
  const displayPreferences = useStore((state) => state.displayPreferences)
  const activeCellMask = useStore((state) => state.activeCellMask)
  const showMaskedCells = useStore((state) => state.showMaskedCells)

  // Create a Set for fast lookup of selected indices
  const selectedSet = useMemo(() => new Set(selectedCellIndices), [selectedCellIndices])

  // Compute bounds from embedding only (so view doesn't reset on color/mask changes)
  const bounds = useMemo(() => {
    const coords = embedding.coordinates

    if (coords.length === 0) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 }
    }

    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity
    for (const [x, y] of coords) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }

    const padX = (maxX - minX) * 0.05
    const padY = (maxY - minY) * 0.05
    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    }
  }, [embedding])

  // Compute data, filtering out masked cells if showMaskedCells is false
  const data = useMemo(() => {
    const coords = embedding.coordinates
    const allData = coords.map((coord, index) => ({
      position: coord,
      index,
    }))

    // If no mask or showing masked cells, return all data
    if (!activeCellMask || showMaskedCells) {
      return allData
    }

    // Filter out masked (inactive) cells
    return allData.filter((d) => activeCellMask[d.index])
  }, [embedding, activeCellMask, showMaskedCells])

  // Compute color function separately (so it can change without affecting view state)
  const getColor = useMemo(() => {
    const opacity = Math.round(displayPreferences.pointOpacity * 255)
    const maskedOpacity = Math.round(opacity * 0.3) // 30% opacity for masked cells
    const maskedColor: [number, number, number, number] = [100, 100, 100, maskedOpacity]

    // Helper to check if cell is masked (inactive)
    const isMasked = (index: number): boolean => {
      return activeCellMask !== null && !activeCellMask[index]
    }

    if (colorMode === 'expression' && expressionData) {
      const { values, min, max } = expressionData
      const range = max - min || 1
      const colorScale = displayPreferences.colorScale

      return (d: { index: number }): [number, number, number, number] => {
        // Show masked cells as gray (when showMaskedCells is true, they're in the data)
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (selectedSet.size > 0 && selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const value = values[d.index]
        if (value === null || value === undefined) {
          return [128, 128, 128, Math.round(opacity * 0.5)]
        }
        const t = (value - min) / range
        const color = getColorFromScale(t, colorScale)
        return [...color, opacity] as [number, number, number, number]
      }
    } else if (colorMode === 'metadata' && colorBy) {
      if (colorBy.dtype === 'category') {
        return (d: { index: number }): [number, number, number, number] => {
          if (isMasked(d.index)) {
            return maskedColor
          }
          if (selectedSet.size > 0 && selectedSet.has(d.index)) {
            return SELECTED_COLOR
          }
          const value = colorBy.values[d.index] as number
          const color = CATEGORY_COLORS[value % CATEGORY_COLORS.length]
          return [...color, opacity] as [number, number, number, number]
        }
      } else if (colorBy.dtype === 'numeric') {
        const values = colorBy.values.filter((v) => v !== null) as number[]
        const min = Math.min(...values)
        const max = Math.max(...values)
        const range = max - min || 1

        return (d: { index: number }): [number, number, number, number] => {
          if (isMasked(d.index)) {
            return maskedColor
          }
          if (selectedSet.size > 0 && selectedSet.has(d.index)) {
            return SELECTED_COLOR
          }
          const value = colorBy.values[d.index]
          if (value === null) return [128, 128, 128, Math.round(opacity * 0.5)]
          const t = ((value as number) - min) / range
          const color = interpolateColor(t)
          return [...color, opacity] as [number, number, number, number]
        }
      }
    }

    // Default color function
    return (d: { index: number }): [number, number, number, number] => {
      if (isMasked(d.index)) {
        return maskedColor
      }
      if (selectedSet.size > 0 && selectedSet.has(d.index)) {
        return SELECTED_COLOR
      }
      return [DEFAULT_COLOR[0], DEFAULT_COLOR[1], DEFAULT_COLOR[2], opacity]
    }
  }, [colorBy, expressionData, colorMode, selectedSet, displayPreferences, activeCellMask])

  // Initialize view state when bounds change
  useEffect(() => {
    const width = bounds.maxX - bounds.minX
    const height = bounds.maxY - bounds.minY
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    const zoom = Math.log2(Math.min(800 / width, 600 / height)) - 1

    setViewState({
      target: [centerX, centerY, 0],
      zoom,
      minZoom: -10,
      maxZoom: 10,
    } as OrthographicViewState)
  }, [bounds])

  const view = useMemo(() => {
    return new OrthographicView({ id: 'main' })
  }, [])

  // Convert screen coordinates to data coordinates
  const screenToData = useCallback((screenX: number, screenY: number): [number, number] | null => {
    if (!containerRef.current || !viewState) return null

    const rect = containerRef.current.getBoundingClientRect()
    const x = screenX - rect.left
    const y = screenY - rect.top
    const width = rect.width
    const height = rect.height

    // Get view state
    const target = viewState.target as [number, number, number]
    const zoomValue = viewState.zoom
    const zoom = typeof zoomValue === 'number' ? zoomValue : (zoomValue?.[0] ?? 0)

    // Calculate scale
    const scale = Math.pow(2, zoom)

    // Convert screen to data coordinates
    // Note: deck.gl OrthographicView has Y increasing downward in screen space
    const dataX = (x - width / 2) / scale + target[0]
    const dataY = (y - height / 2) / scale + target[1]

    return [dataX, dataY]
  }, [viewState])

  // Handle lasso drawing
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (interactionMode !== 'lasso') return

    const point = screenToData(e.clientX, e.clientY)
    if (point) {
      setIsDrawing(true)
      setLassoPoints([point])
    }
  }, [interactionMode, screenToData])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDrawing || interactionMode !== 'lasso') return

    const point = screenToData(e.clientX, e.clientY)
    if (point) {
      setLassoPoints((prev) => [...prev, point])
    }
  }, [isDrawing, interactionMode, screenToData])

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || lassoPoints.length < 3) {
      setIsDrawing(false)
      setLassoPoints([])
      return
    }

    // Find all points inside the polygon
    const selectedIndices: number[] = []
    for (let i = 0; i < embedding.coordinates.length; i++) {
      const [x, y] = embedding.coordinates[i]
      if (pointInPolygon(x, y, lassoPoints)) {
        selectedIndices.push(i)
      }
    }

    onSelectionComplete(selectedIndices)
    setIsDrawing(false)
    setLassoPoints([])
  }, [isDrawing, lassoPoints, embedding.coordinates, onSelectionComplete])

  // Convert lasso points to SVG path
  const lassoPath = useMemo(() => {
    if (lassoPoints.length < 2 || !containerRef.current || !viewState) return ''

    const rect = containerRef.current.getBoundingClientRect()
    const width = rect.width
    const height = rect.height
    const target = viewState.target as [number, number, number]
    const zoomValue = viewState.zoom
    const zoom = typeof zoomValue === 'number' ? zoomValue : (zoomValue?.[0] ?? 0)
    const scale = Math.pow(2, zoom)

    const screenPoints = lassoPoints.map(([dataX, dataY]) => {
      const screenX = (dataX - target[0]) * scale + width / 2
      const screenY = (dataY - target[1]) * scale + height / 2
      return `${screenX},${screenY}`
    })

    return `M ${screenPoints.join(' L ')} Z`
  }, [lassoPoints, viewState])

  const baseRadius = displayPreferences.pointSize
  const selectedRadius = baseRadius + 2
  const maskedRadius = Math.max(1, baseRadius / 3) // 1/3 size for masked cells

  // Helper to get radius for a point
  const getRadius = (d: { index: number }): number => {
    // Masked cells are smaller
    if (activeCellMask !== null && !activeCellMask[d.index]) {
      return maskedRadius
    }
    // Selected cells are larger
    if (selectedSet.has(d.index)) {
      return selectedRadius
    }
    return baseRadius
  }

  const layers = [
    new ScatterplotLayer({
      id: 'scatterplot',
      data,
      getPosition: (d) => d.position,
      getRadius,
      getFillColor: getColor,
      radiusUnits: 'pixels',
      radiusMinPixels: 1,
      radiusMaxPixels: 20,
      pickable: true,
      updateTriggers: {
        getFillColor: [colorMode, colorBy?.name, expressionData?.gene, expressionData?.genes, selectedCellIndices, displayPreferences.colorScale, displayPreferences.pointOpacity, activeCellMask],
        getRadius: [selectedCellIndices, displayPreferences.pointSize, activeCellMask],
      },
    }),
  ]

  if (!viewState) return null

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        backgroundColor: displayPreferences.backgroundColor,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <DeckGL
        views={view}
        viewState={viewState}
        onViewStateChange={({ viewState: newViewState }) => {
          if (interactionMode === 'pan') {
            setViewState(newViewState as OrthographicViewState)
          }
        }}
        controller={interactionMode === 'pan'}
        layers={layers}
        style={{ position: 'absolute', width: '100%', height: '100%', background: 'transparent' }}
        getCursor={() => (interactionMode === 'lasso' ? 'crosshair' : 'grab')}
      />

      {/* Lasso SVG overlay */}
      {isDrawing && lassoPoints.length > 1 && (
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          <path
            d={lassoPath}
            fill="rgba(233, 69, 96, 0.2)"
            stroke="#e94560"
            strokeWidth={2}
            strokeDasharray="5,5"
          />
        </svg>
      )}

      {/* Lasso mode indicator */}
      {interactionMode === 'lasso' && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '6px 12px',
            backgroundColor: 'rgba(233, 69, 96, 0.9)',
            color: 'white',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          Lasso Mode: Click and drag to select cells
        </div>
      )}
    </div>
  )
}
