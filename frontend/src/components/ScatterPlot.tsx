import { useMemo, useState, useRef, useCallback, useEffect } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView, OrthographicViewState } from '@deck.gl/core'
import { EmbeddingData, ObsColumnData, ExpressionData, ColorMode, InteractionMode } from '../store'

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

function viridisColor(t: number): [number, number, number] {
  if (t < 0.25) {
    const s = t / 0.25
    return [
      Math.round(68 + s * (59 - 68)),
      Math.round(1 + s * (82 - 1)),
      Math.round(84 + s * (139 - 84)),
    ]
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25
    return [
      Math.round(59 + s * (33 - 59)),
      Math.round(82 + s * (145 - 82)),
      Math.round(139 + s * (140 - 139)),
    ]
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25
    return [
      Math.round(33 + s * (94 - 33)),
      Math.round(145 + s * (201 - 145)),
      Math.round(140 + s * (98 - 140)),
    ]
  } else {
    const s = (t - 0.75) / 0.25
    return [
      Math.round(94 + s * (253 - 94)),
      Math.round(201 + s * (231 - 201)),
      Math.round(98 + s * (37 - 98)),
    ]
  }
}

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

  // Create a Set for fast lookup of selected indices
  const selectedSet = useMemo(() => new Set(selectedCellIndices), [selectedCellIndices])

  // Compute bounds and colors
  const { bounds, data, getColor } = useMemo(() => {
    const coords = embedding.coordinates
    if (coords.length === 0) {
      return {
        bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
        data: [],
        getColor: () => DEFAULT_COLOR,
      }
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
    const bounds = {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    }

    const data = coords.map((coord, index) => ({
      position: coord,
      index,
    }))

    let getColor: (d: { index: number }) => [number, number, number, number]

    if (colorMode === 'expression' && expressionData) {
      const { values, min, max } = expressionData
      const range = max - min || 1

      getColor = (d) => {
        if (selectedSet.size > 0 && selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const value = values[d.index]
        if (value === null || value === undefined) {
          return [128, 128, 128, 100]
        }
        const t = (value - min) / range
        const color = viridisColor(t)
        return [...color, 220] as [number, number, number, number]
      }
    } else if (colorMode === 'metadata' && colorBy) {
      if (colorBy.dtype === 'category') {
        getColor = (d) => {
          if (selectedSet.size > 0 && selectedSet.has(d.index)) {
            return SELECTED_COLOR
          }
          const value = colorBy.values[d.index] as number
          const color = CATEGORY_COLORS[value % CATEGORY_COLORS.length]
          return [...color, 200] as [number, number, number, number]
        }
      } else if (colorBy.dtype === 'numeric') {
        const values = colorBy.values.filter((v) => v !== null) as number[]
        const min = Math.min(...values)
        const max = Math.max(...values)
        const range = max - min || 1

        getColor = (d) => {
          if (selectedSet.size > 0 && selectedSet.has(d.index)) {
            return SELECTED_COLOR
          }
          const value = colorBy.values[d.index]
          if (value === null) return [128, 128, 128, 100]
          const t = ((value as number) - min) / range
          const color = interpolateColor(t)
          return [...color, 200] as [number, number, number, number]
        }
      } else {
        getColor = (d) => {
          if (selectedSet.size > 0 && selectedSet.has(d.index)) {
            return SELECTED_COLOR
          }
          return DEFAULT_COLOR
        }
      }
    } else {
      getColor = (d) => {
        if (selectedSet.size > 0 && selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        return DEFAULT_COLOR
      }
    }

    return { bounds, data, getColor }
  }, [embedding, colorBy, expressionData, colorMode, selectedSet])

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
    const zoom = viewState.zoom || 0

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
    const zoom = viewState.zoom || 0
    const scale = Math.pow(2, zoom)

    const screenPoints = lassoPoints.map(([dataX, dataY]) => {
      const screenX = (dataX - target[0]) * scale + width / 2
      const screenY = (dataY - target[1]) * scale + height / 2
      return `${screenX},${screenY}`
    })

    return `M ${screenPoints.join(' L ')} Z`
  }, [lassoPoints, viewState])

  const layers = [
    new ScatterplotLayer({
      id: 'scatterplot',
      data,
      getPosition: (d) => d.position,
      getRadius: (d) => (selectedSet.has(d.index) ? 5 : 3),
      getFillColor: getColor,
      radiusMinPixels: 2,
      radiusMaxPixels: 12,
      pickable: true,
      updateTriggers: {
        getFillColor: [colorMode, colorBy?.name, expressionData?.gene, expressionData?.genes, selectedCellIndices],
        getRadius: [selectedCellIndices],
      },
    }),
  ]

  if (!viewState) return null

  return (
    <div
      ref={containerRef}
      style={{ position: 'relative', width: '100%', height: '100%' }}
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
        style={{ position: 'absolute', width: '100%', height: '100%' }}
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
