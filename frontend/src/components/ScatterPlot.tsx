import { useMemo, useState, useRef, useCallback, useEffect } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView, OrthographicViewState } from '@deck.gl/core'
import { useStore, EmbeddingData, ObsColumnData, ExpressionData, BivariateExpressionData, ColorMode, InteractionMode, ColorScale } from '../store'

interface ScatterPlotProps {
  embedding: EmbeddingData
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  bivariateData: BivariateExpressionData | null
  colorMode: ColorMode
  interactionMode: InteractionMode
  selectedCellIndices: number[]
  onSelectionComplete: (indices: number[], additive: boolean) => void
  onLineDrawn: (points: [number, number][]) => void
  onTransformEmbedding: (rotationDegrees: number) => void
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

// Bivariate colormap definitions
// Each has corner colors: c00 (low/low), c10 (high gene1/low gene2), c01 (low gene1/high gene2), c11 (high/high)
import { BivariateColormap } from '../store'

type BivariateCorners = {
  c00: [number, number, number]  // Low/Low
  c10: [number, number, number]  // High gene1/Low gene2
  c01: [number, number, number]  // Low gene1/High gene2
  c11: [number, number, number]  // High/High
}

export const BIVARIATE_COLORMAPS: Record<BivariateColormap, BivariateCorners> = {
  default: {
    c00: [240, 240, 240],  // Gray
    c10: [227, 26, 28],    // Red
    c01: [31, 120, 180],   // Blue
    c11: [255, 255, 0],    // Yellow
  },
  pinkgreen: {
    c00: [240, 240, 240],  // Gray
    c10: [197, 27, 125],   // Pink/Magenta
    c01: [77, 146, 33],    // Green
    c11: [166, 86, 40],    // Brown (mixing pink+green)
  },
  orangepurple: {
    c00: [240, 240, 240],  // Gray
    c10: [230, 97, 1],     // Orange
    c01: [94, 60, 153],    // Purple
    c11: [178, 24, 43],    // Dark red/maroon
  },
  custom: {
    c00: [240, 240, 240],  // Gray (default, can be customized later)
    c10: [227, 26, 28],    // Red
    c01: [31, 120, 180],   // Blue
    c11: [255, 255, 0],    // Yellow
  },
}

// Bilinear interpolation for bivariate coloring
export function getBivariateColor(
  u: number,  // Normalized gene set 1 value [0,1]
  v: number,  // Normalized gene set 2 value [0,1]
  colormap: BivariateColormap = 'default',
): [number, number, number] {
  const { c00, c10, c01, c11 } = BIVARIATE_COLORMAPS[colormap]
  // Bilinear interpolation: blend four corner colors based on (u, v) position
  return [
    Math.round(c00[0] * (1 - u) * (1 - v) + c10[0] * u * (1 - v) + c01[0] * (1 - u) * v + c11[0] * u * v),
    Math.round(c00[1] * (1 - u) * (1 - v) + c10[1] * u * (1 - v) + c01[1] * (1 - u) * v + c11[1] * u * v),
    Math.round(c00[2] * (1 - u) * (1 - v) + c10[2] * u * (1 - v) + c01[2] * (1 - u) * v + c11[2] * u * v),
  ]
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
  bivariateData,
  colorMode,
  interactionMode,
  selectedCellIndices,
  onSelectionComplete,
  onLineDrawn,
  onTransformEmbedding,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [lassoPoints, setLassoPoints] = useState<[number, number][]>([])
  const [linePoints, setLinePoints] = useState<[number, number][]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [viewState, setViewState] = useState<OrthographicViewState | null>(null)

  // Rotation state for adjust mode
  const isRotating = useRef(false)
  const rotateStartAngle = useRef(0)
  const accumulatedRotation = useRef(0)
  const preRotationCoords = useRef<[number, number][] | null>(null)

  // Get display preferences, cell masking, sort state, and drawn lines from store
  const displayPreferences = useStore((state) => state.displayPreferences)
  const activeCellMask = useStore((state) => state.activeCellMask)
  const showMaskedCells = useStore((state) => state.showMaskedCells)
  const cellSortOrder = useStore((state) => state.cellSortOrder)
  const cellSortVersion = useStore((state) => state.cellSortVersion)
  const drawnLines = useStore((state) => state.drawnLines)
  const activeLineId = useStore((state) => state.activeLineId)

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

  // Compute data, filtering out masked cells if showMaskedCells is false, and applying sort order
  const data = useMemo(() => {
    const coords = embedding.coordinates
    let allData = coords.map((coord, index) => ({
      position: coord,
      index,
    }))

    // Apply sort order if set (reorder so high-expression cells render on top)
    if (cellSortOrder) {
      allData = cellSortOrder.map((originalIndex) => ({
        position: coords[originalIndex],
        index: originalIndex,
      }))
    }

    // If no mask or showing masked cells, return all data
    if (!activeCellMask || showMaskedCells) {
      return allData
    }

    // Filter out masked (inactive) cells
    return allData.filter((d) => activeCellMask[d.index])
  }, [embedding, activeCellMask, showMaskedCells, cellSortOrder, cellSortVersion])

  // Compute color function separately (so it can change without affecting view state)
  const getColor = useMemo(() => {
    const opacity = Math.round(displayPreferences.pointOpacity * 255)
    const maskedOpacity = Math.round(opacity * 0.3) // 30% opacity for masked cells
    const maskedColor: [number, number, number, number] = [100, 100, 100, maskedOpacity]

    // Helper to check if cell is masked (inactive)
    const isMasked = (index: number): boolean => {
      return activeCellMask !== null && !activeCellMask[index]
    }

    if (colorMode === 'bivariate' && bivariateData) {
      const { values1, values2 } = bivariateData
      const bivariateColormap = displayPreferences.bivariateColormap

      return (d: { index: number }): [number, number, number, number] => {
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (selectedSet.size > 0 && selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const u = values1[d.index] ?? 0
        const v = values2[d.index] ?? 0
        const color = getBivariateColor(u, v, bivariateColormap)
        return [...color, opacity] as [number, number, number, number]
      }
    } else if (colorMode === 'expression' && expressionData) {
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
  }, [colorBy, expressionData, bivariateData, colorMode, selectedSet, displayPreferences, activeCellMask])

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
    const dataX = (x - width / 2) / scale + target[0]
    const dataY = (y - height / 2) / scale + target[1]

    return [dataX, dataY]
  }, [viewState])

  // Handle lasso, line drawing, and adjust rotation
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (interactionMode === 'lasso') {
      const point = screenToData(e.clientX, e.clientY)
      if (point) {
        setIsDrawing(true)
        setLassoPoints([point])
      }
    } else if (interactionMode === 'draw') {
      const point = screenToData(e.clientX, e.clientY)
      if (point) {
        setIsDrawing(true)
        setLinePoints([point])
      }
    } else if (interactionMode === 'adjust' && e.shiftKey && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const centerY = rect.top + rect.height / 2
      const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX)
      isRotating.current = true
      rotateStartAngle.current = angle
      accumulatedRotation.current = 0
      preRotationCoords.current = embedding.coordinates.map(c => [c[0], c[1]] as [number, number])
    }
  }, [interactionMode, screenToData, embedding.coordinates])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    // Handle adjust rotation independently of isDrawing
    if (isRotating.current && interactionMode === 'adjust' && containerRef.current && preRotationCoords.current) {
      const rect = containerRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const centerY = rect.top + rect.height / 2
      const currentAngle = Math.atan2(e.clientY - centerY, e.clientX - centerX)
      const totalDelta = currentAngle - rotateStartAngle.current
      const totalDegrees = -(totalDelta * 180) / Math.PI
      accumulatedRotation.current = totalDegrees

      // Apply rotation client-side from the pre-rotation snapshot
      const theta = (totalDegrees * Math.PI) / 180
      const cosT = Math.cos(theta)
      const sinT = Math.sin(theta)
      const coords = preRotationCoords.current
      // Compute centroid
      let cx = 0, cy = 0
      for (const [x, y] of coords) { cx += x; cy += y }
      cx /= coords.length; cy /= coords.length
      const rotated: [number, number][] = coords.map(([x, y]) => {
        const dx = x - cx
        const dy = y - cy
        return [cx + dx * cosT - dy * sinT, cy + dx * sinT + dy * cosT]
      })
      const setEmbedding = useStore.getState().setEmbedding
      setEmbedding({ name: embedding.name, coordinates: rotated })
      return
    }

    if (!isDrawing) return

    const point = screenToData(e.clientX, e.clientY)
    if (!point) return

    if (interactionMode === 'lasso') {
      setLassoPoints((prev) => [...prev, point])
    } else if (interactionMode === 'draw') {
      setLinePoints((prev) => [...prev, point])
    }
  }, [isDrawing, interactionMode, screenToData, embedding.name])

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    // Handle adjust rotation release
    if (isRotating.current) {
      isRotating.current = false
      const totalDeg = accumulatedRotation.current
      preRotationCoords.current = null
      if (Math.abs(totalDeg) > 0.1) {
        onTransformEmbedding(totalDeg)
      }
      return
    }

    if (!isDrawing) return

    if (interactionMode === 'lasso') {
      if (lassoPoints.length < 3) {
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

      // Hold Shift to add to existing selection instead of replacing
      onSelectionComplete(selectedIndices, e.shiftKey)
      setIsDrawing(false)
      setLassoPoints([])
    } else if (interactionMode === 'draw') {
      if (linePoints.length >= 2) {
        onLineDrawn(linePoints)
      }
      setIsDrawing(false)
      setLinePoints([])
    }
  }, [isDrawing, interactionMode, lassoPoints, linePoints, embedding.coordinates, onSelectionComplete, onLineDrawn, onTransformEmbedding])

  // Convert data points to screen coordinates
  const dataToScreen = useCallback((points: [number, number][]): string[] => {
    if (!containerRef.current || !viewState) return []

    const rect = containerRef.current.getBoundingClientRect()
    const width = rect.width
    const height = rect.height
    const target = viewState.target as [number, number, number]
    const zoomValue = viewState.zoom
    const zoom = typeof zoomValue === 'number' ? zoomValue : (zoomValue?.[0] ?? 0)
    const scale = Math.pow(2, zoom)

    return points.map(([dataX, dataY]) => {
      const screenX = (dataX - target[0]) * scale + width / 2
      const screenY = (dataY - target[1]) * scale + height / 2
      return `${screenX},${screenY}`
    })
  }, [viewState])

  // Convert lasso points to SVG path
  const lassoPath = useMemo(() => {
    if (lassoPoints.length < 2) return ''
    const screenPoints = dataToScreen(lassoPoints)
    if (screenPoints.length === 0) return ''
    return `M ${screenPoints.join(' L ')} Z`
  }, [lassoPoints, dataToScreen])

  // Convert line points to SVG path (open, not closed)
  const linePath = useMemo(() => {
    if (linePoints.length < 2) return ''
    const screenPoints = dataToScreen(linePoints)
    if (screenPoints.length === 0) return ''
    return `M ${screenPoints.join(' L ')}`
  }, [linePoints, dataToScreen])

  // Filter and convert stored lines to SVG paths (only for current embedding and visible)
  const storedLinePaths = useMemo(() => {
    return drawnLines
      .filter((line) => line.embeddingName === embedding.name && line.visible)
      .map((line) => {
        const points = line.smoothedPoints || line.points
        if (points.length < 2) return { id: line.id, name: line.name, path: '', isActive: line.id === activeLineId }
        const screenPoints = dataToScreen(points)
        if (screenPoints.length === 0) return { id: line.id, name: line.name, path: '', isActive: line.id === activeLineId }
        return {
          id: line.id,
          name: line.name,
          path: `M ${screenPoints.join(' L ')}`,
          isActive: line.id === activeLineId,
          startPoint: screenPoints[0],
          endPoint: screenPoints[screenPoints.length - 1],
        }
      })
  }, [drawnLines, activeLineId, dataToScreen, embedding.name])

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
        getFillColor: [colorMode, colorBy?.name, expressionData?.gene, expressionData?.genes, selectedCellIndices, displayPreferences.colorScale, displayPreferences.bivariateColormap, displayPreferences.pointOpacity, activeCellMask],
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
        getCursor={() => (interactionMode === 'lasso' || interactionMode === 'draw' ? 'crosshair' : interactionMode === 'adjust' ? 'move' : 'grab')}
      />

      {/* SVG overlay for lasso, lines, and stored lines */}
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
        {/* Stored lines */}
        {storedLinePaths.map((line) => line.path && (
          <g key={line.id}>
            <path
              d={line.path}
              fill="none"
              stroke={line.isActive ? '#4ecdc4' : '#888'}
              strokeWidth={line.isActive ? 3 : 2}
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity={line.isActive ? 1 : 0.5}
            />
            {/* Direction arrow at end */}
            {line.endPoint && line.isActive && (
              <circle
                cx={line.endPoint.split(',')[0]}
                cy={line.endPoint.split(',')[1]}
                r={6}
                fill="#4ecdc4"
              />
            )}
            {/* Start indicator */}
            {line.startPoint && line.isActive && (
              <circle
                cx={line.startPoint.split(',')[0]}
                cy={line.startPoint.split(',')[1]}
                r={4}
                fill="none"
                stroke="#4ecdc4"
                strokeWidth={2}
              />
            )}
          </g>
        ))}

        {/* Currently drawing lasso */}
        {isDrawing && interactionMode === 'lasso' && lassoPoints.length > 1 && (
          <path
            d={lassoPath}
            fill="rgba(233, 69, 96, 0.2)"
            stroke="#e94560"
            strokeWidth={2}
            strokeDasharray="5,5"
          />
        )}

        {/* Currently drawing line */}
        {isDrawing && interactionMode === 'draw' && linePoints.length > 1 && (
          <path
            d={linePath}
            fill="none"
            stroke="#4ecdc4"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>

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
          Lasso Mode: Click and drag to select cells (hold Shift to add to selection)
        </div>
      )}

      {/* Draw mode indicator */}
      {interactionMode === 'draw' && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '6px 12px',
            backgroundColor: 'rgba(78, 205, 196, 0.9)',
            color: 'white',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          Draw Mode: Click and drag to draw a line
        </div>
      )}

      {/* Adjust mode indicator */}
      {interactionMode === 'adjust' && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '6px 12px',
            backgroundColor: 'rgba(255, 165, 0, 0.9)',
            color: 'white',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          Adjust Mode: Shift+drag to rotate
        </div>
      )}
    </div>
  )
}
