import { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { EmbeddingData, ObsColumnData } from '../store'

interface ScatterPlotProps {
  embedding: EmbeddingData
  colorBy: ObsColumnData | null
}

// Color palette for categorical data (similar to d3 category10)
const CATEGORY_COLORS: [number, number, number][] = [
  [31, 119, 180],   // blue
  [255, 127, 14],   // orange
  [44, 160, 44],    // green
  [214, 39, 40],    // red
  [148, 103, 189],  // purple
  [140, 86, 75],    // brown
  [227, 119, 194],  // pink
  [127, 127, 127],  // gray
  [188, 189, 34],   // olive
  [23, 190, 207],   // cyan
]

// Default color when no color column selected
const DEFAULT_COLOR: [number, number, number, number] = [100, 149, 237, 200] // cornflower blue

function interpolateColor(t: number): [number, number, number] {
  // Blue to red gradient for numeric data
  const r = Math.round(255 * t)
  const b = Math.round(255 * (1 - t))
  return [r, 50, b]
}

export default function ScatterPlot({ embedding, colorBy }: ScatterPlotProps) {
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

    // Calculate bounds
    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity
    for (const [x, y] of coords) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }

    // Add padding
    const padX = (maxX - minX) * 0.05
    const padY = (maxY - minY) * 0.05
    const bounds = {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    }

    // Create data array with indices
    const data = coords.map((coord, index) => ({
      position: coord,
      index,
    }))

    // Create color function based on colorBy
    let getColor: (d: { index: number }) => [number, number, number, number]

    if (!colorBy) {
      getColor = () => DEFAULT_COLOR
    } else if (colorBy.dtype === 'category') {
      getColor = (d) => {
        const value = colorBy.values[d.index] as number
        const color = CATEGORY_COLORS[value % CATEGORY_COLORS.length]
        return [...color, 200] as [number, number, number, number]
      }
    } else if (colorBy.dtype === 'numeric') {
      // Normalize numeric values for color mapping
      const values = colorBy.values.filter((v) => v !== null) as number[]
      const min = Math.min(...values)
      const max = Math.max(...values)
      const range = max - min || 1

      getColor = (d) => {
        const value = colorBy.values[d.index]
        if (value === null) return [128, 128, 128, 100] // Gray for null
        const t = ((value as number) - min) / range
        const color = interpolateColor(t)
        return [...color, 200] as [number, number, number, number]
      }
    } else {
      getColor = () => DEFAULT_COLOR
    }

    return { bounds, data, getColor }
  }, [embedding, colorBy])

  // Calculate view to fit all points
  const view = useMemo(() => {
    const width = bounds.maxX - bounds.minX
    const height = bounds.maxY - bounds.minY
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    const zoom = Math.log2(Math.min(800 / width, 600 / height))

    return new OrthographicView({
      id: 'main',
    })
  }, [bounds])

  const initialViewState = useMemo(() => {
    const width = bounds.maxX - bounds.minX
    const height = bounds.maxY - bounds.minY
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    const zoom = Math.log2(Math.min(800 / width, 600 / height)) - 1

    return {
      target: [centerX, centerY, 0],
      zoom,
    }
  }, [bounds])

  const layers = [
    new ScatterplotLayer({
      id: 'scatterplot',
      data,
      getPosition: (d) => d.position,
      getRadius: 3,
      getFillColor: getColor,
      radiusMinPixels: 2,
      radiusMaxPixels: 10,
      pickable: true,
      updateTriggers: {
        getFillColor: [colorBy?.name, colorBy?.values],
      },
    }),
  ]

  return (
    <DeckGL
      views={view}
      initialViewState={initialViewState}
      controller={true}
      layers={layers}
      style={{ position: 'relative', width: '100%', height: '100%' }}
    />
  )
}
