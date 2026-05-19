import { useEffect, useMemo, useState, useRef, useCallback } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView, OrthographicViewState } from '@deck.gl/core'
import { Figure, FigurePanel as PanelType, useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import { getColorFromScale, resolveCategoryPalette } from './ScatterPlot'

// Per-panel color data — fetched lazily based on the panel's color mode.
interface PanelColorData {
  kind: 'expression' | 'metadata-numeric' | 'metadata-category' | 'none'
  values?: (number | null)[]                    // length = current n_cells
  min?: number
  max?: number
  categories?: string[]
  categoryColors?: (string | null | undefined)[]
}

const API_BASE = '/api'

async function fetchPanelData(panel: PanelType): Promise<PanelColorData> {
  if (panel.colorMode === 'none') return { kind: 'none' }

  if (panel.colorMode === 'expression') {
    if (panel.selectedGenes.length === 0) return { kind: 'none' }
    const transform = panel.expressionTransform === 'log1p' ? 'log1p' : undefined
    if (panel.selectedGenes.length === 1) {
      const qs = new URLSearchParams()
      if (transform) qs.set('transform', transform)
      const url = appendDataset(`${API_BASE}/expression/${encodeURIComponent(panel.selectedGenes[0])}${qs.toString() ? '?' + qs.toString() : ''}`)
      const res = await fetch(url)
      if (!res.ok) throw new Error('expression fetch failed')
      const data = await res.json()
      return { kind: 'expression', values: data.values, min: data.min, max: data.max }
    } else {
      const res = await fetch(appendDataset(`${API_BASE}/expression/multi`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genes: panel.selectedGenes, transform }),
      })
      if (!res.ok) throw new Error('expression/multi fetch failed')
      const data = await res.json()
      return { kind: 'expression', values: data.values, min: data.min, max: data.max }
    }
  }

  if (panel.colorMode === 'metadata' && panel.selectedColorColumn) {
    const res = await fetch(appendDataset(`${API_BASE}/obs/${encodeURIComponent(panel.selectedColorColumn)}`))
    if (!res.ok) throw new Error('obs fetch failed')
    const data = await res.json()
    if (data.dtype === 'category') {
      return { kind: 'metadata-category', values: data.values, categories: data.categories ?? [], categoryColors: data.colors ?? undefined }
    }
    if (data.dtype === 'numeric') {
      const vals = (data.values as (number | null)[]).filter((v): v is number => v != null)
      const min = Math.min(...vals)
      const max = Math.max(...vals)
      return { kind: 'metadata-numeric', values: data.values, min, max }
    }
  }

  return { kind: 'none' }
}

interface Props {
  figure: Figure
  panel: PanelType
  // Render-only mode used by the export path — disables interaction so the
  // panel renders without controllers, suitable for screenshot capture.
  staticView?: boolean
}

export default function FigurePanel({ figure, panel, staticView = false }: Props) {
  const updateFigure = useStore((s) => s.updateFigure)
  const [colorData, setColorData] = useState<PanelColorData>({ kind: 'none' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Fetch color data when the panel's color mode / source changes.
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchPanelData(panel)
      .then((d) => {
        if (!cancelled) setColorData(d)
      })
      .catch((err) => {
        if (!cancelled) setError((err as Error).message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [panel.colorMode, panel.selectedGenes, panel.selectedColorColumn, panel.expressionTransform])

  // Bounds from the snapshotted coordinates — used for default viewState.
  const bounds = useMemo(() => {
    const coords = figure.coordinates
    if (coords.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const [x, y] of coords) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const padX = (maxX - minX) * 0.05
    const padY = (maxY - minY) * 0.05
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }
  }, [figure.coordinates])

  // Build deck.gl data — each entry is a renderable cell with position + the
  // ORIGINAL adata cell index (for color lookup) preserved.
  const data = useMemo(() => {
    return figure.coordinates.map((coord, i) => ({
      position: coord,
      adataIndex: figure.cellIndices[i],
    }))
  }, [figure.coordinates, figure.cellIndices])

  // Color function — closed over the panel's settings + fetched data.
  const getColor = useMemo(() => {
    const opacity = Math.round(panel.pointOpacity * 255)
    const fallback: [number, number, number, number] = [100, 149, 237, opacity]

    if (colorData.kind === 'expression' && colorData.values && colorData.min !== undefined && colorData.max !== undefined) {
      const { values, min, max } = colorData
      const range = max - min || 1
      const scale = panel.colorScale
      return (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex]
        if (v == null) return [128, 128, 128, Math.round(opacity * 0.5)]
        const t = (v - min) / range
        const rgb = getColorFromScale(t, scale)
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    }
    if (colorData.kind === 'metadata-numeric' && colorData.values && colorData.min !== undefined && colorData.max !== undefined) {
      const { values, min, max } = colorData
      const range = max - min || 1
      const scale = panel.colorScale
      return (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex]
        if (v == null) return [128, 128, 128, Math.round(opacity * 0.5)]
        const t = ((v as number) - min) / range
        const rgb = getColorFromScale(t, scale)
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    }
    if (colorData.kind === 'metadata-category' && colorData.values && colorData.categories) {
      const palette = resolveCategoryPalette(colorData.categories.length, colorData.categoryColors)
      const values = colorData.values
      return (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex] as number
        const rgb = palette[v] ?? palette[0] ?? [128, 128, 128]
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    }
    return (_d: { adataIndex: number }) => fallback
  }, [colorData, panel.colorScale, panel.pointOpacity])

  // Shared viewState: read from figure, write back when user pans/zooms.
  const viewState: OrthographicViewState = useMemo(() => {
    if (figure.sharedZoom !== null && figure.sharedTargetX !== null && figure.sharedTargetY !== null) {
      return {
        target: [figure.sharedTargetX, figure.sharedTargetY, 0],
        zoom: figure.sharedZoom,
        minZoom: -10,
        maxZoom: 10,
      } as OrthographicViewState
    }
    // Lazy default: fit to bounds. Compute a reasonable zoom for the
    // typical figure-panel size (~400×400 logical px).
    const width = bounds.maxX - bounds.minX
    const height = bounds.maxY - bounds.minY
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    const zoom = Math.log2(Math.min(400 / width, 400 / height)) - 1
    return { target: [centerX, centerY, 0], zoom, minZoom: -10, maxZoom: 10 } as OrthographicViewState
  }, [figure.sharedZoom, figure.sharedTargetX, figure.sharedTargetY, bounds])

  // Initialize shared viewState on first render if it hasn't been set.
  useEffect(() => {
    if (figure.sharedZoom === null) {
      const v = viewState
      updateFigure({
        sharedZoom: typeof v.zoom === 'number' ? v.zoom : (v.zoom as number[])[0],
        sharedTargetX: (v.target as number[])[0],
        sharedTargetY: (v.target as number[])[1],
      })
    }
    // Only run when shared state is null on mount — subsequent changes are
    // user-driven through handleViewStateChange.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleViewStateChange = useCallback(
    (params: { viewState: OrthographicViewState }) => {
      const v = params.viewState
      const target = v.target as number[]
      const zoom = typeof v.zoom === 'number' ? v.zoom : (v.zoom as number[])[0]
      updateFigure({
        sharedZoom: zoom,
        sharedTargetX: target[0],
        sharedTargetY: target[1],
      })
    },
    [updateFigure]
  )

  const radius = panel.pointSize / 2

  const deckLayers = [
    new ScatterplotLayer({
      id: `panel-${panel.id}`,
      data,
      getPosition: (d) => d.position,
      getRadius: radius,
      getFillColor: getColor,
      radiusUnits: 'pixels',
      radiusMinPixels: 1,
      stroked: false,
      filled: true,
      updateTriggers: {
        getFillColor: [colorData, panel.colorScale, panel.pointOpacity],
        getRadius: panel.pointSize,
      },
    }),
  ]

  const view = useMemo(() => new OrthographicView({ id: `view-${panel.id}` }), [panel.id])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        backgroundColor: panel.background,
        border: panel.showBorder ? '1px solid #444' : 'none',
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      <DeckGL
        views={view}
        viewState={viewState}
        controller={!staticView}
        layers={deckLayers}
        useDevicePixels={true}
        onViewStateChange={staticView ? undefined : handleViewStateChange}
        style={{ width: '100%', height: '100%' }}
      />
      {panel.title && (
        <div
          style={{
            position: 'absolute',
            top: '8px',
            left: '8px',
            fontSize: '14px',
            fontWeight: 600,
            color: '#fff',
            textShadow: '0 1px 2px rgba(0,0,0,0.8)',
            pointerEvents: 'none',
          }}
        >
          {panel.title}
        </div>
      )}
      {/* Expression legend (mini colorbar) — only for continuous color modes */}
      {(colorData.kind === 'expression' || colorData.kind === 'metadata-numeric') &&
        colorData.min !== undefined && colorData.max !== undefined && (
        <div
          style={{
            position: 'absolute',
            right: '8px',
            bottom: '8px',
            pointerEvents: 'none',
            color: '#fff',
            textShadow: '0 1px 2px rgba(0,0,0,0.8)',
            fontSize: '10px',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span>{colorData.min.toFixed(2)}</span>
            <ColorBar scale={panel.colorScale} width={80} height={8} />
            <span>{colorData.max.toFixed(2)}</span>
          </div>
        </div>
      )}
      {loading && (
        <div style={loadingOverlayStyle}>Loading…</div>
      )}
      {error && (
        <div style={{ ...loadingOverlayStyle, color: '#e94560' }}>{error}</div>
      )}
    </div>
  )
}

function ColorBar({ scale, width, height }: { scale: import('../store').ColorScale; width: number; height: number }) {
  // Render a horizontal gradient stripe sampled from the color scale.
  const stops: string[] = []
  const N = 16
  for (let i = 0; i <= N; i++) {
    const t = i / N
    const rgb = getColorFromScale(t, scale)
    stops.push(`rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}) ${(t * 100).toFixed(0)}%`)
  }
  const bg = `linear-gradient(to right, ${stops.join(', ')})`
  return <div style={{ width, height, background: bg, borderRadius: '2px', border: '1px solid rgba(255,255,255,0.4)' }} />
}

const loadingOverlayStyle = {
  position: 'absolute' as const,
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  padding: '6px 12px',
  backgroundColor: 'rgba(0,0,0,0.6)',
  color: '#fff',
  fontSize: '11px',
  borderRadius: '4px',
}
