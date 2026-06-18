import { useEffect, useMemo, useState, useRef, useCallback } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView, OrthographicViewState } from '@deck.gl/core'
import { Figure, FigurePanel as PanelType, HighlightLayer, HighlightSource, useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import { getColorFromScale, getBivariateColor, resolveCategoryPalette, hexToRgb } from './ScatterPlot'

// Build a per-cell weight function for a HighlightLayer. Mirrors the
// `layerWeightFn` in ScatterPlot.tsx (kept inline here to avoid leaking
// internals; small enough that duplication is fine).
function layerWeightFn(layer: HighlightLayer): (i: number) => number {
  const src: HighlightSource = layer.source
  const intensity = layer.intensity
  if (src.kind === 'cellset') {
    const mask = src.mask
    return (i: number) => (mask[i] ? intensity : 0)
  }
  const { values, thresholdMode, lo, hi } = src
  if (thresholdMode === 'above') {
    return (i: number) => {
      const v = values[i]
      return v != null && v >= lo ? intensity : 0
    }
  }
  if (thresholdMode === 'below') {
    return (i: number) => {
      const v = values[i]
      return v != null && v <= lo ? intensity : 0
    }
  }
  return (i: number) => {
    const v = values[i]
    return v != null && v >= lo && v <= hi ? intensity : 0
  }
}

// Per-panel color data — fetched lazily based on the panel's color mode.
interface PanelColorData {
  kind: 'expression' | 'metadata-numeric' | 'metadata-category' | 'bivariate' | 'none'
  values?: (number | null)[]                    // length = current n_cells
  values1?: number[]                            // bivariate: gene-set 1 score (normalized [0,1])
  values2?: number[]                            // bivariate: gene-set 2 score (normalized [0,1])
  min?: number
  max?: number
  categories?: string[]
  categoryColors?: (string | null | undefined)[]
}

const API_BASE = '/api'

// Resolve a gene-set name to its gene list via the global store (gene sets
// are global, not per-dataset). Returns [] if unknown.
function lookupGeneSet(name: string | null): string[] {
  if (!name) return []
  const cats = useStore.getState().geneSetCategories
  for (const k of Object.keys(cats) as (keyof typeof cats)[]) {
    const cat = cats[k]
    for (const gs of cat.geneSets) {
      if (gs.name === name) return gs.genes
    }
    for (const folder of cat.folders) {
      for (const gs of folder.geneSets) {
        if (gs.name === name) return gs.genes
      }
    }
  }
  return []
}

async function fetchPanelData(panel: PanelType): Promise<PanelColorData> {
  if (panel.colorMode === 'none') return { kind: 'none' }

  if (panel.colorMode === 'bivariate') {
    // Each axis is a single gene (bivariateGeneN) or a gene set (bivariateSetN).
    const genes1 = panel.bivariateGene1 ? [panel.bivariateGene1] : lookupGeneSet(panel.bivariateSet1)
    const genes2 = panel.bivariateGene2 ? [panel.bivariateGene2] : lookupGeneSet(panel.bivariateSet2)
    if (genes1.length === 0 || genes2.length === 0) return { kind: 'none' }
    const transform = panel.expressionTransform === 'log1p' ? 'log1p' : null
    const res = await fetch(appendDataset(`${API_BASE}/expression/bivariate`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genes1, genes2, transform }),
    })
    if (!res.ok) throw new Error('bivariate fetch failed')
    const data = await res.json()
    return { kind: 'bivariate', values1: data.values1, values2: data.values2 }
  }

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
  const highlightLayers = useStore((s) => s.highlightLayers)
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
  }, [panel.colorMode, panel.selectedGenes, panel.selectedColorColumn, panel.expressionTransform, panel.bivariateSet1, panel.bivariateSet2, panel.bivariateGene1, panel.bivariateGene2])

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
  // ORIGINAL adata cell index (for color lookup) preserved. When the panel has
  // a continuous color source (expression or numeric metadata), stable-sort
  // ascending by the cell's score so high-value cells render LAST (= on top
  // in deck.gl draw order), matching the main ScatterPlot's behavior. This
  // keeps high-expression markers from being occluded by overlapping
  // low-expression cells.
  const data = useMemo(() => {
    let arr = figure.coordinates.map((coord, i) => ({
      position: coord,
      adataIndex: figure.cellIndices[i],
    }))
    if ((colorData.kind === 'expression' || colorData.kind === 'metadata-numeric') && colorData.values) {
      const values = colorData.values
      arr = [...arr].sort((a, b) => {
        const va = values[a.adataIndex] ?? -Infinity
        const vb = values[b.adataIndex] ?? -Infinity
        return (va as number) - (vb as number)
      })
    } else if (colorData.kind === 'bivariate' && colorData.values1 && colorData.values2) {
      // Sort by sum of normalized scores — matches the main ScatterPlot's
      // bivariate stacking so high-expression cells stay on top.
      const v1 = colorData.values1
      const v2 = colorData.values2
      arr = [...arr].sort((a, b) => {
        const sa = (v1[a.adataIndex] ?? 0) + (v2[a.adataIndex] ?? 0)
        const sb = (v1[b.adataIndex] ?? 0) + (v2[b.adataIndex] ?? 0)
        return sa - sb
      })
    }

    // Highlight overlay: after any expression/bivariate stacking, stable-sort
    // ascending by max highlight weight so highlighted cells end up LAST (= on
    // top in deck.gl draw order). Without this, an overlapping non-highlighted
    // low/zero-expression cell can be drawn over a highlighted cell and obscure
    // its color. Mirrors the main ScatterPlot's highlight stacking. The sort is
    // stable, so high-expression-on-top is preserved among equal-weight cells.
    if (panel.showHighlightOverlay && highlightLayers.length > 0) {
      const weighters = highlightLayers.map(layerWeightFn)
      const weightAt = (i: number): number => {
        let m = 0
        for (let k = 0; k < weighters.length; k++) {
          const w = weighters[k](i)
          if (w > m) m = w
        }
        return m
      }
      arr = [...arr].sort((a, b) => weightAt(a.adataIndex) - weightAt(b.adataIndex))
    }
    return arr
  }, [figure.coordinates, figure.cellIndices, colorData, panel.showHighlightOverlay, highlightLayers])

  // Color function — closed over the panel's settings + fetched data.
  // When `showHighlightOverlay` is on, the dataset's `highlightLayers`
  // are blended on top of the base color in creation order (same hard-gate
  // blend formula used by the main ScatterPlot).
  const getColor = useMemo(() => {
    const opacity = Math.round(figure.pointOpacity * 255)
    const fallback: [number, number, number, number] = [100, 149, 237, opacity]

    // Compute baseColorFn first.
    let baseColorFn: (d: { adataIndex: number }) => [number, number, number, number]

    if (colorData.kind === 'expression' && colorData.values && colorData.min !== undefined && colorData.max !== undefined) {
      const { values, min, max } = colorData
      const range = max - min || 1
      const scale = panel.colorScale
      baseColorFn = (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex]
        if (v == null) return [128, 128, 128, Math.round(opacity * 0.5)]
        const t = (v - min) / range
        const rgb = getColorFromScale(t, scale)
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    } else if (colorData.kind === 'metadata-numeric' && colorData.values && colorData.min !== undefined && colorData.max !== undefined) {
      const { values, min, max } = colorData
      const range = max - min || 1
      const scale = panel.colorScale
      baseColorFn = (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex]
        if (v == null) return [128, 128, 128, Math.round(opacity * 0.5)]
        const t = ((v as number) - min) / range
        const rgb = getColorFromScale(t, scale)
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    } else if (colorData.kind === 'metadata-category' && colorData.values && colorData.categories) {
      const palette = resolveCategoryPalette(colorData.categories.length, colorData.categoryColors)
      const values = colorData.values
      baseColorFn = (d: { adataIndex: number }): [number, number, number, number] => {
        const v = values[d.adataIndex] as number
        const rgb = palette[v] ?? palette[0] ?? [128, 128, 128]
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    } else if (colorData.kind === 'bivariate' && colorData.values1 && colorData.values2) {
      const v1 = colorData.values1
      const v2 = colorData.values2
      const colormap = panel.bivariateColormap
      baseColorFn = (d: { adataIndex: number }): [number, number, number, number] => {
        const u = v1[d.adataIndex] ?? 0
        const w = v2[d.adataIndex] ?? 0
        const rgb = getBivariateColor(u, w, colormap)
        return [rgb[0], rgb[1], rgb[2], opacity]
      }
    } else {
      baseColorFn = (_d: { adataIndex: number }) => fallback
    }

    // Highlight overlay: blend each layer's color over the base color in
    // creation order. Per-cell weight is hard-gated (in-threshold = intensity,
    // else 0) for geneset layers and a mask lookup for cellset layers.
    if (panel.showHighlightOverlay && highlightLayers.length > 0) {
      const compiled = highlightLayers.map((layer) => ({
        rgb: hexToRgb(layer.color) ?? [34, 197, 94] as [number, number, number],
        weight: layerWeightFn(layer),
      }))
      const fn = baseColorFn
      return (d: { adataIndex: number }): [number, number, number, number] => {
        const base = fn(d)
        let r = base[0], g = base[1], b = base[2]
        for (let k = 0; k < compiled.length; k++) {
          const w = compiled[k].weight(d.adataIndex)
          if (w <= 0) continue
          const ww = w > 1 ? 1 : w
          const lr = compiled[k].rgb
          r = r * (1 - ww) + lr[0] * ww
          g = g * (1 - ww) + lr[1] * ww
          b = b * (1 - ww) + lr[2] * ww
        }
        return [Math.round(r), Math.round(g), Math.round(b), base[3]]
      }
    }

    return baseColorFn
  }, [colorData, panel.colorScale, panel.bivariateColormap, panel.showHighlightOverlay, highlightLayers, figure.pointOpacity])

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

  const radius = figure.pointSize / 2

  // Pack positions into a Float32Array binary attribute. See ScatterPlot.tsx
  // for the rationale — deck.gl 9.2.6's per-cell accessor path has a stride
  // bug at N ≳ 44k that produces a phantom vertical line at X=0.
  const positionsBuf = useMemo(() => {
    const buf = new Float32Array(data.length * 2)
    for (let i = 0; i < data.length; i++) {
      const p = data[i].position
      buf[i * 2] = p[0]
      buf[i * 2 + 1] = p[1]
    }
    return buf
  }, [data])

  const deckLayers = [
    new ScatterplotLayer({
      id: `panel-${panel.id}`,
      data: {
        length: data.length,
        attributes: {
          getPosition: { value: positionsBuf, size: 2 },
        },
      },
      getRadius: radius,
      getFillColor: (_obj: unknown, info: { index: number }) => getColor(data[info.index]),
      radiusUnits: 'pixels',
      radiusMinPixels: 1,
      stroked: false,
      filled: true,
      updateTriggers: {
        getFillColor: [colorData, panel.colorScale, panel.bivariateColormap, panel.showHighlightOverlay, highlightLayers, figure.pointOpacity],
        getRadius: figure.pointSize,
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
      {figure.showGrid && (
        <GridOverlay
          divisions={figure.gridDivisions}
          color={figure.gridColor}
          lineWidth={figure.gridLineWidth}
        />
      )}
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

// Screen-fixed N×N grid overlay (the same N lines regardless of zoom/pan).
// Drawn as an absolutely-positioned SVG that sits on top of the deck.gl
// canvas but ignores pointer events so pan/zoom still work.
function GridOverlay({ divisions, color, lineWidth }: { divisions: number; color: string; lineWidth: number }) {
  const lines: React.ReactElement[] = []
  for (let i = 1; i < divisions; i++) {
    const p = (i / divisions) * 100
    lines.push(
      <line key={`v${i}`} x1={`${p}%`} x2={`${p}%`} y1="0" y2="100%" stroke={color} strokeWidth={lineWidth} />
    )
    lines.push(
      <line key={`h${i}`} x1="0" x2="100%" y1={`${p}%`} y2={`${p}%`} stroke={color} strokeWidth={lineWidth} />
    )
  }
  return (
    <svg
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        shapeRendering: 'crispEdges',
        opacity: 0.6,
      }}
    >
      {lines}
    </svg>
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
