import { useEffect, useRef } from 'react'
import { useStore, type BivariateColormap } from '../store'
import { BIVARIATE_COLORMAPS, getBivariateColor, resolveCategoryPalette } from '../lib/cellColors'

// Plot chrome shared by every embedding view: the draggable panel shell and the
// four legends that sit in it. Lifted out of App.tsx so a dataset pane can be a
// component in its own right rather than a block copied once per slot.

const styles = {
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    marginBottom: '4px',
  },
  legendColor: {
    width: '12px',
    height: '12px',
    borderRadius: '2px',
  },
  colorBar: {
    width: '120px',
    height: '12px',
    borderRadius: '2px',
    marginBottom: '4px',
  },
  colorBarLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '10px',
    color: '#888',
  },
}

// A draggable + minimizable floating panel used for all plot legends. Position
// and minimized state persist in the store keyed by `id`, so they survive
// color-mode switches and re-renders. Until dragged, the panel sits at its
// default bottom-right corner.
export function FloatingPanel({
  id,
  title,
  titleExtra,
  defaultCorner,
  children,
}: {
  id: string
  title: string
  titleExtra?: React.ReactNode
  defaultCorner?: React.CSSProperties
  children: React.ReactNode
}) {
  const panel = useStore((s) => s.legendPanels[id])
  const setPos = useStore((s) => s.setLegendPanelPos)
  const toggleMin = useStore((s) => s.toggleLegendPanelMinimized)
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef<{ mouseX: number; mouseY: number; startX: number; startY: number } | null>(null)

  const moved = !!panel && panel.x != null && panel.y != null
  const minimized = panel?.minimized ?? false

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    const el = ref.current
    if (!el) return
    e.preventDefault()
    // Start from the stored position, or the element's current corner offset
    // the first time it's dragged.
    const startX = moved ? (panel!.x as number) : el.offsetLeft
    const startY = moved ? (panel!.y as number) : el.offsetTop
    drag.current = { mouseX: e.clientX, mouseY: e.clientY, startX, startY }
    const onMove = (ev: MouseEvent) => {
      if (!drag.current || !ref.current) return
      let nx = drag.current.startX + (ev.clientX - drag.current.mouseX)
      let ny = drag.current.startY + (ev.clientY - drag.current.mouseY)
      const parent = ref.current.offsetParent as HTMLElement | null
      if (parent) {
        const maxX = Math.max(0, parent.clientWidth - ref.current.offsetWidth)
        const maxY = Math.max(0, parent.clientHeight - ref.current.offsetHeight)
        nx = Math.max(0, Math.min(nx, maxX))
        ny = Math.max(0, Math.min(ny, maxY))
      }
      setPos(id, nx, ny)
    }
    const onUp = () => {
      drag.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const positionStyle: React.CSSProperties = moved
    ? { left: panel!.x, top: panel!.y, right: 'auto', bottom: 'auto' }
    : (defaultCorner ?? { bottom: 20, right: 20 })

  return (
    <div
      ref={ref}
      style={{
        position: 'absolute',
        ...positionStyle,
        backgroundColor: 'rgba(22, 33, 62, 0.92)',
        borderRadius: 8,
        overflow: 'hidden',
        zIndex: 12,
        boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
      }}
    >
      <div
        onMouseDown={onMouseDown}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          padding: '5px 6px 5px 10px',
          cursor: 'move',
          userSelect: 'none',
          borderBottom: minimized ? 'none' : '1px solid rgba(255,255,255,0.08)',
        }}
        title="Drag to move"
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: '#aaa',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            maxWidth: 200,
          }}
        >
          {title}
          {titleExtra}
        </span>
        <button
          onClick={(e) => { e.stopPropagation(); toggleMin(id) }}
          onMouseDown={(e) => e.stopPropagation()}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#888',
            cursor: 'pointer',
            fontSize: 14,
            lineHeight: 1,
            padding: '0 2px',
          }}
          title={minimized ? 'Expand' : 'Minimize'}
        >
          {minimized ? '▢' : '–'}
        </button>
      </div>
      {!minimized && (
        <div style={{ padding: '8px 12px 12px', maxHeight: '40vh', overflowY: 'auto' }}>
          {children}
        </div>
      )}
    </div>
  )
}

export function CategoryLegend({ colorBy, panelId }: { colorBy: { name: string; categories?: string[]; colors?: string[]; dtype: string }; panelId: string }) {
  if (colorBy.dtype !== 'category' || !colorBy.categories) {
    return null
  }

  const palette = resolveCategoryPalette(colorBy.categories.length, colorBy.colors)

  return (
    <FloatingPanel id={panelId} title={colorBy.name}>
      {colorBy.categories.map((cat, i) => {
        const [r, g, b] = palette[i] ?? palette[0]
        return (
          <div key={cat} style={styles.legendItem}>
            <div
              style={{
                ...styles.legendColor,
                backgroundColor: `rgb(${r}, ${g}, ${b})`,
              }}
            />
            <span>{cat}</span>
          </div>
        )
      })}
    </FloatingPanel>
  )
}

const COLOR_SCALE_GRADIENTS: Record<string, string> = {
  viridis: 'linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))',
  plasma: 'linear-gradient(to right, rgb(13,8,135), rgb(126,3,168), rgb(204,71,120), rgb(248,149,64), rgb(240,249,33))',
  magma: 'linear-gradient(to right, rgb(0,0,4), rgb(81,18,124), rgb(183,55,121), rgb(252,137,97), rgb(252,253,191))',
  inferno: 'linear-gradient(to right, rgb(0,0,4), rgb(66,10,104), rgb(147,38,103), rgb(221,81,58), rgb(252,165,10), rgb(252,255,164))',
  cividis: 'linear-gradient(to right, rgb(0,32,81), rgb(82,95,110), rgb(152,136,62), rgb(253,234,69))',
  coolwarm: 'linear-gradient(to right, rgb(59,76,192), rgb(112,146,208), rgb(197,197,197), rgb(230,128,103), rgb(180,4,38))',
  blues: 'linear-gradient(to right, rgb(247,251,255), rgb(107,174,214), rgb(8,48,107))',
  reds: 'linear-gradient(to right, rgb(255,245,240), rgb(251,106,74), rgb(103,0,13))',
  sunset: 'linear-gradient(to right, rgb(40,11,86), rgb(219,75,109), rgb(255,222,135))',
  ocean: 'linear-gradient(to right, rgb(2,17,51), rgb(28,119,150), rgb(120,255,214))',
  grape: 'linear-gradient(to right, rgb(28,27,92), rgb(123,31,162), rgb(224,64,251))',
  mint: 'linear-gradient(to right, rgb(4,40,63), rgb(0,150,136), rgb(173,255,96))',
}

export function ContinuousLegend({ name, min, max, panelId, colorScale }: { name: string; min: number; max: number; panelId: string; colorScale: string }) {
  return (
    <FloatingPanel id={panelId} title={name}>
      <div
        style={{
          ...styles.colorBar,
          background: COLOR_SCALE_GRADIENTS[colorScale] || COLOR_SCALE_GRADIENTS.viridis,
        }}
      />
      <div style={styles.colorBarLabels}>
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </FloatingPanel>
  )
}

// Continuous colorbar for gene/gene-set expression coloring (honors the chosen
// color scale). Extracted so it can live in a movable/minimizable FloatingPanel.
export function ExpressionLegend({
  panelId,
  title,
  transform,
  min,
  max,
  colorScale,
}: {
  panelId: string
  title: string
  transform?: string
  min: number
  max: number
  colorScale: string
}) {
  return (
    <FloatingPanel
      id={panelId}
      title={title}
      titleExtra={
        transform === 'log1p' ? (
          <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>(log1p)</span>
        ) : undefined
      }
    >
      <div style={{ ...styles.colorBar, background: COLOR_SCALE_GRADIENTS[colorScale] || COLOR_SCALE_GRADIENTS.viridis }} />
      <div style={styles.colorBarLabels}>
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </FloatingPanel>
  )
}

// Canvas-based bivariate legend that accurately reflects the bilinear interpolation
export function BivariateLegend({
  bivariateData,
  colormap,
  sortReversed,
  onToggleSort,
  panelId,
}: {
  bivariateData: { genes1: string[]; genes2: string[]; transform?: string }
  colormap: BivariateColormap
  sortReversed: boolean
  onToggleSort: () => void
  panelId: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const size = 80

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const imageData = ctx.createImageData(size, size)
    const data = imageData.data

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1)  // gene set 1 (horizontal, 0=left, 1=right)
        const v = 1 - y / (size - 1)  // gene set 2 (vertical, 0=bottom, 1=top, but y increases down)
        const color = getBivariateColor(u, v, colormap)
        const idx = (y * size + x) * 4
        data[idx] = color[0]
        data[idx + 1] = color[1]
        data[idx + 2] = color[2]
        data[idx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [colormap])

  // Get corner colors for axis labels
  const corners = BIVARIATE_COLORMAPS[colormap]
  const color1 = `rgb(${corners.c10.join(',')})`  // High gene1 color
  const color2 = `rgb(${corners.c01.join(',')})`  // High gene2 color

  return (
    <FloatingPanel
      id={panelId}
      title="Bivariate Expression"
      titleExtra={
        bivariateData.transform === 'log1p' ? (
          <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>(log1p)</span>
        ) : undefined
      }
    >
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        style={{ borderRadius: '4px', marginBottom: '4px' }}
      />
      <div style={{ fontSize: '10px', color: '#888' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: color1 }}>→</span>
          <span>{bivariateData.genes1.length === 1 ? bivariateData.genes1[0] : `${bivariateData.genes1.length} genes`}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: color2 }}>↑</span>
          <span>{bivariateData.genes2.length === 1 ? bivariateData.genes2[0] : `${bivariateData.genes2.length} genes`}</span>
        </div>
      </div>
      {/* Sort order toggle button */}
      <button
        onClick={onToggleSort}
        style={{
          marginTop: '8px',
          padding: '4px 8px',
          fontSize: '10px',
          backgroundColor: sortReversed ? '#4ecdc4' : '#0f3460',
          color: sortReversed ? '#000' : '#aaa',
          border: '1px solid #1a1a2e',
          borderRadius: '4px',
          cursor: 'pointer',
          width: '100%',
        }}
        title={sortReversed
          ? 'Currently: Low expression on top. Click to show high expression on top.'
          : 'Currently: High expression on top. Click to show low expression on top.'}
      >
        {sortReversed ? '↓ Low on Top' : '↑ High on Top'}
      </button>
    </FloatingPanel>
  )
}
