import { Figure } from '../store'
import { hexToRgb } from '../components/ScatterPlot'

// Composite all FigurePanel deck.gl canvases plus the figure-level styling
// (background, titles) into a single PNG. We read pixels directly from each
// panel's visible canvas — useDevicePixels=true means the source is already
// at native pixel density. The `scale` parameter upscales the OUTPUT
// canvas by an integer factor; the source pixels are scaled by drawImage's
// linear interpolation (so higher than 1× looks softer rather than sharper,
// but produces bigger output dimensions for layout-fit purposes).
export async function exportFigureAsPng(figure: Figure, scale: number = 1): Promise<void> {
  // Find the grid container in the DOM. It's the only element with grid
  // layout right under the FigureBuilder's body.
  const grids = document.querySelectorAll('[data-figure-grid="true"]')
  let grid: HTMLElement | null = null
  if (grids.length > 0) {
    grid = grids[0] as HTMLElement
  } else {
    // Fallback: find by searching for an element containing canvases that match the panel count
    const candidates = Array.from(document.querySelectorAll('div')).filter((el) => {
      const canvases = el.querySelectorAll(':scope > div canvas')
      return canvases.length === figure.panels.length
    })
    grid = (candidates[0] as HTMLElement) ?? null
  }
  if (!grid) {
    alert('Could not locate the figure grid to export. Try clicking on the figure area first.')
    return
  }

  const gridRect = grid.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1

  // Output canvas dimensions = grid CSS size × dpr × user-chosen scale.
  // dpr captures display-pixel density (Retina); scale lets the user request a
  // bigger output (interpolated up for >1).
  const outScale = dpr * scale
  const outWidth = Math.round(gridRect.width * outScale)
  const outHeight = Math.round(gridRect.height * outScale)

  const dest = document.createElement('canvas')
  dest.width = outWidth
  dest.height = outHeight
  const ctx = dest.getContext('2d')
  if (!ctx) {
    alert('Could not get 2D canvas context for export.')
    return
  }

  // Fill background with figure-level color.
  ctx.fillStyle = figure.background
  ctx.fillRect(0, 0, outWidth, outHeight)

  // Iterate panel containers. Each panel container is a direct child div of
  // the grid (so its position can be measured relative to the grid).
  const panelEls = grid.querySelectorAll(':scope > div[data-figure-panel="true"]')
  panelEls.forEach((el) => {
    const panelDiv = el as HTMLElement
    const panelRect = panelDiv.getBoundingClientRect()
    const x = (panelRect.left - gridRect.left) * outScale
    const y = (panelRect.top - gridRect.top) * outScale
    const w = panelRect.width * outScale
    const h = panelRect.height * outScale

    const panelId = panelDiv.dataset.panelId
    const panel = figure.panels.find((p) => p.id === panelId)
    if (panel) {
      // Per-panel background
      ctx.fillStyle = panel.background
      ctx.fillRect(x, y, w, h)
    }

    // Find this panel's deck.gl canvas (deck.gl creates a <canvas> inside its container).
    const canvas = panelDiv.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) {
      try {
        ctx.drawImage(canvas, x, y, w, h)
      } catch (err) {
        console.warn('Failed to copy panel canvas:', err)
      }
    }

    // Grid overlay — N×N evenly-spaced lines, screen-fixed (matches the
    // live SVG GridOverlay). Drawn after the canvas so it sits on top.
    if (figure.showGrid) {
      ctx.save()
      ctx.globalAlpha = 0.6
      ctx.strokeStyle = figure.gridColor
      ctx.lineWidth = figure.gridLineWidth * outScale
      for (let i = 1; i < figure.gridDivisions; i++) {
        const fx = x + (i / figure.gridDivisions) * w
        const fy = y + (i / figure.gridDivisions) * h
        ctx.beginPath()
        ctx.moveTo(fx, y)
        ctx.lineTo(fx, y + h)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(x, fy)
        ctx.lineTo(x + w, fy)
        ctx.stroke()
      }
      ctx.restore()
    }

    // Border
    if (panel?.showBorder) {
      ctx.strokeStyle = '#444'
      ctx.lineWidth = 1 * outScale
      ctx.strokeRect(x, y, w, h)
    }

    // Panel title overlay
    if (panel?.title) {
      ctx.font = `600 ${14 * outScale}px sans-serif`
      ctx.fillStyle = '#fff'
      // Shadow for legibility on any background
      ctx.shadowColor = 'rgba(0,0,0,0.8)'
      ctx.shadowBlur = 4 * outScale
      ctx.fillText(panel.title, x + 8 * outScale, y + 20 * outScale)
      ctx.shadowBlur = 0
    }
  })

  // Figure-level title at top center
  if (figure.title) {
    ctx.font = `600 ${16 * outScale}px sans-serif`
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.shadowColor = 'rgba(0,0,0,0.8)'
    ctx.shadowBlur = 4 * outScale
    ctx.fillText(figure.title, outWidth / 2, 24 * outScale)
    ctx.shadowBlur = 0
    ctx.textAlign = 'start'
  }

  // Encode and download
  const blob = await new Promise<Blob | null>((resolve) => dest.toBlob(resolve, 'image/png'))
  if (!blob) {
    alert('Export failed: could not produce PNG.')
    return
  }
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  const now = new Date()
  const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`
  const safeTitle = figure.title ? figure.title.replace(/[^a-z0-9_-]+/gi, '_').slice(0, 40) : 'figure'
  a.href = url
  a.download = `${safeTitle}_${stamp}.png`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Re-export for consumers that want to construct legend colors etc.
export { hexToRgb }
