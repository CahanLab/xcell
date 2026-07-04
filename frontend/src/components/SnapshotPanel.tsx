import { useLayoutEffect, useRef, useCallback } from 'react'
import { useStore, EmbeddingSnapshot } from '../store'

const HEADER_H = 26
const MIN_W = 120
const MIN_H = 90

function parseHex(hex: string): [number, number, number] {
  let h = (hex || '').trim().replace(/^#/, '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [26, 26, 46]
  return [r, g, b]
}

// Re-render a snapshot's frozen points into a 2D canvas at the given CSS size.
// Points are splatted straight into an ImageData buffer (fast, no per-point
// fillStyle switches), preserving the capture-time draw order so later/top
// points win overlaps — matching how deck.gl stacks them. Cluster labels are
// drawn on top with a dark halo, mirroring the live SVG overlay. Redrawing
// from the stored coordinates (rather than scaling a baked bitmap) keeps the
// snapshot crisp at any panel size.
export function renderSnapshotToCanvas(
  canvas: HTMLCanvasElement,
  snap: EmbeddingSnapshot,
  cssW: number,
  cssH: number,
) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const W = Math.max(1, Math.round(cssW * dpr))
  const H = Math.max(1, Math.round(cssH * dpr))
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const { minX, minY, maxX, maxY } = snap.bounds
  const dataW = maxX - minX || 1
  const dataH = maxY - minY || 1
  const margin = 4 * dpr
  const scale = Math.min((W - 2 * margin) / dataW, (H - 2 * margin) / dataH)
  // Center the (possibly letterboxed) data extent in the canvas.
  const offX = (W - dataW * scale) / 2
  const offY = (H - dataH * scale) / 2
  // deck.gl's OrthographicView is Y_DOWN by default (larger data Y = lower on
  // screen), so we DON'T flip Y here — orientation stays identical to the plot.
  const toX = (x: number) => offX + (x - minX) * scale
  const toY = (y: number) => offY + (y - minY) * scale

  const img = ctx.createImageData(W, H)
  const buf = img.data
  // Background fill straight into the buffer (putImageData ignores prior pixels).
  const [br, bgc, bb] = parseHex(snap.bg)
  for (let p = 0; p < buf.length; p += 4) {
    buf[p] = br
    buf[p + 1] = bgc
    buf[p + 2] = bb
    buf[p + 3] = 255
  }

  // Point radius grows gently with canvas size: crisp single pixels when the
  // panel is small, slightly fuller dots when enlarged.
  const r = Math.max(0, Math.min(3, Math.round(Math.min(W, H) / 260)))
  const pts = snap.points
  const cols = snap.colors
  const n = pts.length >> 1
  for (let i = 0; i < n; i++) {
    const cx = (toX(pts[i * 2]) + 0.5) | 0
    const cy = (toY(pts[i * 2 + 1]) + 0.5) | 0
    const cr = cols[i * 3]
    const cg = cols[i * 3 + 1]
    const cb = cols[i * 3 + 2]
    const y0 = cy - r, y1 = cy + r
    const x0 = cx - r, x1 = cx + r
    for (let yy = y0; yy <= y1; yy++) {
      if (yy < 0 || yy >= H) continue
      const row = yy * W
      for (let xx = x0; xx <= x1; xx++) {
        if (xx < 0 || xx >= W) continue
        const o = (row + xx) * 4
        buf[o] = cr
        buf[o + 1] = cg
        buf[o + 2] = cb
        buf[o + 3] = 255
      }
    }
  }
  ctx.putImageData(img, 0, 0)

  // Cluster labels on top (skipped when the panel is tiny, to avoid clutter).
  if (snap.labels.length > 0 && Math.min(cssW, cssH) > 90) {
    const fontPx = Math.max(9, Math.min(13, Math.round(Math.min(cssW, cssH) / 16))) * dpr
    ctx.font = `700 ${fontPx}px system-ui, -apple-system, sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.lineJoin = 'round'
    ctx.lineWidth = Math.max(2, fontPx / 4)
    ctx.strokeStyle = 'rgba(0,0,0,0.85)'
    ctx.fillStyle = '#fff'
    for (const lab of snap.labels) {
      const lx = toX(lab.x)
      const ly = toY(lab.y)
      if (lx < 0 || lx > W || ly < 0 || ly > H) continue
      ctx.strokeText(lab.text, lx, ly)
      ctx.fillText(lab.text, lx, ly)
    }
  }
}

// One pinned snapshot: a floating, draggable, resizable panel that redraws the
// frozen embedding into a canvas. The store is the single source of truth for
// geometry (x/y/w/h); drag/resize write straight to it, throttled to one commit
// per animation frame so heavy canvases don't thrash. The canvas only redraws
// when size/content change (not on move), so panning a panel stays cheap.
function SnapshotPanel({ snap }: { snap: EmbeddingSnapshot }) {
  const update = useStore((s) => s.updateEmbeddingSnapshot)
  const remove = useStore((s) => s.removeEmbeddingSnapshot)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  // Redraw only when the size or frozen content changes — NOT on x/y moves.
  useLayoutEffect(() => {
    if (snap.minimized) return
    const canvas = canvasRef.current
    if (canvas) renderSnapshotToCanvas(canvas, snap, snap.w, snap.h)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap.w, snap.h, snap.minimized, snap.points, snap.colors, snap.labels, snap.bg, snap.pointSize])

  const parentBounds = () => {
    const parent = rootRef.current?.offsetParent as HTMLElement | null
    return parent
      ? { w: parent.clientWidth, h: parent.clientHeight }
      : { w: Infinity, h: Infinity }
  }

  const startDrag = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const start = { mx: e.clientX, my: e.clientY, x: snap.x, y: snap.y }
    const totalH = HEADER_H + snap.h
    let nx = snap.x, ny = snap.y
    let raf = 0
    const commit = () => { raf = 0; update(snap.id, { x: nx, y: ny }) }
    const onMove = (ev: MouseEvent) => {
      const { w: pw, h: ph } = parentBounds()
      nx = Math.max(0, Math.min(start.x + (ev.clientX - start.mx), Math.max(0, pw - snap.w)))
      ny = Math.max(0, Math.min(start.y + (ev.clientY - start.my), Math.max(0, ph - totalH)))
      if (!raf) raf = requestAnimationFrame(commit)
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      if (raf) cancelAnimationFrame(raf)
      update(snap.id, { x: nx, y: ny })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [snap.x, snap.y, snap.w, snap.h, snap.id, update])

  const startResize = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const start = { mx: e.clientX, my: e.clientY, w: snap.w, h: snap.h }
    let nw = snap.w, nh = snap.h
    let raf = 0
    const commit = () => { raf = 0; update(snap.id, { w: nw, h: nh }) }
    const onMove = (ev: MouseEvent) => {
      const { w: pw, h: ph } = parentBounds()
      const maxW = Math.max(MIN_W, pw - snap.x)
      const maxH = Math.max(MIN_H, ph - snap.y - HEADER_H)
      nw = Math.max(MIN_W, Math.min(start.w + (ev.clientX - start.mx), maxW))
      nh = Math.max(MIN_H, Math.min(start.h + (ev.clientY - start.my), maxH))
      if (!raf) raf = requestAnimationFrame(commit)
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      if (raf) cancelAnimationFrame(raf)
      update(snap.id, { w: nw, h: nh })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [snap.x, snap.y, snap.w, snap.h, snap.id, update])

  return (
    <div
      ref={rootRef}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        left: snap.x,
        top: snap.y,
        width: snap.w,
        backgroundColor: 'rgba(22, 33, 62, 0.95)',
        borderRadius: 8,
        overflow: 'hidden',
        zIndex: 15,
        boxShadow: '0 3px 14px rgba(0,0,0,0.45)',
        border: '1px solid rgba(255,255,255,0.10)',
      }}
    >
      <div
        onMouseDown={startDrag}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 6,
          height: HEADER_H,
          padding: '0 6px 0 9px',
          cursor: 'move',
          userSelect: 'none',
          borderBottom: snap.minimized ? 'none' : '1px solid rgba(255,255,255,0.08)',
        }}
        title="Drag to move"
      >
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: '#cbd5e1',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
          title={snap.title}
        >
          <span style={{ color: '#4ecdc4', marginRight: 5 }}>◈</span>
          {snap.title}
        </span>
        <span style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
          <button
            onClick={(e) => { e.stopPropagation(); update(snap.id, { minimized: !snap.minimized }) }}
            onMouseDown={(e) => e.stopPropagation()}
            style={iconBtn}
            title={snap.minimized ? 'Expand' : 'Minimize'}
          >
            {snap.minimized ? '▢' : '–'}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); remove(snap.id) }}
            onMouseDown={(e) => e.stopPropagation()}
            style={iconBtn}
            title="Remove snapshot"
          >
            ×
          </button>
        </span>
      </div>

      {!snap.minimized && (
        <div style={{ position: 'relative', width: snap.w, height: snap.h, lineHeight: 0 }}>
          <canvas
            ref={canvasRef}
            style={{ width: snap.w, height: snap.h, display: 'block' }}
          />
          {/* Resize handle (bottom-right corner) */}
          <div
            onMouseDown={startResize}
            title="Drag to resize"
            style={{
              position: 'absolute',
              right: 0,
              bottom: 0,
              width: 16,
              height: 16,
              cursor: 'nwse-resize',
              background:
                'linear-gradient(135deg, transparent 0 50%, rgba(255,255,255,0.5) 50% 62%, transparent 62% 74%, rgba(255,255,255,0.5) 74% 86%, transparent 86%)',
            }}
          />
        </div>
      )}
    </div>
  )
}

const iconBtn: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  color: '#8b98a8',
  cursor: 'pointer',
  fontSize: 14,
  lineHeight: 1,
  padding: '2px 4px',
  borderRadius: 3,
}

// Renders every pinned snapshot captured from the given plot slot. Mounted
// inside each ScatterPlot's container so panels float over that plot.
export function SnapshotLayer({ slotKey }: { slotKey: string }) {
  const snapshots = useStore((s) => s.embeddingSnapshots)
  const mine = snapshots.filter((s) => s.slotKey === slotKey)
  if (mine.length === 0) return null
  return (
    <>
      {mine.map((snap) => (
        <SnapshotPanel key={snap.id} snap={snap} />
      ))}
    </>
  )
}
