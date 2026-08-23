import { useRef } from 'react'

// The boundary between two tiled dataset panes. It used to be a 1px border the
// left pane drew on its own edge; making it an element is what lets you grab it.
// The visible line stays 1px — the hit area around it is what widens.

export default function PaneDivider({
  axis = 'x',
  onResizeStart,
  onResize,
  onReset,
}: {
  /** 'x' sits between columns and moves left/right; 'y' between rows. */
  axis?: 'x' | 'y'
  onResizeStart: () => void
  /** Pointer offset from where the drag began, in px. Cumulative, not
   *  incremental: dragging past a pane's minimum and back returns it to where
   *  the pointer says it should be rather than to where it drifted. */
  onResize: (deltaPx: number) => void
  onReset: () => void
}) {
  const start = useRef(0)
  const vertical = axis === 'y'
  const cursor = vertical ? 'row-resize' : 'col-resize'
  const along = (e: { clientX: number; clientY: number }) => (vertical ? e.clientY : e.clientX)

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()   // don't let the pane underneath claim the active slot
    start.current = along(e)
    onResizeStart()
    const move = (ev: PointerEvent) => onResize(along(ev) - start.current)
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    // Held on the body so the cursor survives crossing a WebGL canvas, which
    // sets its own.
    document.body.style.cursor = cursor
    document.body.style.userSelect = 'none'
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  return (
    <div
      onPointerDown={onPointerDown}
      onDoubleClick={(e) => { e.stopPropagation(); onReset() }}
      title="Drag to resize · double-click for equal widths"
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        cursor,
        backgroundColor: 'transparent',
        zIndex: 13,
      }}
    >
      <div
        style={{
          position: 'absolute',
          ...(vertical
            ? { left: 0, right: 0, top: 3, height: 1 }
            : { top: 0, bottom: 0, left: 3, width: 1 }),
          backgroundColor: '#0f3460',
          pointerEvents: 'none',
        }}
      />
    </div>
  )
}
