import { useRef } from 'react'

// The boundary between two tiled dataset panes. It used to be a 1px border the
// left pane drew on its own edge; making it an element is what lets you grab it.
// The visible line stays 1px — the hit area around it is what widens.

export default function PaneDivider({
  onResizeStart,
  onResize,
  onReset,
}: {
  onResizeStart: () => void
  /** Pointer offset from where the drag began, in px. Cumulative, not
   *  incremental: dragging past a pane's minimum and back returns it to where
   *  the pointer says it should be rather than to where it drifted. */
  onResize: (deltaPx: number) => void
  onReset: () => void
}) {
  const startX = useRef(0)

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()   // don't let the pane underneath claim the active slot
    startX.current = e.clientX
    onResizeStart()
    const move = (ev: PointerEvent) => onResize(ev.clientX - startX.current)
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    // Held on the body so the cursor survives crossing a WebGL canvas, which
    // sets its own.
    document.body.style.cursor = 'col-resize'
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
        flex: '0 0 7px',
        position: 'relative',
        cursor: 'col-resize',
        backgroundColor: 'transparent',
        zIndex: 13,
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          left: 3,
          width: 1,
          backgroundColor: '#0f3460',
          pointerEvents: 'none',
        }}
      />
    </div>
  )
}
