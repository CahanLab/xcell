import { useEffect } from 'react'
import { useStore } from '../store'
import { MESSAGES } from '../messages'

export default function GeneMaskModal() {
  const open = useStore((s) => s.geneMaskModalOpen)
  const setOpen = useStore((s) => s.setGeneMaskModalOpen)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  if (!open) return null

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={() => setOpen(false)}
    >
      <div
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '6px',
          padding: '20px',
          minWidth: '420px',
          maxWidth: '520px',
          color: '#ccc',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px',
          }}
        >
          <h3 style={{ margin: 0, fontSize: '14px', color: '#ccc' }}>
            {MESSAGES.geneMask.title}
          </h3>
          <button
            onClick={() => setOpen(false)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              fontSize: '16px',
            }}
            title="Close"
          >
            ✕
          </button>
        </div>
        <div style={{ fontSize: '11px', color: '#888' }}>
          (modal body — populated in next task)
        </div>
      </div>
    </div>
  )
}
