import { useEffect, useRef, type ReactNode } from 'react'

/** A confirm step for anything that wipes: New session, Clear all gene sets,
 *  the Gene Set Manager's bulk delete. Unlike window.confirm it can say what
 *  will go (names, counts) and carry an escape hatch such as a backup button.
 *
 *  Sits above every other modal, and swallows its own Escape so a dialog
 *  opened from the manager closes without taking the manager with it. Cancel
 *  has focus, so a stray Enter never confirms a wipe. */
export default function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  danger = true,
  busy = false,
  error = null,
  confirmDisabled = false,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  children?: ReactNode
  confirmLabel: string
  danger?: boolean
  busy?: boolean
  error?: string | null
  confirmDisabled?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  const cancelRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    cancelRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Capture phase on window runs before any bubble-phase listener, so the
      // modal underneath never sees this Escape.
      e.stopPropagation()
      if (!busy) onCancel()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open, busy, onCancel])

  if (!open) return null

  const accent = danger ? '#e94560' : '#4ecdc4'
  const disabled = busy || confirmDisabled

  return (
    <div
      onClick={() => { if (!busy) onCancel() }}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 1300,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#16213e', border: `1px solid ${danger ? '#5a2a3a' : '#0f3460'}`, borderRadius: '8px',
          padding: '18px 22px', width: 'min(460px, 92vw)', color: '#eee',
          boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '10px', color: danger ? '#f07d92' : '#eee' }}>
          {title}
        </div>
        <div style={{ fontSize: '12px', color: '#ccc', lineHeight: 1.5 }}>{children}</div>
        {error && <div style={{ color: '#e94560', fontSize: '12px', marginTop: '10px' }}>{error}</div>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
          <button
            ref={cancelRef}
            onClick={onCancel}
            disabled={busy}
            style={{
              padding: '6px 14px', fontSize: '12px', backgroundColor: 'transparent', color: '#aaa',
              border: '1px solid #555', borderRadius: '4px', cursor: busy ? 'default' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={disabled}
            style={{
              padding: '6px 14px', fontSize: '12px', fontWeight: 600,
              backgroundColor: disabled ? '#1a1a2e' : accent, color: disabled ? '#666' : (danger ? '#fff' : '#000'),
              border: 'none', borderRadius: '4px', cursor: disabled ? 'not-allowed' : 'pointer',
            }}
          >
            {busy ? 'Working…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
