/**
 * Copy text to the clipboard, returning whether it worked.
 *
 * The async Clipboard API is the normal path. It is absent or rejects on a
 * plain-http origin that is not localhost — a LAN address, say — so fall back
 * to the legacy select-and-execCommand route through a scratch textarea.
 * Never throws: the caller only needs to know whether to show "copied".
 */
export async function copyText(text: string): Promise<boolean> {
  const clipboard = typeof navigator !== 'undefined' ? navigator.clipboard : undefined
  if (clipboard && typeof clipboard.writeText === 'function') {
    try {
      await clipboard.writeText(text)
      return true
    } catch {
      // fall through to the legacy path
    }
  }
  if (typeof document === 'undefined') return false
  const el = document.createElement('textarea')
  el.value = text
  // Keep the scratch element out of the layout and out of the tab order.
  el.setAttribute('readonly', '')
  el.style.position = 'fixed'
  el.style.opacity = '0'
  document.body.appendChild(el)
  try {
    el.select()
    return Boolean(document.execCommand('copy'))
  } catch {
    return false
  } finally {
    document.body.removeChild(el)
  }
}
