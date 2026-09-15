import { afterEach, describe, expect, it, vi } from 'vitest'
import { copyText } from './clipboard'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('copyText', () => {
  it('writes through the async Clipboard API when it is available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    await expect(copyText('Sox9')).resolves.toBe(true)
    expect(writeText).toHaveBeenCalledWith('Sox9')
  })

  it('falls back to a selection copy when the Clipboard API rejects', async () => {
    // Plain-http origins expose navigator.clipboard but reject every call.
    vi.stubGlobal('navigator', { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } })
    const el = { value: '', style: {} as Record<string, string>, select: vi.fn(), setAttribute: vi.fn() }
    const execCommand = vi.fn().mockReturnValue(true)
    const body = { appendChild: vi.fn(), removeChild: vi.fn() }
    vi.stubGlobal('document', { createElement: vi.fn(() => el), body, execCommand })
    await expect(copyText('Col2a1')).resolves.toBe(true)
    expect(el.value).toBe('Col2a1')
    expect(el.select).toHaveBeenCalled()
    expect(execCommand).toHaveBeenCalledWith('copy')
    // The scratch element must not be left in the page.
    expect(body.removeChild).toHaveBeenCalledWith(el)
  })

  it('resolves false rather than throwing when no clipboard path exists', async () => {
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('document', undefined)
    await expect(copyText('Runx2')).resolves.toBe(false)
  })
})
