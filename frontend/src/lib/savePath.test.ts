import { describe, it, expect } from 'vitest'
import { composeSavePath, splitSavePath } from './savePath'

describe('composeSavePath', () => {
  it('joins a directory and a filename', () => {
    expect(composeSavePath('/data/models', 'clf.pkl')).toBe('/data/models/clf.pkl')
  })

  it('does not double the separator when the directory has a trailing slash', () => {
    expect(composeSavePath('/data/models/', 'clf.pkl')).toBe('/data/models/clf.pkl')
  })

  it('handles the filesystem root', () => {
    expect(composeSavePath('/', 'clf.pkl')).toBe('/clf.pkl')
  })

  it('adds .pkl when no pickle extension was given', () => {
    // Typing a bare name is the common case; the backend would append the
    // suffix anyway, so showing it up front keeps the preview honest.
    expect(composeSavePath('/data', 'myclassifier')).toBe('/data/myclassifier.pkl')
  })

  it('leaves an existing pickle extension alone', () => {
    expect(composeSavePath('/data', 'a.pkl')).toBe('/data/a.pkl')
    expect(composeSavePath('/data', 'b.pickle')).toBe('/data/b.pickle')
  })

  it('is case-insensitive about the existing extension', () => {
    expect(composeSavePath('/data', 'c.PKL')).toBe('/data/c.PKL')
  })

  it('does not mistake a dotted name for an extension', () => {
    expect(composeSavePath('/data', 'pbmc.v2')).toBe('/data/pbmc.v2.pkl')
  })

  it('returns empty when there is no filename to save under', () => {
    expect(composeSavePath('/data', '')).toBe('')
    expect(composeSavePath('/data', '   ')).toBe('')
  })

  it('returns empty when no directory has been chosen', () => {
    expect(composeSavePath('', 'clf.pkl')).toBe('')
  })

  it('trims surrounding whitespace', () => {
    expect(composeSavePath(' /data ', ' clf.pkl ')).toBe('/data/clf.pkl')
  })
})

describe('splitSavePath', () => {
  it('splits a full path into directory and name', () => {
    expect(splitSavePath('/data/models/clf.pkl'))
      .toEqual({ dir: '/data/models', name: 'clf.pkl' })
  })

  it('handles a file at the root', () => {
    expect(splitSavePath('/clf.pkl')).toEqual({ dir: '/', name: 'clf.pkl' })
  })

  it('treats a bare name as having no directory', () => {
    expect(splitSavePath('clf.pkl')).toEqual({ dir: '', name: 'clf.pkl' })
  })

  it('returns empties for an empty path', () => {
    expect(splitSavePath('')).toEqual({ dir: '', name: '' })
  })

  it('round-trips with composeSavePath', () => {
    const path = '/data/models/clf.pkl'
    const { dir, name } = splitSavePath(path)
    expect(composeSavePath(dir, name)).toBe(path)
  })
})
