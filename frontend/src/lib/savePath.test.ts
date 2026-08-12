import { describe, it, expect } from 'vitest'
import { composeSavePath, splitSavePath } from './savePath'

describe('composeSavePath', () => {
  it('joins a directory and a filename', () => {
    expect(composeSavePath('/data/models', 'clf.pkl', '.pkl')).toBe('/data/models/clf.pkl')
  })

  it('does not double the separator when the directory has a trailing slash', () => {
    expect(composeSavePath('/data/models/', 'clf.pkl', '.pkl')).toBe('/data/models/clf.pkl')
  })

  it('handles the filesystem root', () => {
    expect(composeSavePath('/', 'clf.pkl', '.pkl')).toBe('/clf.pkl')
  })

  it('adds .pkl when no pickle extension was given', () => {
    // Typing a bare name is the common case; the backend would append the
    // suffix anyway, so showing it up front keeps the preview honest.
    expect(composeSavePath('/data', 'myclassifier', '.pkl')).toBe('/data/myclassifier.pkl')
  })

  it('leaves any accepted extension alone, appending only the first', () => {
    // A list means "append this one, but accept any of these" — .pickle is a
    // perfectly good pickle and must not become b.pickle.pkl.
    const PKL = ['.pkl', '.pickle']
    expect(composeSavePath('/data', 'a.pkl', PKL)).toBe('/data/a.pkl')
    expect(composeSavePath('/data', 'b.pickle', PKL)).toBe('/data/b.pickle')
    expect(composeSavePath('/data', 'c', PKL)).toBe('/data/c.pkl')
  })

  it('is case-insensitive about the existing extension', () => {
    expect(composeSavePath('/data', 'c.PKL', '.pkl')).toBe('/data/c.PKL')
  })

  it('does not mistake a dotted name for an extension', () => {
    expect(composeSavePath('/data', 'pbmc.v2', '.pkl')).toBe('/data/pbmc.v2.pkl')
  })

  it('returns empty when there is no filename to save under', () => {
    expect(composeSavePath('/data', '', '.pkl')).toBe('')
    expect(composeSavePath('/data', '   ', '.pkl')).toBe('')
  })

  it('returns empty when no directory has been chosen', () => {
    expect(composeSavePath('', 'clf.pkl', '.pkl')).toBe('')
  })

  it('trims surrounding whitespace', () => {
    expect(composeSavePath(' /data ', ' clf.pkl ', '.pkl')).toBe('/data/clf.pkl')
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
    expect(composeSavePath(dir, name, '.pkl')).toBe(path)
  })

  it('round-trips for any extension, not just pickles', () => {
    const path = '/data/E11.5_best_102424_xcell.h5ad'
    const { dir, name } = splitSavePath(path)
    expect(composeSavePath(dir, name, '.h5ad')).toBe(path)
  })
})

describe('composeSavePath with other extensions', () => {
  it('appends the extension it is given, not a hardcoded one', () => {
    expect(composeSavePath('/data', 'export', '.h5ad')).toBe('/data/export.h5ad')
  })

  it('leaves a name that already has the extension alone', () => {
    expect(composeSavePath('/data', 'export.h5ad', '.h5ad')).toBe('/data/export.h5ad')
  })

  it('matches the extension case-insensitively', () => {
    // A user who typed .H5AD meant .h5ad, and appending again would give
    // export.H5AD.h5ad.
    expect(composeSavePath('/data', 'EXPORT.H5AD', '.h5ad')).toBe('/data/EXPORT.H5AD')
  })

  it('appends nothing when the extension is empty', () => {
    expect(composeSavePath('/data', 'export', '')).toBe('/data/export')
  })

  it('does not treat a pickle name as satisfying a .tsv request', () => {
    expect(composeSavePath('/data', 'clf.pkl', '.tsv')).toBe('/data/clf.pkl.tsv')
  })
})
