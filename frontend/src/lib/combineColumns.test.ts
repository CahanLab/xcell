import { describe, it, expect } from 'vitest'
import { summarizePolicy, seedPolicy, type ColumnInfo } from './combineColumns'

const cols: ColumnInfo[] = [
  { name: 'nCount_RNA', datasets: ['A', 'B'], kind: 'numeric', suggested: 'merge', reason: 'r' },
  { name: 'clusters', datasets: ['A', 'B'], kind: 'category', suggested: 'separate', reason: 'r' },
  { name: 'only_a', datasets: ['A'], kind: 'numeric', suggested: 'merge', reason: 'r' },
]

describe('seedPolicy', () => {
  it('starts every column on its suggestion', () => {
    expect(seedPolicy(cols)).toEqual({
      nCount_RNA: 'merge', clusters: 'separate', only_a: 'merge',
    })
  })

  it('keeps a choice the user already made', () => {
    expect(seedPolicy(cols, { clusters: 'drop' }).clusters).toBe('drop')
  })

  it('forgets choices for columns that are no longer offered', () => {
    // Removing a file changes the column list; a stale entry would be sent to
    // the backend, which rejects a policy naming a column no dataset has.
    expect(seedPolicy(cols, { gone: 'merge' })).not.toHaveProperty('gone')
  })
})

describe('summarizePolicy', () => {
  it('counts each outcome', () => {
    expect(summarizePolicy(cols, seedPolicy(cols))).toBe('2 merged · 1 kept separate')
  })

  it('mentions drops', () => {
    const p = { ...seedPolicy(cols), only_a: 'drop' }
    expect(summarizePolicy(cols, p)).toBe('1 merged · 1 kept separate · 1 dropped')
  })

  it('says so when nothing is kept', () => {
    const p = Object.fromEntries(cols.map((c) => [c.name, 'drop']))
    expect(summarizePolicy(cols, p)).toBe('3 dropped')
  })

  it('handles an empty column list', () => {
    expect(summarizePolicy([], {})).toBe('none')
  })
})
