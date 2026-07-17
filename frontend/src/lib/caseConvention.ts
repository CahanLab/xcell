/** Force gene symbols to a species' capitalization convention on import.
 *
 *  Most orthologous human and mouse genes share a symbol but differ only by
 *  case (human COL1A1 vs mouse Col1a1). Converting imported symbols to match
 *  the loaded data's convention lets a mouse gene set score human data (and
 *  vice-versa) through the existing exact-match paths, with no backend change.
 *
 *  This is a pure case heuristic: 'human' uppercases, 'mouse' title-cases
 *  (first char upper, rest lower). Orthologs whose symbols genuinely differ
 *  beyond case (or atypical symbols like mt-Nd1) are out of scope — that is
 *  the deferred HGNC HCOP ortholog-table work. */

export type CaseConvention = 'none' | 'human' | 'mouse'

export function applyCaseConvention(symbol: string, c: CaseConvention): string {
  if (c === 'human') return symbol.toUpperCase()
  if (c === 'mouse') {
    return symbol ? symbol[0].toUpperCase() + symbol.slice(1).toLowerCase() : symbol
  }
  return symbol
}

/** Apply a convention to every symbol in an up/down gene-list pair. Returns a
 *  new object; leaves `genesDown` undefined when it started undefined. A no-op
 *  for 'none'. */
export function convertGeneList<T extends { genes: string[]; genesDown?: string[] }>(
  list: T,
  c: CaseConvention,
): T {
  if (c === 'none') return list
  return {
    ...list,
    genes: list.genes.map((g) => applyCaseConvention(g, c)),
    genesDown: list.genesDown?.map((g) => applyCaseConvention(g, c)),
  }
}
