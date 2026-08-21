/** CSV serialisation of a neighborhood-enrichment result.
 *
 * The heatmap answers "what is next to what" at a glance; the numbers behind
 * it are what ends up in a figure legend or a stats table, so both shapes are
 * offered. The long form is one row per ordered pair with every statistic —
 * the shape R and pandas want. The matrix form is the picture as a table.
 *
 * Both take the display order, so an exported file matches the figure the user
 * is looking at rather than the input order they stopped looking at.
 */

export interface NeighborhoodMatrices {
  categories: string[]
  composition: number[][]
  expected: number[][]
  counts: number[][]
  zscores: number[][]
  log2fc: number[][]
  pvals: number[][]
  qvals: number[][]
}

/** RFC-4180 quoting: cell types like "Fibro, distal" would otherwise shift
 *  every column to their right. */
function csvCell(value: string | number): string {
  const s = String(value)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** Round for display without flattening small p/q values to 0: six
 *  significant figures, then trimmed of trailing zeros. */
function num(v: number): string {
  if (!Number.isFinite(v)) return ''
  if (v === 0) return '0'
  const s = Number(v.toPrecision(6)).toString()
  return s
}

export function toLongCsv(result: NeighborhoodMatrices, order: number[]): string {
  const cats = result.categories
  const lines = [
    'row_type,col_type,neighbor_fraction,expected_fraction,n_pairs,zscore,log2fc,pval,qval',
  ]
  for (const i of order) {
    for (const j of order) {
      lines.push([
        csvCell(cats[i]),
        csvCell(cats[j]),
        num(result.composition[i][j]),
        num(result.expected[i][j]),
        num(result.counts[i][j]),
        num(result.zscores[i][j]),
        num(result.log2fc[i][j]),
        num(result.pvals[i][j]),
        num(result.qvals[i][j]),
      ].join(','))
    }
  }
  return lines.join('\n') + '\n'
}

export function toMatrixCsv(
  values: number[][],
  categories: string[],
  order: number[],
): string {
  const lines = [['', ...order.map((j) => csvCell(categories[j]))].join(',')]
  for (const i of order) {
    lines.push([csvCell(categories[i]), ...order.map((j) => num(values[i][j]))].join(','))
  }
  return lines.join('\n') + '\n'
}
