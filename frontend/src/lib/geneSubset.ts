/** Turning a gene-subset selection into what the association routes expect.
 *
 *  Two ways to narrow which genes get tested, and they are deliberately
 *  exclusive: boolean `.var` columns (highly_variable, expressed, …), or gene
 *  sets from the Gene Panel. The backend can combine columns with each other
 *  and can take an explicit gene list, but it cannot mix the two — and mixing
 *  them is the wrong default anyway. Intersecting a curated 40-gene signature
 *  with `highly_variable` quietly discards the genes the user came to test.
 */

import { applyOp } from './geneSetOps'

export type GeneSubsetOperation = 'intersection' | 'union'

/** What `gene_subset` accepts: a column name, an explicit gene list, several
 *  columns to combine, or nothing (meaning every gene). */
export type GeneSubsetSpec =
  | string
  | string[]
  | { columns: string[]; operation: GeneSubsetOperation }
  | null

export interface GeneSubsetSelection {
  columns: string[]
  geneSets: { name: string; genes: string[] }[]
  operation: GeneSubsetOperation
}

const nonEmpty = (sets: GeneSubsetSelection['geneSets']) => sets.filter((gs) => gs.genes.length > 0)

/** Fold the selected gene sets into one list under the chosen operation. */
function combineGeneSets(
  sets: GeneSubsetSelection['geneSets'],
  operation: GeneSubsetOperation,
): string[] {
  return sets.map((gs) => gs.genes).reduce((acc, genes) => applyOp(operation, acc, genes))
}

export function resolveGeneSubset(selection: GeneSubsetSelection): GeneSubsetSpec {
  const sets = nonEmpty(selection.geneSets)
  if (sets.length > 0) {
    // An empty result stays an empty list rather than collapsing to null:
    // null means "all genes", and falling back to the whole transcriptome
    // because an intersection came up empty is not what was asked for.
    return combineGeneSets(sets, selection.operation)
  }
  if (selection.columns.length === 1) return selection.columns[0]
  if (selection.columns.length > 1) {
    return { columns: selection.columns, operation: selection.operation }
  }
  return null
}

const OP_GLYPH: Record<GeneSubsetOperation, string> = { intersection: '∩', union: '∪' }
const OP_WORD: Record<GeneSubsetOperation, string> = { intersection: 'AND', union: 'OR' }

/** One-line description of the selection, for the picker header. */
export function geneSubsetLabel(selection: GeneSubsetSelection): string {
  const sets = nonEmpty(selection.geneSets)
  if (sets.length > 0) {
    const names = sets.map((gs) => gs.name).join(` ${OP_GLYPH[selection.operation]} `)
    const n = combineGeneSets(sets, selection.operation).length
    const size = n === 0 ? 'no genes' : `${n.toLocaleString()} gene${n === 1 ? '' : 's'}`
    return `${names} (${size})`
  }
  if (selection.columns.length === 1) return selection.columns[0]
  if (selection.columns.length > 1) {
    return selection.columns.join(` ${OP_WORD[selection.operation]} `)
  }
  return 'all genes'
}

/** Human-readable form of the `type` the backend reports back on a result:
 *  'all', 'gene_list', 'column:<name>', or '<operation>:<a>+<b>'. */
export function describeGeneSubsetType(type: string): string {
  if (type === 'all') return 'all genes'
  if (type === 'gene_list') return 'gene list'
  const sep = type.indexOf(':')
  if (sep === -1) return type
  const kind = type.slice(0, sep)
  const rest = type.slice(sep + 1)
  if (kind === 'column') return rest
  if (kind === 'intersection') return rest.split('+').join(' AND ')
  if (kind === 'union') return rest.split('+').join(' OR ')
  return rest
}
