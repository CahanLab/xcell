/** Set operations over gene-name arrays. Each returns a de-duplicated array
 *  with stable order = first appearance in A, then new members of B.
 *  Matching is exact-string (gene names already match the .var index). */

export type SetOp = 'union' | 'intersection' | 'difference' | 'symmetric'

const dedupe = (xs: string[]): string[] => Array.from(new Set(xs))

export function union(a: string[], b: string[]): string[] {
  return dedupe([...a, ...b])
}

export function intersection(a: string[], b: string[]): string[] {
  const sb = new Set(b)
  return dedupe(a).filter((g) => sb.has(g))
}

export function difference(a: string[], b: string[]): string[] {
  const sb = new Set(b)
  return dedupe(a).filter((g) => !sb.has(g))
}

export function symmetricDifference(a: string[], b: string[]): string[] {
  const sa = new Set(a)
  const sb = new Set(b)
  return [...dedupe(a).filter((g) => !sb.has(g)), ...dedupe(b).filter((g) => !sa.has(g))]
}

export function applyOp(op: SetOp, a: string[], b: string[]): string[] {
  switch (op) {
    case 'union': return union(a, b)
    case 'intersection': return intersection(a, b)
    case 'difference': return difference(a, b)
    case 'symmetric': return symmetricDifference(a, b)
  }
}

export const OP_SYMBOL: Record<SetOp, string> = {
  union: '∪', intersection: '∩', difference: '−', symmetric: '△',
}

export const OP_LABEL: Record<SetOp, string> = {
  union: 'Union (A ∪ B)',
  intersection: 'Intersection (A ∩ B)',
  difference: 'Difference (A − B)',
  symmetric: 'Symmetric difference (A △ B)',
}
