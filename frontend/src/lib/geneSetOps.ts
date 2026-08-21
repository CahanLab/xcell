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

/** Alphabetical order for gene names: case-insensitive and numeric-aware, so
 *  Hoxd2 precedes Hoxd10 rather than following it. A pure reorder — membership
 *  (duplicates included) is preserved, unlike the set operations above. */
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })

export function sortGenes(genes: string[]): string[] {
  return [...genes].sort(collator.compare)
}

/** Sort one gene set's genes in place within its category, wherever it lives.
 *
 * A gene set sits either directly on the category or inside one of its
 * folders, and the caller (a menu item) knows only its id. Untouched objects
 * keep their identity so React re-renders just the set that changed; an
 * unknown id returns the category itself, so a no-op never invalidates state.
 */
export function sortGeneSetInCategory<
  S extends { id: string; genes: string[]; genesDown?: string[] },
  F extends { geneSets: S[] },
  C extends { geneSets: S[]; folders: F[] },
>(category: C, geneSetId: string): C {
  const sortSet = (gs: S): S => ({
    ...gs,
    genes: sortGenes(gs.genes),
    ...(gs.genesDown ? { genesDown: sortGenes(gs.genesDown) } : {}),
  })

  if (category.geneSets.some((gs) => gs.id === geneSetId)) {
    return {
      ...category,
      geneSets: category.geneSets.map((gs) => (gs.id === geneSetId ? sortSet(gs) : gs)),
    }
  }
  if (category.folders.some((f) => f.geneSets.some((gs) => gs.id === geneSetId))) {
    return {
      ...category,
      folders: category.folders.map((f) =>
        f.geneSets.some((gs) => gs.id === geneSetId)
          ? { ...f, geneSets: f.geneSets.map((gs) => (gs.id === geneSetId ? sortSet(gs) : gs)) }
          : f,
      ),
    }
  }
  return category
}
