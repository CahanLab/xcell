/**
 * Pure helpers for the Decompose gene set modal: how programs become gene
 * sets, how coherence is worded, how a program's bar is scaled.
 */

export interface Program {
  name: string
  index: number
  genes: string[]
  weights: number[]
  genes_down: string[]
  weights_down: number[]
  variance_ratio?: number
  factor_weight?: number
}

export interface Coherence {
  eigengene_pve: number
  suggested_k: number
  eigen_pve?: number[]
  mean_abs_corr?: number | null
  n_genes?: number
  n_cells?: number
  /** Marchenko–Pastur noise edge, as a fraction of the gene count (same units as eigen_pve). */
  noise_edge_pve?: number
  n_significant?: number
  signal_share_top?: number
  one_pattern?: boolean
}

/** Above this share of variance in one pattern, a single score is the honest summary. */
export const ONE_PATTERN_PVE = 0.6

export function programsToGeneSets(
  setName: string,
  _method: string,
  programs: Program[],
): { name: string; genes: string[]; genesDown?: string[] }[] {
  const out: { name: string; genes: string[]; genesDown?: string[] }[] = []
  for (const p of programs) {
    if (p.genes.length === 0 && p.genes_down.length === 0) continue
    const set: { name: string; genes: string[]; genesDown?: string[] } = { name: `${setName} ${p.name}`, genes: p.genes }
    if (p.genes_down.length > 0) set.genesDown = p.genes_down
    out.push(set)
  }
  return out
}

export function coherenceVerdict(c: Coherence): { kind: 'one' | 'several'; text: string } {
  const pct = Math.round(c.eigengene_pve * 100)
  const k = Math.max(1, c.suggested_k)
  // Backend verdict (noise-edge based) wins; the variance-share rule is the
  // fallback for a response that predates it.
  const one = c.one_pattern ?? c.eigengene_pve > ONE_PATTERN_PVE
  if (one) {
    const share = c.signal_share_top != null && c.n_significant != null && c.n_significant > 1
      ? ` and carries ${Math.round(c.signal_share_top * 100)}% of what stands above noise`
      : ''
    return {
      kind: 'one',
      text: `One pattern explains ${pct}% of this set's variance${share} — it behaves as a single program, and a mean or UCell score is probably all you need.`,
    }
  }
  const share = c.signal_share_top != null ? `; the strongest carries only ${Math.round(c.signal_share_top * 100)}% of that` : ''
  return {
    kind: 'several',
    text: `${k} pattern${k === 1 ? '' : 's'} stand${k === 1 ? 's' : ''} above the noise floor${share} (one pattern = ${pct}% of the set's variance). A single score would hide the rest — decompose it.`,
  }
}

/** 0–1 bar length: variance share (PCA) or factor weight (NMF), relative to the largest program. */
export function programWeight(p: Program, all: Program[]): number {
  const w = (q: Program) => q.variance_ratio ?? q.factor_weight ?? 1
  const max = Math.max(...all.map(w), 1e-12)
  return w(p) / max
}
