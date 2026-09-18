/**
 * Presentation helpers for MyGene.info annotations (the ⓘ card in the Genes
 * panel). The record shape mirrors backend/xcell/gene_annotations.py.
 */

export interface GoTerm {
  id: string
  term: string
  evidence: string
}

export interface GeneAnnotation {
  symbol: string
  query: string
  notfound: boolean
  name?: string
  summary?: string
  summary_source?: string | null
  type?: string | null
  aliases?: string[]
  entrez?: number | null
  ensembl?: string | null
  mgi?: string | null
  hgnc?: string | null
  uniprot?: string | null
  taxid?: number | null
  go?: { BP?: GoTerm[]; CC?: GoTerm[]; MF?: GoTerm[] }
  interpro?: { id: string; name: string; short: string }[]
  pathways?: Record<string, { id: string; name: string }[]>
  homologene?: Record<string, number>
  orthologs?: Record<string, { symbol: string; entrez: number }>
  links?: Record<string, string>
  fetched_at?: string
}

/** Experimental > high-throughput > curated > phylogenetic/sequence > electronic. Mirrors the backend table. */
const EVIDENCE_RANK: Record<string, number> = {
  EXP: 5, IDA: 5, IPI: 5, IMP: 5, IGI: 5, IEP: 5,
  HTP: 4, HDA: 4, HMP: 4, HGI: 4, HEP: 4,
  TAS: 3, NAS: 3, IC: 3,
  IBA: 2, IBD: 2, IKR: 2, IRD: 2, ISS: 2, ISO: 2, ISA: 2, ISM: 2, IGC: 2, RCA: 2,
  IEA: 1, ND: 0,
}

export function evidenceRank(code: string): number {
  return EVIDENCE_RANK[code] ?? 0
}

/** Strongest evidence first; ties keep MyGene's order. */
export function topGoTerms(terms: GoTerm[] | undefined, n: number): GoTerm[] {
  if (!terms) return []
  return terms
    .map((t, i) => ({ t, i }))
    .sort((a, b) => evidenceRank(b.t.evidence) - evidenceRank(a.t.evidence) || a.i - b.i)
    .slice(0, n)
    .map((x) => x.t)
}

export function summaryLine(rec: GeneAnnotation): { text: string | null; note: string | null } {
  const text = rec.summary && rec.summary.trim() ? rec.summary.trim() : null
  if (!text) return { text: null, note: null }
  const src = rec.summary_source || ''
  if (src.startsWith('ortholog:')) {
    const who = src.slice('ortholog:'.length)
    const species = rec.taxid === 9606 ? 'mouse' : 'human'
    return { text, note: `No summary of its own; this is the ${species} ortholog ${who}'s.` }
  }
  return { text, note: null }
}

const LINK_ORDER: [string, string][] = [
  ['ncbi', 'NCBI'], ['ensembl', 'Ensembl'], ['mgi', 'MGI'], ['hgnc', 'HGNC'], ['uniprot', 'UniProt'],
]

export function linkList(rec: GeneAnnotation): { label: string; url: string }[] {
  const links = rec.links || {}
  return LINK_ORDER.filter(([k]) => links[k]).map(([k, label]) => ({ label, url: links[k] }))
}
