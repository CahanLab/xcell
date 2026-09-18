import { describe, it, expect } from 'vitest'
import { evidenceRank, topGoTerms, summaryLine, linkList, type GeneAnnotation } from './geneAnnotation'

const rec: GeneAnnotation = {
  symbol: 'Col1a1', query: 'Col1a1', notfound: false, name: 'collagen, type I, alpha 1',
  summary: 'This gene encodes the alpha-1 subunit of type I collagen.', summary_source: 'refseq',
  type: 'protein-coding', aliases: ['Cola1'], entrez: 12842, ensembl: 'ENSMUSG00000001506', mgi: 'MGI:88467', hgnc: null, uniprot: 'P11087', taxid: 10090,
  go: {
    BP: [{ id: 'GO:1', term: 'skeletal system development', evidence: 'IEA' }, { id: 'GO:2', term: 'collagen fibril organization', evidence: 'IDA' }, { id: 'GO:3', term: 'ossification', evidence: 'IMP' }],
    CC: [{ id: 'GO:4', term: 'extracellular region', evidence: 'HDA' }], MF: [],
  },
  interpro: [{ id: 'IPR1', name: 'Fibrillar collagen, C-terminal', short: 'Fib_collagen_C' }],
  pathways: { reactome: [{ id: 'R-1', name: 'Extracellular matrix organization' }] },
  homologene: { '9606': 1277 }, orthologs: { human: { symbol: 'COL1A1', entrez: 1277 } },
  links: { ncbi: 'https://ncbi/12842', mgi: 'https://mgi/MGI:88467', uniprot: 'https://uniprot/P11087' },
}

describe('evidenceRank / topGoTerms', () => {
  it('ranks experimental evidence above electronic and orders terms by it, stably', () => {
    expect(evidenceRank('IDA')).toBeGreaterThan(evidenceRank('IEA'))
    expect(evidenceRank('???')).toBe(0)
    const top = topGoTerms(rec.go!.BP, 2)
    expect(top.map((t) => t.id)).toEqual(['GO:2', 'GO:3'])
    expect(topGoTerms(rec.go!.BP, 10).length).toBe(3)
  })
})

describe('summaryLine', () => {
  it('names the source of a borrowed summary and handles a gene without one', () => {
    expect(summaryLine(rec)).toEqual({ text: 'This gene encodes the alpha-1 subunit of type I collagen.', note: null })
    const borrowed = summaryLine({ ...rec, summary: 'Human text.', summary_source: 'ortholog:COL1A1' })
    expect(borrowed.note).toMatch(/human ortholog COL1A1/)
    expect(summaryLine({ ...rec, summary: '', summary_source: null }).text).toBeNull()
  })
})

describe('linkList', () => {
  it('lists links in a fixed order with readable labels', () => {
    expect(linkList(rec)).toEqual([
      { label: 'NCBI', url: 'https://ncbi/12842' },
      { label: 'MGI', url: 'https://mgi/MGI:88467' },
      { label: 'UniProt', url: 'https://uniprot/P11087' },
    ])
  })
})
