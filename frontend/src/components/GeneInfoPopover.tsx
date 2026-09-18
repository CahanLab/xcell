/**
 * GeneInfoPopover — the ⓘ card: what a gene is, from MyGene.info.
 *
 * Name, RefSeq summary (or the ortholog's, labelled), GO terms ordered by
 * evidence, InterPro domains, pathway membership, orthologs and links out.
 * Anchored beside the row whose ⓘ was clicked; the store holds the target
 * (`geneInfoTarget`; null = closed). Annotations are cached server-side, so
 * a second open is instant and the panel's "Fetch gene annotations for all
 * genes" makes every first open instant too.
 *
 * Rollback: delete this file, its mount and the GeneInfoButton in
 * GenePanel.tsx, `geneInfoTarget` in store.ts, and lib/geneAnnotation.ts.
 */
import React, { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import { linkList, summaryLine, topGoTerms, type GeneAnnotation, type GoTerm } from '../lib/geneAnnotation'

const API_BASE = '/api'
const WIDTH = 400

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625', accent: '#4ecdc4',
  alert: '#e94560', warn: '#e9a23b', muted: '#aaa', dim: '#888', text: '#ddd',
}

const GO_LABEL: Record<'BP' | 'CC' | 'MF', string> = { BP: 'Biological process', CC: 'Cellular component', MF: 'Molecular function' }

export default function GeneInfoPopover() {
  const target = useStore((s) => s.geneInfoTarget)
  const setTarget = useStore((s) => s.setGeneInfoTarget)
  const [rec, setRec] = useState<GeneAnnotation | null>(null)
  const [species, setSpecies] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [showAll, setShowAll] = useState<Record<string, boolean>>({})
  const cardRef = useRef<HTMLDivElement | null>(null)

  const load = React.useCallback(async (gene: string, refresh = false) => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(appendDataset(`${API_BASE}/gene_annotations/${encodeURIComponent(gene)}${refresh ? '?refresh=true' : ''}`))
      const body = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`)
      setRec(body.annotation)
      setSpecies(body.species)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!target) return
    setRec(null); setShowAll({})
    load(target.gene)
  }, [target, load])

  useEffect(() => {
    if (!target) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setTarget(null) }
    const onDown = (e: MouseEvent) => {
      if (cardRef.current && !cardRef.current.contains(e.target as Node)) setTarget(null)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('mousedown', onDown)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('mousedown', onDown) }
  }, [target, setTarget])

  if (!target) return null

  // Keep the card on screen: prefer to the left of the anchor (the Genes
  // panel hugs the right edge), fall back to the right, clamp vertically.
  const vw = window.innerWidth, vh = window.innerHeight
  const left = target.x - WIDTH - 8 >= 8 ? target.x - WIDTH - 8 : Math.min(target.x + 8, vw - WIDTH - 8)
  const top = Math.max(8, Math.min(target.y - 12, vh - 8 - Math.min(vh * 0.7, 560)))

  const summary = rec && !rec.notfound ? summaryLine(rec) : { text: null, note: null }
  const links = rec && !rec.notfound ? linkList(rec) : []

  return (
    <div
      ref={cardRef}
      onMouseDown={(e) => e.stopPropagation()}
      style={{
        position: 'fixed', left, top, width: WIDTH, maxHeight: '70vh', overflowY: 'auto', zIndex: 1500,
        background: dark.panel, border: `1px solid ${dark.accent}`, borderRadius: 8, padding: 12,
        color: dark.text, fontSize: 12, boxShadow: '0 8px 24px rgba(0,0,0,0.5)', lineHeight: 1.45,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: dark.accent }}>{target.gene}</span>
        {rec && !rec.notfound && rec.symbol !== target.gene && <span style={{ color: dark.dim }}>= {rec.symbol}</span>}
        <div style={{ flex: 1 }} />
        {rec && !rec.notfound && (
          <button onClick={() => load(target.gene, true)} title="Re-fetch from MyGene.info" style={{ background: 'transparent', border: 'none', color: dark.dim, cursor: 'pointer', fontSize: 12 }}>↻</button>
        )}
        <button onClick={() => setTarget(null)} style={{ background: 'transparent', border: 'none', color: dark.dim, cursor: 'pointer', fontSize: 13 }} title="Close (Esc)">✕</button>
      </div>

      {loading && !rec && <div style={{ color: dark.dim }}>Asking MyGene.info…</div>}
      {error && <div style={{ color: '#f5b8c4', background: '#3a1f2b', border: `1px solid ${dark.alert}`, borderRadius: 4, padding: '4px 8px' }}>{error}</div>}

      {rec && rec.notfound && (
        <div style={{ color: dark.muted }}>MyGene.info has no record named <b style={{ color: dark.text }}>{target.gene}</b>{species ? ` for ${species}` : ''}. Predicted or provisional genes, and identifiers other than symbols, often have none.</div>
      )}

      {rec && !rec.notfound && (
        <>
          <div style={{ color: dark.text, marginBottom: 2 }}>{rec.name}</div>
          <div style={{ color: dark.dim, fontSize: 11, marginBottom: 8 }}>
            {rec.type || 'gene'}{rec.aliases && rec.aliases.length > 0 ? ` · aka ${rec.aliases.slice(0, 5).join(', ')}` : ''}
            {rec.orthologs && Object.entries(rec.orthologs).map(([sp, o]) => <span key={sp}> · {sp} ortholog <b style={{ color: dark.muted }}>{o.symbol}</b></span>)}
          </div>

          {summary.text ? (
            <div style={{ marginBottom: 8 }}>
              <div>{summary.text}</div>
              {summary.note && <div style={{ color: dark.warn, fontSize: 11, marginTop: 2 }}>{summary.note}</div>}
            </div>
          ) : (
            <div style={{ color: dark.dim, fontSize: 11, marginBottom: 8 }}>No RefSeq summary.</div>
          )}

          {(['BP', 'CC', 'MF'] as const).map((cat) => {
            const all = rec.go?.[cat] ?? []
            if (all.length === 0) return null
            const open = !!showAll[cat]
            const shown = open ? topGoTerms(all, all.length) : topGoTerms(all, 5)
            return (
              <Section key={cat} title={`${GO_LABEL[cat]} (${all.length})`}>
                {shown.map((t: GoTerm) => (
                  <div key={t.id} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
                    <a href={`https://amigo.geneontology.org/amigo/term/${t.id}`} target="_blank" rel="noreferrer" style={{ color: dark.text, textDecoration: 'none' }} title={t.id}>{t.term}</a>
                    <span style={{ color: dark.dim, fontSize: 10 }} title="GO evidence code">{t.evidence}</span>
                  </div>
                ))}
                {all.length > 5 && (
                  <button onClick={() => setShowAll((s) => ({ ...s, [cat]: !open }))} style={{ background: 'transparent', border: 'none', color: dark.accent, cursor: 'pointer', fontSize: 11, padding: 0 }}>
                    {open ? 'show fewer' : `+${all.length - 5} more`}
                  </button>
                )}
              </Section>
            )
          })}

          {rec.interpro && rec.interpro.length > 0 && (
            <Section title={`Domains (${rec.interpro.length})`}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {rec.interpro.map((d) => (
                  <a key={d.id} href={`https://www.ebi.ac.uk/interpro/entry/InterPro/${d.id}/`} target="_blank" rel="noreferrer" title={`${d.id} ${d.short}`}
                    style={{ background: dark.border, color: '#cfe', borderRadius: 10, padding: '1px 8px', fontSize: 11, textDecoration: 'none' }}>{d.name}</a>
                ))}
              </div>
            </Section>
          )}

          {rec.pathways && Object.keys(rec.pathways).length > 0 && (
            <Section title="Pathways">
              {Object.entries(rec.pathways).map(([src, items]) => (
                <div key={src} style={{ marginBottom: 2 }}>
                  <span style={{ color: dark.dim, fontSize: 11 }}>{src}: </span>
                  <span style={{ color: dark.muted }}>{items.slice(0, 6).map((p) => p.name).join(' · ')}{items.length > 6 ? ` · +${items.length - 6}` : ''}</span>
                </div>
              ))}
            </Section>
          )}

          <div style={{ display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {links.map((l) => <a key={l.label} href={l.url} target="_blank" rel="noreferrer" style={{ color: dark.accent, fontSize: 11, textDecoration: 'none' }}>{l.label} ↗</a>)}
            <span style={{ flex: 1 }} />
            <span style={{ color: dark.dim, fontSize: 10 }}>MyGene.info{rec.fetched_at ? ` · ${rec.fetched_at.slice(0, 10)}` : ''}</span>
          </div>
        </>
      )}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: dark.dim, textTransform: 'uppercase', letterSpacing: 0.4, marginBottom: 2 }}>{title}</div>
      {children}
    </div>
  )
}
