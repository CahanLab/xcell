import { Fragment, useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useStore, cfgDefault, GeneSet } from '../store'
import {
  useObsSummaries, appendDataset, pollTask,
  runOverlapEnrichment, startGsea, startGseaBatch, runOraBatch,
  fetchEnrichmentResults, fetchEnrichmentResult, deleteEnrichmentResult, fetchCachedLibraries,
} from '../hooks/useData'
import {
  formatP, curvePath, hitTicks, filterRows, resultsToGeneSets, libraryGroups, rowsToTsv, metricStripBins, reconcileChoice,
  isBatch, batchGroupsSorted, batchToGeneSets,
  type EnrichmentResult, type AnyEnrichmentResult, type BatchCollection, type EnrichmentSummary, type CachedLibrary,
  type GseaResult, type GseaRow, type OraRow,
} from '../lib/enrichment'
import { standaloneSvg, downloadText } from '../lib/svgExport'
import { flattenGeneSets } from './GenePanel'

/** Overlap (hypergeometric) enrichment and preranked GSEA against the
 *  gene-set libraries already in the source cache, plus the user's own sets.
 *  Single contrasts and whole-column batches (one result per group plus a
 *  collection). Global modal: mounted always, gated on `enrichmentSource`. */

const ACCENT = '#4ecdc4'
const ALERT = '#e94560'
const ALL_GROUPS = '__all__'
const MARKERS = '__markers__'
const PRESET = '__preset__'

const styles = {
  backdrop: {
    position: 'fixed' as const, top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: '8px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)', width: '920px', maxWidth: '94vw', maxHeight: '88vh',
    display: 'flex', flexDirection: 'column' as const, overflow: 'hidden',
  },
  header: {
    padding: '14px 20px', borderBottom: '1px solid #0f3460',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  },
  title: { fontSize: '16px', fontWeight: 600, color: ALERT, margin: 0 },
  closeButton: { background: 'none', border: 'none', color: '#aaa', fontSize: '20px', cursor: 'pointer', padding: '4px 8px', lineHeight: 1 },
  body: { flex: 1, overflowY: 'auto' as const, padding: '14px 20px' },
  tabs: { display: 'flex', gap: '4px', marginBottom: '14px', borderBottom: '1px solid #0f3460' },
  tab: (active: boolean) => ({
    padding: '6px 14px', fontSize: '13px', cursor: 'pointer', background: 'none',
    borderTop: 'none', borderLeft: 'none', borderRight: 'none',
    color: active ? ACCENT : '#888', borderBottom: active ? `2px solid ${ACCENT}` : '2px solid transparent',
  }),
  label: { fontSize: '12px', color: '#aaa', marginBottom: '6px', display: 'block', fontWeight: 500 },
  section: { backgroundColor: '#0f1625', borderRadius: '4px', padding: '10px 12px', marginBottom: '12px' },
  row: { display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' as const },
  input: {
    padding: '5px 8px', fontSize: '13px', backgroundColor: '#0f3460', color: '#eee',
    border: '1px solid #1a1a2e', borderRadius: '4px', outline: 'none', width: '80px',
  },
  select: {
    padding: '5px 8px', fontSize: '13px', backgroundColor: '#0f3460', color: '#eee',
    border: '1px solid #1a1a2e', borderRadius: '4px', outline: 'none', maxWidth: '260px',
  },
  textarea: {
    width: '100%', minHeight: '70px', padding: '6px 8px', fontSize: '12px', fontFamily: 'monospace',
    backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px',
    outline: 'none', boxSizing: 'border-box' as const,
  },
  checkRow: { display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#ccc', padding: '2px 0', cursor: 'pointer' },
  muted: { fontSize: '11px', color: '#888' },
  footer: {
    padding: '12px 20px', borderTop: '1px solid #0f3460',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px',
  },
  button: { padding: '8px 16px', fontSize: '13px', borderRadius: '4px', cursor: 'pointer', border: 'none' },
  primaryButton: { backgroundColor: ALERT, color: '#fff' },
  secondaryButton: { backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e' },
  successButton: { backgroundColor: ACCENT, color: '#000' },
  disabledButton: { opacity: 0.5, cursor: 'not-allowed' },
  error: {
    fontSize: '12px', color: ALERT, padding: '8px 12px',
    backgroundColor: 'rgba(233, 69, 96, 0.1)', borderRadius: '4px', marginBottom: '12px',
  },
  table: { width: '100%', borderCollapse: 'collapse' as const, fontSize: '12px' },
  th: {
    position: 'sticky' as const, top: 0, backgroundColor: '#16213e', color: '#888', fontWeight: 500,
    textAlign: 'left' as const, padding: '6px 8px', borderBottom: '1px solid #0f3460', whiteSpace: 'nowrap' as const,
  },
  td: { padding: '5px 8px', borderBottom: '1px solid #0f1625', color: '#ccc', verticalAlign: 'top' as const },
  chip: {
    display: 'inline-block', padding: '1px 6px', margin: '2px', fontSize: '11px', borderRadius: '3px',
    backgroundColor: '#0f3460', color: '#eee', cursor: 'pointer',
  },
  groupChip: (active: boolean) => ({
    display: 'inline-block', padding: '3px 10px', margin: '2px', fontSize: '12px', borderRadius: '12px', cursor: 'pointer',
    backgroundColor: active ? ACCENT : '#0f3460', color: active ? '#000' : '#eee',
  }),
  progressTrack: { height: '6px', backgroundColor: '#0f1625', borderRadius: '3px', overflow: 'hidden', marginTop: '8px' },
  progressBar: (frac: number) => ({ height: '100%', width: `${Math.round(frac * 100)}%`, backgroundColor: ACCENT, transition: 'width 0.3s' }),
}

type Tab = 'ora' | 'gsea'
type Phase = 'config' | 'running' | 'results'
interface BoolColumn { name: string; n_true: number; n_total: number }

function libKey(l: { source: string; id: string }) {
  return `${l.source}/${l.id}`
}

function optionalNumber(text: string): number | null {
  const v = Number(text)
  return text.trim() === '' || !Number.isFinite(v) ? null : v
}

function GseaCurve({ row, ranking, svgRef }: {
  row: GseaRow
  ranking: GseaResult['ranking']
  svgRef: React.RefObject<SVGSVGElement>
}) {
  const W = 560, H = 140, STRIP = 14
  const curve = row.curve!
  const { d, zeroY } = curvePath(curve, W, H, ranking.n_ranked)
  const ticks = hitTicks(curve)
  const bins = metricStripBins(ranking.scores, 100)
  const bmax = Math.max(1e-9, ...bins.map(Math.abs))
  const sx = (x: number) => (ranking.n_ranked > 1 ? (x / (ranking.n_ranked - 1)) * W : 0)
  return (
    <svg ref={svgRef} data-gsea-curve width={W} height={H + STRIP + 30} style={{ display: 'block', marginTop: '8px' }}>
      <line x1={0} x2={W} y1={zeroY} y2={zeroY} stroke="#0f3460" />
      <path d={d} fill="none" stroke={ACCENT} strokeWidth={1.5} />
      {ticks.map((x, i) => (
        <line key={i} x1={sx(x)} x2={sx(x)} y1={H + 1} y2={H + 9} stroke="#eee" strokeWidth={0.8} />
      ))}
      {bins.map((v, i) => (
        <rect
          key={i} x={(i / bins.length) * W} y={H + 12} width={W / bins.length + 0.5} height={STRIP}
          fill={v >= 0 ? `rgba(233,69,96,${Math.abs(v) / bmax})` : `rgba(78,205,196,${Math.abs(v) / bmax})`}
        />
      ))}
      <text x={4} y={12} fill="#aaa" fontSize={10}>
        ES {row.es.toFixed(2)} · NES {row.nes.toFixed(2)} · padj {formatP(row.padj)} · leading edge {row.n_leading_edge}/{row.n_set}
      </text>
      <text x={0} y={H + STRIP + 26} fill="#888" fontSize={10}>rank 1 (highest score)</text>
      <text x={W} y={H + STRIP + 26} fill="#888" fontSize={10} textAnchor="end">rank {ranking.n_ranked}</text>
    </svg>
  )
}

export default function EnrichmentModal() {
  const source = useStore((s) => s.enrichmentSource)
  const setSource = useStore((s) => s.setEnrichmentSource)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const cellSubsets = useStore((s) => s.cellSubsets)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const setSelectedGenes = useStore((s) => s.setSelectedGenes)
  const { summaries } = useObsSummaries()

  const [tab, setTab] = useState<Tab>('ora')
  // Gene sets to test
  const [cached, setCached] = useState<CachedLibrary[]>([])
  const [chosenLibs, setChosenLibs] = useState<Set<string>>(new Set())
  const [ownFolder, setOwnFolder] = useState('')
  const [booleanColumns, setBooleanColumns] = useState<BoolColumn[]>([])
  const [universeCol, setUniverseCol] = useState('')
  // ORA
  const [querySetId, setQuerySetId] = useState('')
  const [pasted, setPasted] = useState('')
  const [minOverlap, setMinOverlap] = useState(() => cfgDefault(['enrichment', 'min_overlap'], 2))
  const [oraMin, setOraMin] = useState(() => cfgDefault(['enrichment', 'min_set_size'], 5))
  const [oraMax, setOraMax] = useState(() => cfgDefault(['enrichment', 'max_set_size'], 500))
  // ORA on the markers of every group
  const [markersColumn, setMarkersColumn] = useState('')
  const [markerTopN, setMarkerTopN] = useState(() => cfgDefault(['marker_genes', 'top_n'], 100))
  const [markerMinIn, setMarkerMinIn] = useState(() => String(cfgDefault(['marker_genes', 'min_in_group_fraction'], '')))
  const [markerMaxOut, setMarkerMaxOut] = useState(() => String(cfgDefault(['marker_genes', 'max_out_group_fraction'], '')))
  const [markerMinFc, setMarkerMinFc] = useState(() => String(cfgDefault(['marker_genes', 'min_fold_change'], '')))
  // GSEA
  const [rankKind, setRankKind] = useState<'diffexp' | 'pca'>('diffexp')
  const [obsColumn, setObsColumn] = useState('')
  const [group, setGroup] = useState('')
  const [reference, setReference] = useState('rest')
  const [method, setMethod] = useState<'wilcoxon' | 't-test'>('wilcoxon')
  const [metric, setMetric] = useState<'score' | 'log2fc'>('score')
  const [cellSubset, setCellSubset] = useState('')
  const [component, setComponent] = useState(1)
  const [nPerm, setNPerm] = useState(() => cfgDefault(['enrichment', 'n_perm'], 1000))
  const [gMin, setGMin] = useState(() => cfgDefault(['enrichment', 'gsea_min_set_size'], 15))
  const [gMax, setGMax] = useState(() => cfgDefault(['enrichment', 'gsea_max_set_size'], 500))
  const [weight, setWeight] = useState(() => cfgDefault(['enrichment', 'weight'], 1))
  const [seed, setSeed] = useState(0)
  // Run / results
  const [phase, setPhase] = useState<Phase>('config')
  const [progress, setProgress] = useState({ frac: 0, message: '' })
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnyEnrichmentResult | null>(null)
  const [selectedGroup, setSelectedGroup] = useState('')
  const [previous, setPrevious] = useState<EnrichmentSummary[]>([])
  const [padjMax, setPadjMax] = useState<number | null>(() => cfgDefault(['enrichment', 'padj_cutoff'], 0.05))
  const [padjText, setPadjText] = useState(() => String(cfgDefault(['enrichment', 'padj_cutoff'], 0.05)))
  const [query, setQuery] = useState('')
  const [hideBelow, setHideBelow] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [copied, setCopied] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)
  // The set the modal was opened from, if any: its genes are the query
  // without a lookup by name (names are not unique across categories).
  const presetRef = useRef<{ name: string; genes: string[] } | null>(null)
  // Groups a batch run is limited to (from Compare Cells); null = every group.
  const batchGroupsRef = useRef<string[] | null>(null)

  const flatSets = useMemo(
    () => flattenGeneSets(geneSetCategories).filter((g) => g.genes.length > 0),
    [geneSetCategories],
  )
  const folders = useMemo(() => {
    const out: { key: string; label: string; sets: GeneSet[] }[] = []
    for (const cat of Object.values(geneSetCategories)) {
      if (cat.geneSets.length > 0) out.push({ key: `${cat.type}/`, label: `${cat.name} (${cat.geneSets.length})`, sets: cat.geneSets })
      for (const f of cat.folders) {
        if (f.geneSets.length > 0) out.push({ key: `${cat.type}/${f.id}`, label: `${cat.name} › ${f.name} (${f.geneSets.length})`, sets: f.geneSets })
      }
    }
    return out
  }, [geneSetCategories])
  const categoricalColumns = useMemo(
    () => summaries.filter((s) => s.dtype === 'category' && (s.categories?.length ?? 0) >= 2),
    [summaries],
  )
  const groupOptions = useMemo(
    () => categoricalColumns.find((c) => c.name === obsColumn)?.categories ?? [],
    [categoricalColumns, obsColumn],
  )
  const pcaKey = cellSubset ? `X_pca_${cellSubset}` : 'X_pca'
  const nPcs = schema?.embedding_dims?.[pcaKey] ?? (schema?.embeddings.includes(pcaKey) ? 50 : 0)

  // Open: reset, pick the tab, remember the preset set, fetch what the pickers need.
  useEffect(() => {
    if (!source) return
    setTab(source.kind)
    presetRef.current = source.kind === 'ora' && source.genes ? { name: source.name ?? 'gene set', genes: source.genes } : null
    batchGroupsRef.current = null
    if (source.kind === 'ora') {
      if (source.markersOf) {
        setQuerySetId(MARKERS)
        setMarkersColumn(source.markersOf.obsColumn)
        batchGroupsRef.current = source.markersOf.groups ?? null
      } else {
        setQuerySetId(presetRef.current ? PRESET : '')
      }
    } else {
      setRankKind('diffexp')
      if (source.obsColumn) setObsColumn(source.obsColumn)
      if (source.obsColumn && source.reference && source.groups && source.groups.length === 1) {
        setGroup(source.groups[0]); setReference(source.reference)
      } else if (source.obsColumn) {
        setGroup(ALL_GROUPS); setReference('rest')
        batchGroupsRef.current = source.groups ?? null
      }
    }
    setPhase('config'); setResult(null); setError(null); setSaved(false); setCopied(false); setExpanded(null)
    fetchCachedLibraries().then(setCached).catch(() => setCached([]))
    fetch(appendDataset('/api/var/boolean_columns', activeSlot))
      .then((r) => r.json())
      .then((cols: BoolColumn[]) => setBooleanColumns(Array.isArray(cols) ? cols : []))
      .catch(() => setBooleanColumns([]))
    fetchEnrichmentResults(activeSlot).then(setPrevious).catch(() => setPrevious([]))
  }, [source, activeSlot])

  // Pickers hold names, and the dataset behind them can change under an
  // always-mounted modal (slot switch, deleted subset or column): drop any
  // value the current options no longer offer.
  useEffect(() => {
    const names = categoricalColumns.map((c) => c.name)
    const pick = categoricalColumns.find((c) => (c.categories?.length ?? 0) <= 200) ?? categoricalColumns[0]
    setObsColumn((v) => reconcileChoice(v, names, pick?.name ?? '') || (pick?.name ?? ''))
    setMarkersColumn((v) => reconcileChoice(v, names, pick?.name ?? '') || (pick?.name ?? ''))
  }, [categoricalColumns])
  useEffect(() => {
    setCellSubset((v) => reconcileChoice(v, cellSubsets.map((s) => s.name), ''))
  }, [cellSubsets])
  useEffect(() => {
    setUniverseCol((v) => reconcileChoice(v, booleanColumns.map((c) => c.name), ''))
  }, [booleanColumns])

  useEffect(() => {
    if (groupOptions.length > 0 && group !== ALL_GROUPS && !groupOptions.some((g) => g.value === group)) setGroup(groupOptions[0].value)
    if (reference !== 'rest' && !groupOptions.some((g) => g.value === reference)) setReference('rest')
  }, [groupOptions, group, reference])

  const handleClose = useCallback(() => setSource(null), [setSource])
  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') handleClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [source, handleClose])

  const libraries = useMemo(
    () => [...chosenLibs].map((k) => {
      const lib = cached.find((c) => libKey(c) === k)
      return { source: lib?.source ?? k.split('/')[0], id: lib?.id ?? k.split('/').slice(1).join('/'), species: lib?.species ?? null }
    }),
    [chosenLibs, cached],
  )
  const inlineSets = useMemo(() => {
    const f = folders.find((x) => x.key === ownFolder)
    return f ? f.sets.map((gs) => ({ name: gs.name, genes: gs.genes, genes_down: gs.genesDown ?? null })) : []
  }, [folders, ownFolder])
  const nothingToTest = libraries.length === 0 && inlineSets.length === 0

  const queryGenes = useMemo(() => {
    if (querySetId === PRESET) return presetRef.current?.genes ?? []
    if (querySetId === MARKERS) return []
    if (querySetId) return flatSets.find((g) => g.id === querySetId)?.genes ?? []
    return pasted.split(/[\s,;]+/).map((g) => g.trim()).filter(Boolean)
  }, [querySetId, flatSets, pasted])
  const queryName = querySetId === PRESET
    ? presetRef.current?.name ?? 'gene set'
    : querySetId ? flatSets.find((g) => g.id === querySetId)?.name ?? 'gene set' : 'pasted list'

  const finishRun = useCallback(async (res: AnyEnrichmentResult, action: string, body: Record<string, unknown>) => {
    setResult(res); setPhase('results'); setSaved(false); setCopied(false); setExpanded(null)
    setSelectedGroup(isBatch(res) ? batchGroupsSorted(res)[0] ?? '' : '')
    addScanpyAction({
      action, params: body,
      result: { key: res.key, n_sets_tested: res.n_sets_tested, n_significant: res.n_significant },
      timestamp: new Date().toISOString(),
    })
    try { setPrevious(await fetchEnrichmentResults(activeSlot)) } catch { /* the list is a convenience */ }
  }, [addScanpyAction, activeSlot])

  const runOra = useCallback(async () => {
    setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Testing overlaps…' })
    try {
      if (querySetId === MARKERS) {
        const body = {
          obs_column: markersColumn, groups: batchGroupsRef.current, top_n: markerTopN,
          min_in_group_fraction: optionalNumber(markerMinIn), max_out_group_fraction: optionalNumber(markerMaxOut),
          min_fold_change: optionalNumber(markerMinFc),
          libraries, sets: inlineSets, gene_subset: universeCol || null,
          min_set_size: oraMin, max_set_size: oraMax, min_overlap: minOverlap,
        }
        setProgress({ frac: 0, message: 'Marker genes, then overlaps for every group…' })
        const res = await runOraBatch(body, activeSlot)
        await finishRun(res, 'enrichment_ora_batch', body)
        return
      }
      const body = {
        genes: queryGenes, name: queryName, libraries, sets: inlineSets, gene_subset: universeCol || null,
        min_set_size: oraMin, max_set_size: oraMax, min_overlap: minOverlap,
      }
      const res = await runOverlapEnrichment(body, activeSlot)
      await finishRun(res, 'enrichment_ora', body)
    } catch (e) {
      setError((e as Error).message); setPhase('config')
    }
  }, [querySetId, markersColumn, markerTopN, markerMinIn, markerMaxOut, markerMinFc, queryGenes, queryName,
    libraries, inlineSets, universeCol, oraMin, oraMax, minOverlap, activeSlot, finishRun])

  const runGsea = useCallback(async () => {
    setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Starting…' })
    try {
      const onProgress = (s: { progress?: number; message?: string }) => setProgress({ frac: s.progress ?? 0, message: s.message ?? '' })
      if (rankKind === 'diffexp' && group === ALL_GROUPS) {
        const body = {
          obs_column: obsColumn, groups: batchGroupsRef.current, reference: 'rest', method, metric,
          cell_subset: cellSubset || null, libraries, sets: inlineSets, gene_subset: universeCol || null,
          n_perm: nPerm, min_set_size: gMin, max_set_size: gMax, weight, seed,
        }
        const { task_id } = await startGseaBatch(body, activeSlot)
        const task = await pollTask(task_id, activeSlot, onProgress)
        if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
        await finishRun(task.result as unknown as BatchCollection, 'enrichment_gsea_batch', body)
        return
      }
      const ranking = rankKind === 'diffexp'
        ? { kind: 'diffexp', obs_column: obsColumn, group, reference, method, metric, cell_subset: cellSubset || null }
        : { kind: 'pca', component: component - 1, cell_subset: cellSubset || null }
      const body = {
        ranking, libraries, sets: inlineSets, gene_subset: universeCol || null,
        n_perm: nPerm, min_set_size: gMin, max_set_size: gMax, weight, seed,
      }
      const { task_id } = await startGsea(body, activeSlot)
      const task = await pollTask(task_id, activeSlot, onProgress)
      if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
      await finishRun(task.result as unknown as GseaResult, 'enrichment_gsea', body)
    } catch (e) {
      setError((e as Error).message); setPhase('config')
    }
  }, [rankKind, obsColumn, group, reference, method, metric, cellSubset, component, libraries, inlineSets,
    universeCol, nPerm, gMin, gMax, weight, seed, activeSlot, finishRun])

  const openPrevious = useCallback(async (key: string) => {
    if (!key) return
    setError(null)
    try {
      const res = await fetchEnrichmentResult(key, activeSlot)
      setResult(res); setPhase('results'); setSaved(false); setCopied(false); setExpanded(null)
      setSelectedGroup(isBatch(res) ? batchGroupsSorted(res)[0] ?? '' : '')
      setTab(res.kind.startsWith('gsea') ? 'gsea' : 'ora')
    } catch (e) {
      setError((e as Error).message)
    }
  }, [activeSlot])

  const removePrevious = useCallback(async (key: string) => {
    try {
      await deleteEnrichmentResult(key, activeSlot)
      setPrevious((p) => p.filter((s) => s.key !== key))
      if (result?.key === key) { setResult(null); setPhase('config') }
    } catch (e) {
      setError((e as Error).message)
    }
  }, [activeSlot, result])

  // The table always shows one single-contrast result: the run itself, or
  // the selected member of a batch.
  const activeResult: EnrichmentResult | null = useMemo(() => {
    if (!result) return null
    if (isBatch(result)) return result.member_results?.[selectedGroup] ?? null
    return result
  }, [result, selectedGroup])

  const visibleRows = useMemo(() => {
    if (!activeResult) return []
    return activeResult.kind === 'ora'
      ? filterRows(activeResult.results, { padjMax, query, hideBelowMinOverlap: hideBelow })
      : filterRows(activeResult.results, { padjMax, query })
  }, [activeResult, padjMax, query, hideBelow])

  const saveSets = useCallback(() => {
    if (!result) return
    if (isBatch(result)) {
      const folders = batchToGeneSets(result, { padjMax, topN: 50 })
      if (folders.length === 0) return
      for (const f of folders) addFolderToCategory('enrichment', f.folder, f.sets)
    } else {
      const sets = resultsToGeneSets(result, { padjMax, topN: 50 })
      if (sets.length === 0) return
      addFolderToCategory('enrichment', result.key, sets)
    }
    setSaved(true)
  }, [result, padjMax, addFolderToCategory])

  const copyTsv = useCallback(() => {
    if (!activeResult) return
    navigator.clipboard?.writeText(rowsToTsv(activeResult)).then(() => setCopied(true)).catch(() => setCopied(false))
  }, [activeResult])

  const downloadSvg = useCallback(() => {
    if (!svgRef.current || !activeResult || !expanded) return
    const safe = expanded.split('\u0000').pop()!.replace(/[^A-Za-z0-9_-]+/g, '_')
    downloadText(`${activeResult.key}_${safe}.svg`, standaloneSvg(svgRef.current.outerHTML, { background: '#16213e' }), 'image/svg+xml')
  }, [activeResult, expanded])

  // Global modals never unmount: every hook above runs whether or not we are open.
  if (!source) return null

  const groups = libraryGroups(cached)
  const canRunOra = !nothingToTest && (querySetId === MARKERS ? Boolean(markersColumn) : queryGenes.length >= 2)
  const canRunGsea = !nothingToTest && (rankKind === 'diffexp' ? Boolean(obsColumn && group) : nPcs > 0)
  const pFloor = activeResult?.kind === 'gsea' ? 1 / (1 + Math.floor(activeResult.n_perm / 2)) : undefined
  const nSetsToAdd = result
    ? (isBatch(result)
      ? batchToGeneSets(result, { padjMax, topN: 50 }).reduce((n, f) => n + f.sets.length, 0)
      : resultsToGeneSets(result, { padjMax, topN: 50 }).length)
    : 0
  const batchNote = batchGroupsRef.current
    ? `limited to ${batchGroupsRef.current.length} group${batchGroupsRef.current.length === 1 ? '' : 's'}: ${batchGroupsRef.current.join(', ')}`
    : 'every group'

  const setsBlock = (
    <div style={styles.section}>
      <span style={styles.label}>Gene sets to test</span>
      {groups.length === 0 ? (
        <div style={styles.muted}>No libraries cached yet. Fetch one from Genes ▸ Gene set library… (MSigDB, GO, Enrichr, OmniPath, MGI).</div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 24px' }}>
          {groups.map((g) => (
            <div key={g.source} style={{ minWidth: '220px' }}>
              <div style={{ ...styles.muted, textTransform: 'uppercase', marginTop: '4px' }}>{g.source}</div>
              {g.libraries.map((lib) => {
                const k = libKey(lib)
                return (
                  <label key={k} style={styles.checkRow}>
                    <input
                      type="checkbox" checked={chosenLibs.has(k)}
                      onChange={() => setChosenLibs((prev) => { const n = new Set(prev); if (n.has(k)) n.delete(k); else n.add(k); return n })}
                    />
                    <span>{lib.name}</span>
                    <span style={styles.muted}>{lib.n_sets} sets · {lib.species}</span>
                  </label>
                )
              })}
            </div>
          ))}
        </div>
      )}
      <div style={{ ...styles.row, marginTop: '10px', marginBottom: 0 }}>
        <span style={{ fontSize: '12px', color: '#ccc' }}>My gene sets:</span>
        <select style={styles.select} value={ownFolder} onChange={(e) => setOwnFolder(e.target.value)}>
          <option value="">none</option>
          {folders.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
        </select>
        <span style={{ fontSize: '12px', color: '#ccc', marginLeft: '12px' }}>Universe:</span>
        <select style={styles.select} value={universeCol} onChange={(e) => setUniverseCol(e.target.value)}>
          <option value="">all visible genes</option>
          {booleanColumns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.n_true.toLocaleString()})</option>)}
        </select>
      </div>
    </div>
  )

  const oraConfig = (
    <>
      <div style={styles.section}>
        <span style={styles.label}>Query gene list</span>
        <div style={styles.row}>
          <select style={styles.select} value={querySetId} onChange={(e) => setQuerySetId(e.target.value)}>
            {presetRef.current && <option value={PRESET}>{presetRef.current.name} ({presetRef.current.genes.length})</option>}
            <option value="">Paste genes…</option>
            <option value={MARKERS}>Marker genes of each group in a column…</option>
            {flatSets.map((g) => <option key={g.id} value={g.id}>{g.name} ({g.genes.length})</option>)}
          </select>
          {querySetId !== MARKERS && <span style={styles.muted}>{queryGenes.length} genes</span>}
        </div>
        {querySetId === '' && (
          <textarea style={styles.textarea} value={pasted} onChange={(e) => setPasted(e.target.value)} placeholder="One symbol per line, or comma / space separated" />
        )}
        {querySetId === MARKERS && (
          <div style={{ ...styles.row, marginBottom: 0 }}>
            <select style={styles.select} value={markersColumn} onChange={(e) => { setMarkersColumn(e.target.value); batchGroupsRef.current = null }}>
              {categoricalColumns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.categories?.length ?? 0} groups)</option>)}
            </select>
            <span style={styles.muted}>{batchNote}</span>
            <span style={{ fontSize: '12px', color: '#ccc' }}>Top N markers</span>
            <input style={styles.input} type="number" min={2} value={markerTopN} onChange={(e) => setMarkerTopN(Math.max(2, Math.round(Number(e.target.value)) || 2))} />
            <span style={{ fontSize: '12px', color: '#ccc' }}>Min in-group frac.</span>
            <input style={{ ...styles.input, width: '60px' }} value={markerMinIn} onChange={(e) => setMarkerMinIn(e.target.value)} placeholder="—" />
            <span style={{ fontSize: '12px', color: '#ccc' }}>Max out-group frac.</span>
            <input style={{ ...styles.input, width: '60px' }} value={markerMaxOut} onChange={(e) => setMarkerMaxOut(e.target.value)} placeholder="—" />
            <span style={{ fontSize: '12px', color: '#ccc' }}>Min fold change</span>
            <input style={{ ...styles.input, width: '60px' }} value={markerMinFc} onChange={(e) => setMarkerMinFc(e.target.value)} placeholder="—" />
          </div>
        )}
      </div>
      {setsBlock}
      <div style={styles.row}>
        <span style={{ fontSize: '12px', color: '#ccc' }}>Min overlap</span>
        <input style={styles.input} type="number" min={1} value={minOverlap} onChange={(e) => setMinOverlap(Math.max(1, Number(e.target.value) || 1))} />
        <span style={{ fontSize: '12px', color: '#ccc' }}>Set size</span>
        <input style={styles.input} type="number" min={1} value={oraMin} onChange={(e) => setOraMin(Math.max(1, Number(e.target.value) || 1))} />
        <span style={styles.muted}>to</span>
        <input style={styles.input} type="number" min={1} value={oraMax} onChange={(e) => setOraMax(Math.max(1, Number(e.target.value) || 1))} />
      </div>
      <div style={styles.muted}>
        Hypergeometric test of each set's overlap with the query, Benjamini–Hochberg across sets. The universe is the dataset's genes (after the gene mask and the Universe column).
        {querySetId === MARKERS && ' Marker genes are one-vs-rest Wilcoxon per group; each group gets its own result.'}
      </div>
    </>
  )

  const gseaConfig = (
    <>
      <div style={styles.section}>
        <span style={styles.label}>Ranking</span>
        <div style={styles.row}>
          <label style={styles.checkRow}><input type="radio" checked={rankKind === 'diffexp'} onChange={() => setRankKind('diffexp')} /> Differential expression</label>
          <label style={styles.checkRow}><input type="radio" checked={rankKind === 'pca'} onChange={() => setRankKind('pca')} /> PCA loading</label>
          <span style={{ fontSize: '12px', color: '#ccc', marginLeft: '12px' }}>Cells:</span>
          <select style={styles.select} value={cellSubset} onChange={(e) => setCellSubset(e.target.value)}>
            <option value="">all cells</option>
            {cellSubsets.map((s) => <option key={s.name} value={s.name}>{s.name} ({s.n_cells.toLocaleString()})</option>)}
          </select>
        </div>
        {rankKind === 'diffexp' ? (
          <div style={{ ...styles.row, marginBottom: 0 }}>
            <select style={styles.select} value={obsColumn} onChange={(e) => { setObsColumn(e.target.value); batchGroupsRef.current = null }}>
              {categoricalColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
            <select style={styles.select} value={group} onChange={(e) => { setGroup(e.target.value); if (e.target.value !== ALL_GROUPS) batchGroupsRef.current = null }}>
              <option value={ALL_GROUPS}>All groups (one vs rest)</option>
              {groupOptions.map((g) => <option key={g.value} value={g.value}>{g.value} ({g.count.toLocaleString()})</option>)}
            </select>
            {group === ALL_GROUPS ? (
              <span style={styles.muted}>{batchNote}; one result per group plus a collection</span>
            ) : (
              <>
                <span style={styles.muted}>vs</span>
                <select style={styles.select} value={reference} onChange={(e) => setReference(e.target.value)}>
                  <option value="rest">rest</option>
                  {groupOptions.filter((g) => g.value !== group).map((g) => <option key={g.value} value={g.value}>{g.value}</option>)}
                </select>
              </>
            )}
            <select style={styles.select} value={method} onChange={(e) => setMethod(e.target.value as 'wilcoxon' | 't-test')}>
              <option value="wilcoxon">Wilcoxon</option>
              <option value="t-test">t-test</option>
            </select>
            <select style={styles.select} value={metric} onChange={(e) => setMetric(e.target.value as 'score' | 'log2fc')}>
              <option value="score">rank by test statistic</option>
              <option value="log2fc">rank by log2 fold change</option>
            </select>
          </div>
        ) : (
          <div style={{ ...styles.row, marginBottom: 0 }}>
            <span style={{ fontSize: '12px', color: '#ccc' }}>Component</span>
            <input style={styles.input} type="number" min={1} max={Math.max(1, nPcs)} value={component} onChange={(e) => setComponent(Math.max(1, Number(e.target.value) || 1))} />
            <span style={styles.muted}>{nPcs > 0 ? `${nPcs} available in ${pcaKey}` : `no ${pcaKey}: run PCA first`}</span>
          </div>
        )}
      </div>
      {setsBlock}
      <div style={styles.row}>
        <span style={{ fontSize: '12px', color: '#ccc' }}>Permutations</span>
        <input style={styles.input} type="number" min={10} step={100} value={nPerm} onChange={(e) => setNPerm(Math.max(10, Number(e.target.value) || 10))} />
        <span style={{ fontSize: '12px', color: '#ccc' }}>Set size</span>
        <input style={styles.input} type="number" min={1} value={gMin} onChange={(e) => setGMin(Math.max(1, Number(e.target.value) || 1))} />
        <span style={styles.muted}>to</span>
        <input style={styles.input} type="number" min={1} value={gMax} onChange={(e) => setGMax(Math.max(1, Number(e.target.value) || 1))} />
        <span style={{ fontSize: '12px', color: '#ccc' }}>Weight</span>
        <input style={styles.input} type="number" min={0} step={0.5} value={weight} onChange={(e) => setWeight(Math.max(0, Number(e.target.value) || 0))} />
        <span style={{ fontSize: '12px', color: '#ccc' }}>Seed</span>
        <input style={styles.input} type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
      </div>
      <div style={styles.muted}>Preranked GSEA (weighted running sum) with a gene-permutation null. p-values are empirical and floor at about 2/permutations; directional sets are tested as "set" and "set (down)".</div>
    </>
  )

  const summaryLine = result && (
    <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '8px' }}>
      <strong style={{ color: '#eee' }}>{result.label}</strong>
      {isBatch(result) ? (
        <>
          {' · '}{result.groups.length} groups · {result.n_sets_tested.toLocaleString()} sets tested in a universe of {result.universe_size.toLocaleString()} genes
          {result.kind === 'gsea_batch' && ` · ${result.n_perm} permutations`}
          {result.kind === 'ora_batch' && ` · top ${result.top_n} markers per group`}
          {' · '}{result.n_significant} rows at padj ≤ 0.05 in all
          {Object.keys(result.skipped).length > 0 && ` · skipped: ${Object.entries(result.skipped).map(([g, why]) => `${g} (${why})`).join('; ')}`}
        </>
      ) : (
        <>
          {' · '}{result.n_sets_tested.toLocaleString()} of {result.n_sets_input.toLocaleString()} sets tested in a universe of {result.universe_size.toLocaleString()} genes
          {result.kind === 'ora' && ` · query ${result.query.n_in_universe}/${result.query.n_input} in universe`}
          {result.kind === 'ora' && result.query.genes_missing.length > 0 && ` (missing: ${result.query.genes_missing.slice(0, 8).join(', ')}${result.query.genes_missing.length > 8 ? '…' : ''})`}
          {result.kind === 'gsea' && ` · ${result.ranking.n_ranked.toLocaleString()} ranked genes, ${result.n_perm} permutations`}
          {' · '}{result.n_significant} at padj ≤ 0.05
        </>
      )}
    </div>
  )

  const groupChips = result && isBatch(result) && (
    <div style={{ marginBottom: '8px' }}>
      {batchGroupsSorted(result).map((g) => (
        <span
          key={g} style={styles.groupChip(g === selectedGroup)}
          onClick={() => { setSelectedGroup(g); setExpanded(null) }}
          title={result.member_results?.[g]?.label}
        >
          {g} ({result.member_results?.[g]?.n_significant ?? 0})
        </span>
      ))}
    </div>
  )

  const renderRow = (row: OraRow | GseaRow) => {
    // Two selected libraries can share a term name; key and expansion are per library.
    const rowKey = `${row.library}\u0000${row.name}`
    const isOpen = expanded === rowKey
    const isOra = activeResult?.kind === 'ora'
    const barFrac = isOra
      ? Math.min(1, -Math.log10(Math.max(row.padj, 1e-10)) / 10)
      : Math.min(1, Math.abs((row as GseaRow).nes) / 3)
    const barColor = isOra || (row as GseaRow).nes >= 0 ? ACCENT : ALERT
    const genes = isOra ? (row as OraRow).genes : (row as GseaRow).leading_edge
    return (
      <Fragment key={rowKey}>
        <tr onClick={() => setExpanded(isOpen ? null : rowKey)} style={{ cursor: 'pointer', backgroundColor: isOpen ? '#0f1625' : undefined }}>
          <td style={{ ...styles.td, color: '#eee', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={row.description || row.name}>{row.name}</td>
          <td style={styles.td}>{row.library}</td>
          <td style={{ ...styles.td, textAlign: 'right' }}>{row.n_set}</td>
          <td style={{ ...styles.td, textAlign: 'right', color: barColor }}>
            {isOra ? `${(row as OraRow).n_overlap}/${row.n_set}` : (row as GseaRow).nes.toFixed(2)}
          </td>
          <td style={{ ...styles.td, textAlign: 'right' }}>{formatP(row.pval, pFloor)}</td>
          <td style={{ ...styles.td, textAlign: 'right', color: row.padj <= 0.05 ? '#eee' : '#888' }}>{formatP(row.padj)}</td>
          <td style={{ ...styles.td, width: '120px' }}>
            <div style={{ height: '8px', backgroundColor: '#0f1625', borderRadius: '2px' }}>
              <div style={{ height: '100%', width: `${Math.round(barFrac * 100)}%`, backgroundColor: barColor, borderRadius: '2px' }} />
            </div>
          </td>
        </tr>
        {isOpen && (
          <tr>
            <td colSpan={7} style={{ ...styles.td, backgroundColor: '#0f1625' }}>
              {row.description && <div style={{ ...styles.muted, marginBottom: '6px' }}>{row.description}</div>}
              {row.url && <a href={row.url} target="_blank" rel="noreferrer" style={{ fontSize: '11px', color: ACCENT }}>{row.url}</a>}
              {!isOra && (row as GseaRow).curve && activeResult?.kind === 'gsea' && (
                <GseaCurve row={row as GseaRow} ranking={activeResult.ranking} svgRef={svgRef} />
              )}
              <div style={{ marginTop: '6px' }}>
                <span style={styles.muted}>{isOra ? 'Overlap' : 'Leading edge'} ({genes.length}): </span>
                {genes.map((g) => (
                  <span key={g} style={styles.chip} title="Colour by this gene" onClick={(e) => { e.stopPropagation(); setSelectedGenes([g]) }}>{g}</span>
                ))}
              </div>
            </td>
          </tr>
        )}
      </Fragment>
    )
  }

  const resultsView = result && (
    <>
      {summaryLine}
      {groupChips}
      {!activeResult && <div style={styles.muted}>No per-group results are stored for this collection.</div>}
      {activeResult && (<>
      <div style={{ ...styles.row, marginBottom: '8px' }}>
        <label style={styles.checkRow}>
          <input type="checkbox" checked={padjMax !== null} onChange={(e) => setPadjMax(e.target.checked ? Number(padjText) || 0.05 : null)} />
          padj ≤
        </label>
        <input
          style={{ ...styles.input, width: '60px' }} value={padjText}
          onChange={(e) => { setPadjText(e.target.value); const v = Number(e.target.value); if (padjMax !== null && Number.isFinite(v) && v > 0) setPadjMax(v) }}
        />
        <input style={{ ...styles.input, width: '180px' }} placeholder="Filter by name or library" value={query} onChange={(e) => setQuery(e.target.value)} />
        {activeResult.kind === 'ora' && (
          <label style={styles.checkRow}><input type="checkbox" checked={hideBelow} onChange={(e) => setHideBelow(e.target.checked)} /> hide below min overlap</label>
        )}
        <span style={styles.muted}>{visibleRows.length} of {activeResult.results.length} shown</span>
      </div>
      <div style={{ maxHeight: '44vh', overflowY: 'auto', border: '1px solid #0f3460', borderRadius: '4px' }}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Set</th>
              <th style={styles.th}>Library</th>
              <th style={{ ...styles.th, textAlign: 'right' }}>Size</th>
              <th style={{ ...styles.th, textAlign: 'right' }}>{activeResult.kind === 'ora' ? 'Overlap' : 'NES'}</th>
              <th style={{ ...styles.th, textAlign: 'right' }}>p</th>
              <th style={{ ...styles.th, textAlign: 'right' }}>padj</th>
              <th style={styles.th}>{activeResult.kind === 'ora' ? '−log10 padj' : '|NES|'}</th>
            </tr>
          </thead>
          <tbody>{visibleRows.map((r) => renderRow(r))}</tbody>
        </table>
        {visibleRows.length === 0 && <div style={{ ...styles.muted, padding: '12px' }}>No sets pass the current filter.</div>}
      </div>
      </>)}
    </>
  )

  return (
    <div style={styles.backdrop} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Gene-set enrichment</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {previous.length > 0 && (
              <>
                <select style={styles.select} value={phase === 'results' && result ? result.key : ''} onChange={(e) => openPrevious(e.target.value)} title="Previous runs stored in this dataset">
                  <option value="">Previous runs…</option>
                  {previous.map((p) => (
                    <option key={p.key} value={p.key}>
                      {p.label} ({p.n_groups ? `${p.n_groups} groups, ` : ''}{p.n_significant}/{p.n_sets_tested})
                    </option>
                  ))}
                </select>
                {phase === 'results' && result && (
                  <button style={styles.closeButton} title="Delete this stored result" onClick={() => removePrevious(result.key)}>🗑</button>
                )}
              </>
            )}
            <button style={styles.closeButton} onClick={handleClose} title="Close">×</button>
          </div>
        </div>
        <div style={styles.body}>
          {phase !== 'results' && (
            <div style={styles.tabs}>
              <button style={styles.tab(tab === 'ora')} onClick={() => setTab('ora')}>Overlap (ORA)</button>
              <button style={styles.tab(tab === 'gsea')} onClick={() => setTab('gsea')}>GSEA (preranked)</button>
            </div>
          )}
          {error && <div style={styles.error}>{error}</div>}
          {phase === 'config' && (tab === 'ora' ? oraConfig : gseaConfig)}
          {phase === 'running' && (
            <div>
              <div style={{ fontSize: '13px', color: '#ccc' }}>{progress.message || 'Running…'}</div>
              <div style={styles.progressTrack}><div style={styles.progressBar(progress.frac)} /></div>
            </div>
          )}
          {phase === 'results' && resultsView}
        </div>
        <div style={styles.footer}>
          <div style={{ display: 'flex', gap: '8px' }}>
            {phase === 'results' ? (
              <button style={{ ...styles.button, ...styles.secondaryButton }} onClick={() => { setPhase('config'); setError(null) }}>← Back</button>
            ) : (
              <button style={{ ...styles.button, ...styles.secondaryButton }} onClick={handleClose}>Cancel</button>
            )}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {phase === 'results' && result && (
              <>
                {isBatch(result) && (
                  <>
                    <button style={{ ...styles.button, ...styles.secondaryButton, ...styles.disabledButton }} disabled title="Coming in the Figures update">Heatmap figure…</button>
                    <button style={{ ...styles.button, ...styles.secondaryButton, ...styles.disabledButton }} disabled title="Coming in the Figures update">Network figure…</button>
                  </>
                )}
                <button style={{ ...styles.button, ...styles.secondaryButton, ...(activeResult ? {} : styles.disabledButton) }} disabled={!activeResult} onClick={copyTsv}>{copied ? 'Copied' : 'Copy TSV'}</button>
                {activeResult?.kind === 'gsea' && expanded && (
                  <button style={{ ...styles.button, ...styles.secondaryButton }} onClick={downloadSvg}>Download SVG</button>
                )}
                <button
                  style={{ ...styles.button, ...styles.successButton, ...(saved || nSetsToAdd === 0 ? styles.disabledButton : {}) }}
                  disabled={saved || nSetsToAdd === 0} onClick={saveSets}
                  title={isBatch(result) ? 'One folder per group, one gene set per row passing the padj filter (top 50)' : 'One gene set per row passing the padj filter (top 50), in the Enrichment category'}
                >
                  {saved ? 'Added to gene sets' : nSetsToAdd === 0 ? 'Nothing to add' : `Add ${nSetsToAdd} to gene sets`}
                </button>
              </>
            )}
            {phase === 'config' && (
              <button
                style={{ ...styles.button, ...styles.primaryButton, ...((tab === 'ora' ? canRunOra : canRunGsea) ? {} : styles.disabledButton) }}
                disabled={tab === 'ora' ? !canRunOra : !canRunGsea}
                onClick={tab === 'ora' ? runOra : runGsea}
              >
                {tab === 'ora' ? (querySetId === MARKERS ? 'Run overlap test per group' : 'Run overlap test') : (group === ALL_GROUPS && rankKind === 'diffexp' ? 'Run GSEA per group' : 'Run GSEA')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
