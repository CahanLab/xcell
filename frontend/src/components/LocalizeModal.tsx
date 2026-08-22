import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask, refreshSchema } from '../hooks/useData'
import { assignRoles, type RefSlot } from '../lib/localizeRoles'
import { adviseParameters, adviseLayers, type PopulationGeometry } from '../lib/localizeAdvice'
import { layerOptionLabel, LayerScaleBadge, type LayerInfo } from './LayerScaleInfo'

/**
 * Localize — predict where each dissociated cell came from, using a spatial
 * dataset as the map.
 *
 * The panel is arranged around the one thing that makes this trustworthy: you
 * are shown the gene overlap *before* running, offered a cross-validation that
 * measures accuracy on the reference itself, and told afterwards how many cells
 * the method declined to place. A coordinate map is smooth and convincing
 * whether or not it is right, so none of that is optional decoration.
 */

const API = '/api'

const dark = {
  panel: '#16213e',
  border: '#0f3460',
  inset: '#0f1625',
  accent: '#4ecdc4',
  alert: '#e94560',
  warn: '#e9a23b',
  text: '#ccc',
  sub: '#aaa',
  faint: '#888',
}

// The wire shape lives with the role logic that consumes it, so the two cannot
// drift — this modal is the only consumer.
type ReferenceSlot = RefSlot

interface GeneColumn {
  name: string
  n_true: number
}

interface Overlap {
  n_shared: number
  n_reference_genes: number
  frac_of_reference: number
  n_missing: number
  missing_from_query: string[]
  sufficient: boolean
  severity: 'ok' | 'warn' | 'error'
}

interface CrossValidation {
  median_error: number | null
  median_error_normalized: number | null
  baseline_centroid: number | null
  baseline_random: number | null
  structure_correlation: number | null
  confidence_error_correlation: number | null
  fraction_within: Record<string, number | null>
  per_group: Record<string, { n: number; median_error: number | null }>
  same_platform: boolean
  n_holdout: number
}

interface LocalizeResult {
  embedding_name: string
  n_cells: number
  n_unplaced: number
  n_shared_genes: number
  n_missing_genes: number
  n_reference_cells: number
  section_col: string | null
  assignment: string | null
  n_candidates: number | null
  confidence_distribution: {
    median: number
    q25: number
    q75: number
    'n_below_0.2': number
    'n_below_0.5': number
  }
}

const TIPS: Record<string, string> = {
  k: 'How many spatial cells vote on each prediction. Too few is noisy; too many drags every cell toward the tissue centre.',
  transform: 'Applied to each dataset separately — that is what removes platform-level differences in per-gene capture. z-score is the safe default across platforms; rank additionally survives any monotone difference.',
  metric: 'How similarity between two cells is measured. Correlation on z-scored data and cosine coincide.',
  reference_layer: 'Which matrix on the spatial reference the profiles come from. A layer smoothed over the reference\u2019s spatial graph is the one worth trying: it averages out spot-level counting noise in the thing every query cell is matched against.',
  query_layer: 'Which matrix on the dataset being localized the profiles come from. Usually .X; a layer smoothed over an expression kNN graph is an option for very sparse query data.',
  aggregation: 'How the k neighbours become one point. weighted_mean is the classic estimator; densest lands the cell in real tissue when its neighbours sit in two separate patches, though it cannot say which patch; best_match snaps to a real reference cell; injective does the same but gives every cell a different one, which removes pile-up and asserts that the query’s composition matches the tissue’s; transport solves for all cells at once, subject to the tissue being occupied, which is the only option that trades between filling the tissue and keeping the gradient rather than sitting at one end.',
  epsilon: 'How much each cell is allowed to hedge across reference spots, in units of the cost matrix’s own spread — so the same value means the same thing whatever transform and metric you chose. Lower fills more of the tissue and behaves more like best match; higher averages more and eventually collapses toward the tissue centre. Measured on an E11.5 limb pair the useful band is roughly 0.03 to 0.3: at 0.05 the map covers 44% of the tissue over 1,774 distinct spots, at 0.5 it is down to 5.5% with the epidermis inverted again. Below 0.03 it converges too slowly to be worth it.',
  min_confidence: 'Cells whose neighbours disagree about location this badly get no coordinate at all, rather than a fabricated one. Leave at 0 to place everything and filter later.',
}

interface MapMetrics {
  embedding: string
  dispersion: {
    area_ratio: number | null; std_ratio_x: number | null
    std_ratio_y: number | null; frac_outside: number | null
  }
  occupancy: {
    n_distinct: number; max_per_spot: number
    effective_n: number | null; n_pred: number
  }
  markers: {
    name: string
    pattern: { correlation: number | null; n_bins_compared: number }
    axis: { reference: number | null; prediction: number | null }
  }[]
}

// 1.0 is right; the limb failure was 0.10 one way and 2.93 the other.
const areaColor = (v: number | null | undefined) =>
  v == null ? dark.faint : (v < 0.5 || v > 1.8) ? '#f0c987' : dark.text
// Negative means the cell type was placed where it is not — the epidermis
// inversion. That is a different kind of wrong from merely weak.
const patternColor = (v: number | null | undefined) =>
  v == null ? dark.faint : v < 0 ? '#ff8fa3' : v < 0.2 ? '#f0c987' : dark.text

export default function LocalizeModal() {
  const isOpen = useStore((s) => s.isLocalizeModalOpen)
  const setOpen = useStore((s) => s.setLocalizeModalOpen)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)

  const [refs, setRefs] = useState<ReferenceSlot[]>([])
  // '' = let assignRoles pick. Seeding this with a literal slot name was the
  // bug: the reference list is only returned by a call that names a valid
  // reference, so a wrong guess 400'd and the picker that would correct it
  // never populated.
  const [preferRef, setPreferRef] = useState('')
  const [overlap, setOverlap] = useState<Overlap | null>(null)
  const [sectionOptions, setSectionOptions] = useState<string[]>([])
  const [geneColumns, setGeneColumns] = useState<GeneColumn[]>([])
  // '' = every shared gene; 'col:<name>' = a reference .var flag;
  // 'set:<id>' = a gene set curated in the Gene panel.
  const [basis, setBasis] = useState('')
  const [showGuide, setShowGuide] = useState(false)
  const categories = useStore((s) => s.geneSetCategories)

  const [k, setK] = useState('15')
  const [transform, setTransform] = useState('zscore')
  const [metric, setMetric] = useState('correlation')
  const [aggregation, setAggregation] = useState('weighted_mean')
  const [epsilon, setEpsilon] = useState('0.05')
  const [minConfidence, setMinConfidence] = useState('0')
  const [sectionCol, setSectionCol] = useState('')
  const [keyAdded, setKeyAdded] = useState('X_spatial_pred')
  // Which matrix each side is read from. 'X' means .X, matching the backend's
  // layer=None. The two are independent: a smoothed ST reference paired with
  // an unsmoothed dissociated query is a normal thing to want.
  const [queryLayer, setQueryLayer] = useState('X')
  const [refLayer, setRefLayer] = useState('X')
  const [queryLayers, setQueryLayers] = useState<LayerInfo[]>([])
  const [refLayers, setRefLayers] = useState<LayerInfo[]>([])
  // Territories drawn on the reference can be carried over with the map.
  const [refTerritories, setRefTerritories] = useState<string[]>([])
  const [importTerritories, setImportTerritories] = useState(false)
  const [assignTerritories, setAssignTerritories] = useState(false)

  const [mapMetrics, setMapMetrics] = useState<MapMetrics[] | null>(null)
  const [scoring, setScoring] = useState(false)
  const [populations, setPopulations] = useState<PopulationGeometry[]>([])

  const [busy, setBusy] = useState<'' | 'check' | 'run'>('')
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cv, setCv] = useState<CrossValidation | null>(null)
  const [result, setResult] = useState<LocalizeResult | null>(null)

  // Every gene set the user has curated, from any category or folder.
  const geneSets = useMemo(() => {
    const out: { id: string; name: string; label: string; genes: string[] }[] = []
    for (const cat of Object.values(categories)) {
      const collect = (sets: { id: string; name: string; genes: string[] }[]) => {
        for (const gs of sets) {
          out.push({
            id: gs.id, name: gs.name, genes: gs.genes,
            label: `${cat.name}: ${gs.name} (${gs.genes.length})`,
          })
        }
      }
      collect(cat.geneSets)
      for (const f of cat.folders) collect(f.geneSets)
    }
    return out
  }, [categories])

  // The wire form of the chosen basis: a .var column name, an explicit gene
  // list, or null for every shared gene.
  const geneSubset = useMemo<string | string[] | null>(() => {
    if (basis.startsWith('col:')) return basis.slice(4)
    if (basis.startsWith('set:')) {
      return geneSets.find((g) => g.id === basis.slice(4))?.genes ?? null
    }
    return null
  }, [basis, geneSets])

  const close = () => {
    setOpen(false)
    setError(null); setCv(null); setResult(null); setBusy(''); setProgress(null)
  }

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen])

  // Roles come from the data, not from whichever slot happens to be active.
  const roles = useMemo(() => assignRoles(refs, preferRef || null), [refs, preferRef])
  const reference = roles.referenceSlot
  const querySlot = roles.querySlot
  const refInfo = refs.find((r) => r.slot === reference)
  const queryInfo = refs.find((r) => r.slot === querySlot)

  // Step 1: the unconditional call. Naming no reference cannot 400 — the route
  // builds the slot list before it validates one — so this always populates the
  // picker, whatever the arrangement.
  const loadSlots = useCallback(async () => {
    setError(null)
    try {
      const r = await fetch(`${API}/localize/suggest`)
      const body = await r.json()
      if (!r.ok) { setError(body.detail || `HTTP ${r.status}`); return }
      setRefs(body.references || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  // Step 2: once roles are known, ask again for the gene-overlap preview and
  // the reference's own .var flags and section columns.
  const loadSuggest = useCallback(async (
    refSlot: string, qSlot: string, column: string,
  ) => {
    setError(null)
    try {
      const q = column ? `&gene_subset=${encodeURIComponent(column)}` : ''
      const r = await fetch(
        appendDataset(`${API}/localize/suggest?reference=${refSlot}${q}`, qSlot),
      )
      const body = await r.json()
      if (!r.ok) {
        setOverlap(null)
        // Keep whatever slot list we already have — dropping it here is what
        // stranded the UI with "no dataset has spatial coordinates".
        if (r.status !== 400) setError(body.detail || `HTTP ${r.status}`)
        return
      }
      if (body.references) setRefs(body.references)
      setOverlap(body.overlap || null)
      setSectionOptions(body.reference_sections || [])
      setGeneColumns(body.reference_gene_columns || [])
      setK((prev) => (prev === '15' && body.suggested_k ? String(body.suggested_k) : prev))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    if (isOpen) loadSlots()
  }, [isOpen, loadSlots])

  // The readable matrices on each side, with the scale classification that
  // makes the choice meaningful. Fetched per slot: the query and the reference
  // are different datasets and rarely carry the same layers.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    const load = (slot: string | null, set: (v: LayerInfo[]) => void) => {
      if (!slot) { set([]); return }
      fetch(appendDataset(`${API}/scanpy/layers`, slot))
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (!cancelled && d?.layers) set(d.layers as LayerInfo[]) })
        .catch(() => { /* the picker falls back to .X */ })
    }
    load(querySlot, setQueryLayers)
    load(reference, setRefLayers)
    if (reference) {
      fetch(appendDataset(`${API}/territories`, reference))
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (!cancelled && d) setRefTerritories(Object.keys(d.territories || {})) })
        .catch(() => { /* the checkbox simply stays hidden */ })
    } else {
      setRefTerritories([])
    }
    return () => { cancelled = true }
  }, [isOpen, querySlot, reference])

  // A layer chosen on one dataset need not exist on the next one loaded.
  useEffect(() => {
    if (queryLayers.length && !queryLayers.some((l) => l.name === queryLayer)) setQueryLayer('X')
  }, [queryLayers, queryLayer])
  useEffect(() => {
    if (refLayers.length && !refLayers.some((l) => l.name === refLayer)) setRefLayer('X')
  }, [refLayers, refLayer])

  useEffect(() => {
    // Only a .var flag can be previewed by name; a gene list is checked when
    // the run starts, and the count below reports it either way.
    if (isOpen && reference && querySlot) {
      loadSuggest(reference, querySlot, basis.startsWith('col:') ? basis.slice(4) : '')
    }
  }, [isOpen, reference, querySlot, basis, loadSuggest])

  // Which of the user's populations this reference would collapse under an
  // averaging estimator. Reference-only, so it runs before any prediction
  // exists — the whole point is to warn while the parameters are being chosen,
  // not after a map has been made and believed.
  useEffect(() => {
    if (!isOpen || !reference || !querySlot || geneSets.length === 0) {
      setPopulations([])
      return
    }
    const sets: Record<string, string[]> = {}
    geneSets.forEach((g) => { if (g.genes.length) sets[g.name] = g.genes })
    if (Object.keys(sets).length === 0) { setPopulations([]); return }

    let cancelled = false
    fetch(appendDataset(`${API}/localize/reference_geometry`, querySlot), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference, dataset: querySlot, gene_sets: sets }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((payload) => {
        if (!cancelled && payload) setPopulations(payload.populations || [])
      })
      // Advice is a bonus. A failure here must not surface as a modal error and
      // must not block the run the user came to make.
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen, reference, querySlot, geneSets])

  const body = () => ({
    reference,
    // Name the query explicitly rather than letting the route fall back to the
    // active slot — the results are written into whichever dataset this names.
    dataset: querySlot,
    k: Number(k) || 15,
    transform, metric, aggregation,
    min_confidence: Number(minConfidence) || 0,
    epsilon: Number(epsilon) || 0.05,
    section_col: sectionCol || null,
    gene_subset: geneSubset,
    layer: queryLayer === 'X' ? null : queryLayer,
    reference_layer: refLayer === 'X' ? null : refLayer,
    import_territories: importTerritories,
    assign_territories: importTerritories && assignTerritories,
  })

  /** Score every predicted embedding in the query against the reference. */
  const scoreMaps = async () => {
    if (!reference || !querySlot) return
    setScoring(true); setError(null)
    try {
      const sets: Record<string, string[]> = {}
      geneSets.forEach((g) => { if (g.genes.length) sets[g.name] = g.genes })
      // Ask the server rather than the store. The modal's refreshSchema() after
      // a run refreshes the *active* slot, which is not necessarily the query,
      // so the cached list can be stale exactly when it matters.
      const sr = await fetch(
        appendDataset(`${API}/schema`, querySlot))
      const all: string[] = sr.ok ? ((await sr.json()).embeddings || []) : []
      // Only things that could be a predicted map. The query's own PCA/UMAP
      // are embeddings too and scoring them against tissue coordinates is
      // meaningless.
      const embeddings = all.filter((e) => /pred/i.test(e) || /spatial/i.test(e))
        .filter((e) => e !== 'spatial' && e !== 'X_spatial')
      if (embeddings.length === 0) {
        setError('No predicted embeddings to score — run Localize first.')
        return
      }
      const r = await fetch(
        appendDataset(`${API}/localize/evaluate_map`, querySlot),
        {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reference, dataset: querySlot, embeddings, gene_sets: sets,
          }),
        },
      )
      const payload = await r.json()
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      setMapMetrics(payload.maps)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setScoring(false)
    }
  }

  const runCheck = async () => {
    setBusy('check'); setError(null); setCv(null); setProgress(0)
    try {
      const r = await fetch(appendDataset(`${API}/localize/cross_validate`, querySlot ?? undefined), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Cross-validation holds out part of the *reference* and predicts it
        // from the rest, so the matrix it reads is the reference's.
        body: JSON.stringify({
          ...body(), holdout_fraction: 0.2,
          layer: refLayer === 'X' ? null : refLayer,
        }),
      })
      const payload = await r.json()
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      const status = await pollTask(payload.task_id, undefined,
        (s) => setProgress(s.progress ?? null))
      if (status.status !== 'completed') throw new Error(status.error || 'Check failed')
      setCv(status.result as unknown as CrossValidation)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(''); setProgress(null)
    }
  }

  const runLocalize = async () => {
    setBusy('run'); setError(null); setResult(null); setProgress(0)
    try {
      const r = await fetch(appendDataset(`${API}/localize/prepare`, querySlot ?? undefined), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body(), key_added: keyAdded }),
      })
      const payload = await r.json()
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      const status = await pollTask(payload.task_id, undefined,
        (s) => setProgress(s.progress ?? null))
      if (status.status !== 'completed') throw new Error(status.error || 'Localize failed')
      setResult(status.result as unknown as LocalizeResult)
      // A new embedding *and* new .obs columns, so both refresh channels are
      // needed: the schema drives the embedding picker, the summaries version
      // drives the Cell Manager's column list.
      await refreshSchema()
      refreshObsSummaries()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(''); setProgress(null)
    }
  }

  if (!isOpen) return null

  const ready = roles.referenceSlot != null
  const blocked = overlap != null && !overlap.sufficient
  const chosenRef = refInfo
  const basisGenes = basis.startsWith('set:')
    ? (geneSets.find((g) => g.id === basis.slice(4))?.genes.length ?? null)
    : (overlap?.n_shared ?? null)
  const queryLayerInfo = queryLayers.find((l) => l.name === queryLayer)
  const refLayerInfo = refLayers.find((l) => l.name === refLayer)
  const advice = [
    ...adviseParameters({
      k: Number(k) || 0,
      transform, metric, aggregation,
      minConfidence: Number(minConfidence) || 0,
      epsilon: Number(epsilon) || 0.05,
      nReferenceCells: chosenRef?.n_cells ?? 0,
      nQueryCells: queryInfo?.n_cells ?? 0,
      nSharedGenes: basisGenes,
      populations,
    }),
    ...adviseLayers({
      queryScale: queryLayerInfo?.scale?.verdict ?? null,
      referenceScale: refLayerInfo?.scale?.verdict ?? null,
      transform,
    }),
  ]

  return (
    <div onClick={close} style={overlayStyle}>
      <div onClick={(e) => e.stopPropagation()} style={cardStyle}>
        <div style={headerStyle}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>
              Localize — predict spatial coordinates
            </div>
            <div style={{ fontSize: 11, color: dark.faint, marginTop: 2 }}>
              Place dissociated cells on a tissue map borrowed from a spatial dataset
            </div>
          </div>
          <button onClick={close} style={{ ...ghost, fontSize: 16 }}>×</button>
        </div>

        <div style={{ overflowY: 'auto', padding: '12px 16px', flex: 1 }}>
          {error && <div style={errBox}>{error}</div>}

          {/* Roles. Derived from which dataset carries coordinates, not from
              whichever slot is active — but stated, so it is a visible
              decision rather than a hidden one. */}
          <div style={sectionLabel}>Datasets</div>
          {!ready ? (
            <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
              {roles.problem} Use <b>File → Load…</b>, then reopen this tool.
            </div>
          ) : (
            <div style={{ ...noticeStyle, display: 'grid', gap: 4 }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ color: dark.faint, width: 74 }}>Reference</span>
                <span>
                  {refInfo?.filename}
                  <span style={{ color: dark.faint }}>
                    {' '}({refInfo?.spatial_key}, {refInfo?.n_cells.toLocaleString()} cells)
                  </span>
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ color: dark.faint, width: 74 }}>Query</span>
                <span>
                  {queryInfo?.filename}
                  <span style={{ color: dark.faint }}>
                    {' '}({queryInfo?.n_cells.toLocaleString()} cells)
                  </span>
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ color: dark.faint, width: 74 }}>Writes to</span>
                <span style={{ color: dark.faint }}>
                  {queryInfo?.filename} · <code>.obsm['{keyAdded}']</code>
                </span>
                {roles.swappable && (
                  <button style={{ ...ghost, fontSize: 11, marginLeft: 'auto' }}
                          onClick={() => setPreferRef(querySlot || '')}>
                    swap
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Gene overlap — shown before anything runs, on purpose. */}
          {overlap && (
            <div style={{
              ...noticeStyle,
              background: overlap.severity === 'ok' ? 'rgba(78,205,196,0.12)'
                : overlap.severity === 'warn' ? 'rgba(233,162,59,0.14)'
                : 'rgba(233,69,96,0.15)',
              color: overlap.severity === 'ok' ? '#bfeae6'
                : overlap.severity === 'warn' ? '#f0c987' : '#ff8fa3',
            }}>
              <b>{overlap.n_shared.toLocaleString()}</b> of the reference's{' '}
              {overlap.n_reference_genes.toLocaleString()} genes are present here
              ({(overlap.frac_of_reference * 100).toFixed(0)}%).
              {!overlap.sufficient && ' Too few to predict from — the result would look confident and mean nothing.'}
              {overlap.sufficient && overlap.severity !== 'ok' &&
                ' Expect a weaker mapping; check the accuracy below before trusting it.'}
              {overlap.n_missing > 0 && (
                <div style={{ marginTop: 4, fontSize: 10, opacity: 0.8 }}>
                  Missing e.g. {overlap.missing_from_query.slice(0, 8).join(', ')}
                  {overlap.n_missing > 8 ? ` … (+${overlap.n_missing - 8})` : ''}
                </div>
              )}
            </div>
          )}

          {/* Which genes similarity is computed over. The reference owns this
              choice: they are the genes shown to carry positional signal. */}
          <div style={{ ...sectionLabel, marginTop: 14 }}>Similarity basis (genes)</div>
          <select value={basis} onChange={(e) => setBasis(e.target.value)}
                  style={{ ...input, width: '100%' }}>
            <option value="">All shared genes</option>
            {geneColumns.length > 0 && (
              <optgroup label="Reference .var flags">
                {geneColumns.map((c) => (
                  <option key={c.name} value={`col:${c.name}`}>
                    {c.name} ({c.n_true.toLocaleString()} genes)
                  </option>
                ))}
              </optgroup>
            )}
            {geneSets.length > 0 && (
              <optgroup label="Gene sets">
                {geneSets.map((g) => (
                  <option key={g.id} value={`set:${g.id}`}>{g.label}</option>
                ))}
              </optgroup>
            )}
          </select>
          <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 4 }}>
            Genes that vary <i>spatially</i> are the ones carrying positional
            information, so <code>spatially_variable</code> (from Spatial
            Autocorrelation on the reference) is the principled choice when it
            exists; <code>highly_variable</code> is a reasonable stand-in. Using
            every gene is fine for a targeted panel, but on whole-transcriptome
            data it dilutes the signal with thousands of genes that say nothing
            about position.
            {geneColumns.length === 0 &&
              ' This reference carries no .var flags yet — run Spatial → Spatial Autocorrelation on it to create one.'}
          </div>

          {/* Source matrix — which matrix each side is read from. Kept next to
              the gene basis: both answer "what expression is being compared",
              and both are invisible in the result if you get them wrong. */}
          <div style={{ ...sectionLabel, marginTop: 14 }}>Source matrix</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <Field label="Reference matrix" tip={TIPS.reference_layer}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <select value={refLayer} onChange={(e) => setRefLayer(e.target.value)} style={input}>
                  {(refLayers.length ? refLayers : [{ name: 'X', density: 0 } as LayerInfo])
                    .map((L) => <option key={L.name} value={L.name}>{layerOptionLabel(L)}</option>)}
                </select>
                <LayerScaleBadge layer={refLayerInfo} align="right" />
              </div>
            </Field>
            <Field label="Query matrix" tip={TIPS.query_layer}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <select value={queryLayer} onChange={(e) => setQueryLayer(e.target.value)} style={input}>
                  {(queryLayers.length ? queryLayers : [{ name: 'X', density: 0 } as LayerInfo])
                    .map((L) => <option key={L.name} value={L.name}>{layerOptionLabel(L)}</option>)}
                </select>
                <LayerScaleBadge layer={queryLayerInfo} align="right" />
              </div>
            </Field>
          </div>
          <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 4 }}>
            Smoothing the <i>reference</i> over its spatial graph (Preprocess →
            Smooth on the spatial dataset) raises the positional signal
            substantially: on an E11.5 limb held-out test it roughly tripled the
            share of spots placed within 10% of the tissue radius, and the
            transform it was smoothed on barely mattered — the per-dataset
            transform below absorbs that. See
            <code> docs/measurements/2026-08-21-smoothing-transform.md</code>.
          </div>

          {refTerritories.length > 0 && (
            <>
              <div style={{ ...sectionLabel, marginTop: 14 }}>Territories</div>
              <label style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 6, cursor: 'pointer' }}>
                <input type="checkbox" checked={importTerritories}
                       onChange={(e) => setImportTerritories(e.target.checked)}
                       style={{ marginTop: 2 }} />
                <span style={{ fontSize: 12 }}>
                  Import territory boundaries from the reference
                  <span style={{ color: dark.faint }}> ({refTerritories.join(', ')})</span>
                </span>
              </label>
              <label style={{
                display: 'flex', gap: 8, alignItems: 'flex-start',
                opacity: importTerritories ? 1 : 0.5,
                cursor: importTerritories ? 'pointer' : 'not-allowed',
              }}>
                <input type="checkbox" checked={importTerritories && assignTerritories}
                       disabled={!importTerritories}
                       onChange={(e) => setAssignTerritories(e.target.checked)}
                       style={{ marginTop: 2 }} />
                <span style={{ fontSize: 12 }}>Assign cells to territories after localizing</span>
              </label>
              <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 4 }}>
                Boundaries are copied in against the predicted embedding, so they
                draw over the localized cells. Every cell that got a coordinate is
                labelled; cells that could not be placed are left blank —
                <code> localize_confidence</code> sits beside the result for filtering.
              </div>
            </>
          )}

          {/* Parameters */}
          <div style={{ ...sectionLabel, marginTop: 14 }}>Parameters</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <Field label="Neighbours (k)" tip={TIPS.k}>
              <input value={k} onChange={(e) => setK(e.target.value)} style={input} />
            </Field>
            <Field label="Expression transform" tip={TIPS.transform}>
              <select value={transform} onChange={(e) => setTransform(e.target.value)} style={input}>
                <option value="zscore">z-score per dataset (default)</option>
                <option value="rank">rank within cell</option>
                <option value="log1p">log1p</option>
                <option value="none">none</option>
              </select>
            </Field>
            <Field label="Similarity" tip={TIPS.metric}>
              <select value={metric} onChange={(e) => setMetric(e.target.value)} style={input}>
                <option value="correlation">correlation</option>
                <option value="cosine">cosine</option>
                <option value="euclidean">euclidean</option>
              </select>
            </Field>
            <Field label="Aggregation" tip={TIPS.aggregation}>
              <select value={aggregation} onChange={(e) => setAggregation(e.target.value)} style={input}>
                <option value="weighted_mean">weighted mean</option>
                <option value="median">median</option>
                <option value="densest">densest cluster</option>
                <option value="best_match">best match</option>
                {/* Gated on the same condition the backend enforces, stated
                    where the choice is made rather than after a failed run. */}
                <option value="injective"
                        disabled={(queryInfo?.n_cells ?? 0) > (refInfo?.n_cells ?? 0)}>
                  injective (a distinct spot each)
                </option>
                <option value="transport">transport (fills the tissue)</option>
              </select>
            </Field>
            {/* Only the aggregation it belongs to: a dial with no effect is
                worse than no dial, because it invites tuning that does nothing. */}
            {aggregation === 'transport' && (
              <Field label="ε (spread)" tip={TIPS.epsilon}>
                <input value={epsilon} onChange={(e) => setEpsilon(e.target.value)}
                       style={input} />
              </Field>
            )}
            <Field label="Min confidence" tip={TIPS.min_confidence}>
              <input value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} style={input} />
            </Field>
            <Field label="Reference section column" tip="Keeps neighbourhoods inside one tissue section, so a coordinate is never averaged across the gap between cuts.">
              <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)} style={input}>
                <option value="">none</option>
                {sectionOptions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </Field>
          </div>

          {/* Live advice about the current selection. */}
          {advice.length > 0 && (
            <div style={{ marginTop: 10, display: 'grid', gap: 5 }}>
              {advice.map((a, i) => (
                <div key={i} style={{
                  fontSize: 11, lineHeight: 1.45,
                  color: a.level === 'warn' ? '#f0c987' : dark.sub,
                  background: a.level === 'warn' ? 'rgba(233,162,59,0.12)' : 'transparent',
                  borderLeft: `2px solid ${a.level === 'warn' ? dark.warn : dark.border}`,
                  padding: '4px 8px', borderRadius: 2,
                }}>
                  {a.level === 'warn' ? '⚠ ' : ''}{a.text}
                </div>
              ))}
            </div>
          )}

          <button onClick={() => setShowGuide((v) => !v)}
                  style={{ ...ghost, marginTop: 8, fontSize: 11 }}>
            {showGuide ? 'Hide' : 'Show'} how to choose these
          </button>
          {showGuide && <ParameterGuide />}

          {/* Map quality — compares maps already produced, against the
              reference. Distinct from Accuracy below, which holds out part of
              the reference and predicts it. */}
          <div style={{ ...sectionLabel, marginTop: 16 }}>Map quality</div>
          <div style={{ fontSize: 11, color: dark.sub, marginBottom: 6 }}>
            Scores every predicted embedding in the query against the reference.
            The numbers are comparative: a pattern score means little alone, and
            an axis correlation means nothing until you see what the reference
            itself reaches. <b>Area</b> 1.0 fills the tissue — well under means
            the map collapsed toward the centre. A <b>negative</b> marker score
            means that cell type was placed where it is not.
          </div>
          <button onClick={scoreMaps} disabled={scoring || !ready}
                  style={{ ...ghost, fontSize: 11 }}>
            {scoring ? 'Scoring…' : 'Score predicted maps'}
          </button>

          {mapMetrics && mapMetrics.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%',
                              fontSize: 11, marginTop: 8 }}>
                <thead>
                  <tr style={{ color: dark.faint, textAlign: 'left' }}>
                    <th style={th}>embedding</th>
                    <th style={th}>area</th>
                    <th style={th}>outside</th>
                    <th style={th}>spots used</th>
                    {mapMetrics[0].markers.map((m) => (
                      <th key={m.name} style={th}>{m.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mapMetrics.map((m) => (
                    <tr key={m.embedding}>
                      <td style={td}><code>{m.embedding}</code></td>
                      <td style={{ ...td, color: areaColor(m.dispersion.area_ratio) }}>
                        {fmt(m.dispersion.area_ratio, 2)}
                      </td>
                      <td style={td}>{pct(m.dispersion.frac_outside)}</td>
                      <td style={td}>
                        {m.occupancy.n_distinct.toLocaleString()}
                        <span style={{ color: dark.faint }}>
                          {' '}/ {m.occupancy.n_pred.toLocaleString()}
                        </span>
                      </td>
                      {m.markers.map((k) => (
                        <td key={k.name}
                            style={{ ...td, color: patternColor(k.pattern.correlation) }}
                            title={`axis ${fmt(k.axis.prediction, 2)} of ${fmt(k.axis.reference, 2)} possible`}>
                          {fmt(k.pattern.correlation, 2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 4 }}>
                Marker columns show spatial pattern fidelity; hover one for its
                axis correlation against the reference's ceiling.
              </div>
            </div>
          )}

          {/* Accuracy check */}
          <div style={{ ...sectionLabel, marginTop: 16 }}>Accuracy</div>
          <div style={{ fontSize: 11, color: dark.sub, marginBottom: 6 }}>
            Holds out a fifth of the reference and predicts it from the rest, with
            these exact parameters. Ground truth is known there, so the error is
            exact — but both halves come from the same platform, so the real
            scRNA-seq mapping will be worse than this.
          </div>
          <button onClick={runCheck} disabled={!!busy || blocked || !ready}
                  style={{ ...ghost, opacity: busy || blocked ? 0.5 : 1 }}>
            {busy === 'check' ? 'Checking…' : 'Check accuracy on the reference'}
          </button>

          {cv && (
            <div style={{ marginTop: 8, padding: 10, background: dark.inset, borderRadius: 4 }}>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: dark.text }}>
                <span><b style={{ color: dark.accent }}>{fmt(cv.median_error)}</b> median error</span>
                <span style={{ color: dark.sub }}>vs {fmt(cv.baseline_centroid)} predicting the tissue centre</span>
                <span style={{ color: dark.sub }}>vs {fmt(cv.baseline_random)} at random</span>
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 6 }}>
                {pct(cv.fraction_within?.['10%'])} of held-out cells land within 10% of
                the tissue diameter. Structure preserved r={fmt(cv.structure_correlation, 2)};
                confidence tracks error r={fmt(cv.confidence_error_correlation, 2)}.
              </div>
              {beatsBaseline(cv) === false && (
                <div style={{ ...noticeStyle, background: 'rgba(233,69,96,0.15)', color: '#ff8fa3' }}>
                  This does not beat simply predicting the middle of the tissue. The
                  shared genes carry little positional information — a map from these
                  parameters would not mean anything.
                </div>
              )}
              {Object.keys(cv.per_group || {}).length > 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: dark.sub }}>
                  {Object.entries(cv.per_group).map(([g, v]) => (
                    <span key={g} style={{ marginRight: 12 }}>{g}: {fmt(v.median_error)}</span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Run */}
          <div style={{ ...sectionLabel, marginTop: 16 }}>Apply</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input value={keyAdded} onChange={(e) => setKeyAdded(e.target.value)}
                   style={{ ...input, flex: 1 }} placeholder="X_spatial_pred" />
            <button onClick={runLocalize} disabled={!!busy || blocked || !ready}
                    style={{
                      ...ghost, background: dark.accent, color: '#0b1020',
                      fontWeight: 600, border: `1px solid ${dark.accent}`,
                      opacity: busy || blocked || !ready ? 0.5 : 1,
                    }}>
              {busy === 'run' ? 'Localizing…' : 'Localize'}
            </button>
          </div>

          {progress !== null && busy && (
            <div style={{ marginTop: 8, height: 4, background: dark.inset, borderRadius: 2 }}>
              <div style={{
                width: `${Math.round((progress || 0) * 100)}%`, height: '100%',
                background: dark.accent, borderRadius: 2, transition: 'width 200ms',
              }} />
            </div>
          )}

          {result && (
            <div style={{ marginTop: 10, padding: 10, background: 'rgba(78,205,196,0.12)', borderRadius: 4 }}>
              <div style={{ fontSize: 12, color: '#bfeae6', fontWeight: 600 }}>
                Wrote <code>.obsm['{result.embedding_name}']</code> for{' '}
                {(result.n_cells - result.n_unplaced).toLocaleString()} of{' '}
                {result.n_cells.toLocaleString()} cells
                {result.n_unplaced > 0 && ` (${result.n_unplaced.toLocaleString()} left unplaced)`}.
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 5 }}>
                Median confidence {result.confidence_distribution.median.toFixed(2)}.{' '}
                <b>{result.confidence_distribution['n_below_0.2'].toLocaleString()}</b> cells
                score below 0.2 — their transcriptional neighbours disagree about where
                they belong, so their coordinates carry little information.
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 5 }}>
                Select <b>{result.embedding_name}</b> in the embedding picker, then
                colour by <b>{result.embedding_name}_confidence</b> to see which regions
                of the map to trust.
              </div>
              {result.assignment === 'candidate' && (
                <div style={{ fontSize: 11, color: '#f0c987', marginTop: 5 }}>
                  Too large to assign exactly, so each cell chose among its{' '}
                  {result.n_candidates} best-matching spots rather than all{' '}
                  {result.n_reference_cells.toLocaleString()}. Every cell still has
                  its own spot and the total similarity is within a fraction of a
                  percent of the exact answer, but this is <b>near-optimal, not
                  optimal</b>.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * What to pick, and what each choice costs. Written as three concrete recipes
 * plus the trade-offs, rather than a restatement of the dropdown labels — a
 * user opening this wants to know which combination to use, not what the words
 * mean.
 */
function ParameterGuide() {
  return (
    <div style={{
      marginTop: 8, padding: 10, background: dark.inset, borderRadius: 4,
      fontSize: 11, lineHeight: 1.5, color: dark.sub,
    }}>
      <div style={{ ...sectionLabel, marginBottom: 6 }}>Starting points</div>
      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 10 }}>
        <thead>
          <tr style={{ color: dark.faint, textAlign: 'left' }}>
            <th style={th}>Situation</th><th style={th}>Use</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={td}>scRNA-seq onto Visium / Xenium <span style={{ color: dark.faint }}>(the usual case)</span></td>
            <td style={td}><code>z-score</code> + <code>correlation</code> + <code>weighted mean</code>, basis <code>spatially_variable</code>, k ≈ √(reference cells)</td>
          </tr>
          <tr>
            <td style={td}>Platforms that differ a lot, or unknown normalization</td>
            <td style={td}><code>rank</code> + <code>cosine</code> — survives any monotone difference between the two, at the cost of ignoring magnitude</td>
          </tr>
          <tr>
            <td style={td}>Both datasets from the same platform and pipeline</td>
            <td style={td}>Any transform; <code>euclidean</code> becomes meaningful and is the most literal comparison</td>
          </tr>
        </tbody>
      </table>

      <div style={{ ...sectionLabel, marginBottom: 6 }}>Trade-offs</div>
      <ul style={{ margin: 0, paddingLeft: 16, display: 'grid', gap: 5 }}>
        <li>
          <b>k</b> trades noise against blur. Small k follows fine structure but
          is driven by a few cells; large k is stable but pulls every prediction
          toward the middle of the tissue, and past ~10% of the reference that
          dominates everything else. Confidence partly self-corrects — a wider
          spread of neighbours lowers it — so a suspiciously high k shows up as
          uniformly low confidence rather than as a silently worse map.
        </li>
        <li>
          <b>Transform</b> is applied to each dataset <i>separately</i>; that is
          the whole point, and it is what cancels per-gene capture efficiency.{' '}
          <code>z-score</code> assumes the query is heterogeneous — standardizing
          a set of nearly identical cells amplifies their noise instead.
        </li>
        <li>
          <b>Aggregation</b> trades plausibility against precision.{' '}
          <code>weighted mean</code> is the best estimate when neighbours agree
          and lands in empty space when they do not. <code>best match</code> is
          always on real tissue but quantized to one reference cell.{' '}
          <code>densest</code> handles a cell type present in two places by
          choosing one — but it cannot know which, so it is right about half the
          time for such cells. <code>injective</code> is <code>best match</code>{' '}
          solved as a set rather than one cell at a time, so no reference spot
          absorbs many cells (measured on an E11.5 limb: 2,683 cells onto 2,683
          distinct spots, up from 865, for a 7% drop in mean similarity). It
          needs at least as many spots as cells, and it assumes the query's
          composition matches the tissue's — which dissociation makes untrue in
          a way that pushes over-represented types where they do not belong. It
          fixes pile-up, not placement. <code>median</code> is the mean's safer
          sibling.
        </li>
        <li>
          <b>Gene basis</b> helps in proportion to how much of the panel is
          uninformative. On whole-transcriptome data, restricting to spatially
          variable genes removes thousands that say nothing about position and
          the map sharpens; on a targeted panel where most genes already carry
          signal, expect little change. The cost is real either way: fewer genes
          shared with the query, and a population defined by genes outside the
          set becomes invisible.
        </li>
        <li>
          <b>Min confidence</b> trades coverage against honesty. It is the one
          parameter that cannot make the map wrong — it only removes cells the
          method could not place.
        </li>
      </ul>

      <div style={{ ...sectionLabel, margin: '10px 0 6px' }}>Avoid</div>
      <ul style={{ margin: 0, paddingLeft: 16, display: 'grid', gap: 5 }}>
        <li><code>none</code> + <code>euclidean</code> across platforms — sequencing depth becomes the strongest signal in the data.</li>
        <li>Tuning k or the basis until the picture looks nicer. A smoother map is what this method produces when it is failing; use the accuracy check, which has ground truth, instead.</li>
        <li>Reading a confidence threshold across runs. It depends on the gene basis and on k, so re-check the distribution after changing either.</li>
        <li>Reading <code>injective</code> or <code>best match</code> as "more accurate" because the map fills the tissue. Filling the tissue and carrying a gradient are the two ends of one trade-off: on the limb pair, <code>weighted mean</code> kept the proximodistal gradient at 0.24 of a 0.34 ceiling while collapsing to 15% of the area, and both single-spot methods held the full area with essentially no gradient at all. Score them under <b>Map quality</b> and pick against what you need.</li>
      </ul>
    </div>
  )
}

const th: React.CSSProperties = {
  padding: '3px 6px', borderBottom: `1px solid ${dark.border}`, fontWeight: 600,
}
const td: React.CSSProperties = {
  padding: '4px 6px', borderBottom: `1px solid rgba(255,255,255,0.05)`,
  verticalAlign: 'top',
}

function beatsBaseline(cv: CrossValidation): boolean | null {
  if (cv.median_error == null || cv.baseline_centroid == null) return null
  return cv.median_error < cv.baseline_centroid
}

function fmt(v: number | null | undefined, digits = 1): string {
  return v == null ? '—' : v.toFixed(digits)
}

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`
}

function Field({ label, tip, children }: {
  label: string; tip?: string; children: React.ReactNode
}) {
  return (
    <label style={{ display: 'block' }} title={tip}>
      <div style={{ fontSize: 11, color: dark.sub, marginBottom: 3 }}>{label}</div>
      {children}
    </label>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
}

const cardStyle: React.CSSProperties = {
  backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
  borderRadius: 8, width: 'min(680px, 94vw)', maxHeight: '90vh',
  display: 'flex', flexDirection: 'column', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
}

const headerStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  padding: '12px 16px', borderBottom: `1px solid ${dark.border}`,
}

const input: React.CSSProperties = {
  background: dark.inset, border: `1px solid ${dark.border}`, borderRadius: 4,
  color: dark.text, fontSize: 12, padding: '6px 8px', outline: 'none', width: '100%',
}

const ghost: React.CSSProperties = {
  background: 'transparent', border: `1px solid ${dark.border}`, borderRadius: 4,
  color: dark.sub, cursor: 'pointer', fontSize: 12, padding: '6px 10px',
}

const sectionLabel: React.CSSProperties = {
  fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6,
  color: dark.faint, marginBottom: 6,
}

const noticeStyle: React.CSSProperties = {
  marginTop: 8, padding: '8px 10px', borderRadius: 4, fontSize: 11.5,
}

const errBox: React.CSSProperties = {
  marginBottom: 10, padding: '8px 10px', borderRadius: 4,
  background: 'rgba(233,69,96,0.15)', color: '#ff8fa3', fontSize: 12,
}
