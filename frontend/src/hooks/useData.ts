import { useEffect, useCallback, useState } from 'react'
import { useStore, DatasetSlot, Schema, EmbeddingData, ObsColumnData, ExpressionData, BivariateExpressionData, DiffExpResult, LineAssociationResult, GeneMaskConfig, PCASubsetSummary } from '../store'
import { MESSAGES } from '../messages'

const API_BASE = '/api'

/** Append `?dataset=<slot>` to a URL for non-primary slots. */
export function appendDataset(url: string, slot?: DatasetSlot): string {
  const s = slot ?? useStore.getState().activeSlot
  if (s === 'primary') return url
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}dataset=${s}`
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

// --- Background task polling for cancellable operations ---

export interface TaskStatus {
  task_id: string
  status: 'running' | 'completed' | 'cancelled' | 'error'
  result?: Record<string, unknown>
  error?: string
}

/** Poll a background task until it reaches a terminal state. */
export async function pollTask(taskId: string, slot?: DatasetSlot): Promise<TaskStatus> {
  const POLL_INTERVAL = 1000
  const MAX_RETRIES = 3

  let retries = 0
  while (true) {
    try {
      const status = await fetchJson<TaskStatus>(
        appendDataset(`${API_BASE}/tasks/${taskId}`, slot)
      )
      retries = 0

      if (status.status !== 'running') {
        return status
      }
    } catch (err) {
      retries++
      if (retries >= MAX_RETRIES) {
        return {
          task_id: taskId,
          status: 'error',
          error: `Lost connection to task: ${(err as Error).message}`,
        }
      }
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
  }
}

/** Cancel a running background task. */
export async function cancelTask(taskId: string, slot?: DatasetSlot): Promise<void> {
  await fetchJson(appendDataset(`${API_BASE}/tasks/${taskId}/cancel`, slot), {
    method: 'POST',
  })
}

export function useSchema() {
  const { schema, setSchema, setLoading, setError, setSelectedEmbedding } = useStore()

  useEffect(() => {
    if (schema) return // Already loaded

    setLoading(true)
    fetchJson<Schema>(appendDataset(`${API_BASE}/schema`))
      .then((data) => {
        setSchema(data)
        // Auto-select embedding by preference: spatial > umap > pca > first available
        if (data.embeddings.length > 0) {
          const preferred = ['spatial', 'umap', 'pca']
          const lower = data.embeddings.map((e) => e.toLowerCase())
          const pick = preferred.find((p) => lower.some((l) => l.includes(p)))
          const idx = pick != null ? lower.findIndex((l) => l.includes(pick)) : 0
          setSelectedEmbedding(data.embeddings[idx])
        }
        // Fetch var identifier columns for gene name switching
        fetchVarIdentifierColumns()
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [schema, setSchema, setLoading, setError, setSelectedEmbedding])

  return schema
}

export function useEmbedding() {
  const { selectedEmbedding, embedding, setEmbedding, setLoading, setError } = useStore()

  useEffect(() => {
    if (!selectedEmbedding) return
    if (embedding?.name === selectedEmbedding) return // Already loaded

    setLoading(true)
    fetchJson<EmbeddingData>(appendDataset(`${API_BASE}/embedding/${selectedEmbedding}`))
      .then(setEmbedding)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [selectedEmbedding, embedding, setEmbedding, setLoading, setError])

  return embedding
}

export function useColorBy() {
  const { selectedColorColumn, colorBy, colorMode, setColorBy, setLoading, setError } = useStore()

  useEffect(() => {
    // Only fetch metadata color if in metadata mode
    if (colorMode !== 'metadata') {
      return
    }
    if (!selectedColorColumn) {
      setColorBy(null)
      return
    }
    if (colorBy?.name === selectedColorColumn) return // Already loaded

    setLoading(true)
    fetchJson<ObsColumnData>(appendDataset(`${API_BASE}/obs/${selectedColorColumn}`))
      .then(setColorBy)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [selectedColorColumn, colorBy, colorMode, setColorBy, setLoading, setError])

  return colorBy
}

// Hook to re-fetch expression data when transform setting changes
export function useExpressionTransformEffect() {
  const {
    selectedGenes,
    colorMode,
    displayPreferences,
    setExpressionData,
    setLoading,
    setError,
    layoutMode,
    activeSlot,
    displayLayer,
  } = useStore()

  useEffect(() => {
    // Only re-fetch if we're in expression mode and have genes selected
    if (colorMode !== 'expression' || selectedGenes.length === 0) {
      return
    }

    const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined
    // 'X' (the default) is omitted from requests; backend treats missing /
    // 'X' identically and reads adata.X.
    const layerArg = displayLayer && displayLayer !== 'X' ? displayLayer : undefined

    const fetchExpression = async () => {
      setLoading(true)
      try {
        const clipPct = displayPreferences.clipPercentile
        if (selectedGenes.length === 1) {
          const qs = new URLSearchParams()
          if (transform) qs.set('transform', transform)
          if (clipPct > 0) qs.set('clip_percentile', String(clipPct))
          if (layerArg) qs.set('layer', layerArg)
          const suffix = qs.toString()
          const url = suffix
            ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(selectedGenes[0])}?${suffix}`)
            : appendDataset(`${API_BASE}/expression/${encodeURIComponent(selectedGenes[0])}`)
          const data = await fetchJson<ExpressionData>(url)
          setExpressionData(data)
        } else {
          const data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/multi`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              genes: selectedGenes,
              transform,
              per_gene_norm: displayPreferences.geneSetPerGeneNorm,
              per_gene_clip: displayPreferences.geneSetPerGeneClip,
              aggregation: displayPreferences.geneSetAggregation,
              clip_percentile: clipPct,
              layer: layerArg,
            }),
          })
          setExpressionData(data)
          if ((data.genes?.length ?? 0) === 0 && (data.n_masked_excluded ?? 0) > 0) {
            setError(MESSAGES.geneMask.allMaskedInSetToast)
          }
        }

        // Mirror to other slot in dual mode. The other slot uses ITS OWN
        // displayLayer (read inside the helper), since layers are per-dataset.
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(
            otherSlot(activeSlot),
            selectedGenes,
            transform,
            {
              perGeneNorm: displayPreferences.geneSetPerGeneNorm,
              perGeneClip: displayPreferences.geneSetPerGeneClip,
              aggregation: displayPreferences.geneSetAggregation,
            },
            clipPct,
          )
        }
      } catch (err) {
        setError((err as Error).message)
        // Reset color mode so the app doesn't get stuck showing an error
        setExpressionData(null)
        useStore.getState().setColorMode('none')
      } finally {
        setLoading(false)
      }
    }

    fetchExpression()
  }, [
    displayPreferences.expressionTransform,
    displayPreferences.geneSetPerGeneNorm,
    displayPreferences.geneSetPerGeneClip,
    displayPreferences.geneSetAggregation,
    displayPreferences.clipPercentile,
    displayLayer,
  ]) // Re-run when any aggregation knob changes (including the source layer).
}

// Hook to re-fetch bivariate data when transform setting changes
export function useBivariateTransformEffect() {
  const {
    bivariateData,
    colorMode,
    displayPreferences,
    setBivariateData,
    setLoading,
    setError,
    layoutMode,
    activeSlot,
    displayLayer,
  } = useStore()

  useEffect(() => {
    // Only re-fetch if we're in bivariate mode and have data
    if (colorMode !== 'bivariate' || !bivariateData) {
      return
    }

    const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined
    const layerArg = displayLayer && displayLayer !== 'X' ? displayLayer : undefined
    const { genes1, genes2 } = bivariateData

    const fetchBivariate = async () => {
      setLoading(true)
      try {
        const clipPct = displayPreferences.clipPercentile
        const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes1,
            genes2,
            transform,
            per_gene_norm: displayPreferences.geneSetPerGeneNorm,
            per_gene_clip: displayPreferences.geneSetPerGeneClip,
            aggregation: displayPreferences.geneSetAggregation,
            clip_percentile: clipPct,
            layer: layerArg,
          }),
        })
        setBivariateData(data)

        // Mirror to other slot in dual mode
        if (layoutMode === 'dual') {
          mirrorBivariateToSlot(
            otherSlot(activeSlot),
            genes1,
            genes2,
            transform,
            {
              perGeneNorm: displayPreferences.geneSetPerGeneNorm,
              perGeneClip: displayPreferences.geneSetPerGeneClip,
              aggregation: displayPreferences.geneSetAggregation,
            },
            clipPct,
          )
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    }

    fetchBivariate()
  }, [
    displayPreferences.expressionTransform,
    displayPreferences.geneSetPerGeneNorm,
    displayPreferences.geneSetPerGeneClip,
    displayPreferences.geneSetAggregation,
    displayPreferences.clipPercentile,
    displayLayer,
  ]) // Re-run when any aggregation knob changes (including the source layer).
}

// Helper: fetch expression data for a specific slot and patch its DatasetState.
// Used to mirror gene coloring to the non-active plot in dual mode.
// Errors are silently ignored (gene may not exist in the other dataset).
// Aggregation knobs that the multi-gene & bivariate endpoints accept.
// Inlined as a small object to avoid threading three separate args through
// the mirror helpers and call sites.
type AggregationParams = {
  perGeneNorm?: string
  perGeneClip?: number
  aggregation?: string
}

async function mirrorExpressionToSlot(
  slot: DatasetSlot,
  genes: string[],
  transform?: string,
  agg?: AggregationParams,
  clipPercentile?: number,
) {
  const state = useStore.getState()
  const ds = state.datasets[slot]
  if (!ds.schema) return
  // Mirror uses the OTHER slot's own displayLayer — layers are per-dataset and
  // a layer name like "smoothed" may exist in one slot but not the other.
  const layerArg = ds.displayLayer && ds.displayLayer !== 'X' ? ds.displayLayer : undefined
  try {
    let data: ExpressionData
    if (genes.length === 1) {
      const qs = new URLSearchParams()
      if (transform) qs.set('transform', transform)
      if (clipPercentile && clipPercentile > 0) qs.set('clip_percentile', String(clipPercentile))
      if (layerArg) qs.set('layer', layerArg)
      const suffix = qs.toString()
      const url = suffix
        ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(genes[0])}?${suffix}`, slot)
        : appendDataset(`${API_BASE}/expression/${encodeURIComponent(genes[0])}`, slot)
      data = await fetchJson<ExpressionData>(url)
    } else {
      data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/multi`, slot), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genes,
          transform,
          per_gene_norm: agg?.perGeneNorm,
          per_gene_clip: agg?.perGeneClip,
          aggregation: agg?.aggregation,
          clip_percentile: clipPercentile ?? 1.0,
          layer: layerArg,
        }),
      })
    }
    // Compute sort order for the other slot
    const indices = Array.from({ length: ds.schema.n_cells }, (_, i) => i)
    indices.sort((a, b) => (data.values[a] ?? -Infinity) - (data.values[b] ?? -Infinity))
    useStore.getState().patchSlotState(slot, {
      expressionData: data,
      selectedGenes: genes,
      colorMode: 'expression',
      selectedColorColumn: null,
      bivariateData: null,
      cellSortOrder: indices,
      cellSortVersion: ds.cellSortVersion + 1,
    })
  } catch {
    // Gene may not exist in this dataset — ignore
  }
}

// Helper: fetch bivariate data for a specific slot and patch its DatasetState.
async function mirrorBivariateToSlot(
  slot: DatasetSlot,
  genes1: string[],
  genes2: string[],
  transform?: string,
  agg?: AggregationParams,
  clipPercentile?: number,
) {
  const state = useStore.getState()
  const ds = state.datasets[slot]
  if (!ds.schema) return
  const layerArg = ds.displayLayer && ds.displayLayer !== 'X' ? ds.displayLayer : undefined
  try {
    const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`, slot), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        genes1,
        genes2,
        transform,
        per_gene_norm: agg?.perGeneNorm,
        per_gene_clip: agg?.perGeneClip,
        aggregation: agg?.aggregation,
        clip_percentile: clipPercentile ?? 1.0,
        layer: layerArg,
      }),
    })
    const { values1, values2 } = data
    const indices = Array.from({ length: ds.schema.n_cells }, (_, i) => i)
    indices.sort((a, b) => ((values1[a] ?? 0) + (values2[a] ?? 0)) - ((values1[b] ?? 0) + (values2[b] ?? 0)))
    useStore.getState().patchSlotState(slot, {
      bivariateData: data,
      colorMode: 'bivariate',
      selectedColorColumn: null,
      expressionData: null,
      selectedGenes: [],
      cellSortOrder: indices,
      cellSortVersion: ds.cellSortVersion + 1,
    })
  } catch {
    // Genes may not exist in this dataset — ignore
  }
}

// Helper: get the other slot (for dual-mode mirroring)
function otherSlot(slot: DatasetSlot): DatasetSlot {
  return slot === 'primary' ? 'secondary' : 'primary'
}

export function useDataActions() {
  const {
    setSelectedEmbedding,
    setSelectedColorColumn,
    setColorMode,
    setExpressionData,
    setBivariateData,
    setSelectedGenes,
    setSelectedGeneSetName,
    clearSelectedGenes,
    clearBivariateMode,
    setLoading,
    setError,
    displayPreferences,
    layoutMode,
    activeSlot,
    patchSlotState,
    displayLayer,
  } = useStore()

  const selectEmbedding = useCallback(
    (name: string) => {
      setSelectedEmbedding(name)
    },
    [setSelectedEmbedding]
  )

  const selectColorColumn = useCallback(
    (name: string | null) => {
      setSelectedColorColumn(name)
      if (name) {
        setColorMode('metadata')
        setSelectedGenes([])
        setExpressionData(null)
      } else {
        setColorMode('none')
      }
    },
    [setSelectedColorColumn, setColorMode, setSelectedGenes, setExpressionData]
  )

  const colorByGene = useCallback(
    async (gene: string, transform?: string) => {
      setLoading(true)
      try {
        // Use provided transform or fall back to display preferences
        const effectiveTransform = transform ?? (displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined)
        const clipPct = displayPreferences.clipPercentile
        const layerArg = displayLayer && displayLayer !== 'X' ? displayLayer : undefined
        const qs = new URLSearchParams()
        if (effectiveTransform) qs.set('transform', effectiveTransform)
        if (clipPct > 0) qs.set('clip_percentile', String(clipPct))
        if (layerArg) qs.set('layer', layerArg)
        const suffix = qs.toString()
        const url = suffix
          ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(gene)}?${suffix}`)
          : appendDataset(`${API_BASE}/expression/${encodeURIComponent(gene)}`)
        const data = await fetchJson<ExpressionData>(url)
        setExpressionData(data)
        setSelectedGenes([gene])
        setSelectedGeneSetName(null)
        setColorMode('expression')
        setSelectedColorColumn(null)

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(otherSlot(activeSlot), [gene], effectiveTransform, undefined as AggregationParams | undefined, clipPct)
        }
      } catch (err) {
        setError((err as Error).message)
        // Reset so the app can recover — don't leave colorMode stuck on 'expression'
        setExpressionData(null)
        setColorMode('none')
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setSelectedGeneSetName, setColorMode, setSelectedColorColumn, setError, displayPreferences.expressionTransform, displayPreferences.clipPercentile, layoutMode, activeSlot, displayLayer]
  )

  const colorByGenes = useCallback(
    async (genes: string[], transform?: string, geneSetName?: string) => {
      if (genes.length === 0) {
        clearSelectedGenes()
        // Mirror clear to other slot in dual mode
        if (layoutMode === 'dual') {
          patchSlotState(otherSlot(activeSlot), {
            selectedGenes: [], selectedGeneSetName: null, expressionData: null, colorMode: 'none',
            cellSortOrder: null, cellSortVersion: 0,
          })
        }
        return
      }
      if (genes.length === 1) {
        return colorByGene(genes[0], transform)
      }

      setLoading(true)
      try {
        // Use provided transform or fall back to display preferences
        const effectiveTransform = transform ?? (displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined)
        const clipPct = displayPreferences.clipPercentile
        const layerArg = displayLayer && displayLayer !== 'X' ? displayLayer : undefined
        const data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/multi`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes,
            transform: effectiveTransform,
            per_gene_norm: displayPreferences.geneSetPerGeneNorm,
            per_gene_clip: displayPreferences.geneSetPerGeneClip,
            aggregation: displayPreferences.geneSetAggregation,
            clip_percentile: clipPct,
            layer: layerArg,
          }),
        })
        setExpressionData(data)
        setSelectedGenes(genes)
        setSelectedGeneSetName(geneSetName ?? null)
        setColorMode('expression')
        setSelectedColorColumn(null)

        if ((data.genes?.length ?? 0) === 0 && (data.n_masked_excluded ?? 0) > 0) {
          setError(MESSAGES.geneMask.allMaskedInSetToast)
        }

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(
            otherSlot(activeSlot),
            genes,
            effectiveTransform,
            {
              perGeneNorm: displayPreferences.geneSetPerGeneNorm,
              perGeneClip: displayPreferences.geneSetPerGeneClip,
              aggregation: displayPreferences.geneSetAggregation,
            },
            clipPct,
          )
        }
      } catch (err) {
        setError((err as Error).message)
        setExpressionData(null)
        setColorMode('none')
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setSelectedGeneSetName, setColorMode, setSelectedColorColumn, setError, clearSelectedGenes, colorByGene, displayPreferences.expressionTransform, displayPreferences.geneSetPerGeneNorm, displayPreferences.geneSetPerGeneClip, displayPreferences.geneSetAggregation, displayPreferences.clipPercentile, layoutMode, activeSlot, patchSlotState, displayLayer]
  )

  const clearExpressionColor = useCallback(() => {
    clearSelectedGenes()
    // Mirror clear to other slot in dual mode
    if (layoutMode === 'dual') {
      patchSlotState(otherSlot(activeSlot), {
        selectedGenes: [], expressionData: null, colorMode: 'none',
        cellSortOrder: null, cellSortVersion: 0,
      })
    }
  }, [clearSelectedGenes, layoutMode, activeSlot, patchSlotState])

  const colorByBivariate = useCallback(
    async (genes1: string[], genes2: string[]) => {
      if (genes1.length === 0 || genes2.length === 0) {
        return
      }

      setLoading(true)
      try {
        const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined
        const clipPct = displayPreferences.clipPercentile
        const layerArg = displayLayer && displayLayer !== 'X' ? displayLayer : undefined
        const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes1,
            genes2,
            transform,
            per_gene_norm: displayPreferences.geneSetPerGeneNorm,
            per_gene_clip: displayPreferences.geneSetPerGeneClip,
            aggregation: displayPreferences.geneSetAggregation,
            clip_percentile: clipPct,
            layer: layerArg,
          }),
        })
        setBivariateData(data)
        setColorMode('bivariate')
        setSelectedColorColumn(null)
        setExpressionData(null)
        setSelectedGenes([])

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorBivariateToSlot(
            otherSlot(activeSlot),
            genes1,
            genes2,
            transform,
            {
              perGeneNorm: displayPreferences.geneSetPerGeneNorm,
              perGeneClip: displayPreferences.geneSetPerGeneClip,
              aggregation: displayPreferences.geneSetAggregation,
            },
            clipPct,
          )
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setBivariateData, setColorMode, setSelectedColorColumn, setExpressionData, setSelectedGenes, setError, displayPreferences.expressionTransform, displayPreferences.geneSetPerGeneNorm, displayPreferences.geneSetPerGeneClip, displayPreferences.geneSetAggregation, displayPreferences.clipPercentile, layoutMode, activeSlot, displayLayer]
  )

  const clearBivariateColor = useCallback(() => {
    clearBivariateMode()
    // Mirror clear to other slot in dual mode
    if (layoutMode === 'dual') {
      patchSlotState(otherSlot(activeSlot), {
        bivariateData: null, colorMode: 'none',
        cellSortOrder: null, cellSortVersion: 0,
      })
    }
  }, [clearBivariateMode, layoutMode, activeSlot, patchSlotState])

  return {
    selectEmbedding,
    selectColorColumn,
    colorByGene,
    colorByGenes,
    clearExpressionColor,
    colorByBivariate,
    clearBivariateColor,
  }
}

// Hook for gene search
export function useGeneSearch() {
  const [results, setResults] = useState<string[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const searchGenes = useCallback(async (query: string) => {
    if (!query || query.length < 1) {
      setResults([])
      return
    }

    setIsSearching(true)
    try {
      const data = await fetchJson<{ genes: string[] }>(
        appendDataset(`${API_BASE}/genes/search?q=${encodeURIComponent(query)}&limit=20`)
      )
      setResults(data.genes)
    } catch (err) {
      console.error('Gene search failed:', err)
      setResults([])
    } finally {
      setIsSearching(false)
    }
  }, [])

  const clearResults = useCallback(() => {
    setResults([])
  }, [])

  return { results, isSearching, searchGenes, clearResults }
}

// Hook for paginated gene browsing
export interface GeneBrowseResult {
  genes: string[]
  offset: number
  limit: number
  total: number
}

export function useGeneBrowse(pageSize: number = 50) {
  const [page, setPage] = useState<GeneBrowseResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const fetchPage = useCallback(async (offset: number) => {
    setIsLoading(true)
    try {
      const data = await fetchJson<GeneBrowseResult>(
        appendDataset(`${API_BASE}/genes/browse?offset=${offset}&limit=${pageSize}`)
      )
      setPage(data)
    } catch (err) {
      console.error('Gene browse failed:', err)
    } finally {
      setIsLoading(false)
    }
  }, [pageSize])

  const reset = useCallback(() => {
    setPage(null)
  }, [])

  return { page, isLoading, fetchPage, reset }
}

// Fetch available var identifier columns for gene name switching
export async function fetchVarIdentifierColumns(slot?: DatasetSlot): Promise<void> {
  try {
    const data = await fetchJson<{ columns: string[]; current: string }>(
      appendDataset(`${API_BASE}/var/identifier_columns`, slot)
    )
    const store = useStore.getState()
    const targetSlot = slot ?? store.activeSlot
    store.patchSlotState(targetSlot, {
      varIdentifierColumns: data.columns,
      currentVarIndex: data.current,
    })
    // Also update flat fields if active slot
    if (targetSlot === store.activeSlot) {
      store.setVarIdentifierColumns(data.columns)
      store.setCurrentVarIndex(data.current)
    }
  } catch (err) {
    console.error('Failed to fetch var identifier columns:', err)
  }
}

// Swap the var index to a different column (changes gene identifiers)
export async function swapVarIndex(column: string, slot?: DatasetSlot): Promise<void> {
  const store = useStore.getState()
  const targetSlot = slot ?? store.activeSlot

  const data = await fetchJson<{
    schema: Schema
    old_genes: string[]
    new_genes: string[]
  }>(
    appendDataset(`${API_BASE}/var/swap_index`, targetSlot),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ column }),
    }
  )

  // Build positional old→new mapping
  const oldToNew = new Map<string, string>()
  for (let i = 0; i < data.old_genes.length; i++) {
    if (data.old_genes[i] !== data.new_genes[i]) {
      oldToNew.set(data.old_genes[i], data.new_genes[i])
    }
  }

  // Remap gene names in store (gene sets, selectedGenes)
  if (oldToNew.size > 0) {
    store.remapAllGeneNames(oldToNew)
  }

  // Update schema without resetting everything
  store.setSchema(data.schema)

  // Refresh identifier columns
  await fetchVarIdentifierColumns(targetSlot)

  // Mask may have been regenerated or cleared by the backend swap hook.
  await fetchGeneMask(targetSlot)

  // If coloring by expression, clear and let user re-select
  const currentState = useStore.getState()
  if (currentState.colorMode === 'expression' && currentState.selectedGenes.length > 0) {
    store.clearSelectedGenes()
  }
}

// -------- Gene Mask API helpers --------

interface GeneMaskResponse {
  active: boolean
  keep_columns: string[]
  hide_columns: string[]
  keep_combine_mode: 'or' | 'and'
  n_visible: number
  n_total: number
  visible_gene_names: string[] | null
}

function responseToConfig(data: GeneMaskResponse): GeneMaskConfig {
  return {
    active: data.active,
    keepColumns: data.keep_columns,
    hideColumns: data.hide_columns,
    keepCombineMode: data.keep_combine_mode,
    nVisible: data.n_visible,
    nTotal: data.n_total,
    visibleGeneNames: data.visible_gene_names,
  }
}

export async function fetchGeneMask(slot?: DatasetSlot): Promise<GeneMaskConfig> {
  const data = await fetchJson<GeneMaskResponse>(
    appendDataset(`${API_BASE}/gene_mask`, slot)
  )
  const config = responseToConfig(data)
  const store = useStore.getState()
  const targetSlot = slot ?? store.activeSlot
  store.patchSlotState(targetSlot, { geneMaskConfig: config })
  if (targetSlot === store.activeSlot) {
    store.setGeneMaskConfig(config)
  }
  return config
}

export interface GeneMaskApplyInput {
  keepColumns: string[]
  hideColumns: string[]
  keepCombineMode: 'or' | 'and'
}

function reactToMaskChange(targetSlot: DatasetSlot, config: GeneMaskConfig) {
  // If the currently-colored gene(s) became masked, clear coloring and
  // surface an error toast via setError (the existing dismissable toast).
  const store = useStore.getState()
  if (targetSlot !== store.activeSlot) return
  if (!config.active || !config.visibleGeneNames) return
  const visible = new Set(config.visibleGeneNames)

  // Single-gene / multi-gene expression coloring
  if (store.colorMode === 'expression' && store.selectedGenes.length > 0) {
    const masked = store.selectedGenes.filter((g) => !visible.has(g))
    if (masked.length === store.selectedGenes.length) {
      store.clearSelectedGenes()
      store.setError(MESSAGES.geneMask.coloringClearedToast(masked[0]))
      return
    }
  }

  // Bivariate coloring
  if (store.colorMode === 'bivariate' && store.bivariateData) {
    const g1Masked = store.bivariateData.genes1.some((g) => !visible.has(g))
    const g2Masked = store.bivariateData.genes2.some((g) => !visible.has(g))
    if (g1Masked || g2Masked) {
      store.clearBivariateMode()
      store.setError(
        MESSAGES.geneMask.coloringClearedToast(
          [...store.bivariateData.genes1, ...store.bivariateData.genes2].find(
            (g) => !visible.has(g)
          ) ?? 'gene'
        )
      )
    }
  }
}

export async function applyGeneMask(
  input: GeneMaskApplyInput,
  slot?: DatasetSlot
): Promise<GeneMaskConfig> {
  const store = useStore.getState()
  const targetSlot = slot ?? store.activeSlot

  const data = await fetchJson<GeneMaskResponse>(
    appendDataset(`${API_BASE}/gene_mask`, slot),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        keep_columns: input.keepColumns,
        hide_columns: input.hideColumns,
        keep_combine_mode: input.keepCombineMode,
      }),
    }
  )
  const config = responseToConfig(data)
  store.patchSlotState(targetSlot, { geneMaskConfig: config })
  if (targetSlot === store.activeSlot) {
    store.setGeneMaskConfig(config)
  }
  reactToMaskChange(targetSlot, config)
  return config
}

export async function clearGeneMask(slot?: DatasetSlot): Promise<GeneMaskConfig> {
  const data = await fetchJson<GeneMaskResponse>(
    appendDataset(`${API_BASE}/gene_mask`, slot),
    { method: 'DELETE' }
  )
  const config = responseToConfig(data)
  const store = useStore.getState()
  const targetSlot = slot ?? store.activeSlot
  store.patchSlotState(targetSlot, { geneMaskConfig: config })
  if (targetSlot === store.activeSlot) {
    store.setGeneMaskConfig(config)
  }
  return config
}

export interface PCALoadingGene {
  gene: string
  loading: number
}

export interface PCALoadingEntry {
  index: number                     // zero-based
  variance_ratio: number | null
  positive: PCALoadingGene[]
  negative: PCALoadingGene[]
}

export interface PCALoadingsResponse {
  n_pcs: number
  top_n: number
  n_genes_loaded: number
  n_genes_total: number
  pcs: PCALoadingEntry[]
}

export async function fetchPcaLoadings(
  topN: number,
  slot?: DatasetSlot,
): Promise<PCALoadingsResponse> {
  const url = appendDataset(`/api/scanpy/pca_loadings?top_n=${topN}`, slot)
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Failed to fetch PCA loadings')
  }
  return res.json()
}

export function usePcaLoadings(topN: number, enabled: boolean): {
  loadings: PCALoadingsResponse | null
  loading: boolean
  error: string | null
  reload: () => void
} {
  const activeSlot = useStore((s) => s.activeSlot)
  const [loadings, setLoadings] = useState<PCALoadingsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchPcaLoadings(topN, activeSlot)
      .then((data) => { if (!cancelled) setLoadings(data) })
      .catch((e) => { if (!cancelled) setError(e.message || 'Failed to fetch loadings') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [topN, enabled, activeSlot, reloadToken])

  const reload = useCallback(() => setReloadToken((n) => n + 1), [])
  return { loadings, loading, error, reload }
}

export async function fetchPcaSubsets(slot?: DatasetSlot): Promise<PCASubsetSummary[]> {
  const url = appendDataset('/api/scanpy/pca_subsets', slot)
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Failed to fetch PCA subsets')
  }
  const data = await res.json()
  const subsets: PCASubsetSummary[] = (data.subsets || []).map((s: any) => ({
    obsmKey: s.obsm_key,
    suffix: s.suffix,
    droppedPcs: s.dropped_pcs || [],
    nPcsKept: s.n_pcs_kept,
  }))
  // Write directly into the active dataset's store so the Neighbors dropdown
  // and PCA Loadings subsets list stay in sync.
  const targetSlot = slot ?? useStore.getState().activeSlot
  useStore.getState().patchSlotState(targetSlot, { pcaSubsets: subsets })
  return subsets
}

export async function createPcaSubset(
  dropPcIndices: number[],
  suffix: string | null,
  slot?: DatasetSlot,
): Promise<PCASubsetSummary> {
  const url = appendDataset('/api/scanpy/pca_subsets', slot)
  const body: { drop_pc_indices: number[]; suffix?: string } = {
    drop_pc_indices: dropPcIndices,
  }
  if (suffix && suffix.trim() !== '') body.suffix = suffix.trim()

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    const err = new Error(detail.detail || 'Failed to create PC subset') as Error & { status?: number }
    err.status = res.status
    throw err
  }
  const data = await res.json()
  const summary: PCASubsetSummary = {
    obsmKey: data.obsm_key,
    suffix: data.suffix,
    droppedPcs: data.dropped_pcs || [],
    nPcsKept: data.n_pcs_kept,
  }
  // Optimistically append to the store.
  const targetSlot = slot ?? useStore.getState().activeSlot
  const current = useStore.getState().datasets[targetSlot]?.pcaSubsets || []
  useStore.getState().patchSlotState(targetSlot, {
    pcaSubsets: [...current, summary],
  })
  return summary
}

export async function deletePcaSubset(
  obsmKey: string,
  slot?: DatasetSlot,
): Promise<void> {
  const url = appendDataset(`/api/scanpy/pca_subsets/${encodeURIComponent(obsmKey)}`, slot)
  const res = await fetch(url, { method: 'DELETE' })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    const err = new Error(detail.detail || 'Failed to delete PC subset') as Error & { status?: number }
    err.status = res.status
    throw err
  }
  const targetSlot = slot ?? useStore.getState().activeSlot
  const current = useStore.getState().datasets[targetSlot]?.pcaSubsets || []
  useStore.getState().patchSlotState(targetSlot, {
    pcaSubsets: current.filter((s) => s.obsmKey !== obsmKey),
  })
}

export interface BooleanColumnValuesResponse {
  n_genes: number
  columns: Record<string, number[]>
}

export async function fetchBooleanColumnValues(
  slot?: DatasetSlot
): Promise<BooleanColumnValuesResponse> {
  return fetchJson<BooleanColumnValuesResponse>(
    appendDataset(`${API_BASE}/var/boolean_column_values`, slot)
  )
}

// Types for obs summaries
export interface CategoryValue {
  value: string
  count: number
}

export interface ObsSummary {
  name: string
  dtype: 'category' | 'numeric' | 'string'
  categories?: CategoryValue[]
  min?: number
  max?: number
  mean?: number
}

// Hook for fetching all obs summaries
export function useObsSummaries() {
  const [summaries, setSummaries] = useState<ObsSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refreshCounter, setRefreshCounter] = useState(0)
  const obsSummariesVersion = useStore((s) => s.obsSummariesVersion)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)

  useEffect(() => {
    if (!schema) return // No dataset loaded yet
    setIsLoading(true)
    fetchJson<ObsSummary[]>(appendDataset(`${API_BASE}/obs/summaries`))
      .then(setSummaries)
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false))
  }, [refreshCounter, obsSummariesVersion, activeSlot, schema])

  const refresh = useCallback(() => {
    setRefreshCounter((c) => c + 1)
  }, [])

  return { summaries, isLoading, error, refresh }
}

// Annotation API functions
export async function createAnnotation(name: string, defaultValue: string = 'unassigned', slot?: DatasetSlot): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(appendDataset(`${API_BASE}/annotations`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, default_value: defaultValue }),
  })
}

export async function addLabelToAnnotation(annotationName: string, label: string, slot?: DatasetSlot): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(appendDataset(`${API_BASE}/annotations/${encodeURIComponent(annotationName)}/labels`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label }),
  })
}

export async function labelCells(
  annotationName: string,
  label: string,
  cellIndices: number[],
  slot?: DatasetSlot
): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(appendDataset(`${API_BASE}/annotations/${encodeURIComponent(annotationName)}/label-cells`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, cell_indices: cellIndices }),
  })
}

export async function deleteAnnotation(name: string, slot?: DatasetSlot): Promise<void> {
  await fetchJson<{ status: string }>(appendDataset(`${API_BASE}/annotations/${encodeURIComponent(name)}`, slot), {
    method: 'DELETE',
  })
}

export async function exportAnnotations(columns?: string[], slot?: DatasetSlot): Promise<string> {
  const response = await fetch(appendDataset(`${API_BASE}/annotations/export`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ columns: columns || null }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.text()
}

export interface RenameObsLabelResult {
  column: string
  old_label: string
  new_label: string
  n_cells_renamed: number
}

export async function renameObsLabel(
  column: string,
  oldLabel: string,
  newLabel: string,
  slot?: DatasetSlot
): Promise<RenameObsLabelResult> {
  return fetchJson<RenameObsLabelResult>(
    appendDataset(`${API_BASE}/obs/${encodeURIComponent(column)}/rename_label`, slot),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_label: oldLabel, new_label: newLabel }),
    }
  )
}

export interface MergeObsLabelsResult {
  column: string
  merged_labels: string[]
  new_label: string
  n_cells_merged: number
}

export async function mergeObsLabels(
  column: string,
  labels: string[],
  newLabel: string,
  slot?: DatasetSlot
): Promise<MergeObsLabelsResult> {
  return fetchJson<MergeObsLabelsResult>(
    appendDataset(`${API_BASE}/obs/${encodeURIComponent(column)}/merge_labels`, slot),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels, new_label: newLabel }),
    }
  )
}

// Differential expression parameters
export interface DiffExpParams {
  method?: string
  corrMethod?: string
  minFoldChange?: number | null
  minInGroupFraction?: number | null
  maxOutGroupFraction?: number | null
  maxPvalAdj?: number | null
  geneSubset?: string | null
}

// Differential expression API function
export async function runDiffExp(
  group1: number[],
  group2: number[],
  topN: number = 25,
  slot?: DatasetSlot,
  params?: DiffExpParams
): Promise<DiffExpResult> {
  const body: Record<string, unknown> = { group1, group2, top_n: topN }
  if (params?.method) body.method = params.method
  if (params?.corrMethod) body.corr_method = params.corrMethod
  if (params?.minFoldChange != null) body.min_fold_change = params.minFoldChange
  if (params?.minInGroupFraction != null) body.min_in_group_fraction = params.minInGroupFraction
  if (params?.maxOutGroupFraction != null) body.max_out_group_fraction = params.maxOutGroupFraction
  if (params?.maxPvalAdj != null) body.max_pval_adj = params.maxPvalAdj
  if (params?.geneSubset) body.gene_subset = params.geneSubset
  return fetchJson<DiffExpResult>(appendDataset(`${API_BASE}/diffexp`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// Hook for differential expression
export function useDiffExp() {
  const {
    comparison,
    diffExpResult,
    isDiffExpLoading,
    setDiffExpResult,
    setDiffExpLoading,
    setDiffExpModalOpen,
    activeCellMask,
  } = useStore()

  const runComparison = useCallback(async (topN: number = 25, params?: DiffExpParams) => {
    if (!comparison.group1 || !comparison.group2) {
      throw new Error('Both groups must be set before running comparison')
    }

    // Filter out masked cells if mask is active
    let group1 = comparison.group1
    let group2 = comparison.group2

    if (activeCellMask) {
      group1 = group1.filter((i) => activeCellMask[i])
      group2 = group2.filter((i) => activeCellMask[i])

      if (group1.length === 0 || group2.length === 0) {
        throw new Error('After filtering masked cells, one or both groups are empty')
      }
    }

    setDiffExpLoading(true)
    try {
      const result = await runDiffExp(group1, group2, topN, undefined, params)
      setDiffExpResult(result)
      setDiffExpModalOpen(true)
      return result
    } catch (err) {
      setDiffExpResult(null)
      throw err
    } finally {
      setDiffExpLoading(false)
    }
  }, [comparison.group1, comparison.group2, activeCellMask, setDiffExpLoading, setDiffExpResult, setDiffExpModalOpen])

  return {
    comparison,
    diffExpResult,
    isDiffExpLoading,
    runComparison,
  }
}

// Line association API function
export interface LineAssociationParams {
  lineName: string
  cellIndices?: number[]
  geneSubset?: string | string[] | { columns: string[]; operation: string } | null
  testVariable?: 'position' | 'distance'
  nSplineKnots?: number
  minCells?: number
  fdrThreshold?: number
  topN?: number
  clusterGenes?: boolean
}

// Sync lines to backend
async function syncLinesToBackend(lines: { name: string; embeddingName: string; points: [number, number][]; smoothedPoints: [number, number][] | null }[], slot?: DatasetSlot) {
  const payload = lines.map((line) => ({
    name: line.name,
    embeddingName: line.embeddingName,
    points: line.points,
    smoothedPoints: line.smoothedPoints,
  }))
  await fetchJson(appendDataset(`${API_BASE}/lines`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lines: payload }),
  })
}

export async function runLineAssociation(params: LineAssociationParams, slot?: DatasetSlot): Promise<LineAssociationResult> {
  const response = await fetch(appendDataset(`${API_BASE}/lines/association`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      line_name: params.lineName,
      cell_indices: params.cellIndices,
      gene_subset: params.geneSubset ?? null,
      test_variable: params.testVariable ?? 'position',
      n_spline_knots: params.nSplineKnots ?? 5,
      min_cells: params.minCells ?? 20,
      fdr_threshold: params.fdrThreshold ?? 0.05,
      top_n: params.topN ?? 50,
      cluster_genes: params.clusterGenes ?? false,
    }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  const data = await response.json()

  if (data.task_id && data.status === 'running') {
    const { setActiveTaskId } = useStore.getState()
    setActiveTaskId(data.task_id)
    try {
      const taskResult = await pollTask(data.task_id, slot)
      if (taskResult.status === 'cancelled') {
        throw new Error('cancelled')
      }
      if (taskResult.status === 'error') {
        throw new Error(taskResult.error || 'Task failed')
      }
      return taskResult.result as unknown as LineAssociationResult
    } finally {
      setActiveTaskId(null)
    }
  }

  return data as LineAssociationResult
}

// Multi-line association types and API function
export interface MultiLineEntry {
  name: string
  cellIndices: number[]
  reversed: boolean
}

export interface MultiLineAssociationParams {
  lines: MultiLineEntry[]
  geneSubset?: string | string[] | { columns: string[]; operation: string } | null
  testVariable?: 'position' | 'distance'
  nSplineKnots?: number
  minCells?: number
  fdrThreshold?: number
  topN?: number
  clusterGenes?: boolean
}

export async function runMultiLineAssociation(params: MultiLineAssociationParams, slot?: DatasetSlot): Promise<LineAssociationResult> {
  const response = await fetch(appendDataset(`${API_BASE}/lines/multi-association`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      lines: params.lines.map((l) => ({
        name: l.name,
        cell_indices: l.cellIndices,
        reversed: l.reversed,
      })),
      gene_subset: params.geneSubset ?? null,
      test_variable: params.testVariable ?? 'position',
      n_spline_knots: params.nSplineKnots ?? 5,
      min_cells: params.minCells ?? 20,
      fdr_threshold: params.fdrThreshold ?? 0.05,
      top_n: params.topN ?? 50,
      cluster_genes: params.clusterGenes ?? false,
    }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  const data = await response.json()

  if (data.task_id && data.status === 'running') {
    const { setActiveTaskId } = useStore.getState()
    setActiveTaskId(data.task_id)
    try {
      const taskResult = await pollTask(data.task_id, slot)
      if (taskResult.status === 'cancelled') {
        throw new Error('cancelled')
      }
      if (taskResult.status === 'error') {
        throw new Error(taskResult.error || 'Task failed')
      }
      return taskResult.result as unknown as LineAssociationResult
    } finally {
      setActiveTaskId(null)
    }
  }

  return data as LineAssociationResult
}

// Hook for line association testing
export function useLineAssociation() {
  const {
    drawnLines,
    lineAssociationResult,
    isLineAssociationLoading,
    setLineAssociationResult,
    setLineAssociationLoading,
    setLineAssociationModalOpen,
  } = useStore()

  const runAssociation = useCallback(async (lineName: string, params?: Partial<LineAssociationParams>) => {
    setLineAssociationLoading(true)
    try {
      // First sync all lines to the backend
      await syncLinesToBackend(drawnLines)

      // Then run the association test
      const result = await runLineAssociation({
        lineName,
        ...params,
      })
      setLineAssociationResult(result)
      setLineAssociationModalOpen(true)
      return result
    } catch (err) {
      const message = (err as Error).message
      if (message === 'cancelled') {
        return null
      }
      setLineAssociationResult(null)
      throw err
    } finally {
      setLineAssociationLoading(false)
    }
  }, [drawnLines, setLineAssociationLoading, setLineAssociationResult, setLineAssociationModalOpen])

  const runMultiAssociation = useCallback(async (params: MultiLineAssociationParams) => {
    setLineAssociationLoading(true)
    try {
      await syncLinesToBackend(drawnLines)
      const result = await runMultiLineAssociation(params)
      setLineAssociationResult(result)
      setLineAssociationModalOpen(true)
      return result
    } catch (err) {
      const message = (err as Error).message
      if (message === 'cancelled') {
        return null
      }
      setLineAssociationResult(null)
      throw err
    } finally {
      setLineAssociationLoading(false)
    }
  }, [drawnLines, setLineAssociationLoading, setLineAssociationResult, setLineAssociationModalOpen])

  return {
    drawnLines,
    lineAssociationResult,
    isLineAssociationLoading,
    runAssociation,
    runMultiAssociation,
  }
}

// Marker genes (one-vs-rest) API types and function
export interface MarkerGeneEntry {
  gene: string
  log2fc: number
  pval: number
  pval_adj: number
}

export interface MarkerGenesGroupResult {
  group: string
  genes: MarkerGeneEntry[]
}

export interface MarkerGenesResponse {
  obs_column: string
  results: MarkerGenesGroupResult[]
}

export async function runMarkerGenes(params: {
  obs_column: string
  groups?: string[]
  top_n?: number
  min_in_group_fraction?: number
  max_out_group_fraction?: number
  min_fold_change?: number
  gene_subset?: string | null
}, slot?: DatasetSlot): Promise<MarkerGenesResponse> {
  return fetchJson<MarkerGenesResponse>(appendDataset(`${API_BASE}/marker-genes`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
}

// Create projection embedding API function
export interface CreateLineEmbeddingParams {
  lineName: string
  cellIndices?: number[]
}

export interface CreateLineEmbeddingResult {
  embedding_name: string
  n_cells: number
  position_range: [number, number]
  distance_range: [number, number]
}

export async function createLineEmbedding(
  params: CreateLineEmbeddingParams,
  lines: { name: string; embeddingName: string; points: [number, number][]; smoothedPoints: [number, number][] | null }[],
  slot?: DatasetSlot
): Promise<CreateLineEmbeddingResult> {
  // First sync lines to backend
  const payload = lines.map((line) => ({
    name: line.name,
    embeddingName: line.embeddingName,
    points: line.points,
    smoothedPoints: line.smoothedPoints,
  }))
  await fetchJson(appendDataset(`${API_BASE}/lines`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lines: payload }),
  })

  // Then create the embedding
  return fetchJson<CreateLineEmbeddingResult>(appendDataset(`${API_BASE}/lines/create-embedding`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      line_name: params.lineName,
      cell_indices: params.cellIndices,
    }),
  })
}

export async function runClusterGeneSet(
  params: {
    geneNames: string[]
    method: 'hierarchical' | 'kmeans' | 'dbscan'
    /** Required for hierarchical / kmeans; ignored for dbscan. */
    k?: number
    cellContext: 'all' | 'selection' | 'annotation'
    cellIndices?: number[]
    annotationColumn?: string
    annotationValues?: string[]
    /** DBSCAN only: neighbor radius in correlation-distance space (0–2). */
    eps?: number
    /** DBSCAN only: minimum genes within `eps` for a gene to be a core point. */
    minSamples?: number
    /** Source matrix to read expression from. Undefined / 'X' → adata.X. */
    layer?: string
  },
  slot?: DatasetSlot,
): Promise<{ clusters: string[][] }> {
  const response = await fetch(appendDataset(`${API_BASE}/cluster_gene_set`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gene_names: params.geneNames,
      method: params.method,
      k: params.k,
      cell_context: params.cellContext,
      cell_indices: params.cellIndices,
      annotation_column: params.annotationColumn,
      annotation_values: params.annotationValues,
      eps: params.eps,
      min_samples: params.minSamples,
      layer: params.layer,
    }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || 'Cluster gene set failed')
  }
  return response.json()
}
