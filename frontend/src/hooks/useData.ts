import { useEffect, useCallback, useState } from 'react'
import { useStore, DatasetSlot, Schema, EmbeddingData, ObsColumnData, ExpressionData, BivariateExpressionData, DiffExpResult, LineAssociationResult } from '../store'

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
  } = useStore()

  useEffect(() => {
    // Only re-fetch if we're in expression mode and have genes selected
    if (colorMode !== 'expression' || selectedGenes.length === 0) {
      return
    }

    const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined

    const fetchExpression = async () => {
      setLoading(true)
      try {
        if (selectedGenes.length === 1) {
          const url = transform
            ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(selectedGenes[0])}?transform=${transform}`)
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
              scoring_method: displayPreferences.geneSetScoringMethod,
            }),
          })
          setExpressionData(data)
        }

        // Mirror to other slot in dual mode
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(otherSlot(activeSlot), selectedGenes, transform, displayPreferences.geneSetScoringMethod)
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    }

    fetchExpression()
  }, [displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod]) // Re-run when transform or scoring method changes
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
  } = useStore()

  useEffect(() => {
    // Only re-fetch if we're in bivariate mode and have data
    if (colorMode !== 'bivariate' || !bivariateData) {
      return
    }

    const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined
    const { genes1, genes2 } = bivariateData

    const fetchBivariate = async () => {
      setLoading(true)
      try {
        const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes1,
            genes2,
            transform,
            clip_percentile: 1.0,
            scoring_method: displayPreferences.geneSetScoringMethod,
          }),
        })
        setBivariateData(data)

        // Mirror to other slot in dual mode
        if (layoutMode === 'dual') {
          mirrorBivariateToSlot(otherSlot(activeSlot), genes1, genes2, transform, displayPreferences.geneSetScoringMethod)
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    }

    fetchBivariate()
  }, [displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod]) // Re-run when transform or scoring method changes
}

// Helper: fetch expression data for a specific slot and patch its DatasetState.
// Used to mirror gene coloring to the non-active plot in dual mode.
// Errors are silently ignored (gene may not exist in the other dataset).
async function mirrorExpressionToSlot(
  slot: DatasetSlot,
  genes: string[],
  transform?: string,
  scoringMethod?: string,
) {
  const state = useStore.getState()
  const ds = state.datasets[slot]
  if (!ds.schema) return
  try {
    let data: ExpressionData
    if (genes.length === 1) {
      const url = transform
        ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(genes[0])}?transform=${transform}`, slot)
        : appendDataset(`${API_BASE}/expression/${encodeURIComponent(genes[0])}`, slot)
      data = await fetchJson<ExpressionData>(url)
    } else {
      data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/multi`, slot), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genes, transform, scoring_method: scoringMethod }),
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
  scoringMethod?: string,
) {
  const state = useStore.getState()
  const ds = state.datasets[slot]
  if (!ds.schema) return
  try {
    const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`, slot), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genes1, genes2, transform, clip_percentile: 1.0, scoring_method: scoringMethod }),
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
    clearSelectedGenes,
    clearBivariateMode,
    setLoading,
    setError,
    displayPreferences,
    layoutMode,
    activeSlot,
    patchSlotState,
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
        const url = effectiveTransform
          ? appendDataset(`${API_BASE}/expression/${encodeURIComponent(gene)}?transform=${effectiveTransform}`)
          : appendDataset(`${API_BASE}/expression/${encodeURIComponent(gene)}`)
        const data = await fetchJson<ExpressionData>(url)
        setExpressionData(data)
        setSelectedGenes([gene])
        setColorMode('expression')
        setSelectedColorColumn(null)

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(otherSlot(activeSlot), [gene], effectiveTransform)
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError, displayPreferences.expressionTransform, layoutMode, activeSlot]
  )

  const colorByGenes = useCallback(
    async (genes: string[], transform?: string) => {
      if (genes.length === 0) {
        clearSelectedGenes()
        // Mirror clear to other slot in dual mode
        if (layoutMode === 'dual') {
          patchSlotState(otherSlot(activeSlot), {
            selectedGenes: [], expressionData: null, colorMode: 'none',
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
        const data = await fetchJson<ExpressionData>(appendDataset(`${API_BASE}/expression/multi`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes,
            transform: effectiveTransform,
            scoring_method: displayPreferences.geneSetScoringMethod,
          }),
        })
        setExpressionData(data)
        setSelectedGenes(genes)
        setColorMode('expression')
        setSelectedColorColumn(null)

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorExpressionToSlot(otherSlot(activeSlot), genes, effectiveTransform, displayPreferences.geneSetScoringMethod)
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError, clearSelectedGenes, colorByGene, displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod, layoutMode, activeSlot, patchSlotState]
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
        const data = await fetchJson<BivariateExpressionData>(appendDataset(`${API_BASE}/expression/bivariate`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            genes1,
            genes2,
            transform,
            clip_percentile: 1.0,
            scoring_method: displayPreferences.geneSetScoringMethod,
          }),
        })
        setBivariateData(data)
        setColorMode('bivariate')
        setSelectedColorColumn(null)
        setExpressionData(null)
        setSelectedGenes([])

        // Mirror to other slot in dual mode (fire-and-forget)
        if (layoutMode === 'dual') {
          mirrorBivariateToSlot(otherSlot(activeSlot), genes1, genes2, transform, displayPreferences.geneSetScoringMethod)
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setBivariateData, setColorMode, setSelectedColorColumn, setExpressionData, setSelectedGenes, setError, displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod, layoutMode, activeSlot]
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

  useEffect(() => {
    setIsLoading(true)
    fetchJson<ObsSummary[]>(appendDataset(`${API_BASE}/obs/summaries`))
      .then(setSummaries)
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false))
  }, [refreshCounter, obsSummariesVersion])

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

// Differential expression API function
export async function runDiffExp(
  group1: number[],
  group2: number[],
  topN: number = 25,
  slot?: DatasetSlot
): Promise<DiffExpResult> {
  return fetchJson<DiffExpResult>(appendDataset(`${API_BASE}/diffexp`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group1, group2, top_n: topN }),
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

  const runComparison = useCallback(async (topN: number = 25) => {
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
      const result = await runDiffExp(group1, group2, topN)
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
  return fetchJson<LineAssociationResult>(appendDataset(`${API_BASE}/lines/association`, slot), {
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
    }),
  })
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
