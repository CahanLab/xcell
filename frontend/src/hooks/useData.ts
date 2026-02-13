import { useEffect, useCallback, useState } from 'react'
import { useStore, Schema, EmbeddingData, ObsColumnData, ExpressionData, BivariateExpressionData, DiffExpResult, LineAssociationResult } from '../store'

const API_BASE = '/api'

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
    fetchJson<Schema>(`${API_BASE}/schema`)
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
    fetchJson<EmbeddingData>(`${API_BASE}/embedding/${selectedEmbedding}`)
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
    fetchJson<ObsColumnData>(`${API_BASE}/obs/${selectedColorColumn}`)
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
            ? `${API_BASE}/expression/${encodeURIComponent(selectedGenes[0])}?transform=${transform}`
            : `${API_BASE}/expression/${encodeURIComponent(selectedGenes[0])}`
          const data = await fetchJson<ExpressionData>(url)
          setExpressionData(data)
        } else {
          const data = await fetchJson<ExpressionData>(`${API_BASE}/expression/multi`, {
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
        const data = await fetchJson<BivariateExpressionData>(`${API_BASE}/expression/bivariate`, {
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
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    }

    fetchBivariate()
  }, [displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod]) // Re-run when transform or scoring method changes
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
          ? `${API_BASE}/expression/${encodeURIComponent(gene)}?transform=${effectiveTransform}`
          : `${API_BASE}/expression/${encodeURIComponent(gene)}`
        const data = await fetchJson<ExpressionData>(url)
        setExpressionData(data)
        setSelectedGenes([gene])
        setColorMode('expression')
        setSelectedColorColumn(null)
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError, displayPreferences.expressionTransform]
  )

  const colorByGenes = useCallback(
    async (genes: string[], transform?: string) => {
      if (genes.length === 0) {
        clearSelectedGenes()
        return
      }
      if (genes.length === 1) {
        return colorByGene(genes[0], transform)
      }

      setLoading(true)
      try {
        // Use provided transform or fall back to display preferences
        const effectiveTransform = transform ?? (displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined)
        const data = await fetchJson<ExpressionData>(`${API_BASE}/expression/multi`, {
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
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError, clearSelectedGenes, colorByGene, displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod]
  )

  const clearExpressionColor = useCallback(() => {
    clearSelectedGenes()
  }, [clearSelectedGenes])

  const colorByBivariate = useCallback(
    async (genes1: string[], genes2: string[]) => {
      if (genes1.length === 0 || genes2.length === 0) {
        return
      }

      setLoading(true)
      try {
        const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : undefined
        const data = await fetchJson<BivariateExpressionData>(`${API_BASE}/expression/bivariate`, {
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
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [setLoading, setBivariateData, setColorMode, setSelectedColorColumn, setExpressionData, setSelectedGenes, setError, displayPreferences.expressionTransform, displayPreferences.geneSetScoringMethod]
  )

  const clearBivariateColor = useCallback(() => {
    clearBivariateMode()
  }, [clearBivariateMode])

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
        `${API_BASE}/genes/search?q=${encodeURIComponent(query)}&limit=20`
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
        `${API_BASE}/genes/browse?offset=${offset}&limit=${pageSize}`
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
    fetchJson<ObsSummary[]>(`${API_BASE}/obs/summaries`)
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
export async function createAnnotation(name: string, defaultValue: string = 'unassigned'): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(`${API_BASE}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, default_value: defaultValue }),
  })
}

export async function addLabelToAnnotation(annotationName: string, label: string): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(`${API_BASE}/annotations/${encodeURIComponent(annotationName)}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label }),
  })
}

export async function labelCells(
  annotationName: string,
  label: string,
  cellIndices: number[]
): Promise<ObsSummary> {
  return fetchJson<ObsSummary>(`${API_BASE}/annotations/${encodeURIComponent(annotationName)}/label-cells`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, cell_indices: cellIndices }),
  })
}

export async function deleteAnnotation(name: string): Promise<void> {
  await fetchJson<{ status: string }>(`${API_BASE}/annotations/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
}

export async function exportAnnotations(columns?: string[]): Promise<string> {
  const response = await fetch(`${API_BASE}/annotations/export`, {
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
  topN: number = 25
): Promise<DiffExpResult> {
  return fetchJson<DiffExpResult>(`${API_BASE}/diffexp`, {
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
async function syncLinesToBackend(lines: { name: string; embeddingName: string; points: [number, number][]; smoothedPoints: [number, number][] | null }[]) {
  const payload = lines.map((line) => ({
    name: line.name,
    embeddingName: line.embeddingName,
    points: line.points,
    smoothedPoints: line.smoothedPoints,
  }))
  await fetchJson(`${API_BASE}/lines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lines: payload }),
  })
}

export async function runLineAssociation(params: LineAssociationParams): Promise<LineAssociationResult> {
  return fetchJson<LineAssociationResult>(`${API_BASE}/lines/association`, {
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
  lines: { name: string; embeddingName: string; points: [number, number][]; smoothedPoints: [number, number][] | null }[]
): Promise<CreateLineEmbeddingResult> {
  // First sync lines to backend
  const payload = lines.map((line) => ({
    name: line.name,
    embeddingName: line.embeddingName,
    points: line.points,
    smoothedPoints: line.smoothedPoints,
  }))
  await fetchJson(`${API_BASE}/lines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lines: payload }),
  })

  // Then create the embedding
  return fetchJson<CreateLineEmbeddingResult>(`${API_BASE}/lines/create-embedding`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      line_name: params.lineName,
      cell_indices: params.cellIndices,
    }),
  })
}
