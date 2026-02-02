import { useEffect, useCallback, useState } from 'react'
import { useStore, Schema, EmbeddingData, ObsColumnData, ExpressionData, DiffExpResult } from '../store'

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
        // Auto-select first embedding
        if (data.embeddings.length > 0) {
          setSelectedEmbedding(data.embeddings[0])
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

export function useDataActions() {
  const {
    setSelectedEmbedding,
    setSelectedColorColumn,
    setColorMode,
    setExpressionData,
    setSelectedGenes,
    clearSelectedGenes,
    setLoading,
    setError,
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
    async (gene: string) => {
      setLoading(true)
      try {
        const data = await fetchJson<ExpressionData>(`${API_BASE}/expression/${encodeURIComponent(gene)}`)
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
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError]
  )

  const colorByGenes = useCallback(
    async (genes: string[]) => {
      if (genes.length === 0) {
        clearSelectedGenes()
        return
      }
      if (genes.length === 1) {
        return colorByGene(genes[0])
      }

      setLoading(true)
      try {
        const data = await fetchJson<ExpressionData>(`${API_BASE}/expression/multi`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(genes),
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
    [setLoading, setExpressionData, setSelectedGenes, setColorMode, setSelectedColorColumn, setError, clearSelectedGenes, colorByGene]
  )

  const clearExpressionColor = useCallback(() => {
    clearSelectedGenes()
  }, [clearSelectedGenes])

  return {
    selectEmbedding,
    selectColorColumn,
    colorByGene,
    colorByGenes,
    clearExpressionColor,
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

  useEffect(() => {
    setIsLoading(true)
    fetchJson<ObsSummary[]>(`${API_BASE}/obs/summaries`)
      .then(setSummaries)
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false))
  }, [refreshCounter])

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
