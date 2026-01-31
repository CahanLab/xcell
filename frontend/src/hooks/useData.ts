import { useEffect, useCallback } from 'react'
import { useStore, Schema, EmbeddingData, ObsColumnData } from '../store'

const API_BASE = '/api'

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
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
  const { selectedColorColumn, colorBy, setColorBy, setLoading, setError } = useStore()

  useEffect(() => {
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
  }, [selectedColorColumn, colorBy, setColorBy, setLoading, setError])

  return colorBy
}

export function useDataActions() {
  const { setSelectedEmbedding, setSelectedColorColumn } = useStore()

  const selectEmbedding = useCallback(
    (name: string) => {
      setSelectedEmbedding(name)
    },
    [setSelectedEmbedding]
  )

  const selectColorColumn = useCallback(
    (name: string | null) => {
      setSelectedColorColumn(name)
    },
    [setSelectedColorColumn]
  )

  return { selectEmbedding, selectColorColumn }
}
