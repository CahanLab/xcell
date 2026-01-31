import { create } from 'zustand'

export interface Schema {
  n_cells: number
  n_genes: number
  embeddings: string[]
  obs_columns: string[]
  obs_dtypes: Record<string, string>
}

export interface EmbeddingData {
  name: string
  coordinates: [number, number][]
}

export interface ObsColumnData {
  name: string
  values: (number | string | null)[]
  dtype: 'category' | 'numeric' | 'string'
  categories?: string[]
}

interface AppState {
  // Data
  schema: Schema | null
  embedding: EmbeddingData | null
  colorBy: ObsColumnData | null

  // UI State
  selectedEmbedding: string | null
  selectedColorColumn: string | null
  isLoading: boolean
  error: string | null

  // Actions
  setSchema: (schema: Schema) => void
  setEmbedding: (embedding: EmbeddingData) => void
  setColorBy: (colorBy: ObsColumnData | null) => void
  setSelectedEmbedding: (name: string) => void
  setSelectedColorColumn: (name: string | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  schema: null,
  embedding: null,
  colorBy: null,
  selectedEmbedding: null,
  selectedColorColumn: null,
  isLoading: false,
  error: null,

  // Actions
  setSchema: (schema) => set({ schema }),
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy }),
  setSelectedEmbedding: (name) => set({ selectedEmbedding: name }),
  setSelectedColorColumn: (name) => set({ selectedColorColumn: name }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}))
