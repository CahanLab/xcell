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

export interface ExpressionData {
  gene?: string
  genes?: string[]
  values: (number | null)[]
  min: number
  max: number
}

export interface GeneSet {
  name: string
  genes: string[]
}

// Color mode: what determines cell colors
export type ColorMode = 'none' | 'metadata' | 'expression'

// Interaction mode for the scatter plot
export type InteractionMode = 'pan' | 'lasso'

// Color scale options for expression data
export type ColorScale = 'viridis' | 'plasma' | 'magma' | 'inferno' | 'cividis' | 'coolwarm' | 'blues' | 'reds'

// Display preferences
export interface DisplayPreferences {
  pointSize: number  // Base point size (1-10)
  backgroundColor: string  // Hex color
  colorScale: ColorScale  // Color scale for expression data
  pointOpacity: number  // 0-1
}

interface AppState {
  // Data
  schema: Schema | null
  embedding: EmbeddingData | null
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null

  // Gene management
  geneSets: GeneSet[]
  selectedGenes: string[]  // Currently selected genes for expression coloring

  // Cell selection
  selectedCellIndices: number[]  // Indices of selected cells
  interactionMode: InteractionMode  // Current interaction mode

  // UI State
  selectedEmbedding: string | null
  selectedColorColumn: string | null
  colorMode: ColorMode
  isLoading: boolean
  error: string | null

  // Display preferences
  displayPreferences: DisplayPreferences

  // Actions
  setSchema: (schema: Schema) => void
  setEmbedding: (embedding: EmbeddingData) => void
  setColorBy: (colorBy: ObsColumnData | null) => void
  setExpressionData: (data: ExpressionData | null) => void
  setSelectedEmbedding: (name: string) => void
  setSelectedColorColumn: (name: string | null) => void
  setColorMode: (mode: ColorMode) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // Gene set actions
  addGeneSet: (name: string, genes: string[]) => void
  removeGeneSet: (name: string) => void
  addGenesToSet: (setName: string, genes: string[]) => void
  removeGenesFromSet: (setName: string, genes: string[]) => void
  renameGeneSet: (oldName: string, newName: string) => void
  setSelectedGenes: (genes: string[]) => void
  clearSelectedGenes: () => void

  // Cell selection actions
  setSelectedCellIndices: (indices: number[]) => void
  addToSelection: (indices: number[]) => void
  clearSelection: () => void
  setInteractionMode: (mode: InteractionMode) => void

  // Display preferences actions
  setDisplayPreferences: (prefs: Partial<DisplayPreferences>) => void
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  schema: null,
  embedding: null,
  colorBy: null,
  expressionData: null,
  geneSets: [],
  selectedGenes: [],
  selectedCellIndices: [],
  interactionMode: 'pan',
  selectedEmbedding: null,
  selectedColorColumn: null,
  colorMode: 'none',
  isLoading: false,
  error: null,
  displayPreferences: {
    pointSize: 3,
    backgroundColor: '#1a1a2e',
    colorScale: 'viridis',
    pointOpacity: 0.85,
  },

  // Actions
  setSchema: (schema) => set({ schema }),
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy }),
  setExpressionData: (data) => set({ expressionData: data }),
  setSelectedEmbedding: (name) => set({ selectedEmbedding: name }),
  setSelectedColorColumn: (name) => set({ selectedColorColumn: name }),
  setColorMode: (mode) => set({ colorMode: mode }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  // Gene set actions
  addGeneSet: (name, genes) =>
    set((state) => ({
      geneSets: [...state.geneSets, { name, genes }],
    })),

  removeGeneSet: (name) =>
    set((state) => ({
      geneSets: state.geneSets.filter((gs) => gs.name !== name),
    })),

  addGenesToSet: (setName, genes) =>
    set((state) => ({
      geneSets: state.geneSets.map((gs) =>
        gs.name === setName
          ? { ...gs, genes: [...new Set([...gs.genes, ...genes])] }
          : gs
      ),
    })),

  removeGenesFromSet: (setName, genes) =>
    set((state) => ({
      geneSets: state.geneSets.map((gs) =>
        gs.name === setName
          ? { ...gs, genes: gs.genes.filter((g) => !genes.includes(g)) }
          : gs
      ),
    })),

  renameGeneSet: (oldName, newName) =>
    set((state) => ({
      geneSets: state.geneSets.map((gs) =>
        gs.name === oldName ? { ...gs, name: newName } : gs
      ),
    })),

  setSelectedGenes: (genes) => set({ selectedGenes: genes }),
  clearSelectedGenes: () => set({ selectedGenes: [], expressionData: null, colorMode: 'none' }),

  // Cell selection actions
  setSelectedCellIndices: (indices) => set({ selectedCellIndices: indices }),
  addToSelection: (indices) =>
    set((state) => ({
      selectedCellIndices: [...new Set([...state.selectedCellIndices, ...indices])],
    })),
  clearSelection: () => set({ selectedCellIndices: [] }),
  setInteractionMode: (mode) => set({ interactionMode: mode }),

  // Display preferences
  setDisplayPreferences: (prefs) =>
    set((state) => ({
      displayPreferences: { ...state.displayPreferences, ...prefs },
    })),
}))
