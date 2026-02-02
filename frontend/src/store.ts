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

// Differential expression types
export interface DiffExpGene {
  gene: string
  log2fc: number
  pval: number
  pval_adj: number
}

export interface DiffExpResult {
  positive: DiffExpGene[]
  negative: DiffExpGene[]
  group1_count: number
  group2_count: number
}

export interface ComparisonState {
  group1: number[] | null
  group2: number[] | null
  group1Label: string | null
  group2Label: string | null
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

  // Comparison state for differential expression
  comparison: ComparisonState
  diffExpResult: DiffExpResult | null
  isDiffExpLoading: boolean
  isDiffExpModalOpen: boolean

  // Cell panel column management
  hiddenColumns: Set<string>  // Column names to hide from cell panel
  columnDisplayNames: Record<string, string>  // Map of original name → display name

  // Cell masking - null means all cells are active (no mask)
  activeCellMask: boolean[] | null
  showMaskedCells: boolean  // Whether to display masked (inactive) cells

  // Cell ordering for z-stacking
  cellSortOrder: number[] | null  // Custom sort order for rendering (indices), null = default order
  cellSortVersion: number  // Incremented when sort is triggered

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

  // Comparison actions
  setComparisonGroup1: (indices: number[], label: string) => void
  setComparisonGroup2: (indices: number[], label: string) => void
  clearComparison: () => void
  setDiffExpResult: (result: DiffExpResult | null) => void
  setDiffExpLoading: (loading: boolean) => void
  setDiffExpModalOpen: (open: boolean) => void

  // Column management actions
  hideColumn: (name: string) => void
  showColumn: (name: string) => void
  setColumnDisplayName: (originalName: string, displayName: string) => void
  clearColumnDisplayName: (originalName: string) => void

  // Cell masking actions
  setActiveCellsFromSelection: () => void  // Set current selection as active cells
  addSelectionToActive: () => void  // Add current selection to active cells
  removeSelectionFromActive: () => void  // Remove current selection from active cells
  resetActiveCells: () => void  // Clear mask, make all cells active
  setShowMaskedCells: (show: boolean) => void

  // Cell ordering actions
  sortCellsByExpression: () => void  // Sort cells so high-expression renders on top
  resetCellOrder: () => void  // Reset to default order
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
  comparison: {
    group1: null,
    group2: null,
    group1Label: null,
    group2Label: null,
  },
  diffExpResult: null,
  isDiffExpLoading: false,
  isDiffExpModalOpen: false,
  hiddenColumns: new Set<string>(),
  columnDisplayNames: {},
  activeCellMask: null,
  showMaskedCells: true,
  cellSortOrder: null,
  cellSortVersion: 0,

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

  // Comparison actions
  setComparisonGroup1: (indices, label) =>
    set((state) => ({
      comparison: { ...state.comparison, group1: indices, group1Label: label },
    })),
  setComparisonGroup2: (indices, label) =>
    set((state) => ({
      comparison: { ...state.comparison, group2: indices, group2Label: label },
    })),
  clearComparison: () =>
    set({
      comparison: { group1: null, group2: null, group1Label: null, group2Label: null },
      diffExpResult: null,
    }),
  setDiffExpResult: (result) => set({ diffExpResult: result }),
  setDiffExpLoading: (loading) => set({ isDiffExpLoading: loading }),
  setDiffExpModalOpen: (open) => set({ isDiffExpModalOpen: open }),

  // Column management actions
  hideColumn: (name) =>
    set((state) => ({
      hiddenColumns: new Set([...state.hiddenColumns, name]),
    })),
  showColumn: (name) =>
    set((state) => {
      const next = new Set(state.hiddenColumns)
      next.delete(name)
      return { hiddenColumns: next }
    }),
  setColumnDisplayName: (originalName, displayName) =>
    set((state) => ({
      columnDisplayNames: { ...state.columnDisplayNames, [originalName]: displayName },
    })),
  clearColumnDisplayName: (originalName) =>
    set((state) => {
      const next = { ...state.columnDisplayNames }
      delete next[originalName]
      return { columnDisplayNames: next }
    }),

  // Cell masking actions
  setActiveCellsFromSelection: () =>
    set((state) => {
      if (state.selectedCellIndices.length === 0 || !state.schema) return {}
      const mask = new Array(state.schema.n_cells).fill(false)
      state.selectedCellIndices.forEach((i) => {
        mask[i] = true
      })
      return { activeCellMask: mask }
    }),

  addSelectionToActive: () =>
    set((state) => {
      if (state.selectedCellIndices.length === 0 || !state.schema) return {}
      // If no mask exists, all cells are currently active - create mask with all true
      const mask = state.activeCellMask
        ? [...state.activeCellMask]
        : new Array(state.schema.n_cells).fill(true)
      state.selectedCellIndices.forEach((i) => {
        mask[i] = true
      })
      return { activeCellMask: mask }
    }),

  removeSelectionFromActive: () =>
    set((state) => {
      if (state.selectedCellIndices.length === 0 || !state.schema) return {}
      // If no mask exists, all cells are currently active - create mask with all true first
      const mask = state.activeCellMask
        ? [...state.activeCellMask]
        : new Array(state.schema.n_cells).fill(true)
      state.selectedCellIndices.forEach((i) => {
        mask[i] = false
      })
      return { activeCellMask: mask }
    }),

  resetActiveCells: () => set({ activeCellMask: null }),

  setShowMaskedCells: (show) => set({ showMaskedCells: show }),

  // Cell ordering actions
  sortCellsByExpression: () =>
    set((state) => {
      if (!state.expressionData || !state.schema) return {}
      const values = state.expressionData.values
      // Create array of indices and sort by expression value (ascending, so high values render last/on top)
      const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
      indices.sort((a, b) => {
        const va = values[a] ?? -Infinity
        const vb = values[b] ?? -Infinity
        return va - vb  // Ascending: low values first, high values last (on top)
      })
      return { cellSortOrder: indices, cellSortVersion: state.cellSortVersion + 1 }
    }),

  resetCellOrder: () => set({ cellSortOrder: null, cellSortVersion: 0 }),
}))
