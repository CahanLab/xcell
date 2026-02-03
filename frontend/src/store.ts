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
  transform?: string  // 'log1p' if transformation was applied
}

export interface BivariateExpressionData {
  genes1: string[]
  genes2: string[]
  values1: number[]  // Normalized [0,1] for gene set 1
  values2: number[]  // Normalized [0,1] for gene set 2
  transform?: string
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
export type ColorMode = 'none' | 'metadata' | 'expression' | 'bivariate'

// Drawn line/shape for trajectory/gradient analysis
export interface DrawnLine {
  id: string
  name: string
  embeddingName: string  // Which embedding this was drawn on
  points: [number, number][]  // Raw points in data coordinates
  smoothedPoints: [number, number][] | null  // Smoothed version (if applied)
  visible: boolean  // Whether to display on the embedding
  projections: CellProjection[]  // Cells projected onto this line
}

// Cell projection onto a line
export interface CellProjection {
  cellIndex: number
  positionOnLine: number  // 0 = start, 1 = end (normalized)
  distanceToLine: number  // Perpendicular distance to nearest point on line
}

// Smoothing parameters for lines
export interface LineSmoothingParams {
  windowSize: number  // Moving average window size
  iterations: number  // Number of smoothing passes
}

// Interaction mode for the scatter plot
export type InteractionMode = 'pan' | 'lasso' | 'draw'

// Color scale options for expression data
export type ColorScale = 'viridis' | 'plasma' | 'magma' | 'inferno' | 'cividis' | 'coolwarm' | 'blues' | 'reds'

// Expression transform options
export type ExpressionTransform = 'none' | 'log1p'

// Display preferences
export interface DisplayPreferences {
  pointSize: number  // Base point size (1-10)
  backgroundColor: string  // Hex color
  colorScale: ColorScale  // Color scale for expression data
  pointOpacity: number  // 0-1
  expressionTransform: ExpressionTransform  // Transformation for expression values
}

// Scanpy action history entry
export interface ScanpyActionRecord {
  action: string
  params: Record<string, unknown>
  result: Record<string, unknown>
  timestamp: string
}

interface AppState {
  // Data
  schema: Schema | null
  embedding: EmbeddingData | null
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null

  // Bivariate expression data
  bivariateData: BivariateExpressionData | null

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

  // Drawn lines/shapes for trajectory analysis
  drawnLines: DrawnLine[]
  activeLineId: string | null  // Currently selected line for editing/smoothing
  lineSmoothingParams: LineSmoothingParams

  // Scanpy modal state
  isScanpyModalOpen: boolean
  scanpyActionHistory: ScanpyActionRecord[]

  // Actions
  setSchema: (schema: Schema) => void
  setEmbedding: (embedding: EmbeddingData) => void
  setColorBy: (colorBy: ObsColumnData | null) => void
  setExpressionData: (data: ExpressionData | null) => void
  setBivariateData: (data: BivariateExpressionData | null) => void
  clearBivariateMode: () => void
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
  sortCellsByBivariate: () => void  // Sort cells so high-bivariate product renders on top
  resetCellOrder: () => void  // Reset to default order

  // Line/shape drawing actions
  addLine: (name: string, points: [number, number][], embeddingName: string) => void
  removeLine: (id: string) => void
  setActiveLine: (id: string | null) => void
  renameLine: (id: string, name: string) => void
  smoothLine: (id: string) => void
  setLineSmoothingParams: (params: Partial<LineSmoothingParams>) => void
  setLineVisibility: (id: string, visible: boolean) => void
  projectSelectedCellsOntoLine: (lineId: string) => void  // Project currently selected cells onto a specific line
  clearLineProjections: (lineId: string) => void

  // Scanpy modal actions
  setScanpyModalOpen: (open: boolean) => void
  setScanpyActionHistory: (history: ScanpyActionRecord[]) => void
  addScanpyAction: (action: ScanpyActionRecord) => void
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  schema: null,
  embedding: null,
  colorBy: null,
  expressionData: null,
  bivariateData: null,
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
    expressionTransform: 'none',
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
  drawnLines: [],
  activeLineId: null,
  lineSmoothingParams: {
    windowSize: 5,
    iterations: 2,
  },
  isScanpyModalOpen: false,
  scanpyActionHistory: [],

  // Actions
  setSchema: (schema) => set({ schema }),
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy }),
  setExpressionData: (data) =>
    set((state) => {
      if (!data || !state.schema) {
        return { expressionData: data, cellSortOrder: null, cellSortVersion: 0 }
      }
      // Auto-sort by expression (ascending, so high values render last/on top)
      const values = data.values
      const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
      indices.sort((a, b) => {
        const va = values[a] ?? -Infinity
        const vb = values[b] ?? -Infinity
        return va - vb
      })
      return {
        expressionData: data,
        cellSortOrder: indices,
        cellSortVersion: state.cellSortVersion + 1,
      }
    }),
  setBivariateData: (data) =>
    set((state) => {
      if (!data || !state.schema) {
        return { bivariateData: data, cellSortOrder: null, cellSortVersion: 0 }
      }
      // Auto-sort by product of both values (ascending, so high values render last/on top)
      const { values1, values2 } = data
      const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
      indices.sort((a, b) => {
        const prodA = (values1[a] ?? 0) * (values2[a] ?? 0)
        const prodB = (values1[b] ?? 0) * (values2[b] ?? 0)
        return prodA - prodB
      })
      return {
        bivariateData: data,
        cellSortOrder: indices,
        cellSortVersion: state.cellSortVersion + 1,
      }
    }),
  clearBivariateMode: () => set({ bivariateData: null, colorMode: 'none', cellSortOrder: null, cellSortVersion: 0 }),
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
  clearSelectedGenes: () => set({ selectedGenes: [], expressionData: null, colorMode: 'none', cellSortOrder: null, cellSortVersion: 0 }),

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

  sortCellsByBivariate: () =>
    set((state) => {
      if (!state.bivariateData || !state.schema) return {}
      const { values1, values2 } = state.bivariateData
      // Sort by product of both normalized values (ascending, so high values render last/on top)
      const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
      indices.sort((a, b) => {
        const prodA = (values1[a] ?? 0) * (values2[a] ?? 0)
        const prodB = (values1[b] ?? 0) * (values2[b] ?? 0)
        return prodA - prodB  // Ascending: low values first, high values last (on top)
      })
      return { cellSortOrder: indices, cellSortVersion: state.cellSortVersion + 1 }
    }),

  resetCellOrder: () => set({ cellSortOrder: null, cellSortVersion: 0 }),

  // Line/shape drawing actions
  addLine: (name, points, embeddingName) =>
    set((state) => {
      const id = `line_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      const newLine: DrawnLine = {
        id,
        name,
        embeddingName,
        points,
        smoothedPoints: null,
        visible: true,
        projections: [],
      }
      return {
        drawnLines: [...state.drawnLines, newLine],
        activeLineId: id,  // Auto-select newly drawn line
      }
    }),

  removeLine: (id) =>
    set((state) => ({
      drawnLines: state.drawnLines.filter((l) => l.id !== id),
      activeLineId: state.activeLineId === id ? null : state.activeLineId,
    })),

  setActiveLine: (id) =>
    set({ activeLineId: id }),

  renameLine: (id, name) =>
    set((state) => ({
      drawnLines: state.drawnLines.map((l) => (l.id === id ? { ...l, name } : l)),
    })),

  smoothLine: (id) =>
    set((state) => {
      const line = state.drawnLines.find((l) => l.id === id)
      if (!line || line.points.length < 3) return {}

      const { windowSize, iterations } = state.lineSmoothingParams
      let smoothed = [...line.points] as [number, number][]

      // Apply moving average smoothing multiple times
      for (let iter = 0; iter < iterations; iter++) {
        const newSmoothed: [number, number][] = []
        for (let i = 0; i < smoothed.length; i++) {
          const halfWindow = Math.floor(windowSize / 2)
          let sumX = 0, sumY = 0, count = 0
          for (let j = Math.max(0, i - halfWindow); j <= Math.min(smoothed.length - 1, i + halfWindow); j++) {
            sumX += smoothed[j][0]
            sumY += smoothed[j][1]
            count++
          }
          newSmoothed.push([sumX / count, sumY / count])
        }
        smoothed = newSmoothed
      }

      return {
        drawnLines: state.drawnLines.map((l) =>
          l.id === id ? { ...l, smoothedPoints: smoothed } : l
        ),
      }
    }),

  setLineSmoothingParams: (params) =>
    set((state) => ({
      lineSmoothingParams: { ...state.lineSmoothingParams, ...params },
    })),

  setLineVisibility: (id, visible) =>
    set((state) => ({
      drawnLines: state.drawnLines.map((l) => (l.id === id ? { ...l, visible } : l)),
    })),

  projectSelectedCellsOntoLine: (lineId) =>
    set((state) => {
      const line = state.drawnLines.find((l) => l.id === lineId)
      if (!line || !state.embedding || state.selectedCellIndices.length === 0) return {}

      const linePoints = line.smoothedPoints || line.points
      if (linePoints.length < 2) return {}

      // Compute cumulative distances along the line
      const cumulativeDistances: number[] = [0]
      for (let i = 1; i < linePoints.length; i++) {
        const dx = linePoints[i][0] - linePoints[i - 1][0]
        const dy = linePoints[i][1] - linePoints[i - 1][1]
        cumulativeDistances.push(cumulativeDistances[i - 1] + Math.sqrt(dx * dx + dy * dy))
      }
      const totalLength = cumulativeDistances[cumulativeDistances.length - 1]

      // Get existing projection cell indices to avoid duplicates
      const existingCellIndices = new Set(line.projections.map((p) => p.cellIndex))

      // Project each selected cell onto the line
      const newProjections: CellProjection[] = []
      const coords = state.embedding.coordinates

      for (const cellIndex of state.selectedCellIndices) {
        // Skip if already projected
        if (existingCellIndices.has(cellIndex)) continue

        const [cx, cy] = coords[cellIndex]
        let minDist = Infinity
        let bestPosition = 0

        // Find closest point on each line segment
        for (let i = 0; i < linePoints.length - 1; i++) {
          const [x1, y1] = linePoints[i]
          const [x2, y2] = linePoints[i + 1]

          // Project point onto line segment
          const dx = x2 - x1
          const dy = y2 - y1
          const segmentLength = Math.sqrt(dx * dx + dy * dy)

          if (segmentLength === 0) continue

          // Parameter t for projection (clamped to [0, 1] for segment)
          let t = ((cx - x1) * dx + (cy - y1) * dy) / (segmentLength * segmentLength)
          t = Math.max(0, Math.min(1, t))

          // Closest point on segment
          const projX = x1 + t * dx
          const projY = y1 + t * dy

          // Distance from cell to closest point
          const dist = Math.sqrt((cx - projX) * (cx - projX) + (cy - projY) * (cy - projY))

          if (dist < minDist) {
            minDist = dist
            // Position along the entire line (normalized 0-1)
            bestPosition = (cumulativeDistances[i] + t * segmentLength) / totalLength
          }
        }

        newProjections.push({
          cellIndex,
          positionOnLine: bestPosition,
          distanceToLine: minDist,
        })
      }

      // Merge new projections with existing ones
      return {
        drawnLines: state.drawnLines.map((l) =>
          l.id === lineId
            ? { ...l, projections: [...l.projections, ...newProjections] }
            : l
        ),
      }
    }),

  clearLineProjections: (lineId) =>
    set((state) => ({
      drawnLines: state.drawnLines.map((l) =>
        l.id === lineId ? { ...l, projections: [] } : l
      ),
    })),

  // Scanpy modal actions
  setScanpyModalOpen: (open) => set({ isScanpyModalOpen: open }),
  setScanpyActionHistory: (history) => set({ scanpyActionHistory: history }),
  addScanpyAction: (action) =>
    set((state) => ({
      scanpyActionHistory: [...state.scanpyActionHistory, action],
    })),
}))
