import { create } from 'zustand'

export interface Schema {
  n_cells: number
  n_genes: number
  embeddings: string[]
  obs_columns: string[]
  obs_dtypes: Record<string, string>
  filename?: string
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
  n_masked_excluded?: number  // multi-gene only: count of genes dropped by gene mask
}

export interface BivariateExpressionData {
  genes1: string[]
  genes2: string[]
  values1: number[]  // Normalized [0,1] for gene set 1
  values2: number[]  // Normalized [0,1] for gene set 2
  transform?: string
}

export interface GeneSet {
  id: string
  name: string
  genes: string[]
  pinned?: boolean
}

// Category types for organizing gene sets
export type GeneSetCategoryType = 'manual' | 'gene_clusters' | 'similar_genes' | 'diff_exp' | 'spatial' | 'marker_genes' | 'line_association'

export interface GeneSetFolder {
  id: string
  name: string
  expanded: boolean
  geneSets: GeneSet[]
  createdAt: string
  pinned?: boolean
}

export interface GeneSetCategory {
  type: GeneSetCategoryType
  name: string  // Display name
  expanded: boolean
  folders: GeneSetFolder[]  // Subfolders (for gene_clusters, diff_exp)
  geneSets: GeneSet[]  // Direct gene sets (for similar_genes, manual)
  visible?: boolean
}

// Default category configuration
const createDefaultCategories = (): Record<GeneSetCategoryType, GeneSetCategory> => ({
  manual: {
    type: 'manual',
    name: 'Manual',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  gene_clusters: {
    type: 'gene_clusters',
    name: 'Gene Clusters',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  similar_genes: {
    type: 'similar_genes',
    name: 'Similar Genes',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  diff_exp: {
    type: 'diff_exp',
    name: 'Differential Expression',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  spatial: {
    type: 'spatial',
    name: 'Spatially Variable',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  marker_genes: {
    type: 'marker_genes',
    name: 'Marker Genes',
    expanded: true,
    folders: [],
    geneSets: [],
  },
  line_association: {
    type: 'line_association',
    name: 'Line Association',
    expanded: true,
    folders: [],
    geneSets: [],
  },
})

// Helper to generate unique IDs
let geneSetIdCounter = 0
export const generateGeneSetId = () => `gs_${Date.now()}_${++geneSetIdCounter}`
export const generateFolderId = () => `folder_${Date.now()}_${++geneSetIdCounter}`

// Look up a user-config override; returns `fallback` if the path doesn't exist.
// Shape:
//   userConfigGet(cfg, ['scanpy', 'filter_cells', 'min_genes'], 25)
export function userConfigGet<T>(cfg: Record<string, unknown>, path: string[], fallback: T): T {
  let cursor: unknown = cfg
  for (const key of path) {
    if (cursor && typeof cursor === 'object' && key in (cursor as Record<string, unknown>)) {
      cursor = (cursor as Record<string, unknown>)[key]
    } else {
      return fallback
    }
  }
  return (cursor === undefined ? fallback : (cursor as T))
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

// Source for the "Select cells by expression" modal.
// `null` means the modal is closed. Mirrors the ClusterGeneSetModal pattern.
export type SelectByExpressionSource =
  | { type: 'gene'; gene: string }
  | { type: 'geneSet'; name: string; genes: string[] }

// Color mode: what determines cell colors
export type ColorMode = 'none' | 'metadata' | 'expression' | 'bivariate'

// Selection tool types
export type SelectionTool = 'lasso' | 'polygon'

// Drawn line/shape for trajectory/gradient analysis
// Draw tool types
export type DrawTool = 'pencil' | 'lasso' | 'polygon' | 'segmented' | 'smooth_curve'

export interface DrawnLine {
  id: string
  name: string
  embeddingName: string  // Which embedding this was drawn on
  points: [number, number][]  // Raw points in data coordinates
  smoothedPoints: [number, number][] | null  // Smoothed version (if applied)
  visible: boolean  // Whether to display on the embedding
  projections: CellProjection[]  // Cells projected onto this line
  drawType: DrawTool  // How this shape was drawn
  closed: boolean  // Whether the shape is closed (polygon)
  strokeColor: string  // Line color (hex)
  strokeWidth: number  // Line width in pixels
  fillColor: string | null  // Fill color (null = no fill)
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

// Line association test result (gene)
export interface LineAssociationGene {
  gene: string
  f_stat: number
  pval: number
  fdr: number
  r_squared: number
  amplitude: number
  direction: number  // Positive = increases along line, negative = decreases
  profile?: number[]  // Smoothed expression profile (normalized 0-1)
  peak_position?: number  // Position along line where expression peaks
}

// Diagnostic info from line association test
export interface LineAssociationDiagnostics {
  n_genes_tested: number
  n_pval_below_05: number
  n_pval_below_01: number
  position_range: [number, number]
  position_std: number
  expression_range: [number, number]
  expression_mean: number
  n_zero_genes: number
  spline_df: number
}

// A module of genes with similar expression profiles along a line
export interface LineAssociationModule {
  module_id: number
  pattern: string                    // 'increasing', 'decreasing', 'peak', 'trough', 'complex'
  n_genes: number
  representative_profile: number[]   // normalized 0-1 profile
  profile_positions: number[]        // corresponding position values (0-1)
  genes: LineAssociationGene[]
}

// Line association test results
export interface LineAssociationResult {
  positive: LineAssociationGene[]  // Genes increasing along line (backward compat)
  negative: LineAssociationGene[]  // Genes decreasing along line (backward compat)
  modules: LineAssociationModule[] // All significant genes grouped by profile shape
  all_genes?: LineAssociationGene[] // Stats for every gene tested (for CSV export / GSEA)
  n_cells: number
  n_significant: number
  n_positive: number
  n_negative: number
  n_modules: number
  line_name: string
  test_variable: string  // 'position' or 'distance'
  fdr_threshold: number
  n_lines?: number
  lines_used?: string[]
  diagnostics?: LineAssociationDiagnostics
}

// Center panel view mode
export type CenterPanelView = 'scatter' | 'heatmap'

// Layout mode for scatter view
export type LayoutMode = 'single' | 'dual'

// Heatmap configuration
export interface HeatmapConfig {
  selectedGeneSets: { name: string; genes: string[] }[]
  cellOrdering: 'none' | 'category' | 'line_position' | 'line_distance' | 'category_then_position'
  obsColumn: string | null
  lineName: string | null
  geneOrdering: 'as_provided' | 'peak_position'
  aggregateGeneSets: boolean
  nBins: number
}

// Gene mask — mirrors backend /api/gene_mask response
export interface GeneMaskConfig {
  active: boolean
  keepColumns: string[]
  hideColumns: string[]
  keepCombineMode: 'or' | 'and'
  nVisible: number
  nTotal: number
  visibleGeneNames: string[] | null  // null when inactive
}

export interface PCASubsetSummary {
  obsmKey: string       // e.g. 'X_pca_noPC2_5'
  suffix: string        // e.g. 'noPC2_5'
  droppedPcs: number[]  // 1-indexed
  nPcsKept: number
}

// Interaction mode for the scatter plot
export type InteractionMode = 'pan' | 'lasso' | 'draw' | 'adjust' | 'quilt'

// Color scale options for expression data
export type ColorScale = 'viridis' | 'plasma' | 'magma' | 'inferno' | 'cividis' | 'coolwarm' | 'blues' | 'reds'

// Bivariate colormap options (corner colors: [lowLow, highLow, lowHigh, highHigh])
export type BivariateColormap = 'default' | 'pinkgreen' | 'orangepurple' | 'custom'

// Gene set scoring method options
// Per-gene normalization applied before aggregating across genes.
// 'none'        — raw values
// 'zscore_mad'  — mean-center + scale by MAD (fallback SD); robust to outliers
// 'zscore_sd'   — mean-center + scale by SD
// 'minmax'      — clip at per_gene_clip then rescale to [0,1] (Contourize-style)
// 'rank'        — average-rank within each gene, divided by N → [0,1]
export type GeneSetPerGeneNorm = 'none' | 'zscore_mad' | 'zscore_sd' | 'minmax' | 'rank'

// How per-gene values are combined into one per-cell summary.
export type GeneSetAggregation = 'mean' | 'median' | 'sum' | 'max'

// Expression transform options
export type ExpressionTransform = 'none' | 'log1p'

// Display preferences
export interface DisplayPreferences {
  pointSize: number  // Base point size (1-10)
  backgroundColor: string  // Hex color
  colorScale: ColorScale  // Color scale for expression data
  bivariateColormap: BivariateColormap  // Colormap for bivariate expression
  // Gene set aggregation pipeline (used for both single-gene-set coloring and bivariate).
  // Pipeline: source → per-gene normalize → aggregate across genes → per-cell clip.
  geneSetPerGeneNorm: GeneSetPerGeneNorm
  geneSetPerGeneClip: number  // percentile (0–5%); only used by 'minmax' per-gene norm
  geneSetAggregation: GeneSetAggregation
  pointOpacity: number  // 0-1
  expressionTransform: ExpressionTransform  // Transformation for expression values
  clipPercentile: number  // Symmetric percentile clip for color-ramp anchors (0 = off)
}

// Scanpy action history entry
export interface ScanpyActionRecord {
  action: string
  params: Record<string, unknown>
  result: Record<string, unknown>
  timestamp: string
}

// Multi-dataset slot identifiers
export type DatasetSlot = 'primary' | 'secondary'

// Per-dataset state: fields that are unique to each loaded dataset
export interface DatasetState {
  schema: Schema | null
  embedding: EmbeddingData | null
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  bivariateData: BivariateExpressionData | null
  selectedGenes: string[]
  selectedGeneSetName: string | null
  selectedEmbedding: string | null
  selectedColorColumn: string | null
  colorMode: ColorMode
  selectedCellIndices: number[]
  activeCellMask: boolean[] | null
  showMaskedCells: boolean
  cellSortOrder: number[] | null
  cellSortVersion: number
  bivariateSortReversed: boolean
  displayPreferences: DisplayPreferences
  drawnLines: DrawnLine[]
  hiddenColumns: Set<string>
  columnDisplayNames: Record<string, string>
  obsSummariesVersion: number
  scanpyActionHistory: ScanpyActionRecord[]
  varIdentifierColumns: string[]
  currentVarIndex: string
  geneMaskConfig: GeneMaskConfig | null
  pcaSubsets: PCASubsetSummary[]
  // Layer used for visualization-side gene expression coloring (single-gene,
  // multi-gene, bivariate). 'X' (default) reads adata.X; any other value reads
  // adata.layers[displayLayer]. Per-dataset because layers are per-dataset.
  displayLayer: string
}

// Hardcoded fallback defaults for display preferences. Kept as a factory so
// every consumer gets a fresh object (no aliasing across slots).
export function defaultDisplayPreferences(): DisplayPreferences {
  return {
    pointSize: 3,
    backgroundColor: '#1a1a2e',
    colorScale: 'viridis',
    bivariateColormap: 'default',
    geneSetPerGeneNorm: 'zscore_mad',
    geneSetPerGeneClip: 1.0,
    geneSetAggregation: 'mean',
    pointOpacity: 0.85,
    expressionTransform: 'none',
    clipPercentile: 1.0,
  }
}

// Map a `display:` section of the user config (snake_case keys) to a
// Partial<DisplayPreferences> overlay. Unknown / mistyped keys are ignored
// silently — same forgiving philosophy as the rest of the config loader.
export function displayPreferencesFromConfig(
  cfg: Record<string, unknown>
): Partial<DisplayPreferences> {
  const display = cfg.display
  if (!display || typeof display !== 'object') return {}
  const d = display as Record<string, unknown>
  const out: Partial<DisplayPreferences> = {}
  const num = (v: unknown): number | undefined =>
    typeof v === 'number' && Number.isFinite(v) ? v : undefined
  const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined)

  const pointSize = num(d.point_size)
  if (pointSize !== undefined) out.pointSize = pointSize
  const pointOpacity = num(d.point_opacity)
  if (pointOpacity !== undefined) out.pointOpacity = pointOpacity
  const bg = str(d.background_color)
  if (bg !== undefined) out.backgroundColor = bg
  const cs = str(d.color_scale)
  if (cs !== undefined) out.colorScale = cs as ColorScale
  const bc = str(d.bivariate_colormap)
  if (bc !== undefined) out.bivariateColormap = bc as BivariateColormap
  const et = str(d.expression_transform)
  if (et !== undefined) out.expressionTransform = et as ExpressionTransform
  const clip = num(d.clip_percentile)
  if (clip !== undefined) out.clipPercentile = clip
  const gsn = str(d.gene_set_per_gene_norm)
  if (gsn !== undefined) out.geneSetPerGeneNorm = gsn as GeneSetPerGeneNorm
  const gsc = num(d.gene_set_per_gene_clip)
  if (gsc !== undefined) out.geneSetPerGeneClip = gsc
  const gsa = str(d.gene_set_aggregation)
  if (gsa !== undefined) out.geneSetAggregation = gsa as GeneSetAggregation
  return out
}

export function createDefaultDatasetState(
  displayOverrides: Partial<DisplayPreferences> = {}
): DatasetState {
  return {
    schema: null,
    embedding: null,
    colorBy: null,
    expressionData: null,
    bivariateData: null,
    selectedGenes: [],
    selectedGeneSetName: null,
    selectedEmbedding: null,
    selectedColorColumn: null,
    colorMode: 'none',
    selectedCellIndices: [],
    activeCellMask: null,
    showMaskedCells: true,
    cellSortOrder: null,
    cellSortVersion: 0,
    bivariateSortReversed: false,
    displayPreferences: { ...defaultDisplayPreferences(), ...displayOverrides },
    drawnLines: [],
    hiddenColumns: new Set<string>(),
    columnDisplayNames: {},
    obsSummariesVersion: 0,
    scanpyActionHistory: [],
    varIdentifierColumns: [],
    currentVarIndex: '_index',
    geneMaskConfig: null,
    pcaSubsets: [],
    displayLayer: 'X',
  }
}

interface AppState {
  // Data (flat per-dataset fields — backward compat)
  schema: Schema | null
  embedding: EmbeddingData | null
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null

  // Bivariate expression data
  bivariateData: BivariateExpressionData | null

  // Gene management (hierarchical)
  geneSetCategories: Record<GeneSetCategoryType, GeneSetCategory>
  selectedGenes: string[]  // Currently selected genes for expression coloring
  selectedGeneSetName: string | null  // Name of the gene set being colored (null for single genes)

  // Legacy flat gene sets (for backward compatibility during migration)
  geneSets: GeneSet[]

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
  clusterModalSourceSet: {
    name: string
    genes: string[]
    categoryType: GeneSetCategoryType
    folderId: string | null
  } | null
  selectByExpressionSource: SelectByExpressionSource | null

  // Line association state
  lineAssociationResult: LineAssociationResult | null
  isLineAssociationLoading: boolean
  isLineAssociationModalOpen: boolean

  // Background task tracking for cancellable operations
  activeTaskId: string | null

  // Cell panel column management
  hiddenColumns: Set<string>  // Column names to hide from cell panel
  columnDisplayNames: Record<string, string>  // Map of original name → display name

  // Cell masking - null means all cells are active (no mask)
  activeCellMask: boolean[] | null
  showMaskedCells: boolean  // Whether to display masked (inactive) cells

  // Cell ordering for z-stacking
  cellSortOrder: number[] | null  // Custom sort order for rendering (indices), null = default order
  cellSortVersion: number  // Incremented when sort is triggered
  bivariateSortReversed: boolean  // Whether bivariate sort is reversed (high values first)

  // Drawn lines/shapes for trajectory analysis
  drawnLines: DrawnLine[]
  activeLineId: string | null  // Currently selected line for editing/smoothing
  lineSmoothingParams: LineSmoothingParams
  drawTool: DrawTool  // Currently selected draw tool
  selectionTool: SelectionTool  // Currently selected selection tool

  // Import modal state
  isImportModalOpen: boolean

  // Gene mask modal state (global)
  geneMaskModalOpen: boolean

  // User config (loaded from backend once at app mount; overrides param defaults)
  // Shape: loose — { scanpy: { <fn>: { <param>: value } }, line_association: { <param>: value } }
  userConfig: Record<string, unknown>

  // Scanpy modal state
  isScanpyModalOpen: boolean
  scanpyActionHistory: ScanpyActionRecord[]

  // Observable summaries refresh trigger
  obsSummariesVersion: number

  // Var identifier column switching (per-dataset, flat mirror)
  varIdentifierColumns: string[]
  currentVarIndex: string

  // Gene mask config (per-dataset, flat mirror)
  geneMaskConfig: GeneMaskConfig | null

  // PCA subsets (per-dataset, flat mirror)
  pcaSubsets: PCASubsetSummary[]

  // Display layer for visualization-side expression coloring (per-dataset).
  // 'X' = adata.X (default); any other value = adata.layers[displayLayer].
  displayLayer: string

  // Marker genes modal state
  isMarkerGenesModalOpen: boolean
  markerGenesColumn: string | null

  // Checkbox-based comparison state
  comparisonCheckedColumn: string | null
  comparisonCheckedCategories: Set<string>

  // Heatmap tab state (rollback: remove this block)
  centerPanelView: CenterPanelView
  heatmapConfig: HeatmapConfig | null

  // Layout mode (single vs side-by-side dual scatter)
  layoutMode: LayoutMode

  // Quilt mode phase
  quiltPhase: 'lasso' | 'transform'
  quiltUndoDepth: number

  // Multi-dataset support
  datasets: Record<DatasetSlot, DatasetState>
  activeSlot: DatasetSlot

  // Actions
  setSchema: (schema: Schema) => void
  setEmbedding: (embedding: EmbeddingData | null) => void
  setColorBy: (colorBy: ObsColumnData | null) => void
  setExpressionData: (data: ExpressionData | null) => void
  setBivariateData: (data: BivariateExpressionData | null) => void
  clearBivariateMode: () => void
  setSelectedEmbedding: (name: string) => void
  setSelectedColorColumn: (name: string | null) => void
  setColorMode: (mode: ColorMode) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // Gene set actions (legacy - for backward compatibility)
  addGeneSet: (name: string, genes: string[]) => void
  removeGeneSet: (name: string) => void
  addGenesToSet: (setName: string, genes: string[]) => void
  removeGenesFromSet: (setName: string, genes: string[]) => void
  renameGeneSet: (oldName: string, newName: string) => void
  setSelectedGenes: (genes: string[]) => void
  setSelectedGeneSetName: (name: string | null) => void
  clearSelectedGenes: () => void

  // Gene set category actions (hierarchical)
  toggleCategoryExpanded: (categoryType: GeneSetCategoryType) => void
  toggleFolderExpanded: (categoryType: GeneSetCategoryType, folderId: string) => void
  addGeneSetToCategory: (categoryType: GeneSetCategoryType, name: string, genes: string[]) => void
  addFolderToCategory: (categoryType: GeneSetCategoryType, folderName: string, geneSets: { name: string; genes: string[] }[]) => void
  addGeneSetToFolder: (categoryType: GeneSetCategoryType, folderId: string, name: string, genes: string[]) => void
  removeGeneSetFromCategory: (categoryType: GeneSetCategoryType, geneSetId: string) => void
  removeGeneSetFromFolder: (categoryType: GeneSetCategoryType, folderId: string, geneSetId: string) => void
  removeFolder: (categoryType: GeneSetCategoryType, folderId: string) => void
  addGenesToCategorySet: (categoryType: GeneSetCategoryType, geneSetId: string, genes: string[]) => void
  removeGenesFromCategorySet: (categoryType: GeneSetCategoryType, geneSetId: string, genes: string[]) => void
  renameCategoryGeneSet: (categoryType: GeneSetCategoryType, geneSetId: string, newName: string) => void
  toggleSetPinned: (categoryType: GeneSetCategoryType, folderId: string | null, geneSetId: string) => void
  toggleFolderPinned: (categoryType: GeneSetCategoryType, folderId: string) => void
  toggleCategoryVisible: (categoryType: GeneSetCategoryType) => void
  renameFolder: (categoryType: GeneSetCategoryType, folderId: string, newName: string) => void
  moveGeneSetToFolder: (
    categoryType: GeneSetCategoryType,
    geneSetId: string,
    destFolderId: string | null,
    sourceFolderId: string | null,
  ) => void
  reorderGeneSet: (
    categoryType: GeneSetCategoryType,
    folderId: string | null,
    geneSetId: string,
    targetIndex: number,
  ) => void
  reorderFolder: (categoryType: GeneSetCategoryType, folderId: string, targetIndex: number) => void

  // Cell selection actions
  setSelectedCellIndices: (indices: number[]) => void
  addToSelection: (indices: number[]) => void
  clearSelection: () => void
  invertSelection: () => void
  setInteractionMode: (mode: InteractionMode) => void
  setQuiltPhase: (phase: 'lasso' | 'transform') => void
  setQuiltUndoDepth: (depth: number) => void

  // Display preferences actions
  setDisplayPreferences: (prefs: Partial<DisplayPreferences>) => void

  // Comparison actions
  setComparisonGroup1: (indices: number[], label: string) => void
  setComparisonGroup2: (indices: number[], label: string) => void
  clearComparison: () => void
  setDiffExpResult: (result: DiffExpResult | null) => void
  setDiffExpLoading: (loading: boolean) => void
  setDiffExpModalOpen: (open: boolean) => void
  setClusterModalSourceSet: (src: {
    name: string
    genes: string[]
    categoryType: GeneSetCategoryType
    folderId: string | null
  } | null) => void
  setSelectByExpressionSource: (src: SelectByExpressionSource | null) => void

  // Line association actions
  setLineAssociationResult: (result: LineAssociationResult | null) => void
  setLineAssociationLoading: (loading: boolean) => void
  setLineAssociationModalOpen: (open: boolean) => void
  setActiveTaskId: (taskId: string | null) => void

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
  toggleBivariateSortOrder: () => void  // Toggle between normal and reversed sort order
  resetCellOrder: () => void  // Reset to default order

  // Line/shape drawing actions
  setDrawTool: (tool: DrawTool) => void
  setSelectionTool: (tool: SelectionTool) => void
  updateLineAppearance: (id: string, updates: { strokeColor?: string; strokeWidth?: number; fillColor?: string | null; closed?: boolean }) => void
  addLine: (name: string, points: [number, number][], embeddingName: string, drawType?: DrawTool, closed?: boolean) => void
  removeLine: (id: string) => void
  setActiveLine: (id: string | null) => void
  renameLine: (id: string, name: string) => void
  smoothLine: (id: string) => void
  setLineSmoothingParams: (params: Partial<LineSmoothingParams>) => void
  setLineVisibility: (id: string, visible: boolean) => void
  projectSelectedCellsOntoLine: (lineId: string) => void  // Project currently selected cells onto a specific line
  clearLineProjections: (lineId: string) => void

  // Import modal actions
  setImportModalOpen: (open: boolean) => void

  // Gene mask actions
  setGeneMaskModalOpen: (open: boolean) => void
  setGeneMaskConfig: (config: GeneMaskConfig | null) => void
  setPcaSubsets: (subsets: PCASubsetSummary[]) => void

  // Display layer (per-dataset) — chooses adata.X or a layer for visualization.
  setDisplayLayer: (layer: string) => void

  // User config action
  setUserConfig: (config: Record<string, unknown>) => void
  // Apply the `display:` section of the user config to all dataset slots and
  // the flat top-level mirror. Called once after the config is fetched at
  // startup so the very first frame already reflects user defaults.
  applyDisplayDefaultsFromConfig: () => void

  // Scanpy modal actions
  setScanpyModalOpen: (open: boolean) => void
  setScanpyActionHistory: (history: ScanpyActionRecord[]) => void
  addScanpyAction: (action: ScanpyActionRecord) => void

  // Observable summaries refresh
  refreshObsSummaries: () => void

  // Marker genes modal actions
  setMarkerGenesModalOpen: (open: boolean) => void
  setMarkerGenesColumn: (column: string | null) => void

  // Var identifier switching actions
  setVarIdentifierColumns: (columns: string[]) => void
  setCurrentVarIndex: (name: string) => void
  remapAllGeneNames: (oldToNew: Map<string, string>) => void

  // Checkbox-based comparison actions
  toggleComparisonCategory: (column: string, category: string) => void
  clearComparisonCategories: () => void

  // Heatmap tab actions (rollback: remove this block)
  setCenterPanelView: (view: CenterPanelView) => void
  setHeatmapConfig: (config: HeatmapConfig | null) => void

  // Layout mode actions
  setLayoutMode: (mode: LayoutMode) => void

  // Multi-dataset actions
  setActiveSlot: (slot: DatasetSlot) => void
  loadDatasetIntoSlot: (slot: DatasetSlot, schema: Schema) => void
  patchSlotState: (slot: DatasetSlot, patch: Partial<DatasetState>) => void
}

export const useStore = create<AppState>((set, get) => {
  // Helper: dual-write a per-dataset patch to both flat fields and datasets[activeSlot]
  function dsUpdate(patch: Partial<DatasetState>): Partial<AppState> {
    const { activeSlot, datasets } = get()
    return {
      ...patch,
      datasets: {
        ...datasets,
        [activeSlot]: { ...datasets[activeSlot], ...patch },
      },
    } as Partial<AppState>
  }

  // Helper: callback-style dual-write where the patch depends on current state
  function dsUpdateFn(fn: (state: AppState) => Partial<DatasetState>): Partial<AppState> {
    const state = get()
    const patch = fn(state)
    if (Object.keys(patch).length === 0) return {}
    const { activeSlot, datasets } = state
    return {
      ...patch,
      datasets: {
        ...datasets,
        [activeSlot]: { ...datasets[activeSlot], ...patch },
      },
    } as Partial<AppState>
  }

  // Helper: copy all per-dataset fields from a slot to flat top-level fields
  function syncFlatFields(slot: DatasetSlot, datasets: Record<DatasetSlot, DatasetState>): Partial<AppState> {
    const ds = datasets[slot]
    return {
      schema: ds.schema,
      embedding: ds.embedding,
      colorBy: ds.colorBy,
      expressionData: ds.expressionData,
      bivariateData: ds.bivariateData,
      selectedGenes: ds.selectedGenes,
      selectedGeneSetName: ds.selectedGeneSetName,
      selectedEmbedding: ds.selectedEmbedding,
      selectedColorColumn: ds.selectedColorColumn,
      colorMode: ds.colorMode,
      selectedCellIndices: ds.selectedCellIndices,
      activeCellMask: ds.activeCellMask,
      showMaskedCells: ds.showMaskedCells,
      cellSortOrder: ds.cellSortOrder,
      cellSortVersion: ds.cellSortVersion,
      bivariateSortReversed: ds.bivariateSortReversed,
      displayPreferences: ds.displayPreferences,
      drawnLines: ds.drawnLines,
      hiddenColumns: ds.hiddenColumns,
      columnDisplayNames: ds.columnDisplayNames,
      obsSummariesVersion: ds.obsSummariesVersion,
      scanpyActionHistory: ds.scanpyActionHistory,
      varIdentifierColumns: ds.varIdentifierColumns,
      currentVarIndex: ds.currentVarIndex,
      geneMaskConfig: ds.geneMaskConfig,
      pcaSubsets: ds.pcaSubsets,
      displayLayer: ds.displayLayer,
    }
  }

  return {
    // Initial state
    schema: null,
    embedding: null,
    colorBy: null,
    expressionData: null,
    bivariateData: null,
    geneSetCategories: createDefaultCategories(),
    geneSets: [],  // Legacy, kept for backward compatibility
    selectedGenes: [],
    selectedGeneSetName: null,
    selectedCellIndices: [],
    interactionMode: 'pan',
    selectedEmbedding: null,
    selectedColorColumn: null,
    colorMode: 'none',
    isLoading: false,
    error: null,
    displayPreferences: defaultDisplayPreferences(),
    comparison: {
      group1: null,
      group2: null,
      group1Label: null,
      group2Label: null,
    },
    diffExpResult: null,
    isDiffExpLoading: false,
    isDiffExpModalOpen: false,
    clusterModalSourceSet: null,
    selectByExpressionSource: null,
    lineAssociationResult: null,
    isLineAssociationLoading: false,
    isLineAssociationModalOpen: false,
    activeTaskId: null,
    hiddenColumns: new Set<string>(),
    columnDisplayNames: {},
    activeCellMask: null,
    showMaskedCells: true,
    cellSortOrder: null,
    cellSortVersion: 0,
    bivariateSortReversed: false,
    drawnLines: [],
    activeLineId: null,
    lineSmoothingParams: {
      windowSize: 5,
      iterations: 2,
    },
    drawTool: 'pencil' as DrawTool,
    selectionTool: 'lasso' as SelectionTool,
    isImportModalOpen: false,
    geneMaskModalOpen: false,
    geneMaskConfig: null,
    pcaSubsets: [],
    displayLayer: 'X',
    userConfig: {},
    isScanpyModalOpen: false,
    scanpyActionHistory: [],
    obsSummariesVersion: 0,
    varIdentifierColumns: [],
    currentVarIndex: '_index',
    isMarkerGenesModalOpen: false,
    markerGenesColumn: null,
    comparisonCheckedColumn: null,
    comparisonCheckedCategories: new Set<string>(),
    centerPanelView: 'scatter',
    heatmapConfig: null,

    // Layout mode
    layoutMode: 'single' as LayoutMode,

    // Quilt mode phase
    quiltPhase: 'lasso' as const,
    quiltUndoDepth: 0,

    // Multi-dataset state
    datasets: {
      primary: createDefaultDatasetState(),
      secondary: createDefaultDatasetState(),
    },
    activeSlot: 'primary' as DatasetSlot,

    // === Per-dataset actions (dual-write) ===

    setSchema: (schema) => set(dsUpdate({ schema })),
    setEmbedding: (embedding) => set(dsUpdate({ embedding })),
    setColorBy: (colorBy) => set(dsUpdate({ colorBy })),

    setExpressionData: (data) =>
      set(dsUpdateFn((state) => {
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
      })),

    setBivariateData: (data) =>
      set(dsUpdateFn((state) => {
        if (!data || !state.schema) {
          return { bivariateData: data, cellSortOrder: null, cellSortVersion: 0 }
        }
        // Sort by sum of both values
        // Respect bivariateSortReversed setting
        const reversed = state.bivariateSortReversed
        const { values1, values2 } = data
        const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
        indices.sort((a, b) => {
          const sumA = (values1[a] ?? 0) + (values2[a] ?? 0)
          const sumB = (values1[b] ?? 0) + (values2[b] ?? 0)
          return reversed ? sumB - sumA : sumA - sumB
        })
        return {
          bivariateData: data,
          cellSortOrder: indices,
          cellSortVersion: state.cellSortVersion + 1,
        }
      })),

    clearBivariateMode: () => set(dsUpdate({ bivariateData: null, colorMode: 'none', cellSortOrder: null, cellSortVersion: 0 })),
    setSelectedEmbedding: (name) => set(dsUpdate({ selectedEmbedding: name })),
    setSelectedColorColumn: (name) => set(dsUpdate({ selectedColorColumn: name })),
    setColorMode: (mode) => set(dsUpdate({ colorMode: mode })),

    // Global-only actions (no dual-write needed)
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),

    // Gene set actions (global)
    addGeneSet: (name, genes) =>
      set((state) => {
        const newGeneSet = { id: generateGeneSetId(), name, genes }
        return {
          // Add to legacy flat list
          geneSets: [...state.geneSets, newGeneSet],
          // Also add to manual category in hierarchical structure
          geneSetCategories: {
            ...state.geneSetCategories,
            manual: {
              ...state.geneSetCategories.manual,
              geneSets: [...state.geneSetCategories.manual.geneSets, newGeneSet],
            },
          },
        }
      }),

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

    setSelectedGenes: (genes) => set(dsUpdate({ selectedGenes: genes })),
    setSelectedGeneSetName: (name) => set(dsUpdate({ selectedGeneSetName: name })),
    clearSelectedGenes: () => set(dsUpdate({ selectedGenes: [], selectedGeneSetName: null, expressionData: null, colorMode: 'none', cellSortOrder: null, cellSortVersion: 0 })),

    // Gene set category actions (hierarchical — global)
    toggleCategoryExpanded: (categoryType) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            expanded: !state.geneSetCategories[categoryType].expanded,
          },
        },
      })),

    toggleFolderExpanded: (categoryType, folderId) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: state.geneSetCategories[categoryType].folders.map((f) =>
              f.id === folderId ? { ...f, expanded: !f.expanded } : f
            ),
          },
        },
      })),

    addGeneSetToCategory: (categoryType, name, genes) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            geneSets: [
              ...state.geneSetCategories[categoryType].geneSets,
              { id: generateGeneSetId(), name, genes },
            ],
          },
        },
      })),

    addFolderToCategory: (categoryType, folderName, geneSets) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: [
              ...state.geneSetCategories[categoryType].folders,
              {
                id: generateFolderId(),
                name: folderName,
                expanded: true,
                createdAt: new Date().toISOString(),
                geneSets: geneSets.map((gs) => ({
                  id: generateGeneSetId(),
                  name: gs.name,
                  genes: gs.genes,
                })),
              },
            ],
          },
        },
      })),

    addGeneSetToFolder: (categoryType, folderId, name, genes) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: state.geneSetCategories[categoryType].folders.map((f) =>
              f.id === folderId
                ? {
                    ...f,
                    geneSets: [...f.geneSets, { id: generateGeneSetId(), name, genes }],
                  }
                : f
            ),
          },
        },
      })),

    removeGeneSetFromCategory: (categoryType, geneSetId) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            geneSets: state.geneSetCategories[categoryType].geneSets.filter(
              (gs) => gs.id !== geneSetId
            ),
          },
        },
      })),

    removeGeneSetFromFolder: (categoryType, folderId, geneSetId) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: state.geneSetCategories[categoryType].folders.map((f) =>
              f.id === folderId
                ? { ...f, geneSets: f.geneSets.filter((gs) => gs.id !== geneSetId) }
                : f
            ),
          },
        },
      })),

    removeFolder: (categoryType, folderId) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: state.geneSetCategories[categoryType].folders.filter(
              (f) => f.id !== folderId
            ),
          },
        },
      })),

    addGenesToCategorySet: (categoryType, geneSetId, genes) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]
        // Check if it's a direct gene set
        const directSet = category.geneSets.find((gs) => gs.id === geneSetId)
        if (directSet) {
          return {
            geneSetCategories: {
              ...state.geneSetCategories,
              [categoryType]: {
                ...category,
                geneSets: category.geneSets.map((gs) =>
                  gs.id === geneSetId
                    ? { ...gs, genes: [...new Set([...gs.genes, ...genes])] }
                    : gs
                ),
              },
            },
          }
        }
        // Check folders
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: category.folders.map((f) => ({
                ...f,
                geneSets: f.geneSets.map((gs) =>
                  gs.id === geneSetId
                    ? { ...gs, genes: [...new Set([...gs.genes, ...genes])] }
                    : gs
                ),
              })),
            },
          },
        }
      }),

    removeGenesFromCategorySet: (categoryType, geneSetId, genes) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]
        const genesSet = new Set(genes)
        // Check if it's a direct gene set
        const directSet = category.geneSets.find((gs) => gs.id === geneSetId)
        if (directSet) {
          return {
            geneSetCategories: {
              ...state.geneSetCategories,
              [categoryType]: {
                ...category,
                geneSets: category.geneSets.map((gs) =>
                  gs.id === geneSetId
                    ? { ...gs, genes: gs.genes.filter((g) => !genesSet.has(g)) }
                    : gs
                ),
              },
            },
          }
        }
        // Check folders
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: category.folders.map((f) => ({
                ...f,
                geneSets: f.geneSets.map((gs) =>
                  gs.id === geneSetId
                    ? { ...gs, genes: gs.genes.filter((g) => !genesSet.has(g)) }
                    : gs
                ),
              })),
            },
          },
        }
      }),

    renameCategoryGeneSet: (categoryType, geneSetId, newName) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]
        // Check if it's a direct gene set
        const directSet = category.geneSets.find((gs) => gs.id === geneSetId)
        if (directSet) {
          return {
            geneSetCategories: {
              ...state.geneSetCategories,
              [categoryType]: {
                ...category,
                geneSets: category.geneSets.map((gs) =>
                  gs.id === geneSetId ? { ...gs, name: newName } : gs
                ),
              },
            },
          }
        }
        // Check folders
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: category.folders.map((f) => ({
                ...f,
                geneSets: f.geneSets.map((gs) =>
                  gs.id === geneSetId ? { ...gs, name: newName } : gs
                ),
              })),
            },
          },
        }
      }),

    toggleSetPinned: (categoryType, folderId, geneSetId) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]
        if (folderId === null) {
          return {
            geneSetCategories: {
              ...state.geneSetCategories,
              [categoryType]: {
                ...category,
                geneSets: category.geneSets.map((gs) =>
                  gs.id === geneSetId ? { ...gs, pinned: !gs.pinned } : gs
                ),
              },
            },
          }
        }
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: category.folders.map((f) =>
                f.id === folderId
                  ? {
                      ...f,
                      geneSets: f.geneSets.map((gs) =>
                        gs.id === geneSetId ? { ...gs, pinned: !gs.pinned } : gs
                      ),
                    }
                  : f
              ),
            },
          },
        }
      }),

    toggleFolderPinned: (categoryType, folderId) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            folders: state.geneSetCategories[categoryType].folders.map((f) =>
              f.id === folderId ? { ...f, pinned: !f.pinned } : f
            ),
          },
        },
      })),

    toggleCategoryVisible: (categoryType) =>
      set((state) => ({
        geneSetCategories: {
          ...state.geneSetCategories,
          [categoryType]: {
            ...state.geneSetCategories[categoryType],
            visible: state.geneSetCategories[categoryType].visible === false,
          },
        },
      })),

    renameFolder: (categoryType, folderId, newName) =>
      set((state) => {
        const trimmed = newName.trim()
        if (!trimmed) return {}
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...state.geneSetCategories[categoryType],
              folders: state.geneSetCategories[categoryType].folders.map((f) =>
                f.id === folderId ? { ...f, name: trimmed } : f
              ),
            },
          },
        }
      }),

    moveGeneSetToFolder: (categoryType, geneSetId, destFolderId, sourceFolderId) =>
      set((state) => {
        // Same-container "move" is not a move — use reorderGeneSet for that.
        // Bail out to avoid accidentally appending the set to the end of its
        // own container.
        if (sourceFolderId === destFolderId) return {}

        const category = state.geneSetCategories[categoryType]

        let movedSet: GeneSet | null = null
        let newTopLevel = category.geneSets
        let newFolders = category.folders

        if (sourceFolderId === null) {
          const idx = category.geneSets.findIndex((gs) => gs.id === geneSetId)
          if (idx === -1) return {}
          movedSet = category.geneSets[idx]
          newTopLevel = [
            ...category.geneSets.slice(0, idx),
            ...category.geneSets.slice(idx + 1),
          ]
        } else {
          const sourceFolder = category.folders.find((f) => f.id === sourceFolderId)
          if (!sourceFolder) return {}
          const idx = sourceFolder.geneSets.findIndex((gs) => gs.id === geneSetId)
          if (idx === -1) return {}
          movedSet = sourceFolder.geneSets[idx]
          newFolders = category.folders.map((f) =>
            f.id === sourceFolderId
              ? {
                  ...f,
                  geneSets: [
                    ...f.geneSets.slice(0, idx),
                    ...f.geneSets.slice(idx + 1),
                  ],
                }
              : f
          )
        }

        if (!movedSet) return {}

        if (destFolderId === null) {
          newTopLevel = [...newTopLevel, movedSet]
        } else {
          newFolders = newFolders.map((f) =>
            f.id === destFolderId
              ? { ...f, geneSets: [...f.geneSets, movedSet!] }
              : f
          )
        }

        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              geneSets: newTopLevel,
              folders: newFolders,
            },
          },
        }
      }),

    reorderGeneSet: (categoryType, folderId, geneSetId, targetIndex) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]

        const reorderList = (list: GeneSet[]): GeneSet[] => {
          const idx = list.findIndex((gs) => gs.id === geneSetId)
          if (idx === -1) return list
          const next = [...list]
          const [moved] = next.splice(idx, 1)
          // targetIndex is expected to be a post-removal index — the caller
          // in GenePanel.tsx adjusts for source-before-target so we can
          // just splice directly. We still clamp defensively.
          const clamped = Math.max(0, Math.min(targetIndex, next.length))
          next.splice(clamped, 0, moved)
          return next
        }

        if (folderId === null) {
          return {
            geneSetCategories: {
              ...state.geneSetCategories,
              [categoryType]: {
                ...category,
                geneSets: reorderList(category.geneSets),
              },
            },
          }
        }
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: category.folders.map((f) =>
                f.id === folderId ? { ...f, geneSets: reorderList(f.geneSets) } : f
              ),
            },
          },
        }
      }),

    reorderFolder: (categoryType, folderId, targetIndex) =>
      set((state) => {
        const category = state.geneSetCategories[categoryType]
        const idx = category.folders.findIndex((f) => f.id === folderId)
        if (idx === -1) return {}
        const next = [...category.folders]
        const [moved] = next.splice(idx, 1)
        const clamped = Math.max(0, Math.min(targetIndex, next.length))
        next.splice(clamped, 0, moved)
        return {
          geneSetCategories: {
            ...state.geneSetCategories,
            [categoryType]: {
              ...category,
              folders: next,
            },
          },
        }
      }),

    // Cell selection actions (per-dataset)
    setSelectedCellIndices: (indices) => set(dsUpdate({ selectedCellIndices: indices })),
    addToSelection: (indices) =>
      set(dsUpdateFn((state) => ({
        selectedCellIndices: [...new Set([...state.selectedCellIndices, ...indices])],
      }))),
    clearSelection: () => set(dsUpdate({ selectedCellIndices: [] })),
    invertSelection: () =>
      set(dsUpdateFn((state) => {
        if (!state.schema || state.selectedCellIndices.length === 0) return {}
        const selected = new Set(state.selectedCellIndices)
        const inverted: number[] = []
        for (let i = 0; i < state.schema.n_cells; i++) {
          if (!selected.has(i)) inverted.push(i)
        }
        return { selectedCellIndices: inverted }
      })),

    // Global-only
    setInteractionMode: (mode) => set({ interactionMode: mode, quiltPhase: 'lasso', quiltUndoDepth: 0 }),
    setQuiltPhase: (phase) => set({ quiltPhase: phase }),
    setQuiltUndoDepth: (depth) => set({ quiltUndoDepth: depth }),

    // Display preferences (per-dataset)
    setDisplayPreferences: (prefs) =>
      set(dsUpdateFn((state) => ({
        displayPreferences: { ...state.displayPreferences, ...prefs },
      }))),

    // Comparison actions (global)
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
    setClusterModalSourceSet: (src) => set({ clusterModalSourceSet: src }),
    setSelectByExpressionSource: (src) => set({ selectByExpressionSource: src }),

    // Line association actions (global)
    setLineAssociationResult: (result) => set({ lineAssociationResult: result }),
    setLineAssociationLoading: (loading) => set({ isLineAssociationLoading: loading }),
    setLineAssociationModalOpen: (open) => set({ isLineAssociationModalOpen: open }),
    setActiveTaskId: (taskId) => set({ activeTaskId: taskId }),

    // Column management actions (per-dataset)
    hideColumn: (name) =>
      set(dsUpdateFn((state) => ({
        hiddenColumns: new Set([...state.hiddenColumns, name]),
      }))),
    showColumn: (name) =>
      set(dsUpdateFn((state) => {
        const next = new Set(state.hiddenColumns)
        next.delete(name)
        return { hiddenColumns: next }
      })),
    setColumnDisplayName: (originalName, displayName) =>
      set(dsUpdateFn((state) => ({
        columnDisplayNames: { ...state.columnDisplayNames, [originalName]: displayName },
      }))),
    clearColumnDisplayName: (originalName) =>
      set(dsUpdateFn((state) => {
        const next = { ...state.columnDisplayNames }
        delete next[originalName]
        return { columnDisplayNames: next }
      })),

    // Cell masking actions (per-dataset)
    setActiveCellsFromSelection: () =>
      set(dsUpdateFn((state) => {
        if (state.selectedCellIndices.length === 0 || !state.schema) return {}
        const mask = new Array(state.schema.n_cells).fill(false)
        state.selectedCellIndices.forEach((i) => {
          mask[i] = true
        })
        return { activeCellMask: mask }
      })),

    addSelectionToActive: () =>
      set(dsUpdateFn((state) => {
        if (state.selectedCellIndices.length === 0 || !state.schema) return {}
        // If no mask exists, all cells are currently active - create mask with all true
        const mask = state.activeCellMask
          ? [...state.activeCellMask]
          : new Array(state.schema.n_cells).fill(true)
        state.selectedCellIndices.forEach((i) => {
          mask[i] = true
        })
        return { activeCellMask: mask }
      })),

    removeSelectionFromActive: () =>
      set(dsUpdateFn((state) => {
        if (state.selectedCellIndices.length === 0 || !state.schema) return {}
        // If no mask exists, all cells are currently active - create mask with all true first
        const mask = state.activeCellMask
          ? [...state.activeCellMask]
          : new Array(state.schema.n_cells).fill(true)
        state.selectedCellIndices.forEach((i) => {
          mask[i] = false
        })
        return { activeCellMask: mask }
      })),

    resetActiveCells: () => set(dsUpdate({ activeCellMask: null })),

    setShowMaskedCells: (show) => set(dsUpdate({ showMaskedCells: show })),

    // Cell ordering actions (per-dataset)
    sortCellsByExpression: () =>
      set(dsUpdateFn((state) => {
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
      })),

    sortCellsByBivariate: () =>
      set(dsUpdateFn((state) => {
        if (!state.bivariateData || !state.schema) return {}
        const { values1, values2 } = state.bivariateData
        const reversed = state.bivariateSortReversed
        // Sort by sum of both normalized values
        // Normal: ascending (low first, high last/on top)
        // Reversed: descending (high first, low last/on top)
        const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
        indices.sort((a, b) => {
          const sumA = (values1[a] ?? 0) + (values2[a] ?? 0)
          const sumB = (values1[b] ?? 0) + (values2[b] ?? 0)
          return reversed ? sumB - sumA : sumA - sumB
        })
        return { cellSortOrder: indices, cellSortVersion: state.cellSortVersion + 1 }
      })),

    toggleBivariateSortOrder: () =>
      set(dsUpdateFn((state) => {
        if (!state.bivariateData || !state.schema) return {}
        const { values1, values2 } = state.bivariateData
        const newReversed = !state.bivariateSortReversed
        // Re-sort with new direction
        const indices = Array.from({ length: state.schema.n_cells }, (_, i) => i)
        indices.sort((a, b) => {
          const sumA = (values1[a] ?? 0) + (values2[a] ?? 0)
          const sumB = (values1[b] ?? 0) + (values2[b] ?? 0)
          return newReversed ? sumB - sumA : sumA - sumB
        })
        return {
          bivariateSortReversed: newReversed,
          cellSortOrder: indices,
          cellSortVersion: state.cellSortVersion + 1,
        }
      })),

    resetCellOrder: () => set(dsUpdate({ cellSortOrder: null, cellSortVersion: 0, bivariateSortReversed: false })),

    // Line/shape drawing actions — mixed (drawnLines is per-dataset, activeLineId is global)
    setDrawTool: (tool) => set({ drawTool: tool }),
    setSelectionTool: (tool) => set({ selectionTool: tool }),

    updateLineAppearance: (id, updates) => {
      const state = get()
      const dsPatch: Partial<DatasetState> = {
        drawnLines: state.drawnLines.map((l) =>
          l.id === id ? { ...l, ...updates } : l
        ),
      }
      const { activeSlot, datasets } = state
      set({
        ...dsPatch,
        datasets: {
          ...datasets,
          [activeSlot]: { ...datasets[activeSlot], ...dsPatch },
        },
      })
    },

    addLine: (name, points, embeddingName, drawType, closed) => {
      const id = `line_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      const newLine: DrawnLine = {
        id,
        name,
        embeddingName,
        points,
        smoothedPoints: null,
        visible: true,
        projections: [],
        drawType: drawType || 'pencil',
        closed: closed || false,
        strokeColor: '#4ecdc4',
        strokeWidth: 2,
        fillColor: null,
      }
      const state = get()
      const dsPatch: Partial<DatasetState> = {
        drawnLines: [...state.drawnLines, newLine],
      }
      const { activeSlot, datasets } = state
      set({
        ...dsPatch,
        activeLineId: id,  // global field
        datasets: {
          ...datasets,
          [activeSlot]: { ...datasets[activeSlot], ...dsPatch },
        },
      })
    },

    removeLine: (id) => {
      const state = get()
      const dsPatch: Partial<DatasetState> = {
        drawnLines: state.drawnLines.filter((l) => l.id !== id),
      }
      const { activeSlot, datasets } = state
      set({
        ...dsPatch,
        activeLineId: state.activeLineId === id ? null : state.activeLineId,  // global
        datasets: {
          ...datasets,
          [activeSlot]: { ...datasets[activeSlot], ...dsPatch },
        },
      })
    },

    // Global-only line actions
    setActiveLine: (id) => set({ activeLineId: id }),

    // Per-dataset line actions
    renameLine: (id, name) =>
      set(dsUpdateFn((state) => ({
        drawnLines: state.drawnLines.map((l) => (l.id === id ? { ...l, name } : l)),
      }))),

    smoothLine: (id) =>
      set(dsUpdateFn((state) => {
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
      })),

    // Global-only
    setLineSmoothingParams: (params) =>
      set((state) => ({
        lineSmoothingParams: { ...state.lineSmoothingParams, ...params },
      })),

    // Per-dataset
    setLineVisibility: (id, visible) =>
      set(dsUpdateFn((state) => ({
        drawnLines: state.drawnLines.map((l) => (l.id === id ? { ...l, visible } : l)),
      }))),

    projectSelectedCellsOntoLine: (lineId) =>
      set(dsUpdateFn((state) => {
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
      })),

    clearLineProjections: (lineId) =>
      set(dsUpdateFn((state) => ({
        drawnLines: state.drawnLines.map((l) =>
          l.id === lineId ? { ...l, projections: [] } : l
        ),
      }))),

    // Import modal actions (global)
    setImportModalOpen: (open) => set({ isImportModalOpen: open }),

    // Gene mask actions
    setGeneMaskModalOpen: (open) => set({ geneMaskModalOpen: open }),
    setGeneMaskConfig: (config) => set(dsUpdate({ geneMaskConfig: config })),
    setPcaSubsets: (subsets) => set(dsUpdate({ pcaSubsets: subsets })),

    setDisplayLayer: (layer) => set(dsUpdate({ displayLayer: layer })),

    setUserConfig: (config) => set({ userConfig: config }),

    applyDisplayDefaultsFromConfig: () => {
      const state = get()
      const overrides = displayPreferencesFromConfig(state.userConfig)
      if (Object.keys(overrides).length === 0) return
      const newDatasets = {
        ...state.datasets,
        primary: {
          ...state.datasets.primary,
          displayPreferences: { ...state.datasets.primary.displayPreferences, ...overrides },
        },
        secondary: {
          ...state.datasets.secondary,
          displayPreferences: { ...state.datasets.secondary.displayPreferences, ...overrides },
        },
      }
      set({
        datasets: newDatasets,
        displayPreferences: { ...state.displayPreferences, ...overrides },
      })
    },

    // Scanpy modal actions
    setScanpyModalOpen: (open) => set({ isScanpyModalOpen: open }),  // global
    setScanpyActionHistory: (history) => set(dsUpdate({ scanpyActionHistory: history })),  // per-dataset
    addScanpyAction: (action) =>
      set(dsUpdateFn((state) => ({
        scanpyActionHistory: [...state.scanpyActionHistory, action],
      }))),

    // Observable summaries refresh (per-dataset)
    refreshObsSummaries: () =>
      set(dsUpdateFn((state) => ({ obsSummariesVersion: state.obsSummariesVersion + 1 }))),

    // Marker genes modal actions (global)
    setMarkerGenesModalOpen: (open) => set({ isMarkerGenesModalOpen: open }),
    setMarkerGenesColumn: (column) => set({ markerGenesColumn: column }),

    // Var identifier switching actions
    setVarIdentifierColumns: (columns) => set(dsUpdate({ varIdentifierColumns: columns })),
    setCurrentVarIndex: (name) => set(dsUpdate({ currentVarIndex: name })),

    remapAllGeneNames: (oldToNew) => set((state) => {
      // Remap selectedGenes
      const newSelectedGenes = state.selectedGenes.map(g => oldToNew.get(g) ?? g)

      // Deep-clone and remap all gene set categories
      const newCategories = { ...state.geneSetCategories }
      for (const catType of Object.keys(newCategories) as Array<keyof typeof newCategories>) {
        const cat = newCategories[catType]
        newCategories[catType] = {
          ...cat,
          geneSets: cat.geneSets.map(gs => ({
            ...gs,
            genes: gs.genes.map(g => oldToNew.get(g) ?? g),
          })),
          folders: cat.folders.map(folder => ({
            ...folder,
            geneSets: folder.geneSets.map(gs => ({
              ...gs,
              genes: gs.genes.map(g => oldToNew.get(g) ?? g),
            })),
          })),
        }
      }

      // Also remap legacy geneSets
      const newGeneSets = state.geneSets.map(gs => ({
        ...gs,
        genes: gs.genes.map(g => oldToNew.get(g) ?? g),
      }))

      return {
        selectedGenes: newSelectedGenes,
        geneSetCategories: newCategories,
        geneSets: newGeneSets,
      }
    }),

    // Checkbox-based comparison actions (global)
    toggleComparisonCategory: (column, category) =>
      set((state) => {
        if (state.comparisonCheckedColumn !== column) {
          // Switching columns: clear previous and start fresh
          return {
            comparisonCheckedColumn: column,
            comparisonCheckedCategories: new Set([category]),
          }
        }
        const next = new Set(state.comparisonCheckedCategories)
        if (next.has(category)) {
          next.delete(category)
        } else {
          next.add(category)
        }
        return {
          comparisonCheckedColumn: next.size > 0 ? column : null,
          comparisonCheckedCategories: next,
        }
      }),
    clearComparisonCategories: () =>
      set({ comparisonCheckedColumn: null, comparisonCheckedCategories: new Set<string>() }),

    // Heatmap tab actions (global)
    setCenterPanelView: (view) => set({ centerPanelView: view }),
    setHeatmapConfig: (config) => set({ heatmapConfig: config }),

    // Layout mode actions
    setLayoutMode: (mode) => set({ layoutMode: mode }),

    // Multi-dataset actions
    setActiveSlot: (slot) => {
      const state = get()
      if (slot === state.activeSlot) return
      set({
        activeSlot: slot,
        ...syncFlatFields(slot, state.datasets),
      })
    },

    loadDatasetIntoSlot: (slot, schema) => {
      const state = get()
      // Apply user-config display defaults so a freshly loaded dataset shows
      // the user's preferred point size / colormap / etc. on the first frame.
      const overrides = displayPreferencesFromConfig(state.userConfig)
      const freshDs = createDefaultDatasetState(overrides)
      freshDs.schema = schema
      // Auto-select embedding by preference: spatial > umap > pca > first
      if (schema.embeddings.length > 0) {
        const preferred = ['spatial', 'umap', 'pca']
        const lower = schema.embeddings.map(e => e.toLowerCase())
        const pick = preferred.find(p => lower.some(l => l.includes(p)))
        const idx = pick != null ? lower.findIndex(l => l.includes(pick)) : 0
        freshDs.selectedEmbedding = schema.embeddings[idx]
      }
      const newDatasets = { ...state.datasets, [slot]: freshDs }
      if (slot === state.activeSlot) {
        set({ datasets: newDatasets, ...syncFlatFields(slot, newDatasets) })
      } else {
        set({ datasets: newDatasets })
      }
    },

    patchSlotState: (slot, patch) => {
      const state = get()
      const newDatasets = {
        ...state.datasets,
        [slot]: { ...state.datasets[slot], ...patch },
      }
      if (slot === state.activeSlot) {
        set({ ...patch, datasets: newDatasets } as Partial<AppState>)
      } else {
        set({ datasets: newDatasets })
      }
    },
  }
})

/** Convenience hook: returns the DatasetState for the active slot. */
export function useActiveDataset(): DatasetState {
  return useStore((state) => state.datasets[state.activeSlot])
}
