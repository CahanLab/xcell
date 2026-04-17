/**
 * Centralized UI messages for xcell frontend.
 * Edit strings here instead of hunting through component files.
 */

export const MESSAGES = {
  // Center panel / scatter plot
  noDataLoaded: 'No data loaded. Click Load to open a dataset.',
  noEmbedding: 'Dataset loaded but has no embeddings. Use Scanpy → Cell Analysis to run PCA and UMAP first.',
  noEmbeddingSecondary: 'No embedding loaded',
  loading: 'Loading...',

  // Load modal
  loadBrowseLoading: 'Loading...',
  loadBrowseEmpty: 'No folders or data files here',
  loadPathPlaceholder: '/path/to/data.h5ad or .h5',
  loadPathLabel: 'Or enter path directly:',

  // Expression errors
  geneNotFoundPrefix: 'Gene not found',

  // Generic
  errorPrefix: 'Error:',

  // Analysis cancellation
  analysisCancelled: 'Analysis cancelled',
  analysisCancelledDetail: 'The operation was stopped. No changes were made.',
  taskNotFound: 'Task not found — it may have expired. Try running again.',

  // SelectByExpressionModal
  selectByExpression: {
    titleGene: (gene: string) => `Select cells by ${gene}`,
    titleGeneSet: (name: string) => `Select cells by ${name}`,
    loading: 'Loading expression values…',
    zeroVariance: (value: number) =>
      `All cells have the same value (${value.toFixed(2)}) — nothing to threshold on.`,
    matchCounter: (match: number, total: number) =>
      `Matching: ${match.toLocaleString()} / ${total.toLocaleString()} cells`,
    annotationCollision: (name: string) =>
      `An annotation named "${name}" already exists — choose a different name.`,
    emptyContextTooltip: 'Current selection is empty',
    zeroVarianceTooltip: 'Cannot threshold: all cells have the same value',
    collisionTooltip: 'Annotation name already exists',
    emptyNameError: 'Please enter an annotation name.',
    emptyContextError: 'Current selection is empty — choose All cells or make a selection.',
    labelingButton: 'Labeling…',
    applyButton: 'Apply',
    cancelButton: 'Cancel',
    closeButton: 'Close',
    openDiffExpButton: 'Open Diff Exp ▸',
    successFooter: (highCount: number, highLabel: string, lowCount: number, lowLabel: string) =>
      `Labeled ${highCount.toLocaleString()} cells ${highLabel}, ${lowCount.toLocaleString()} cells ${lowLabel}.`,
    thresholdModeLabel: 'Threshold mode:',
    thresholdInputLabel: 'Threshold: ',
    loInputLabel: 'Lo: ',
    hiInputLabel: 'Hi: ',
    actionLabel: 'Action:',
    updateSelectionLabel: 'Update selection',
    labelCellsLabel: 'Label cells',
    annotationNameLabel: 'Annotation name:',
    moreOptionsOpen: '▾ More options',
    moreOptionsClosed: '▸ More options',
    highLabelFieldLabel: 'High label:',
    lowLabelFieldLabel: 'Low label:',
    contextLabel: 'Context:',
    contextAllCellsLabel: 'All cells',
    contextCurrentSelectionLabel: (n: number) =>
      `Current selection (${n.toLocaleString()} cells)`,
    noExistingSelectionTooltip: 'No existing selection',
    defaultHighLabel: 'high',
    defaultLowLabel: 'low',
    failedToLabelCells: 'Failed to label cells',
  },

  // GeneMaskModal
  geneMask: {
    title: 'Gene Mask',
    description: 'Filter genes by .var boolean columns.',
    noBoolColumns: 'No boolean .var columns found. Run Highly Variable Genes or Spatial Autocorrelation first.',
    columnLabel: (n_true: number, n_total: number) =>
      `${n_true.toLocaleString()} / ${n_total.toLocaleString()} True`,
    stateOff: 'Off',
    stateKeep: 'Keep',
    stateHide: 'Hide',
    combineLabel: 'Combine Keep columns:',
    combineAny: 'Match ANY',
    combineAll: 'Match ALL',
    previewLabel: (visible: number, total: number) =>
      `Preview: ${visible.toLocaleString()} of ${total.toLocaleString()} visible`,
    clearButton: 'Clear',
    cancelButton: 'Cancel',
    applyButton: 'Apply',
    noneVisibleError: 'Mask would leave zero visible genes. Adjust your selection.',
    coloringClearedToast: (gene: string) =>
      `${gene} is masked; coloring cleared.`,
    allMaskedInSetToast: 'All genes in this set are masked — score is zero.',
    maskClearedAfterFilter: 'Gene mask was cleared because referenced columns were removed.',
    hiddenSuffix: (n: number) => `(${n.toLocaleString()} hidden)`,
    visibleBadge: (visible: number, total: number) =>
      `${visible.toLocaleString()} / ${total.toLocaleString()}`,
    menuItem: 'Gene mask…',
  },

  // PCA Loadings Explorer
  pcaLoadings: {
    description:
      'Inspect top-loading genes per PC, then create a derived subset that excludes selected PCs for downstream analysis.',
    topNLabel: 'Top-N genes per side:',
    colPC: 'PC',
    colVariance: 'Var %',
    colPositive: 'Top + loading genes',
    colNegative: 'Top − loading genes',
    suffixLabel: 'Suffix (optional):',
    suffixAutoPrefix: 'auto: ',
    createButton: 'Create PC subset →',
    createBusyButton: 'Creating…',
    deleteButton: '✕',
    checkedSummary: (nChecked: number, nKept: number) =>
      `${nChecked} PC${nChecked === 1 ? '' : 's'} checked, ${nKept} would remain`,
    noneChecked: 'Check at least one PC to drop',
    allDropped: 'Cannot drop all PCs',
    prereqMissing: 'Run PCA first to explore loadings.',
    empty: 'No PC loadings available — re-run PCA to populate loadings.',
    subsetCaption: (loaded: number, total: number) =>
      `Showing ${loaded.toLocaleString()} of ${total.toLocaleString()} genes (subset PCA)`,
    loading: 'Loading loadings…',
    fetchError: 'Failed to load PC loadings.',
    existingSubsetsHeader: 'Existing PC subsets',
    noSubsets: 'No derived PC subsets yet.',
    subsetSummary: (suffix: string, nKept: number, dropped: number[]) =>
      dropped.length > 0
        ? `${suffix} · ${nKept} kept · dropped ${dropped.join(', ')}`
        : `${suffix} · ${nKept} kept`,
    createdToast: (suffix: string, nKept: number) =>
      `Created PC subset "${suffix}" (${nKept} PCs kept)`,
    collisionToast: (suffix: string) =>
      `A PC subset named "${suffix}" already exists.`,
    clearedToast: (n: number) =>
      `PCA recomputed — cleared ${n.toLocaleString()} derived PC subset${n === 1 ? '' : 's'}.`,
    neighborsSourceLabel: 'PC source',
    neighborsSourceDescription:
      'Which PC embedding to use. Create derived subsets via PCA Loadings.',
    neighborsSourceBaseLabel: 'X_pca (all PCs)',
  },
} as const
