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
} as const
