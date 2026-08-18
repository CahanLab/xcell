/** Default for the split view's second pane: the view that complements the
 * one already shown. Beside a spatial map you want the expression manifold;
 * beside an expression map you want the tissue. */
export function pickSecondEmbedding(
  embeddings: string[],
  current: string | null,
): string | null {
  const others = embeddings.filter((e) => e !== current)
  if (others.length === 0) return null
  const currentIsSpatial = current != null && current.toLowerCase().includes('spatial')
  const preferred = currentIsSpatial ? ['umap', 'pca', 'spatial'] : ['spatial', 'umap', 'pca']
  for (const kind of preferred) {
    const hit = others.find((e) => e.toLowerCase().includes(kind))
    if (hit) return hit
  }
  return others[0]
}
