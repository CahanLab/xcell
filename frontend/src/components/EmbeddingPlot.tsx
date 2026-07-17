import type { ComponentProps } from 'react'
import { useStore } from '../store'
import ScatterPlot from './ScatterPlot'
import ScatterPlot3D from './ScatterPlot3D'

// Picks the 2D or 3D renderer for an embedding. Takes the exact same props the
// three <ScatterPlot> call sites already pass; renders ScatterPlot3D only when
// the user is in 3D view mode AND the embedding actually carries a z column
// (useEmbedding populates embedding.z only in 3D with a z dim chosen), otherwise
// the unchanged 2D ScatterPlot. ScatterPlot3D ignores the 2D-only callbacks.
type Props = ComponentProps<typeof ScatterPlot>

export default function EmbeddingPlot(props: Props) {
  const viewMode = useStore((s) => s.viewMode)
  const use3D = viewMode === '3d' && !!props.embedding.z && props.embedding.z.length > 0
  return use3D ? <ScatterPlot3D {...props} /> : <ScatterPlot {...props} />
}
