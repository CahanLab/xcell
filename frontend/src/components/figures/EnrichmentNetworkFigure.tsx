import { forwardRef, useMemo } from 'react'
import { divergingColor, truncate, type EnrichmentNetworkData } from '../../lib/figures'

/** Bipartite graph of groups (squares) and gene sets (circles): enrichment
 *  edges coloured by sign, optional dashed overlap edges between sets. */

export interface EnrichmentNetworkFigureProps {
  data: EnrichmentNetworkData
  params: Record<string, unknown>
  title?: string
  width?: number
  height?: number
  background?: string
}

const FONT = 'Inter, system-ui, sans-serif'
const ACCENT = '#4ecdc4'

export const EnrichmentNetworkFigure = forwardRef<SVGSVGElement, EnrichmentNetworkFigureProps>(function EnrichmentNetworkFigure(
  { data, params, title, width = 760, height = 560, background = '#16213e' },
  ref,
) {
  const labelMax = Math.max(8, Number(params.label_max_chars ?? 30))
  const colormap = String(params.colormap ?? 'rdbu')
  const sizeBy = String(params.node_size_by ?? 'degree')
  const widthBy = String(params.edge_width_by ?? 'value')
  const titleH = title ? 24 : 0
  const pad = 70
  const W = Math.max(320, width, title ? 24 + title.length * 7 : 0), H = Math.max(240, height)

  const { vmax, maxSize } = useMemo(() => {
    let vm = 0, ms = 0
    for (const e of data.edges) if (e.kind === 'enrichment') vm = Math.max(vm, Math.abs(e.value))
    for (const n of data.nodes) if (n.kind === 'set') ms = Math.max(ms, sizeBy === 'n_set' ? n.size : n.degree)
    return { vmax: vm || 1, maxSize: ms || 1 }
  }, [data, sizeBy])

  const [x0, y0, x1, y1] = data.bounds
  const sx = (x: number) => pad + ((x - x0) / Math.max(1e-9, x1 - x0)) * (W - 2 * pad)
  const sy = (y: number) => titleH + pad * 0.6 + ((y - y0) / Math.max(1e-9, y1 - y0)) * (H - titleH - pad * 1.2)
  const byId = new Map(data.nodes.map((n) => [n.id, n]))
  const radius = (n: EnrichmentNetworkData['nodes'][number]) => {
    if (n.kind === 'group') return 9
    if (sizeBy === 'constant') return 6
    const s = sizeBy === 'n_set' ? n.size : n.degree
    return 4 + 8 * Math.sqrt(Math.max(0, s) / maxSize)
  }

  if (data.nodes.length === 0 || data.edges.length === 0) {
    return (
      <svg ref={ref} data-figure-svg width={Math.max(480, width)} height={80} style={{ background, fontFamily: FONT }}>
        {title && <text x={12} y={17} fill="#eee" fontSize={13} fontWeight={600}>{title}</text>}
        <text x={12} y={titleH + 30} fill="#aaa" fontSize={12}>{data.note ?? 'Nothing to draw'}</text>
      </svg>
    )
  }

  const overlap = data.edges.filter((e) => e.kind === 'overlap')
  const enrich = data.edges.filter((e) => e.kind === 'enrichment')

  return (
    <svg ref={ref} data-figure-svg width={W} height={H + 30} style={{ background, fontFamily: FONT, display: 'block' }}>
      {title && <text x={12} y={17} fill="#eee" fontSize={13} fontWeight={600}>{title}</text>}
      {overlap.map((e, i) => {
        const a = byId.get(e.source)!, b = byId.get(e.target)!
        return <line key={`o${i}`} x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)} stroke="#8892a6" strokeWidth={0.8 + 2 * e.value} strokeDasharray="4 3" opacity={0.8}>
          <title>{`${a.label} ~ ${b.label}: Jaccard ${e.value.toFixed(2)}`}</title>
        </line>
      })}
      {enrich.map((e, i) => {
        const a = byId.get(e.source)!, b = byId.get(e.target)!
        const w = widthBy === 'value' ? 0.8 + 4 * Math.abs(e.value) / vmax : 1.5
        return <line key={`e${i}`} x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)} stroke={divergingColor(e.value, vmax, colormap)} strokeWidth={w} opacity={0.9}>
          <title>{`${a.label} → ${b.label}: ${e.value.toFixed(2)}${e.padj != null ? `, padj ${e.padj.toExponential(1)}` : ''}`}</title>
        </line>
      })}
      {data.nodes.map((n) => {
        const cx = sx(n.x), cy = sy(n.y), r = radius(n)
        return (
          <g key={n.id}>
            {n.kind === 'group'
              ? <rect x={cx - r} y={cy - r} width={2 * r} height={2 * r} fill={ACCENT} stroke="#0f1625" strokeWidth={1} />
              : <circle cx={cx} cy={cy} r={r} fill="#d7dbe6" stroke="#0f1625" strokeWidth={1} />}
            <title>{n.kind === 'group' ? `${n.label} (group, ${n.degree} sets)` : `${n.label} (${n.library ?? ''}, ${n.size} genes, ${n.degree} links)`}</title>
            <text x={cx + r + 3} y={cy + 3.5} fill={n.kind === 'group' ? '#eee' : '#ccc'} fontSize={n.kind === 'group' ? 11 : 9} fontWeight={n.kind === 'group' ? 700 : 400} stroke={background} strokeWidth={3} paintOrder="stroke">
              {truncate(n.label, labelMax)}
            </text>
          </g>
        )
      })}
      {/* legend */}
      <g transform={`translate(12, ${H + 8})`}>
        <rect x={0} y={-8} width={12} height={12} fill={ACCENT} />
        <text x={16} y={2} fill="#aaa" fontSize={10}>group</text>
        <circle cx={60} cy={-2} r={5} fill="#d7dbe6" />
        <text x={70} y={2} fill="#aaa" fontSize={10}>gene set</text>
        <line x1={130} x2={150} y1={-2} y2={-2} stroke={divergingColor(vmax, vmax, colormap)} strokeWidth={2.5} />
        <text x={154} y={2} fill="#aaa" fontSize={10}>up</text>
        <line x1={180} x2={200} y1={-2} y2={-2} stroke={divergingColor(-vmax, vmax, colormap)} strokeWidth={2.5} />
        <text x={204} y={2} fill="#aaa" fontSize={10}>down</text>
        {overlap.length > 0 && <>
          <line x1={240} x2={260} y1={-2} y2={-2} stroke="#8892a6" strokeWidth={1.5} strokeDasharray="4 3" />
          <text x={264} y={2} fill="#aaa" fontSize={10}>set overlap</text>
        </>}
        {data.note && <text x={0} y={16} fill="#e9a23b" fontSize={9}>{data.note}</text>}
      </g>
    </svg>
  )
})

export default EnrichmentNetworkFigure
