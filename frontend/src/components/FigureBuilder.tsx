import { useState, useMemo } from 'react'
import { useStore, FigurePanel as PanelType, FigureColorMode, ColorScale } from '../store'
import { useObsSummaries } from '../hooks/useData'
import { flattenGeneSets } from './GenePanel'
import FigurePanel from './FigurePanel'
import { exportFigureAsPng } from '../utils/exportFigure'

const LAYOUT_PRESETS: { label: string; rows: number; cols: number }[] = [
  { label: '1×1', rows: 1, cols: 1 },
  { label: '1×2', rows: 1, cols: 2 },
  { label: '2×1', rows: 2, cols: 1 },
  { label: '1×3', rows: 1, cols: 3 },
  { label: '2×2', rows: 2, cols: 2 },
  { label: '2×3', rows: 2, cols: 3 },
  { label: '3×3', rows: 3, cols: 3 },
]

const COLOR_SCALES: ColorScale[] = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'coolwarm', 'blues', 'reds']

export default function FigureBuilder() {
  const figure = useStore((s) => s.activeFigure)
  const updateFigure = useStore((s) => s.updateFigure)
  const updateFigurePanel = useStore((s) => s.updateFigurePanel)
  const setFigureLayout = useStore((s) => s.setFigureLayout)
  const closeFigure = useStore((s) => s.closeFigure)
  const setCenterPanelView = useStore((s) => s.setCenterPanelView)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const { summaries } = useObsSummaries()

  const [activePanelId, setActivePanelId] = useState<string | null>(null)
  const [exportScale, setExportScale] = useState<number>(2)
  const [isExporting, setIsExporting] = useState(false)

  const allGeneSets = useMemo(() => flattenGeneSets(geneSetCategories).filter((g) => g.genes.length > 0), [geneSetCategories])
  const categoricalCols = useMemo(() => summaries.filter((s) => s.dtype === 'category' || s.dtype === 'string'), [summaries])
  const numericCols = useMemo(() => summaries.filter((s) => s.dtype === 'numeric'), [summaries])

  if (!figure) {
    return (
      <div style={emptyStyle}>
        <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '8px' }}>No figure open.</div>
        <div style={{ fontSize: '12px', color: '#888' }}>
          Select cells in the Embedding tab and click <strong>Create figure</strong> in the Cells panel to start.
        </div>
      </div>
    )
  }

  const activePanel = figure.panels.find((p) => p.id === activePanelId) || figure.panels[0]

  const handleExport = async () => {
    setIsExporting(true)
    try {
      await exportFigureAsPng(figure, exportScale)
    } finally {
      setIsExporting(false)
    }
  }

  const handleClose = () => {
    closeFigure()
    setCenterPanelView('scatter')
  }

  const handleLayoutChange = (rows: number, cols: number) => {
    setFigureLayout(rows, cols)
    if (activePanelId && figure.panels.findIndex((p) => p.id === activePanelId) >= rows * cols) {
      setActivePanelId(null)
    }
  }

  return (
    <div style={containerStyle}>
      {/* Top toolbar */}
      <div style={toolbarStyle}>
        <input
          type="text"
          placeholder="Figure title (optional)"
          value={figure.title}
          onChange={(e) => updateFigure({ title: e.target.value })}
          style={titleInputStyle}
        />
        <span style={{ ...labelStyle, marginLeft: '8px' }}>Layout</span>
        <select
          value={`${figure.rows}x${figure.cols}`}
          onChange={(e) => {
            const [r, c] = e.target.value.split('x').map(Number)
            handleLayoutChange(r, c)
          }}
          style={selectStyle}
        >
          {LAYOUT_PRESETS.map((p) => (
            <option key={p.label} value={`${p.rows}x${p.cols}`}>{p.label}</option>
          ))}
        </select>
        <span style={{ ...labelStyle, marginLeft: '12px' }}>Background</span>
        <input
          type="color"
          value={figure.background}
          onChange={(e) => updateFigure({ background: e.target.value })}
          style={colorInputStyle}
        />
        <span style={{ ...labelStyle, marginLeft: '12px' }}>Point size</span>
        <input
          type="range"
          min={1}
          max={10}
          step={0.5}
          value={figure.pointSize}
          onChange={(e) => updateFigure({ pointSize: parseFloat(e.target.value) })}
          style={{ width: '90px' }}
          title="Point radius in CSS pixels (shared across all panels)"
        />
        <span style={{ ...labelStyle, fontVariantNumeric: 'tabular-nums', width: '24px' }}>{figure.pointSize.toFixed(1)}</span>
        <span style={{ ...labelStyle, marginLeft: '8px' }}>Opacity</span>
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={figure.pointOpacity}
          onChange={(e) => updateFigure({ pointOpacity: parseFloat(e.target.value) })}
          style={{ width: '90px' }}
          title="Point opacity (shared across all panels)"
        />
        <span style={{ ...labelStyle, fontVariantNumeric: 'tabular-nums', width: '30px' }}>{Math.round(figure.pointOpacity * 100)}%</span>
        <div style={{ flex: 1 }} />
        <span style={labelStyle}>Export DPI</span>
        <select value={exportScale} onChange={(e) => setExportScale(Number(e.target.value))} style={selectStyle}>
          <option value={1}>1× (screen)</option>
          <option value={2}>2× (~150 DPI)</option>
          <option value={3}>3× (~225 DPI)</option>
          <option value={4}>4× (~300 DPI)</option>
        </select>
        <button onClick={handleExport} disabled={isExporting} style={primaryButtonStyle}>
          {isExporting ? 'Exporting…' : 'Export PNG'}
        </button>
        <button onClick={handleClose} style={secondaryButtonStyle}>Close</button>
      </div>

      {/* Body: grid + sidebar */}
      <div style={bodyStyle}>
        {/* Grid of panels */}
        <div
          data-figure-grid="true"
          style={{
            flex: 1,
            backgroundColor: figure.background,
            display: 'grid',
            gridTemplateRows: `repeat(${figure.rows}, 1fr)`,
            gridTemplateColumns: `repeat(${figure.cols}, 1fr)`,
            gap: '8px',
            padding: figure.title ? '36px 12px 12px' : '12px',
            position: 'relative',
            overflow: 'hidden',
            minWidth: 0,
          }}
        >
          {figure.title && (
            <div
              style={{
                position: 'absolute',
                top: '8px',
                left: 0,
                right: 0,
                textAlign: 'center',
                color: '#fff',
                fontSize: '16px',
                fontWeight: 600,
                textShadow: '0 1px 2px rgba(0,0,0,0.8)',
                pointerEvents: 'none',
              }}
            >
              {figure.title}
            </div>
          )}
          {figure.panels.map((p) => (
            <div
              key={p.id}
              data-figure-panel="true"
              data-panel-id={p.id}
              onClick={() => setActivePanelId(p.id)}
              style={{
                position: 'relative',
                cursor: 'pointer',
                outline: activePanelId === p.id ? '2px solid #4ecdc4' : 'none',
                outlineOffset: '-2px',
              }}
            >
              <FigurePanel figure={figure} panel={p} />
            </div>
          ))}
        </div>

        {/* Sidebar: per-panel controls */}
        <div style={sidebarStyle}>
          {activePanel ? (
            <PanelControls
              panel={activePanel}
              allGeneSets={allGeneSets}
              categoricalCols={categoricalCols.map((c) => c.name)}
              numericCols={numericCols.map((c) => c.name)}
              onUpdate={(patch) => updateFigurePanel(activePanel.id, patch)}
            />
          ) : (
            <div style={{ fontSize: '11px', color: '#888' }}>Click a panel to edit its settings.</div>
          )}
        </div>
      </div>

      <div style={footerStyle}>
        {figure.cellIndices.length.toLocaleString()} cells · embedding: {figure.embeddingName}
      </div>
    </div>
  )
}

function PanelControls({
  panel,
  allGeneSets,
  categoricalCols,
  numericCols,
  onUpdate,
}: {
  panel: PanelType
  allGeneSets: { id: string; name: string; genes: string[] }[]
  categoricalCols: string[]
  numericCols: string[]
  onUpdate: (patch: Partial<Omit<PanelType, 'id'>>) => void
}) {
  const [customGene, setCustomGene] = useState<string>('')

  const handleColorModeChange = (mode: FigureColorMode) => {
    // Reset source fields when switching modes
    onUpdate({
      colorMode: mode,
      selectedGenes: [],
      selectedGeneSetName: null,
      selectedColorColumn: null,
    })
    setCustomGene('')
  }

  return (
    <div>
      <div style={{ fontSize: '12px', fontWeight: 600, color: '#aaa', textTransform: 'uppercase', marginBottom: '8px' }}>
        Selected panel
      </div>

      <ControlRow label="Title">
        <input
          type="text"
          value={panel.title}
          onChange={(e) => onUpdate({ title: e.target.value })}
          style={inputStyle}
        />
      </ControlRow>

      <ControlRow label="Color">
        <select value={panel.colorMode} onChange={(e) => handleColorModeChange(e.target.value as FigureColorMode)} style={inputStyle}>
          <option value="none">none</option>
          <option value="expression">gene / gene set</option>
          <option value="metadata">metadata</option>
        </select>
      </ControlRow>

      {panel.colorMode === 'expression' && (
        <>
          <ControlRow label="Gene set">
            <select
              value={panel.selectedGeneSetName ?? ''}
              onChange={(e) => {
                const setName = e.target.value
                if (!setName) {
                  onUpdate({ selectedGenes: [], selectedGeneSetName: null })
                  return
                }
                const gs = allGeneSets.find((g) => g.name === setName)
                if (gs) {
                  onUpdate({ selectedGenes: gs.genes, selectedGeneSetName: gs.name })
                  setCustomGene('')
                }
              }}
              style={inputStyle}
            >
              <option value="">— pick gene set —</option>
              {allGeneSets.map((gs) => (
                <option key={gs.id} value={gs.name}>{gs.name} ({gs.genes.length})</option>
              ))}
            </select>
          </ControlRow>
          <ControlRow label="Or gene">
            <input
              type="text"
              placeholder="e.g. Prrx1"
              value={customGene}
              onChange={(e) => setCustomGene(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && customGene.trim()) {
                  onUpdate({ selectedGenes: [customGene.trim()], selectedGeneSetName: null })
                }
              }}
              onBlur={() => {
                if (customGene.trim() && (panel.selectedGenes.length !== 1 || panel.selectedGenes[0] !== customGene.trim())) {
                  onUpdate({ selectedGenes: [customGene.trim()], selectedGeneSetName: null })
                }
              }}
              style={inputStyle}
            />
          </ControlRow>
          {panel.selectedGenes.length > 0 && !panel.selectedGeneSetName && (
            <div style={hintStyle}>Showing: {panel.selectedGenes.join(', ')}</div>
          )}
          <ControlRow label="Color scale">
            <select value={panel.colorScale} onChange={(e) => onUpdate({ colorScale: e.target.value as ColorScale })} style={inputStyle}>
              {COLOR_SCALES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </ControlRow>
          <ControlRow label="Transform">
            <select value={panel.expressionTransform} onChange={(e) => onUpdate({ expressionTransform: e.target.value as 'none' | 'log1p' })} style={inputStyle}>
              <option value="none">none</option>
              <option value="log1p">log1p</option>
            </select>
          </ControlRow>
        </>
      )}

      {panel.colorMode === 'metadata' && (
        <>
          <ControlRow label="Column">
            <select value={panel.selectedColorColumn ?? ''} onChange={(e) => onUpdate({ selectedColorColumn: e.target.value || null })} style={inputStyle}>
              <option value="">— pick column —</option>
              {categoricalCols.length > 0 && <optgroup label="Categorical">
                {categoricalCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </optgroup>}
              {numericCols.length > 0 && <optgroup label="Numeric">
                {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </optgroup>}
            </select>
          </ControlRow>
          {panel.selectedColorColumn && numericCols.includes(panel.selectedColorColumn) && (
            <ControlRow label="Color scale">
              <select value={panel.colorScale} onChange={(e) => onUpdate({ colorScale: e.target.value as ColorScale })} style={inputStyle}>
                {COLOR_SCALES.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </ControlRow>
          )}
        </>
      )}

      <div style={{ height: '1px', backgroundColor: '#333', margin: '12px 0' }} />

      <ControlRow label="Background">
        <input
          type="color"
          value={panel.background}
          onChange={(e) => onUpdate({ background: e.target.value })}
          style={colorInputStyle}
        />
      </ControlRow>
      <ControlRow label="Border">
        <input
          type="checkbox"
          checked={panel.showBorder}
          onChange={(e) => onUpdate({ showBorder: e.target.checked })}
        />
      </ControlRow>
    </div>
  )
}

function ControlRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
      <span style={{ ...labelStyle, width: '70px', flexShrink: 0 }}>{label}</span>
      {children}
    </div>
  )
}

const containerStyle = {
  display: 'flex',
  flexDirection: 'column' as const,
  width: '100%',
  height: '100%',
  backgroundColor: '#0f1625',
}

const toolbarStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  padding: '8px 12px',
  borderBottom: '1px solid #1a1a2e',
  backgroundColor: '#16213e',
}

const bodyStyle = {
  flex: 1,
  display: 'flex',
  minHeight: 0,
  overflow: 'hidden',
}

const sidebarStyle = {
  width: '280px',
  borderLeft: '1px solid #1a1a2e',
  backgroundColor: '#16213e',
  padding: '12px',
  overflowY: 'auto' as const,
  flexShrink: 0,
}

const footerStyle = {
  padding: '4px 12px',
  fontSize: '10px',
  color: '#666',
  borderTop: '1px solid #1a1a2e',
  backgroundColor: '#16213e',
}

const emptyStyle = {
  display: 'flex',
  flexDirection: 'column' as const,
  alignItems: 'center',
  justifyContent: 'center',
  width: '100%',
  height: '100%',
  backgroundColor: '#0f1625',
  padding: '32px',
}

const labelStyle = { fontSize: '11px', color: '#888' }
const hintStyle = { fontSize: '10px', color: '#888', marginBottom: '6px', marginLeft: '76px' }

const inputStyle = {
  flex: 1,
  minWidth: 0,
  padding: '4px 6px',
  fontSize: '11px',
  backgroundColor: '#1a1a2e',
  color: '#eee',
  border: '1px solid #0f3460',
  borderRadius: '3px',
}

const titleInputStyle = {
  ...inputStyle,
  flex: 'none' as const,
  width: '180px',
}

const selectStyle = {
  ...inputStyle,
  flex: 'none' as const,
  cursor: 'pointer',
}

const colorInputStyle = {
  width: '28px',
  height: '22px',
  padding: 0,
  border: '1px solid #0f3460',
  borderRadius: '3px',
  backgroundColor: '#1a1a2e',
  cursor: 'pointer',
}

const primaryButtonStyle = {
  padding: '4px 12px',
  fontSize: '12px',
  fontWeight: 500,
  backgroundColor: '#4ecdc4',
  color: '#000',
  border: 'none',
  borderRadius: '3px',
  cursor: 'pointer',
}

const secondaryButtonStyle = {
  padding: '4px 12px',
  fontSize: '12px',
  backgroundColor: 'transparent',
  color: '#aaa',
  border: '1px solid #444',
  borderRadius: '3px',
  cursor: 'pointer',
}
