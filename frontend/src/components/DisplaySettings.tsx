import { useState, useRef, useEffect } from 'react'
import { useStore, ColorScale, ExpressionTransform, BivariateColormap, GeneSetScoringMethod } from '../store'
import { getBivariateColor } from './ScatterPlot'

const styles = {
  container: {
    position: 'relative' as const,
  },
  button: {
    padding: '6px 12px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  buttonActive: {
    backgroundColor: '#1a3a5c',
    borderColor: '#e94560',
  },
  panel: {
    position: 'absolute' as const,
    top: '100%',
    right: 0,
    marginTop: '8px',
    width: '280px',
    backgroundColor: '#16213e',
    border: '1px solid #0f3460',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    zIndex: 1000,
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    borderBottom: '1px solid #0f3460',
    fontSize: '13px',
    fontWeight: 600,
    color: '#e94560',
  },
  content: {
    padding: '12px 16px',
  },
  settingGroup: {
    marginBottom: '16px',
  },
  label: {
    fontSize: '12px',
    color: '#aaa',
    marginBottom: '6px',
    display: 'block',
  },
  sliderContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  slider: {
    flex: 1,
    height: '4px',
    appearance: 'none' as const,
    backgroundColor: '#0f3460',
    borderRadius: '2px',
    outline: 'none',
    cursor: 'pointer',
  },
  sliderValue: {
    fontSize: '12px',
    color: '#eee',
    minWidth: '30px',
    textAlign: 'right' as const,
  },
  colorInput: {
    width: '100%',
    height: '32px',
    padding: '2px',
    border: '1px solid #0f3460',
    borderRadius: '4px',
    backgroundColor: '#0a0f1a',
    cursor: 'pointer',
  },
  colorPresets: {
    display: 'flex',
    gap: '6px',
    marginTop: '8px',
    flexWrap: 'wrap' as const,
  },
  colorPreset: {
    width: '24px',
    height: '24px',
    borderRadius: '4px',
    border: '2px solid transparent',
    cursor: 'pointer',
  },
  colorPresetActive: {
    border: '2px solid #e94560',
  },
  select: {
    width: '100%',
    padding: '8px 12px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    outline: 'none',
  },
  colorScalePreview: {
    height: '12px',
    borderRadius: '2px',
    marginTop: '8px',
  },
  toggleContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  toggle: {
    position: 'relative' as const,
    width: '40px',
    height: '20px',
    backgroundColor: '#0f3460',
    borderRadius: '10px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  toggleActive: {
    backgroundColor: '#e94560',
  },
  toggleKnob: {
    position: 'absolute' as const,
    top: '2px',
    left: '2px',
    width: '16px',
    height: '16px',
    backgroundColor: '#eee',
    borderRadius: '50%',
    transition: 'transform 0.2s',
  },
  toggleKnobActive: {
    transform: 'translateX(20px)',
  },
  toggleLabel: {
    fontSize: '12px',
    color: '#aaa',
  },
  toggleDescription: {
    fontSize: '10px',
    color: '#666',
    marginTop: '4px',
  },
}

const BACKGROUND_PRESETS = [
  { color: '#1a1a2e', name: 'Dark Blue' },
  { color: '#0a0a0a', name: 'Black' },
  { color: '#1a1a1a', name: 'Dark Gray' },
  { color: '#2d2d44', name: 'Slate' },
  { color: '#0d1b2a', name: 'Navy' },
  { color: '#ffffff', name: 'White' },
  { color: '#f5f5f5', name: 'Light Gray' },
  { color: '#fafafa', name: 'Off White' },
]

const COLOR_SCALES: { value: ColorScale; label: string; gradient: string }[] = [
  {
    value: 'viridis',
    label: 'Viridis',
    gradient: 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)',
  },
  {
    value: 'plasma',
    label: 'Plasma',
    gradient: 'linear-gradient(to right, #0d0887, #7e03a8, #cc4778, #f89540, #f0f921)',
  },
  {
    value: 'magma',
    label: 'Magma',
    gradient: 'linear-gradient(to right, #000004, #51127c, #b73779, #fc8961, #fcfdbf)',
  },
  {
    value: 'inferno',
    label: 'Inferno',
    gradient: 'linear-gradient(to right, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)',
  },
  {
    value: 'cividis',
    label: 'Cividis',
    gradient: 'linear-gradient(to right, #002051, #525f6e, #98883e, #fdea45)',
  },
  {
    value: 'coolwarm',
    label: 'Coolwarm',
    gradient: 'linear-gradient(to right, #3b4cc0, #7092d0, #c5c5c5, #e68067, #b40426)',
  },
  {
    value: 'blues',
    label: 'Blues',
    gradient: 'linear-gradient(to right, #f7fbff, #6baed6, #08306b)',
  },
  {
    value: 'reds',
    label: 'Reds',
    gradient: 'linear-gradient(to right, #fff5f0, #fb6a4a, #67000d)',
  },
]

const BIVARIATE_COLORMAP_OPTIONS: { value: BivariateColormap; label: string }[] = [
  { value: 'default', label: 'Red / Blue / Yellow' },
  { value: 'pinkgreen', label: 'Pink / Green / Brown' },
  { value: 'orangepurple', label: 'Orange / Purple / Maroon' },
]

// Canvas-based bivariate colormap preview
function BivariateColormapPreview({ colormap, size = 40 }: { colormap: BivariateColormap; size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const imageData = ctx.createImageData(size, size)
    const data = imageData.data

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1)  // gene set 1 (horizontal)
        const v = 1 - y / (size - 1)  // gene set 2 (vertical, flipped so high is at top)
        const color = getBivariateColor(u, v, colormap)
        const idx = (y * size + x) * 4
        data[idx] = color[0]
        data[idx + 1] = color[1]
        data[idx + 2] = color[2]
        data[idx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [colormap, size])

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ borderRadius: '4px', border: '1px solid #0f3460' }}
    />
  )
}

export default function DisplaySettings() {
  const { displayPreferences, setDisplayPreferences } = useStore()
  const [isOpen, setIsOpen] = useState(false)

  const handlePointSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDisplayPreferences({ pointSize: parseFloat(e.target.value) })
  }

  const handleOpacityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDisplayPreferences({ pointOpacity: parseFloat(e.target.value) })
  }

  const handleBackgroundChange = (color: string) => {
    setDisplayPreferences({ backgroundColor: color })
  }

  const handleColorScaleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDisplayPreferences({ colorScale: e.target.value as ColorScale })
  }

  const handleBivariateColormapChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDisplayPreferences({ bivariateColormap: e.target.value as BivariateColormap })
  }

  const handleTransformToggle = () => {
    const newTransform: ExpressionTransform =
      displayPreferences.expressionTransform === 'none' ? 'log1p' : 'none'
    setDisplayPreferences({ expressionTransform: newTransform })
  }

  const handleScoringMethodToggle = () => {
    const newMethod: GeneSetScoringMethod =
      displayPreferences.geneSetScoringMethod === 'mean' ? 'zscore' : 'mean'
    setDisplayPreferences({ geneSetScoringMethod: newMethod })
  }

  const currentScale = COLOR_SCALES.find((s) => s.value === displayPreferences.colorScale)
  const isTransformEnabled = displayPreferences.expressionTransform === 'log1p'
  const isZscoreEnabled = displayPreferences.geneSetScoringMethod === 'zscore'

  return (
    <div style={styles.container}>
      <button
        style={{
          ...styles.button,
          ...(isOpen ? styles.buttonActive : {}),
        }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>Display</span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop to close on click outside */}
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 999,
            }}
            onClick={() => setIsOpen(false)}
          />
          <div style={styles.panel}>
            <div style={styles.header}>Display Settings</div>
            <div style={styles.content}>
              {/* Point Size */}
              <div style={styles.settingGroup}>
                <label style={styles.label}>Point Size</label>
                <div style={styles.sliderContainer}>
                  <input
                    type="range"
                    min="0.5"
                    max="10"
                    step="0.1"
                    value={displayPreferences.pointSize}
                    onChange={handlePointSizeChange}
                    style={styles.slider}
                  />
                  <span style={styles.sliderValue}>{displayPreferences.pointSize.toFixed(1)}</span>
                </div>
              </div>

              {/* Point Opacity */}
              <div style={styles.settingGroup}>
                <label style={styles.label}>Point Opacity</label>
                <div style={styles.sliderContainer}>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={displayPreferences.pointOpacity}
                    onChange={handleOpacityChange}
                    style={styles.slider}
                  />
                  <span style={styles.sliderValue}>{Math.round(displayPreferences.pointOpacity * 100)}%</span>
                </div>
              </div>

              {/* Background Color */}
              <div style={styles.settingGroup}>
                <label style={styles.label}>Background Color</label>
                <input
                  type="color"
                  value={displayPreferences.backgroundColor}
                  onChange={(e) => handleBackgroundChange(e.target.value)}
                  style={styles.colorInput}
                />
                <div style={styles.colorPresets}>
                  {BACKGROUND_PRESETS.map((preset) => (
                    <div
                      key={preset.color}
                      style={{
                        ...styles.colorPreset,
                        backgroundColor: preset.color,
                        ...(displayPreferences.backgroundColor === preset.color
                          ? styles.colorPresetActive
                          : {}),
                      }}
                      onClick={() => handleBackgroundChange(preset.color)}
                      title={preset.name}
                    />
                  ))}
                </div>
              </div>

              {/* Color Scale */}
              <div style={styles.settingGroup}>
                <label style={styles.label}>Expression Color Scale</label>
                <select
                  value={displayPreferences.colorScale}
                  onChange={handleColorScaleChange}
                  style={styles.select}
                >
                  {COLOR_SCALES.map((scale) => (
                    <option key={scale.value} value={scale.value}>
                      {scale.label}
                    </option>
                  ))}
                </select>
                {currentScale && (
                  <div
                    style={{
                      ...styles.colorScalePreview,
                      background: currentScale.gradient,
                    }}
                  />
                )}
              </div>

              {/* Bivariate Colormap */}
              <div style={styles.settingGroup}>
                <label style={styles.label}>Bivariate Colormap</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <select
                    value={displayPreferences.bivariateColormap}
                    onChange={handleBivariateColormapChange}
                    style={{ ...styles.select, flex: 1 }}
                  >
                    {BIVARIATE_COLORMAP_OPTIONS.map((cmap) => (
                      <option key={cmap.value} value={cmap.value}>
                        {cmap.label}
                      </option>
                    ))}
                  </select>
                  <BivariateColormapPreview colormap={displayPreferences.bivariateColormap} size={40} />
                </div>
                <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
                  For comparing two gene sets simultaneously
                </div>
              </div>

              {/* Gene Set Scoring Method */}
              <div style={styles.settingGroup}>
                <div style={styles.toggleContainer}>
                  <div>
                    <span style={styles.toggleLabel}>Z-score Scaling</span>
                    <div style={styles.toggleDescription}>
                      {isZscoreEnabled
                        ? 'Mean-center + MAD scale each gene'
                        : 'Simple mean across genes'}
                    </div>
                  </div>
                  <div
                    style={{
                      ...styles.toggle,
                      ...(isZscoreEnabled ? styles.toggleActive : {}),
                    }}
                    onClick={handleScoringMethodToggle}
                  >
                    <div
                      style={{
                        ...styles.toggleKnob,
                        ...(isZscoreEnabled ? styles.toggleKnobActive : {}),
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Expression Transform */}
              <div style={styles.settingGroup}>
                <div style={styles.toggleContainer}>
                  <div>
                    <span style={styles.toggleLabel}>Normalize + Log1p</span>
                    <div style={styles.toggleDescription}>
                      Apply count depth scaling and log(x+1)
                    </div>
                  </div>
                  <div
                    style={{
                      ...styles.toggle,
                      ...(isTransformEnabled ? styles.toggleActive : {}),
                    }}
                    onClick={handleTransformToggle}
                  >
                    <div
                      style={{
                        ...styles.toggleKnob,
                        ...(isTransformEnabled ? styles.toggleKnobActive : {}),
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
