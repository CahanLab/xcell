import { useState } from 'react'
import { useStore, ColorScale } from '../store'

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

  const currentScale = COLOR_SCALES.find((s) => s.value === displayPreferences.colorScale)

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
            </div>
          </div>
        </>
      )}
    </div>
  )
}
