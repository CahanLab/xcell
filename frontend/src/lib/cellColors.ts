import { useMemo } from 'react'
import {
  DisplayPreferences,
  ColorMode,
  ObsColumnData,
  ExpressionData,
  BivariateExpressionData,
  HighlightLayer,
  ColorScale,
  BivariateColormap,
} from '../store'

// Base palette for categorical data (d3 category10). Beyond 10 categories we
// extend with golden-angle hue stepping in HSL — see generateCategoryPalette.
const BASE_CATEGORY_COLORS: [number, number, number][] = [
  [31, 119, 180],
  [255, 127, 14],
  [44, 160, 44],
  [214, 39, 40],
  [148, 103, 189],
  [140, 86, 75],
  [227, 119, 194],
  [127, 127, 127],
  [188, 189, 34],
  [23, 190, 207],
]

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const C = (1 - Math.abs(2 * l - 1)) * s
  const hh = h / 60
  const X = C * (1 - Math.abs((hh % 2) - 1))
  let r1 = 0, g1 = 0, b1 = 0
  if (hh < 1) { r1 = C; g1 = X }
  else if (hh < 2) { r1 = X; g1 = C }
  else if (hh < 3) { g1 = C; b1 = X }
  else if (hh < 4) { g1 = X; b1 = C }
  else if (hh < 5) { r1 = X; b1 = C }
  else { r1 = C; b1 = X }
  const m = l - C / 2
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ]
}

// Return N visually distinct RGB triplets. First 10 follow d3-cat10; the rest
// step through HSL hue space by the golden angle (137.5°) so adjacent indices
// land far apart on the color wheel, with alternating saturation/lightness to
// disambiguate when hues wrap close together.
export function generateCategoryPalette(n: number): [number, number, number][] {
  if (n <= BASE_CATEGORY_COLORS.length) return BASE_CATEGORY_COLORS.slice(0, n)
  const out: [number, number, number][] = BASE_CATEGORY_COLORS.slice()
  const GOLDEN_ANGLE = 137.508
  for (let i = BASE_CATEGORY_COLORS.length; i < n; i++) {
    const k = i - BASE_CATEGORY_COLORS.length
    const hue = (k * GOLDEN_ANGLE + 25) % 360
    const sat = k % 2 === 0 ? 0.70 : 0.85
    const lit = k % 3 === 0 ? 0.55 : k % 3 === 1 ? 0.45 : 0.60
    out.push(hslToRgb(hue, sat, lit))
  }
  return out
}

// Parse a #RGB / #RRGGBB / #RRGGBBAA hex color into an RGB triplet.
export function hexToRgb(hex: string): [number, number, number] | null {
  if (typeof hex !== 'string') return null
  let h = hex.trim().replace(/^#/, '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  if (h.length !== 6 && h.length !== 8) return null
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null
  return [r, g, b]
}

// Resolve a per-category palette. If `existingColors` (e.g. from adata.uns
// [`${col}_colors`]) is provided and parseable, those win; missing/invalid
// entries fall through to the generated palette. Output length = n.
export function resolveCategoryPalette(
  n: number,
  existingColors?: (string | null | undefined)[],
): [number, number, number][] {
  const generated = generateCategoryPalette(n)
  if (!existingColors || existingColors.length === 0) return generated
  return generated.map((fallback, i) => {
    const hex = existingColors[i]
    if (!hex) return fallback
    return hexToRgb(hex) ?? fallback
  })
}

export const DEFAULT_COLOR: [number, number, number, number] = [100, 149, 237, 200]
export const SELECTED_COLOR: [number, number, number, number] = [255, 255, 0, 255]

// Linear interpolation helper
function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t)
}

// Interpolate between color stops
function interpolateStops(
  t: number,
  stops: { pos: number; color: [number, number, number] }[]
): [number, number, number] {
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].pos && t <= stops[i + 1].pos) {
      const localT = (t - stops[i].pos) / (stops[i + 1].pos - stops[i].pos)
      return [
        lerp(stops[i].color[0], stops[i + 1].color[0], localT),
        lerp(stops[i].color[1], stops[i + 1].color[1], localT),
        lerp(stops[i].color[2], stops[i + 1].color[2], localT),
      ]
    }
  }
  return stops[stops.length - 1].color
}

// Color scale definitions
const COLOR_SCALES: Record<ColorScale, { pos: number; color: [number, number, number] }[]> = {
  viridis: [
    { pos: 0, color: [68, 1, 84] },
    { pos: 0.25, color: [59, 82, 139] },
    { pos: 0.5, color: [33, 145, 140] },
    { pos: 0.75, color: [94, 201, 98] },
    { pos: 1, color: [253, 231, 37] },
  ],
  plasma: [
    { pos: 0, color: [13, 8, 135] },
    { pos: 0.25, color: [126, 3, 168] },
    { pos: 0.5, color: [204, 71, 120] },
    { pos: 0.75, color: [248, 149, 64] },
    { pos: 1, color: [240, 249, 33] },
  ],
  magma: [
    { pos: 0, color: [0, 0, 4] },
    { pos: 0.25, color: [81, 18, 124] },
    { pos: 0.5, color: [183, 55, 121] },
    { pos: 0.75, color: [252, 137, 97] },
    { pos: 1, color: [252, 253, 191] },
  ],
  inferno: [
    { pos: 0, color: [0, 0, 4] },
    { pos: 0.2, color: [66, 10, 104] },
    { pos: 0.4, color: [147, 38, 103] },
    { pos: 0.6, color: [221, 81, 58] },
    { pos: 0.8, color: [252, 165, 10] },
    { pos: 1, color: [252, 255, 164] },
  ],
  cividis: [
    { pos: 0, color: [0, 32, 81] },
    { pos: 0.33, color: [82, 95, 110] },
    { pos: 0.66, color: [152, 136, 62] },
    { pos: 1, color: [253, 234, 69] },
  ],
  coolwarm: [
    { pos: 0, color: [59, 76, 192] },
    { pos: 0.25, color: [112, 146, 208] },
    { pos: 0.5, color: [197, 197, 197] },
    { pos: 0.75, color: [230, 128, 103] },
    { pos: 1, color: [180, 4, 38] },
  ],
  blues: [
    { pos: 0, color: [247, 251, 255] },
    { pos: 0.5, color: [107, 174, 214] },
    { pos: 1, color: [8, 48, 107] },
  ],
  reds: [
    { pos: 0, color: [255, 245, 240] },
    { pos: 0.5, color: [251, 106, 74] },
    { pos: 1, color: [103, 0, 13] },
  ],
  // Modern sequential gradients (learnui.design-style smooth multi-stops).
  sunset: [
    { pos: 0, color: [40, 11, 86] },
    { pos: 0.5, color: [219, 75, 109] },
    { pos: 1, color: [255, 222, 135] },
  ],
  ocean: [
    { pos: 0, color: [2, 17, 51] },
    { pos: 0.5, color: [28, 119, 150] },
    { pos: 1, color: [120, 255, 214] },
  ],
  grape: [
    { pos: 0, color: [28, 27, 92] },
    { pos: 0.5, color: [123, 31, 162] },
    { pos: 1, color: [224, 64, 251] },
  ],
  mint: [
    { pos: 0, color: [4, 40, 63] },
    { pos: 0.5, color: [0, 150, 136] },
    { pos: 1, color: [173, 255, 96] },
  ],
}

export function getColorFromScale(t: number, scale: ColorScale): [number, number, number] {
  return interpolateStops(t, COLOR_SCALES[scale])
}

// Bivariate colormap definitions
// Each has corner colors: c00 (low/low), c10 (high gene1/low gene2), c01 (low gene1/high gene2), c11 (high/high)
type BivariateCorners = {
  c00: [number, number, number]  // Low/Low
  c10: [number, number, number]  // High gene1/Low gene2
  c01: [number, number, number]  // Low gene1/High gene2
  c11: [number, number, number]  // High/High
}

export const BIVARIATE_COLORMAPS: Record<BivariateColormap, BivariateCorners> = {
  default: {
    c00: [240, 240, 240],  // Gray
    c10: [227, 26, 28],    // Red
    c01: [31, 120, 180],   // Blue
    c11: [255, 255, 0],    // Yellow
  },
  pinkgreen: {
    c00: [240, 240, 240],  // Gray
    c10: [197, 27, 125],   // Pink/Magenta
    c01: [77, 146, 33],    // Green
    c11: [166, 86, 40],    // Brown (mixing pink+green)
  },
  orangepurple: {
    c00: [240, 240, 240],  // Gray
    c10: [230, 97, 1],     // Orange
    c01: [94, 60, 153],    // Purple
    c11: [178, 24, 43],    // Dark red/maroon
  },
  custom: {
    c00: [240, 240, 240],  // Gray (default, can be customized later)
    c10: [227, 26, 28],    // Red
    c01: [31, 120, 180],   // Blue
    c11: [255, 255, 0],    // Yellow
  },
}

// Bilinear interpolation for bivariate coloring
export function getBivariateColor(
  u: number,  // Normalized gene set 1 value [0,1]
  v: number,  // Normalized gene set 2 value [0,1]
  colormap: BivariateColormap = 'default',
): [number, number, number] {
  const { c00, c10, c01, c11 } = BIVARIATE_COLORMAPS[colormap]
  // Bilinear interpolation: blend four corner colors based on (u, v) position
  return [
    Math.round(c00[0] * (1 - u) * (1 - v) + c10[0] * u * (1 - v) + c01[0] * (1 - u) * v + c11[0] * u * v),
    Math.round(c00[1] * (1 - u) * (1 - v) + c10[1] * u * (1 - v) + c01[1] * (1 - u) * v + c11[1] * u * v),
    Math.round(c00[2] * (1 - u) * (1 - v) + c10[2] * u * (1 - v) + c01[2] * (1 - u) * v + c11[2] * u * v),
  ]
}

// Build a per-cell weight function for a highlight layer. Hard gate: in-range
// cells get full intensity; out-of-range get 0. Cellset layers consult their
// mask directly.
export function layerWeightFn(layer: HighlightLayer): (i: number) => number {
  const src = layer.source
  const intensity = layer.intensity
  if (src.kind === 'cellset') {
    const mask = src.mask
    return (i: number) => (mask[i] ? intensity : 0)
  }
  const { values, thresholdMode, lo, hi } = src
  if (thresholdMode === 'above') {
    return (i: number) => {
      const v = values[i]
      return v != null && v >= lo ? intensity : 0
    }
  }
  if (thresholdMode === 'below') {
    return (i: number) => {
      const v = values[i]
      return v != null && v <= lo ? intensity : 0
    }
  }
  return (i: number) => {
    const v = values[i]
    return v != null && v >= lo && v <= hi ? intensity : 0
  }
}

export interface CellColorParams {
  displayPreferences: DisplayPreferences
  activeCellMask: (boolean[] | Uint8Array | null)
  colorMode: ColorMode
  colorBy: ObsColumnData | null
  expressionData: ExpressionData | null
  bivariateData: BivariateExpressionData | null
  highlightLayers: HighlightLayer[]
  selectedSet: Set<number>
}

// Compute the per-cell color function (depends on colorMode, active data, and
// any highlight overlays). Extracted from ScatterPlot.tsx so ScatterPlot3D
// can reuse the exact same coloring logic.
export function useCellColor(p: CellColorParams): (d: { index: number }) => [number, number, number, number] {
  return useMemo(() => {
    const opacity = Math.round(p.displayPreferences.pointOpacity * 255)
    const maskedOpacity = Math.round(opacity * 0.3) // 30% opacity for masked cells
    const maskedColor: [number, number, number, number] = [100, 100, 100, maskedOpacity]

    // Helper to check if cell is masked (inactive)
    const isMasked = (index: number): boolean => {
      return p.activeCellMask !== null && !p.activeCellMask[index]
    }

    // Compute the base color function (depends on colorMode); we may wrap it
    // below with a highlight overlay blender.
    let baseColorFn: (d: { index: number }) => [number, number, number, number]

    if (p.colorMode === 'bivariate' && p.bivariateData) {
      const { values1, values2 } = p.bivariateData
      const bivariateColormap = p.displayPreferences.bivariateColormap

      baseColorFn = (d: { index: number }): [number, number, number, number] => {
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const u = values1[d.index] ?? 0
        const v = values2[d.index] ?? 0
        const color = getBivariateColor(u, v, bivariateColormap)
        return [...color, opacity] as [number, number, number, number]
      }
    } else if (p.colorMode === 'expression' && p.expressionData) {
      const { values, min, max } = p.expressionData
      const range = max - min || 1
      const colorScale = p.displayPreferences.colorScale

      baseColorFn = (d: { index: number }): [number, number, number, number] => {
        // Show masked cells as gray (when showMaskedCells is true, they're in the data)
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const value = values[d.index]
        if (value === null || value === undefined) {
          return [128, 128, 128, Math.round(opacity * 0.5)]
        }
        const t = (value - min) / range
        const color = getColorFromScale(t, colorScale)
        return [...color, opacity] as [number, number, number, number]
      }
    } else if (p.colorMode === 'metadata' && p.colorBy && p.colorBy.dtype === 'category') {
      const colorBy = p.colorBy
      const nCats = colorBy.categories?.length ?? 0
      const palette = resolveCategoryPalette(Math.max(nCats, 1), colorBy.colors)
      baseColorFn = (d: { index: number }): [number, number, number, number] => {
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const value = colorBy.values[d.index] as number
        const color = palette[value] ?? palette[0]
        return [...color, opacity] as [number, number, number, number]
      }
    } else if (p.colorMode === 'metadata' && p.colorBy && p.colorBy.dtype === 'numeric') {
      const colorBy = p.colorBy
      const values = colorBy.values.filter((v) => v !== null) as number[]
      const min = Math.min(...values)
      const max = Math.max(...values)
      const range = max - min || 1
      const colorScale = p.displayPreferences.colorScale

      baseColorFn = (d: { index: number }): [number, number, number, number] => {
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        const value = colorBy.values[d.index]
        if (value === null) return [128, 128, 128, Math.round(opacity * 0.5)]
        const t = ((value as number) - min) / range
        // Continuous .obs (e.g. LR scores) honors the chosen color scale, like
        // expression coloring does.
        const color = getColorFromScale(t, colorScale)
        return [...color, opacity] as [number, number, number, number]
      }
    } else {
      // Default color function (no active color mode, or metadata without dtype)
      baseColorFn = (d: { index: number }): [number, number, number, number] => {
        if (isMasked(d.index)) {
          return maskedColor
        }
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) {
          return SELECTED_COLOR
        }
        return [DEFAULT_COLOR[0], DEFAULT_COLOR[1], DEFAULT_COLOR[2], opacity]
      }
    }

    // Highlight overlay: blend each layer's color over the running base color,
    // in creation order. Weight per cell per layer is a hard gate
    // (in-threshold = full intensity, otherwise 0) for geneset layers and a
    // mask lookup for cellset layers. Skipped for selected and masked cells.
    if (p.highlightLayers.length > 0) {
      const compiled = p.highlightLayers.map((layer) => ({
        rgb: hexToRgb(layer.color) ?? [34, 197, 94] as [number, number, number],
        weight: layerWeightFn(layer),
      }))
      const fn = baseColorFn
      return (d: { index: number }): [number, number, number, number] => {
        const base = fn(d)
        if (isMasked(d.index)) return base
        if (p.selectedSet.size > 0 && p.selectedSet.has(d.index)) return base
        let r = base[0], g = base[1], b = base[2]
        for (let k = 0; k < compiled.length; k++) {
          const w = compiled[k].weight(d.index)
          // `!(w > 0)` also rejects NaN (NaN > 0 is false). The old `w <= 0`
          // let a NaN weight through, and a NaN weight poisons the blend ->
          // NaN color components -> the cell renders black.
          if (!(w > 0)) continue
          const ww = w > 1 ? 1 : w
          const lr = compiled[k].rgb
          r = r * (1 - ww) + lr[0] * ww
          g = g * (1 - ww) + lr[1] * ww
          b = b * (1 - ww) + lr[2] * ww
        }
        return [Math.round(r), Math.round(g), Math.round(b), base[3]]
      }
    }

    return baseColorFn
  }, [p.colorBy, p.expressionData, p.bivariateData, p.highlightLayers,
      p.colorMode, p.selectedSet, p.displayPreferences, p.activeCellMask])
}
