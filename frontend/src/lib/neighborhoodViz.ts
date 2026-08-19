/** Pure color/format helpers for the Neighborhood enrichment heatmap.
 *
 * The diverging z ramp is built for xcell's dark surface: the midpoint sits
 * near the panel background so z ≈ 0 reads as "nothing", and salience grows
 * with |z| toward a light cool pole (avoidance) and a light warm pole
 * (attraction) — histoCAT's red/blue convention with dark-mode-correct
 * lightness anchoring.
 */

export const Z_COOL: [number, number, number] = [130, 170, 242] // avoidance pole
export const Z_MID: [number, number, number] = [37, 43, 61]     // ≈ panel surface
export const Z_WARM: [number, number, number] = [242, 136, 107] // attraction pole

const SEQ_LO: [number, number, number] = [18, 34, 48]   // near-surface teal
const SEQ_HI: [number, number, number] = [124, 232, 218] // bright accent teal

const clamp01 = (t: number) => Math.min(1, Math.max(0, t))

function mix(a: [number, number, number], b: [number, number, number], t: number): string {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

/** Symmetric color domain for a z matrix: the 90th percentile of |z|,
 * floored at 3 so weak results never look saturated, and robust to a few
 * huge z values flattening the rest of the scale. */
export function zDomain(zscores: number[][]): number {
  const abs = zscores.flat().map(Math.abs).filter((v) => isFinite(v))
  if (!abs.length) return 3
  abs.sort((x, y) => x - y)
  const p90 = abs[Math.floor(0.9 * (abs.length - 1))]
  return Math.max(3, p90)
}

/** Diverging cell color for a z-score given the symmetric domain. */
export function divergingColor(z: number, domain: number): string {
  const t = clamp01(Math.abs(z) / domain)
  return z >= 0 ? mix(Z_MID, Z_WARM, t) : mix(Z_MID, Z_COOL, t)
}

/** Sequential (dark→light teal) color for a 0..1 fraction. */
export function seqColor(t: number): string {
  return mix(SEQ_LO, SEQ_HI, clamp01(t))
}

/** Ink that stays readable on a given cell color. */
export function textColorFor(rgb: [number, number, number]): string {
  const [r, g, b] = rgb
  const luma = 0.299 * r + 0.587 * g + 0.114 * b
  return luma > 140 ? '#101418' : '#eee'
}

/** Signed z with one decimal; never prints "-0.0". */
export function formatZ(z: number): string {
  const s = z.toFixed(1)
  if (s === '-0.0' || s === '0.0') return '0.0'
  return z > 0 ? `+${s}` : s
}

/** Percent with a decimal only below 10%. */
export function formatPct(frac: number): string {
  const pct = frac * 100
  if (pct > 0 && pct < 10) return `${pct.toFixed(1)}%`
  return `${Math.round(pct)}%`
}
