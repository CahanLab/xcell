/** Scale-bar sizing: a bar must be a round physical length, not a round
 * number of pixels. */

export interface ScaleBarSpec {
  um: number
  px: number
  label: string
}

const NICE = [1, 2, 5]

/** The largest 1/2/5 × 10^k µm length whose bar fits in targetPx pixels.
 *
 * @param umPerPx  physical size of one screen pixel, in µm
 */
export function pickScaleBar(umPerPx: number, targetPx = 120): ScaleBarSpec | null {
  if (!isFinite(umPerPx) || umPerPx <= 0) return null
  const rawUm = targetPx * umPerPx
  const exponent = Math.floor(Math.log10(rawUm))
  let best: number | null = null
  for (const exp of [exponent, exponent - 1]) {
    for (const n of NICE) {
      const candidate = n * Math.pow(10, exp)
      if (candidate <= rawUm && (best == null || candidate > best)) best = candidate
    }
  }
  if (best == null || best <= 0) return null
  // Snap away float dust (0.19999999…) without disturbing round values.
  const um = parseFloat(best.toPrecision(12))
  const px = um / umPerPx
  const label = um >= 1000
    ? `${parseFloat((um / 1000).toPrecision(12))} mm`
    : `${um} µm`
  return { um, px: parseFloat(px.toPrecision(12)), label }
}
