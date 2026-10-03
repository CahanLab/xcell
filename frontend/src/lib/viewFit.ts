/**
 * The camera that fits an embedding, and how far it may zoom from there.
 *
 * deck.gl's zoom is log2(screen pixels per data unit), so a fixed [-10, 10]
 * clamp means something different on every embedding. It suited UMAP's tens
 * of units (fit ≈ 4), but a diffusion map spans a fraction of a unit and fits
 * at ≈ 11: above the ceiling, so the first wheel event snapped the view out to
 * 10 and no zoom-in could bring it back. The limits are relative to the fit.
 */

export interface Bounds2D {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

type Coord = readonly (number | null)[] | null | undefined

/** The extent of the cells that have coordinates, padded by ``pad`` of it on
 * each side. A subset's embedding is null outside the subset, and ``null < x``
 * reads null as 0, so those cells used to pull the fit towards the origin. */
export function boundsOf(coords: readonly Coord[], pad = 0.05): Bounds2D {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const c of coords) {
    if (!c) continue
    const x = c[0], y = c[1]
    if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (minX > maxX) return { minX: 0, maxX: 1, minY: 0, maxY: 1 }
  const padX = (maxX - minX) * pad
  const padY = (maxY - minY) * pad
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }
}

/** How far, in powers of two, the camera may zoom either way from the fit. */
export const ZOOM_RANGE = 10

export const zoomLimits = (fit: number) => ({ minZoom: fit - ZOOM_RANGE, maxZoom: fit + ZOOM_RANGE })

/** Fit ``b`` into a ``width`` × ``height`` view with a margin. An axis with no
 * extent (a constant diffusion component) is left to the other; no extent at
 * all gives zoom 0 rather than Infinity. */
export function fitView(b: Bounds2D, width: number, height: number) {
  const scale = (pixels: number, span: number) =>
    span > 0 && Number.isFinite(span) ? pixels / span : Infinity
  const s = Math.min(scale(width, b.maxX - b.minX), scale(height, b.maxY - b.minY))
  const zoom = Number.isFinite(s) ? Math.log2(s) - 1 : 0
  const mid = (lo: number, hi: number) => (Number.isFinite(lo + hi) ? (lo + hi) / 2 : 0)
  return {
    target: [mid(b.minX, b.maxX), mid(b.minY, b.maxY), 0] as [number, number, number],
    zoom,
    ...zoomLimits(zoom),
  }
}
