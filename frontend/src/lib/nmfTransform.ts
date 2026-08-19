/**
 * Picking the right source transform for NMF from a layer's detected scale.
 *
 * Every other gene-expression path in xcell defaults to normalize_total +
 * log1p, which is right for raw counts and wrong for anything already on a log
 * scale — there it silently factorizes log(1 + log(1 + counts)). NMF is more
 * sensitive to that than coloring is, because the extra compression flattens
 * exactly the high-expression differences the factors are built from, so the
 * modal picks its default from the scale xcell already detects
 * (`/api/scanpy/layers` → `scale.verdict`, see backend/xcell/layer_scale.py).
 */
export type NmfTransform = 'log1p' | 'none'

/** Scales that are already logged, or that log1p would actively damage. */
const NO_TRANSFORM = new Set([
  'log_normalized',
  'log_transformed',
  'binary',
  'z_scored',
])

export function defaultTransformFor(verdict: string | undefined): NmfTransform {
  if (verdict && NO_TRANSFORM.has(verdict)) return 'none'
  // raw_counts, normalized_linear, empty, unknown, or no detection at all —
  // log1p, matching the rest of xcell.
  return 'log1p'
}

const NOTES: Record<string, string> = {
  log_normalized: 'This matrix is already log-normalized, so it is fed to NMF as-is.',
  log_transformed: 'This matrix is already on a log scale, so it is fed to NMF as-is.',
  binary: 'This matrix is binary, so it is fed to NMF as-is.',
  z_scored:
    'This matrix is z-scored and contains negative values — NMF will reject it. '
    + 'Pick a counts or log-normalized layer instead.',
}

/** One line explaining a non-obvious default, or null when none is needed. */
export function transformNoteFor(verdict: string | undefined): string | null {
  if (!verdict) return null
  return NOTES[verdict] ?? null
}
