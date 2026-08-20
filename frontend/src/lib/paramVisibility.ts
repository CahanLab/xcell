/**
 * Whether a Scanpy-panel parameter should be shown, given the current values.
 *
 * Most conditions are an exact match on another parameter. `'*'` means "shown
 * once that parameter has any value at all", which is what options depending
 * on an optional picker need — the HVG union/intersection columns only exist
 * once a split column has been chosen, and there is no fixed value to match.
 */
export interface VisibleWhen {
  param: string
  value: string
}

export function isParamVisible(
  condition: VisibleWhen | undefined,
  paramValues: Record<string, unknown>,
): boolean {
  if (!condition) return true
  const value = paramValues[condition.param]
  if (condition.value === '*') {
    return value !== undefined && value !== null && value !== ''
  }
  return String(value) === condition.value
}
