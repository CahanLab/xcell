import type { GeneSet } from '../store'

/**
 * Turn an arbitrary folder name into something safe to use as a filename stem.
 * Trims, replaces any run of characters outside [A-Za-z0-9._-] with a single
 * underscore, and falls back to "gene_sets" if the result is empty.
 */
export function sanitizeFilename(name: string): string {
  const cleaned = name.trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '')
  return cleaned || 'gene_sets'
}

function triggerDownload(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Export a folder's gene sets as JSON. Matches the existing global-export
 * schema so round-trips via the Import modal continue to work.
 * Empty sets are included.
 */
export function exportFolderAsJson(folderName: string, sets: GeneSet[]): void {
  const payload = sets.map((gs) => ({ name: gs.name, genes: gs.genes }))
  const json = JSON.stringify(payload, null, 2)
  triggerDownload(`${sanitizeFilename(folderName)}.json`, json, 'application/json')
}

/**
 * Export a folder's gene sets as GMT. Standard Broad format:
 *   set_name<TAB>description<TAB>gene1<TAB>gene2...
 * Description is the literal "na". Empty sets are SKIPPED because GMT readers
 * commonly reject lines with fewer than 3 fields.
 */
export function exportFolderAsGmt(folderName: string, sets: GeneSet[]): void {
  const lines = sets
    .filter((gs) => gs.genes.length > 0)
    .map((gs) => [gs.name, 'na', ...gs.genes].join('\t'))
  const gmt = lines.join('\n') + (lines.length ? '\n' : '')
  triggerDownload(`${sanitizeFilename(folderName)}.gmt`, gmt, 'text/tab-separated-values')
}

/**
 * Export a folder's gene sets as long-form CSV (two columns: set_name,gene).
 * RFC 4180 quoting is applied defensively to any field containing commas,
 * quotes, or newlines. Empty sets are skipped (no rows to emit for them).
 */
export function exportFolderAsCsv(folderName: string, sets: GeneSet[]): void {
  const quote = (value: string): string => {
    if (/[",\n\r]/.test(value)) {
      return `"${value.replace(/"/g, '""')}"`
    }
    return value
  }
  const rows: string[] = ['set_name,gene']
  for (const gs of sets) {
    for (const gene of gs.genes) {
      rows.push(`${quote(gs.name)},${quote(gene)}`)
    }
  }
  const csv = rows.join('\n') + '\n'
  triggerDownload(`${sanitizeFilename(folderName)}.csv`, csv, 'text/csv')
}
