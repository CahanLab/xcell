/** Turn an on-screen <svg> into a file that opens outside the browser.
 *
 * Browsers infer the SVG namespace from the DOM; a serialised file has no such
 * context and needs the xmlns declared, or Illustrator and Inkscape refuse it.
 * xcell's figures are drawn for a dark panel, so a transparent background
 * would come out as white-on-white in most viewers — hence the painted rect.
 */

export function standaloneSvg(markup: string, opts: { background?: string }): string {
  let out = markup
  if (!out.includes('xmlns=')) {
    out = out.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
  }
  if (opts.background) {
    const openTagEnd = out.indexOf('>') + 1
    out =
      out.slice(0, openTagEnd) +
      `<rect width="100%" height="100%" fill="${opts.background}"/>` +
      out.slice(openTagEnd)
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n${out}`
}

/** Hand a generated file to the user. Same object-URL dance as the gene-set
 *  export in App.tsx; kept here so callers do not each re-derive it. */
export function downloadText(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
