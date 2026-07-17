# Gene-set import: case-convention parameter

**Date:** 2026-07-17
**Status:** Approved, ready to implement

## Problem

External gene sets are species-specific. Most orthologous human and mouse genes
share the same symbol but differ only by case (human `COL1A1` vs mouse `Col1a1`).
Users want to load a mouse gene set and use it against human data, and vice-versa.

Today every gene-set scoring/coloring path resolves symbols against the dataset's
`var_names` with an **exact, case-sensitive** match:

- `score_gene_sets_matrix` — folder → `.obsm` (`adaptor.py:659`)
- `score_gene_sets_ucell` — directional → `.obs` (`adaptor.py:1479`)
- `get_multi_gene_expression` — color-by-set, mean (`adaptor.py:1588`)
- `ucell_score_values` — color-by-set, UCell (`adaptor.py:1417`)
- `get_expression` — click a single gene to color (`adaptor.py:1203`)

So a mouse set (`Col1a1`) scored against human data (`COL1A1`) silently drops
every gene, and clicking a single mouse gene name 404s.

## Decision

Ship the smallest, safest fix now: an **explicit case-convention parameter in the
gene-set import modal**. When the user selects a convention, every imported gene
symbol is transformed to match their data's convention *at import time*, before it
enters the store. Because the stored symbols then match the dataset, they flow
through all five exact-match paths above unchanged — including the single-gene
click. No backend changes, no scoring-code changes.

This is deliberately scoped down from the fuller alternatives discussed
(a runtime ortholog-aware resolver + a bundled HGNC HCOP ortholog table). Those
remain the eventual "real" solution and are recorded under Deferred, below.

## Design

### The control

Add one selector to `frontend/src/components/ImportModal.tsx`, above the Import
button:

> **Convert gene symbols to match your data:**
> - Leave as-is *(default)*
> - Human — UPPERCASE (e.g. `COL1A1`)
> - Mouse — Title case (e.g. `Col1a1`)

Default is `Leave as-is`, so behavior is unchanged unless the user opts in.
The selection is modal-local state, reset when the modal closes.

### The conversion helper

A pure function in a new module `frontend/src/lib/caseConvention.ts`:

```ts
export type CaseConvention = 'none' | 'human' | 'mouse'

export function applyCaseConvention(symbol: string, c: CaseConvention): string {
  if (c === 'human') return symbol.toUpperCase()          // Col1a1 -> COL1A1
  if (c === 'mouse') {                                     // COL1A1 -> Col1a1
    return symbol ? symbol[0].toUpperCase() + symbol.slice(1).toLowerCase() : symbol
  }
  return symbol
}
```

A small list-level wrapper applies it to a `ParsedGeneList` / `LibrarySet`,
transforming **both** `genes` (up) and `genesDown` (down).

### Where it's applied

At import time, in the two places sets enter the store:

- `handleImport` — uploaded files (`.gmt` / `.csv` / `.txt` / `.tsv` / `.json`).
- `handleLoadBundle` — the shipped curated libraries, so a human library can be
  loaded onto mouse data (and vice-versa) too.

The parsers are untouched; the transform is applied to the parsed lists just
before they are handed to `addGeneSetToCategory` / `addFolderToCategory`.

### What this buys

The stored symbols match the dataset's convention, so every existing path works
with no further change: folder → `.obsm` scoring, UCell, color-by-set (mean and
UCell), and the single-gene click (`get_expression`). A converted `COL1A1`
clicked in the folder simply exact-matches.

## Known limitations (documented, intentional)

- **Case-only heuristic.** Orthologs whose symbols genuinely differ beyond case
  (non-1:1 families, renamed genes) are not handled. That is what the deferred
  HGNC HCOP ortholog table would fix.
- **Atypical casing.** Symbols like mitochondrial `mt-Nd1` get the naive rule
  (`MT-ND1` for human, `Mt-nd1` for mouse), which won't always equal the true
  symbol.
- **Baked at import.** Conversion is applied once, at import, and stored. This
  fits the one-species-per-session workflow. A runtime resolver would be needed
  to make a single gene-set library resolve correctly against datasets of
  different species within one session.

## Deferred (future work)

A runtime, ortholog-aware resolver: bundle a compact `orthologs_human_mouse.csv`
distilled from **HGNC HCOP** (integrates ~14 orthology resources with a `support`
consensus count; build script modeled on `build_lr_db.py`), and resolve each
symbol against the active dataset **exact → ortholog → case-insensitive → miss**,
routed through all five call sites (and eventually LigRec's ad-hoc uppercase
matching at `adaptor.py:7234`). This enables same-session cross-species
comparison and keeps the global gene-set library reusable across species.
Tracked separately; not part of this change.

## Testing

The frontend has no unit-test harness (backend uses pytest; frontend behavior is
verified end-to-end via Playwright, matching this project's established pattern).
Verification for this change:

1. `tsc` typecheck passes (`npm run build` runs `tsc && vite build`).
2. End-to-end in the running app: open Import, load a lowercase mouse gene list
   with **Human** selected, confirm the stored set's symbols are uppercased
   (both up and down lists), and that the shipped-library path applies the
   convention too.

If a frontend test runner is introduced later, `applyCaseConvention` is a pure
function and is the natural first unit under test.
