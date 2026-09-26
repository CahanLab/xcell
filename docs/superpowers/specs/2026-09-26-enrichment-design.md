# Gene-set enrichment: overlap (ORA) and preranked GSEA

**Date:** 2026-09-26
**Status:** approved by delegation ("proceed w/o my feedback unless needed")

## Goal

Answer "which annotated gene sets does this gene list / this contrast point
at?" inside xcell, against the gene-set libraries the app already caches
(MSigDB, GO, Enrichr, OmniPath, MGI) and against the user's own Gene Panel
sets. Two complementary tests:

- **Overlap enrichment (ORA).** A gene list (a Gene Panel set, a pasted
  list) against a library: hypergeometric test of overlap with each set,
  Benjamini–Hochberg across sets. Universe = the dataset's genes, after the
  session gene mask and an optional gene-subset column.
- **Preranked GSEA.** A signed per-gene ranking (group vs rest / group vs
  group differential expression, or a PCA loading) against a library:
  Subramanian-2005 weighted running sum, gene-permutation null, NES, empirical
  p, BH. Leading-edge genes and the running-score curve are returned for
  plotting.

Both write their results into the AnnData so they export with the file, log
to the analysis record, and can be turned into Gene Panel sets.

## Non-goals

- No `gseapy` / `fgsea` dependency. scipy + numpy implement everything; the
  multilevel exact p-value refinement of fgsea is out of scope — the empirical
  p is floored at `1/(n_perm+1)` and the UI says so.
- No new library downloads from this feature. It runs only against libraries
  already in `~/.cache/xcell/gene_set_sources/` (the Library modal fetches
  them). An uncached library is a 400 with a hint.
- No phenotype (sample-label) permutation GSEA; single-cell contrasts do not
  have exchangeable samples in the sense that test assumes.
- No per-cluster batch mode ("GSEA for every Leiden cluster at once"). One
  contrast per run; the modal can be re-run.

## Backend

### `backend/xcell/enrichment.py` (pure)

Takes arrays and plain dicts; never imports the adaptor. Public functions:

```python
def bh_adjust(pvals: np.ndarray) -> np.ndarray
```
Benjamini–Hochberg, monotone, clipped to 1. NaN inputs stay NaN.

```python
def resolve_sets(sets, universe, *, min_size, max_size, directional='union')
    -> list[ResolvedSet]
```
`sets` are `{name, genes, genes_down?, library?, description?, url?}`.
Each set's symbols are matched to `universe` (list of dataset gene names)
exactly first, then case-insensitively (the `gene_set_overlap` rule), and
de-duplicated. `directional='union'` merges `genes ∪ genes_down` (ORA);
`directional='split'` emits `<name>` for `genes` and `<name> (down)` for a
non-empty `genes_down` (GSEA, where sign matters). Sets whose resolved size
falls outside `[min_size, max_size]` are dropped; the count is reported.
A `ResolvedSet` is `{name, library, description, url, n_input, indices:
np.ndarray[int]}` where `indices` index into `universe`.

```python
def overlap_enrichment(query_indices, resolved_sets, n_universe, *,
                       min_overlap=2) -> list[dict]
```
For each set: `k` = |query ∩ set|, `K` = |set|, `n` = |query|, `N` =
`n_universe`. `pval = hypergeom.sf(k-1, N, K, n)`; `expected = n*K/N`;
`fold_enrichment = k/expected`; `odds_ratio` from the 2×2 table (Haldane 0.5
correction when a cell is zero, so it is never `inf`). Sets with
`k < min_overlap` are still tested (they count toward BH — dropping them
would bias the correction) but are flagged `below_min_overlap: True` so the
UI can hide them. Returned records, sorted by `pval` then name:
`{name, library, description, url, n_set, n_overlap, expected,
fold_enrichment, odds_ratio, pval, padj, genes: [overlap gene names],
below_min_overlap}`. All floats finite.

```python
def preranked_gsea(scores, resolved_sets, *, n_perm=1000, min_size=15,
                   max_size=500, weight=1.0, seed=0, curve_top_n=50,
                   report=None) -> dict
```
`scores` is a float array aligned to the universe. Genes with non-finite
scores are dropped first. Ranking is descending by score, stable (ties keep
universe order). Each set's `indices` are mapped to rank positions.

ES for a set with sorted hit positions `p_1 < … < p_k` (0-based) and weights
`w_j = |s_{p_j}|^weight`: post-hit value `P_hit(j) − (p_j − (j−1))/(N−k)`,
pre-hit value `P_hit(j−1) − (p_j − (j−1))/(N−k)`. ES is the post-hit maximum
or the pre-hit minimum, whichever has larger magnitude (the running sum is
monotone between hits, so those are the only candidate extrema). This is
vectorised across permutations with one `cumsum` per size class. If all
weights are zero (a set whose genes all score 0), ES is 0.

Null: `n_perm` random permutations of `arange(N)` drawn once; for size `k`
the null set is the first `k` entries of each permutation (a uniform random
`k`-subset, the fgsea trick), so every distinct set size shares the same
permutation pool and each size class is computed once and cached.

`NES = ES / mean(|null ES| of the same sign)`; `pval = (1 + #{null of same
sign with |null| ≥ |ES|}) / (1 + #{null of same sign})`; `padj = bh_adjust`
across all tested sets. Leading edge: hits at positions `≤ argmax` for
`ES > 0`, `≥ argmin` for `ES < 0`. For the `curve_top_n` sets with smallest
`pval` the record carries `curve: [[x, y], …]` — vertices `(0,0)`, then per
hit `(p_j, pre)`, `(p_j, post)`, then `(N−1, 0)` — for the others `curve` is
`None`. `report(fraction, message)` is called per size class when supplied.

Returns `{n_ranked, n_perm, results: [{name, library, description, url,
n_set, es, nes, pval, padj, leading_edge: [gene names], n_leading_edge,
curve}]}` sorted by `pval` then `-|nes|`. Score of the ranking is returned
alongside as `ranking: {genes: [...], scores: [...]}` only from the adaptor
(see below), not from this function.

### Adaptor (`adaptor.py`)

- `_enrichment_sets(libraries, sets)` — resolves each `{source, id,
  species?}` through `gene_set_sources.find_library`; missing → `ValueError("Library
  'x' from msigdb is not cached; fetch it from the Gene set library first")`.
  Each library set is tagged `library = <library name>`; inline `sets`
  (`{name, genes, genes_down?}`) are tagged `library = 'My gene sets'`. Empty
  combined list → `ValueError`.
- `run_overlap_enrichment(genes, *, name=None, libraries=None, sets=None,
  gene_subset=None, min_set_size=5, max_set_size=500, min_overlap=2,
  key=None)` — synchronous. Universe = `var_names[_resolve_gene_mask(gene_subset)]`.
  Query symbols resolve against the universe with the same exact-then-
  case-insensitive rule; unresolved ones are reported. Fewer than 2 resolved
  query genes → `ValueError`. Stores and logs (below). Returns
  `{key, kind: 'ora', query: {name, n_input, n_in_universe, genes_missing},
  universe_size, gene_subset_type, n_sets_input, n_sets_tested, results,
  params}`.
- `prepare_gsea(ranking, *, libraries=None, sets=None, gene_subset=None,
  n_perm=1000, min_set_size=15, max_set_size=500, weight=1.0, seed=0,
  key=None) -> (compute_fn, apply_fn)`. `ranking` is one of:
  - `{'kind': 'diffexp', 'obs_column', 'group', 'reference': 'rest' | '<other group>',
    'method': 'wilcoxon' | 't-test', 'metric': 'score' | 'log2fc',
    'cell_subset': None | '<subset name>'}` — validated synchronously
    (column exists and is categorical, group and reference exist, subset
    exists, ≥ 2 cells each side). `prepare` snapshots a throwaway AnnData of
    the involved cells × universe genes; `compute_fn` runs
    `sc.tl.rank_genes_groups(..., use_raw=False)` on it and takes `scores`
    (the Wilcoxon z / t statistic) or `logfoldchanges` as the ranking.
  - `{'kind': 'pca', 'component': int, 'cell_subset': None | name}` — the
    column of `varm['PCs']` (or `PCs_<subset>`), restricted to the universe.
    Genes outside the PCA's gene mask have loading 0 and are dropped
    (non-finite handling reuses: set them to NaN before calling).
  - `{'kind': 'scores', 'genes': [...], 'scores': [...]}` — a client-supplied
    ranking, resolved to the universe; lets the UI (or a script) rank by
    anything later without a backend change.
  `compute_fn(report)` builds the ranking then calls `preranked_gsea`.
  `apply_fn` stores and logs, returning `{key, kind: 'gsea', ranking:
  {kind, label, n_ranked, genes, scores}, universe_size, gene_subset_type,
  n_sets_input, n_sets_tested, n_perm, results, params}`. `genes`/`scores`
  are the full ranked list (≤ ~20k entries) so the UI can draw the metric
  strip under the curve.
- **Storage.** `uns['xcell_enrichment']` is a dict `key → json.dumps(result)`.
  JSON strings are the existing precedent for record-shaped data (drawn
  lines, territories) because h5ad cannot write lists of dicts. Keys are
  sanitised (`_sanitize_subset_name` rule) and default to `ora_<query name>` /
  `gsea_<column>_<group>_vs_<reference>` / `gsea_pca<n>`; a taken key gets a
  numeric suffix rather than overwriting. `get_enrichment_results()` decodes
  and lists them newest first; `delete_enrichment_result(key)` removes one.
  `_log_action('enrichment_ora' | 'enrichment_gsea', params, {key,
  n_sets_tested, n_significant (padj ≤ 0.05)})`.

### Routes (`api/routes.py`)

```
POST   /enrichment/overlap           200  → run_overlap_enrichment result
POST   /enrichment/gsea              202  → {task_id, status}
GET    /enrichment/results           200  → {results: [summary...]}   (no per-set records)
GET    /enrichment/results/{key}     200  → full stored result
DELETE /enrichment/results/{key}     200  → {deleted: key}
```
Models directly above the routes: `EnrichmentLibraryRef {source, id,
species: str | None = None}`, `EnrichmentInlineSet {name, genes,
genes_down: list[str] | None = None}`, `OverlapEnrichmentRequest`,
`GseaRankingSpec` (flat: `kind`, `obs_column`, `group`, `reference`,
`method`, `metric`, `cell_subset`, `component`, `genes`, `scores`, all
optional with defaults), `GseaRequest`. `gene_subset` is typed as the full
`str | list[str] | GeneSubsetSpec | None` union. Error mapping per CLAUDE.md.

### Config (`config.yaml`)

```yaml
enrichment:
  min_set_size: 5          # ORA
  max_set_size: 500
  min_overlap: 2
  gsea_min_set_size: 15
  gsea_max_set_size: 500
  n_perm: 1000
  weight: 1.0
  padj_cutoff: 0.05        # results table default filter
```

## Frontend

### `lib/enrichment.ts` (pure, vitest)

- `formatP(p)` — `1.2e-5` / `0.032` / `<1e-3 (perm floor)` style.
- `curveToPath(curve, width, height, nRanked)` — SVG path string with the
  y-range symmetric around 0.
- `hitTicks(curve)` — x positions for the hit rug (one per hit vertex pair).
- `filterResults(results, {padjMax, query, hideBelowMinOverlap})`.
- `resultsToGeneSets(kind, results, {padjMax, topN})` — ORA rows → overlap
  genes; GSEA rows → leading edge; names `<set> (k/K)` / `<set> (NES x.x)`.
- `defaultTableRows` ordering helper; `rankingLabel(spec)`.
- `libraryOptions(availability)` — groups `cached` by source with species tags.

### Store

- `enrichmentSource: EnrichmentSource | null` and setter, where
  `EnrichmentSource = { kind: 'ora'; name?: string; genes?: string[] } |
  { kind: 'gsea' }`. `null` = closed.
- `'enrichment'` added to `GeneSetCategoryType`, `createDefaultCategories`
  (name "Enrichment"), the Gene Panel's category order and any exhaustive
  switch. Existing stored payloads survive via `mergeHydratedCategories`.

### `components/EnrichmentModal.tsx`

Global modal in `App.tsx`, gated on `enrichmentSource`. Two tabs, **Overlap**
and **GSEA**, preselected from `kind`. Layout mirrors `MarkerGenesModal`
(inline `styles`, `cfgDefault` seeds, click-outside/Escape close), width
~860px to fit the table.

Shared **Gene sets to test** block: checkbox list from `GET
/gene_set_sources` → `cached`, grouped by source, each row `name · n_sets ·
species`; plus a "My gene sets" picker (category → folder) that adds the
Gene Panel's sets inline. Empty cache → a line pointing at "Gene set
library…". Shared **Universe** select of boolean `.var` columns (as
MarkerGenesModal's gene subset select), default "all visible genes".

**Overlap tab.** Query: dropdown of Gene Panel sets across categories
(preselected when opened from a set's menu) or a "Paste genes" textarea;
shows `n resolved / n input` after a run. Params: min overlap, set-size
range. Run is a single POST.

**GSEA tab.** Ranking: radio *Differential expression* (obs column select,
group select, reference select = "rest" or another group, method, metric)
or *PCA loading* (embedding with `PCs`, component number). Optional named
cell subset select (from `cellSubsets`). Params: permutations, set-size
range, weight, seed. Run POSTs, then `pollTask` with a progress bar.

**Results** (both): summary line (universe size, sets tested, query
resolution), filter row (padj ≤ cutoff, search, "hide below min overlap"),
then a table sorted by p: name, library, size, overlap `k/K` or NES, p,
padj, and an inline bar (`−log10 padj` for ORA; signed NES for GSEA).
Clicking a row expands it: the gene list (overlap / leading edge, clickable
chips → `setSelectedGene`) and, for GSEA rows with a curve, an SVG panel:
running score, hit rug, and the ranking-metric strip. Buttons: **Add to gene
sets** (folder `<key>` in the Enrichment category, rows passing the filter,
capped at 50), **Copy TSV**, **Download SVG** (curve panel; reuses
`lib/svgExport.ts`), **Close**. After a run: `addScanpyAction` with the
params and `{key, n_sets_tested, n_significant}`; no schema refresh is
needed (nothing in obs/obsm changes).

**Entry points.** GenePanel per-set menu "Enrichment (overlap)…" →
`{kind:'ora', name, genes}`; GenePanel toolbar menu "Enrichment analysis…"
→ `{kind:'ora'}`; ScanpyModal custom function `enrichment` ("Gene-set
enrichment", opens the tool). `useData.ts` gains `runOverlapEnrichment`,
`startGsea`, `fetchEnrichmentResults` and the result types.

**Reopening old results.** A "Previous runs" select at the top of the
results area lists `GET /enrichment/results`; choosing one loads it via
`GET /enrichment/results/{key}`. Delete via a small × next to the select.

## Error handling

- Backend validation is synchronous in both paths, so bad input is a 400
  with a sentence the modal shows in its alert colour.
- Uncached library, unknown column/group/subset, empty universe (gene mask
  hid everything), < 2 resolved query genes, no sets in size range: all
  `ValueError` → 400.
- GSEA task failure surfaces through `pollTask`'s synthesized error status.
- All returned floats are finite (`odds_ratio` corrected, NaN scores dropped,
  NES 0 when null mean is 0).

## Testing

Backend (`backend/tests/`, self-contained, seeded):
- `test_enrichment.py` — `bh_adjust` against known values; hypergeometric p
  equals `scipy.stats.fisher_exact` one-sided on the same table; `odds_ratio`
  finite on zero cells; `resolve_sets` case-insensitive + directional split;
  GSEA: a set planted at the top of the ranking has ES ≈ 1, NES > 1, p at the
  floor, leading edge = the planted genes; a random set has |NES| ≈ 1 and p >
  0.05; a set planted at the bottom has ES < 0 and a leading edge from the
  tail; determinism under `seed`; ES from the vectorised routine equals a
  naive running-sum loop on a small example (the key numerical contract);
  ties and duplicates; `curve` present only for `curve_top_n`; weight 0
  reduces to the classic KS statistic.
- `test_enrichment_adaptor.py` — universe honours the session gene mask and a
  `gene_subset` column; results land in `uns['xcell_enrichment']` as JSON
  and round-trip through `get_enrichment_results`; key uniqueness; diffexp
  ranking (a gene up-regulated in the group ranks first with `score`
  metric, `reference='rest'` and `reference=<group>` both work); PCA
  ranking; `scores` ranking resolves case-insensitively; uncached library →
  `ValueError`; inline sets are tagged "My gene sets"; `_log_action` entries.
- `test_enrichment_routes.py` — 200 shape for overlap; 202 + task completion
  for GSEA via `task_manager`; dict `gene_subset` accepted (the 422 trap);
  400 mapping; results list/get/delete.

Frontend: `lib/enrichment.test.ts` for every helper; `npx tsc --noEmit`.

Browser: isolated stack (worktree, :8100/:5273) with the bundled toy data
and the mouse Hallmark library copied into a scratch cache dir via
`XCELL_...`/config `gene_set_sources.cache_dir`; seed a Gene Panel set via
`PUT /api/gene_sets`; run ORA from the set menu and GSEA on a Leiden cluster;
confirm the table, an expanded curve, "Add to gene sets" producing a folder,
and the analysis record entry. The live :8000 backend is not touched.
