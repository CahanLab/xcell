# External gene annotation resources, gene maps, and gene-set decomposition

Date: 2026-09-17. Status: designed without review — Patrick asked for this to be
built without his feedback unless something was unclear. Decisions he did not
make are marked **[decision]** so they are easy to find and reverse.

## The problem

xcell's Gene Panel is where a biological question turns into a set of genes,
but the only genes it can start from are the ones already in a file on disk.
Asking "how are the collagens expressed in this tissue?" means leaving xcell
to find a collagen list, reformatting it, and importing it — and then
discovering that "the collagens" are not one expression pattern but several,
which a single mean or UCell score flattens into one number.

Three things are missing:

1. **A way in for curated knowledge.** MSigDB, Enrichr's 228 libraries (GO,
   Reactome, KEGG, WikiPathways, PanglaoDB, CellMarker, Tabula Muris …) and
   STRING's interaction network are all one HTTP request away, and none of
   them is reachable from the Gene Panel.
2. **A view of genes as objects.** xcell can already build a gene–gene graph
   from expression (`gene_pca` → `gene_neighbors` → `cluster_genes`) but never
   shows it; and it knows nothing about similarity that comes from
   *annotation* rather than expression.
3. **A way to take a gene set apart.** When a set's members do not co-vary,
   the informative object is the set of *expression configurations* those
   genes take in the data, which is a dimensionality reduction of the
   cells × set-genes submatrix, not a score.

## What we are building

Four increments, each usable on its own, in this order:

- **SP1 — Gene Set Library.** Backend source adapters (MSigDB, Enrichr,
  STRING) with a local cache; routes to list, fetch, search and overlap-check
  libraries; a browser modal in the Gene Panel that imports chosen sets.
- **SP2 — The two simple use cases, end to end.** "Cluster cells on this
  gene set" from a gene-set row, and a hand-off from Cluster Genes to the
  Heatmap tab that honours the cell subset the clustering ran on.
- **SP3 — Decompose a gene set.** PCA or NMF on the cells × set-genes
  submatrix from a gene-set row, producing per-cell program scores (a score
  matrix), per-gene loadings, and one gene set per program.
- **SP4 — Gene map.** A gene–gene similarity built from expression,
  annotation membership, and STRING, with weights; a 2-D gene embedding and
  modules; a canvas scatter with lasso-to-gene-set, and a clustered
  similarity heatmap.

Everything the existing surveys found reusable is reused: `_resolve_gene_mask`
and `GeneSubsetSpec` for gene subsets, the `cell_context` vocabulary and
`_resolve_cell_context` for cell scope, `task_manager` for anything slow,
`.obsm` + `uns['xcell_score_matrices']` for per-cell scores,
`gene_coexpression.corr_matrix` for expression similarity, `gene_nmf` for the
NMF branch, `lib/hclust.ts` for dendrogram order, and the Gene Panel's
`OverflowMenu` rows and `addFolderToCategory` for every result.

## Design decisions

### Sources are adapters behind one interface, and the network is optional

`backend/xcell/gene_set_sources.py` is a pure module (no adaptor import) with
one class per source implementing:

```
catalogue(species) -> list[LibraryInfo]      # what can be fetched
fetch(library_id, species, report) -> Library  # download + parse
```

`LibraryInfo = {id, name, description, n_sets, species, version, url}` and
`Library = LibraryInfo + {fetched_at, sets: [{name, genes, description?, url?}]}`.
Adding a source is one class and one registry entry. The registry is what
`GET /api/gene_set_sources` reports, with `availability()` per source in the
`pyscn.py` sense (`{available, error, install_hint}`) so a blocked network
degrades to "cached libraries only" rather than a 500.

**[decision]** Only the Python standard library is used for HTTP
(`urllib.request` with a 60 s timeout and a `User-Agent`). `requests`,
`httpx` and `gseapy` are present in the environment but only transitively;
depending on them means declaring them, and the fetches here are three GETs.

**[decision]** The three sources are MSigDB (per-collection GMT + JSON
metadata from `data.broadinstitute.org`, latest release discovered from the
release index and pinned to `2026.1` when the index is unreachable), Enrichr
(`datasetStatistics` for the catalogue, `geneSetLibrary?mode=text` for a
library; weighted `GENE,1.0` tokens are stripped to the symbol), and STRING
(`get_string_ids`, `interaction_partners`, `network` on the REST API,
version 12). GO, Reactome, KEGG and WikiPathways come through Enrichr and
MSigDB rather than their own adapters, because those two already carry
current releases of all four and a fourth adapter would add nothing but a
second parser.

### A library is fetched once and searched locally

**[decision]** Cache directory: `$XDG_CACHE_HOME/xcell/gene_set_sources/`
(default `~/.cache/xcell/…`), overridable as `gene_set_sources.cache_dir` in
`config.yaml`. One JSON file per library
(`<source>/<species>/<library_id>.json`) and one catalogue file per source
(`<source>/catalogue.<species>.json`, refreshed when older than 7 days or on
demand). This is the first user-level cache in xcell; the only precedent is
`visium_hd.py`'s cache-next-to-the-file, which has no "file" here.

Searching, filtering and paging happen over the cached JSON in the backend,
which also computes the **overlap** of every returned set with the loaded
dataset: how many members are present, how many fall in each requested
boolean `.var` column (`highly_variable`, `spatially_variable`, …), and the
member names resolved into the dataset's symbol space. That is the number the
user filters on ("give me the ECM sets that actually have HVGs here"), so it
belongs with the search, not in a second round trip.

### Species and symbol case are handled once, server-side

Libraries are human-symbol (Enrichr, MSigDB Hs) or mouse-symbol (MSigDB Mm,
a few Enrichr libraries). The frontend already offers a case-convention
switch on import, but a browser that shows overlap counts needs to match
*before* import.

**[decision]** The overlap step matches exactly first, then case-insensitively
(the LigRec precedent, `adaptor.py` `upper_to_var`), and reports both counts.
The imported gene set always carries the dataset's spelling. The dataset's
species is inferred from Ensembl prefixes when the index is Ensembl ids
(`gene_symbols.detect_species`) and otherwise from symbol case (majority
Title-case → mouse, majority upper-case → human); the browser defaults its
species toggle to that and the user can override it. True orthology (symbols
that differ beyond case) stays out of scope, as the case-convention design
already records.

### Imported sets remember where they came from

**[decision]** A `GeneSet` gains an optional `source?: {source, library,
name, version}` field. The backend store is opaque to it (by design); the
frontend shows it in the row tooltip and the Library browser marks sets that
are already imported. Nothing else reads it.

### Decomposition writes what NMF already writes

`prepare_gene_nmf` is the reference: `.obsm[key]` + registry entry (per-cell
program scores, NaN for cells outside the scope), `.varm[f'{key}_loadings']`
(all genes, zeros outside the set), `uns['xcell_gene_set_decomposition'][key]`
(params, program gene lists, variance explained), `_log_action` and a
`codegen` entry. **[decision]** The PCA branch is added alongside NMF rather
than reusing `run_pca`, because `run_pca` overwrites the dataset's `X_pca`
and registers no score matrix; the decomposition of one gene set must not
disturb the cell embedding everyone else is looking at.

**[decision]** The PCA branch runs on `normalize_total + log1p` values with
per-gene z-scoring, so a highly expressed collagen does not own PC1 by
magnitude alone. Each component yields two gene lists — positive and
negative loadings above a threshold — filed as `genes` / `genesDown` of one
gene set, which is exactly the directional format UCell scores. The NMF
branch keeps `gene_nmf`'s defaults except `max_genes`, which is capped at the
set size.

**[decision]** Before either method runs, the modal shows a one-number
diagnostic: the fraction of the set's variance explained by its first
eigengene (from `gene_coexpression`'s coherence). A set at 0.8 is one
pattern and a score is the right tool; a set at 0.3 is what decomposition is
for. The number is computed synchronously by a small route so the user sees
it before choosing k.

**[decision, revised during implementation]** The variance fraction alone
does not decide "one pattern or several" on real single-cell data: for the
33 collagens on the 24k-cell hindlimb the first eigengene explains 4.6 % and
the mean pairwise |r| is 0.009, and the Kaiser rule (eigenvalues > 1) named
ten patterns because dozens of noise eigenvalues sit just above 1 when cells
vastly outnumber genes. How many patterns are real is therefore judged
against the Marchenko–Pastur noise edge `(1 + √(g/n))²`, the largest
eigenvalue independent genes produce by chance; `suggested_k` is the count
above it and `one_pattern` is true when at most one clears it or the first
carries over 70 % of what does. On the hindlimb that gives two collagen
patterns, which the decomposition confirmed (fibrillar vs basement-membrane
collagens). The modal draws the edge on the scree.

### A gene map is a similarity, then an embedding, then a picture

`backend/xcell/gene_similarity.py` (pure) builds a genes × genes similarity
from up to three channels, each in [0, 1]:

- **expression** — `gene_coexpression.corr_matrix` (bicor by default) on the
  log-normalised matrix, over the chosen cell scope, mapped to `(r + 1) / 2`;
- **annotation** — cosine similarity of gene × term membership vectors over
  the chosen cached libraries (a gene is a row, every set it belongs to is a
  column); genes with no annotation get 0 to everyone and are flagged;
- **STRING** — the combined score of each edge returned by `network` for the
  gene list, 0 where STRING has no edge.

The channels combine as a weighted mean over the channels present, with
weights from the modal (default expression 1.0, annotation 1.0, STRING 0.5
when enabled). **[decision]** Two-dimensional coordinates come from UMAP on
the precomputed distance `1 - S` when `umap-learn` is importable, else
classical MDS; modules from Leiden on the kNN graph of `S`. The result is
JSON-serialisable and lives in `uns['xcell_gene_maps'][key]` with `genes`,
`coords`, `modules`, per-channel availability and the similarity matrix
itself (float32, capped at 3,000 genes — a larger request is a 400 with the
cap in the message).

**[decision]** The map is drawn on a plain canvas inside a modal, not through
deck.gl. Every prop and overlay of the cell scatter is cell-indexed, gene
counts stay in the low thousands, and two canvas plots already exist in the
codebase to copy from. The lasso is a pure `lib/genePlot.ts` function that
returns gene names, and a lassoed selection becomes a gene set in one click.

## Architecture

```
external sources ──► gene_set_sources.py ──► ~/.cache/xcell/gene_set_sources/*.json
                              │
                     routes: /api/gene_set_sources/…      (catalogue, fetch [202], search, string)
                              │
adaptor: gene_set_overlap(sets, columns) ──► resolved names + counts   (POST /api/gene_sets/overlap)
adaptor: prepare_cluster_cells_by_gene_set ──► obsm X_pca_<key>, obsp, obs leiden_<key>, obsm X_umap_<key>
adaptor: gene_set_coherence ──► {n_genes, eigengene_pve, mean_corr}
adaptor: prepare_gene_set_decomposition ──► obsm[key] + registry, varm, uns, gene sets
adaptor: prepare_gene_map ──► uns['xcell_gene_maps'][key]        gene_similarity.py (pure)
                              │
frontend: GeneSetLibraryModal, ClusterCellsByGeneSetModal, DecomposeGeneSetModal, GeneMapModal
          + gene-set row ⋯ items; pure lib/geneSetLibrary.ts, lib/genePlot.ts
```

Every route that touches the dataset takes `dataset: str | None = Query(None)`
last. Source and cache routes are slot-less, like `/api/gene_sets`.

## Routes

| Route | Kind | Purpose |
|---|---|---|
| `GET /api/gene_set_sources` | sync | sources, availability, cache dir, cached libraries |
| `GET /api/gene_set_sources/{source}/libraries?species=&refresh=` | sync | catalogue (cached unless stale/refresh) |
| `POST /api/gene_set_sources/{source}/libraries/{id}/fetch` | 202 | download + cache one library |
| `GET /api/gene_set_sources/{source}/libraries/{id}/sets?species=&q=&gene=&offset=&limit=` | sync | search a cached library |
| `POST /api/gene_sets/overlap` | sync, dataset | overlap of given sets with the dataset and boolean columns |
| `POST /api/gene_set_sources/string/partners` | sync | STRING partners of seed genes → gene set + edges |
| `POST /api/gene_set_sources/string/network` | sync | STRING edges among a gene list |
| `POST /api/gene_sets/coherence` | sync, dataset | eigengene PVE + mean correlation of a set |
| `POST /api/gene_sets/decompose` | 202, dataset | PCA / NMF decomposition of a set |
| `GET /api/gene_sets/decompositions[/{key}]` | sync, dataset | list / read results |
| `POST /api/gene_sets/cluster_cells` | 202, dataset | PCA → neighbours → Leiden (→ UMAP) on a set |
| `POST /api/gene_map/run` | 202, dataset | build similarity + embedding + modules |
| `GET /api/gene_map/{key}` | sync, dataset | read a map (coords, modules, optional similarity) |

Errors: `ValueError → 400` (bad library id, unknown genes, over the gene cap,
network failure with the URL and reason in the message), `KeyError → 404`,
no data → 503. A network failure inside a task is an `error` task status
whose message names the URL.

## Parameters and defaults

Added to `config.yaml` under `gene_set_sources:` (`cache_dir`,
`catalogue_ttl_days: 7`, `string_required_score: 400`,
`string_partner_limit: 50`), `gene_set_decomposition:` (`method: pca`,
`k: 3`, `loading_threshold: 0.2`), `gene_map:` (`metric: bicor`,
`weights: {expression: 1.0, annotation: 1.0, string: 0.5}`,
`n_neighbors: 15`, `resolution: 1.0`, `max_genes: 3000`), and
`cluster_cells_by_gene_set:` (`n_comps: 20`, `n_neighbors: 15`,
`resolution: 1.0`, `run_umap: true`). All mirrored in
`docs/config.example.yaml`.

## Testing

Backend, all offline: source parsers on captured fixture strings (a GMT
line with weights, a MSigDB JSON record, a STRING partner record), cache
round-trips in `tmp_path`, catalogue staleness, overlap counting including
case-insensitive resolution, coherence on a planted one-pattern set vs a
planted two-pattern set, decomposition on a planted two-program set (the
right genes land in the right program, `.obsm`/`.varm`/registry state is
asserted), cluster-cells on planted populations, similarity channels on
hand-built memberships and edge lists, and route tests with
`monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)` and
the network functions monkeypatched. Live-network tests are gated with
`skipif` on an environment flag and never run in CI.

Frontend: pure `lib/` modules with vitest (library search ranking and
selection state, overlap presentation, lasso hit-testing, program
labelling). Every modal is verified in the browser on the isolated stack,
and SP1 is verified against the real sources.

## Out of scope

- Orthology beyond symbol case (HGNC HCOP), already deferred elsewhere.
- Enrichment analysis (over-representation / GSEA) — the library is a way
  *in*, not a statistics tool; `gseapy` is a separate decision.
- Bulk download of STRING's full network; only per-query REST calls.
- Persisting gene maps across `--reload` beyond what `uns` gives.
- A center-panel tab for the gene map; the modal can be promoted later.
