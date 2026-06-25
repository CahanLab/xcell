# UCell-style Gene-Set Scoring — Design

Date: 2026-06-25
Status: Approved (brainstorm) → ready for plan

## Goal

Add UCell-style gene-set scoring to xcell. UCell signatures are *directional*:
they carry genes expected **up** (high) in a cell population and genes expected
**down** (low / undetected). Scoring is a per-cell rank-based AUC statistic that
is robust to dataset size and normalization.

Two consumption paths (both requested):
1. **Persisted** — a "Score with UCell" action writes scores to `.obs` columns
   (`UCell_<name>`), which are then colorable like any `.obs` column and ride
   along in the existing `/export/h5ad` download.
2. **Interactive** — UCell as a scoring method in the existing
   *Cells → color cells by selected gene set* flow, recomputed on demand, not
   persisted.

## The UCell algorithm (canonical, from carmonalab/UCell `HelperFunctions.R`)

### Per-cell ranks
For each cell, rank genes by expression **descending** (rank 1 = highest-expressed),
`ties.method = "average"`. Cap at `maxRank`: any rank `>= maxRank` is treated as
`maxRank`. Stored sparse (rank `>= maxRank` → 0). Because ranks are invariant to
per-cell monotonic transforms (per-cell scaling + log1p preserve within-cell
order), the **default source layer is raw `counts`** (fallback `.X`).

### `u_stat` for a gene set of size `n` (AUC form)
```
rank_sum     = sum over the set's genes of (capped rank), per cell
rank_sum_min = n * (n + 1) / 2
u_score      = 1 − (rank_sum − rank_sum_min) / (n * maxRank − rank_sum_min)
```
`u_score` ∈ [0, 1]: 1.0 when the set's genes are the top-expressed in the cell;
0.0 when all sit at/below `maxRank`.

### Combine up + down (per cell)
```
u_p   = u_stat(up_genes)
u_n   = u_stat(down_genes)        # 0 if no down-genes
score = max(u_p − w_neg * u_n, 0) # w_neg default 1.0, clipped at 0
```

### Consequences honored by this design
- **Down-only sets score 0 everywhere** → warn & skip in UI; every set needs ≥1 up-gene.
- **Missing genes are skipped** (filtered to genes present in `.var`), matching the
  rest of the app. This is UCell's `"skip"` mode, not the R default `"impute"`;
  numbers can differ slightly from UCell-R. Documented, intentional.
- **Signature longer than `maxRank`** → auto-raise `maxRank` to the longest
  signature length (with a note in the response), rather than erroring.

## Data model — directional gene sets

Extend the frontend `GeneSet` interface (`frontend/src/store.ts`) with an
optional down-list:

```ts
export interface GeneSet {
  id: string
  name: string
  genes: string[]        // the UP / positive list (unchanged)
  genesDown?: string[]   // NEW: the DOWN / negative list (optional)
  pinned?: boolean
}
```

- A set with no `genesDown` behaves exactly as today everywhere. Existing
  features (set-ops / Combine modal, `.var`-columns-as-sets, mean coloring) keep
  operating on `genes` only — the down-list is **UCell-specific** and ignored by
  those features (YAGNI: no down-aware set algebra).
- **Backend store needs no change**: `backend/xcell/gene_set_store.py` is opaque
  JSON that round-trips the frontend shape verbatim, so `genesDown` rides along.

## Input file format (locked — users generate lists against this)

Recommended JSON:
```json
{
  "sets": [
    { "name": "CD8_T_effector",
      "up":   ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG"],
      "down": ["SELL", "CCR7", "TCF7"] },
    { "name": "Epithelial",
      "up":   ["EPCAM", "KRT8", "KRT18"] }
  ]
}
```

Field rules:
- `name` — required string; stored score column is `UCell_<name>`.
- `up` — genes expected high. Synonyms: `positive`, `genesUp`, `genes`.
- `down` — genes expected low/undetected. Synonyms: `negative`, `genesDown`. Optional.
- Entries are gene-symbol strings; objects with a `symbol` field are also accepted
  (geneset-builder `.gsb.json` compatibility).
- Symbols must match `.var` exactly (case/species sensitive). Absent genes skipped.

Also accepted (identical semantics):
- Top-level array: `[{ "name": ..., "up": [...], "down": [...] }, ...]`.
- The UCell **`-` suffix** convention in any flat list (JSON `genes` array, GMT
  line, CSV column): a token ending in `-` → down-list (suffix stripped); a
  trailing `+` is stripped and treated as up. GMT line:
  `name <TAB> description <TAB> CD8A <TAB> CD8B <TAB> SELL- <TAB> CCR7-`.

## Backend (adaptor + routes)

New adaptor methods (`backend/xcell/adaptor.py`):

- `_ucell_ranks(layer, maxRank)` — returns the capped sparse rank matrix
  (cells × genes), lazily computed and cached on the adaptor in
  `self._ucell_rank_cache` keyed by `(layer, maxRank)` and validated against
  `id(self.adata)` (+ shape). Computed in cell-chunks to bound peak memory:
  per chunk, densify, descending average-tie rank, cap at `maxRank`, store
  sparse. **Not** written into `adata` → never bloats `/export/h5ad`. Cache is
  dropped whenever an op reassigns `self.adata`.

- `_ucell_score_one(up_idx, down_idx, ranks, maxRank, w_neg)` — per-cell score
  array via the `u_stat` + `max(u_p − w_neg·u_n, 0)` formula above.

- `score_gene_sets_ucell(sets, layer='counts', maxRank=1500, w_neg=1.0, store=True)`
  — `sets` is a list of `{name, up: [...], down: [...]}`. Filters each list to
  present genes (skip missing), auto-raises `maxRank` if a signature is longer,
  computes/uses cached ranks, scores each set. When `store=True`, writes
  `.obs['UCell_<sanitized name>']` with a collision-safe suffix. Returns
  `{results: [{name, obs_column, min, max, n_up_used, n_down_used, skipped?}], maxRank, layer}`.

Routes (`backend/xcell/api/routes.py`):
- `POST /scanpy/score_genes_ucell` — persisted batch. Body: `{sets, layer, maxRank, w_neg}`.
  Frontend calls `refreshSchema()` afterwards so new columns become colorable.
- `POST /expression/ucell` — interactive, single set → `{values, min, max}`,
  reuses the cached ranks, writes nothing. Kept **separate** from
  `get_multi_gene_expression` (per-gene-norm + aggregate) because rank-AUC and
  normalize+aggregate are different enough that overloading would muddy both.

## Frontend UX

- **GenePanel** — a "Score with UCell" action on a selected set (and on a
  multi-selection) opens a small form: `maxRank` (1500), source layer (default
  `counts`, from `list_layers`), and `w_neg` (1.0, under an "advanced" toggle).
  Submitting POSTs the batch endpoint, then `refreshSchema()` + a toast naming
  the new `UCell_<name>` column(s).
- **Interactive color-by-set** — add "UCell" as a scoring method alongside the
  current mean / per-gene-norm options; directional sets auto-use their
  down-list; nothing is written. Calls `POST /expression/ucell`.
- **Directional display** — a set carrying a down-list shows up/down counts
  (e.g. `12↑ 4↓`); down genes are styled distinctly in the chip view. Minimal.
- **Import** — extend `ImportModal.parseJSON` + `ParsedGeneList` to recognize the
  up/down JSON forms and the `-`/`+` suffix convention across JSON/GMT/CSV, and
  carry `genesDown` through `addGeneSetToCategory` / `addFolderToCategory`.

## Edge cases / validation

- Down-only set → identically 0; UI warns & skips, response marks `skipped`.
- Missing genes → filtered (skip); response reports `n_up_used` / `n_down_used`.
- Signature length > maxRank → auto-raise maxRank, note in response.
- Empty `up` after filtering → skip set with a message.
- Rank cache keyed on `(layer, maxRank, id(self.adata), shape)`; recomputed on mismatch.

## Testing

- **Backend pytest** (`backend/tests/`): `u_stat` vs a hand-computed small matrix;
  up-only, up+down, down-only→0, missing-gene skip, `maxRank` capping, `w_neg`
  weighting, obs-column write + collision suffix, rank-cache reuse +
  invalidation, layer `counts`→`X` fallback, interactive endpoint parity with the
  stored path.
- **Frontend**: `npx tsc --noEmit` + `npm run build`. Playwright smoke on the
  isolated stack (backend :8001 + vite :5180): load a directional set, run Score
  with UCell, color cells by the resulting `UCell_*` obs column. Clean up servers
  + PNGs.

## Out of scope (YAGNI)

- Down-aware set algebra in the Combine modal.
- Persisting the rank matrix into the h5ad.
- `missing_genes="impute"` mode toggle (we always skip).
- Cross-session rank caching to disk.
