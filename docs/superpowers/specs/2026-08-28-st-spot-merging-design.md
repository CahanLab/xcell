# Merging ST spots into single large cells

**Date:** 2026-08-28
**Status:** approved, ready for implementation planning

## The problem

A spatial transcriptomics spot is a fixed patch of tissue, not a cell. When the
cell is much larger than the spot — a hypertrophic chondrocyte is 30–40 µm
across, against a 2–16 µm Visium HD bin — one cell is spread over several
adjacent spots. Each spot then carries a fraction of that cell's transcripts, so
every spot in the region looks shallow, and no single spot is a usable profile of
the cell that is actually there.

The fix is to put the pieces back together: within a region the user identifies
as large cells, merge neighbouring spots up to the size of one cell and sum their
counts. The output is a dataset whose spots approximate cells rather than
patches.

## What we are building

**Analyze → Spatial → Merge Spots.** The user lassoes a region, sets a cell
diameter, previews the grouping, and commits. Committing produces a **new
dataset in a new slot** — a sibling tab — holding the whole tissue with the
region's spots replaced by merged ones. The source dataset is never modified.

## Design decisions

Each was settled in brainstorming, and each rules out a simpler alternative for
a stated reason.

### The merged result is a new dataset, not an edit

Merging creates rows that never existed. Every per-cell artifact already
computed — `.obsm` embeddings, Leiden labels, drawn-line projections,
comparison groups, prior DE tables — is either aggregated or stale, and there is
no aggregation that is right for an embedding.

Editing in place would silently invalidate all of it with no undo short of
reloading the file. A sibling dataset keeps the original open and comparable,
and xcell already holds N datasets as tabs. The cost is a second copy in memory
and re-running clustering on the merged one, which is the correct thing to do
anyway.

### Geometry decides which spots merge; similarity only vetoes

The spots most worth merging are the low-count ones, and those are exactly the
spots whose profiles are least able to say whether they should merge — a
correlation between two 40-count profiles is largely sampling noise. Ranking
candidate pairs by similarity therefore produces groupings that change between
runs on identical data.

So the merge is driven by geometry, which is stable and has biological meaning:
spots agglomerate until the footprint reaches one cell diameter. Similarity
enters only as a **veto** — a permissive floor that blocks a merge across a
clear boundary — never as a ranking.

Pure geometry was rejected because it merges across real boundaries whenever two
cell types share a footprint. Grid tiling was rejected on top of that: a tile
edge landing mid-cell splits it regardless of parameters, and rotating the
tissue changes the answer.

### Greedy agglomeration, not seeded growth

Seeded growth produces rounder merged spots, which is closer to how a cell sits.
But results swing on the seed set, and a poor one leaves ragged leftovers at the
region edge — tuning seeding rather than biology. Greedy pairwise agglomeration
over a neighbour graph is constrained hierarchical clustering with a diameter
cap: a well-understood shape, deterministic given a tie-break, and the veto
drops in as an edge filter rather than a special case.

### The merge sums raw counts; the veto correlates normalized profiles

Two different jobs, two different matrices. Summing is only meaningful on raw
counts. Correlating raw counts, though, makes the veto toothless: any two spots
correlate highly because both are dominated by the same few high expressors. The
veto therefore sees CPM-normalized, log1p'd profiles, computed on the fly and
never stored, where a genuine difference in cell type is what shows up.

### The whole tissue comes along; nothing derived does

The merged dataset covers the whole dataset, not just the region — spots outside
it pass through untouched — so it can be viewed and compared as a dataset rather
than a fragment.

But it carries no derived numbers at all. Numeric and QC `.obs` columns are
dropped, not averaged and not recomputed. Averaging a per-spot score across
members is an approximation the summed counts cannot justify; and keeping such
columns only for the untouched spots would leave a column populated in half the
tissue and blank in the other, which invites exactly the analysis that should not
be run. Anything wanted on the merged data is computed on it explicitly, correct
by construction.

`.obsm` follows the same rule: only the spatial key survives, holding the new
centroids. A UMAP computed on the original spots would still render for merged
rows while meaning nothing.

### Eligibility is a parameter, not a policy

The region is the user's statement of intent, so by default every spot in it
merges, giving one comparable population. An absolute count threshold or a
quantile is available for the case where well-covered spots should be left
alone — with the understanding that the output is then a mixture whose count
distributions differ by construction.

## Architecture

```
lasso selection ─► POST /api/spots/merge ─► DataAdaptor.merge_spots()
                                                     │
                                            spot_merge.py (pure)
                                                     │
                                          merged AnnData ─► set_adaptor(new slot)
```

Three units, each testable alone:

**`backend/xcell/spot_merge.py`** — pure. Coordinates, a counts matrix, a
normalized matrix for the veto, and parameters in; a group-label array and
statistics out. Knows nothing of AnnData or the adaptor. This is where the
algorithm lives and where tuning happens, and it holds the bulk of the tests.

**`DataAdaptor.merge_spots(...)`** — extracts the arrays, calls the module,
assembles the merged AnnData. Read-only on `self`.

**`POST /api/spots/merge`** — wraps the result in a `DataAdaptor`, registers it
under the requested slot, returns the schema. Synchronous, following `/combine`,
which is the existing precedent for a route that produces a new dataset.

Synchronous is a judgement about size: a lassoed region is hundreds to low
thousands of spots, and greedy agglomeration over a Delaunay graph at that scale
is well under a second. If regions turn out large in practice this moves to
`task_manager` behind the same route — the migration path, not built
speculatively.

## The algorithm

1. **Region and eligibility.** The region is `active_cell_indices`, the same
   selection mechanism every preprocessing route takes. Within it, `eligibility`
   picks candidates: `all` (default), `min_counts`, or `quantile`. A spot in
   the region that fails eligibility is not merged **and cannot be absorbed
   into a group** — it passes through untouched, which is the whole point of
   the non-default modes.

2. **Pitch estimation.** Median nearest-neighbour distance among region spots,
   converted to µm via `uns['xcell_spatial_scale']['um_per_unit']`. It sets
   `max_spots` automatically — `ceil(π·(d/2)² / pitch²)`, the number of lattice
   points a disc of that diameter covers — and is reported, so a wrong scale
   shows up as an absurd pitch rather than silently mis-capping every merge.

3. **Neighbour graph.** Delaunay triangulation over region coordinates, with
   edges longer than `max_diameter_um` dropped. Delaunay rather than a fixed
   radius because it adapts to irregular spacing and does not require the pitch
   to be known in advance.

4. **Greedy agglomeration.** Edges in a min-heap ordered by centroid distance.
   Pop the shortest and merge if all three hold:
   - merged footprint diameter ≤ `max_diameter_um`, where a group's diameter
     is the maximum pairwise distance among its members
   - member count ≤ `max_spots`
   - Pearson r of the two groups' normalized mean profiles ≥ `min_correlation`

   On a merge, recompute the group's summed counts, normalized profile and
   centroid, and push its surviving edges. Stop when the heap empties.

5. **Determinism.** Distance ties break on the two groups' lowest member
   source indices, compared as a sorted pair. Identical input yields identical
   labels on every run.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_diameter_um` | 40 | Cap on merged footprint — the cell-size knob |
| `max_spots` | `ceil(π·(diameter/2)² / pitch²)` | Hard ceiling; overridable |
| `min_correlation` | 0.15 | Veto floor; blocks clearly-different neighbours |
| `eligibility` | `all` | or `min_counts` / `quantile` |
| `min_counts` / `quantile` | — | Threshold for the non-default eligibility modes |
| `gene_subset` | `null` (all genes) | Genes the veto correlates over |
| `layer` | `counts` | Matrix to sum |
| `purity_column` | first categorical `.obs` | Column `merge_purity` votes on |
| `dry_run` | `true` | Preview only; registers no slot |
| `slot` | — | Slot for the merged dataset; required only when `dry_run` is false |

`gene_subset` takes exactly what `GeneSubsetPicker` already emits — a `.var`
column name, a `{columns, operation}` combination, or an explicit gene list,
which is what a Gene Panel gene set resolves to. The veto correlates the
**expression of those genes**, a column subset of the matrix; it never computes
a gene-set score.

## What the merged dataset contains

| Part | Content |
|---|---|
| `.X`, `layers['counts']` | Summed raw counts |
| other layers | Dropped — no defensible way to sum a smoothed or scaled matrix |
| spatial `.obsm` key | Unweighted centroid of member spots |
| other `.obsm` | Dropped |
| `.var` | Unchanged |
| categorical `.obs` | Majority vote; ties break on the dataset's category order |
| numeric and QC `.obs` | Dropped |
| `.uns['xcell_spot_merge']` | Parameters, source filename, spots before/after, resolved gene subset |

Two new `.obs` columns always: `merged_n_spots` (1 for untouched spots) and
`merge_group` (categorical id, blank outside the region).

A third, `merge_purity`, appears only when the dataset has a categorical `.obs`
column to vote on: the majority-vote fraction within each group, which is how a
merge that crossed a boundary the veto missed becomes visible. Which column it
votes on is the `purity_column` parameter, defaulting to the first categorical
`.obs` column so the number is never computed from a silently-chosen one.

Coordinates are the unweighted centroid: count-weighting would pull the point
toward the brightest member, which is not where the cell is.

## The tuning loop

`dry_run` defaults to true. The same route and the same code path run, but no
slot is registered and the response carries only what is needed to judge the
parameters:

```
region          412 spots
estimated pitch 15.2 µm  →  up to 6 spots per merge at 40 µm
result          412 → 118 merged spots
merge sizes     1:12  2:31  3:44  4:26  5:5
counts/spot     median 84 → 310
veto blocked    37 candidate pairs
correlations    at the cut: p10 0.11 · p50 0.34 · p90 0.72
purity          0.94 median (cell_type)
```

The last two lines are the instrument for `min_correlation`. If nothing is
blocked, the veto is not earning its place; if the correlation distribution
straddles the threshold, it is doing real work and is worth moving. `purity`
appears only when the dataset has a categorical `.obs` column to vote on.

## Error handling

Every one a 400 carrying the fix:

- No spatial coordinates on this dataset.
- No region selected — lasso the spots to merge first.
- `um_per_unit` unset — `max_diameter_um` is meaningless without it; points at
  the spatial scale setting.
- Region smaller than two spots.
- No integer counts available — names the `layer` parameter rather than
  silently summing normalized values.
- Gene subset resolving to nothing — reuses `_resolve_gene_mask`'s messages.

## Testing

**`spot_merge.py`** carries the bulk, being arrays in and arrays out: a
synthetic grid with planted groups; the diameter cap respected exactly; the veto
blocking across a planted boundary and not blocking within one; each eligibility
mode; determinism, asserted by running twice and comparing labels; and the
degenerate cases — a single spot, every pair vetoed, and collinear coordinates
that defeat Delaunay.

**Adaptor tests** assert the returned statistics *and* the resulting AnnData:
summed counts exact against a hand-computed group, centroids correct, categoricals
majority-voted, numeric and QC columns absent, `.obsm` reduced to the spatial key
alone, provenance in `.uns`, and untouched spots identical to their source rows.

**Route tests**: the slot appears, the source dataset is unchanged, and
`dry_run` registers nothing.

**Frontend**: the pure parameter and derivation logic, plus the `GeneSubsetPicker`
reuse.

Browser verification covers the loop that matters — lasso, preview, adjust
`min_correlation`, commit, and confirm the sibling tab holds the merged spots
while the original is untouched.

## Out of scope

- Drawing the merge groups on the spatial plot. The preview statistics answer
  the tuning question; a group overlay is a follow-up if they prove insufficient.
- Splitting spots, the inverse operation.
- Automatic cell-diameter estimation from the data. The diameter is a biological
  claim the user makes.
