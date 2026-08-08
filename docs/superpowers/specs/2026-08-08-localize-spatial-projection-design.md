# Localize: predicting spatial coordinates for dissociated cells

**Date:** 2026-08-08
**Status:** Approved, ready to implement

## Problem

A lab has two datasets of the same tissue: scRNA-seq (deep transcriptomes, no
positions) and spatial (positions, shallower or panel-limited expression). The
question — *where in the tissue was this cell?* — is answerable because
expression carries positional information, and the spatial dataset supplies the
map from one to the other.

xcell can load both at once (`primary` / `secondary`) but has no way to relate
them. This adds **Analyze → Spatial → Localize**: predict a coordinate for every
cell in the scRNA-seq dataset from a spatial reference, and write it as a new
embedding so the whole existing visualization stack — colouring, contours,
spatial neighbours, ligand-receptor — applies to it.

## Method: kNN projection

For each query cell:

1. Restrict both matrices to the shared gene set (∩ a chosen subset).
2. Transform each dataset **independently** (see below).
3. Find its *k* nearest reference cells in expression space.
4. Aggregate those neighbours' coordinates into a prediction.
5. Score how much that prediction should be believed.

Simple, transparent, and no training. It is essentially what CellTrek and
Seurat's coordinate transfer do, and it is the right first method here: the
parameters are all interpretable, and it fails in ways that can be *measured*
rather than ways that hide inside a fitted model.

## The failure mode this design is built around

**Averaging the coordinates of k transcriptionally-similar cells is only
meaningful if those cells are spatially co-located.**

If a cell type is dispersed through the tissue — immune cells, fibroblasts — the
k nearest spatial cells are scattered, and their centroid lands in the middle of
the tissue, quite possibly somewhere that cell type never occurs. The output
looks superb: every cell gets a plausible coordinate, the picture is smooth, and
it is systematically wrong. A bimodal type present in two distant patches is
worse: the mean lands squarely in the gap between them.

Nothing about the prediction itself reveals this. So the design carries **two
independent confidence axes**, both written to `.obs`, and neither optional:

| score | question it answers | how it fails |
|---|---|---|
| `similarity` | Does this cell resemble anything in the reference at all? | A cell type absent from the spatial data still gets placed *somewhere* |
| `confidence` | Do its matches agree on a location? | A dispersed or bimodal type produces a confident-looking centroid of nothing |

`confidence` is the weighted RMS spatial spread of the k neighbours around the
prediction, normalized by the reference tissue's own RMS radius, as
`max(0, 1 - spread / radius)`. 1 means every neighbour sat on the same spot; 0
means they were as scattered as the tissue itself — i.e. the prediction carries
no information. Normalizing by the tissue makes it comparable across datasets
and units.

Cells below `min_confidence` get **NaN coordinates** rather than a fabricated
point. Default is 0 (place everything, flag it), because silently dropping cells
is its own kind of lie — but the UI shows the distribution so the choice is
informed.

## Aggregation

Four ways to turn k neighbours into one point, each earning its place:

- **`weighted_mean`** (default) — similarity-weighted centroid. Best when the
  neighbourhood is unimodal; the classic estimator.
- **`median`** — coordinate-wise weighted median. Same idea, robust to a few
  scattered neighbours dragging the mean.
- **`best_match`** — the top-1 neighbour's exact coordinate. Predictions land on
  real tissue rather than in interpolated space, which matters when the tissue
  has holes or the reference is a discrete grid. Quantized, but never invents a
  location that does not exist.
- **`densest`** — find the tightest cluster among the k neighbours (the
  neighbour with the most companions within a radius derived from the reference's
  own spacing) and average only those. **This is the answer to bimodality**: mean
  lands between two patches, `densest` picks one.

## Cross-platform comparison

Raw expression correlation between scRNA-seq and Visium/Xenium is dominated by
technical differences, not biology. The transform is therefore not a detail:

- **`zscore`** (default) — per-gene standardization **within each dataset
  separately**, which is exactly what removes platform-level per-gene scaling.
- **`rank`** — per-cell ranks; with cosine this is Spearman, robust to any
  monotone platform difference.
- **`log1p`**, **`none`** — for same-platform or pre-normalized input.

Metric: `correlation` (default), `cosine`, `euclidean`. Correlation on z-scored
data and cosine coincide; both are offered because users think in both.

**Gene overlap is reported before anything runs**, following the precedent set by
the PySingleCellNet coverage check: proceeding silently with twelve shared genes
produces confident nonsense. The subset is resolved on the *reference* — the
spatial side owns which genes carry positional signal — and `spatially_variable`
(from `spatial_autocorr`, already in xcell) is the scientifically right choice
when available.

## Multi-section references

xcell supports spatial datasets holding several sections laid out side by side
with a gap. Averaging coordinates across sections produces a point in empty
space between them. With `section_col` set, each query cell is assigned the
weighted-majority section among its neighbours, and only that section's
neighbours contribute to the coordinate. The assignment is written to
`.obs['<key>_section']`.

## Evaluation — the part that makes this trustworthy

Two functions, both pure.

### `cross_validate_localization` — before you trust anything

Hold out a fraction of the *reference* and predict those cells' positions from
the rest, with the exact parameters the user has chosen. Ground truth is known,
so error is exact.

This is an **upper bound**: there is no platform gap within one dataset, so real
scRNA-seq→spatial error will be worse. The UI says so rather than letting a good
number be over-read.

### `evaluate_localization` — when truth is known

For a query with known coordinates (the toy benchmark, or a spatial dataset used
as a query to test the method).

Both report:

- `median_error`, `mean_error` in data units, and normalized by tissue diameter
  so the number means something without knowing the units.
- **`baseline_random`** — expected error from assigning a random reference cell's
  location. **`baseline_centroid`** — error from predicting the tissue centroid
  for everything. Without these the error is unfalsifiable: on a compact tissue
  the centroid baseline is deceptively strong, and a method that fails to beat it
  is worthless however small its error looks.
- `fraction_within` at 5% / 10% / 25% of tissue diameter.
- `structure_correlation` — Spearman between predicted and true pairwise
  distances over a subsample. Per-cell error and preserved global structure are
  different things; a method can do well on one and badly on the other.
- **per-group error** by a categorical column. A single median hides that the
  dispersed type is ten times worse than everything else, which is precisely the
  thing a user needs to know.
- **`confidence_error_correlation`** — Spearman between `confidence` and actual
  error. If the confidence score does not predict error, it is decoration, and
  the report says so. This is the design's own falsification test.

## The toy benchmark

`generate_toy_localize.py` emits a **pair** with exact ground truth:

- `toy_localize_spatial.h5ad` — the reference. An elliptical tissue with four
  region-restricted programs.
- `toy_localize_scrna.h5ad` — the query. The same biology re-sampled at a
  different depth with different dropout (so the transform choice matters), no
  coordinates in `.obsm['spatial']`, and the truth preserved in
  `.obsm['spatial_true']` plus `.obs['true_x'/'true_y']`.

The tissue is designed to exercise the failure modes rather than flatter the
method:

| population | why it is there |
|---|---|
| 4 region-restricted types | the easy case — should map well, sets the floor |
| `Immune`, uniformly dispersed | neighbours scatter → low `confidence`, large error. Without this, the confidence machinery is untested against the thing it exists for |
| `Bipolar`, in two distant patches | `weighted_mean` lands in the gap; `densest` picks a patch. Demonstrates why the aggregation choice exists |
| `Circulating`, query-only | absent from the reference → low `similarity`. Exercises the other confidence axis |

Deterministic (fixed seed), ~1200 reference and ~900 query cells over 120 genes:
small enough for the test suite, structured enough to be a real benchmark.

## Architecture

Follows the house rules: one pure module, an adaptor that owns AnnData, thin
routes.

```
reference adaptor ──spatial_reference_bundle()──► plain arrays
                                                     │
query adaptor ──prepare_localize(bundle, …)──────────┤
                                                     ▼
                                            localize.py (pure)
                                                     │
                          .obsm['X_spatial_pred'] ◄──┘ + .obs confidences
```

`backend/xcell/localize.py` is pure — arrays in, dicts out, no AnnData, no
adaptor import — which is what lets the projection and every metric be tested
directly.

**Two slots is new.** Every existing data route resolves one adaptor. Rather
than let the route reach into the reference's `.X` (which would break "the
adaptor is the only thing that owns AnnData"), the reference exposes
`spatial_reference_bundle(gene_subset, layer, section_col)` returning plain
arrays, and the query's `prepare_localize` consumes that. Both adaptors keep
their own data; the route just wires them together.

Long-running, so `prepare_localize` follows the `task_manager` contract:
validate synchronously (gene overlap, missing coordinates, same-slot misuse),
snapshot parameters, compute with no side effects, apply into the live AnnData.

## API

| route | purpose |
|---|---|
| `GET /api/localize/suggest` | what can act as a reference; gene-overlap preview; suggested k |
| `POST /api/localize/prepare` | run the projection (202, cancellable) |
| `POST /api/localize/cross_validate` | hold-out accuracy on the reference (202) |
| `POST /api/localize/evaluate` | score an existing prediction against known truth |

## UI

`frontend/src/components/LocalizeModal.tsx`, opened from Analyze → Spatial →
Localize. Reference slot picker; gene overlap shown with a severity band before
anything runs; k / transform / metric / aggregation / section column /
`min_confidence`; a **Check accuracy** button running the cross-validation and
showing error against both baselines and the per-type breakdown; then Apply,
which reports the confidence distribution and how many cells fell below the
threshold.

## Testing

`backend/tests/test_localize.py` (pure), `test_localize_adaptor.py`,
`test_localize_routes.py` — pytest, self-contained, seeded, per house
convention. The contracts worth pinning:

- **It beats the baselines on the toy pair.** A method that does not beat
  predict-the-centroid is not working; this is asserted, not eyeballed.
- **The dispersed type scores low confidence and the localized types score
  high.** The confidence axis is tested against data built to break it.
- **`densest` beats `weighted_mean` on the bimodal type**, which is the entire
  reason it exists.
- **The query-only type gets low `similarity`** while still receiving a
  coordinate — placement and plausibility are separate claims.
- Gene overlap is computed on names, not positions, and a tiny overlap raises
  rather than proceeding.
- Sections never mix: a prediction's coordinates come from one section only.
- `min_confidence` yields NaN, and NaN survives to `.obsm` without breaking the
  schema or the embedding endpoints.
- Metrics are JSON-safe — no `inf`, no `NaN` (use `None`), per house rule.

Then in the browser: load both toy datasets into the two slots, run it, and look
at the predicted embedding coloured by confidence.

## Out of scope

Deconvolution of multi-cell spots; optimal-transport or deep methods (Tangram,
novoSpaRc); mapping in the other direction (imputing expression onto spatial
spots); 3-D reconstruction; using the prediction to reorder the reference.
