# Multi-contour tissue annotation — design

**Date:** 2026-06-16
**Status:** Approved, proceeding to implementation plan

## Problem / goal

xcell's **Contourize** (Analyze → Spatial) turns one gene set into a banded
spatial-expression column. Users want to annotate a whole tissue section by
running several tissue modules (cartilage, muscle, tendon, interzone, skin,
dermis…) and **fusing them into one `.obs` annotation** where each spot is
labeled with the tissue it belongs to.

This adds **Analyze → Spatial → Multi-contour**: select N gene sets, contourize
each, binarize each to "high / not-high", and assign every spot a single tissue
label — resolving spots that are high in multiple modules via their spatial +
expression neighborhood.

Secondary goal: **parameter guidance** for contourize (what each knob does,
data-aware defaults), folded into this feature rather than shipped separately.

## Current contourize (for reference)

`run_contourize(genes, contour_levels=6, log_transform=True, smooth_sigma=2.0,
grid_res=200, clip_percentiles=(1,99))` in `adaptor.py`:
per gene → log1p → percentile-clip → min-max to [0,1]; average across genes →
per-spot summary; interpolate onto `grid_res²` grid (cubic); Gaussian-smooth
(`smooth_sigma`, in grid-pixel units); cut into `contour_levels` evenly-spaced
thresholds in [0, vmax]; label each spot with the highest band it meets.
Stored as an ordered categorical. The continuous smoothed score sampled at each
spot (the value before banding) is the key reusable quantity.

## Decisions (from brainstorming)

- **Entry:** new Multi-contour modal under Analyze → Spatial.
- **Per module:** contourize with a *low* level count (default **3**).
- **Binarize:** "high" = band ≥ a per-module cutoff. **Auto** cutoff = top band;
  **per-module override** via the review UI.
- **Assignment per spot** (`H(s)` = modules where spot is high):
  - `|H|=0` → `unassigned`
  - `|H|=1` → that module's tissue label (= gene-set name)
  - `|H|≥2` → conflict → resolve via spatial-then-profile kNN (below)
- **Conflict resolution:** candidate pool = spot's spatial-graph neighbors
  (`obsp['spatial_connectivities']` if present, else coordinate-kNN built on the
  fly); keep only **unambiguous** neighbors (`|H|=1`); rank by **Euclidean
  distance in `X_pca`**; take nearest **k** (default 15); assign **majority**
  tissue. **Tie or empty pool → `unassigned`.** **Single pass** (resolve against
  original unambiguous labels only).
- **`X_pca` is a hard prerequisite** — refuse to run without it (error tells the
  user to run PCA first). Profile space = full `X_pca` (not gene-set-restricted).
- **Output:** one **unordered categorical** `.obs` column (default name `tissue`),
  categories = selected gene-set names + `unassigned`.
- **QC columns (default OFF):** a single "save QC columns" checkbox adds
  `tissue_status` (`single` / `resolved` / `unassigned`) and per-module
  `<setname>_high` booleans.
- **Single-contourize** gets the same improved tooltips + data-aware defaults
  (same code path).

## Architecture

Two-phase backend so the per-module binning review is interactive:

- **`prepare_multicontour(gene_sets, contour params)`** — validates prereqs;
  computes each module's continuous per-spot score (shared scoring core), level
  bands, auto cutoff, and a histogram of spots per band. Writes nothing; returns
  the review payload and caches per-module scores server-side under a token.
- **`finalize_multicontour(token | recompute, cutoffs, profile_k, out_name,
  save_qc)`** — binarize → assign → resolve → write `.obs` column(s); log action.

New module **`backend/xcell/multicontour.py`** (pure, testable):
`compute_module_score`, `auto_cutoff`, `binarize`, `assign_single_high`,
`resolve_conflicts_knn`. The adaptor exposes thin `prepare_multicontour` /
`finalize_multicontour` orchestration methods (matching the existing
prepare/apply cancellable pattern); API routes are thin wrappers.

**Refactor (in scope):** extract the contourize scoring core (gene normalize →
average → grid interpolate → smooth → sample-at-spots) from `run_contourize`
into a shared function used by both single contourize and `compute_module_score`.
Characterization test guards existing behavior.

Alternatives rejected: one-shot auto-only (drops the override step);
binarize/assign in the frontend (splits logic across tiers).

## UX flow (modal)

1. Multi-select gene sets from the gene-set store. Global params with smart
   defaults: `contour_levels`=3, `smooth_sigma`, `grid_res` (data-aware, §below),
   `log_transform`.
2. **Compute** → `prepare` (cancellable). 
3. **Review & bin:** one row per module — histogram of spots per band, auto
   cutoff preselected (top band), a control to move the high/not-high boundary;
   live "high" spot count.
4. Resolution params: `profile_k` (default 15), output column name (default
   `tissue`), "save QC columns" checkbox (default off).
5. **Finalize** → writes column(s); toast summarizes #per tissue, #resolved,
   #unassigned. Cell Manager refreshes to show the new column.

## Parameter guidance (folded in)

- **Tooltips with direction-of-effect** on every param (multi-contour *and*
  single contourize). Example for `smooth_sigma`: "higher = smoother, fewer/larger
  zones; too high merges adjacent tissues. Measured in grid pixels, so its
  real-world radius scales with grid_res."
- **Data-aware suggested defaults:** `grid_res` from spot count / spatial extent;
  `smooth_sigma` from median nearest-neighbor spot spacing. Shown as the
  prefilled default with reasoning in the tooltip. (Old `grid_res=200` is coarse
  for 344k-bin Visium HD.)
- The per-module histogram + live high-count in step 3 **is** the preview — no
  separate preview endpoint.

## Errors / edge cases

- Missing `X_pca` → clear error ("Run PCA first").
- < 2 gene sets, or a set with genes absent from `var_names` → error listing the
  problem.
- No spatial coords → error (existing `has_spatial` prereq).
- Module with degenerate score (all equal) → its bands collapse; auto cutoff
  yields 0 high spots; surfaced in the review histogram (not an error).
- Token cache miss on finalize (e.g. server restart) → finalize recomputes
  scores from the same params rather than failing.

## Testing

Unit tests over pure functions in `multicontour.py` on a small synthetic spatial
AnnData (hand-placed spots, 2–3 toy gene sets, known `X_pca`, known spatial
graph):
- single-high spot → correct tissue;
- 0-high spot → `unassigned`;
- constructed ≥2-high spot → resolves to the correct neighbor-majority tissue;
- engineered tie / no unambiguous neighbors → `unassigned`;
- absent `obsp['spatial_connectivities']` → coordinate-kNN fallback path taken;
- `auto_cutoff` selects the top band; an override cutoff changes the high set;
- categories == gene-set names + `unassigned`, dtype unordered categorical.
Plus a characterization test that the extracted scoring core reproduces current
`run_contourize` output on a fixture.

## Out of scope

- Live full-field contour preview rendering (the histogram is the preview).
- Iterative conflict propagation (single pass only).
- argmax fallback for conflicts (decided: leave `unassigned`).
- Profile spaces other than `X_pca`.
