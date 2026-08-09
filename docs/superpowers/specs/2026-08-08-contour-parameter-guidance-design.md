# Contour: parameter guidance in the modal

**Date:** 2026-08-08
**Status:** Approved, ready to implement

## Problem

Analyze → Spatial → Contour asks for six numbers before it will do anything —
contour levels, grid resolution, smoothing sigma, log transform, section
column, and (multi-set) profile k. Today each carries a one-line `title=`
tooltip that restates the label. Two of them are prefilled from a data-aware
suggestion and then never mentioned again, so a user who edits one has no way
to know they have left the range the data supports.

The parameters are not independent, and the coupling is invisible:

**Sigma is measured in grid pixels, so grid resolution silently rescales it.**
`_interp_smooth_subset` interpolates onto a `grid_res × grid_res` grid spanning
the tissue extent, then applies `gaussian_filter(sigma=smooth_sigma)`. One grid
pixel is `extent / grid_res` in data units, so the physical smoothing radius is
`smooth_sigma × extent / grid_res`. **Doubling grid resolution halves the
actual smoothing.** `suggest_smooth_sigma` knows this — it solves for a radius
of about two spot spacings — but the moment a user raises `grid_res` on its
own, the smoothing they were given quietly shrinks by the same factor, and the
bands break into speckle for reasons nothing on screen explains.

**Bands are equal-width, not equal-count.** Thresholds are
`linspace(0, vmax, levels + 2)[1:-1]` over the smoothed field. On a skewed
field — the normal case, since a module is high in one region and near zero
elsewhere — the top band can hold a handful of spots. The review-phase
histogram shows this, but not why, and the natural reading ("my gene set barely
expresses") is wrong.

**`log_transform` defaults differently on each side.** The route models default
it to `True` (`ContourizeRequest`, `MultiContourPrepareRequest`); the modal
defaults it to `False` and always sends explicitly, so the effective default in
the UI is off. The tooltip says "turn on for raw counts" — and the user has no
way to know whether their `.X` holds raw counts. xcell already knows:
`layer_scale.assess_matrix_scale` runs on every entry from
`/api/scanpy/layers` and returns a verdict of `raw_counts`, `log_normalized`,
`z_scored`, and so on.

This adds the guidance pattern Localize established — per-field tips, live
advice on the current settings, and a collapsible guide — to both phases of the
Contour modal.

## Approach

### Express advice in spot spacings, not in "×the suggestion"

"Your grid is 3× the suggested value" tells a user they disagree with a number
they never chose. The quantities that mean something are:

- **pixel size** `px = extent / grid_res`, compared to `median_spacing`.
  `px < spacing` means the interpolation is producing detail at a scale finer
  than the sampling, i.e. inventing it.
- **smoothing radius** `r = smooth_sigma × px`, in the same data units, as a
  multiple of `median_spacing`. Below 1 the filter does not reach the next
  spot, so the field keeps per-spot noise and the bands speckle. Above about 6
  it reaches across a whole zone and adjacent tissues merge.

Both are computed in the frontend from values the backend supplies, and both
are quoted back to the user in the message ("smoothing reaches 0.4 spot
spacings"). This also makes the grid×sigma coupling visible: change one and the
other line moves.

### Backend change: one endpoint, additively

`suggest_contour_params()` currently returns `{grid_res, smooth_sigma}`. It
gains `n_spots`, `median_spacing`, and `extent` — every one of which it already
computes inside `suggest_smooth_sigma` and throws away. Existing callers are
unaffected.

Everything else the advice needs is already served:

| input | source |
|---|---|
| `.X` scale verdict | `/api/scanpy/layers` → `X` entry → `scale.verdict` |
| categorical obs columns | `/api/obs/summaries` (modal already fetches) |
| `X_pca` present | `/api/schema` → `embeddings` |

### `adviseContour()` — live checks

A pure exported function, mirroring `adviseParameters` in `LocalizeModal.tsx`,
returning `{level: 'warn' | 'info', text}[]`. Select phase:

| fires when | says |
|---|---|
| pixel size < spot spacing | grid is finer than the sampling; cubic interpolation is inventing structure between spots |
| pixel size > ~3 spot spacings | grid coarser than the tissue; zones smaller than a few spots disappear |
| smoothing radius < 1 spacing | filter does not reach the neighbouring spot — bands will speckle rather than form zones |
| smoothing radius > ~6 spacings | filter spans a whole zone; adjacent tissues merge into one band |
| grid edited off the suggestion while sigma was left at it, and neither radius rule above fired | names the *new* physical radius, since changing the grid rescaled it. Suppressed when a radius warning already said the same thing more directly |
| log off and `X` verdict is `raw_counts` | quotes the observed max; percentile clipping is at 1/99, so a few high spots still set the range |
| log on and verdict is `log_normalized` / `log_transformed` | second log flattens the field and pulls every band toward the middle |
| log on and verdict is `z_scored` | `log1p` of negative values — the score is meaningless |
| no section column set, but a categorical column with 2–20 levels exists | interpolation spans the gap between sections; expression bleeds across |
| ≥2 sources and no `X_pca` | says so now instead of failing at prepare |
| a source with ≤2 genes | the module score is essentially one gene's field |
| `contour_levels` ≥ 6 | levels only subdivide the cutoff choice; they do not change the underlying field |

Review phase:

| fires when | says |
|---|---|
| chosen cutoff selects 0 spots | nothing is high — this module cannot contribute a tissue |
| chosen cutoff selects > 50% of spots | high nearly everywhere; it will win conflicts by size rather than by signal |
| two modules both high in > 40% | large overlap, so most assignments come from neighbour voting, not from the contours |
| `profile_k` > 50 | the vote reaches well beyond the local neighbourhood and pulls toward whatever is regionally dominant |

Data-dependent thresholds carry a *why* comment at the definition, per the
repo's comment convention.

### `<ContourGuide />` — collapsible, per phase

Same shape as `ParameterGuide`, behind a "Show how to choose these" ghost
button, default collapsed.

*Starting points* — a table keyed on what the user has, not on parameter names:
Visium (55 µm spots, ~2–5k), Visium HD / binned, Xenium & other single-cell
resolution, and multi-section slides.

*Trade-offs* — leads with the grid×sigma coupling, because it is the one thing
a user cannot infer from the form. Then: what banding does (equal-width, not
equal-count, and why the top band is usually small); what the cutoff means in
the multi-set path; and that `log_transform` and the 1/99 percentile clip
interact, since clipping already tames the tail that log is usually reached for.

*Avoid* — raising the grid to sharpen a blurry map (it sharpens by removing
smoothing, which is not the same as resolving structure); contouring across
sections without a section column; reading band thresholds across runs, since
`vmax` is per-run and per-section-set; picking cutoffs by which map looks
cleanest rather than by the spot counts shown next to them.

The existing `Param` tooltips get rewritten to state cost rather than restate
the label.

## Tests

**Backend** — `suggest_contour_params` returns the three new fields, with
`median_spacing` matching a hand-computed nearest-neighbour median on a regular
grid fixture, and `extent` matching the larger axis span. Existing callers keep
working (the multi-contour and route tests already cover them).

**Frontend** — `adviseContour` is exported and pure, so it is unit-tested
directly: one case per rule firing, one case per rule staying silent, and the
grid×sigma interaction (raise `grid_res` alone → the radius warning appears).
Ordering is asserted, since warnings must precede infos.

## Out of scope

- Reconciling the `log_transform` default mismatch between the route models and
  the modal. The advice makes the current state legible; changing the default
  would alter results for anyone whose saved config relies on it, and is a
  separate decision.
- A preview of the contoured field inside the modal. It would answer these
  questions better than any prose, and it is a much larger piece of work.
