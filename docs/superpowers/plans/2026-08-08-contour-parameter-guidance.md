# Contour Parameter Guidance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Contour modal explain what its parameters cost, and warn — live, against the actual data — when the current settings are wrong for it.

**Architecture:** The advice logic is pure and lives in `frontend/src/lib/contourAdvice.ts`, so vitest exercises it directly rather than through the DOM. It takes the current form state plus a few facts about the dataset and returns `{level, text}[]`. `MultiContourModal` renders that list, plus a collapsible `<ContourGuide />`. One additive backend change supplies the dataset facts; everything else is already served by existing endpoints.

**Tech Stack:** React 18 + TypeScript + vitest; Python 3 / FastAPI / numpy / scipy; pytest.

**Spec:** `docs/superpowers/specs/2026-08-08-contour-parameter-guidance-design.md`

## Global Constraints

- Branch `feat/contour-guidance` off `main`; merge back with `--no-ff`.
- **Do not start this until `feat/graph-choice` is merged.** They touch different files, but sequencing keeps the browser verification honest.
- TDD: failing test, run it, watch it fail, implement.
- Backend tests from `backend/`: `pixi run -e dev pytest`. Frontend: `cd frontend && npm test`, typecheck `npx tsc --noEmit`.
- Pure frontend logic goes in `frontend/src/lib/` with a sibling `.test.ts` — the established pattern (`savePath.ts`, `datasetIdentity.ts`, `lasso3d.ts`).
- Styling is inline. `MultiContourModal.tsx` uses a module-level `styles` object with the palette: panel `#16213e`, border/field `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`/`#888`.
- Stats crossing the API must be JSON-serializable — no `inf`, no `NaN`. Use `None` for "not computable".
- Comments explain *why*, especially where a threshold was chosen.

**Reference implementation.** `frontend/src/components/LocalizeModal.tsx` already does all three parts of this: `adviseParameters` (~line 99), the advice renderer (~line 457), and `ParameterGuide` (~line 579). Read it before starting. Match its voice — concrete, quantitative, willing to say a setting is wrong.

**The two facts the guidance exists to convey**, both verified in `backend/xcell/adaptor.py:_interp_smooth_subset` and `backend/xcell/multicontour.py`:

1. **Sigma is in grid pixels.** One pixel is `extent / grid_res` in data units, so the physical smoothing radius is `smooth_sigma × extent / grid_res`. Raising the grid resolution alone *shrinks the actual smoothing by the same factor*. `suggest_smooth_sigma` solves for a radius of ~2 spot spacings; edit the grid and that silently no longer holds.
2. **Bands are equal-width, not equal-count.** `thresholds = linspace(0, vmax, levels + 2)[1:-1]` over the smoothed field. On a skewed field — the normal case — the top band holds few spots, and the natural reading ("my gene set barely expresses") is wrong.

---

### Task 1: Backend supplies the dataset facts

`suggest_smooth_sigma` already computes the median nearest-neighbour spacing and the extent, then throws both away. Return them.

**Files:**
- Modify: `backend/xcell/multicontour.py` — `suggest_smooth_sigma` (~line 169)
- Modify: `backend/xcell/adaptor.py` — `suggest_contour_params` (~line 7509)
- Test: `backend/tests/test_contour_suggest.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `multicontour.spot_geometry(coords) -> tuple[float, float]` returning `(median_spacing, extent)`
  - `suggest_contour_params()` returns `{grid_res, smooth_sigma, n_spots, median_spacing, extent}`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_contour_suggest.py`:

```python
"""The data-aware facts the Contour modal reasons from.

Sigma is measured in grid pixels, so the modal cannot tell whether a setting is
sane without knowing the spot spacing and the tissue extent. These are the
numbers that make that possible.
"""

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.multicontour import spot_geometry


def _adata(step=2.0, n_side=10):
    """A regular grid, so the nearest-neighbour spacing is exactly `step`."""
    coords = np.array([[i * step, j * step]
                       for i in range(n_side) for j in range(n_side)])
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, (n, 4)).astype(np.float32)))
    ad.var_names = ["a", "b", "c", "d"]
    ad.obsm["spatial"] = coords
    return ad


def test_spot_geometry_on_a_regular_grid():
    coords = _adata(step=2.0, n_side=10).obsm["spatial"]
    spacing, extent = spot_geometry(coords)
    assert spacing == pytest.approx(2.0)
    # 10 points at stride 2 spans 0..18 on both axes.
    assert extent == pytest.approx(18.0)


def test_spot_geometry_survives_a_degenerate_input():
    """One spot has no spacing and no extent. None means 'not computable' —
    NaN would not survive the JSON round trip."""
    spacing, extent = spot_geometry(np.array([[1.0, 1.0]]))
    assert spacing is None
    assert extent is None


def test_suggest_contour_params_reports_the_geometry():
    a = DataAdaptor("x.h5ad", adata=_adata(step=2.0, n_side=10))
    out = a.suggest_contour_params()

    assert out["n_spots"] == 100
    assert out["median_spacing"] == pytest.approx(2.0)
    assert out["extent"] == pytest.approx(18.0)
    # Unchanged, so existing callers keep working.
    assert out["grid_res"] == 10          # round(sqrt(100)), clamped to [50, 600] -> 50
    assert out["smooth_sigma"] > 0


def test_suggest_contour_params_is_json_safe():
    import json
    a = DataAdaptor("x.h5ad", adata=_adata())
    json.dumps(a.suggest_contour_params())   # raises on NaN/inf
```

Note on `grid_res`: `suggest_grid_res` clamps to `[50, 600]`, so 100 spots gives **50**, not 10. Run the test, read the real number, and fix the assertion to match the implementation — do not change the clamp.

- [ ] **Step 2: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_contour_suggest.py -q
```

Expected: `ImportError: cannot import name 'spot_geometry'`.

- [ ] **Step 3: Extract the geometry**

In `backend/xcell/multicontour.py`, add above `suggest_smooth_sigma`:

```python
def spot_geometry(coords):
    """Median nearest-neighbour spacing and the larger axis span.

    These are the units the contour parameters actually live in: one grid pixel
    is ``extent / grid_res``, and sigma is measured in those pixels — so without
    both numbers there is no way to say whether a smoothing setting reaches the
    next spot or spans a whole zone.

    Returns (None, None) when there is nothing to measure; NaN would not
    survive the JSON round trip to the UI.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None, None
    d, _ = cKDTree(coords).query(coords, k=2)
    median_spacing = float(np.median(d[:, 1]))
    extent = float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])))
    if not np.isfinite(median_spacing) or not np.isfinite(extent):
        return None, None
    return median_spacing, extent
```

Then rewrite `suggest_smooth_sigma` to use it, so the two cannot disagree:

```python
def suggest_smooth_sigma(coords, grid_res):
    """~2 grid-pixels at the median spot spacing; clamped to [1, 6]."""
    median_spacing, extent = spot_geometry(coords)
    if median_spacing is None:
        return 2.0
    px = (extent or 1.0) / grid_res
    sigma = 2.0 * (median_spacing / px) if px > 0 else 2.0
    return float(min(6.0, max(1.0, sigma)))
```

- [ ] **Step 4: Return them from the adaptor**

In `suggest_contour_params`, replace the body's tail:

```python
        grid_res = mc.suggest_grid_res(self.adata.n_obs)
        coords = self.adata.obsm[spatial_key]
        smooth_sigma = mc.suggest_smooth_sigma(coords, grid_res)
        median_spacing, extent = mc.spot_geometry(coords)
        return {
            'grid_res': grid_res,
            'smooth_sigma': round(float(smooth_sigma), 2),
            'n_spots': int(self.adata.n_obs),
            # None rather than NaN: the UI drops the advice that needs these
            # rather than rendering a broken comparison.
            'median_spacing': round(median_spacing, 4) if median_spacing else None,
            'extent': round(extent, 4) if extent else None,
        }
```

Update the docstring to list the new keys.

- [ ] **Step 5: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_contour_suggest.py -q
cd backend && pixi run -e dev pytest -q
```

Expected: new file passes; no regressions (the existing multicontour tests exercise `suggest_smooth_sigma`).

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/multicontour.py backend/xcell/adaptor.py backend/tests/test_contour_suggest.py
git commit -m "feat(contour): report spot spacing and extent alongside the suggestions"
```

---

### Task 2: Advice for the select phase

**Files:**
- Create: `frontend/src/lib/contourAdvice.ts`
- Create: `frontend/src/lib/contourAdvice.test.ts`

**Interfaces:**
- Consumes: the payload from Task 1.
- Produces:

```ts
export interface Advice { level: 'warn' | 'info'; text: string }

export interface ContourGeometry {
  nSpots: number
  medianSpacing: number | null
  extent: number | null
  suggestedGridRes: number
  suggestedSigma: number
}

export interface ContourSettings {
  gridRes: number
  smoothSigma: number
  contourLevels: number
  logTransform: boolean
  sectionCol: string
  sectionCandidates: string[]   // categorical obs columns with 2..20 levels
  nSources: number
  smallestSourceSize: number | null
  hasPca: boolean
  matrixScale: string | null    // layer_scale verdict for .X
  matrixMax: number | null
}

export function adviseContour(s: ContourSettings, g: ContourGeometry | null): Advice[]
```

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/lib/contourAdvice.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { adviseContour, type ContourSettings, type ContourGeometry } from './contourAdvice'

// A regular grid: 400 spots, spacing 2, extent 38. suggest_grid_res clamps to
// 50, and the suggested sigma puts the smoothing radius at ~2 spacings.
const GEOM: ContourGeometry = {
  nSpots: 400, medianSpacing: 2, extent: 38,
  suggestedGridRes: 50, suggestedSigma: 2.63,
}

const OK: ContourSettings = {
  gridRes: 50, smoothSigma: 2.63, contourLevels: 3, logTransform: false,
  sectionCol: '', sectionCandidates: [], nSources: 2, smallestSourceSize: 12,
  hasPca: true, matrixScale: 'log_normalized', matrixMax: 6.2,
}

const texts = (a: { text: string }[]) => a.map((x) => x.text).join(' | ')

describe('adviseContour', () => {
  it('says nothing when the settings match the data', () => {
    expect(adviseContour(OK, GEOM)).toEqual([])
  })

  it('returns nothing at all when the geometry is unknown', () => {
    // A dataset with one spot has no spacing to compare against; silence beats
    // a comparison against null.
    expect(adviseContour(OK, null)).toEqual([])
  })

  it('warns when the grid is finer than the spot spacing', () => {
    // px = 38/600 = 0.063 data units, i.e. 0.03 spacings -- structure between
    // spots is interpolated, not measured.
    const a = adviseContour({ ...OK, gridRes: 600 }, GEOM)
    expect(texts(a)).toMatch(/finer than/i)
    expect(a.some((x) => x.level === 'warn')).toBe(true)
  })

  it('warns when the grid is coarser than the tissue', () => {
    const a = adviseContour({ ...OK, gridRes: 5 }, GEOM)
    expect(texts(a)).toMatch(/coarser|larger than/i)
  })

  it('warns when the smoothing does not reach the next spot', () => {
    // radius = 0.3 * (38/50) = 0.23 data units = 0.11 spacings.
    const a = adviseContour({ ...OK, smoothSigma: 0.3 }, GEOM)
    expect(texts(a)).toMatch(/speckle/i)
  })

  it('warns when the smoothing spans a whole zone', () => {
    // radius = 30 * 0.76 = 22.8 data units = 11 spacings.
    const a = adviseContour({ ...OK, smoothSigma: 30 }, GEOM)
    expect(texts(a)).toMatch(/merge/i)
  })

  it('catches the grid-times-sigma coupling', () => {
    // Raising the grid alone shrinks the physical smoothing by the same
    // factor. This is the interaction nothing on the form reveals.
    const a = adviseContour({ ...OK, gridRes: 200 }, GEOM)
    expect(texts(a)).toMatch(/smoothing/i)
    expect(texts(a)).toMatch(/spot spacing/i)
  })

  it('warns when log is off and the matrix looks like raw counts', () => {
    const a = adviseContour(
      { ...OK, logTransform: false, matrixScale: 'raw_counts', matrixMax: 8134 },
      GEOM,
    )
    expect(texts(a)).toMatch(/raw counts/i)
    expect(texts(a)).toMatch(/8134|8,134/)
  })

  it('warns when log is on and the matrix is already log-normalized', () => {
    const a = adviseContour({ ...OK, logTransform: true }, GEOM)
    expect(texts(a)).toMatch(/second log|already/i)
  })

  it('warns when log is on and the matrix is z-scored', () => {
    const a = adviseContour(
      { ...OK, logTransform: true, matrixScale: 'z_scored', matrixMax: 4.1 },
      GEOM,
    )
    expect(texts(a)).toMatch(/negative/i)
  })

  it('says nothing about log when the scale is unknown', () => {
    const a = adviseContour({ ...OK, matrixScale: null, matrixMax: null }, GEOM)
    expect(texts(a)).not.toMatch(/log/i)
  })

  it('mentions an unused section column', () => {
    const a = adviseContour({ ...OK, sectionCandidates: ['section'] }, GEOM)
    expect(texts(a)).toMatch(/section/i)
    expect(texts(a)).toMatch(/bleed|across/i)
  })

  it('stays quiet once a section column is chosen', () => {
    const a = adviseContour(
      { ...OK, sectionCol: 'section', sectionCandidates: ['section'] }, GEOM,
    )
    expect(texts(a)).not.toMatch(/bleed/i)
  })

  it('warns about multi-set contouring without PCA before the run fails', () => {
    const a = adviseContour({ ...OK, nSources: 3, hasPca: false }, GEOM)
    expect(texts(a)).toMatch(/PCA/)
  })

  it('does not mention PCA for a single-set contour', () => {
    const a = adviseContour({ ...OK, nSources: 1, hasPca: false }, GEOM)
    expect(texts(a)).not.toMatch(/PCA/)
  })

  it('notes a source with almost no genes', () => {
    const a = adviseContour({ ...OK, smallestSourceSize: 1 }, GEOM)
    expect(texts(a)).toMatch(/one gene|single gene/i)
  })

  it('explains that more levels only subdivide the cutoff', () => {
    const a = adviseContour({ ...OK, contourLevels: 8 }, GEOM)
    expect(texts(a)).toMatch(/cutoff/i)
  })

  it('puts warnings before infos', () => {
    const a = adviseContour(
      { ...OK, gridRes: 600, contourLevels: 8, sectionCandidates: ['section'] },
      GEOM,
    )
    const firstInfo = a.findIndex((x) => x.level === 'info')
    const lastWarn = a.map((x) => x.level).lastIndexOf('warn')
    expect(firstInfo === -1 || lastWarn < firstInfo).toBe(true)
  })
})
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd frontend && npm test -- contourAdvice
```

Expected: `Cannot find module './contourAdvice'`.

- [ ] **Step 3: Implement**

Create `frontend/src/lib/contourAdvice.ts`:

```ts
/**
 * What is wrong with the current contour settings, given this dataset.
 *
 * Contour's parameters are not independent and the coupling is invisible on the
 * form: sigma is measured in *grid pixels*, so one pixel is `extent / gridRes`
 * in data units and the physical smoothing radius is `sigma × extent / gridRes`.
 * Raising the grid alone shrinks the actual smoothing by the same factor. Every
 * rule below is therefore expressed in **spot spacings** — a unit the user can
 * see in the plot — rather than as a disagreement with a suggested number they
 * never chose.
 */

export interface Advice { level: 'warn' | 'info'; text: string }

export interface ContourGeometry {
  nSpots: number
  medianSpacing: number | null
  extent: number | null
  suggestedGridRes: number
  suggestedSigma: number
}

export interface ContourSettings {
  gridRes: number
  smoothSigma: number
  contourLevels: number
  logTransform: boolean
  sectionCol: string
  sectionCandidates: string[]
  nSources: number
  smallestSourceSize: number | null
  hasPca: boolean
  matrixScale: string | null
  matrixMax: number | null
}

const round1 = (v: number) => Math.round(v * 10) / 10

export function adviseContour(
  s: ContourSettings,
  g: ContourGeometry | null,
): Advice[] {
  const warns: Advice[] = []
  const infos: Advice[] = []

  // Without spacing and extent there is nothing to compare against, and a
  // comparison against a missing number is worse than silence.
  const geomKnown =
    g != null && g.medianSpacing != null && g.extent != null &&
    g.medianSpacing > 0 && g.extent > 0 && s.gridRes > 0

  if (geomKnown) {
    const px = g!.extent! / s.gridRes            // data units per grid pixel
    const pxPerSpacing = px / g!.medianSpacing!
    const radius = s.smoothSigma * px            // data units
    const radiusInSpacings = radius / g!.medianSpacing!

    // Below half a spacing per pixel, cubic griddata is producing detail at a
    // scale finer than the sampling — the structure is interpolated, not
    // measured.
    if (pxPerSpacing < 0.5) {
      warns.push({ level: 'warn', text:
        `A grid of ${s.gridRes} over ${g!.nSpots.toLocaleString()} spots puts ` +
        `${round1(1 / pxPerSpacing)} pixels between neighbouring spots — finer than ` +
        `the data samples. The detail between spots is interpolated, not measured. ` +
        `About ${g!.suggestedGridRes} matches the spacing.` })
    }
    // Above ~3 spacings per pixel a zone a few spots across falls inside one
    // pixel and cannot appear at all.
    else if (pxPerSpacing > 3) {
      warns.push({ level: 'warn', text:
        `Each grid pixel spans ${round1(pxPerSpacing)} spot spacings, so a zone ` +
        `smaller than about ${Math.ceil(pxPerSpacing)} spots across cannot appear. ` +
        `About ${g!.suggestedGridRes} matches the spacing.` })
    }

    // The Gaussian has to reach the neighbouring spot to average anything.
    if (radiusInSpacings < 1) {
      warns.push({ level: 'warn', text:
        `Smoothing reaches ${round1(radiusInSpacings)} spot spacings ` +
        `(σ ${s.smoothSigma} × ${round1(px)} per pixel), so it does not reach the ` +
        `next spot. The bands will follow per-spot noise rather than forming zones.` })
    } else if (radiusInSpacings > 6) {
      warns.push({ level: 'warn', text:
        `Smoothing reaches ${round1(radiusInSpacings)} spot spacings, wide enough to ` +
        `span a whole zone — adjacent tissues will merge into one band.` })
    }
    // The interaction itself, only when neither radius rule already said it
    // more directly: sigma is in pixels, so editing the grid rescaled it.
    else if (s.gridRes !== g!.suggestedGridRes &&
             Math.abs(s.smoothSigma - g!.suggestedSigma) < 0.01) {
      infos.push({ level: 'info', text:
        `σ is measured in grid pixels, so changing the grid to ${s.gridRes} moved the ` +
        `smoothing to ${round1(radiusInSpacings)} spot spacings without you editing σ. ` +
        `σ ${round1(2 * g!.medianSpacing! / px)} would restore the original width.` })
    }
  }

  // Scale of .X. Only speak when layer_scale actually reached a verdict.
  if (s.matrixScale === 'raw_counts' && !s.logTransform) {
    const max = s.matrixMax != null ? ` (max ${s.matrixMax.toLocaleString()})` : ''
    warns.push({ level: 'warn', text:
      `.X looks like raw counts${max} and log transform is off. Per-gene clipping is ` +
      `at the 1st/99th percentile, so a handful of high spots still set the range and ` +
      `the field collapses toward them.` })
  }
  if (s.logTransform &&
      (s.matrixScale === 'log_normalized' || s.matrixScale === 'log_transformed')) {
    warns.push({ level: 'warn', text:
      `.X is already on a log scale, so this applies a second log. That flattens the ` +
      `field and pulls every band toward the middle.` })
  }
  if (s.logTransform && s.matrixScale === 'z_scored') {
    warns.push({ level: 'warn', text:
      `.X is centred and scaled, so it holds negative values — log1p of those is not ` +
      `a number the score can use.` })
  }

  if (!s.sectionCol && s.sectionCandidates.length > 0) {
    infos.push({ level: 'info', text:
      `No section column set, but ${s.sectionCandidates.slice(0, 3).join(', ')} ` +
      `${s.sectionCandidates.length === 1 ? 'looks like one' : 'look like candidates'}. ` +
      `Interpolation spans the gap between sections, so expression bleeds across it.` })
  }

  if (s.nSources >= 2 && !s.hasPca) {
    warns.push({ level: 'warn', text:
      `Fusing ${s.nSources} modules needs X_pca to resolve spots that are high in more ` +
      `than one. Run PCA first, or contour a single set.` })
  }

  if (s.smallestSourceSize != null && s.smallestSourceSize <= 2) {
    infos.push({ level: 'info', text:
      `A gene set of ${s.smallestSourceSize} makes the module score essentially ` +
      `${s.smallestSourceSize === 1 ? 'one gene' : 'two genes'}' spatial field, ` +
      `dropout included.` })
  }

  if (s.contourLevels >= 6) {
    infos.push({ level: 'info', text:
      `${s.contourLevels} levels only subdivide the choice of cutoff — the underlying ` +
      `field is unchanged, and thresholds are equally spaced in value, not in spot ` +
      `count, so the upper bands stay small.` })
  }

  return [...warns, ...infos]
}
```

- [ ] **Step 4: Run the tests**

```bash
cd frontend && npm test -- contourAdvice
```

Expected: all pass. If a threshold test fails, recompute the arithmetic in the test comment before touching the implementation — the numbers there are the specification.

- [ ] **Step 5: Typecheck and commit**

```bash
cd frontend && npx tsc --noEmit
git add frontend/src/lib/contourAdvice.ts frontend/src/lib/contourAdvice.test.ts
git commit -m "feat(contour): live advice for the select-phase parameters"
```

---

### Task 3: Advice for the review phase

**Files:**
- Modify: `frontend/src/lib/contourAdvice.ts`
- Modify: `frontend/src/lib/contourAdvice.test.ts`

**Interfaces:**
- Consumes: `Advice` from Task 2.
- Produces:

```ts
export interface ModuleHighs { name: string; highCount: number }
export function adviseModules(
  modules: ModuleHighs[], nSpots: number, profileK: number,
): Advice[]
```

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/lib/contourAdvice.test.ts`:

```ts
import { adviseModules } from './contourAdvice'

describe('adviseModules', () => {
  const N = 400

  it('says nothing when each module marks a distinct minority', () => {
    expect(adviseModules(
      [{ name: 'A', highCount: 80 }, { name: 'B', highCount: 60 }], N, 15,
    )).toEqual([])
  })

  it('warns when a module has no high spots', () => {
    const a = adviseModules(
      [{ name: 'A', highCount: 0 }, { name: 'B', highCount: 60 }], N, 15,
    )
    expect(a.map((x) => x.text).join(' ')).toMatch(/A/)
    expect(a.map((x) => x.text).join(' ')).toMatch(/no spots|nothing/i)
  })

  it('warns when a module is high nearly everywhere', () => {
    const a = adviseModules(
      [{ name: 'A', highCount: 300 }, { name: 'B', highCount: 60 }], N, 15,
    )
    expect(a.map((x) => x.text).join(' ')).toMatch(/75%|most/i)
  })

  it('flags a large overlap between two broad modules', () => {
    const a = adviseModules(
      [{ name: 'A', highCount: 200 }, { name: 'B', highCount: 180 }], N, 15,
    )
    expect(a.map((x) => x.text).join(' ')).toMatch(/overlap|neighbour|neighbor/i)
  })

  it('warns when profile k reaches past the local neighbourhood', () => {
    const a = adviseModules([{ name: 'A', highCount: 80 }], N, 80)
    expect(a.map((x) => x.text).join(' ')).toMatch(/80/)
  })

  it('is silent with no modules', () => {
    expect(adviseModules([], N, 15)).toEqual([])
  })
})
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd frontend && npm test -- contourAdvice
```

Expected: `adviseModules is not a function`.

- [ ] **Step 3: Implement**

Append to `frontend/src/lib/contourAdvice.ts`:

```ts
export interface ModuleHighs { name: string; highCount: number }

/**
 * Whether the chosen cutoffs can actually produce a tissue map.
 *
 * A module high nowhere contributes nothing; a module high everywhere wins
 * conflicts by size rather than by signal, which is the quiet way a fused
 * annotation turns into one label with decoration.
 */
export function adviseModules(
  modules: ModuleHighs[], nSpots: number, profileK: number,
): Advice[] {
  const out: Advice[] = []
  if (modules.length === 0 || nSpots <= 0) return out

  const frac = (m: ModuleHighs) => m.highCount / nSpots

  for (const m of modules) {
    if (m.highCount === 0) {
      out.push({ level: 'warn', text:
        `${m.name} has no spots above its cutoff, so it cannot contribute a tissue. ` +
        `Lower the cutoff, or drop the set.` })
    } else if (frac(m) > 0.5) {
      out.push({ level: 'warn', text:
        `${m.name} is high in ${Math.round(frac(m) * 100)}% of spots. A module high ` +
        `nearly everywhere wins overlaps by size rather than by signal.` })
    }
  }

  // Two broad modules mean most spots are ambiguous, so the annotation is
  // mostly the neighbour vote rather than the contours the user reviewed.
  const broad = modules.filter((m) => frac(m) > 0.4).map((m) => m.name)
  if (broad.length >= 2) {
    out.push({ level: 'info', text:
      `${broad.join(' and ')} are each high in over 40% of spots, so most assignments ` +
      `will come from the neighbour vote rather than from the contours themselves.` })
  }

  if (profileK > 50) {
    out.push({ level: 'info', text:
      `Profile k of ${profileK} votes over a wide neighbourhood, which pulls ambiguous ` +
      `spots toward whatever is regionally dominant rather than locally adjacent.` })
  }

  return out
}
```

- [ ] **Step 4: Run the tests, typecheck, commit**

```bash
cd frontend && npm test -- contourAdvice && npx tsc --noEmit
git add frontend/src/lib/contourAdvice.ts frontend/src/lib/contourAdvice.test.ts
git commit -m "feat(contour): live advice for the review-phase cutoffs"
```

---

### Task 4: The collapsible guide

**Files:**
- Modify: `frontend/src/components/MultiContourModal.tsx` — add `<ContourGuide />` beside the existing `Param` / `BandBars` helpers (~line 457), and extend `styles`.

**Interfaces:**
- Consumes: nothing.
- Produces: `function ContourGuide(): JSX.Element` (module-local, not exported).

- [ ] **Step 1: Read the reference**

Open `frontend/src/components/LocalizeModal.tsx` lines 573–655 (`ParameterGuide`). Match its structure — a *Starting points* table, a *Trade-offs* list, an *Avoid* list — and its voice.

- [ ] **Step 2: Write the component**

Add to `MultiContourModal.tsx`, above the `styles` object:

```tsx
/**
 * What to pick and what each choice costs — not a restatement of the labels.
 *
 * Leads with the grid×σ coupling because it is the one thing the form actively
 * hides: σ is in grid pixels, so raising the grid shrinks the real smoothing.
 */
function ContourGuide() {
  return (
    <div style={styles.guide}>
      <div style={styles.guideHead}>Starting points</div>
      <table style={styles.guideTable}>
        <thead>
          <tr style={{ color: '#888', textAlign: 'left' }}>
            <th style={styles.th}>Data</th><th style={styles.th}>Use</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={styles.td}>Visium <span style={{ color: '#888' }}>(55 µm spots, 2–5k)</span></td>
            <td style={styles.td}>The prefilled grid and σ. 3 levels, cutoff at the top band.</td>
          </tr>
          <tr>
            <td style={styles.td}>Visium HD, or binned to a fine grid</td>
            <td style={styles.td}>Prefilled grid; raise σ with it so the smoothing still reaches 2 spot spacings.</td>
          </tr>
          <tr>
            <td style={styles.td}>Xenium / single-cell resolution</td>
            <td style={styles.td}>Spacing is irregular — check the reported radius rather than trusting σ.</td>
          </tr>
          <tr>
            <td style={styles.td}>Several sections on one slide</td>
            <td style={styles.td}>Set the section column. Without it the grid spans the gap and expression bleeds across.</td>
          </tr>
        </tbody>
      </table>

      <div style={styles.guideHead}>Trade-offs</div>
      <ul style={styles.guideList}>
        <li>
          <b>Grid and σ are one setting, not two.</b> σ is measured in grid
          pixels, and a pixel is <code>extent / grid</code>. Doubling the grid
          halves the real smoothing radius. If a map looks blurry, raising the
          grid appears to sharpen it — but it sharpened by removing smoothing,
          which is not the same as resolving structure.
        </li>
        <li>
          <b>Bands are equal-width, not equal-count.</b> Thresholds are spaced
          evenly between 0 and the field's maximum. A module high in one corner
          gives a strongly skewed field, so the top band holds few spots — that
          is the shape of the field, not weak expression.
        </li>
        <li>
          <b>Levels only subdivide the cutoff.</b> More levels give finer
          control over where "high" starts. They do not change the interpolated
          field and cannot reveal structure the grid and σ did not preserve.
        </li>
        <li>
          <b>Log transform and the percentile clip overlap.</b> Each gene is
          already clipped to its 1st–99th percentile before averaging, so the
          extreme tail is handled. Log is for when the <i>bulk</i> of the
          distribution is multiplicative — raw counts — not for one hot spot.
        </li>
        <li>
          <b>The cutoff is where the tissue call happens.</b> Everything before
          it is a continuous field; the cutoff turns it into a claim. The spot
          count next to each choice is the honest signal, not the picture.
        </li>
      </ul>

      <div style={styles.guideHead}>Avoid</div>
      <ul style={styles.guideList}>
        <li>Raising the grid to sharpen a blurry map — that removes smoothing rather than adding resolution.</li>
        <li>Contouring several sections without a section column.</li>
        <li>Comparing band thresholds between runs. <code>vmax</code> is per-run, so the same number means different things.</li>
        <li>Choosing cutoffs by which map looks cleanest. A clean map is what this produces when a module is high nearly everywhere.</li>
      </ul>
    </div>
  )
}
```

- [ ] **Step 3: Add the styles**

Append to the `styles` object, matching the existing palette:

```ts
  guide: { marginTop: 8, padding: 10, background: '#0f1625', borderRadius: 4, fontSize: 11, lineHeight: 1.5, color: '#bbb' },
  guideHead: { color: '#4ecdc4', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, margin: '10px 0 6px' },
  guideTable: { borderCollapse: 'collapse', width: '100%', marginBottom: 4 },
  th: { padding: '2px 6px 4px 0', fontWeight: 500, borderBottom: '1px solid #0f3460' },
  td: { padding: '4px 6px 4px 0', verticalAlign: 'top', borderBottom: '1px solid rgba(15,52,96,0.5)' },
  guideList: { margin: 0, paddingLeft: 16, display: 'grid', gap: 5 },
  ghost: { background: 'none', border: '1px solid #0f3460', color: '#aaa', borderRadius: 4, padding: '4px 10px', fontSize: 11, cursor: 'pointer' },
  advice: { display: 'grid', gap: 5, marginBottom: 12 },
```

`styles` is typed `Record<string, React.CSSProperties>`; `textTransform` needs the literal type, so write `textTransform: 'uppercase' as const` if tsc objects.

- [ ] **Step 4: Typecheck and commit**

```bash
cd frontend && npx tsc --noEmit
git add frontend/src/components/MultiContourModal.tsx
git commit -m "feat(contour): a guide to choosing the parameters"
```

---

### Task 5: Wire it into the modal

**Files:**
- Modify: `frontend/src/components/MultiContourModal.tsx` — state and fetches (~lines 110–147), select-phase render (~line 265), review-phase render (~line 331), `Param` tips (~lines 288–314, 369–378)

**Interfaces:**
- Consumes: `adviseContour`, `adviseModules`, `ContourGeometry`, `Advice` (Tasks 2–3); `ContourGuide` (Task 4); `suggest_contour_params`'s new fields (Task 1).
- Produces: no exported names.

- [ ] **Step 1: Collect the inputs**

The modal already fetches `/api/obs/summaries` and `/api/scanpy/contour_suggest`. Extend both, and add one fetch for the matrix scale.

Keep the whole suggestion payload rather than only the two prefilled numbers:

```tsx
const [geom, setGeom] = useState<ContourGeometry | null>(null)
const [matrixScale, setMatrixScale] = useState<string | null>(null)
const [matrixMax, setMatrixMax] = useState<number | null>(null)
const [showGuide, setShowGuide] = useState(false)
const [sectionCandidates, setSectionCandidates] = useState<string[]>([])
const hasPca = !!useStore((s) => s.schema)?.embeddings?.includes('X_pca')
```

In the existing `contour_suggest` effect, after the two `setX` calls:

```tsx
        setGeom({
          nSpots: d.n_spots, medianSpacing: d.median_spacing, extent: d.extent,
          suggestedGridRes: d.grid_res, suggestedSigma: d.smooth_sigma,
        })
```

In the existing `/obs/summaries` effect, alongside `setObsColumns` — a section column has a handful of levels, not hundreds:

```tsx
        setSectionCandidates(
          (d.summaries || d || [])
            .filter((s: { dtype?: string; categories?: unknown[] }) =>
              s.dtype === 'category' &&
              (s.categories?.length ?? 0) >= 2 && (s.categories?.length ?? 0) <= 20)
            .map((s: { name: string }) => s.name),
        )
```

Add a new effect for the matrix scale. `/api/scanpy/layers` returns the `X` entry first, carrying a `scale` block from `layer_scale.py`:

```tsx
  // Whether .X holds raw counts is the one thing the log-transform checkbox
  // cannot be answered without, and datasets rarely say.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return
        const x = (d.layers || []).find((l: { name: string }) => l.name === 'X')
        setMatrixScale(x?.scale?.verdict ?? null)
        setMatrixMax(x?.scale?.stats?.max ?? null)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen])
```

- [ ] **Step 2: Compute the advice**

Above the `return`, after `const nSources = sources.length`:

```tsx
  const advice = useMemo(() => adviseContour({
    gridRes: parseInt(gridRes || '0', 10) || 0,
    smoothSigma: parseFloat(smoothSigma || '0') || 0,
    contourLevels, logTransform, sectionCol, sectionCandidates, nSources,
    smallestSourceSize: sources.length
      ? Math.min(...sources.map((s) => s.genes.length)) : null,
    hasPca, matrixScale, matrixMax,
  }, geom), [gridRes, smoothSigma, contourLevels, logTransform, sectionCol,
             sectionCandidates, nSources, sources, hasPca, matrixScale,
             matrixMax, geom])

  const moduleAdvice = useMemo(() => (prep
    ? adviseModules(
        prep.modules.map((m) => ({
          name: m.name, highCount: highCount(m, cutoffs[m.name] ?? m.auto_cutoff),
        })),
        geom?.nSpots ?? 0, profileK)
    : []), [prep, cutoffs, profileK, geom])
```

`highCount` is defined above the `return` already; move it above these `useMemo`s if the ordering complains.

- [ ] **Step 3: Render**

Add a small local renderer beside `Param`:

```tsx
function AdviceList({ items }: { items: Advice[] }) {
  if (items.length === 0) return null
  return (
    <div style={styles.advice}>
      {items.map((a, i) => (
        <div key={i} style={{
          fontSize: 11, lineHeight: 1.45, padding: '4px 8px', borderRadius: 2,
          color: a.level === 'warn' ? '#f0c987' : '#aaa',
          background: a.level === 'warn' ? 'rgba(233,162,59,0.12)' : 'transparent',
          borderLeft: `2px solid ${a.level === 'warn' ? '#e9a23b' : '#0f3460'}`,
        }}>
          {a.level === 'warn' ? '⚠ ' : ''}{a.text}
        </div>
      ))}
    </div>
  )
}
```

In the **select** phase, directly after the closing `</div>` of `styles.paramGrid` and before `styles.actions`:

```tsx
            <AdviceList items={advice} />
            <button style={styles.ghost} onClick={() => setShowGuide((v) => !v)}>
              {showGuide ? 'Hide' : 'Show'} how to choose these
            </button>
            {showGuide && <ContourGuide />}
```

In the **review** phase, after its `styles.paramGrid` block:

```tsx
            <AdviceList items={moduleAdvice} />
```

- [ ] **Step 4: Rewrite the tooltips**

Replace the `tip` strings so each states a cost rather than restating the label:

```tsx
<Param label="Contour levels" tip="How finely the score range is divided. Thresholds are equally spaced in value, not in spot count, so the top bands stay small on a skewed field. More levels give finer control over the cutoff; they do not change the field.">
<Param label="Grid resolution" tip="Interpolation grid per axis. Sigma is measured in these pixels, so changing this rescales the smoothing by the same factor. Prefilled to roughly one pixel per spot.">
<Param label="Smoothing sigma" tip="Gaussian width in grid pixels. What matters is the radius it works out to in spot spacings — under 1 it cannot reach the next spot and the bands speckle; over about 6 it spans a zone and tissues merge.">
<Param label="Log transform" tip="log1p before per-gene clipping. Each gene is already clipped to its 1st–99th percentile, so this is for a bulk multiplicative scale — raw counts — not for one hot spot. Applying it to an already-logged matrix flattens the field.">
<Param label="Section column" tip="Contours each section on its own grid. Without it the interpolation spans the gap between sections and expression bleeds across. Use Define Sections to create one.">
<Param label="Profile k" tip="Nearest unambiguous spatial neighbours, ranked in PCA space, that vote on a spot high in more than one module. Large k reaches past the local neighbourhood and votes for whatever is regionally dominant.">
```

- [ ] **Step 5: Typecheck**

```bash
cd frontend && npx tsc --noEmit && npm test
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/MultiContourModal.tsx
git commit -m "feat(contour): show the guidance in the modal"
```

---

### Task 6: Browser verification and docs

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Start both servers**

```bash
pixi run backend
pixi run dev
```

Stop them with `pkill -f "uvicorn xcell.main:app"; pkill -f "vite"` — `lsof -ti :8000 | xargs kill` leaves a reloader that re-binds and serves stale data.

- [ ] **Step 2: Check each live rule actually fires**

Load the bundled toy spatial dataset, open Analyze → Spatial → Contour, pick one gene set, and drive the form:

1. Note the prefilled grid and σ — **no advice should show**.
2. Set grid to 600. The over-fine warning appears, quoting pixels-per-spacing.
3. Return grid to the suggestion, set σ to 0.2. The speckle warning appears.
4. Set σ to 30. The merge warning appears.
5. Return σ, set grid to 200 without touching σ. The coupling info appears, naming the new radius.
6. Toggle log transform. Whether a warning appears depends on the toy dataset's `.X` — confirm the text matches the verdict at `curl -s localhost:8000/api/scanpy/layers | python3 -m json.tool | grep -A3 verdict`.
7. If the toy has a `section` column, confirm the bleed info appears and disappears when you choose it.

Any rule whose text is wrong for what the data actually is: fix the rule, not the test.

- [ ] **Step 3: Check the review phase**

Pick two gene sets, Compute, then in the review phase drag one module's cutoff to the lowest band. The ">50% of spots" warning should appear, and the spot counts beside the selector should agree with it.

- [ ] **Step 4: Check the guide renders**

Expand "Show how to choose these". Confirm the table and lists are readable against the panel background and that the modal still scrolls — it is `maxHeight: '85vh', overflowY: 'auto'`, and the guide is tall.

- [ ] **Step 5: Confirm nothing regressed**

Run one single-set contour and one two-set contour end to end. Both must still produce their `.obs` column and colour the plot.

- [ ] **Step 6: Stop the servers, update the docs**

`README.md` — note that the Contour modal explains its parameters and warns when they do not match the data.

`CHANGELOG.md` — an entry covering the live advice, the guide, and the two facts it exists to convey (σ measured in grid pixels; equal-width bands).

- [ ] **Step 7: Commit and merge**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: contour parameter guidance"
git checkout main
git merge --no-ff feat/contour-guidance
```

---

## Self-Review

**Spec coverage.** `suggest_contour_params` extension → Task 1. Every select-phase rule in the spec's table → Task 2, one test each. Every review-phase rule → Task 3. `<ContourGuide />` with Starting points / Trade-offs / Avoid → Task 4. Tooltip rewrite and wiring → Task 5. Backend tests → Task 1; `adviseContour`/`adviseModules` unit tests → Tasks 2–3; browser verification → Task 6.

**One refinement over the spec:** the spec's "grid edited off the suggestion" rule is implemented as an `else if` after the two radius rules, so it cannot double up on a warning that already said the same thing. The spec anticipated this; the plan pins down the mechanism.

**Names used consistently:** `Advice`, `ContourGeometry`, `ContourSettings`, `ModuleHighs`, `adviseContour`, `adviseModules`, `ContourGuide`, `AdviceList` — each defined once and spelled the same everywhere. Backend: `spot_geometry`, and the payload keys `n_spots` / `median_spacing` / `extent` map to `nSpots` / `medianSpacing` / `extent` at the fetch site in Task 5 Step 1.

**Known soft spots**, each with an in-plan instruction: the real clamped `grid_res` for the test fixture (Task 1 Step 1), `textTransform`'s literal type (Task 4 Step 3), `highCount`'s declaration order relative to the new `useMemo`s (Task 5 Step 2), and whether the toy dataset's `.X` verdict makes the log rules fire (Task 6 Step 2).
