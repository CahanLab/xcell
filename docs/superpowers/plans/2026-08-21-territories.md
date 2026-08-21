# Territories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user hand-draw named regions on spatial data, save them in the AnnData, annotate cells by which region they occupy, and carry those regions onto dissociated cells through Localize.

**Architecture:** Store the *dividing cuts*, derive the regions between them. A cut is one object referenced by both neighbours, so dragging it moves both — shared-edge editing falls out of the representation and overlaps/gaps become unrepresentable rather than merely detectable. Geometry lives in a pure backend module (`shapely`); the adaptor owns persistence to `uns['xcell_territories_json']`; the frontend draws cuts with the existing draw tools and renders derived faces as SVG overlay paths.

**Tech Stack:** Python 3.11, shapely 2.1, numpy, pandas, AnnData, FastAPI; React 18 + Zustand + inline styles, vitest.

**Spec:** `docs/superpowers/specs/2026-08-21-territories-design.md`

## Global Constraints

- Analysis modules are **pure**: arrays/dicts in, dicts out; never import the adaptor, never mutate live state (`CLAUDE.md`).
- Every data-touching route takes `dataset: str | None = Query(None)` as its **last** parameter and resolves via `get_adaptor(dataset)`.
- Error mapping: `ValueError → 400`, `KeyError → 404`, no data → 503. `except HTTPException: raise` before any catch-all.
- Anything crossing the API must be JSON-serializable: no `inf`, no `NaN`. Use `None` for "not computable".
- New `.obs` columns are tool-prefixed: `territory_<type>`.
- Anything that mutates calls `self._log_action(action, params, result)`.
- A feature creating a new `.obs` column needs **both** `refreshSchema()` and `refreshObsSummaries()` on the frontend.
- Frontend styling is inline. Palette: panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`/`#888`.
- Tests: `backend/tests/`, no `conftest.py`, module-level `_adata()` / `_adaptor()` factories, seeded RNG, `csr_matrix` + `float32`. Assert on the resulting AnnData state, not only the returned dict.
- Branch off `main`, merge back with `--no-ff`.

**Deviation from the spec, recorded deliberately:** the spec says a face is named by clicking inside it on the plot and that click becomes the anchor. Phase 2 names faces from the panel's face list instead, with the anchor auto-placed at the face's representative point — functionally equivalent for name persistence, and it avoids adding a new interaction mode to a 1,500-line plot component before the vertex-editing work in Phase 3 opens that file anyway. Plot-click naming lands in Phase 3, Task 15.

---

## File Structure

**Created:**
- `backend/xcell/territories.py` — pure geometry: extend, derive, name, assign.
- `backend/tests/test_territories.py` — pure geometry tests.
- `backend/tests/test_territories_adaptor.py` — persistence + assignment tests.
- `backend/tests/test_territories_routes.py` — route tests.
- `frontend/src/lib/territoryGeometry.ts` — pure helpers (ring from cells, face colors, SVG path building).
- `frontend/src/lib/territoryGeometry.test.ts`
- `frontend/src/components/TerritoryPanel.tsx` — the drawing panel.
- `frontend/src/components/AssignTerritoriesModal.tsx` — the assignment modal.

**Modified:**
- `backend/xcell/adaptor.py` — territory persistence, assignment, bundle export, import.
- `backend/xcell/api/routes.py` — territory routes; `LocalizeRequest` fields.
- `pixi.toml` — declare `shapely`.
- `frontend/src/store.ts` — territory state + cut-routing flag.
- `frontend/src/App.tsx` — route drawn cuts to territories; mount the panel/modal.
- `frontend/src/components/ScatterPlot.tsx` — render faces; (Phase 3) vertex editing.
- `frontend/src/components/ScanpyModal.tsx` — Spatial launchers.
- `CHANGELOG.md`.

---

## Task 1: Pure geometry — extending cuts to the ring

**Files:**
- Create: `backend/xcell/territories.py`
- Create: `backend/tests/test_territories.py`
- Modify: `pixi.toml` (declare shapely)

**Interfaces:**
- Consumes: nothing.
- Produces: `extend_cuts(ring: list[list[float]], cuts: list[dict]) -> list` returning shapely geometries clipped to the ring. A cut dict is `{"id": str, "points": list[list[float]], "closed": bool}`.

- [ ] **Step 1: Declare shapely explicitly**

It arrives today only as a squidpy transitive dependency. In `pixi.toml`, under `[dependencies]`, after the `squidpy` line:

```toml
# Territory geometry (xcell/territories.py): noding + polygonize. Present
# transitively via squidpy, declared here because we depend on it directly.
shapely = ">=2.0"
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_territories.py`:

```python
"""Territory geometry — cuts in, faces out.

The invariant every test here defends: faces derived from a ring plus its cuts
tile the ring exhaustively and disjointly. Overlaps and gaps are not errors to
be detected, they are states the representation cannot reach.
"""
import numpy as np
import pytest

from xcell import territories as terr


SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def _cut(points, closed=False, cid="c1"):
    return {"id": cid, "points": points, "closed": closed}


def test_a_cut_stopping_short_is_extended_to_the_ring():
    """Freehand cuts never land on the edge; a cut that divides nothing would
    be a silently missing boundary."""
    short = _cut([[2.0, 5.0], [8.0, 5.0]])
    (geom,) = terr.extend_cuts(SQUARE, [short])
    xs = [p[0] for p in geom.coords]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(10.0)


def test_overshoot_is_trimmed_at_the_ring():
    over = _cut([[-50.0, 5.0], [60.0, 5.0]])
    (geom,) = terr.extend_cuts(SQUARE, [over])
    xs = [p[0] for p in geom.coords]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(10.0)


def test_a_closed_cut_is_kept_as_a_loop_not_extended():
    loop = _cut([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]], closed=True)
    (geom,) = terr.extend_cuts(SQUARE, [loop])
    assert geom.is_closed


def test_a_cut_entirely_outside_the_ring_is_dropped():
    outside = _cut([[20.0, 20.0], [30.0, 20.0]])
    assert terr.extend_cuts(SQUARE, [outside]) == []


def test_a_degenerate_cut_of_one_point_is_dropped():
    assert terr.extend_cuts(SQUARE, [_cut([[5.0, 5.0]])]) == []
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xcell.territories'`

- [ ] **Step 4: Write the implementation**

Create `backend/xcell/territories.py`:

```python
"""Hand-drawn spatial territories: dividing cuts in, regions out.

A territory type stores the curves that *divide* a space, not the regions
between them. The regions are derived, so a boundary is one object shared by
both of its neighbours: move it and both follow. Overlaps and gaps are not
detected and repaired here because the representation cannot express them.

Pure: plain lists and arrays in, plain lists and arrays out. Never imports the
adaptor, never touches AnnData.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def extend_cuts(ring: list[list[float]], cuts: list[dict[str, Any]]) -> list[LineString]:
    """Project each open cut's ends outward, then clip every cut to the ring.

    A freehand cut rarely lands on the tissue edge, and a cut that stops short
    divides nothing. Each end is extended along its own terminal direction by
    more than the ring's diagonal and the result intersected with the ring, so
    the extension stops exactly at the boundary and overshoot is trimmed the
    same way. The caller's drawn points are never modified — extension is
    derived every time, so repeated editing cannot compound.
    """
    ring_poly = Polygon(ring)
    minx, miny, maxx, maxy = ring_poly.bounds
    reach = float(np.hypot(maxx - minx, maxy - miny)) or 1.0

    out: list[LineString] = []
    for cut in cuts:
        pts = [(float(x), float(y)) for x, y in cut.get("points", [])]
        if len(pts) < 2:
            continue

        if cut.get("closed"):
            geom = LineString(pts + [pts[0]])
        else:
            geom = LineString([_project(pts[0], pts[1], reach)] + pts
                              + [_project(pts[-1], pts[-2], reach)])

        clipped = geom.intersection(ring_poly)
        if clipped.is_empty:
            continue
        out.append(clipped)
    return out


def _project(end: tuple[float, float], inward: tuple[float, float],
             reach: float) -> tuple[float, float]:
    """`end` pushed away from `inward` by `reach`."""
    dx, dy = end[0] - inward[0], end[1] - inward[1]
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        return end
    return (end[0] + dx / norm * reach, end[1] + dy / norm * reach)
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: 5 passed.

Note: `intersection` of a LineString with a Polygon can return a `MultiLineString` when a cut leaves and re-enters the ring. That is correct and `polygonize` handles it in Task 2; no special casing needed.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/territories.py backend/tests/test_territories.py pixi.toml
git commit -m "feat: territory cut extension and clipping"
```

---

## Task 2: Pure geometry — deriving faces

**Files:**
- Modify: `backend/xcell/territories.py`
- Modify: `backend/tests/test_territories.py`

**Interfaces:**
- Consumes: `extend_cuts` from Task 1.
- Produces: `derive_faces(ring, cuts) -> list[Polygon]`, deterministically ordered top-to-bottom then left-to-right.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_territories.py`:

```python
def test_one_cut_splits_the_ring_into_two_faces_that_tile_it():
    faces = terr.derive_faces(SQUARE, [_cut([[2.0, 5.0], [8.0, 5.0]])])
    assert len(faces) == 2
    assert sum(f.area for f in faces) == pytest.approx(100.0)
    assert faces[0].intersection(faces[1]).area == pytest.approx(0.0)


def test_two_crossing_cuts_make_four_faces():
    cuts = [_cut([[1.0, 5.0], [9.0, 5.0]], cid="h"),
            _cut([[5.0, 1.0], [5.0, 9.0]], cid="v")]
    faces = terr.derive_faces(SQUARE, cuts)
    assert len(faces) == 4
    assert sum(f.area for f in faces) == pytest.approx(100.0)


def test_a_closed_cut_makes_an_island_and_its_surround():
    loop = _cut([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]], closed=True)
    faces = terr.derive_faces(SQUARE, [loop])
    assert len(faces) == 2
    areas = sorted(f.area for f in faces)
    assert areas[0] == pytest.approx(4.0)     # the island
    assert areas[1] == pytest.approx(96.0)    # everything else


def test_no_cuts_leaves_the_ring_as_a_single_face():
    faces = terr.derive_faces(SQUARE, [])
    assert len(faces) == 1
    assert faces[0].area == pytest.approx(100.0)


def test_face_order_is_deterministic_so_the_ui_does_not_reshuffle():
    cuts = [_cut([[1.0, 5.0], [9.0, 5.0]])]
    first = terr.derive_faces(SQUARE, cuts)
    second = terr.derive_faces(SQUARE, list(reversed(cuts)))
    assert [f.centroid.y for f in first] == [f.centroid.y for f in second]
    # top face first: reading order, not shapely's internal order
    assert first[0].centroid.y > first[1].centroid.y


def test_a_self_intersecting_freehand_cut_still_yields_valid_faces():
    """Noding in unary_union handles the crossing; the extra loop becomes its
    own face rather than an invalid geometry."""
    squiggle = _cut([[2.0, 4.0], [8.0, 6.0], [8.0, 4.0], [2.0, 6.0]])
    faces = terr.derive_faces(SQUARE, [squiggle])
    assert all(f.is_valid for f in faces)
    assert sum(f.area for f in faces) == pytest.approx(100.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: FAIL — `AttributeError: module 'xcell.territories' has no attribute 'derive_faces'`

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/territories.py`:

```python
def derive_faces(ring: list[list[float]], cuts: list[dict[str, Any]]) -> list[Polygon]:
    """The regions the cuts divide the ring into.

    ``unary_union`` nodes every crossing (including a freehand cut crossing
    itself), and ``polygonize`` then builds the faces bounded by that noded
    network. Faces come out exhaustive and disjoint by construction, which is
    the whole reason boundaries rather than regions are the stored object.

    Ordered top-to-bottom then left-to-right so the UI's face list is stable
    across re-derivations.
    """
    ring_poly = Polygon(ring)
    network = unary_union([ring_poly.exterior] + extend_cuts(ring, cuts))
    faces = [f for f in polygonize(network)
             if f.representative_point().within(ring_poly)]
    return sorted(faces, key=lambda f: (-round(f.centroid.y, 9),
                                        round(f.centroid.x, 9)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/territories.py backend/tests/test_territories.py
git commit -m "feat: derive territory faces from ring and cuts"
```

---

## Task 3: Pure geometry — naming faces and assigning cells

**Files:**
- Modify: `backend/xcell/territories.py`
- Modify: `backend/tests/test_territories.py`

**Interfaces:**
- Consumes: `derive_faces`.
- Produces:
  - `name_faces(faces, anchors) -> list[str | None]` where an anchor is `{"name": str, "x": float, "y": float}`.
  - `assign(coords, faces, names, unassigned="unassigned") -> np.ndarray` of dtype object.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_territories.py`:

```python
def test_an_anchor_names_the_face_that_contains_it():
    faces = terr.derive_faces(SQUARE, [_cut([[1.0, 5.0], [9.0, 5.0]])])
    names = terr.name_faces(faces, [{"name": "proximal", "x": 5.0, "y": 8.0},
                                    {"name": "distal", "x": 5.0, "y": 2.0}])
    assert names == ["proximal", "distal"]


def test_a_face_with_no_anchor_is_unnamed_rather_than_guessed():
    faces = terr.derive_faces(SQUARE, [_cut([[1.0, 5.0], [9.0, 5.0]])])
    assert terr.name_faces(faces, [{"name": "proximal", "x": 5.0, "y": 8.0}]) \
        == ["proximal", None]


def test_names_survive_moving_a_cut():
    """The point of anchoring names to points instead of to faces: edit the
    boundary and the labels stay put."""
    anchors = [{"name": "proximal", "x": 5.0, "y": 8.0},
               {"name": "distal", "x": 5.0, "y": 2.0}]
    before = terr.name_faces(terr.derive_faces(SQUARE, [_cut([[1.0, 5.0], [9.0, 5.0]])]), anchors)
    after = terr.name_faces(terr.derive_faces(SQUARE, [_cut([[1.0, 6.5], [9.0, 6.5]])]), anchors)
    assert before == after == ["proximal", "distal"]


def test_every_cell_inside_the_ring_gets_exactly_one_label():
    faces = terr.derive_faces(SQUARE, [_cut([[1.0, 5.0], [9.0, 5.0]])])
    names = ["proximal", "distal"]
    coords = np.array([[5.0, 9.0], [5.0, 1.0], [2.0, 7.0]])
    assert list(terr.assign(coords, faces, names)) == ["proximal", "proximal", "proximal"][:1] + ["distal", "proximal"]


def test_a_cell_outside_the_ring_is_unassigned_not_nearest():
    faces = terr.derive_faces(SQUARE, [])
    labels = terr.assign(np.array([[50.0, 50.0]]), faces, ["whole"])
    assert list(labels) == ["unassigned"]


def test_a_cell_in_an_unnamed_face_is_unassigned():
    faces = terr.derive_faces(SQUARE, [_cut([[1.0, 5.0], [9.0, 5.0]])])
    labels = terr.assign(np.array([[5.0, 2.0]]), faces, ["proximal", None])
    assert list(labels) == ["unassigned"]


def test_assignment_handles_no_faces_without_erroring():
    assert list(terr.assign(np.array([[1.0, 1.0]]), [], [])) == ["unassigned"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'name_faces'`

- [ ] **Step 3: Write the implementation**

Append to `backend/xcell/territories.py`:

```python
def name_faces(faces: list[Polygon],
               anchors: list[dict[str, Any]]) -> list[str | None]:
    """Each face takes the name of an anchor point it contains.

    Names live on points rather than on faces so that editing a boundary
    cannot orphan a label: re-derive the faces, and every anchor is still
    inside whichever face now surrounds it.
    """
    out: list[str | None] = []
    for face in faces:
        hit = None
        for anchor in anchors:
            if face.contains(Point(float(anchor["x"]), float(anchor["y"]))):
                hit = str(anchor["name"])
                break
        out.append(hit)
    return out


def assign(coords, faces: list[Polygon], names: list[str | None],
           *, unassigned: str = "unassigned") -> np.ndarray:
    """Label every coordinate by the face containing it.

    A coordinate outside every face, or inside an unnamed one, gets
    ``unassigned`` — never the nearest face. Snapping to the nearest region
    would invent an annotation the user never drew.
    """
    import shapely
    from shapely import STRtree

    coords = np.asarray(coords, dtype=float)
    labels = np.full(len(coords), unassigned, dtype=object)
    if not faces or len(coords) == 0:
        return labels

    tree = STRtree(faces)
    # predicate is evaluated as tree_geometry.contains(input_geometry)
    input_idx, tree_idx = tree.query(shapely.points(coords), predicate="contains")

    claimed = np.zeros(len(coords), dtype=bool)
    for i, t in zip(input_idx, tree_idx):
        # A point landing exactly on a shared edge matches both neighbours;
        # first match wins, deterministically, because faces are ordered.
        if claimed[i]:
            continue
        claimed[i] = True
        if names[t] is not None:
            labels[i] = names[t]
    return labels
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories.py -q`
Expected: 18 passed.

If `test_every_cell_inside_the_ring_gets_exactly_one_label` fails on the expected list, fix the *test* — the expected value is `["proximal", "distal", "proximal"]` for coords at y=9, y=1, y=7.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/territories.py backend/tests/test_territories.py
git commit -m "feat: name territory faces by anchor and assign coordinates"
```

---

## Task 4: Adaptor — persistence

**Files:**
- Modify: `backend/xcell/adaptor.py`
- Create: `backend/tests/test_territories_adaptor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces on `DataAdaptor`:
  - `save_territories(type_name: str, payload: dict) -> dict`
  - `get_territories() -> dict[str, dict]`
  - `delete_territories(type_name: str) -> dict`
  - Constant `TERRITORY_UNS_KEY = 'xcell_territories_json'`.

A stored type is:
```python
{"embedding": "spatial", "section_col": "section" | None,
 "sections": {"<section>": {"ring": [[x, y], ...],
                            "cuts": [{"id": str, "points": [[x, y], ...], "closed": bool}],
                            "anchors": [{"name": str, "x": float, "y": float}]}},
 "source": "drawn", "created": "<iso>"}
```

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_territories_adaptor.py`:

```python
"""Territory persistence and cell assignment on the adaptor.

Territories are written into the *live* adata, not only at export time — the
whole point is that a dataset carries its own regions.
"""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n=60, n_genes=8, seed=0, sections=True):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(1.0, (n, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    # A grid so membership is easy to reason about: x in [0,10), y in [0,10)
    xs = rng.uniform(0.5, 9.5, n)
    ys = rng.uniform(0.5, 9.5, n)
    ad.obsm["spatial"] = np.column_stack([xs, ys])
    if sections:
        ad.obs["section"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B"))
    return ad


def _adaptor(**kw):
    return DataAdaptor("t.h5ad", adata=_adata(**kw))


def _payload(section_col="section", sections=("A", "B")):
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    per = {
        s: {
            "ring": ring,
            "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
            "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0},
                        {"name": "distal", "x": 5.0, "y": 2.0}],
        }
        for s in sections
    }
    return {"embedding": "spatial", "section_col": section_col,
            "sections": per, "source": "drawn"}


def test_saving_writes_into_the_live_adata_not_only_on_export():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    assert DataAdaptor.TERRITORY_UNS_KEY in a.adata.uns
    stored = json.loads(a.adata.uns[DataAdaptor.TERRITORY_UNS_KEY])
    assert stored["prox-dist"]["sections"]["A"]["anchors"][0]["name"] == "proximal"


def test_saving_stamps_a_creation_date():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    assert a.get_territories()["prox-dist"]["created"]


def test_a_second_type_does_not_clobber_the_first():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    assert sorted(a.get_territories()) == ["ant-post", "prox-dist"]


def test_saving_the_same_name_replaces_it():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    p = _payload()
    p["sections"]["A"]["anchors"][0]["name"] = "renamed"
    a.save_territories("prox-dist", p)
    got = a.get_territories()["prox-dist"]["sections"]["A"]["anchors"][0]["name"]
    assert got == "renamed"


def test_get_returns_empty_when_nothing_was_ever_saved():
    assert _adaptor().get_territories() == {}


def test_delete_removes_one_type_and_leaves_the_rest():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    a.delete_territories("prox-dist")
    assert list(a.get_territories()) == ["ant-post"]


def test_deleting_an_unknown_type_is_an_error_not_a_silent_no_op():
    a = _adaptor()
    with pytest.raises(KeyError, match="nope"):
        a.delete_territories("nope")


def test_a_type_naming_an_embedding_the_dataset_lacks_is_refused():
    a = _adaptor()
    p = _payload()
    p["embedding"] = "X_not_here"
    with pytest.raises(ValueError, match="X_not_here"):
        a.save_territories("prox-dist", p)


def test_a_type_with_no_sections_is_refused():
    a = _adaptor()
    p = _payload()
    p["sections"] = {}
    with pytest.raises(ValueError, match="no sections"):
        a.save_territories("prox-dist", p)


def test_territories_survive_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    path = tmp_path / "round.h5ad"
    a.adata.write_h5ad(path)

    reloaded = DataAdaptor(str(path), adata=anndata.read_h5ad(path))
    assert reloaded.get_territories()["prox-dist"]["sections"]["A"]["cuts"][0]["points"] \
        == [[1.0, 5.0], [9.0, 5.0]]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_adaptor.py -q`
Expected: FAIL — `AttributeError: type object 'DataAdaptor' has no attribute 'TERRITORY_UNS_KEY'`

- [ ] **Step 3: Write the implementation**

In `backend/xcell/adaptor.py`, add a new section immediately **before** the `# Localize` section marker (search for `# Localize — predicting spatial coordinates from a spatial reference`):

```python
    # =========================================================================
    # Territories — hand-drawn regions that annotate cells by occupancy
    # See xcell/territories.py for the geometry.
    # =========================================================================

    # Ragged vertex lists do not survive an h5ad round trip as nested uns
    # structures, so the whole payload is one JSON string — the same choice
    # xcell_lines_json and xcell_analysis_record already make.
    TERRITORY_UNS_KEY = 'xcell_territories_json'

    def get_territories(self) -> dict[str, Any]:
        """Every saved territory type, or {} when none were ever drawn."""
        raw = self.adata.uns.get(self.TERRITORY_UNS_KEY)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            # A corrupt blob must not take the dataset down with it.
            return {}

    def save_territories(self, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Write one territory type into the live AnnData.

        Validated here rather than at draw time so a payload arriving from an
        import or a script cannot put geometry in that names an embedding this
        dataset does not have — which would assign an empty column and look
        like the drawing was wrong.
        """
        if not type_name or not str(type_name).strip():
            raise ValueError('Territory type needs a name')
        embedding = payload.get('embedding') or 'spatial'
        if embedding not in self.adata.obsm:
            raise ValueError(
                f"Territories reference embedding '{embedding}', which this "
                f"dataset does not have. Available: {sorted(self.adata.obsm)}"
            )
        sections = payload.get('sections') or {}
        if not sections:
            raise ValueError(f"Territory type '{type_name}' has no sections")
        for name, block in sections.items():
            if len(block.get('ring') or []) < 3:
                raise ValueError(f"Section '{name}' has no ring")

        stored = self.get_territories()
        stored[type_name] = {
            **payload,
            'embedding': embedding,
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        result = {
            'type': type_name,
            'n_sections': len(sections),
            'n_cuts': sum(len(b.get('cuts') or []) for b in sections.values()),
            'n_named': len({a['name'] for b in sections.values()
                            for a in (b.get('anchors') or [])}),
        }
        self._log_action('save_territories', {'type': type_name}, result)
        return result

    def delete_territories(self, type_name: str) -> dict[str, Any]:
        """Forget one territory type. Its .obs columns are left alone."""
        stored = self.get_territories()
        if type_name not in stored:
            raise KeyError(f"No territory type '{type_name}'")
        del stored[type_name]
        self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        self._log_action('delete_territories', {'type': type_name}, {'type': type_name})
        return {'type': type_name, 'remaining': sorted(stored)}
```

Check the top of `adaptor.py` for `import json` and `from datetime import datetime`; add whichever is missing.

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_adaptor.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_territories_adaptor.py
git commit -m "feat: persist territories in the live AnnData"
```

---

## Task 5: Adaptor — assigning cells to territories

**Files:**
- Modify: `backend/xcell/adaptor.py`
- Modify: `backend/tests/test_territories_adaptor.py`

**Interfaces:**
- Consumes: `get_territories` (Task 4); `derive_faces`, `name_faces`, `assign` (Tasks 2–3).
- Produces: `assign_territories(types: list[str], combine: bool = False, embedding: str | None = None) -> dict` writing `territory_<type>` columns and optionally `territory_<a>__<b>`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_territories_adaptor.py`:

```python
def test_assignment_writes_a_prefixed_categorical_column():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    col = a.adata.obs["territory_prox-dist"]
    assert isinstance(col.dtype, pd.CategoricalDtype)
    assert set(col.dropna().unique()) <= {"proximal", "distal", "unassigned"}


def test_every_cell_above_the_cut_is_proximal():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    ys = a.adata.obsm["spatial"][:, 1]
    labels = a.adata.obs["territory_prox-dist"].astype(str).values
    assert set(labels[ys > 5.0]) == {"proximal"}
    assert set(labels[ys < 5.0]) == {"distal"}


def test_the_result_reports_counts_per_territory():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    result = a.assign_territories(["prox-dist"])
    counts = result["types"]["prox-dist"]["counts"]
    assert counts["proximal"] + counts["distal"] == a.n_cells


def test_a_section_is_only_assigned_by_its_own_faces():
    """Sections sit side by side in one coordinate space; a section-blind
    implementation looks perfectly correct here and is wrong."""
    a = _adaptor()
    payload = _payload(sections=("A",))          # geometry for A only
    a.save_territories("prox-dist", payload)
    a.assign_territories(["prox-dist"])
    labels = a.adata.obs["territory_prox-dist"].astype(str).values
    is_b = (a.adata.obs["section"].astype(str) == "B").values
    assert set(labels[is_b]) == {"unassigned"}
    assert "proximal" in set(labels[~is_b])


def test_the_same_name_in_two_sections_yields_one_label():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    labels = a.adata.obs["territory_prox-dist"].astype(str)
    by_section = labels.groupby(a.adata.obs["section"].astype(str)).unique()
    assert set(by_section["A"]) == set(by_section["B"])


def test_a_dataset_without_the_section_column_uses_the_single_ring():
    ad = _adata(sections=False)
    a = DataAdaptor("t.h5ad", adata=ad)
    a.save_territories("prox-dist", _payload(section_col=None, sections=("all",)))
    a.assign_territories(["prox-dist"])
    assert "unassigned" not in set(a.adata.obs["territory_prox-dist"].astype(str))


def test_combining_two_types_writes_a_third_column():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.save_territories("ant-post", _payload())
    a.assign_territories(["prox-dist", "ant-post"], combine=True)
    combined = a.adata.obs["territory_prox-dist__ant-post"].astype(str)
    assert combined.iloc[0] == "|".join([
        a.adata.obs["territory_prox-dist"].astype(str).iloc[0],
        a.adata.obs["territory_ant-post"].astype(str).iloc[0],
    ])


def test_combining_needs_at_least_two_types():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    with pytest.raises(ValueError, match="at least two"):
        a.assign_territories(["prox-dist"], combine=True)


def test_assigning_an_unknown_type_is_an_error():
    a = _adaptor()
    with pytest.raises(KeyError, match="nope"):
        a.assign_territories(["nope"])


def test_a_cell_with_no_coordinate_gets_na_not_unassigned():
    """NaN means "no position"; unassigned means "positioned outside every
    territory". Collapsing them would hide unplaced cells."""
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    coords = a.adata.obsm["spatial"].copy()
    coords[0] = [np.nan, np.nan]
    a.adata.obsm["spatial"] = coords
    a.assign_territories(["prox-dist"])
    assert pd.isna(a.adata.obs["territory_prox-dist"].iloc[0])


def test_assignment_is_recorded_in_the_analysis_record():
    a = _adaptor()
    a.save_territories("prox-dist", _payload())
    a.assign_territories(["prox-dist"])
    assert any(e["action"] == "assign_territories" for e in a._action_history)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_adaptor.py -q`
Expected: FAIL — `AttributeError: 'DataAdaptor' object has no attribute 'assign_territories'`

- [ ] **Step 3: Write the implementation**

Append to the territories section of `backend/xcell/adaptor.py`:

```python
    TERRITORY_PREFIX = 'territory_'
    TERRITORY_UNASSIGNED = 'unassigned'

    def assign_territories(
        self,
        types: list[str],
        combine: bool = False,
        embedding: str | None = None,
    ) -> dict[str, Any]:
        """Label every cell by the territory it occupies, one column per type.

        Coordinates come from the embedding each type was drawn in, so a type
        imported from a spatial reference reads the *predicted* coordinates on
        this dataset while a locally drawn type reads the real ones.
        """
        from xcell import territories as terr

        stored = self.get_territories()
        missing = [t for t in types if t not in stored]
        if missing:
            raise KeyError(f"No territory type(s): {missing}")
        if combine and len(types) < 2:
            raise ValueError('Combining needs at least two types')

        out: dict[str, Any] = {'types': {}}
        written: list[str] = []

        for type_name in types:
            spec = stored[type_name]
            emb = embedding or spec.get('embedding') or 'spatial'
            if emb not in self.adata.obsm:
                raise ValueError(
                    f"Territory type '{type_name}' was drawn in embedding "
                    f"'{emb}', which this dataset does not have."
                )
            coords = np.asarray(self.adata.obsm[emb], dtype=float)[:, :2]
            placed = np.isfinite(coords).all(axis=1)

            labels = np.full(self.n_cells, self.TERRITORY_UNASSIGNED, dtype=object)
            section_col = spec.get('section_col')
            if section_col and section_col in self.adata.obs.columns:
                section_of = self.adata.obs[section_col].astype(str).values
            else:
                # One tissue: every cell belongs to whatever single section the
                # geometry defines.
                only = next(iter(spec['sections']))
                section_of = np.full(self.n_cells, only, dtype=object)

            for section_name, block in spec['sections'].items():
                mask = (section_of == section_name) & placed
                if not mask.any():
                    continue
                faces = terr.derive_faces(block['ring'], block.get('cuts') or [])
                names = terr.name_faces(faces, block.get('anchors') or [])
                labels[mask] = terr.assign(
                    coords[mask], faces, names,
                    unassigned=self.TERRITORY_UNASSIGNED,
                )

            series = pd.Series(labels, index=self.adata.obs_names, dtype=object)
            series[~placed] = np.nan          # no coordinate is not "outside"
            column = f'{self.TERRITORY_PREFIX}{type_name}'
            self.adata.obs[column] = pd.Categorical(series)
            written.append(column)

            counts = series.dropna().value_counts()
            out['types'][type_name] = {
                'column': column,
                'counts': {str(k): int(v) for k, v in counts.items()},
                'n_unplaced': int((~placed).sum()),
            }

        if combine:
            parts = [self.adata.obs[f'{self.TERRITORY_PREFIX}{t}'].astype(str)
                     for t in types]
            joined = parts[0]
            for p in parts[1:]:
                joined = joined.str.cat(p, sep='|')
            column = f'{self.TERRITORY_PREFIX}{"__".join(types)}'
            self.adata.obs[column] = pd.Categorical(joined)
            written.append(column)
            out['combined_column'] = column

        out['columns'] = written
        self._log_action('assign_territories',
                         {'types': list(types), 'combine': bool(combine)}, out)
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_adaptor.py -q`
Expected: 20 passed.

- [ ] **Step 5: Run the whole backend suite for regressions**

Run: `cd backend && pixi run -e dev pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_territories_adaptor.py
git commit -m "feat: assign cells to territories by occupancy"
```

---

## Task 6: Routes

**Files:**
- Modify: `backend/xcell/api/routes.py`
- Create: `backend/tests/test_territories_routes.py`

**Interfaces:**
- Consumes: all adaptor methods from Tasks 4–5, and `derive_faces`/`name_faces` for the stateless preview.
- Produces:
  - `POST /api/territories/faces` — stateless preview: `{ring, cuts, anchors}` → `{faces: [{polygon, name, area}]}`.
  - `GET /api/territories`, `PUT /api/territories/{type_name}`, `DELETE /api/territories/{type_name}`
  - `POST /api/territories/assign`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_territories_routes.py`:

```python
"""Territory routes. The faces endpoint is deliberately stateless: it is called
on every edit while drawing, and must not be able to observe or mutate a
dataset mid-drag."""
import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

client = TestClient(app)
SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def _install(n=40):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, (n, 5)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(5)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    ad.obsm["spatial"] = np.column_stack([rng.uniform(0.5, 9.5, n),
                                          rng.uniform(0.5, 9.5, n)])
    ad.obs["section"] = pd.Categorical(["A"] * n)
    a = DataAdaptor("t.h5ad", adata=ad)
    routes.set_adaptor(a, slot="primary")
    return a


def _payload():
    return {"embedding": "spatial", "section_col": "section", "source": "drawn",
            "sections": {"A": {"ring": SQUARE,
                               "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]],
                                         "closed": False}],
                               "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0},
                                           {"name": "distal", "x": 5.0, "y": 2.0}]}}}


def test_the_faces_preview_needs_no_dataset():
    resp = client.post("/api/territories/faces", json={
        "ring": SQUARE,
        "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
        "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0}],
    })
    assert resp.status_code == 200
    faces = resp.json()["faces"]
    assert len(faces) == 2
    assert faces[0]["name"] == "proximal"
    assert faces[1]["name"] is None
    assert faces[0]["polygon"][0] == faces[0]["polygon"][-1]  # closed ring


def test_the_faces_preview_reports_area_as_a_finite_number():
    resp = client.post("/api/territories/faces", json={"ring": SQUARE, "cuts": [], "anchors": []})
    area = resp.json()["faces"][0]["area"]
    assert isinstance(area, float) and np.isfinite(area)


def test_a_ring_with_too_few_points_is_a_400():
    resp = client.post("/api/territories/faces",
                       json={"ring": [[0.0, 0.0]], "cuts": [], "anchors": []})
    assert resp.status_code == 400


def test_put_then_get_round_trips_a_type():
    _install()
    assert client.put("/api/territories/prox-dist", json=_payload()).status_code == 200
    got = client.get("/api/territories").json()["territories"]
    assert list(got) == ["prox-dist"]


def test_put_with_an_unknown_embedding_is_a_400():
    _install()
    payload = _payload()
    payload["embedding"] = "X_nope"
    resp = client.put("/api/territories/prox-dist", json=payload)
    assert resp.status_code == 400
    assert "X_nope" in resp.json()["detail"]


def test_deleting_an_unknown_type_is_a_404():
    _install()
    assert client.delete("/api/territories/nope").status_code == 404


def test_assign_writes_the_column_and_reports_counts():
    a = _install()
    client.put("/api/territories/prox-dist", json=_payload())
    resp = client.post("/api/territories/assign",
                       json={"types": ["prox-dist"], "combine": False})
    assert resp.status_code == 200
    assert "territory_prox-dist" in a.adata.obs.columns
    assert resp.json()["types"]["prox-dist"]["counts"]


def test_assigning_an_unknown_type_is_a_404():
    _install()
    resp = client.post("/api/territories/assign", json={"types": ["nope"]})
    assert resp.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_routes.py -q`
Expected: FAIL — 404s, because the routes do not exist.

- [ ] **Step 3: Write the implementation**

In `backend/xcell/api/routes.py`, add before the Localize section:

```python
# =========================================================================
# Territories — hand-drawn regions that annotate cells by occupancy
# =========================================================================


class TerritoryCut(BaseModel):
    id: str
    points: list[list[float]]
    closed: bool = False


class TerritoryAnchor(BaseModel):
    name: str
    x: float
    y: float


class TerritoryFacesRequest(BaseModel):
    ring: list[list[float]]
    cuts: list[TerritoryCut] = []
    anchors: list[TerritoryAnchor] = []


@router.post("/territories/faces")
def territory_faces(request: TerritoryFacesRequest):
    """Derive the regions a set of cuts divides a ring into.

    Deliberately stateless — no dataset, no adaptor. The drawing panel calls
    this on every edit, and a preview that could observe (or block on) live
    dataset state would couple a drag gesture to whatever else is running.
    """
    from xcell import territories as terr

    if len(request.ring) < 3:
        raise HTTPException(status_code=400, detail='A ring needs at least 3 points')
    try:
        cuts = [c.model_dump() for c in request.cuts]
        faces = terr.derive_faces(request.ring, cuts)
        names = terr.name_faces(faces, [a.model_dump() for a in request.anchors])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        'faces': [
            {
                'polygon': [[float(x), float(y)] for x, y in f.exterior.coords],
                'name': n,
                'area': float(f.area),
            }
            for f, n in zip(faces, names)
        ]
    }


class TerritorySection(BaseModel):
    ring: list[list[float]]
    cuts: list[TerritoryCut] = []
    anchors: list[TerritoryAnchor] = []


class TerritoryTypeRequest(BaseModel):
    embedding: str = 'spatial'
    section_col: str | None = None
    sections: dict[str, TerritorySection]
    source: str = 'drawn'


@router.get("/territories")
def list_territories(dataset: str | None = Query(None)):
    """Every territory type saved on this dataset."""
    adaptor = get_adaptor(dataset)
    return {"territories": adaptor.get_territories()}


@router.put("/territories/{type_name}")
def put_territory(
    type_name: str,
    request: TerritoryTypeRequest,
    dataset: str | None = Query(None),
):
    """Save (or replace) one territory type on this dataset."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.save_territories(type_name, request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/territories/{type_name}")
def delete_territory(type_name: str, dataset: str | None = Query(None)):
    """Forget one territory type. Any .obs columns it produced are left alone."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.delete_territories(type_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class TerritoryAssignRequest(BaseModel):
    types: list[str]
    combine: bool = False
    embedding: str | None = None


@router.post("/territories/assign")
def assign_territories(
    request: TerritoryAssignRequest,
    dataset: str | None = Query(None),
):
    """Annotate cells by which territory they occupy."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.assign_territories(
            types=request.types,
            combine=request.combine,
            embedding=request.embedding,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_routes.py -q`
Expected: 8 passed.

- [ ] **Step 5: Run the whole backend suite**

Run: `cd backend && pixi run -e dev pytest -q`

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_territories_routes.py
git commit -m "feat: territory routes, with a stateless faces preview"
```

---

## Task 7: Frontend pure helpers

**Files:**
- Create: `frontend/src/lib/territoryGeometry.ts`
- Create: `frontend/src/lib/territoryGeometry.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ringFromCells(coords: [number, number][], padFraction?: number): [number, number][]`
  - `faceColor(index: number, total: number): string`
  - `polygonPath(screenPoints: string[]): string`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/territoryGeometry.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { ringFromCells, faceColor, polygonPath } from './territoryGeometry'

describe('ringFromCells', () => {
  it('returns a padded bounding box in clockwise corner order', () => {
    const ring = ringFromCells([[0, 0], [10, 10]], 0)
    expect(ring).toEqual([[0, 0], [10, 0], [10, 10], [0, 10]])
  })

  it('pads by a fraction of the larger extent so cuts can reach past the cells', () => {
    const ring = ringFromCells([[0, 0], [10, 10]], 0.1)
    expect(ring[0]).toEqual([-1, -1])
    expect(ring[2]).toEqual([11, 11])
  })

  it('ignores non-finite coordinates rather than producing a NaN ring', () => {
    const ring = ringFromCells([[0, 0], [10, 10], [NaN, 5]], 0)
    expect(ring.flat().every((v) => Number.isFinite(v))).toBe(true)
  })

  it('gives a degenerate single-cell input a ring with area', () => {
    const ring = ringFromCells([[5, 5]], 0.1)
    expect(ring[0][0]).toBeLessThan(ring[1][0])
  })

  it('returns an empty ring when there is nothing to bound', () => {
    expect(ringFromCells([], 0.1)).toEqual([])
  })
})

describe('faceColor', () => {
  it('gives different faces different colors', () => {
    expect(faceColor(0, 4)).not.toBe(faceColor(1, 4))
  })

  it('is stable for the same position', () => {
    expect(faceColor(2, 5)).toBe(faceColor(2, 5))
  })

  it('returns an rgba string carrying the fill transparency', () => {
    expect(faceColor(0, 3)).toMatch(/^rgba\(\d+,\d+,\d+,0\.\d+\)$/)
  })
})

describe('polygonPath', () => {
  it('builds a closed SVG path', () => {
    expect(polygonPath(['1,2', '3,4', '5,6'])).toBe('M 1,2 L 3,4 L 5,6 Z')
  })

  it('returns an empty string when there is nothing to draw', () => {
    expect(polygonPath(['1,2'])).toBe('')
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/lib/territoryGeometry.test.ts`
Expected: FAIL — cannot resolve `./territoryGeometry`.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/lib/territoryGeometry.ts`:

```typescript
/** Pure helpers for territory drawing and rendering.
 *
 * The ring is the outer boundary a section's cuts divide. It is a padded
 * bounding box rather than a hull of the cells: assignment only ever concerns
 * cells, cells only exist inside the tissue, and a box is robust where an alpha
 * shape is fiddly. The padding matters — it gives a freehand cut somewhere to
 * land beyond the outermost cells.
 */

export function ringFromCells(
  coords: [number, number][],
  padFraction = 0.05,
): [number, number][] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of coords) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX)) return []

  // A single cell (or a perfectly flat row) has no extent to pad; fall back to
  // a unit box so the ring always encloses area.
  const span = Math.max(maxX - minX, maxY - minY) || 1
  const pad = span * padFraction
  return [
    [minX - pad, minY - pad],
    [maxX + pad, minY - pad],
    [maxX + pad, maxY + pad],
    [minX - pad, maxY + pad],
  ]
}

/** Evenly spaced hues, mid-lightness, translucent so cells read through. */
export function faceColor(index: number, total: number): string {
  const hue = (index * 360) / Math.max(total, 1)
  const [r, g, b] = hslToRgb(hue / 360, 0.55, 0.55)
  return `rgba(${r},${g},${b},0.22)`
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h * 12) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)]
}

export function polygonPath(screenPoints: string[]): string {
  if (screenPoints.length < 3) return ''
  return `M ${screenPoints.join(' L ')} Z`
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run src/lib/territoryGeometry.test.ts`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/territoryGeometry.ts frontend/src/lib/territoryGeometry.test.ts
git commit -m "feat: pure helpers for territory rings, colors and paths"
```

---

## Task 8: Store state and cut routing

**Files:**
- Modify: `frontend/src/store.ts`
- Modify: `frontend/src/App.tsx:1156-1162` (`handleLineDrawn`)

**Interfaces:**
- Consumes: nothing.
- Produces on the store:
  - `territoryDraft: TerritoryDraft | null` and `setTerritoryDraft`
  - `addTerritoryCut(points: [number, number][], closed: boolean)`
  - `removeTerritoryCut(id: string)`
  - `setTerritorySection(section: string)`
  - `isTerritoryPanelOpen` / `setTerritoryPanelOpen`
  - `isAssignTerritoriesOpen` / `setAssignTerritoriesOpen`

```typescript
export interface TerritoryCut { id: string; points: [number, number][]; closed: boolean }
export interface TerritoryDraft {
  typeName: string
  embedding: string
  sectionCol: string | null
  activeSection: string
  sections: Record<string, { ring: [number, number][]; cuts: TerritoryCut[]; anchors: { name: string; x: number; y: number }[] }>
}
```

- [ ] **Step 1: Add the types and state**

In `frontend/src/store.ts`, next to the `DrawnLine` interface, add the three interfaces above. In the store interface, beside `isDefineSectionsOpen`, add:

```typescript
  isTerritoryPanelOpen: boolean
  setTerritoryPanelOpen: (open: boolean) => void
  isAssignTerritoriesOpen: boolean
  setAssignTerritoriesOpen: (open: boolean) => void
  // Non-null while the Territory panel is armed: drawn cuts route here instead
  // of becoming ordinary shapes in `drawnLines`.
  territoryDraft: TerritoryDraft | null
  setTerritoryDraft: (draft: TerritoryDraft | null) => void
  addTerritoryCut: (points: [number, number][], closed: boolean) => void
  removeTerritoryCut: (id: string) => void
  setTerritorySection: (section: string) => void
```

- [ ] **Step 2: Implement the actions**

In the store body, beside the other modal setters:

```typescript
    isTerritoryPanelOpen: false,
    setTerritoryPanelOpen: (open) => set({ isTerritoryPanelOpen: open }),
    isAssignTerritoriesOpen: false,
    setAssignTerritoriesOpen: (open) => set({ isAssignTerritoriesOpen: open }),
    territoryDraft: null,
    setTerritoryDraft: (draft) => set({ territoryDraft: draft }),

    addTerritoryCut: (points, closed) =>
      set((state) => {
        const draft = state.territoryDraft
        if (!draft) return {}
        const section = draft.sections[draft.activeSection]
        if (!section) return {}
        const cut = { id: `cut_${Date.now()}_${section.cuts.length}`, points, closed }
        return {
          territoryDraft: {
            ...draft,
            sections: {
              ...draft.sections,
              [draft.activeSection]: { ...section, cuts: [...section.cuts, cut] },
            },
          },
        }
      }),

    removeTerritoryCut: (id) =>
      set((state) => {
        const draft = state.territoryDraft
        if (!draft) return {}
        const section = draft.sections[draft.activeSection]
        if (!section) return {}
        return {
          territoryDraft: {
            ...draft,
            sections: {
              ...draft.sections,
              [draft.activeSection]: {
                ...section,
                cuts: section.cuts.filter((c) => c.id !== id),
              },
            },
          },
        }
      }),

    setTerritorySection: (section) =>
      set((state) =>
        state.territoryDraft
          ? { territoryDraft: { ...state.territoryDraft, activeSection: section } }
          : {},
      ),
```

- [ ] **Step 3: Route drawn cuts away from `drawnLines`**

Replace `handleLineDrawn` in `frontend/src/App.tsx` (currently lines 1156-1162):

```typescript
  const handleLineDrawn = useCallback((points: [number, number][]) => {
    if (!selectedEmbedding) return
    const closed = drawTool === 'polygon' || drawTool === 'lasso'
    // While the Territory panel is armed, a drawn curve is a *cut*, not a
    // shape: it belongs to the territory draft and never enters drawnLines.
    if (useStore.getState().territoryDraft) {
      useStore.getState().addTerritoryCut(points, closed)
      return
    }
    // Auto-name and add immediately. Don't exit draw mode — user can keep drawing.
    // (Lines can be renamed in-place from the Shapes panel.)
    addLine(`Line ${drawnLines.length + 1}`, points, selectedEmbedding, drawTool, closed)
  }, [drawnLines.length, selectedEmbedding, drawTool, addLine])
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/store.ts frontend/src/App.tsx
git commit -m "feat: territory draft state and cut routing"
```

---

## Task 9: Render territory faces on the plot

**Files:**
- Modify: `frontend/src/components/ScatterPlot.tsx` (near `storedLinePaths`, ~line 968)

**Interfaces:**
- Consumes: `territoryDraft` (Task 8), `faceColor` / `polygonPath` (Task 7).
- Produces: derived faces rendered as translucent SVG fills beneath the drawn-line overlay.

- [ ] **Step 1: Add face state and fetching**

Near the other `useStore` reads in `ScatterPlot.tsx`:

```typescript
  const territoryDraft = useStore((state) => state.territoryDraft)
  const [territoryFaces, setTerritoryFaces] = useState<{ polygon: [number, number][]; name: string | null }[]>([])
```

Then, after the other effects:

```typescript
  // Faces are derived by the backend on every edit. Debounced: a freehand cut
  // fires many updates and each one is a fresh polygonize.
  useEffect(() => {
    const section = territoryDraft?.sections[territoryDraft.activeSection]
    if (!section || section.ring.length < 3) { setTerritoryFaces([]); return }
    if (territoryDraft.embedding !== embedding.name) { setTerritoryFaces([]); return }
    let cancelled = false
    const timer = setTimeout(() => {
      fetch('/api/territories/faces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ring: section.ring, cuts: section.cuts, anchors: section.anchors }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (!cancelled && d) setTerritoryFaces(d.faces) })
        .catch(() => { /* preview only; the panel reports save failures */ })
    }, 120)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [territoryDraft, embedding.name])
```

- [ ] **Step 2: Build the paths**

Beside `storedLinePaths`:

```typescript
  const territoryFacePaths = useMemo(() => {
    return territoryFaces.map((face, i) => ({
      path: polygonPath(dataToScreen(face.polygon)),
      fill: faceColor(i, territoryFaces.length),
      name: face.name,
    }))
  }, [territoryFaces, dataToScreen])
```

Import at the top: `import { faceColor, polygonPath } from '../lib/territoryGeometry'`

- [ ] **Step 3: Render them under the existing line overlay**

In the SVG overlay, **before** the element that renders `storedLinePaths` (so faces sit beneath the cuts):

```tsx
        {territoryFacePaths.map((f, i) => f.path && (
          <path key={`face-${i}`} d={f.path} fill={f.fill} stroke="#4ecdc4"
                strokeWidth={1} strokeOpacity={0.5} pointerEvents="none" />
        ))}
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc --noEmit`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ScatterPlot.tsx
git commit -m "feat: render derived territory faces on the spatial plot"
```

---

## Task 10: The Territory panel

**Files:**
- Create: `frontend/src/components/TerritoryPanel.tsx`
- Modify: `frontend/src/App.tsx` (mount it)
- Modify: `frontend/src/components/ScanpyModal.tsx` (Spatial launcher)

**Interfaces:**
- Consumes: store actions from Task 8, `ringFromCells` from Task 7, routes from Task 6.
- Produces: a working draw → name → save loop.

- [ ] **Step 1: Build the panel**

Create `frontend/src/components/TerritoryPanel.tsx`. It follows `DefineSectionsPanel`: floating, non-blocking, so the plot stays usable while drawing.

Structure (write it in this order):

1. Reads: `territoryDraft`, `setTerritoryDraft`, `addTerritoryCut`, `removeTerritoryCut`, `setTerritorySection`, `schema`, `setInteractionMode`, `setDrawTool`, `setSelectedEmbedding`.
2. **Setup phase** — type name input (default `territory`), section-column select populated from `/api/obs/summaries` categorical columns plus a "— treat as one tissue —" option, and a Start button.
3. On Start: fetch the embedding coordinates for the spatial embedding, compute `ringFromCells` per section value (filtering cells by that section's value), build the draft, `setSelectedEmbedding('spatial')`, `setInteractionMode('draw')`, `setDrawTool('smooth_curve')`.
4. **Drawing phase** — a section switcher (one button per section, showing cut counts), a cut list with delete buttons, and a face list.
5. The face list comes from the same `/api/territories/faces` call the plot makes; each row shows the face's area share and either its name or "unnamed", with an inline text input to name it. Naming sets an anchor at the face's representative point, which the endpoint returns.
6. Save: `PUT /api/territories/{typeName}` with the draft's sections; then `setTerritoryDraft(null)`, `setInteractionMode('pan')`, and call `refreshSchema()`.

Key detail for step 5 — extend the faces response to include the representative point so the panel can place an anchor without geometry code. In `routes.py`, in the `territory_faces` response, add to each face dict:

```python
                'anchor': [float(f.representative_point().x),
                           float(f.representative_point().y)],
```

and add a route test in `backend/tests/test_territories_routes.py`:

```python
def test_the_faces_preview_returns_an_anchor_inside_each_face():
    resp = client.post("/api/territories/faces", json={
        "ring": SQUARE,
        "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
        "anchors": [],
    })
    faces = resp.json()["faces"]
    # top face's anchor is above the cut, bottom face's below
    assert faces[0]["anchor"][1] > 5.0
    assert faces[1]["anchor"][1] < 5.0
```

- [ ] **Step 2: Run the route test**

Run: `cd backend && pixi run -e dev pytest tests/test_territories_routes.py -q`
Expected: 9 passed.

- [ ] **Step 3: Mount the panel**

In `App.tsx`, beside `<DefineSectionsPanel />`, add `<TerritoryPanel />`.

In `ScanpyModal.tsx`, add a `territories` entry to the Spatial tab's function list with a launcher button, following exactly the `neighborhood` pattern at the `selectedFunction === 'neighborhood'` branch (~line 2662). The description:

```
Draw named regions on the spatial plot and save them with the dataset. You draw
the dividing cuts; xcell derives the regions between them, so neighbouring
territories share a boundary exactly and cannot overlap or leave gaps. Opens the
Territory tool.
```

- [ ] **Step 4: Typecheck and test**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`

- [ ] **Step 5: Verify in the browser**

Start an isolated stack (worktree, backend on :8100, Vite on :5273 with `XCELL_BACKEND`), load
`chef-gene/real_st_data/E11.5_hindlimbs_102424_xcell_qced-08132026.h5ad` (three real sections in one coordinate space), and confirm:
- drawing a cut across section A shows two faces immediately;
- switching to section B shows *its* cuts, not A's;
- naming a face updates the list and survives adding another cut;
- Save then reload the page: `GET /api/territories` returns the type.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/TerritoryPanel.tsx frontend/src/App.tsx frontend/src/components/ScanpyModal.tsx backend/xcell/api/routes.py backend/tests/test_territories_routes.py
git commit -m "feat: Territory panel — draw cuts, name regions, save"
```

---

## Task 11: Vertex editing

**Files:**
- Modify: `frontend/src/components/ScatterPlot.tsx`
- Modify: `frontend/src/store.ts` (add `updateTerritoryCut`)

**Interfaces:**
- Consumes: `territoryDraft`.
- Produces: `updateTerritoryCut(id: string, points: [number, number][])`; drag-a-vertex interaction; plot-click face naming.

- [ ] **Step 1: Add the store action**

```typescript
    updateTerritoryCut: (id, points) =>
      set((state) => {
        const draft = state.territoryDraft
        if (!draft) return {}
        const section = draft.sections[draft.activeSection]
        if (!section) return {}
        return {
          territoryDraft: {
            ...draft,
            sections: {
              ...draft.sections,
              [draft.activeSection]: {
                ...section,
                cuts: section.cuts.map((c) => (c.id === id ? { ...c, points } : c)),
              },
            },
          },
        }
      }),
```

- [ ] **Step 2: Render vertex handles**

When `territoryDraft` is set and the active section has cuts, render a small circle at every vertex of every cut (screen coords via `dataToScreen`), `r={4}`, fill `#4ecdc4`, plus a larger transparent circle `r={10}` as the hit target.

- [ ] **Step 3: Drag them**

Add `draggingVertex: { cutId: string; index: number } | null` state. On `mousedown` on a handle, set it and stop propagation so the plot does not start panning. On `mousemove` while set, `screenToData` the pointer and call `updateTerritoryCut` with that vertex replaced. On `mouseup`, clear.

The existing debounce in Task 9 means faces re-derive during the drag, which is the point: **both** neighbouring regions follow the boundary because they are derived from it.

- [ ] **Step 4: Click a face to name it**

With the draft armed and no vertex being dragged, a plain click inside a face selects that face in the panel list. This is the spec's original naming interaction, landing here because the plot's interaction code is already open.

- [ ] **Step 5: Verify in the browser**

Confirm dragging a cut's vertex moves the boundary and that *both* adjacent faces update in the same frame, and that names stay attached to the right regions afterwards.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ScatterPlot.tsx frontend/src/store.ts
git commit -m "feat: drag a territory cut's vertices; both neighbours follow"
```

---

## Task 12: Assign modal

**Files:**
- Create: `frontend/src/components/AssignTerritoriesModal.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/ScanpyModal.tsx`

**Interfaces:**
- Consumes: `GET /api/territories`, `POST /api/territories/assign`.
- Produces: the assignment UI.

- [ ] **Step 1: Build the modal**

Global modal gated on `isAssignTerritoriesOpen`, following `ScanpyModal`'s conventions. It:
1. fetches `GET /api/territories` on open and lists each type with a checkbox, its section count and its territory names;
2. shows a "also write a combined column" checkbox, enabled only when ≥2 types are ticked, with a preview of the resulting column name (`territory_a__b`) and an example value (`proximal|anterior`);
3. posts to `/api/territories/assign`;
4. on success shows counts per territory, then calls **both** `refreshSchema()` and `refreshObsSummaries()` — a new `.obs` column needs both channels;
5. reports failures in the standard `errBox`.

- [ ] **Step 2: Mount and launch**

Mount in `App.tsx`; add the Spatial launcher in `ScanpyModal.tsx` next to the Territory one.

- [ ] **Step 3: Typecheck and test**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`

- [ ] **Step 4: Verify in the browser**

On the three-section limb data: assign one type, confirm the new column appears in the Cell Manager without a reload and colors the plot sensibly; then assign two types with combine and confirm the third column's values are `a|b`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/AssignTerritoriesModal.tsx frontend/src/App.tsx frontend/src/components/ScanpyModal.tsx
git commit -m "feat: assign cells to territories from the UI"
```

---

## Task 13: Localize transfer

**Files:**
- Modify: `backend/xcell/adaptor.py` (`spatial_reference_bundle`, new `import_territories`, `prepare_localize`)
- Modify: `backend/xcell/api/routes.py` (`LocalizeRequest`)
- Modify: `backend/tests/test_localize_adaptor.py`
- Modify: `frontend/src/components/LocalizeModal.tsx`

**Interfaces:**
- Consumes: `get_territories`, `save_territories`, `assign_territories`.
- Produces: `import_territories(payload: dict, embedding: str) -> dict`; `LocalizeRequest.import_territories: bool`, `.assign_territories: bool`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize_adaptor.py`:

```python
# --- territory transfer ----------------------------------------------------

def _territory_payload(coords):
    xs, ys = coords[:, 0], coords[:, 1]
    pad = 0.05 * max(xs.ptp(), ys.ptp())
    ring = [[xs.min() - pad, ys.min() - pad], [xs.max() + pad, ys.min() - pad],
            [xs.max() + pad, ys.max() + pad], [xs.min() - pad, ys.max() + pad]]
    mid = float((ys.min() + ys.max()) / 2)
    return {"embedding": "spatial", "section_col": None, "source": "drawn",
            "sections": {"all": {
                "ring": ring,
                "cuts": [{"id": "c1", "closed": False,
                          "points": [[float(xs.min()), mid], [float(xs.max()), mid]]}],
                "anchors": [{"name": "top", "x": float(xs.mean()), "y": float(ys.max())},
                            {"name": "bottom", "x": float(xs.mean()), "y": float(ys.min())}]}}}


def test_the_bundle_carries_the_reference_territories():
    ref_ad, coords = _spatial()
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    ref.save_territories('band', _territory_payload(coords))
    bundle = ref.spatial_reference_bundle()
    assert 'band' in bundle['territories']


def test_a_reference_without_territories_still_bundles():
    ref, _, _ = _pair()
    assert ref.spatial_reference_bundle()['territories'] == {}


def test_importing_rewrites_the_embedding_to_the_predicted_key():
    """Geometry drawn on the reference's `spatial` is only meaningful over the
    query's *predicted* coordinates, so the key has to be rewritten or the
    faces would be drawn over the query's own UMAP."""
    ref_ad, coords = _spatial()
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    ref.save_territories('band', _territory_payload(coords))
    _, query, _ = _pair()
    query.adata.obsm['X_spatial_pred'] = np.zeros((query.n_cells, 2))

    query.import_territories(ref.get_territories(), embedding='X_spatial_pred')
    imported = query.get_territories()['band']
    assert imported['embedding'] == 'X_spatial_pred'
    assert imported['source'].startswith('imported:')


def test_localize_can_import_and_assign_in_one_run():
    ref_ad, coords = _spatial()
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    ref.save_territories('band', _territory_payload(coords))
    q_ad, truth = _query_from(ref_ad, coords)
    query = DataAdaptor('scrna.h5ad', adata=q_ad)

    bundle = ref.spatial_reference_bundle()
    compute_fn, apply_fn = query.prepare_localize(
        bundle, import_territories=True, assign_territories=True)
    result = apply_fn(compute_fn())

    assert 'territory_band' in query.adata.obs.columns
    assert set(query.adata.obs['territory_band'].dropna().astype(str)) <= {'top', 'bottom', 'unassigned'}
    assert result['territories']['imported'] == ['band']


def test_unplaced_cells_get_no_territory():
    ref_ad, coords = _spatial()
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    ref.save_territories('band', _territory_payload(coords))
    q_ad, _ = _query_from(ref_ad, coords)
    query = DataAdaptor('scrna.h5ad', adata=q_ad)

    bundle = ref.spatial_reference_bundle()
    compute_fn, apply_fn = query.prepare_localize(
        bundle, min_confidence=2.0,      # nothing can clear this: all unplaced
        import_territories=True, assign_territories=True)
    apply_fn(compute_fn())
    assert query.adata.obs['territory_band'].isna().all()


def test_assigning_without_importing_is_refused():
    ref, query, _ = _pair()
    bundle = ref.spatial_reference_bundle()
    with pytest.raises(ValueError, match='import'):
        query.prepare_localize(bundle, import_territories=False, assign_territories=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && pixi run -e dev pytest tests/test_localize_adaptor.py -q`
Expected: FAIL — `KeyError: 'territories'` and unexpected keyword arguments.

- [ ] **Step 3: Implement**

1. In `spatial_reference_bundle`, add to the returned dict:

```python
            'territories': self.get_territories(),
```

2. Add to the territories section of the adaptor:

```python
    def import_territories(self, payload: dict[str, Any], embedding: str) -> dict[str, Any]:
        """Adopt another dataset's territories, retargeted at a local embedding.

        The geometry belongs to the *reference's* coordinate space, which on
        this dataset is the predicted embedding — so the embedding key is
        rewritten, and the source is stamped so a later reader can tell drawn
        regions from borrowed ones.
        """
        if embedding not in self.adata.obsm:
            raise ValueError(
                f"Cannot import territories into '{embedding}': this dataset "
                f"has no such embedding."
            )
        stored = self.get_territories()
        imported: list[str] = []
        stamp = datetime.now().isoformat(timespec='seconds')
        for name, spec in (payload or {}).items():
            stored[name] = {
                **spec,
                'embedding': embedding,
                'source': f'imported:{self.filepath.name}@{stamp}',
            }
            imported.append(name)
        if imported:
            self.adata.uns[self.TERRITORY_UNS_KEY] = json.dumps(stored)
        self._log_action('import_territories',
                         {'embedding': embedding}, {'imported': imported})
        return {'imported': imported}
```

3. In `prepare_localize`, add the two keyword arguments (default `False`), validate synchronously:

```python
        if assign_territories and not import_territories:
            raise ValueError(
                'Assigning territories needs the boundaries imported too — '
                'tick "import territory boundaries".'
            )
```

Snapshot `bundle.get('territories') or {}` into a local, and in `apply_fn`, after the predicted embedding is written:

```python
            territory_report: dict[str, Any] = {'imported': [], 'assigned': []}
            if import_territories and snap_territories:
                territory_report.update(
                    self.import_territories(snap_territories, embedding=key_added))
                if assign_territories and territory_report['imported']:
                    assigned = self.assign_territories(
                        territory_report['imported'], embedding=key_added)
                    territory_report['assigned'] = assigned['columns']
            result['territories'] = territory_report
```

4. In `routes.py`, add to `LocalizeRequest`:

```python
    import_territories: bool = False
    assign_territories: bool = False
```

and pass both through in `localize_prepare`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend && pixi run -e dev pytest tests/test_localize_adaptor.py -q`

- [ ] **Step 5: Add the UI**

In `LocalizeModal.tsx`, in the Source-matrix section, add two checkboxes shown only when the reference reports territories (read from `GET /api/territories?dataset=<referenceSlot>` on role resolution):

- "Import territory boundaries from the reference"
- "Assign cells to territories after localizing" (disabled unless the first is ticked)

Wire both into `body()`. Under them, a caption:

```
Boundaries are copied into this dataset against the predicted embedding, so they
draw over the localized cells. Every cell that got a coordinate is labelled;
cells that could not be placed are left blank — localize_confidence sits beside
the result for filtering.
```

- [ ] **Step 6: Full suites, then browser**

Run: `cd backend && pixi run -e dev pytest -q`, `cd frontend && npx tsc --noEmit && npx vitest run`.

In the browser: draw territories on the E11.5 hindlimb reference, load a second dataset as the query, localize with both boxes ticked, and confirm `territory_*` appears on the query and the faces render over `X_spatial_pred`.

- [ ] **Step 7: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_localize_adaptor.py frontend/src/components/LocalizeModal.tsx
git commit -m "feat: carry territories onto localized cells"
```

---

## Task 14: Changelog and merge

- [ ] **Step 1: Write the CHANGELOG entry**

Add to `## [Unreleased]` → `### Added`, in the established voice: what was wrong (Define Sections discards the polygon), what territories do, the load-bearing decision (cuts stored, regions derived — overlaps and gaps unrepresentable rather than merely detectable), and the Localize transfer.

- [ ] **Step 2: Full verification**

```bash
cd backend && pixi run -e dev pytest -q
cd frontend && npx tsc --noEmit && npx vitest run
```

- [ ] **Step 3: Merge**

```bash
git checkout main
git merge --no-ff feat/territories -m "Merge: territories"
```

---

## Self-Review

**Spec coverage:**
- Multi-polygon territory per type → Task 5 (`test_the_same_name_in_two_sections_yields_one_label`).
- Boundaries stored, regions derived → Tasks 1–2.
- Curved/segmented cuts → Task 1 (`points` is a polyline; Task 10 arms `smooth_curve`).
- Cut auto-extension → Task 1.
- Anchor-based naming → Task 3 (`test_names_survive_moving_a_cut`).
- Frozen ring → Task 7 (`ringFromCells` computed once at Start, stored in the draft) + Task 4 (persisted).
- Combined column → Task 5, Task 12.
- Assign everything placed, NaN for unplaced → Task 5, Task 13.
- `uns['xcell_territories_json']` + h5ad round trip → Task 4.
- `embedding` recorded and rewritten on import → Task 13.
- Stateless faces route → Task 6.
- shapely declared → Task 1.
- Shared-edge editing → Task 11.

**Placeholder scan:** none — every code step carries real code. Task 10 and Task 12 describe component structure in numbered detail rather than full JSX, which is deliberate: both are long inline-styled components whose shape is fixed by the named existing components they mirror (`DefineSectionsPanel`, `ScanpyModal`), and the executor reads those.

**Type consistency:** `TerritoryCut` / `TerritoryDraft` (Task 8) match the payload shape in Tasks 4 and 6. `assign_territories(types, combine, embedding)` is called identically in Tasks 5, 6 and 13. `TERRITORY_UNS_KEY`, `TERRITORY_PREFIX` and `TERRITORY_UNASSIGNED` are defined once in Task 4/5 and referenced by name thereafter. `faceColor(index, total)` and `polygonPath(screenPoints)` (Task 7) are used with those signatures in Task 9.

**One spec correction:** the spec says self-intersecting cuts are repaired with `buffer(0)`. That is the idiom for invalid *polygons*; for lines the actual mechanism is noding inside `unary_union`, which is what Task 2 relies on and tests. The spec line should be corrected when this plan is executed.
