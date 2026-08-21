# Territories: hand-drawn spatial regions that travel with the data

**Date:** 2026-08-21
**Status:** approved, ready for implementation planning

## The problem

Define Sections already lets a user carve a spatial dataset into named regions:
lasso a region, name it, label the cells. But the polygon is thrown away — only
the labels survive. That means the boundary cannot be edited, cannot be
visualized afterwards, cannot be re-applied to a filtered dataset, and above all
cannot be carried onto another dataset.

Carrying it is the point. A user who has drawn proximal / medial / distal on an
E11.5 limb section wants dissociated cells localized onto that same tissue to
inherit those labels. That requires the *geometry* to be the saved artifact, not
the labels it happened to produce.

## What we are building

A **territory type** is a named partition of a spatial coordinate space — for
example `prox-dist`, or `ant-post`. It contains **territories**: named regions
that tile the space exhaustively and disjointly. Each type becomes one `.obs`
column when assigned.

Territories are drawn by hand, saved into the AnnData, rendered over the
spatial plot, used to annotate cells by occupancy, and optionally carried onto a
query dataset through Localize.

## Design decisions

These were settled in brainstorming; each rules out a simpler alternative for a
stated reason.

### A territory is a multi-polygon, not a polygon

Sections (A / B / C) sit side by side in one coordinate space. "Proximal" is
drawn once per section but is one label. So a territory holds N regions, one per
section, and all of them yield the same `.obs` value.

### Boundaries are stored; regions are derived

The alternative — storing each territory as its own closed polygon — makes
overlaps and gaps *possible*, and therefore makes detection, reporting and
repair necessary. Rejected in favour of storing the dividing **cuts** and
deriving the faces between them.

A cut is one object referenced by both of its neighbours, so dragging it moves
both at once: shared-edge editing falls out of the representation rather than
being maintained by repair logic. Overlaps and gaps are not detected — they are
unrepresentable.

Cuts are arbitrary polylines or freehand curves (the existing draw tools and
smoothing), not straight lines. A closed cut is an island: a loop inside a face
splits it into inside and outside, which is how a region like the AER is drawn
inside distal.

### Cut ends auto-extend

A freehand cut rarely lands exactly on the tissue edge. Each open cut's
endpoints are projected along their terminal direction until they meet the ring
or another cut; overshoot is trimmed there. The user's drawn points are stored
verbatim and the extension is derived, so repeated editing never compounds.

### Names attach to anchor points, not to faces

The user names a face by clicking inside it; that click *is* the anchor. After a
cut is dragged and faces are re-derived, each face takes the name of whichever
anchor it now contains. Names survive edits with no face-identity bookkeeping.

### The ring is frozen at save time

The outer boundary of a section is the padded bounding box of that section's
cells, computed once and **stored**. Recomputing it would let a later cell
filter silently shift saved boundaries. Storing it is also what lets an imported
reference geometry work on a query dataset whose own cells sit nowhere near it.

A bounding box rather than a concave hull: assignment is about cells, cells only
exist inside the tissue, and a box is robust where an alpha shape is fiddly.
Faces extending past the tissue edge capture nothing that is not there.

### Combined types are a column, not geometry

`prox-dist × ant-post` is written as an extra `.obs` column
`territory_prox-dist__ant-post`, pasting the two labels together as
`proximal|anterior`. It was tempting to make it a real derived
type — polygonizing the union of both cut sets costs almost nothing given the
architecture — but nothing needs those grid polygons to exist, so they do not.

### Transfer assigns everything placed

A localized cell with a coordinate gets whatever territory it lands in. No
confidence filter: `localize_confidence` already sits in `.obs` next to the
result, and duplicating that policy inside territory assignment would hide it.
Cells with no coordinate get NaN.

## Data model

`adata.uns['xcell_territories_json']` — a JSON string, following the
`xcell_lines_json` and `xcell_analysis_record` precedent, because ragged vertex
lists do not survive an h5ad round trip as nested `uns` structures.

```jsonc
{
  "prox-dist": {
    "embedding": "spatial",        // which coordinate space the geometry lives in
    "section_col": "section",      // null = treat the space as one tissue
    "sections": {
      "A": {
        "ring":    [[x, y], ...],                    // frozen outer boundary
        "cuts":    [{"id": "c1", "points": [[x, y], ...], "closed": false}],
        "anchors": [{"name": "proximal", "x": 1.0, "y": 2.0}]
      }
    },
    "source": "drawn",             // or "imported:<file>@<iso-date>"
    "created": "<iso-date>"
  }
}
```

**Written into the live `adata.uns` on save**, not only at export. This is a
deliberate break from drawn lines, which live in `_drawn_lines` in memory and
only reach `.uns` in `prepare_export_with_lines`. The user asked for territories
saved as part of the AnnData.

`embedding` is load-bearing for rendering: geometry drawn on `spatial` is only
meaningful over `spatial`. On import into a query it is rewritten to the
predicted embedding key, so the faces draw correctly over localized cells.

## Components

### `backend/xcell/territories.py` — pure geometry

Arrays in, dicts out. Never imports the adaptor, never mutates state, following
the house rule for analysis modules.

- `extend_cuts(ring, cuts) -> list[LineString]` — project open endpoints to the
  ring or to another cut, trim overshoot.
- `derive_faces(ring, cuts) -> list[Polygon]` — `unary_union` (which nodes every
  crossing) then `shapely.ops.polygonize`. Exhaustive and disjoint by
  construction.
- `name_faces(faces, anchors) -> list[str | None]` — anchor containment.
- `assign(coords, faces, names) -> np.ndarray` — `shapely.STRtree` query, one
  vectorized pass.

shapely 2.1.2 is already installed via squidpy but is currently a transitive
dependency; it gets declared explicitly in `pixi.toml` rather than relying on
squidpy's dependency graph holding still.

### Adaptor surface

- `save_territories(type_name, payload)` / `get_territories()` /
  `delete_territories(type_name)` — read and write `uns['xcell_territories_json']`.
- `assign_territories(types, combine, embedding=None)` — resolves each cell's
  section, derives faces per section, assigns, writes `territory_<type>` columns
  plus the optional combined column, logs via `_log_action`.
- `spatial_reference_bundle` gains the territory payload.
- `import_territories(payload, embedding)` — writes another dataset's geometry
  in, retagged with provenance and the local embedding key.

### Routes

- `POST /api/territories/faces` — **stateless**: `{ring, cuts}` in, face polygons
  out. Called on every edit during drawing, debounced. No adaptor state, so it
  cannot be affected by a concurrent mutation.
- `GET/PUT/DELETE /api/territories` — the saved types for a slot.
- `POST /api/territories/assign` — the assignment run.
- `LocalizeRequest` gains `import_territories: bool` and
  `assign_territories: bool`.

### Frontend

- `TerritoryPanel.tsx` — floating, non-blocking (the plot must stay usable while
  drawing), modelled on `DefineSectionsPanel`. Name the type, pick the section
  column, draw cuts per section, click a face to name it, save.
- Face rendering in `ScatterPlot.tsx` — translucent filled SVG paths in the
  existing overlay, filtered to the type's `embedding` exactly as drawn lines
  are filtered to theirs.
- Vertex-level editing — hit-test a cut's vertices, drag one, re-derive. This is
  new interaction code: the plot moves whole shapes today and has no
  vertex-level handling. It is the largest single piece of the build.
- `AssignTerritoriesModal.tsx` — checkbox per type, combined-column option.
- `pure lib: territoryColors.ts` — face palette, reusing `resolveCategoryPalette`.

## Error handling

- A type whose section has cuts but no anchors saves fine; unnamed faces assign
  to `unassigned` and the panel shows them hatched.
- A cut that divides nothing after extension (fully outside the ring) is
  reported, not silently dropped.
- Assigning a type whose `embedding` is missing from the target dataset is a 400
  naming the embedding, not an empty column.
- A cell with a coordinate outside every face gets `unassigned`; a cell with no
  coordinate gets NaN. The two are distinct on purpose.
- Self-intersecting freehand cuts need no repair: `unary_union` nodes a line
  where it crosses itself, and `polygonize` then makes the extra loop its own
  face. (`buffer(0)` is the idiom for invalid *polygons*, not for lines.)

## Testing

Backend, `backend/tests/test_territories.py` (pure) and
`test_territories_adaptor.py` / `test_territories_routes.py`:

- Faces from a single cut across a square ring: two faces, areas summing to the
  ring, no overlap.
- Two crossing cuts: four faces.
- A closed loop inside a face: an island plus its surround.
- A cut stopping short: extension reaches the ring; the derived partition is
  still exhaustive.
- Anchor naming survives moving a cut — the same names come back attached to the
  shifted faces.
- Assignment: every cell lands in exactly one face; counts match; cells outside
  the ring are `unassigned`.
- Multi-section: a territory named in two sections yields one label; a cell in
  section B is never assigned by section A's faces.
- Round trip: save, write h5ad, reload, assign — identical labels.
- Localize: bundle carries territories; import rewrites the embedding key;
  unplaced cells get NaN while placed cells are all labelled.

Every geometric test asserts on the resulting AnnData state as well as the
returned dict, per the house rule.

Frontend: pure helpers only (`territoryColors`), plus browser verification on
the real three-section E11.5 hindlimb data — the case with sections laid out
side by side is exactly where a section-blind implementation looks correct and
is wrong.

## Build order

1. **Geometry engine + storage + assignment** — headless, fully testable, the
   foundation everything else leans on.
2. **Define Territories panel** — draw cuts, live face preview, name, save.
3. **Vertex editing** — drag a cut, faces follow.
4. **Assign modal** — type checkboxes, combined column.
5. **Localize transfer** — import and assign on localization.

Each phase is independently useful: after 1 the geometry is scriptable, after 2
it is drawable, after 4 it annotates, after 5 it transfers.

## Explicitly out of scope

- Grid polygons for combined types.
- Confidence-weighted or boundary-distance-weighted assignment.
- Editing imported territories on the query dataset (they are provenance-tagged
  and read-only there).
- Alpha-shape tissue outlines.
