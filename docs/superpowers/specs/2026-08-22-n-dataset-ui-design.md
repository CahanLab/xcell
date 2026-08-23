# More than two datasets: the UI

2026-08-22. Step 4 of `docs/measurements/2026-08-22-more-than-two-datasets.md`.
Steps 1–3 made the frontend N-slot capable without changing anything visible.
This is the part that changes the product.

Patrick's direction: **tabs to switch between datasets, and resizable panes.**
He asked for it to be built without further input, so the decisions below were
made rather than asked; each one records why.

## What a slot is called

Two names, not one:

- **The slot key** is the backend identifier — what goes in `?dataset=`.
  `'primary'` must survive as a name, because `appendDataset()` omits the query
  parameter entirely for it. The second stays `'secondary'`. Beyond that,
  `slot3`, `slot4`, … `nextSlotKey()` fills the first free name in that
  sequence, so unloading the secondary and loading another reuses `secondary`
  rather than climbing forever.
- **The label** is what a person reads, and it comes from the data:
  `schema.filename` with its directory and extension stripped. A tab reading
  `E11.5_hindlimbs_102424_xcell_qced` is worth having; four tabs reading
  Primary, Secondary, Slot3, Slot4 are not. It falls back to the capitalized
  slot key when a dataset has no filename.

Deriving keys from filenames was the alternative. Rejected: it needs sanitising
and collision handling for something no user ever sees, and `primary` has to be
special-cased regardless.

## Layout

`layoutMode: 'single' | 'dual'` becomes `'single' | 'tiled'`. With three
datasets "dual" is a lie, and the union is read in eleven places that all mean
"is more than one dataset on screen".

- **single** — the active dataset, full width. Unchanged.
- **tiled** — every *loaded* dataset, side by side, in slot order.

Panes tile in one row. Not a grid: with the realistic two or three datasets a
row is right, and wrapping interacts badly with the resizing below. Many
datasets at once give narrow panes, which is what the resize handles are for.

The Split button keeps its name — Patrick's word for the feature — and toggles
single/tiled.

## Resizing

A real divider element between panes, not the 1px border the pane used to draw
on its own edge. Dragging it transfers width between exactly the two panes it
sits between, so the others do not move.

Widths live in `paneWidths: Record<slot, number>` as flex weights, absent
meaning 1. Global UI state, not per-dataset: it describes the layout, not the
data. `resizePanes()` is pure and does the arithmetic — it converts weights to
pixels, applies the drag, clamps both neighbours to a 120px minimum, and
converts back with the total weight preserved.

Double-clicking a divider resets every pane to equal width.

## Unloading

`DELETE /api/datasets/{slot}` has existed all along with nothing calling it.
The tab's × calls it, then drops the slot from the store.

- **Confirmed first**, via `window.confirm` — the established pattern here for
  destructive actions (GenePanel, AnalysisRecordPanel). Unloading discards
  every analysis done on that dataset, which is not recoverable by reloading
  the file.
- **The last loaded dataset cannot be unloaded.** The app has nowhere to go.
- If the unloaded slot was active, the previous slot in order becomes active,
  or the next one if it was first. `slotAfterUnload()` is pure and tested.

## Loading

The Load dialog's "Load into:" picker lists the loaded datasets by label plus
**New dataset**, which takes the next free key. Loading into an existing slot
replaces it, as it does today.

## The two-slot leftovers this removes

- The header `<select>` listing Primary and Secondary by hand → the tab strip.
- `useSpatialScale('primary')` / `useSpatialScale('secondary')`, called twice by
  hand → a `<SpatialScaleLoader slot>` rendered once per loaded slot. Hooks
  cannot be called in a loop, so the loop has to be over components.
- `applyUserConfig` wrote display preferences into `primary` and `secondary` by
  name — a third dataset silently kept the built-in defaults. Now every slot.

## Deliberately not in scope

- A grid layout for many panes; drag-to-reorder tabs; renaming a dataset by
  hand (the filename is the name); persisting pane widths across a reload.
- Per-dataset visibility in tiled mode — tiled shows everything loaded. If
  three datasets are loaded and only two are wanted side by side, the answer
  today is to unload one. Worth revisiting if it bites.
