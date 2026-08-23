# Can xcell hold more than two datasets?

2026-08-22. Measured on the bundled toy data plus two of Patrick's real files.

**Yes — and the backend already does. The two-dataset limit is a frontend type
and a hand-duplicated pane, not an architectural one.** Memory is not the
barrier people assume it is: an extra 24k-cell dataset costs ~230 MB, while the
process already sits at 4.7 GB for reasons that have nothing to do with how many
datasets are loaded.

## The backend has no two-slot limit today

`backend/xcell/api/routes.py` keeps datasets in `_adaptors: dict[str,
DataAdaptor]`, keyed by an arbitrary string. `LoadRequest.slot` is a plain `str`
with no enum and no validation, every data route already ends in `dataset: str |
None = Query(None)`, and `GET /api/datasets` / `DELETE /api/datasets/{slot}`
already list and unload by name.

Loading five datasets into a stock backend — including two under names it has
never seen — worked with no code change at all:

```
             primary     900 cells x     76 genes
           secondary   1,000 cells x     76 genes
            tertiary   1,600 cells x     13 genes
   sample_E11.5_rep3     900 cells x    120 genes
        real_spatial  24,172 cells x 12,236 genes
5 slots live
```

Each answered `/api/obs/summaries?dataset=<slot>` with its own columns. Slot
names can be meaningful (`sample_E11.5_rep3`), not just ordinal.

## What a dataset actually costs

Measured by RSS delta on the isolated backend, loading Patrick's real spatial
h5ad (24,172 × 12,236) as a fifth slot:

| | |
|---|---|
| RSS before | 420 MB |
| RSS after | 654 MB |
| **Marginal cost** | **+228 MB** |
| Of which `.X` + `counts` (CSR, 8 B/nnz) | 133 MB |

So a dataset costs roughly **1.7× its stored matrices** — the rest is the
`.obs`/`.var` frames, `.obsm` embeddings, and read buffers.

Now compare that with what the live session is actually holding. Patrick's
backend, with two datasets loaded and a session's worth of analysis on them:

| | |
|---|---|
| Process RSS | **4,746 MB** |
| Primary `.X` + `counts` (4,212 × 16,196, 23.2% dense) | 253 MB |
| Secondary `.X` + `counts` (9,145 × 12,236, 3.4% dense) | 61 MB |
| **Datasets as a share of RSS** | **6.6%** |

The other 93% is fixed interpreter and library cost (~400 MB measured on a
backend holding only toy data) plus peak memory from analyses that was freed to
the allocator but never returned to the OS — PCA over 16k genes, UMAP, Leiden,
NMF, LigRec.

**The conclusion that matters:** loading more datasets is cheap; *analysing*
them is expensive. Going from 2 to 6 datasets of Patrick's size adds well under
1 GB. One more NMF run costs more than four more datasets. Any "keep it
efficient" work should target analysis peaks, not dataset count.

## What actually blocks N datasets

All of it is in `frontend/`:

1. **`store.ts:498` — `export type DatasetSlot = 'primary' | 'secondary'`.**
   Widening it to `string` and `datasets` to `Record<string, DatasetState>` is
   the mechanical core of the change. Every consumer indexes by slot already.

2. **The dual pane in `App.tsx` (~2039–2220) is written out twice** — one ~90
   line block reading `datasets.primary.*`, a near-identical one reading
   `datasets.secondary.*`. This is the real work. It wants to become a
   `<DatasetPane slot={s} />` rendered from a list before any of this scales.

3. **`useData.ts:526 — `otherSlot()`** returns "the one that isn't this one".
   Meaningless with three. Used for dual-mode mirroring; needs to become an
   explicit set of mirror targets.

4. **Fixed two-option UI**: the slot `<select>` (App.tsx:1545), the hardcoded
   `useSpatialScale('primary')` / `useSpatialScale('secondary')` pair
   (App.tsx:835), and `layoutMode: 'single' | 'dual'`.

`lib/localizeRoles.ts` is already ahead of this — it types slots as `string` and
derives reference/query from which dataset carries coordinates rather than from
slot names. It is the model for the rest.

## The failure mode to design against

Today's fix (`fix: Compare is per-dataset, not global`) is the canary. Three
features named "Compare" kept dataset-scoped state in *global* store fields, so
switching slots carried the previous dataset's state along. The worst of them
sent the primary's cell indices to the secondary: all in range for the larger
matrix, so it returned a confident answer about the wrong cells.

With two slots this is a bug you can stumble into. With N slots it is the
default outcome for every piece of state anyone forgets to scope. Before adding
slots, the invariant needs to be enforceable rather than remembered:

- Anything keyed to a dataset — cell indices, column names, embedding names,
  gene names — belongs in `DatasetState`, never at the top level.
- `syncFlatFields` is the checklist. A field in `DatasetState` and missing from
  it is a silent staleness bug.
- The flat top-level mirrors exist for component convenience. They are a
  cache of the active slot, and that is the only thing they may ever be.

## What has since been done

All four steps below are **done** (2026-08-22).

The audit in step 3 found five top-level fields that named something inside a
dataset. Four were staleness bugs you would notice; one returned a wrong answer
you would not:

| | |
|---|---|
| `embeddingDims` | keyed by embedding *name*, and both of Patrick's datasets have an `X_pca`. PCs chosen on one silently became how the other was projected. |
| `markerGenesColumn` | an `.obs` column |
| `comparisonCheckedColumn` / `Categories` | an `.obs` column and its ticked categories |
| `lineAssociationResult` | a table about one line on one dataset |

Deliberately left global, each with its reason, recorded beside `datasets` in
`AppState`: `embeddingLabelColumn`, `activeLineId`, `activeTaskId`,
`activeFigure`, the gene-set state, and `heatmapConfig` — the last being the
next candidate, since it names both an `.obs` column and a drawn line.

**The invariant is now mechanical.** `store.test.ts` derives the set of mirrored
fields from `DatasetState` and the top-level state themselves and asserts every
one of them follows a slot switch. It cannot see the other direction — a
dataset-shaped field that was never in `DatasetState` at all — which is what the
list above is for.

## Suggested order

1. **Extract `<DatasetPane slot={...} />`** from the duplicated dual-layout
   blocks. No behaviour change, no new slots — pure deduplication, verifiable
   against the current two-pane view.
2. **Widen `DatasetSlot` to `string`** and `datasets` to a `Record<string, …>`
   with lookups that tolerate a missing key. Keep exactly two slots in the UI.
3. **Audit the store against the checklist above** — every remaining top-level
   field that names something dataset-shaped.
4. **Then** add the UI: a dataset list with load/unload/rename, and a layout
   that tiles 1–N panes instead of switching between `single` and `dual`.

Steps 1–3 are invisible to the user and independently verifiable. Only step 4
changes the product, and by then it is small.

### What step 4 did

Built as designed in `docs/superpowers/specs/2026-08-22-n-dataset-ui-design.md`:
a tab per dataset, `layoutMode: 'single' | 'tiled'`, draggable pane dividers,
unload wired to the `DELETE` route that had never been called, and slot keys
generated by `nextSlotKey()`. Verified with three datasets loaded at once
through the UI.

Two hardcoded pairs went with it: `useSpatialScale('primary')` /
`useSpatialScale('secondary')` became one `<SpatialScaleLoader>` per loaded
slot, and `applyConfigDefaults` stopped writing display preferences to two
slots by name — which had been leaving a third dataset on the built-in
defaults.

Deliberately still open: a grid layout for many panes (they tile in one row),
drag-to-reorder tabs, renaming a dataset by hand, persisting pane widths across
a reload, and choosing *which* subset of loaded datasets to tile — Split shows
all of them.

### What step 4 had to touch

Everything below is what remains after steps 1–3. Patrick's direction for it:
**tabs to switch between datasets, and resizable panes.**

- The two-option slot `<select>` and the Load dialog's slot picker, both of
  which list `primary` and `secondary` by hand.
- `layoutMode: 'single' | 'dual'` — a count, or a set of visible slots, rather
  than two names. `paneDescriptors()` already takes a list and returns
  `{slot, label, showDivider}`, so the pane row itself needs nothing.
- `useSpatialScale('primary')` / `useSpatialScale('secondary')`, called once
  per slot by hand in `App.tsx`.
- An **unload** action. The store has none — `DELETE /api/datasets/{slot}`
  exists on the backend and nothing calls it. The store already tolerates a
  slot disappearing (that is what the no-op guards are for), but nothing can
  make it happen yet.
- Slot **naming**. `loadDatasetIntoSlot` takes any string today and pane labels
  are derived from the slot name, so `sample_E11.5_rep3` already works end to
  end — but nothing in the UI can produce a name other than the two fixed ones.
