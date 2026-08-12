# File → Export writes where you choose

## The problem

All three exports stream a browser download: `handleExportH5ad` sets
`a.download = 'xcell_export.h5ad'` and lets the browser drop it in Downloads.
You cannot choose the directory, you cannot choose the name, and every export of
the same dataset overwrites or duplicates the last one under an identical
filename. A user who exports three variants ends up with `xcell_export.h5ad`,
`xcell_export (1).h5ad`, `xcell_export (2).h5ad` and no way to tell them apart.

For an h5ad it is also a needless round trip: a multi-gigabyte file is written
to a temp file on the server, streamed through the browser, and written again —
to a directory the user did not pick.

## What already exists, and is therefore not being rebuilt

- `FileBrowser` (`components/FileBrowser.tsx`) — directory navigation with
  shortcuts, a per-`kind` remembered directory in `localStorage`, and an
  `onNavigate(dir, entries)` callback.
- `savePath.ts` — `composeSavePath` / `splitSavePath`, keeping a directory and a
  typed filename joinable in both directions so a path field can stay the single
  source of truth. 15 tests.
- `PyscnModal` — already wires those two into a working save picker for the
  classifier output path.
- `POST /api/record/export` — the precedent for writing server-side into a
  directory the user picked: it validates the directory, sanitises the stem, and
  reports what it wrote.

So this feature is mostly **extracting the picker PyscnModal already contains**
and giving the export routes somewhere to write.

## Approach

### 1. A reusable picker, and PyscnModal deliberately left alone

`components/SavePathPicker.tsx` — a `FileBrowser` plus a filename field plus the
composed absolute path, emitting one `path` string, and warning when the name
already exists in the folder being shown.

**PyscnModal is not refactored onto it**, and that was decided after looking at
what it would take. Its picker is woven into its own style tokens, an `onError`
sink, a "Done" button that collapses a `Browse` toggle, and an outer path field
it keeps in sync. Serving it would mean adding `footer` and `onError` props to a
component that otherwise needs neither — and a shared component that grows props
to satisfy its second caller is evidence the two uses are not the same thing.
The duplication is a recorded choice rather than an oversight: the alternative
was destabilising a working feature with no UI tests for a tidiness gain.

### 2. `savePath.ts` becomes extension-agnostic

`composeSavePath(dir, filename, ext)` takes the extension it should append
rather than hardcoding `.pkl`. Every call site is updated; TypeScript finds them
all. The pickle default is removed rather than kept as a fallback, because a
silent default is exactly how a `.h5ad` would end up named `.pkl`.

### 3. Server-side write, one route per meaning

`POST /api/export/save` with `{path, kind}` where `kind` is `h5ad`, `metadata`
or `gene_sets`. It validates that the parent directory exists, refuses a path
that is a directory, writes, and returns the path written and its size. Errors
map the usual way: a bad directory is a `400` the user can read, not a stack
trace.

The existing streaming `GET /api/export/h5ad` **stays**. Downloading to the
browser is still the right thing when the backend is not on your machine, and
removing it would break a URL that works today.

### 4. Each format proposes a name, and warns before overwriting

The default filename is derived from the loaded dataset — `E11.5_best_102424`
becomes `E11.5_best_102424_xcell.h5ad` — rather than a constant, so two exports
of different datasets do not collide.

Because `FileBrowser` reports the entries of the directory it is showing, the
picker can say **"this will overwrite an existing file"** before the button is
pressed. This writes to the user's filesystem, so a name collision must be
visible in advance rather than discovered afterwards.

### 5. Browse kinds gain the shapes being written

`BROWSE_KINDS` gains `tabular` (`.tsv`, `.csv`, `.txt`) and `geneset` (`.json`,
`.gmt`). h5ad export reuses `data`. Each kind remembers its own directory, so
where you keep exported tables is not forced to be where you keep h5ads.

## Testing

Backend, `backend/tests/test_export_save.py`:

1. Each `kind` writes a file that exists afterwards and is non-empty.
2. An h5ad written this way reads back with the same shape and the new `.obs`
   columns.
3. A non-existent parent directory is a `400` naming the directory.
4. A path that is an existing directory is a `400`, not a traceback.
5. An unknown `kind` is a `400` listing the valid ones.
6. The response is JSON-serializable and reports the path and byte size.

Frontend, `src/lib/savePath.test.ts`: the existing 15 cases move to the explicit
extension, plus cases for an extension already present, a different extension,
and the empty-extension case.

## Out of scope

A native OS save sheet (`showSaveFilePicker`) — Chrome/Edge only, so it would
need this path as its fallback anyway. Exporting figures, the analysis record
(which already writes server-side by its own route), or any new export format.
Remote-backend file writing beyond what the browse endpoint already exposes.
