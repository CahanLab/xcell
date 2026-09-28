# Version label, fresh-session reset, clear gene sets, Gene Set Manager

Date: 2026-09-28 · Branch: `feat/gene-set-manager`

## What was asked

1. Show xcell's version in small type under the "xcell" title.
2. **File → reset to a fresh state**: no old AnnData left, ready for a new analysis.
   **Clear all gene sets** so the Gene pane is fresh. Both confirm before wiping.
   Consider reaching them from the top of their panes.
3. A **Gene Set Manager** that makes these quick:
   - delete many folders at once (after importing the 23 signalling pathways,
     each folder currently needs ✕ + confirm);
   - merge more than two sets into one (union by default);
   - create a set by typing or pasting gene names instead of dragging each gene.

Patrick asked for this to go ahead without check-ins, so the decisions below are
mine. This note records them.

## 1. Version label

- `vite.config.ts` defines `__XCELL_VERSION__` (from `package.json`) and
  `__XCELL_COMMIT__` (`git rev-parse --short HEAD` at dev-server start, `''` when
  git is unavailable). No fetch, so it is there on first paint even when the
  backend is down.
- The header shows `v0.1.0` in 10 px muted type under the title. The tooltip adds
  the commit. The commit is kept out of the label because it is read once when
  Vite starts and can go stale in a long-running dev server.
- The version is written in five places: `pixi.toml`, `backend/pyproject.toml`,
  `xcell/__init__.py`, `frontend/package.json`, and two literals in `main.py`.
  `main.py` now reads `__version__`, and a backend test checks that the rest
  agree, so bumping one without the others fails CI.

## 2. Reset session and clear gene sets

**Backend.** `POST /api/session/reset` with body `{clear_gene_sets: bool = false}`:
- cancels every running background task (`TaskManager.cancel_all()`);
- drops every dataset slot;
- optionally empties `gene_set_store`.

It returns `{unloaded, cancelled_tasks, gene_sets_cleared}`. It is slot-less, like
`/gene_sets`, because it is about the whole process.

**Frontend reset** (`File → New session…`, always present) opens a confirm dialog
that shows:
- the datasets that will be unloaded, by name and cell count;
- a reminder that anything not exported is lost, with an *Export…* shortcut when
  data is loaded;
- a checkbox *Also delete all gene sets (N)*, off by default. Gene sets are
  cross-dataset by design, and a new analysis often reuses them.

On confirm it POSTs, forgets the stored workspace arrangement (`xcell_workspace`)
and reloads the page. A reload is the only honest "fresh". Global modals never
unmount ([[global-modals-never-unmount]]), and many components keep local state
and refs (`hydratedGeneSetsRef`, `restoreStartedRef`, …), so any in-place store
reset would leave some of that state behind. Conveniences survive: recent files,
last browse directories, and the library species.

**The empty state has to be clean**, because after a reset it is the normal state
rather than an edge case:
- `ensureActiveSlot` records `backendEmpty` when the backend reports no datasets.
- `useSchema` does not fetch in that state. It would 503 and flash a red "No data
  loaded for slot 'primary'".
- The centre panel's "No data loaded" message gains a **Load a dataset…** button.

**Clear all gene sets** is a store action, `clearAllGeneSets()`. It restores
`createDefaultCategories()`, and the existing debounced PUT persists that. It is
reachable from three places:
- File → *Clear all gene sets…*;
- Genes pane ⋯ → *Clear all gene sets…* (the top of its pane, as asked);
- the manager's toolbar.

Its confirm dialog states counts (sets, folders, categories) and offers *Download
a backup (JSON)* first. The backup uses the same flat format as File → Export →
gene sets, plus `genesDown` and `folder`, which the Import modal already reads, so
a backup re-imports.

**The reset gets no pane shortcut.** It is app-wide and there is no pane it
belongs to. A second copy would only be a second way to hit it by accident.

One shared `ConfirmDialog` component (title, body, danger button, Escape and
click-outside to cancel, busy/error states) replaces `window.confirm` for these
two wipes and the manager's bulk delete.

## 3. Gene Set Manager

A large modal (`GeneSetManagerModal`), store-held source object:
`geneSetManager: {tab, dest?, focus?} | null`, where `null` means closed.

**Entry points:**
- a **Manage** button beside Library / Import / Browse at the top of the Genes pane;
- Genes ⋯ → *Manage gene sets…*;
- a set row's ⋯ → *Edit genes…*, which opens the Edit tab on that set;
- the Manual category's **+**, which now opens the New tab with the destination
  preset instead of creating an empty set to drag genes into;
- a folder's **+**, which does the same with the destination set to that folder.

Genes stay optional in the New tab, so the drag workflow still works.

**Left: the tree, for selecting.**
- A filter box matches set names, folder names and gene symbols
  (case-insensitive). With a filter active, matching folders expand.
- Every category, folder and set has a checkbox. A folder's checkbox selects the
  folder *and* its sets, and shows indeterminate when only some are ticked.
  Unticking a set inside a ticked folder unticks the folder.
- Folders start collapsed, so 23 pathway folders are 23 rows.
- Toolbar:
  - *Select all shown*;
  - *Clear*;
  - **Delete (N)**: one confirm for everything ticked. Ticked folders go
    entirely, ticked sets individually, and the dialog states counts;
  - **Move to…**: any category's top level, any folder, or a new Manual folder;
  - *Clear all…*.

**Right: three tabs.**
- **New** — name, destination, and a textarea for pasted genes (spaces, commas,
  semicolons, tabs, new lines; quotes and bullets stripped; duplicates dropped).
  An optional *down genes* box makes a directional set. A live check against the
  active dataset's genes (`GET /api/genes`, fetched once per open) reports:
  - how many genes were found;
  - which were case-corrected (`sox9 → Sox9`, stored in the dataset's spelling);
  - which were not found. These are dropped unless *Keep genes not in this
    dataset* is ticked.

  With no dataset loaded, genes are kept as typed and the panel says so.
  Enter in the name field or ⌘/Ctrl+Enter in the textarea creates the set.
- **Edit** — the focused set: rename, edit genes and down genes as text (one per
  line), with the same check. Save / Revert, Duplicate.
- **Merge** — every ticked set is an operand, listed with its size. The rule is:
  - **Union — in any set** (the default);
  - **Intersection — in every set**;
  - **In at least k sets**, for a consensus.

  Order is first appearance. Down lists merge by the same rule and drop anything
  already in the up list. For intersection, a set without a down list therefore
  empties the down side. The panel previews the result count and first genes,
  and suggests a name (`A ∪ B ∪ C` for up to three operands, else `Union of 7
  sets`). You pick a destination, and *Delete the source sets after merging* is
  optional.

**Pure logic** (`lib/geneSetManager.ts`, vitest) holds every operation above.
Each is a function `categories → categories` or `lists → list`, so the modal is
mostly wiring:
- `parseGeneText`, `resolveGenes`, `mergeGeneLists` / `mergeGeneSets`;
- `deleteSelection`, `insertGeneSet`, `updateGeneSet`, `moveSets`;
- `filterTree`, `selectionSummary`, `gatherSets`.

IDs are passed in rather than generated in the lib. The store gains one action,
`replaceGeneSetCategories(next)`.

**Kept as is:** the two-operand *Combine sets…* modal. It is the only place that
does set algebra with boolean `.var` columns.

## Testing

- Backend:
  - `test_session_reset.py`: cancels running tasks, unloads every slot,
    gene-set store cleared only on request, schema 503 afterwards;
  - `TaskManager.cancel_all` covered as well;
  - `test_version.py`.
- Frontend (vitest):
  - `geneSetManager.test.ts` covers parsing, resolution, the merge rules
    including directional sets, delete, insert, move, update, filter and
    selection summary;
  - store tests for `clearAllGeneSets` / `replaceGeneSetCategories` /
    `backendEmpty`.
- Browser, on an isolated stack (`:8100` / `:5273`):
  - import the signalling pathways and delete all 23 folders in one confirm;
  - merge 3+ sets;
  - paste a mixed-case list with an unknown gene;
  - clear all gene sets;
  - File → New session, landing on the clean empty state with no failed
    request other than those expected.
