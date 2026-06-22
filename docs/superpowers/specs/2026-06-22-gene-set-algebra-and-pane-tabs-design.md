# Gene-set algebra (.var columns + set operations) and pane de-crowding

**Date:** 2026-06-22
**Status:** Implemented 2026-06-22 (plan
`docs/superpowers/plans/2026-06-22-gene-set-algebra-and-pane-tabs.md`). 134
backend tests pass; frontend tsc + build clean; verified live end-to-end on the
adE11 dataset (Cells Categorical|Continuous tabs; Genes Sets|Color tabs with
Bivariate/Highlight under Color; `.var` columns `mt`/`ribo` materialized as gene
sets; Combine `mt ∪ ribo` → 114-gene Manual set).

## 1. Problem / goal

Two related asks for the Genes (and Cells) panes:

1. **Access `.var` boolean columns as gene sets** — e.g. `highly_variable`,
   `spatially_variable` — so their genes can be used like any gene set.
2. **Set operations** over gene sets and `.var` columns (union / intersection /
   difference) to build focused gene sets.
3. The Genes and Cells panes are **crowded**; adding the above would worsen it,
   so reorganize both with tabs.

## 2. Decisions (locked)

- **Frozen snapshots.** A derived gene set is materialized to a fixed gene-name
  list at creation (an ordinary `GeneSet`). No live re-evaluation. (Recipe
  metadata for a future live mode is out of scope.)
- **Boolean `.var` columns only** for v1 (numeric-with-threshold is a later
  follow-up).
- **UI:** a browsable `.var`-columns list **plus** a Combine modal.
- **Genes pane:** two tabs — **Sets** | **Color**.
- **Cells pane:** two tabs — **Categorical** | **Continuous**.
- **Created sets land in the Manual category.**
- **Operators:** Union (A∪B), Intersection (A∩B), Difference (A−B), Symmetric
  difference (A△B).

## 3. Current state (what we build on)

- Gene sets: `GeneSet { id, name, genes: string[] }` in categories
  (`store.ts`); `manual` category already exists; store actions
  `addGeneSetToCategory` / `addFolderToCategory`.
- `.var` boolean columns already exposed:
  - `GET /var/boolean_columns` → `[{name, n_true/count, ...}]`
    (`adaptor.get_var_boolean_columns`).
  - `GET /var/boolean_column_values` → per-column True **positional indices**
    (uses `adaptor._column_to_bool_array`).
  - Note: these give indices, not gene **names** — see the one new endpoint.
- `CellPanel.tsx`: "Add Annotation" + collapsible **Categorical** / **Continuous**
  / **Hidden** sections (`categoricalColumns` = dtype category/string;
  `continuousColumns` = dtype numeric).
- `GenePanel.tsx` (default export, ~line 1665): gene search/browse
  (`GeneSearch`), the gene-set tree, a **Bivariate Coloring** section (two
  `BivariateAxisPicker`s + Apply), and `HighlightOverlayPanel`.

## 4. Architecture by phase

Three phases, each independently shippable and committable.

### Phase 1 — Pane reorganization (UI only, no new data flow)

- **`CellPanel.tsx`:** keep "Add Annotation" at top; replace the two collapsible
  Categorical/Continuous sections with a **tab strip** `Categorical (n) |
  Continuous (n)` selecting which list renders. "Hidden" stays as a small
  collapsible footer below the active list. Tab state: `useState` in CellPanel.
  No change to per-column behavior (color/hide/rename, Add Annotation).
- **`GenePanel.tsx`:** add a **tab strip** `Sets | Color` under the "Genes"
  header. **Sets** renders `GeneSearch` + the gene-set tree (+ Phase 2/3
  pieces). **Color** renders the existing Bivariate Coloring block +
  `HighlightOverlayPanel`, moved verbatim. Tab state: `useState` in GenePanel.
  The header (title, ⋯, Import, Browse) stays above the tabs.

### Phase 2 — `.var` columns as gene sets (Sets tab)

- **New backend endpoint** `GET /var/column_genes?column=<name>` →
  `{ "column": str, "genes": [str, ...] }`: `var_names` where the boolean
  column is True. Validates the column exists and is boolean (reuse the
  `get_var_boolean_columns` allow-list + `_column_to_bool_array`); 400 otherwise.
  Adaptor method `column_to_gene_names(column) -> list[str]`.
- **`useData.ts`:** `fetchVarBooleanColumns()` (wraps `/var/boolean_columns`)
  and `fetchVarColumnGenes(column)` (wraps the new endpoint).
- **Sets tab UI:** a collapsible **".var columns"** subsection listing each
  boolean column as `name (count)` with a **`+`** button. `+` fetches the
  column's genes and creates a frozen `GeneSet` named after the column in
  **Manual** (via `addGeneSetToCategory('manual', …)`).

### Phase 3 — Combine modal (set operations)

- **New `frontend/src/lib/geneSetOps.ts`** — pure functions over gene-name
  arrays: `union`, `intersection`, `difference(a, b)`, `symmetricDifference`.
  Each de-duplicates and returns a stable-ordered array (order = first
  appearance in A then B). Case handling: exact-string match (gene-set names
  already match the `.var` index; no normalization), documented.
- **New `frontend/src/components/CombineGeneSetsModal.tsx`**, opened from a
  **"⨂ Combine sets…"** button in the Sets tab. State: operand A, operator,
  operand B. Each operand is picked from a unified dropdown listing **all gene
  sets** (label `"<category>: <name> (n)"`) and **`.var` boolean columns**
  (label `".var: <name> (n)"`). Selecting resolves to a gene-name list
  (sets from the store; columns via `fetchVarColumnGenes`, cached per column).
  Shows a **live result count**, an editable **name** input auto-suggested from
  operands+operator (e.g. `highly_variable ∩ my_markers`), and **Create** →
  frozen `GeneSet` in **Manual**. Escape/backdrop closes; disabled Create until
  both operands chosen and name non-empty.
- Store: a `combineModalOpen` boolean (mirrors existing modal-open flags).

## 5. Data model

No schema change. Outputs are ordinary `GeneSet`s in `manual`. The `.var`-columns
list is a *browse surface* (fetched, not stored). Materializing a column or a
combine result copies gene names into a new set; later `.var` changes do not
affect existing sets (frozen).

## 6. Error / edge handling

- Missing/non-boolean column → backend 400; frontend surfaces a small inline
  error in the `.var` list / combine modal.
- Empty result of a set op → still creatable (a 0-gene set) but the Create
  button warns ("0 genes"); user may proceed or adjust. (Decision: allow, with
  a visible count, to avoid surprising blocks.)
- Duplicate set names allowed (each set has a unique id); no rename prompt.
- A `.var` column with 0 True genes still lists (count 0); `+` makes an empty
  set (consistent with above).

## 7. Testing

- **Backend (pytest, TDD):**
  - `column_to_gene_names('highly_variable')` returns exactly the True genes'
    names, in `.var` order.
  - Missing column and non-boolean column raise `ValueError` → route 400.
  - Route `GET /var/column_genes` returns `{column, genes}` for a synthetic
    adata with a boolean `.var` column.
- **Frontend:**
  - `geneSetOps` is pure; verified by `tsc` + a Playwright smoke that creates
    one set per operator and asserts the resulting counts (union ≥ each input,
    intersection ⊆, difference, symmetric difference). (Optional later: add
    vitest to unit-test `geneSetOps` directly — no runner exists today.)
  - Tab reorg verified by `tsc --noEmit` + `vite build` + the Playwright smoke
    (Cells tabs switch; Genes Sets/Color tabs switch; Bivariate/Highlight live
    under Color; `.var` list and Combine under Sets).
- Run the smoke on an **isolated stack** (backend :8001 + vite :5180), never the
  user's :8000/:5173; clean up servers + artifacts after.

## 8. Out of scope

- Live/dynamic (recipe-based) re-evaluating sets.
- Numeric `.var` columns via threshold predicates.
- Set operations involving categorical/continuous `.var` (cells) — this is
  gene-side only.
- Persisting derived sets to the backend gene-set store beyond existing behavior.

## 9. Build order

1. **Phase 1** — Cells tabs, then Genes Sets/Color tabs (`tsc`/build/smoke).
2. **Phase 2** — `/var/column_genes` (TDD) → `useData` wrappers → `.var` list +
   materialize `+`.
3. **Phase 3** — `geneSetOps` util → `CombineGeneSetsModal` → wire the button +
   store flag.
Each phase: commit; final end-to-end smoke on the isolated stack.
