# Barplot and Expression Heatmap join the Figure System (Part C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The composition barplot and the expression heatmap become saveable figure kinds: "Save as figure" from their tabs, rendered in the Figures tab by the same components, editable through the generic params form, exported and attached like any figure.

**Architecture:** `figure_data` dispatches the two kinds to the existing computations (`adaptor.crosstab`, `heatmap.compute_heatmap_data`) — nothing numerical changes. The barplot's SVG moves into `figures/CompositionBarplotFigure.tsx` (used by both the tab and the Figures view); the heatmap's `HeatmapCanvas` moves into `figures/ExpressionHeatmapFigure.tsx` (canvas; PNG-only export). Two pure helpers map the tabs' configs to figure records.

**Tech Stack:** Python; React + SVG/canvas; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-09-26-figures-and-batch-enrichment-design.md` (Part C)

## Global Constraints

- No numerical change to the crosstab or heatmap computations.
- Figure inputs for the heatmap may carry `cell_indices` (an ad-hoc selection, like the analysis record's selections) so a heatmap drawn on a selection stays reproducible; a named `cell_subset` is preferred when one is active.
- The heatmap kind exports PNG only; the Figures toolbar says so and disables SVG for it.
- Worktree / test commands as in the previous plans.

## Review Focus

1. A barplot figure saved under the ephemeral cell mask must record that fact (inputs carry `cell_subset` or `cell_indices`), never silently draw all cells — Task 1 test.
2. A heatmap figure whose `line_name` no longer exists must fail with a clear message, not a stack trace — Task 1 test.
3. The barplot rendered in the Figures tab must equal the tab's drawing for the same data and params (shared component) — Task 2 by construction; browser check compares bar counts.
4. `share_of` options must come from the live crosstab's `b_categories` — Task 3 browser check.
5. Saving a figure from a tab must not change the tab's own config or view — Task 3 browser check.

---

### Task 1: Backend `figure_data` for the two kinds

**Files:** `backend/xcell/adaptor.py` (figures section), `backend/tests/test_figures_adaptor.py`.

**Interfaces:** `figure_data` returns for `composition_barplot`: the `crosstab` dict (`a_categories, b_categories, counts, b_colors?, n_cells, n_total`) plus `{'params': {...}}`; for `expression_heatmap`: `compute_heatmap_data`'s dict (`matrix, row_labels, row_groups, column_groups, n_bins, n_cells, n_genes_hidden`). `_validate_figure_inputs('expression_heatmap', …)` accepts `cell_indices: list[int] | None` and validates `line_name` against `self._drawn_lines` (or however lines are looked up — `grep -n "def .*line" adaptor.py`); `composition_barplot` accepts `cell_indices` too.

- [ ] **Step 1: Failing tests** (append): barplot data equals `a.crosstab('grp','batch')`; with `inputs.cell_subset='half'` the counts cover 40 cells (`n_cells == 40`) and with `cell_indices=[0..9]` 10; heatmap data with two gene sets returns `matrix` with `len(row_labels) == 6` and `column_groups` for `obs_column='grp'`; `aggregate_gene_sets=True` gives 2 rows; unknown `line_name` → `ValueError` mentioning the name; `create_figure('expression_heatmap', inputs={'gene_sets': []})` → 400-style `ValueError`.
- [ ] **Step 2: RED. Step 3: implement** the two `figure_data` branches: resolve `cell_indices` = `get_cell_subset_indices(cell_subset)` if `cell_subset` else `inputs.get('cell_indices')`; barplot → `self.crosstab(a, b, active_cell_indices=cell_indices)`; heatmap → `from xcell.heatmap import compute_heatmap_data`; `genes` = concatenation of the sets' genes (de-duplicated, order kept), `gene_set_groups = [{'name': s['name'], 'genes': s['genes']}]` (check the shape `HeatmapView` sends in its POST — mirror it exactly), `aggregate_gene_sets`, `cell_ordering`, `obs_column`, `line_name`, `gene_ordering`, `n_bins`, `cell_indices`. Line validation: look up how `/api/heatmap/data` validates `line_name` and reuse.
- [ ] **Step 4: GREEN. Step 5: Commit** `feat(figures): barplot and expression heatmap figure data`.

---

### Task 2: Shared renderers extracted from the tabs

**Files:** Create `frontend/src/components/figures/CompositionBarplotFigure.tsx`, `frontend/src/components/figures/ExpressionHeatmapFigure.tsx`; modify `BarplotView.tsx`, `HeatmapView.tsx`; `lib/figures.ts` (+test): `barplotFigureFromConfig(config, cellSubset, cellIndices)` and `heatmapFigureFromConfig(config, cellSubset)` returning `{kind, inputs, params}`; `CrosstabData`, `ExpressionHeatmapData` types; `paramsToBarplotConfig(params)` for the renderer.

- [ ] **Step 1: Failing vitest** for the two mapping helpers (inputs/params field names match the backend schema; `share_of: null` when `shareOf` is null; `cell_subset` wins over `cell_indices`; heatmap `gene_sets` carry `{name, genes}` only).
- [ ] **Step 2: RED. Step 3:** `CompositionBarplotFigure` = the SVG from `BarplotView` (axis, bars, labels, optional value text) as a `forwardRef<SVGSVGElement>` taking `{data: Crosstab, params, title?, width, height, background, legend: 'inline' | 'none', onHover?}`; `legend: 'inline'` draws a legend inside the SVG (swatch + label per `b_categories`, right of the plot) so an export is self-contained. `BarplotView` renders it with `legend='none'` and keeps its FloatingPanel + tooltip via `onHover`. `ExpressionHeatmapFigure` = `HeatmapCanvas` + `viridisColor` moved out of `HeatmapView.tsx` unchanged, plus a `forwardRef<HTMLCanvasElement>` and a `title` prop; `HeatmapView` imports it. `lib/figures.ts` gains `canvasToPngBlob(canvas)`.
- [ ] **Step 4:** tsc + vitest green; the Barplot and Heatmap tabs still behave (browser in Task 4). **Commit** `refactor(figures): shared barplot and expression-heatmap renderers`.

---

### Task 3: Save as figure + Figures tab support

**Files:** `BarplotView.tsx`, `HeatmapView.tsx` (toolbar buttons), `FiguresView.tsx` (renderer dispatch, canvas export path, dynamic `share_of` options), `lib/figureSchemas.ts` (`share_of` gets `optionsFrom: 'b_categories'`), `NewFigureModal.tsx` unchanged (these kinds are saved from their tabs).

- [ ] **Step 1:** `ParamField` gains `optionsFrom?: 'b_categories'`; `FiguresView`'s `ParamInput` receives `dynamicOptions` built from the current data (`b_categories` for a crosstab) when `optionsFrom` is set, with a leading "— none —" option mapping to `null`.
- [ ] **Step 2:** `FiguresView`: `dataMatchesKind` for `'counts' in data` (barplot) and `'matrix' in data` (heatmap); renderer dispatch; `exportSvg` disabled with title "PNG only for this kind" when the active kind is `expression_heatmap`; `exportPng`/`attach` use `canvasToPngBlob(canvasRef.current)` for it.
- [ ] **Step 3:** Barplot tab toolbar **Save as figure** → `createFigure(barplotFigureFromConfig(config, activeSubsetName, activeCellMask ? indicesFromMask(activeCellMask) : null))` → `refreshFigures(); setActiveFigureId(id); setCenterPanelView('figures')`. Heatmap tab toolbar **Save as figure** → `createFigure(heatmapFigureFromConfig(heatmapConfig, activeSubsetName))` likewise (`cellIndices` from the config's `cellIndices` when set).
- [ ] **Step 4:** tsc + vitest. **Commit** `feat(figures): Save as figure from the Barplot and Heatmap tabs; Figures tab renders both kinds`.

---

### Task 4: Browser verification + docs

Isolated stack, toy data: Barplot tab → choose `cell_type` × `cell_type`? (the toy has one categorical column; add a second via the API: `POST /api/scanpy/run` `add_obs`? simpler: run Leiden? — use `PUT`… check what exists; otherwise create a boolean/categorical column with the Cells panel "Add annotation" from a selection, or use `toy_spatial_3sections.h5ad` which has a section column). Save as figure → Figures tab shows the same number of bars as the tab; edit `normalize` → preview; `share_of` select lists the b categories; Export SVG downloads; Attach works. Heatmap tab → configure with the seeded gene set → Save as figure → Figures tab renders the canvas; Export SVG disabled; Export PNG downloads; reload → both figures re-render. Confirm the tabs' own configs are untouched after saving. CHANGELOG + README additions. Stop servers, clean up, commit `docs: barplot and heatmap figures`.
