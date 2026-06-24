# Custom .obs embeddings + multi-species (PDX) counts

**Date:** 2026-06-24
**Status:** Implemented 2026-06-24 (plan
`docs/superpowers/plans/2026-06-24-obs-embedding-and-multispecies-counts.md`).
148 backend tests pass; frontend tsc + build clean; verified live on the adE11
dataset — created an `.obs` embedding (total_counts × n_genes_by_counts, log
both) that auto-selected and rendered as an operable scatter; `sum_counts_by_pattern`
+ `assign_species` produced count columns and a `species` categorical visible in
the Cells pane.

## 1. Goals

Two related features, both implemented as **ScanpyModal registry functions**
(the same pattern that already creates UMAP/PCA embeddings and adds .var/.obs
columns), so they get the generic form, task handling, history, and post-run
refresh for free.

1. **Embedding from two quantitative .obs columns.** The user picks two numeric
   `.obs` columns → a 2-D embedding (x = col A, y = col B), with an optional
   per-axis log toggle. Once created it behaves like any embedding (select in
   the Embedding dropdown; lasso / select / color all already work).
2. **Multi-species counts (PDX).** CellRanger aligned to a combined genome
   produces species-prefixed gene symbols (e.g. `GRCh38_A1BG` human,
   `mm10___Xkr4` mouse). A function sums counts of genes matching a prefix (or
   regex) into a numeric `.obs` column; a second function assigns each cell a
   species from those count columns.

The two compose into the classic **barnyard plot**: sum per-species counts →
build a log–log `.obs` embedding of the two count columns → lasso the human /
mouse / doublet clouds (or use the automatic species assignment).

## 2. Current state (what we build on)

- **Embeddings** are any `.obsm` key that is a 2-D array with ≥2 columns
  (`get_schema` lists them, `adaptor.py:463`; `get_embedding` returns the first
  two columns, `:496`). Storing `obsm['X_<name>']` is sufficient for it to appear
  and be operable.
- **Counts:** `layers['counts']` is snapshotted from `.X` when `.X` looks like
  integer counts (`_maybe_snapshot_counts_layer`, `:293`).
- **Obs columns:** added by assigning `adata.obs[name] = …`; categorical via
  `pd.Categorical` (see `create_annotation`, `:1342`). `get_schema` /
  `/obs/summaries` pick them up.
- **ScanpyModal registry** (`frontend/src/components/ScanpyModal.tsx`):
  `SCANPY_FUNCTIONS` (`:171`) maps category → functions → params. Each function
  dispatches to `POST /api/scanpy/<key>` (`:1116`); the handler supports sync or
  task results, reports `embedding_name` (`:1186`), and post-run refreshes
  schema + obs summaries for a hardcoded key list (`:1260`) and auto-selects new
  embeddings for another key list (`:1281`). Param types include
  `obs_column_select` (currently category-only, `:761`/`:1852`/`:2049`), `select`,
  `text`, `textarea`, `number`, `layer_select`.
- **Prerequisites:** `GET /scanpy/prerequisites/<fn>` gates Run; new functions
  must be handled (return satisfied for ours).

## 3. Shared frontend change

Extend the existing `obs_column_select` param type with an optional
`obsDtype: 'numeric' | 'category'` field (default `'category'` →
backward-compatible). The renderer filters the obs columns it offers by that
dtype, using the obs summaries already fetched. No new param type, no new modal.

## 4. Feature 1 — Embedding from two .obs columns

**Adaptor** `create_obs_embedding(col_x, col_y, log_axes='none', name=None) -> dict`:
- Validate `col_x`, `col_y` exist and are numeric (else `ValueError`).
- `log_axes ∈ {'none','x','y','both'}`; for a logged axis apply `np.log1p` after
  validating the column is non-negative (else `ValueError`
  "log requires non-negative values").
- `name`: default `f"X_{col_x}_vs_{col_y}"`; if the user supplies one, prefix
  with `X_` when absent. Reject if the key already exists in `.obsm`.
- Store `obsm[name] = np.column_stack([x, y]).astype(float)`.
- Return `{"embedding_name": name, "n_cells": n_cells}`.

**Route** `POST /scanpy/embedding_from_obs` (request: `col_x`, `col_y`,
`log_axes='none'`, `name: str | None`). 400 on `ValueError`.

**Registry** entry under *Cell Analysis*: params `col_x` / `col_y`
(`obs_column_select`, `obsDtype: 'numeric'`), `log_axes`
(`select`: None / X / Y / Both), `name` (`text`, optional). Add
`embedding_from_obs` to the post-run schema-refresh list (`:1260`) and the
embedding-auto-select list (`:1281`) so it appears and is selected immediately.

## 5. Feature 2 — Multi-species counts + assignment

**Adaptor** `sum_counts_by_pattern(pattern, match_mode='prefix', obs_name=None,
layer='counts') -> dict`:
- `match_mode ∈ {'prefix','regex'}`. `prefix` → `var_names.str.startswith(pattern)`;
  `regex` → `var_names.str.match(pattern)` (`re`-based). Empty pattern →
  `ValueError`.
- Resolve the matrix: `layers[layer]` if present else `.X` (raw counts
  preferred; `layer='counts'` default).
- `obs[obs_name]` = per-cell sum over matching gene columns (float). Default
  `obs_name` = sanitized pattern + `_counts` (e.g. `GRCh38_` → `GRCh38_counts`).
- `ValueError` if no genes match.
- Return `{"obs_name": obs_name, "n_genes_matched": k}`.

**Adaptor** `assign_species(count_columns, labels=None, obs_name='species',
threshold=0.9) -> dict`:
- `count_columns`: ≥2 existing numeric `.obs` columns. `labels` default = the
  column names (with a trailing `_counts` stripped for readability).
- Per cell: `total = Σ count_columns`; `frac_i = col_i/total`. Assign `labels[i]`
  where `i = argmax(frac)` **iff** `frac_i ≥ threshold`, else `'mixed'`;
  `total == 0 → 'unassigned'`.
- Store `obs[obs_name]` = `pd.Categorical` over the used labels (+ `'mixed'`,
  `'unassigned'` as present).
- Return `{"obs_name": obs_name, "counts": {label: n_cells, ...}}`.

**Routes** `POST /scanpy/sum_counts_by_pattern`, `POST /scanpy/assign_species`
(400 on `ValueError`). **Registry** entries under a new *Multi-genome* category:
- `sum_counts_by_pattern`: `pattern` (text), `match_mode` (select: Prefix/Regex),
  `obs_name` (text, optional), `layer` (layer_select, default `counts`).
- `assign_species`: `count_columns` (text, comma-separated obs names),
  `labels` (text, comma-separated, optional), `obs_name` (text, default
  `species`), `threshold` (number, default 0.9).
Add both keys to the post-run obs-refresh list (`:1260`).

(Comma-separated text is used for `count_columns`/`labels` to avoid a
multi-select param type in v1; it is the lowest-risk addition. A multi-select
type is a clean later enhancement.)

## 6. Prerequisites

`GET /scanpy/prerequisites/<fn>` must return a satisfied result for
`embedding_from_obs`, `sum_counts_by_pattern`, `assign_species` (no hard
prerequisites; validation happens in the adaptor). Wire these keys into the
prerequisites endpoint's known-function handling (default satisfied).

## 7. Error / edge handling

- Non-numeric obs column for an axis / negative values with log → 400 with a
  clear message.
- Pattern matches no genes → 400.
- `assign_species` with <2 columns, or a named column missing/non-numeric → 400.
- Duplicate **embedding** name → 400 (don't clobber e.g. an existing UMAP).
  **Obs** columns (counts / species) **overwrite on re-run** — re-running to
  tweak a pattern or threshold is the expected workflow.
- Cells with `total == 0` species counts → `'unassigned'` (not a crash).

## 8. Testing

- **Backend (pytest, TDD):**
  - `create_obs_embedding`: stores a (n_cells, 2) `obsm` key; `log_axes='x'`
    applies `log1p` to x only; rejects non-numeric column and negative-with-log;
    rejects duplicate name.
  - `sum_counts_by_pattern`: prefix and regex matching select the right genes;
    per-cell sums equal the matrix row-sums over matched columns; derived
    `obs_name`; raises on no match.
  - `assign_species`: a cell that is 95% human → `human`; 50/50 → `mixed`;
    all-zero → `unassigned`; respects `threshold` and custom `labels`.
  - Route tests for all three (200 happy path; 400 on bad input).
- **Frontend:** `tsc --noEmit` + `vite build`; Playwright smoke on the isolated
  stack (backend :8001 + vite :5180, never the user's :8000/:5173):
  - Build an `.obs` embedding from two numeric columns → it appears in the
    Embedding dropdown and is selected; lasso works.
  - On a PDX-style dataset (or a synthetic one with `GRCh38_`/`mm10_` genes),
    run `sum_counts_by_pattern` twice and `assign_species` → the new `.obs`
    columns + `species` categorical appear in the Cells pane.
  Clean up servers + screenshots afterward.

## 9. Out of scope
- Live/recomputed embeddings (these are static snapshots like UMAP/PCA).
- A dedicated multi-select param type (comma-text used for v1).
- Per-cell species doublet-rate modeling beyond the fraction threshold.
- Editing/deleting custom embeddings beyond what existing embedding tooling does.

## 10. Build order
1. Backend `create_obs_embedding` + route (+ prereqs) — TDD.
2. Backend `sum_counts_by_pattern` + route — TDD.
3. Backend `assign_species` + route — TDD.
4. Frontend: `obs_column_select` `obsDtype` filter + the three registry entries
   (new *Multi-genome* category) + post-run refresh/auto-select wiring.
5. Verify: full suite, tsc/build, Playwright smoke on the isolated stack.
