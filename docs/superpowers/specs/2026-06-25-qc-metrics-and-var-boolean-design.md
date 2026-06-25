# calculate_qc_metrics + pattern-based boolean .var columns

**Date:** 2026-06-25

## 1. Goals

Two composable ScanpyModal registry functions:

1. **Add a boolean `.var` column** (Analyze → Genes) from a gene-name prefix or
   regex — to flag, e.g., mitochondrial genes or a species of origin.
2. **`sc.pp.calculate_qc_metrics`** (Analyze → Preprocess) with a `qc_vars`
   picker that selects those boolean `.var` columns; defaults `inplace=True`,
   `log1p=True`.

They compose: create a boolean column (e.g. `mt` from `^mt-`), then run QC
metrics with `mt` checked → `pct_counts_mt`, `total_counts_mt` land in `.obs`.

## 2. Current state (build on)

- **ScanpyModal registry** (`SCANPY_FUNCTIONS`): category → function → params;
  each dispatches `POST /api/scanpy/<key>`; post-run refreshes schema/obs for a
  hardcoded key list and supports sync results. Param types include `text`,
  `select`, `obs_column_select` (now with `obsDtype`), `layer_select`,
  `gene_subset`.
- **Boolean `.var` columns** are already fetched into a `booleanColumns` state
  (`/var/boolean_columns`) and rendered as toggle chips for the `gene_subset`
  param. The `qc_vars` picker reuses `booleanColumns`.
- **Backend:** `import scanpy as sc` is present (`adaptor.py:15`).
  `sum_counts_by_pattern` (added 2026-06-24) already matches `var_names` by
  prefix/regex — factor that into a shared helper.
- Routes are individual `@router.post("/scanpy/<key>")` handlers;
  `check_prerequisites` defaults unknown actions to satisfied.

## 3. Feature B — Add boolean `.var` column (build first)

**Adaptor** `_match_var_names(pattern, match_mode) -> np.ndarray[bool]`
(refactored from `sum_counts_by_pattern`, which now calls it):
- `match_mode ∈ {'prefix','regex'}`; empty pattern → `ValueError`.

**Adaptor** `add_var_boolean_column(name, pattern, match_mode='prefix') -> dict`:
- Reject empty `name`.
- Reject `name` already present **and not boolean** (don't clobber a numeric
  metric column); a boolean column of the same name may be overwritten
  (re-runnable).
- `mask = _match_var_names(pattern, match_mode)`; reject **no match** (consistent
  with `sum_counts_by_pattern`).
- `adata.var[name] = mask` (bool dtype).
- Return `{"name": name, "n_genes_matched": int(mask.sum())}`.

**Route** `POST /scanpy/add_var_boolean` (model: `name`, `pattern`,
`match_mode='prefix'`). 400 on `ValueError`.

**Registry** under *Genes* (`gene_analysis`): `name` (text),
`pattern` (text, e.g. `^mt-`), `match_mode` (select prefix/regex). Add the key
to the post-run schema-refresh list so the new boolean column is immediately
available to the gene-mask tool and the `qc_vars` picker.

## 4. Feature A — calculate_qc_metrics

**Adaptor** `run_calculate_qc_metrics(qc_vars=None, percent_top=None,
log1p=True) -> dict`:
- `qc_vars`: list (or comma-separated string) of `.var` column names; each must
  exist and be boolean (else `ValueError`). Empty/None → `[]`.
- `percent_top`: None (default) or a list of ints; None skips the
  `pct_counts_in_top_N_genes` columns (avoids scanpy's "fewer genes than
  max(percent_top)" error on small panels). Accept a comma-separated string too.
- Call `sc.pp.calculate_qc_metrics(self.adata, qc_vars=qc_vars,
  percent_top=percent_top, log1p=log1p, inplace=True)`.
- Return `{"qc_vars": qc_vars, "n_obs_columns": <count added>,
  "n_var_columns": <count added>}` (compute added columns by diffing
  `.obs`/`.var` column sets before/after).

**Route** `POST /scanpy/calculate_qc_metrics` (model: `qc_vars: str | None`
(comma-separated), `percent_top: str | None`, `log1p: bool = True`). 400 on
`ValueError`.

**Registry** under *Preprocess* (`preprocessing`): `qc_vars`
(**`var_bool_multiselect`**), `log1p` (select true/false, default true),
`percent_top` (text, optional, comma-separated ints; blank → None). `inplace`
fixed True (not exposed). Add the key to the post-run schema + obs refresh list.

## 5. New frontend param type `var_bool_multiselect`

- Add `'var_bool_multiselect'` to the `ParamDef.type` union.
- Render: toggle chips from `booleanColumns` (already fetched), exactly like the
  `gene_subset` chips, but membership is stored as a **comma-separated string**
  in `paramValues[param.name]` (toggle adds/removes a name). Empty when none
  selected.
- Ensure `booleanColumns` is fetched when a function exposes a
  `var_bool_multiselect` param (extend the existing fetch trigger, which today
  keys off `gene_subset`).
- The run handler sends `paramValues[name]` verbatim (the comma string); the
  backend splits it.

## 6. Error / edge handling

- Empty name/pattern, no gene match, qc_var that is missing or non-boolean,
  name collision with a non-boolean `.var` column → 400 with a clear message.
- Boolean `.var` column re-run with same name → overwrite (idempotent).
- `calculate_qc_metrics` re-run → scanpy overwrites the QC columns (expected).

## 7. Testing

- **Backend (pytest, TDD):**
  - `_match_var_names` / `add_var_boolean_column`: prefix + regex produce the
    correct boolean mask; empty pattern, no-match, and non-boolean name-collision
    raise; boolean column appears in `get_var_boolean_columns()`.
  - `run_calculate_qc_metrics`: adds `total_counts`, `n_genes_by_counts`,
    `log1p_total_counts` to `.obs` and (with `qc_vars=['mt']`) `pct_counts_mt`,
    `total_counts_mt`; `log1p=False` omits the `log1p_` columns; a non-boolean
    qc_var raises.
  - Route tests for both (200 happy path; 400 on bad input).
- **Frontend:** `tsc --noEmit` + `vite build`; Playwright smoke on the isolated
  stack — Genes → Add boolean column (`^mt-` → `mt`), then Preprocess → QC
  metrics with `mt` checked → confirm `pct_counts_mt` shows in the Cells pane.

## 8. Out of scope
- A general multi-select param type beyond boolean `.var` columns.
- `percent_top` UI beyond an optional text field.
- Editing/deleting `.var` columns (separate concern).

## 9. Build order
1. Backend `_match_var_names` refactor + `add_var_boolean_column` + route — TDD.
2. Backend `run_calculate_qc_metrics` + route — TDD.
3. Frontend `var_bool_multiselect` param type.
4. Frontend registry entries (Genes: add_var_boolean; Preprocess:
   calculate_qc_metrics) + post-run refresh wiring.
5. Verify: full suite, tsc/build, Playwright smoke.
