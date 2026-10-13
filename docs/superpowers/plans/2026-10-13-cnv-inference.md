# CNV Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** inferCNV-style copy-number inference with subclone calling, a modal, and a CNV heatmap figure kind.

**Architecture:** pure `backend/xcell/cnv.py` (numpy/scipy/scanpy, no new deps) called from `DataAdaptor.prepare_cnv` as a two-phase background task; results land in `.obsm`/`.obs`/`uns` under one key; a `cnv_heatmap` figure kind reads them back. Frontend: `CnvModal.tsx` + `lib/cnv.ts`, `CnvHeatmapFigure.tsx` canvas renderer.

**Tech Stack:** FastAPI, numpy, scipy (`cluster.hierarchy`), scanpy (`tl.leiden`), sklearn `NearestNeighbors`, React + canvas.

**Spec:** `docs/superpowers/specs/2026-10-13-cnv-inference-design.md`

## Global Constraints

- Heavy imports function-local; the module never imports the adaptor.
- Request model fields must not be narrower than the adaptor arguments.
- Every `_log_action` name needs a `codegen.REGISTRY` entry.
- JSON across the API: no NaN/inf; `None` for not computable.
- New `.obs` columns → modal calls `refreshSchema()` and `refreshObsSummaries()`.
- Figure kind: `FIGURE_KINDS`, `_FIGURE_PARAM_KEYS`, `config.yaml figures.cnv_heatmap`, `_validate_figure_inputs`, `figure_data`; frontend `FIGURE_SCHEMAS`, renderer, `FiguresView` dispatch.

## Review Focus

1. Gene names with Ensembl version suffixes (`ENSG00000000003.14`) must match the table — test in Task 1.
2. A chromosome with fewer genes than `window` must produce one window, not crash — Task 1.
3. A reference category with zero cells inside the active mask must 400, not divide by zero — Task 3.
4. `cluster='hierarchical'` on more than 20k cells is quadratic memory; refuse above 20k with a message naming Leiden — Task 1/3.
5. The figure's `obs_column` deleted after creation must raise `FigureInputMissing` (409), not draw ungrouped — Task 5.

---

## File map

- Create `backend/xcell/cnv.py` — positions, ordering, inference, clustering, summaries, heatmap assembly.
- Create `backend/xcell/data/gene_positions_{human,mouse}.tsv.gz` — built by `build_gene_positions.py` (committed).
- Modify `backend/xcell/adaptor.py` — `cnv_status`, `prepare_cnv`, `cnv_results`, figure kind branches.
- Modify `backend/xcell/api/routes.py` — `/cnv/status`, `/cnv`, `/cnv/results`.
- Modify `backend/xcell/codegen.py` — `'cnv'` entry.
- Modify `backend/xcell/config.yaml` — `figures.cnv_heatmap`.
- Tests: `backend/tests/test_cnv.py`, `test_cnv_adaptor.py`, `test_cnv_routes.py`.
- Frontend: create `lib/cnv.ts` (+test), `components/CnvModal.tsx`, `components/figures/CnvHeatmapFigure.tsx`; modify `store.ts`, `App.tsx`, `ScanpyModal.tsx`, `lib/figures.ts`, `lib/figureSchemas.ts` (+test), `components/FiguresView.tsx`, `hooks/useData.ts` (if a fetch helper is needed).
- Docs: `CHANGELOG.md` Unreleased, `README.md` feature line, CLAUDE.md note.

## Interfaces (cnv.py)

```python
CHROMOSOME_ORDER = {'human': [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'],
                    'mouse': [f'chr{i}' for i in range(1, 20)] + ['chrX', 'chrY']}
def load_positions(species) -> dict[str, tuple[str, int, int]]   # id and symbol keys -> (chrom, start, end)
def match_positions(var_names, table) -> (dict[str, tuple], dict n_matched_by: {'ensembl','symbol','symbol_upper'})
def positions_from_columns(var_names, chrom_values, start_values) -> dict[str, tuple]
def gene_order(positions: dict, var_names, include: list[str]) -> list[dict]   # [{chromosome, indices(np.ndarray), starts}] in karyotype order
def plan_windows(order, window, step) -> (windows: list[dict{chromosome,start,end,n_genes,gene_idx: ndarray}], chromosomes: list[dict{name,start,size}])
def reference_subtract(X_log_chunk, ref_means: ndarray (n_ref_groups, n_genes)) -> ndarray
def smooth_windows(centered_chunk, windows) -> ndarray (n_chunk, n_windows)
def infer(get_chunk, n_cells, gene_cols, windows, ref_means, lfc_clip, center_cells, chunk=2000, report=None) -> ndarray float32
def apply_noise_threshold(M, k) -> M
def cnv_score(M) -> ndarray
def chromosome_means(M, chromosomes) -> ndarray (n_cells, n_chrom)
def cluster_clones(M, method, resolution, n_clusters, n_pcs, seed) -> ndarray int labels (ranked by mean score)
def clone_summary(labels, score, chrom_means, chrom_names, ref_mask, ref_labels) -> list[dict]
def heatmap_data(M, groups, group_order, row_mode, max_rows, vmax, chromosomes) -> dict
```

## Tasks

### Task 1: pure module `cnv.py` with `test_cnv.py` (TDD)
Tests: version-suffix match; symbol upper-case fallback; karyotype order and exclusion of X/Y/M; one window for a short chromosome; window count = floor((n−window)/step)+1; bounds rule with two reference groups; chunked == whole; noise threshold zeroing; synthetic tumor (3 clones with planted gains/losses, 1 normal group) → ARI > 0.9 for both methods, gained windows positive, reference cells lowest score; hierarchical refused above 20k.

### Task 2: build and commit the position tables
Run `build_gene_positions.py`; load test reads the shipped mouse table and finds `Actb` on chr5.

### Task 3: adaptor `cnv_status` / `prepare_cnv` / `cnv_results`, codegen entry, `test_cnv_adaptor.py`
Validation 400s (bad column, bad category, empty reference under mask, z-scored input, no positions); outputs keys and NaN/unassigned under a mask; registry entry; log action; `codegen.REGISTRY['cnv']`; results list.

### Task 4: routes + `test_cnv_routes.py`
Status 200 shape; POST 202 then poll completes; results.

### Task 5: `cnv_heatmap` figure kind (backend) + tests in `test_cnv_adaptor.py`
Config defaults, validation, `figure_data` shape; missing column → `FigureInputMissing`.

### Task 6: frontend lib + modal + ScanpyModal entry + store/App wiring
`lib/cnv.ts` builds params; vitest pins shape. Modal per spec.

### Task 7: frontend figure kind
`FigureKind` union, labels, schema + test, default title, `CnvHeatmapFigure.tsx`, `FiguresView` dispatch and PNG export.

### Task 8: browser verification, docs, merge
Synthetic tumor h5ad from a real mouse dataset's genes; run modal; heatmap; CHANGELOG/README/CLAUDE.md; stop servers; merge `--no-ff` after checking the live dataset.
