# Analysis Record: reproducible export to Jupyter / Markdown

**Date:** 2026-08-07
**Status:** Approved, ready to implement

## Problem

xcell is a GUI. A user clicks through thirty operations and arrives at a result.
Nothing about that path is publishable, re-runnable, or even legible six months
later. The methods section has to be reconstructed from memory.

The raw material already exists and is going to waste. `DataAdaptor` keeps
`self._action_history`, and 32 mutating methods call `_log_action(action,
params, result)`. There is even a route — `GET /api/scanpy/history` — that
returns it. The frontend never calls that route. The history is:

- **incomplete** — the load itself isn't recorded, so there is no statement of
  what the analysis started from; ~15 mutating routes never log at all;
- **blind to selections** — ten methods accept `active_cell_indices` and none
  record it, so `leiden` on a 4,000-cell lasso selection is logged
  indistinguishably from `leiden` on all 40,000 cells;
- **volatile** — it lives in RAM and dies with the adaptor;
- **unexportable** — there is no path from it to anything a person would read.

## What "reproducible" has to mean here — the honesty constraint

The tempting version of this feature emits a tidy notebook of scanpy calls that
*looks* authoritative. That version is worse than nothing: a methods supplement
that confidently describes an analysis nobody ran. The whole value of the
feature is that a reader can trust it.

So the governing rule is: **every emitted code cell is labelled with how
faithful it is, and the exporter never emits code it can't stand behind.**

Each step carries a `fidelity`:

| fidelity | meaning | example |
|---|---|---|
| `exact` | the emitted line is the library call xcell actually made | `sc.pp.normalize_total(adata, target_sum=10000.0)` |
| `xcell` | reproducible only through xcell's own Python API | `adaptor.run_contourize(...)` |
| `manual` | no code equivalent; described in prose only | a hand-drawn lasso selection |

`exact` is verifiable, not aspirational. `run_normalize_total` with no active
selection literally calls `sc.pp.normalize_total(self.adata, **kwargs)`; that is
why it earns `exact`. The same is true of `log1p`, `pca`, `neighbors`, `umap`,
`leiden`, `highly_variable_genes`, `filter_cells`, `filter_genes`.

Orthogonal to fidelity, each step records `on_subset`. When xcell ran an
operation on an active cell selection, the emitted code is the whole-dataset
call and the step is flagged — loudly, in the notebook, not just in a data
structure. We deliberately do **not** hand-roll subset + write-back code per
operation: each of the ten subset-capable methods has different write-back
semantics (UMAP writes NaN for inactive cells, Leiden writes `'unassigned'`,
`normalize_total` splices sparse data rows), and generated code that is subtly
wrong is exactly the failure this feature exists to prevent. Instead the
selection itself is preserved (see below) so the reader can restore it.

A step whose action has no registry entry still exports — as prose with its
parameters rendered — and counts as `manual`. Coverage gaps are visible, never
silent.

The notebook opens with a reproducibility summary that says, in numbers, how
much of it re-runs.

## Architecture

```
DataAdaptor._log_action ─┬─► _action_history      (unchanged, back-compat)
                         └─► AnalysisRecord       (new)
                                  │
   frontend notes + captured PNGs ┤
                                  ▼
                          codegen.translate()  ──►  notebook_export
                                                      ├─► .ipynb
                                                      └─► .md (+ figures/)
```

Three new backend modules, all pure in the sense CLAUDE.md means — they take
plain data and return plain data, never import the adaptor, never touch live
AnnData:

### `backend/xcell/analysis_record.py`

The data model and nothing else.

```python
@dataclass
class Step:
    index: int
    action: str
    params: dict
    result: dict
    timestamp: str
    note: str | None = None          # user-written annotation
    figure_ids: list[str] = field(default_factory=list)
    n_active: int | None = None      # cells the op ran on, if a subset
    n_total: int | None = None
    selection: list[int] | None = None   # recorded when small enough

@dataclass
class Figure:
    id: str
    png_b64: str
    caption: str = ''
    step_index: int | None = None    # None = a standalone figure at the tail
    timestamp: str = ''

@dataclass
class AnalysisRecord:
    title: str
    abstract: str
    source: dict                     # file path, n_cells, n_genes, loaded_at
    steps: list[Step]
    figures: dict[str, Figure]
    report_start: int                # index of the first step to export
```

Plus `add_step`, `set_note`, `add_figure`, `mark_start`, `steps_for_report()`,
and `to_dict` / `from_dict` for the `uns` round-trip.

**Selections.** `selection` holds the actual cell indices when
`n_active <= SELECTION_CAP` (50,000 — roughly 400 KB of JSON, the point where
carrying it stops being free). Above the cap only the counts are kept and the
step says so. On export, any recorded selections are written to a sibling
`<name>_selections.json` and referenced from the notebook.

**Persistence.** `uns['xcell_analysis_record']` holds `to_dict()` as a **JSON
string** — h5ad's handling of deeply nested heterogeneous dicts is a known
source of grief, and a string round-trips exactly. Written in
`prepare_export_with_lines`, restored in `DataAdaptor.__init__`. So the record
survives export → reload, and an h5ad handed to a colleague carries its own
provenance.

### `backend/xcell/codegen.py`

The registry that turns a logged action into prose and code. One entry per
action, each owning everything the exporters need to know about it:

```python
@dataclass
class ActionSpec:
    label: str                                   # "Leiden clustering"
    fidelity: str                                # exact | xcell | manual
    summary: Callable[[dict, dict], str]         # boilerplate annotation
    code: Callable[[dict, dict], list[str]] | None
    imports: tuple[str, ...] = ()
```

`translate(step) -> TranslatedStep(title, summary, code_lines, fidelity,
imports, warnings)`. This is where reproducibility actually lives, and because
it is a pure function of two dicts it is testable with no AnnData at all — one
table-driven test per action.

The `summary` callables are the "boiler-plate xcell provided" annotation the
feature calls for: `leiden` renders as *"Leiden clustering at resolution 1.0 →
12 clusters in `.obs['leiden']`."* — assembled from the params and the recorded
result, so the numbers are real.

### `backend/xcell/notebook_export.py`

Two pure emitters over `(record, translated_steps)`:

- `to_notebook(...) -> dict` — nbformat 4.5 JSON. Figures become
  `display_data` outputs with `image/png` base64, so the `.ipynb` is one
  self-contained file that renders offline.
- `to_markdown(...) -> str` — figures as relative links into a `figures/`
  sibling directory (the caller writes the bytes), or omitted.

Document shape:

```
# <title>
<abstract>

> Generated by xcell on <date> from <path> (12,000 cells x 2,000 genes).
> 11 of 14 steps re-execute as written; 2 require xcell; 1 is manual.
> 3 steps ran on a cell selection — see the flags below.

## Setup                                      [code]
import scanpy as sc
adata = sc.read_h5ad("/path/to/data.h5ad")

## 1. Normalize total                         [markdown]
Normalized each cell to 10,000 counts.
<user note, if any>
                                              [code]
sc.pp.normalize_total(adata, target_sum=10000.0)
                                              [figure output, if attached]
...

## Steps that do not re-execute automatically  [markdown]
- Step 7 — Draw contour: hand-drawn, cannot be regenerated from parameters.
```

Writing files is the route's job, not the module's, so the emitters stay
trivially testable.

## Recording is always on; the *report* is what the user starts

The obvious reading of "activated by the user" is a record button that gates
logging. We reject that: users forget to arm it before doing the interesting
thing, and the data is then gone forever. `_log_action` already fires
unconditionally and costs nothing, so the recorder always accumulates.

What the user controls is the **span they want to tell a story about**.
"Start report here" sets `report_start` to the current step count; export takes
`steps_for_report()`. The requested UX is met exactly — mark a beginning, do the
work, hit export — with the failure mode removed. `report_start` defaults to 0,
so an export with no marker covers the whole session.

## Filling the logging holes

Three fixes to the existing history, all prerequisites for the export to mean
anything:

1. **The load.** `__init__` seeds step 0 with the source path, dimensions, and
   file size. Without it the notebook has nothing to open.
2. **Selections.** The ten methods taking `active_cell_indices` pass
   `subset=indices` to `_log_action` explicitly at the call site. Explicit
   beats a stashed `self._pending_subset`, which would silently misattribute a
   selection to the next operation that happens to log.
3. **Unlogged mutators.** Add `_log_action` to the high-value writers that
   currently record nothing: `rename_label`, `merge_labels`,
   `score_gene_sets_matrix`, `score_genes_ucell`, `embedding_from_obs`,
   `calculate_qc_metrics`, `rename_genes`, `add_var_boolean`,
   `sum_counts_by_pattern`, `assign_species`, `cluster_gene_set`,
   `create_annotation` / `label_cells`, `diffexp`.

## Figures

The frontend owns the pixels. Capture reuses the proven snapshot path rather
than reading back from WebGL: `ScatterPlot.captureSnapshot` already walks the
visible cells and resolves each one's colour through the same `getColor`
deck.gl uses, and `renderSnapshotToCanvas` splats that into a plain 2D canvas.
Rendering the same payload to an offscreen canvas and calling `toDataURL` is
deterministic and has no `preserveDrawingBuffer` hazard — deck.gl does not set
it here, so a direct `toDataURL` on the live canvas is not safe.

`POST /api/record/figure` takes `{png_b64, caption, step_index?}`; with no
`step_index` the figure attaches to the most recent step.

## API

All under `/api/record`, each taking the standard trailing
`dataset: str | None = Query(None)`:

| route | purpose |
|---|---|
| `GET /api/record` | the whole record, figures as ids + captions only (no base64) |
| `PUT /api/record/meta` | set title / abstract |
| `POST /api/record/mark` | set `report_start` to the current step count |
| `POST /api/record/step/{index}/note` | set or clear a step's note |
| `POST /api/record/figure` | attach a captured PNG |
| `DELETE /api/record/figure/{id}` | drop a figure |
| `POST /api/record/export` | render and write; returns paths + a preview |
| `DELETE /api/record` | clear steps (keeps the load step) |

`GET` deliberately omits base64 payloads — a dozen figures would make the
polling response megabytes.

**Export** takes `{format: 'ipynb' | 'md' | 'both', output_dir, filename,
include_figures, include_code}` and **writes server-side**, using the existing
folder-browse UI (the same pattern as the PySCN Train tab's output folder). A
notebook needs to live next to the data to be re-runnable, and a server-side
write solves figure sidecars without inventing a zip-download path. The
response carries the rendered markdown so the modal can show a preview.

## UI

One new component, `frontend/src/components/AnalysisRecordPanel.tsx`, opened
from the **File** menu (where Load and Export already live):

```
Analysis record                                    [x]
Title    [ Cortex atlas — clustering              ]
Abstract [ ...                                    ]

  #  Step                                    Fidelity
  0  Loaded cortex.h5ad (12,000 x 31,053)     exact
  1  Normalize total (target 10,000)          exact
  2  log1p                                    exact
  3  Leiden res 1.0 -> 12 clusters  [subset]  exact     [note] [fig]
     > "resolution chosen from the silhouette sweep"
  ...
                                          [Start report here]

[Capture current plot as figure]

Format (o) Notebook ( ) Markdown ( ) Both
Folder [ /Users/... ] [Browse]   Filename [ analysis ]
[x] figures  [x] code cells                         [Export]
```

Store additions are per-dataset, mirroring `scanpyActionHistory`. After an
export the panel shows the written paths and a preview.

## Testing

`backend/tests/test_analysis_record.py`, `test_codegen.py`,
`test_notebook_export.py`, `test_record_routes.py` — pytest, self-contained,
module-level `_adata()` / `_adaptor()` factories, seeded RNG, `csr_matrix` +
`float32`, per house convention.

The contracts worth pinning down:

- **Codegen is table-driven.** Every action in the registry gets a case
  asserting its generated code and fidelity. A test enumerates every action name
  appearing in a `_log_action` call in `adaptor.py` and asserts the registry
  covers it — so adding a logged action without a spec fails CI rather than
  silently degrading an export to prose.
- **The notebook is valid.** `to_notebook` output round-trips through
  `json.dumps` and satisfies nbformat's structural requirements (cell types,
  `outputs` and `execution_count` on code cells, `nbformat: 4`).
- **`exact` means exact.** An integration test runs a real pipeline through the
  adaptor (normalize → log1p → pca → neighbors → leiden), exports, and asserts
  the emitted lines match the scanpy calls the adaptor makes.
- **Subsets are visible.** Running leiden with `active_cell_indices` produces a
  step with `n_active` set, and the exported markdown says so.
- **Round-trip.** `from_dict(to_dict(record))` is lossless, including figures
  and notes; `uns` persistence survives a write/read of the h5ad.
- **JSON-safety.** No `inf`/`NaN` reaches the API — recorded results are
  sanitized on the way into the record, since `marker_genes` and `diffexp`
  results carry floats from scipy.

Then in the browser, per this project's rule: run an analysis, export, and
**actually execute the generated notebook** against the toy dataset. A notebook
that doesn't run is the only failure that matters, and it is not one pytest will
catch.

## Out of scope

Editing or reordering steps; re-running a notebook inside xcell; HTML/PDF
export; recording the secondary slot's steps into the primary's report; R /
Seurat codegen.
