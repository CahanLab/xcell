"""Rendering an :class:`AnalysisRecord` as a Jupyter notebook or markdown.

Pure: record in, notebook dict or markdown string out. Writing files is the
route's job — keeping I/O out of here is what makes the document itself
testable.

Both formats carry the same content and, importantly, the same reproducibility
header: a reader is told up front how many steps re-run as written, how many
need xcell, how many are manual, and how many ran on a cell selection rather
than the whole dataset.
"""
from __future__ import annotations

import base64
import binascii
import datetime
from typing import Any

from xcell.analysis_record import AnalysisRecord, Figure, Step
from xcell.codegen import EXACT, MANUAL, XCELL, TranslatedStep, translate

SELECTIONS_SUFFIX = '_selections.json'

# Prelude for xcell-tier steps: wrap the in-memory AnnData in an adaptor (the
# `adata=` kwarg skips disk I/O) and provide a runner for xcell's two-phase
# prepare_*/apply operations.
_XCELL_PRELUDE = [
    'import inspect',
    'from xcell.adaptor import DataAdaptor',
    '',
    'xa = DataAdaptor(SOURCE_PATH, adata=adata)',
    '',
    '',
    'def _xcell_run(prepared):',
    '    """Drive one of xcell\'s two-phase (prepare_*) operations to completion."""',
    '    compute, apply = prepared',
    '    # A compute_fn taking one argument is opting into progress reporting.',
    '    needs_report = bool(inspect.signature(compute).parameters)',
    '    return apply(compute(lambda *a, **k: None) if needs_report else compute())',
]


# --- assembling the translated report -------------------------------------

def _translated(record: AnalysisRecord) -> list[tuple[Step, TranslatedStep]]:
    """The steps the report actually narrates.

    Loads are excluded: the first one becomes the setup cell, and a later one is
    a session artifact (the user exported an h5ad and re-opened it) rather than
    an analysis step. Counting them would make the header disagree with the
    numbered steps a reader can see.
    """
    return [
        (s, translate(s))
        for s in record.steps_for_report()
        if s.action != 'load_dataset'
    ]


def _load_step(record: AnalysisRecord) -> tuple[Step, TranslatedStep] | None:
    """The load, wherever it sits.

    It is normally step 0 and so may fall outside a marked report span — but the
    notebook still has to open the right file, so it is looked up separately.
    """
    for step in record.steps:
        if step.action == 'load_dataset':
            return step, translate(step)
    return None


def _source_path(record: AnalysisRecord) -> str | None:
    load = _load_step(record)
    if load and load[0].params.get('kind') != 'memory':
        return load[0].params.get('path')
    return record.source.get('path') or None


def _setup_lines(record: AnalysisRecord, pairs: list[tuple[Step, TranslatedStep]]) -> list[str]:
    """The opening code cell: imports, the source file, and any prelude the
    body actually needs."""
    load = _load_step(record)
    imports: set[str] = set()
    for _, t in pairs:
        imports.update(t.imports)
    if load:
        imports.update(load[1].imports)
    # The adaptor prelude brings its own import, and is emitted in full below.
    imports.discard('from xcell.adaptor import DataAdaptor')

    lines = sorted(imports)
    path = _source_path(record)
    if path:
        lines += ['', f'SOURCE_PATH = {path!r}']

    if load and load[1].code:
        # The step's own code carries the path as a literal so it stands alone
        # in the UI; in the notebook it reads better through the constant, and
        # one line then re-points the whole document.
        lines += ['']
        lines += [line.replace(repr(path), 'SOURCE_PATH') if path else line
                  for line in load[1].code]
    elif path:
        lines += ['', 'adata = sc.read_h5ad(SOURCE_PATH)'] if 'import scanpy as sc' in lines \
            else ['', '# Load the dataset here — the original source was not recorded.']
    else:
        lines += ['', '# The source dataset was built in memory and cannot be re-read.',
                  '# Assign it to `adata` before running the steps below.']

    if any(s.selection for s, _ in pairs):
        lines += [
            '',
            'import json',
            '',
            '# Cell selections xcell recorded, so subset steps can be reproduced.',
            f'with open(NOTEBOOK_NAME + {SELECTIONS_SUFFIX!r}) as fh:',
            '    SELECTIONS = json.load(fh)',
        ]

    if any(t.fidelity == XCELL for _, t in pairs):
        if not path:
            lines += ['', "SOURCE_PATH = 'in-memory'  # xcell steps need a path argument"]
        lines += [''] + _XCELL_PRELUDE

    return lines


def _counts(pairs: list[tuple[Step, TranslatedStep]]) -> dict[str, int]:
    return {
        'total': len(pairs),
        EXACT: sum(1 for _, t in pairs if t.fidelity == EXACT),
        XCELL: sum(1 for _, t in pairs if t.fidelity == XCELL),
        MANUAL: sum(1 for _, t in pairs if t.fidelity == MANUAL),
        'subset': sum(1 for s, _ in pairs if s.n_active is not None),
    }


def report_counts(record: AnalysisRecord) -> dict[str, int]:
    """How much of the report re-runs — the number the UI and the header share."""
    return _counts(_translated(record))


def _header(record: AnalysisRecord, pairs: list[tuple[Step, TranslatedStep]]) -> str:
    c = _counts(pairs)
    title = record.title.strip() or 'xcell analysis record'
    out = [f'# {title}', '']
    if record.abstract.strip():
        out += [record.abstract.strip(), '']

    src = record.source
    where = src.get('path') or 'an in-memory dataset'
    shape = ''
    if src.get('n_cells') is not None and src.get('n_genes') is not None:
        shape = f" ({src['n_cells']:,} cells x {src['n_genes']:,} genes)"
    stamp = datetime.date.today().isoformat()

    out += [
        f'> Recorded by [xcell](https://github.com/CahanLab/xcell) on {stamp} '
        f'from `{where}`{shape}.',
        '>',
        f'> **{c["total"]} step{"" if c["total"] == 1 else "s"}.** '
        f'{c[EXACT]} re-run as written; '
        f'{c[XCELL]} need the xcell Python API; '
        f'{c[MANUAL]} are manual and are described but not executed.',
    ]
    if c['subset']:
        out += [
            '>',
            f'> **{c["subset"]} step{"" if c["subset"] == 1 else "s"} ran on a cell '
            f'selection** rather than the whole dataset. The code below runs on '
            f'everything; each affected step is flagged.',
        ]
    if c[XCELL]:
        out += [
            '>',
            '> xcell steps are reproduced by calling xcell\'s own API with the '
            'parameters as recorded.',
        ]
    return '\n'.join(out)


def _step_markdown(
    record: AnalysisRecord, n: int, step: Step, t: TranslatedStep, *, include_figures: bool,
) -> str:
    badge = {EXACT: '', XCELL: ' — *xcell API*', MANUAL: ' — *not reproduced in code*'}[t.fidelity]
    out = [f'## {n}. {t.title}{badge}', '', t.summary]
    for warning in t.warnings:
        out += ['', f'> ⚠️ {warning}']
    if step.note:
        out += ['', step.note.strip()]
    if include_figures:
        for fig in record.figures_for(step):
            if fig.caption:
                out += ['', f'*{fig.caption}*']
    return '\n'.join(out)


# --- figures --------------------------------------------------------------

def figure_filename(figure: Figure) -> str:
    return f'{figure.id}.png'


def figure_payloads(record: AnalysisRecord) -> dict[str, bytes]:
    """Decoded PNG bytes keyed by filename, for a caller that writes files.

    A figure that won't decode is dropped rather than allowed to abort an
    export the user has been assembling all session.
    """
    out: dict[str, bytes] = {}
    for fig in record.figures.values():
        try:
            out[figure_filename(fig)] = base64.b64decode(fig.png_b64, validate=True)
        except (binascii.Error, ValueError):
            continue
    return out


def _is_decodable(figure: Figure) -> bool:
    try:
        base64.b64decode(figure.png_b64, validate=True)
        return True
    except (binascii.Error, ValueError):
        return False


def selections_payload(record: AnalysisRecord) -> dict[str, list[int]]:
    """The cell selections worth preserving, keyed as the generated code
    expects."""
    return {
        f'step_{s.index}': s.selection
        for s in record.steps_for_report()
        if s.selection
    }


# --- notebook -------------------------------------------------------------

def _md_cell(text: str) -> dict[str, Any]:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': _split(text)}


def _code_cell(lines: list[str], outputs: list[dict] | None = None) -> dict[str, Any]:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': outputs or [],
        'source': _split('\n'.join(lines)),
    }


def _split(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline
    except the last."""
    lines = text.split('\n')
    return [f'{line}\n' for line in lines[:-1]] + [lines[-1]]


def _figure_outputs(record: AnalysisRecord, step: Step) -> list[dict[str, Any]]:
    outputs = []
    for fig in record.figures_for(step):
        if not _is_decodable(fig):
            continue
        outputs.append({
            'output_type': 'display_data',
            'metadata': {},
            'data': {
                'image/png': fig.png_b64,
                'text/plain': [fig.caption or '<figure>'],
            },
        })
    return outputs


def to_notebook(
    record: AnalysisRecord,
    *,
    include_figures: bool = True,
    include_code: bool = True,
    notebook_name: str = 'analysis',
) -> dict[str, Any]:
    """Render the record as an nbformat-4 notebook dict."""
    pairs = _translated(record)
    cells: list[dict[str, Any]] = [_md_cell(_header(record, pairs))]

    if include_code:
        setup = _setup_lines(record, pairs)
        cells.append(_code_cell([f'NOTEBOOK_NAME = {notebook_name!r}', ''] + setup))

    for n, (step, t) in enumerate(pairs, start=1):
        cells.append(_md_cell(_step_markdown(record, n, step, t,
                                             include_figures=include_figures)))
        outputs = _figure_outputs(record, step) if include_figures else []
        if include_code and t.code:
            cells.append(_code_cell(t.code, outputs))
        elif outputs:
            # A manual step has no code cell to hang an output on; an empty
            # cell keeps the figure attached to its step.
            cells.append(_code_cell(['# (figure captured at this point)'], outputs))

    standalone = [f for f in record.standalone_figures() if _is_decodable(f)]
    if include_figures and standalone:
        cells.append(_md_cell('## Figures'))
        for fig in standalone:
            cells.append(_md_cell(f'*{fig.caption}*' if fig.caption else '&nbsp;'))
            cells.append(_code_cell(['# (captured figure)'], [{
                'output_type': 'display_data',
                'metadata': {},
                'data': {'image/png': fig.png_b64,
                         'text/plain': [fig.caption or '<figure>']},
            }]))

    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python'},
            'xcell': {'generated': datetime.datetime.now().isoformat()},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


# --- markdown -------------------------------------------------------------

def to_markdown(
    record: AnalysisRecord,
    *,
    include_figures: bool = True,
    include_code: bool = True,
    figure_dir: str | None = None,
    notebook_name: str = 'analysis',
) -> str:
    """Render the record as a markdown document.

    Figures are linked into ``figure_dir`` (a path relative to the document);
    with no directory there is nowhere to point, so they are described but not
    shown.
    """
    pairs = _translated(record)
    show_figures = include_figures and figure_dir is not None
    parts = [_header(record, pairs)]

    if include_code:
        setup = _setup_lines(record, pairs)
        parts.append('## Setup\n\n```python\n' +
                     '\n'.join([f'NOTEBOOK_NAME = {notebook_name!r}', ''] + setup) +
                     '\n```')

    for n, (step, t) in enumerate(pairs, start=1):
        parts.append(_step_markdown(record, n, step, t, include_figures=include_figures))
        if include_code and t.code:
            parts.append('```python\n' + '\n'.join(t.code) + '\n```')
        if show_figures:
            for fig in record.figures_for(step):
                if _is_decodable(fig):
                    parts.append(f'![{fig.caption}]({figure_dir}/{figure_filename(fig)})')

    standalone = [f for f in record.standalone_figures() if _is_decodable(f)]
    if include_figures and standalone:
        parts.append('## Figures')
        for fig in standalone:
            if show_figures:
                parts.append(f'![{fig.caption}]({figure_dir}/{figure_filename(fig)})')
            elif fig.caption:
                parts.append(f'*{fig.caption}*')

    return '\n\n'.join(parts) + '\n'
