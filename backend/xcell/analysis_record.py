"""The Analysis Record: what a GUI session did, in a form that can be exported.

Pure data. This module knows nothing about AnnData or the adaptor — it holds the
ordered list of operations a session performed, the user's annotations, and any
figures captured along the way, and it round-trips losslessly through JSON so
the whole thing can ride along in ``uns['xcell_analysis_record']``.

Two modules consume it: ``codegen`` turns each step into prose and Python, and
``notebook_export`` renders the result as a notebook or markdown.
"""
from __future__ import annotations

import datetime
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

# Above this many cells we keep the count but drop the index list. A selection
# is worth carrying — it is the difference between documenting a subset and
# preserving it — but 50k int32 indices is already ~400 KB of JSON per step,
# and .uns has to hold every step.
SELECTION_CAP = 50_000


def sanitize_for_json(value: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    Recorded results come straight from scipy and numpy (``marker_genes`` and
    ``diffexp`` in particular), so NaN and inf turn up routinely. Neither is
    valid JSON, and the record is served over HTTP and written into h5ad, so
    they are converted at the boundary rather than at every consumer.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    # numpy scalars expose .item(); everything else passes through untouched.
    item = getattr(value, 'item', None)
    if callable(item) and hasattr(value, 'dtype'):
        return sanitize_for_json(item())
    return value


def _now() -> str:
    return datetime.datetime.now().isoformat()


@dataclass
class Step:
    """One recorded operation."""

    index: int
    action: str
    params: dict[str, Any]
    result: dict[str, Any]
    timestamp: str
    note: str | None = None
    figure_ids: list[str] = field(default_factory=list)
    # Set only when the operation ran on an active cell selection.
    n_active: int | None = None
    n_total: int | None = None
    selection: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'action': self.action,
            'params': self.params,
            'result': self.result,
            'timestamp': self.timestamp,
            'note': self.note,
            'figure_ids': list(self.figure_ids),
            'n_active': self.n_active,
            'n_total': self.n_total,
            'selection': self.selection,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(
            index=int(d.get('index', 0)),
            action=str(d.get('action', 'unknown')),
            params=d.get('params') or {},
            result=d.get('result') or {},
            timestamp=d.get('timestamp') or '',
            note=d.get('note'),
            figure_ids=list(d.get('figure_ids') or []),
            n_active=d.get('n_active'),
            n_total=d.get('n_total'),
            selection=d.get('selection'),
        )


@dataclass
class Figure:
    """A PNG captured from the UI, base64-encoded (no data: prefix)."""

    id: str
    png_b64: str
    caption: str = ''
    step_index: int | None = None
    timestamp: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'png_b64': self.png_b64,
            'caption': self.caption,
            'step_index': self.step_index,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Figure:
        return cls(
            id=str(d.get('id', '')),
            png_b64=str(d.get('png_b64', '')),
            caption=d.get('caption') or '',
            step_index=d.get('step_index'),
            timestamp=d.get('timestamp') or '',
        )


@dataclass
class AnalysisRecord:
    """Everything needed to write up a session."""

    source: dict[str, Any] = field(default_factory=dict)
    title: str = ''
    abstract: str = ''
    steps: list[Step] = field(default_factory=list)
    figures: dict[str, Figure] = field(default_factory=dict)
    # Index of the first step the exported report should cover. The recorder is
    # always on — the user marks where the story starts, rather than arming a
    # logger they might forget to arm.
    report_start: int = 0

    # --- steps -----------------------------------------------------------

    def add_step(
        self,
        action: str,
        params: dict[str, Any] | None,
        result: dict[str, Any] | None,
        *,
        selection: list[int] | None = None,
        n_total: int | None = None,
    ) -> Step:
        """Append a step. ``selection`` is the active cell subset, if any."""
        n_active = len(selection) if selection is not None else None
        kept = list(selection) if selection is not None and n_active <= SELECTION_CAP else None
        step = Step(
            index=len(self.steps),
            action=action,
            params=sanitize_for_json(params or {}),
            result=sanitize_for_json(result or {}),
            timestamp=_now(),
            n_active=n_active,
            n_total=n_total,
            selection=kept,
        )
        self.steps.append(step)
        return step

    def set_note(self, index: int, note: str | None) -> Step:
        """Set (or clear, with an empty string) a step's user annotation."""
        step = self.steps[index]
        step.note = note or None
        return step

    def mark_start(self) -> int:
        """Treat everything from here on as the report. Discards nothing."""
        self.report_start = len(self.steps)
        return self.report_start

    def steps_for_report(self) -> list[Step]:
        return self.steps[self.report_start:]

    def clear(self) -> None:
        """Drop the history, keeping the load step so the export still opens
        the right file."""
        self.steps = [s for s in self.steps[:1] if s.action == 'load_dataset']
        for i, s in enumerate(self.steps):
            s.index = i
            s.figure_ids = []
        self.figures = {}
        self.report_start = 0

    # --- figures ---------------------------------------------------------

    def add_figure(
        self,
        png_b64: str,
        *,
        caption: str = '',
        step_index: int | None = None,
    ) -> Figure:
        """Attach a captured PNG. Defaults to the most recent step; with no
        steps yet it stands alone and renders at the end of the report."""
        if step_index is None and self.steps:
            step_index = len(self.steps) - 1
        if step_index is not None and not (0 <= step_index < len(self.steps)):
            raise IndexError(f"No step at index {step_index}")
        fig = Figure(
            id=f"fig_{uuid.uuid4().hex[:12]}",
            png_b64=png_b64,
            caption=caption,
            step_index=step_index,
            timestamp=_now(),
        )
        self.figures[fig.id] = fig
        if step_index is not None:
            self.steps[step_index].figure_ids.append(fig.id)
        return fig

    def remove_figure(self, figure_id: str) -> None:
        fig = self.figures.pop(figure_id, None)
        if fig is None:
            raise KeyError(f"No figure {figure_id}")
        for step in self.steps:
            if figure_id in step.figure_ids:
                step.figure_ids.remove(figure_id)

    def standalone_figures(self) -> list[Figure]:
        return [f for f in self.figures.values() if f.step_index is None]

    def figures_for(self, step: Step) -> list[Figure]:
        return [self.figures[i] for i in step.figure_ids if i in self.figures]

    # --- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': 1,
            'source': self.source,
            'title': self.title,
            'abstract': self.abstract,
            'report_start': self.report_start,
            'steps': [s.to_dict() for s in self.steps],
            'figures': {k: v.to_dict() for k, v in self.figures.items()},
        }

    @classmethod
    def from_dict(cls, d: Any) -> AnalysisRecord:
        if not isinstance(d, dict):
            raise ValueError('An analysis record must be a JSON object')
        return cls(
            source=d.get('source') or {},
            title=d.get('title') or '',
            abstract=d.get('abstract') or '',
            report_start=int(d.get('report_start') or 0),
            steps=[Step.from_dict(s) for s in (d.get('steps') or [])],
            figures={
                k: Figure.from_dict(v) for k, v in (d.get('figures') or {}).items()
            },
        )
