"""In-memory, server-lifetime store for user-defined gene sets.

Gene sets (and their folder/category layout) are maintained on the frontend
Zustand store, but that state lives in the tab — if the user reloads, all
sets they built up in the session are lost. This module mirrors the
frontend's ``geneSetCategories`` dict on the backend process as an opaque
JSON-serialisable blob so the frontend can hydrate from ``GET /api/gene_sets``
on mount and push updates on every mutation.

It's deliberately dumb: the backend doesn't interpret the structure. If the
frontend shape evolves, only the frontend has to understand the new fields
— as long as they round-trip through JSON, this module doesn't care.

Persistence is only for the lifetime of the server process — matches the
existing pattern for drawn lines. Server restart clears the store.
"""
from __future__ import annotations

from threading import Lock
from typing import Any


_state: dict[str, Any] = {}
_lock = Lock()


def get_gene_sets() -> dict[str, Any]:
    """Return a shallow copy of the current gene-set state (possibly empty)."""
    with _lock:
        return dict(_state)


def set_gene_sets(payload: dict[str, Any]) -> None:
    """Replace the entire gene-set state. Payload must be a JSON-safe dict."""
    if not isinstance(payload, dict):
        raise ValueError("gene set payload must be a mapping at the top level")
    with _lock:
        _state.clear()
        _state.update(payload)


def reset() -> None:
    """Clear the store (used by tests / admin flows)."""
    with _lock:
        _state.clear()
