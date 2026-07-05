"""Discovery and parsing of gene-set *bundles* shipped with xcell.

Curated gene-set collections live in ``xcell/data/gene_sets/*.json`` and ship
with the package. They are a read-only **library**: the frontend lists them
(in the Import dialog) and the user loads a bundle into their own editable
gene sets on demand. Bundles are never auto-injected into the user's workspace
and never touch the mutable ``gene_set_store``.

Two file shapes are accepted (both round-trip with the Import dialog, so a
bundle file can also just be dragged in):

  * bare array   ``[ {"name": ..., "genes": [...], "folder": ...?}, ... ]``
  * wrapped      ``{"name": ..., "description": ...?, "sets": [ ... ]}``

Per-set fields:
  ``name``       (required) display name of the set
  ``genes``      (required) UP / positive symbols (synonyms: ``up``, ``positive``)
  ``genesDown``  (optional) DOWN / negative symbols (synonyms: ``down``, ``negative``)
  ``folder``     (optional) sub-group; sets sharing a folder load together

To add a standard collection, drop a ``.json`` file in the directory — no code
change needed. A malformed file is skipped with a warning rather than failing
the whole endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LIBRARY_DIR = Path(__file__).parent / "data" / "gene_sets"


def _as_symbols(val: Any) -> list[str]:
    """Coerce a gene list into cleaned string symbols. Accepts plain strings
    or ``{"symbol": ...}`` objects (the .gsb.json shape)."""
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for g in val:
        if isinstance(g, str):
            s = g.strip()
            if s:
                out.append(s)
        elif isinstance(g, dict) and isinstance(g.get("symbol"), str):
            s = g["symbol"].strip()
            if s:
                out.append(s)
    return out


def _parse_set(item: Any) -> dict[str, Any] | None:
    """Normalise one raw set into ``{name, genes, genesDown?, folder?}`` or None."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    genes = _as_symbols(item.get("genes") or item.get("up") or item.get("positive"))
    down = _as_symbols(item.get("genesDown") or item.get("down") or item.get("negative"))
    if not genes and not down:
        return None
    out: dict[str, Any] = {"name": name.strip(), "genes": genes}
    if down:
        out["genesDown"] = down
    folder = item.get("folder")
    if isinstance(folder, str) and folder.strip():
        out["folder"] = folder.strip()
    return out


def _parse_bundle(path: Path) -> dict[str, Any] | None:
    """Parse one bundle file into a normalised, frontend-agnostic dict."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"[xcell.gene_set_library] Skipping {path.name}: {e}")
        return None

    name = path.stem
    description = ""
    if isinstance(data, dict):
        raw_sets = data.get("sets")
        if isinstance(data.get("name"), str) and data["name"].strip():
            name = data["name"].strip()
        if isinstance(data.get("description"), str):
            description = data["description"].strip()
    elif isinstance(data, list):
        raw_sets = data
    else:
        raw_sets = None

    if not isinstance(raw_sets, list):
        print(f"[xcell.gene_set_library] Skipping {path.name}: no gene sets found")
        return None

    sets = [s for s in (_parse_set(x) for x in raw_sets) if s]
    if not sets:
        print(f"[xcell.gene_set_library] Skipping {path.name}: no valid gene sets")
        return None

    return {
        "id": path.stem,
        "name": name,
        "description": description,
        "count": len(sets),
        "sets": sets,
    }


def list_bundles() -> list[dict[str, Any]]:
    """Return all shipped gene-set bundles, sorted by display name."""
    if not LIBRARY_DIR.is_dir():
        return []
    bundles: list[dict[str, Any]] = []
    for path in sorted(LIBRARY_DIR.glob("*.json")):
        bundle = _parse_bundle(path)
        if bundle:
            bundles.append(bundle)
    bundles.sort(key=lambda b: b["name"].lower())
    return bundles
