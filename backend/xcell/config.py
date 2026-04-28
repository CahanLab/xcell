"""User-editable configuration for xcell defaults.

Loads a YAML or JSON file at startup and exposes the parsed dict via
``get_user_config()``. The file is intentionally a single source of truth
that both backend and frontend read — the frontend fetches it through
``GET /api/config/defaults`` and merges the values into its modal param
defaults.

Discovery order (first hit wins):
  1. ``$XCELL_CONFIG_PATH`` — explicit override (full path)
  2. ``~/.xcell/config.yaml``
  3. ``~/.xcell/config.yml``
  4. ``~/.xcell/config.json``

Missing file is not an error — defaults then fall back to the hardcoded
values baked into routes / ScanpyModal.

Shape (loose — unknown keys are just ignored by whichever consumer reads
them, so adding a new override never breaks startup):

    scanpy:
      <function_name>:
        <param_name>: <value>
    line_association:
      <param_name>: <value>

Example ~/.xcell/config.yaml::

    scanpy:
      filter_cells:
        min_genes: 15
      filter_genes:
        min_cells: 10
      neighbors:
        n_neighbors: 20
    line_association:
      n_spline_knots: 7
      fdr_threshold: 0.1
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_USER_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None
_CONFIG_ERROR: str | None = None


def _candidate_paths() -> list[Path]:
    override = os.environ.get("XCELL_CONFIG_PATH")
    if override:
        return [Path(override).expanduser()]
    home = Path.home() / ".xcell"
    return [home / "config.yaml", home / "config.yml", home / "config.json"]


def _parse(path: Path) -> dict[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        # PyYAML is an optional import — fall back to JSON if unavailable
        # so users on a minimal install still get config support.
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Cannot parse {path}: PyYAML is not installed. "
                "Install it (`pip install pyyaml`) or use config.json instead."
            ) from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config at {path} must be a mapping at the top level.")
    return data


def load_user_config() -> dict[str, Any]:
    """Load user config. Safe to call more than once; result is cached."""
    global _USER_CONFIG, _CONFIG_PATH, _CONFIG_ERROR
    if _USER_CONFIG is not None:
        return _USER_CONFIG

    for candidate in _candidate_paths():
        if candidate.exists():
            try:
                _USER_CONFIG = _parse(candidate)
                _CONFIG_PATH = candidate
                _CONFIG_ERROR = None
                print(f"[xcell.config] Loaded user config from {candidate}")
                return _USER_CONFIG
            except Exception as e:
                _USER_CONFIG = {}
                _CONFIG_PATH = candidate
                _CONFIG_ERROR = str(e)
                print(f"[xcell.config] Failed to parse {candidate}: {e}")
                return _USER_CONFIG

    _USER_CONFIG = {}
    _CONFIG_PATH = None
    _CONFIG_ERROR = None
    return _USER_CONFIG


def get_user_config() -> dict[str, Any]:
    """Return the loaded user config (possibly empty)."""
    if _USER_CONFIG is None:
        return load_user_config()
    return _USER_CONFIG


def get_config_meta() -> dict[str, Any]:
    """Return metadata about the loaded config (path, parse error)."""
    if _USER_CONFIG is None:
        load_user_config()
    return {
        "path": str(_CONFIG_PATH) if _CONFIG_PATH else None,
        "loaded": _CONFIG_PATH is not None and _CONFIG_ERROR is None,
        "error": _CONFIG_ERROR,
    }


def get_scanpy_param_default(function_name: str, param_name: str, fallback: Any) -> Any:
    """Look up a scanpy modal param default. Returns fallback if not set."""
    cfg = get_user_config()
    try:
        return cfg["scanpy"][function_name][param_name]
    except (KeyError, TypeError):
        return fallback


def get_line_association_default(param_name: str, fallback: Any) -> Any:
    """Look up a line-association default. Returns fallback if not set."""
    cfg = get_user_config()
    try:
        return cfg["line_association"][param_name]
    except (KeyError, TypeError):
        return fallback
