"""User-editable configuration for xcell defaults.

Loads a YAML or JSON file at startup and exposes the parsed dict via
``get_user_config()``. The config is intentionally a single source of truth
that both backend and frontend read — the frontend fetches it through
``GET /api/config/defaults`` and merges the values into its modal param
defaults and display preferences.

Two layers are merged (deep, key-by-key; the override wins at each leaf):

  1. Base — ``xcell/config.yaml`` shipped with the repo. This gives every
     UI-adjustable parameter a default, so a fresh checkout already reflects
     the project's chosen defaults with no per-user setup.
  2. Override — the first user config found, deep-merged over the base:
       a. ``$XCELL_CONFIG_PATH`` — explicit path (wins over the home files)
       b. ``~/.xcell/config.yaml``
       c. ``~/.xcell/config.yml``
       d. ``~/.xcell/config.json``

Because it is a deep merge, a user's file only needs the keys they want to
change — every other key still comes from the shipped base. Missing user
config is not an error (the base defaults apply); a missing base file is not
an error either (consumers fall back to hardcoded defaults). Unknown keys are
ignored by whichever consumer reads them, so adding a new override never
breaks startup.

Shape (loose):

    scanpy:
      <function_name>:
        <param_name>: <value>
    display:
      <param_name>: <value>
    line_association:
      <param_name>: <value>

Example ~/.xcell/config.yaml (override only the keys you care about)::

    scanpy:
      filter_cells:
        min_genes: 15
    display:
      point_size: 2
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_USER_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None
_CONFIG_ERROR: str | None = None


def _repo_default_path() -> Path:
    """The config.yaml shipped alongside this module (the base defaults)."""
    return Path(__file__).parent / "config.yaml"


def _candidate_paths() -> list[Path]:
    """User override locations, highest precedence first (first existing wins)."""
    override = os.environ.get("XCELL_CONFIG_PATH")
    if override:
        return [Path(override).expanduser()]
    home = Path.home() / ".xcell"
    return [home / "config.yaml", home / "config.yml", home / "config.json"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Nested dicts merge key-by-key; any non-dict value in ``override`` (including
    lists and ``None``) replaces the corresponding base value outright.
    """
    result: dict[str, Any] = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


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
    """Load config (shipped base deep-merged with the user override).

    Safe to call more than once; result is cached.
    """
    global _USER_CONFIG, _CONFIG_PATH, _CONFIG_ERROR
    if _USER_CONFIG is not None:
        return _USER_CONFIG

    # Base layer: the config.yaml shipped with the repo. Always tried first so
    # every UI-adjustable parameter has a default even with no user config.
    base: dict[str, Any] = {}
    base_error: str | None = None
    base_path = _repo_default_path()
    if base_path.exists():
        try:
            base = _parse(base_path)
            print(f"[xcell.config] Loaded shipped defaults from {base_path}")
        except Exception as e:
            base_error = str(e)
            print(f"[xcell.config] Failed to parse shipped defaults {base_path}: {e}")

    # Override layer: the first user config found, deep-merged over the base so a
    # partial user file overrides individual keys without dropping shipped defaults.
    override: dict[str, Any] = {}
    override_path: Path | None = None
    override_error: str | None = None
    for candidate in _candidate_paths():
        if candidate.exists():
            override_path = candidate
            try:
                override = _parse(candidate)
                print(f"[xcell.config] Loaded user config from {candidate}")
            except Exception as e:
                override_error = str(e)
                print(f"[xcell.config] Failed to parse {candidate}: {e}")
            break

    _USER_CONFIG = _deep_merge(base, override)
    # Surface the user override path/error when present, else the shipped base.
    _CONFIG_PATH = override_path or (base_path if base_path.exists() else None)
    _CONFIG_ERROR = override_error or base_error
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
