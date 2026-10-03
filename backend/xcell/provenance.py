"""Which xcell produced a result: the version, and the commit in a checkout.

A version alone says little between releases (0.1.0 covered eight months of
main), so a git checkout adds its commit and whether it had uncommitted
changes. Read once per process: the dev backend restarts on every backend
edit, so the commit stays current; a frontend-only edit does not restart it.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Any

from xcell import __version__

# The repository root, when this file is in a checkout of it.
_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(['git', *args], cwd=_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    """``{'version', 'commit', 'dirty', 'tag'}``; the last three are None
    outside a git checkout of xcell."""
    info: dict[str, Any] = {'version': __version__, 'commit': None, 'dirty': None, 'tag': None}
    # Only xcell's own checkout: an installed copy can sit inside some
    # unrelated repository, whose commit would be a lie.
    if not ((_ROOT / '.git').exists() and (_ROOT / 'backend' / 'xcell').is_dir()):
        return info
    commit = _git('rev-parse', '--short=7', 'HEAD')
    if not commit:
        return info
    status = _git('status', '--porcelain', '--untracked-files=no')
    info.update(
        commit=commit,
        dirty=None if status is None else bool(status),
        tag=_git('describe', '--tags', '--exact-match', 'HEAD'),
    )
    return info


def build_label(info: dict[str, Any] | None = None) -> str:
    """``0.2.0`` at a clean release tag (or outside a checkout), otherwise
    ``0.2.0+g<commit>``, with ``.dirty`` for uncommitted changes."""
    info = info or build_info()
    version = info['version']
    if not info.get('commit'):
        return version
    if info.get('tag') == f'v{version}' and not info.get('dirty'):
        return version
    return f"{version}+g{info['commit']}" + ('.dirty' if info.get('dirty') else '')
