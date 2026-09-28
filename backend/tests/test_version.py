"""xcell's version is written in several manifests; they must agree.

The header shows the frontend's `package.json` version, the backend reports
`xcell.__version__`, and pixi / setuptools each carry their own copy. Bumping
one and forgetting the rest would put a version on screen that the backend
does not claim to be.
"""
import json
import re
from pathlib import Path

import pytest

import xcell
from xcell.main import app

REPO = Path(__file__).resolve().parents[2]


def _toml_version(path: Path) -> str:
    # The first `version = "..."` in the file is the project's own: both
    # manifests declare it in their leading table.
    m = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(), re.MULTILINE)
    assert m, f'no version in {path}'
    return m.group(1)


def test_backend_manifests_agree():
    assert _toml_version(REPO / 'backend' / 'pyproject.toml') == xcell.__version__


def test_fastapi_app_reports_package_version():
    assert app.version == xcell.__version__


@pytest.mark.skipif(not (REPO / 'frontend' / 'package.json').exists(),
                    reason='frontend not present in this checkout')
def test_frontend_version_matches_backend():
    pkg = json.loads((REPO / 'frontend' / 'package.json').read_text())
    assert pkg['version'] == xcell.__version__


@pytest.mark.skipif(not (REPO / 'pixi.toml').exists(), reason='no pixi manifest')
def test_pixi_workspace_version_matches_backend():
    assert _toml_version(REPO / 'pixi.toml') == xcell.__version__
