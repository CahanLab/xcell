"""The release script: one version in five files, a changelog that rolls, notes
that fit on a GitHub release.

The script lives in scripts/ (repo tooling, not part of the xcell package) and
uses only the standard library, so the release workflow can run it on a bare
runner. These tests run it against copies of the real manifests, so a format
change in any of them shows up here rather than half-way through a release.
"""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('release', REPO / 'scripts' / 'release.py')
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

URL = 'https://github.com/CahanLab/xcell'


def _repo(tmp_path, version=None):
    """The five version-carrying files, copied from the real repo."""
    for rel in release.VERSION_FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    if version is not None:
        release.write_version(tmp_path, version)
    return tmp_path


# --- versions ---------------------------------------------------------------

def test_the_real_repo_agrees_on_one_version():
    versions = release.read_versions(REPO)
    assert set(versions) == set(release.VERSION_FILES)
    assert len(set(versions.values())) == 1


def test_bump_writes_every_file_and_only_the_version(tmp_path):
    root = _repo(tmp_path, '0.1.0')
    before = {rel: (root / rel).read_text() for rel in release.VERSION_FILES}
    release.write_version(root, '0.2.0')
    assert set(release.read_versions(root).values()) == {'0.2.0'}
    for rel, text in before.items():
        after = (root / rel).read_text()
        changed = [(a, b) for a, b in zip(text.splitlines(), after.splitlines()) if a != b]
        # The lockfile carries the version twice (its root and packages[""]).
        assert len(changed) == (2 if rel.endswith('package-lock.json') else 1), rel
        assert all('0.1.0' in a and '0.2.0' in b for a, b in changed)
        assert len(text.splitlines()) == len(after.splitlines())


def test_the_lockfile_keeps_npm_formatting(tmp_path):
    root = _repo(tmp_path, '0.1.0')
    release.write_version(root, '0.2.0')
    text = (root / 'frontend' / 'package-lock.json').read_text()
    assert text.endswith('}\n')
    data = json.loads(text)
    assert data['version'] == data['packages']['']['version'] == '0.2.0'


def test_disagreeing_files_are_refused(tmp_path):
    root = _repo(tmp_path, '0.1.0')
    pkg = root / 'frontend' / 'package.json'
    pkg.write_text(pkg.read_text().replace('"version": "0.1.0"', '"version": "0.1.1"', 1))
    with pytest.raises(release.ReleaseError, match='disagree'):
        release.current_version(root)


@pytest.mark.parametrize('bad', ['0.2', 'v0.2.0', '0.2.0rc1', '0.2.0-rc.1', '1.0.0.0', ''])
def test_only_plain_major_minor_patch_is_accepted(bad):
    # PEP 440 and npm spell pre-releases differently, and the manifests must
    # hold the same string, so pre-releases are out until they are needed.
    with pytest.raises(release.ReleaseError):
        release.parse_version(bad)


def test_a_version_must_move_forward():
    release.check_newer('0.1.0', '0.2.0')
    release.check_newer('0.9.0', '0.10.0')  # numeric, not string, order
    with pytest.raises(release.ReleaseError, match='newer'):
        release.check_newer('0.2.0', '0.2.0')
    with pytest.raises(release.ReleaseError, match='newer'):
        release.check_newer('0.10.0', '0.9.0')


def test_check_tag(tmp_path):
    root = _repo(tmp_path, '0.2.0')
    release.check_tag(root, 'v0.2.0')
    with pytest.raises(release.ReleaseError, match='0.2.0'):
        release.check_tag(root, 'v0.2.1')
    with pytest.raises(release.ReleaseError):
        release.check_tag(root, '0.2.0')  # tags carry the v


# --- changelog --------------------------------------------------------------

CHANGELOG = """# Changelog

All notable changes to xcell are documented in this file.

## [Unreleased]

### Added
- **A feature.** It does things.

### Fixed
- A bug.
"""


def test_rolling_the_changelog_opens_a_dated_section_under_a_fresh_unreleased():
    out = release.roll_changelog(CHANGELOG, '0.2.0', '2026-10-03', URL)
    assert '## [Unreleased]\n\n## [0.2.0] - 2026-10-03\n\n### Added\n- **A feature.**' in out
    assert out.index('## [Unreleased]') < out.index('## [0.2.0]')
    assert out.endswith(
        f'[Unreleased]: {URL}/compare/v0.2.0...HEAD\n[0.2.0]: {URL}/releases/tag/v0.2.0\n')


def test_a_second_release_compares_against_the_first():
    first = release.roll_changelog(CHANGELOG, '0.2.0', '2026-10-03', URL)
    second = first.replace('## [Unreleased]\n', '## [Unreleased]\n\n### Fixed\n- Another bug.\n', 1)
    out = release.roll_changelog(second, '0.3.0', '2026-11-01', URL)
    assert out.index('## [0.3.0] - 2026-11-01') < out.index('## [0.2.0] - 2026-10-03')
    links = out[out.index(f'[Unreleased]: {URL}'):]
    assert links == (f'[Unreleased]: {URL}/compare/v0.3.0...HEAD\n'
                     f'[0.3.0]: {URL}/compare/v0.2.0...v0.3.0\n'
                     f'[0.2.0]: {URL}/releases/tag/v0.2.0\n')


def test_an_empty_unreleased_section_is_refused():
    rolled = release.roll_changelog(CHANGELOG, '0.2.0', '2026-10-03', URL)
    with pytest.raises(release.ReleaseError, match='nothing under'):
        release.roll_changelog(rolled, '0.3.0', '2026-11-01', URL)


def test_a_version_already_in_the_changelog_is_refused():
    rolled = release.roll_changelog(CHANGELOG, '0.2.0', '2026-10-03', URL)
    again = rolled.replace('## [Unreleased]\n', '## [Unreleased]\n\n- More.\n', 1)
    with pytest.raises(release.ReleaseError, match='already'):
        release.roll_changelog(again, '0.2.0', '2026-10-04', URL)


def test_notes_are_the_section_without_its_heading_or_the_links():
    rolled = release.roll_changelog(CHANGELOG, '0.2.0', '2026-10-03', URL)
    notes = release.release_notes(rolled, '0.2.0')
    assert notes == '### Added\n- **A feature.** It does things.\n\n### Fixed\n- A bug.\n'
    with pytest.raises(release.ReleaseError, match='0.9.0'):
        release.release_notes(rolled, '0.9.0')


def test_notes_too_long_for_github_end_at_a_whole_entry_with_a_link():
    bullets = ''.join(f'- Entry {i}: ' + 'x' * 80 + '\n' for i in range(200))
    text = CHANGELOG.replace('- A bug.\n', bullets)
    rolled = release.roll_changelog(text, '0.2.0', '2026-10-03', URL)
    notes = release.release_notes(rolled, '0.2.0', limit=2000,
                                  full_url=f'{URL}/blob/v0.2.0/CHANGELOG.md')
    assert len(notes) <= 2000
    body, tail = notes.rsplit('\n\n', 1)
    assert body.splitlines()[-1].startswith('- Entry ') and body.endswith('x')
    assert f'{URL}/blob/v0.2.0/CHANGELOG.md' in tail


def test_the_real_changelog_rolls_and_its_notes_fit_a_github_release():
    text = (REPO / 'CHANGELOG.md').read_text()
    rolled = release.roll_changelog(text, '9.9.9', '2026-10-03', URL)
    notes = release.release_notes(rolled, '9.9.9', limit=release.GITHUB_NOTES_LIMIT,
                                  full_url=f'{URL}/blob/v9.9.9/CHANGELOG.md')
    assert 0 < len(notes) <= release.GITHUB_NOTES_LIMIT


# --- prepare: the local half of a release -----------------------------------

def _git_repo(tmp_path):
    import subprocess
    root = _repo(tmp_path, '0.1.0')
    (root / 'CHANGELOG.md').write_text(CHANGELOG)

    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '-q', '-b', 'main')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    git('add', '-A')
    git('commit', '-q', '-m', 'init')
    return root, git


def test_prepare_commits_and_tags_without_pushing(tmp_path):
    root, git = _git_repo(tmp_path)
    assert release.prepare('0.2.0', root=root, run_tests=False, date='2026-10-03') == 'v0.2.0'
    assert release.current_version(root) == '0.2.0'
    assert '## [0.2.0] - 2026-10-03' in (root / 'CHANGELOG.md').read_text()
    assert git('log', '-1', '--format=%s') == 'release: v0.2.0'
    assert git('cat-file', '-t', 'v0.2.0') == 'tag'  # annotated, so describe sees it
    assert git('rev-parse', 'v0.2.0^{commit}') == git('rev-parse', 'HEAD')
    assert git('status', '--porcelain') == ''  # every changed file was committed
    assert git('remote') == ''


def test_prepare_refuses_and_changes_nothing(tmp_path):
    root, git = _git_repo(tmp_path)
    head = git('rev-parse', 'HEAD')

    with pytest.raises(release.ReleaseError, match='newer'):
        release.prepare('0.1.0', root=root, run_tests=False)
    git('checkout', '-q', '-b', 'feature')
    with pytest.raises(release.ReleaseError, match='main'):
        release.prepare('0.2.0', root=root, run_tests=False)
    git('checkout', '-q', 'main')
    (root / 'pixi.toml').write_text((root / 'pixi.toml').read_text() + '\n# edit\n')
    with pytest.raises(release.ReleaseError, match='uncommitted'):
        release.prepare('0.2.0', root=root, run_tests=False)
    git('checkout', '--', 'pixi.toml')
    git('tag', 'v0.2.0')
    with pytest.raises(release.ReleaseError, match='already exists'):
        release.prepare('0.2.0', root=root, run_tests=False)

    assert git('rev-parse', 'HEAD') == head
    assert release.current_version(root) == '0.1.0'
