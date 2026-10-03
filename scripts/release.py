#!/usr/bin/env python3
"""Cut an xcell release: one version in five files, a dated changelog, a tag.

    pixi run version             # the version, and a check that the files agree
    pixi run release 0.2.0       # check, test, bump, commit and tag -- locally
    git push origin main v0.2.0  # publish: .github/workflows/release.yml then
                                 # creates the GitHub release from the changelog

``prepare`` (what ``pixi run release`` runs) stops at a local commit and tag,
so nothing leaves the machine until the push; ``git tag -d v0.2.0`` and
``git reset --hard HEAD~1`` undo it. The other subcommands are its parts:
``bump`` rewrites the files, ``notes`` prints a version's changelog section,
``check-tag`` is the workflow's guard that a tag matches the files.

Standard library only: the release workflow runs this on a bare runner.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Every file that states xcell's version; backend/tests/test_version.py checks
#: they agree, and ``bump`` rewrites them together.
VERSION_FILES = (
    'backend/xcell/__init__.py',
    'backend/pyproject.toml',
    'pixi.toml',
    'frontend/package.json',
    'frontend/package-lock.json',
)

# GitHub refuses a release body over 125,000 characters; the margin covers the
# difference between Python's count and GitHub's.
GITHUB_NOTES_LIMIT = 120_000

FALLBACK_URL = 'https://github.com/CahanLab/xcell'

_VERSION = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
# The first match in each file is the project's own version: both TOML
# manifests declare it in their leading table, package.json near the top.
_PATTERNS = {
    '.py': re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.M),
    '.toml': re.compile(r'^(version\s*=\s*")([^"]+)(")', re.M),
    '.json': re.compile(r'^(\s*"version":\s*")([^"]+)(")', re.M),
}
_HEADING = re.compile(r'^## \[([^\]]+)\]', re.M)
_LINK = re.compile(r'^\[[^\]]+\]:\s*\S+\s*$')


class ReleaseError(Exception):
    """A release cannot go ahead; the message says why and what to do."""


# --- versions ---------------------------------------------------------------

def parse_version(text: str) -> tuple[int, int, int]:
    # Plain MAJOR.MINOR.PATCH only: PEP 440 and npm spell pre-releases
    # differently (0.2.0rc1 / 0.2.0-rc.1), and every manifest must hold the
    # same string.
    m = _VERSION.match(text or '')
    if not m:
        raise ReleaseError(f"'{text}' is not a version: use MAJOR.MINOR.PATCH, e.g. 0.2.0.")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def check_newer(old: str, new: str) -> None:
    if parse_version(new) <= parse_version(old):
        raise ReleaseError(f'{new} is not newer than the current version, {old}.')


def _is_lockfile(rel: str) -> bool:
    return rel.endswith('package-lock.json')


def _read(root: Path, rel: str) -> str:
    text = (root / rel).read_text(encoding='utf-8')
    if _is_lockfile(rel):
        data = json.loads(text)
        top, pkg = data.get('version'), data.get('packages', {}).get('', {}).get('version')
        if top != pkg:
            raise ReleaseError(f"{rel}: its version ({top}) and packages[''] version ({pkg}) disagree.")
        return top
    m = _PATTERNS[Path(rel).suffix].search(text)
    if not m:
        raise ReleaseError(f'No version found in {rel}.')
    return m.group(2)


def read_versions(root: Path = REPO) -> dict[str, str]:
    return {rel: _read(root, rel) for rel in VERSION_FILES}


def current_version(root: Path = REPO) -> str:
    versions = read_versions(root)
    if len(set(versions.values())) != 1:
        listing = ', '.join(f'{rel} {v}' for rel, v in versions.items())
        raise ReleaseError(f'The version files disagree: {listing}.')
    return next(iter(versions.values()))


def write_version(root: Path, version: str) -> None:
    parse_version(version)
    for rel in VERSION_FILES:
        path = root / rel
        text = path.read_text(encoding='utf-8')
        if _is_lockfile(rel):
            # npm writes JSON.stringify(lock, null, 2) plus a newline, which
            # json.dumps reproduces byte for byte (test_release pins it).
            data = json.loads(text)
            data['version'] = version
            data['packages']['']['version'] = version
            new = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
        else:
            new, n = _PATTERNS[path.suffix].subn(rf'\g<1>{version}\g<3>', text, count=1)
            if n != 1:
                raise ReleaseError(f'No version found in {rel}.')
        path.write_text(new, encoding='utf-8')


def check_tag(root: Path, tag: str) -> str:
    """The version a ``v``-prefixed tag names, if it is the one in the files."""
    if not tag.startswith('v'):
        raise ReleaseError(f"Release tags look like v0.2.0; got '{tag}'.")
    version = tag[1:]
    parse_version(version)
    current = current_version(root)
    if current != version:
        raise ReleaseError(f'Tag {tag} does not match the version in the files ({current}).')
    return version


# --- changelog --------------------------------------------------------------

def _links_start(text: str) -> int:
    """Where the trailing block of link references begins (len(text) if none)."""
    lines = text.splitlines(keepends=True)
    i = len(lines)
    while i and not lines[i - 1].strip():
        i -= 1
    j = i
    while j and (_LINK.match(lines[j - 1]) or not lines[j - 1].strip()):
        j -= 1
    while j < i and not lines[j].strip():
        j += 1
    if j == i:
        return len(text)
    return sum(len(line) for line in lines[:j])


def _links(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text[_links_start(text):].splitlines():
        if _LINK.match(line):
            label, url = line.split(']:', 1)
            out.append((label[1:], url.strip()))
    return out


def roll_changelog(text: str, version: str, date: str, repo_url: str) -> str:
    """Move what is under ``## [Unreleased]`` into ``## [version] - date``,
    leave a fresh empty Unreleased above it, and update the link references."""
    parse_version(version)
    m = re.search(r'^## \[Unreleased\][^\n]*\n', text, re.M)
    if not m:
        raise ReleaseError("CHANGELOG.md has no '## [Unreleased]' heading.")
    if re.search(rf'^## \[{re.escape(version)}\]', text, re.M):
        raise ReleaseError(f'CHANGELOG.md already has a section for {version}.')
    links_at = _links_start(text)
    nxt = _HEADING.search(text, m.end())
    end = min(nxt.start() if nxt else len(text), links_at)
    body = text[m.end():end].strip('\n')
    if not body.strip():
        raise ReleaseError("There is nothing under '## [Unreleased]' to release.")

    older = text[end:links_at].rstrip('\n')
    previous = nxt.group(1) if nxt and nxt.start() < links_at else None
    links = [('Unreleased', f'{repo_url}/compare/v{version}...HEAD'),
             (version, f'{repo_url}/compare/v{previous}...v{version}' if previous
              else f'{repo_url}/releases/tag/v{version}')]
    links += [(label, url) for label, url in _links(text) if label not in ('Unreleased', version)]

    out = text[:m.start()] + '## [Unreleased]\n\n' + f'## [{version}] - {date}\n\n' + body + '\n'
    if older:
        out += '\n' + older + '\n'
    return out + '\n' + ''.join(f'[{label}]: {url}\n' for label, url in links)


def release_notes(text: str, version: str, *, limit: int | None = None,
                  full_url: str | None = None) -> str:
    """A version's changelog section, cut at a whole entry if over ``limit``."""
    m = re.search(rf'^## \[{re.escape(version)}\][^\n]*\n', text, re.M)
    if not m:
        raise ReleaseError(f'CHANGELOG.md has no section for {version}.')
    nxt = _HEADING.search(text, m.end())
    end = min(nxt.start() if nxt else len(text), _links_start(text))
    notes = text[m.end():end].strip('\n') + '\n'
    if limit is None or len(notes) <= limit:
        return notes

    where = f'[CHANGELOG.md]({full_url})' if full_url else 'CHANGELOG.md'
    tail = f'*This release is too long for a GitHub release; the rest is in {where}.*\n'
    room = limit - len(tail) - 2
    # Cut only where an entry or a heading starts, so no entry is cut in half.
    starts = [i for i in (0, *(k + 1 for k, ch in enumerate(notes) if ch == '\n'))
              if notes.startswith(('- ', '#'), i)]
    cut = max((i for i in starts if len(notes[:i].rstrip('\n')) <= room), default=0)
    kept = notes[:cut].rstrip('\n').split('\n')
    while kept and kept[-1].startswith('#'):  # a heading left with nothing under it
        kept.pop()
        while kept and not kept[-1].strip():
            kept.pop()
    return '\n'.join(kept) + '\n\n' + tail


# --- git --------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    r = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
    if r.returncode:
        raise ReleaseError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def repo_url(root: Path = REPO) -> str:
    """origin as an https URL, for changelog links and release notes."""
    try:
        url = _git(root, 'remote', 'get-url', 'origin')
    except (ReleaseError, OSError):
        return FALLBACK_URL
    url = re.sub(r'^git@github\.com:', 'https://github.com/', url)
    url = re.sub(r'\.git$', '', url)
    return url if url.startswith('https://') else FALLBACK_URL


def _run_tests(root: Path) -> None:
    for cmd, cwd in ((['pixi', 'run', '-e', 'dev', 'pytest', '-q'], root / 'backend'),
                     (['pixi', 'run', 'npx', 'vitest', 'run'], root / 'frontend'),
                     (['pixi', 'run', 'npx', 'tsc', '--noEmit'], root / 'frontend')):
        print(f"release: {' '.join(cmd)}  (in {cwd.name}/)", flush=True)
        if subprocess.run(cmd, cwd=cwd).returncode:
            raise ReleaseError(f"{' '.join(cmd)} failed; nothing was changed.")


def prepare(version: str, *, root: Path = REPO, run_tests: bool = True,
            date: str | None = None) -> str:
    """Check, test, bump, commit and tag. Returns the tag; pushes nothing."""
    current = current_version(root)
    check_newer(current, version)
    branch = _git(root, 'rev-parse', '--abbrev-ref', 'HEAD')
    if branch != 'main':
        raise ReleaseError(f'Releases are cut from main; this checkout is on {branch}.')
    if _git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ReleaseError('There are uncommitted changes; commit or stash them first.')
    tag = f'v{version}'
    if _git(root, 'tag', '--list', tag):
        raise ReleaseError(f'Tag {tag} already exists.')

    changelog = root / 'CHANGELOG.md'
    # Rolled before the tests run, so a changelog problem fails in seconds.
    rolled = roll_changelog(changelog.read_text(encoding='utf-8'), version,
                            date or datetime.date.today().isoformat(), repo_url(root))
    if run_tests:
        _run_tests(root)

    write_version(root, version)
    changelog.write_text(rolled, encoding='utf-8')
    _git(root, 'add', *VERSION_FILES, 'CHANGELOG.md')
    _git(root, 'commit', '-m', f'release: {tag}')
    _git(root, 'tag', '-a', tag, '-m', f'xcell {version}')
    return tag


# --- command line -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='release.py', description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('version', help='print the version; fail if the files disagree')
    b = sub.add_parser('bump', help='rewrite the version files and roll the changelog')
    b.add_argument('version')
    b.add_argument('--date', help='release date for the changelog (default: today)')
    n = sub.add_parser('notes', help="print a version's changelog section")
    n.add_argument('version', help='0.2.0 or v0.2.0')
    n.add_argument('--limit', type=int, default=GITHUB_NOTES_LIMIT)
    t = sub.add_parser('check-tag', help='fail unless the tag names the version in the files')
    t.add_argument('tag')
    pr = sub.add_parser('prepare', help='check, test, bump, commit and tag (no push)')
    pr.add_argument('version')
    pr.add_argument('--skip-tests', action='store_true')
    args = p.parse_args(argv)

    try:
        if args.cmd == 'version':
            print(current_version())
        elif args.cmd == 'bump':
            check_newer(current_version(), args.version)
            changelog = REPO / 'CHANGELOG.md'
            rolled = roll_changelog(changelog.read_text(encoding='utf-8'), args.version,
                                    args.date or datetime.date.today().isoformat(), repo_url())
            write_version(REPO, args.version)
            changelog.write_text(rolled, encoding='utf-8')
            print(f'release: bumped to {args.version}; review with git diff.')
        elif args.cmd == 'notes':
            version = args.version.removeprefix('v')
            text = (REPO / 'CHANGELOG.md').read_text(encoding='utf-8')
            sys.stdout.write(release_notes(text, version, limit=args.limit,
                                           full_url=f'{repo_url()}/blob/v{version}/CHANGELOG.md'))
        elif args.cmd == 'check-tag':
            print(check_tag(REPO, args.tag))
        elif args.cmd == 'prepare':
            tag = prepare(args.version, run_tests=not args.skip_tests)
            print(f'release: committed and tagged {tag} locally. To publish:\n'
                  f'    git push origin main {tag}\n'
                  f'The release workflow then creates the GitHub release. To undo instead:\n'
                  f'    git tag -d {tag} && git reset --hard HEAD~1')
    except ReleaseError as e:
        print(f'release: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
