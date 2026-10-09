"""Build the developmental GO slim bundles.

Run this to regenerate ``gene_sets/dev_goslim_{coarse,fine}_{mouse,human}.json``
beside it. ``dev_goslim.tsv`` is the source of truth: a curated GO slim for
developmental biology at two depths, with the fine terms nested under the
coarse ones. The design note is ``docs/superpowers/specs/2026-10-08-dev-goslim-design.md``.

Table columns::

    level        coarse | fine
    parent       a fine row's coarse go_id (a GO id or a ``union:<id>``)
    go_id        the term, or ``union:<id>`` for a coarse row with no single GO term
    name         the GO name — checked against the ontology, so a wrong id fails
    mouse_genes  rewritten by the build (closed membership on the MGI GAF)
    cross        ``x`` when a fine term does not descend from its coarse term in GO

A set is a term's closed membership (ancestors included, as the GO library's
sets are); a ``union:<id>`` coarse set is the union of its fine sets. There are no
size bounds — a slim is curated — but the build prints every size and any
sibling pair whose Jaccard exceeds :data:`OVERLAP_THRESHOLD`, so redundancy
is a number on the screen before the table is accepted. A cross-linked fine
term appears in its folder but not in its coarse set, since the coarse set is
the GO term's closure; the ``x`` flag makes that placement a recorded choice.

Needs go-basic.obo and both species' GAFs from the gene-set-source cache; they
are downloaded once if absent (~60 MB). Human symbols come straight from
goa_human, not from the mouse set by ortholog mapping: GO annotates the two
species independently.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
OUT_DIR = HERE / 'gene_sets'
TABLE_PATH = HERE / 'dev_goslim.tsv'
COLUMNS = ['level', 'parent', 'go_id', 'name', 'mouse_genes', 'cross']
SPECIES = ('mouse', 'human')
OVERLAP_THRESHOLD = 0.5
UNION = 'union:'


def is_union(go_id: str) -> bool:
    return go_id.startswith(UNION)

DESCRIPTION = {
    'coarse': ('A GO slim for developmental biology, coarse depth: one set per major developmental '
               'process or organ system (pattern specification, cell fate commitment, gastrulation, the '
               'signaling pathways, nervous, circulatory, skeletal, limb, …), each the GO term\'s annotated '
               'genes with descendants included. Sized for enrichment of markers and DE lists. The fine '
               'bundle nests 3–11 narrower terms under each of these.'),
    'fine': ('A GO slim for developmental biology, fine depth: one folder per coarse term, holding its '
             'narrower terms (chondrocyte differentiation under skeletal, left/right pattern formation under '
             'pattern specification, the named pathways under signaling), each the GO term\'s annotated genes '
             'with descendants included. Sized for scoring cells. A folder may hold a term from elsewhere in '
             'GO when biology puts it there (osteoblast differentiation under skeletal).'),
}


def _root():
    sys.path.insert(0, str(HERE.parent.parent))


# --- table -----------------------------------------------------------------------

def read_table(path: Path) -> list[dict[str, Any]]:
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if reader.fieldnames != COLUMNS:
            raise ValueError(f'{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}')
        rows = []
        for r in reader:
            r = {k: (v or '').strip() for k, v in r.items()}
            try:
                r['mouse_genes'] = int(r['mouse_genes'] or 0)
            except ValueError:
                r['mouse_genes'] = 0
            rows.append(r)
    return rows


def validate(rows: list[dict[str, Any]], onto: dict[str, Any]) -> list[str]:
    """Every way the table can be wrong, each named with its row."""
    problems: list[str] = []
    coarse_ids: set[str] = set()
    seen: dict[str, set[str]] = {'coarse': set(), 'fine': set()}
    cache: dict[str, frozenset[str]] = {}
    for i, r in enumerate(rows, start=2):
        where = f'line {i} ({r.get("name") or r.get("go_id")})'
        level, tid = r['level'], r['go_id']
        if level not in seen:
            problems.append(f'{where}: level must be coarse or fine, not {level!r}')
            continue
        if level == 'coarse':
            coarse_ids.add(tid)
        if is_union(tid):
            if level != 'coarse':
                problems.append(f'{where}: only a coarse row may be a union')
            if tid in seen[level]:
                problems.append(f'{where}: {tid} appears twice at the {level} level')
            seen[level].add(tid)
            continue
        if tid in seen[level]:
            problems.append(f'{where}: {tid} appears twice at the {level} level')
        seen[level].add(tid)
        if tid not in onto['name']:
            if tid in onto['alt']:
                problems.append(f'{where}: {tid} is an alt_id of {onto["alt"][tid]}; use the live id')
            else:
                problems.append(f'{where}: {tid} is not in the ontology (obsolete or unknown)')
            continue
        if onto['name'][tid] != r['name']:
            problems.append(f'{where}: name does not match the ontology, which calls {tid} {onto["name"][tid]!r}')
        if onto['namespace'][tid] != 'biological_process':
            problems.append(f'{where}: {tid} is {onto["namespace"][tid]}, not biological_process')
    _root()
    from xcell import go_semantic as gos  # noqa: PLC0415

    for i, r in enumerate(rows, start=2):
        if r['level'] != 'fine':
            continue
        where = f'line {i} ({r["name"]})'
        parent = r['parent']
        if parent not in coarse_ids:
            problems.append(f'{where}: parent {parent!r} is not a coarse row')
            continue
        if is_union(parent) or r['go_id'] not in onto['name']:
            continue
        descends = parent in gos.ancestors(onto['parents'], r['go_id'], cache)
        if not descends and r['cross'] != 'x':
            problems.append(f'{where}: does not descend from {parent} in GO; mark cross=x if that is intended')
        if descends and r['cross'] == 'x':
            problems.append(f'{where}: descends from {parent}, so cross=x is wrong')
    return problems


# --- bundles -------------------------------------------------------------------

def _set_name(r: dict[str, Any]) -> str:
    return r['name'] if is_union(r['go_id']) else f'{r["name"]} ({r["go_id"]})'


def _coarse_rows(rows):
    return [r for r in rows if r['level'] == 'coarse']


def _fine_rows(rows, parent):
    return [r for r in rows if r['level'] == 'fine' and r['parent'] == parent]


def build(rows: list[dict[str, Any]], onto: dict[str, Any], gaf: dict[str, Any], species: str,
          level: str) -> dict[str, Any]:
    """One bundle. ``level`` is ``coarse`` (flat) or ``fine`` (a folder per coarse term)."""
    _root()
    from xcell import go_semantic as gos  # noqa: PLC0415

    members = gos.closed_members(onto, gaf)
    release = f'GO {onto.get("version") or "?"}, {species} GAF {gaf.get("date") or "?"}'
    sets: list[dict[str, Any]] = []
    empty: list[str] = []

    def add(r, genes, folder=None, under=None):
        if not genes:
            empty.append(_set_name(r))
            return
        desc = ('' if is_union(r['go_id']) else f'{r["go_id"]} · ') + f'{len(genes)} genes, descendants included'
        if under is not None:
            desc += f' · under {under["name"]}' + ('' if is_union(under['go_id']) else f' ({under["go_id"]})')
        s = {'name': _set_name(r), 'description': f'{desc} · {release}', 'genes': sorted(genes)}
        if folder:
            s['folder'] = folder
        sets.append(s)

    for c in _coarse_rows(rows):
        if level == 'coarse':
            if is_union(c['go_id']):
                genes = set().union(*(members.get(f['go_id'], set()) for f in _fine_rows(rows, c['go_id'])))
            else:
                genes = members.get(c['go_id'], set())
            add(c, genes)
        else:
            for f in _fine_rows(rows, c['go_id']):
                add(f, members.get(f['go_id'], set()), folder=c['name'], under=c)
    return {
        'name': f'Developmental GO slim, {level} ({species})',
        'description': f'{DESCRIPTION[level]} Built from {release}.',
        'sets': sets,
        'empty': empty,
    }


def sibling_overlaps(rows, members: dict[str, set[str]], *, threshold: float = OVERLAP_THRESHOLD):
    """Fine siblings whose Jaccard exceeds ``threshold`` — the redundancy the table accepts."""
    out = []
    for c in _coarse_rows(rows):
        fines = _fine_rows(rows, c['go_id'])
        for i, a in enumerate(fines):
            for b in fines[i + 1:]:
                ga, gb = members.get(a['go_id'], set()), members.get(b['go_id'], set())
                if not ga or not gb:
                    continue
                j = len(ga & gb) / len(ga | gb)
                if j > threshold:
                    out.append((a['go_id'], b['go_id'], round(j, 2)))
    return out


def _size(r, rows, members: dict[str, set[str]]) -> int:
    if is_union(r['go_id']):
        return len(set().union(*(members.get(f['go_id'], set()) for f in _fine_rows(rows, r['go_id']))))
    return len(members.get(r['go_id'], set()))


def rewrite_sizes(path: Path, rows, members: dict[str, set[str]]) -> None:
    for r in rows:
        r['mouse_genes'] = _size(r, rows, members)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[k] for k in COLUMNS])


# --- main ------------------------------------------------------------------------

def _load(species_list):
    """The ontology and each species' GAF, downloading what the cache lacks."""
    _root()
    from xcell import gene_set_sources as gss  # noqa: PLC0415
    from xcell import go_semantic as gos  # noqa: PLC0415

    def say(frac, msg):
        print(f'  {msg}')

    for sp in species_list:
        if not (gos.obo_path().is_file() and gos.gaf_path(sp).is_file()):
            print(f'downloading GO files for {sp} into {gos.go_dir()}')
            gss.GOSource._download_files(sp, say)
    onto = gos.load_ontology()
    return onto, {sp: gos.load_gaf(sp) for sp in species_list}


def main() -> int:
    _root()
    from xcell import go_semantic as gos  # noqa: PLC0415

    rows = read_table(TABLE_PATH)
    onto, gafs = _load(SPECIES)
    problems = validate(rows, onto)
    if problems:
        print('\n'.join(problems))
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    members = {sp: gos.closed_members(onto, gafs[sp]) for sp in SPECIES}
    for sp in SPECIES:
        for level in ('coarse', 'fine'):
            bundle = build(rows, onto, gafs[sp], sp, level)
            path = OUT_DIR / f'dev_goslim_{level}_{sp}.json'
            path.write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + '\n')
            print(f'wrote {path.name}: {len(bundle["sets"])} sets, '
                  f'{sum(len(s["genes"]) for s in bundle["sets"])} genes'
                  + (f'; empty on {sp}: {", ".join(bundle["empty"])}' if bundle['empty'] else ''))
    rewrite_sizes(TABLE_PATH, rows, members['mouse'])
    print(f'\n{"term":<62}{"mouse":>7}{"human":>7}')
    for c in _coarse_rows(rows):
        print(f'{c["name"]:<62}{c["mouse_genes"]:>7}{_size(c, rows, members["human"]):>7}')
        for f in _fine_rows(rows, c['go_id']):
            flag = ' x' if f['cross'] else ''
            print(f'    {f["name"] + flag:<58}{f["mouse_genes"]:>7}{len(members["human"].get(f["go_id"], set())):>7}')
    names = {r['go_id']: r['name'] for r in rows}
    for sp in SPECIES:
        pairs = sibling_overlaps(rows, members[sp])
        if pairs:
            print(f'\nsibling overlap above {OVERLAP_THRESHOLD} on {sp}:')
            for a, b, j in pairs:
                print(f'  {names[a]} ~ {names[b]}: {j}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
