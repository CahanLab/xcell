"""Propose fine-term candidates for the developmental GO slim.

This is the first, mechanical step of the derivation described in
``docs/dev-goslim.md``: for each coarse term (or a root you name), list the
biological-process descendants whose closed membership on the mouse GAF falls
in a size window, largest first, skipping any that overlaps an already listed
candidate by Jaccard at or above a cap. A curator then keeps, drops and adds
terms by hand; the result is ``dev_goslim.tsv`` and ``build_dev_goslim.py``
validates it. Nothing here writes to the table.

Parameters are the ones used for the shipped table (2026-10-08)::

    --min 40 --max 350   closed mouse genes per candidate
    --jaccard 0.5        a candidate at or above this overlap with one already
                         listed under the same root is skipped
    --per-root 14        candidates listed per root

Run from ``backend/``::

    pixi run -e dev python xcell/data/carve_dev_goslim_candidates.py
    pixi run -e dev python xcell/data/carve_dev_goslim_candidates.py --root 'limb development' --min 20

Needs go-basic.obo and the mouse GAF in the gene-set-source cache (fetch the
GO library once in the app, or run ``build_dev_goslim.py``, which downloads
them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
TABLE_PATH = HERE / 'dev_goslim.tsv'


def descendants(parents: dict[str, set[str]], root: str) -> set[str]:
    children: dict[str, set[str]] = {}
    for t, ps in parents.items():
        for p in ps:
            children.setdefault(p, set()).add(t)
    out, stack = set(), [root]
    while stack:
        t = stack.pop()
        if t in out:
            continue
        out.add(t)
        stack.extend(children.get(t, ()))
    out.discard(root)
    return out


def carve(onto, members, root: str, *, lo: int, hi: int, jaccard: float, per_root: int) -> list[str]:
    """Candidate term ids under ``root``: BP, size in [lo, hi], largest first, non-redundant."""
    cand = sorted((t for t in descendants(onto['parents'], root)
                   if onto['namespace'].get(t) == 'biological_process' and lo <= len(members.get(t, ())) <= hi),
                  key=lambda t: -len(members[t]))
    chosen: list[str] = []
    for t in cand:
        if all(len(members[t] & members[c]) / len(members[t] | members[c]) < jaccard for c in chosen):
            chosen.append(t)
        if len(chosen) >= per_root:
            break
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', action='append', help='a GO id or exact BP term name; repeatable. Default: the coarse rows of dev_goslim.tsv')
    ap.add_argument('--min', type=int, default=40)
    ap.add_argument('--max', type=int, default=350)
    ap.add_argument('--jaccard', type=float, default=0.5)
    ap.add_argument('--per-root', type=int, default=14)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE.parent.parent))
    from xcell import go_semantic as gos  # noqa: PLC0415

    onto = gos.load_ontology()
    members = gos.closed_members(onto, gos.load_gaf('mouse'))
    names = onto['name']
    by_name = {v: k for k, v in names.items()}

    roots: list[str] = []
    if args.root:
        for r in args.root:
            tid = r if r.startswith('GO:') else by_name.get(r)
            if not tid or tid not in names:
                print(f'no such term: {r!r}', file=sys.stderr)
                return 1
            roots.append(tid)
    else:
        import csv  # noqa: PLC0415
        with open(TABLE_PATH, newline='') as fh:
            roots = [r['go_id'] for r in csv.DictReader(fh, delimiter='\t')
                     if r['level'] == 'coarse' and r['go_id'].startswith('GO:')]

    print(f'window {args.min}-{args.max} mouse genes, Jaccard < {args.jaccard}, up to {args.per_root} per root; '
          f'GO {onto.get("version")}')
    for root in roots:
        chosen = carve(onto, members, root, lo=args.min, hi=args.max, jaccard=args.jaccard, per_root=args.per_root)
        print(f'\n## {names[root]} {root}  ({len(members.get(root, ()))} genes)')
        for t in chosen:
            print(f'  {t}  {names[t]}  ({len(members[t])})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
