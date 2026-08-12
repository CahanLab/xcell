"""Map Ensembl gene IDs to official symbols, from tables shipped with xcell.

A dataset whose ``.var`` holds only ``ENSMUSG00000002603`` rather than ``Tgfb1``
is intact but unusable by a human: gene search finds nothing, imported gene sets
match nothing, and every panel that names a gene is unreadable.

Pure: takes lists of strings, returns plain dicts. The tables live in
``xcell/data`` and are cached per species, so a session pays the read once.

Shipped rather than fetched from an API, for three reasons: it works with no
network, it returns instantly instead of a round trip per session, and it is
*deterministic* -- the same dataset maps the same way in a year, which a live
service cannot promise.
"""
from __future__ import annotations

import gzip
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).resolve().parent / 'data'

# Ensembl encodes the species in the prefix, so it never has to be asked for.
SPECIES_PREFIXES = {'ENSG': 'human', 'ENSMUSG': 'mouse'}

# Named, rather than lumped into "unsupported", so a dataset xcell cannot map
# gets a message that says which animal it is and why nothing happened.
_OTHER_PREFIXES = {
    'ENSDARG': 'zebrafish', 'ENSRNOG': 'rat', 'ENSGALG': 'chicken',
    'ENSXETG': 'frog', 'ENSCAFG': 'dog', 'ENSSSCG': 'pig',
    'ENSMMUG': 'macaque', 'ENSBTAG': 'cow',
}


def strip_version(gene_id: str) -> str:
    """``ENSG00000141510.15`` -> ``ENSG00000141510``.

    Whether a dataset keeps the version suffix depends on which tool wrote the
    file, not on the biology, so it is removed on both sides of every lookup.
    Only Ensembl ids are touched: a symbol containing a dot must survive.
    """
    gid = str(gene_id)
    if not gid.startswith('ENS'):
        return gid
    return gid.split('.', 1)[0]


def _prefix_of(gene_id: str) -> str | None:
    gid = strip_version(gene_id)
    # Longest first, so ENSMUSG is not shadowed by ENSG.
    for pfx in sorted([*SPECIES_PREFIXES, *_OTHER_PREFIXES], key=len, reverse=True):
        if gid.startswith(pfx):
            return pfx
    return None


def detect_species(ids: list[str]) -> dict[str, Any]:
    """Which species these Ensembl ids are, by majority, or ``None``.

    Majority rather than unanimity: a real dataset can carry a handful of spike
    ins or a stray id from another build, and refusing over those would be
    useless. The split is reported so the user sees which way it went.
    """
    counts: Counter[str] = Counter()
    other: Counter[str] = Counter()
    for gid in ids:
        pfx = _prefix_of(gid)
        if pfx in SPECIES_PREFIXES:
            counts[SPECIES_PREFIXES[pfx]] += 1
        elif pfx is not None:
            other[_OTHER_PREFIXES[pfx]] += 1

    n_recognized = sum(counts.values())
    total = max(len(ids), 1)
    return {
        'species': counts.most_common(1)[0][0] if counts else None,
        'counts': dict(counts),
        'n_recognized': n_recognized,
        'fraction': n_recognized / total,
        'unsupported': other.most_common(1)[0][0] if (not counts and other) else None,
    }


@lru_cache(maxsize=4)
def load_table(species: str) -> dict[str, str]:
    """Ensembl id -> symbol for one species, read once per session."""
    if species not in set(SPECIES_PREFIXES.values()):
        raise ValueError(
            f"no symbol table for '{species}' — xcell ships human and mouse")
    path = _DATA / f'ensembl_symbols_{species}.tsv.gz'
    if not path.exists():
        raise ValueError(
            f'the {species} symbol table is missing from this install '
            f'({path.name}). Rebuild it with '
            '`python xcell/data/build_gene_symbol_maps.py`.')
    table: dict[str, str] = {}
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            gid, _, sym = line.rstrip('\n').partition('\t')
            if gid and sym:
                table[gid] = sym
    return table


def map_ids(ids: list[str], table: dict[str, str]) -> dict[str, Any]:
    """Map each id to a symbol, keeping the id itself where there is no match."""
    symbols: list[str] = []
    examples: list[dict[str, str]] = []
    n_mapped = 0
    for gid in ids:
        sym = table.get(strip_version(gid))
        if sym:
            n_mapped += 1
            if len(examples) < 5:
                examples.append({'id': str(gid), 'symbol': sym})
        # An unmapped gene keeps its own id. Blank would break the >90%-unique
        # test that makes a column offerable as an identifier, and would make
        # those genes unaddressable — worse than an ugly label.
        symbols.append(sym or str(gid))

    dupes = [s for s, n in Counter(symbols).items() if n > 1]
    return {
        'symbols': symbols,
        'n_mapped': n_mapped,
        'n_unmapped': len(ids) - n_mapped,
        'n_duplicate_symbols': len(dupes),
        'examples': examples,
    }
