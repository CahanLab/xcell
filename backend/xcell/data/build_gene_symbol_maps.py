"""Build the shipped Ensembl gene ID -> symbol tables from GENCODE.

Run this to regenerate `ensembl_symbols_{human,mouse}.tsv.gz` beside it. The
output is committed, so xcell maps symbols offline and instantly; this script
exists so the provenance of that data is reproducible rather than folklore.

    pixi run -e dev python xcell/data/build_gene_symbol_maps.py

**Two releases per species, newest winning.** A single current release misses
retired IDs, and datasets in the wild are often older than the current
annotation -- 10x's `refdata-gex-GRCh38-2020-A`, still one of the most widely
used references, is GENCODE v32/M23. Merging the current release with that one
covers both without materially growing the file.
"""
from __future__ import annotations

import gzip
import sys
import urllib.request
from pathlib import Path

BASE = 'https://ftp.ebi.ac.uk/pub/databases/gencode'

# Newest last: later releases overwrite earlier ones, so the current symbol
# wins wherever an ID was renamed between releases.
SOURCES = {
    'human': [
        (f'{BASE}/Gencode_human/release_32/gencode.v32.primary_assembly.annotation.gtf.gz',
         'GENCODE v32 (10x refdata-gex-GRCh38-2020-A)'),
        (f'{BASE}/Gencode_human/release_50/gencode.v50.primary_assembly.annotation.gtf.gz',
         'GENCODE v50'),
    ],
    'mouse': [
        (f'{BASE}/Gencode_mouse/release_M23/gencode.vM23.primary_assembly.annotation.gtf.gz',
         'GENCODE M23 (10x refdata-gex-mm10-2020-A)'),
        (f'{BASE}/Gencode_mouse/release_M39/gencode.vM39.primary_assembly.annotation.gtf.gz',
         'GENCODE M39'),
    ],
}

OUT_DIR = Path(__file__).resolve().parent


def _attr(attrs: str, key: str) -> str | None:
    """Pull one value out of a GTF attribute string."""
    needle = f'{key} "'
    start = attrs.find(needle)
    if start < 0:
        return None
    start += len(needle)
    end = attrs.find('"', start)
    return attrs[start:end] if end > 0 else None


def harvest(url: str) -> dict[str, str]:
    """gene_id (version stripped) -> gene_name, from the gene lines of a GTF."""
    out: dict[str, str] = {}
    with urllib.request.urlopen(url, timeout=600) as resp, gzip.open(resp, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.split('\t', 9)
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            gid = _attr(parts[8], 'gene_id')
            name = _attr(parts[8], 'gene_name')
            if not gid or not name:
                continue
            # Ensembl IDs carry a version suffix that datasets may or may not
            # keep. The map is keyed without it and lookups strip it too.
            out[gid.split('.', 1)[0]] = name
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for species, sources in SOURCES.items():
        merged: dict[str, str] = {}
        for url, label in sources:
            print(f'{species}: fetching {label} ...', flush=True)
            got = harvest(url)
            print(f'  {len(got):,} genes', flush=True)
            merged.update(got)

        path = OUT_DIR / f'ensembl_symbols_{species}.tsv.gz'
        with gzip.open(path, 'wt', compresslevel=9) as fh:
            for gid in sorted(merged):
                fh.write(f'{gid}\t{merged[gid]}\n')
        size = path.stat().st_size / 1e6
        print(f'{species}: wrote {len(merged):,} ids -> {path.name} ({size:.1f} MB)\n',
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
