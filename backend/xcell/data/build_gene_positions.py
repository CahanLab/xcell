"""Build the shipped gene position tables (`gene_positions_{human,mouse}.tsv.gz`).

CNV inference (xcell/cnv.py) needs each gene's chromosome and start so it can
order genes along the genome and smooth expression in genomic windows. Like
the symbol maps beside this script, the tables are committed so the backend
works offline; this script records where they came from.

    pixi run -e dev python xcell/data/build_gene_positions.py

Same two GENCODE releases per species as build_gene_symbol_maps.py, newest
winning, so a gene retired after the 10x 2020-A references still has a
position. Only genes on the primary chromosomes (1-22/19, X, Y, M) are kept;
scaffolds cannot be placed on a karyogram.

Columns: gene_id (version stripped), symbol, chromosome (``chr`` prefix
kept), start, end. Coordinates are GRCh38 / GRCm39 (GENCODE M39 is GRCm39;
M23 is GRCm38 — a gene present only in M23 keeps its GRCm38 position, which
is a negligible error at the 100-gene window scale CNV smoothing uses).
"""
from __future__ import annotations

import gzip
import sys
import urllib.request
from pathlib import Path

from build_gene_symbol_maps import SOURCES, _attr

OUT_DIR = Path(__file__).resolve().parent


def harvest(url: str) -> dict[str, tuple[str, str, int, int]]:
    """gene_id -> (symbol, chromosome, start, end) from the gene lines of a GTF."""
    out: dict[str, tuple[str, str, int, int]] = {}
    with urllib.request.urlopen(url, timeout=600) as resp, gzip.open(resp, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.split('\t', 9)
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if not chrom.startswith('chr') or '_' in chrom:
                continue
            gid = _attr(parts[8], 'gene_id')
            name = _attr(parts[8], 'gene_name')
            if not gid or not name:
                continue
            out[gid.split('.', 1)[0]] = (name, chrom, int(parts[3]), int(parts[4]))
    return out


def main() -> int:
    for species, sources in SOURCES.items():
        merged: dict[str, tuple[str, str, int, int]] = {}
        for url, label in sources:
            print(f'{species}: fetching {label} ...', flush=True)
            got = harvest(url)
            print(f'  {len(got):,} genes', flush=True)
            merged.update(got)
        path = OUT_DIR / f'gene_positions_{species}.tsv.gz'
        with gzip.open(path, 'wt', compresslevel=9) as fh:
            fh.write('gene_id\tsymbol\tchromosome\tstart\tend\n')
            for gid in sorted(merged):
                sym, chrom, start, end = merged[gid]
                fh.write(f'{gid}\t{sym}\t{chrom}\t{start}\t{end}\n')
        print(f'{species}: wrote {len(merged):,} genes -> {path.name} '
              f'({path.stat().st_size / 1e6:.1f} MB)\n', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
