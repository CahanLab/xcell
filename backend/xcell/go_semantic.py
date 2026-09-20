"""Gene Ontology: the ontology graph, a species' annotations, and semantic similarity.

The gene map's *annotation* channel treats every gene set alike, so two genes
sharing "biological process" count as much as two sharing "chondrocyte
differentiation". GO is a DAG with information content, and that is what
this module adds: each gene is annotated to its terms *and all their
ancestors*, every term is weighted by how rare it is, and two genes are
compared with SimGIC (Pesquita et al. 2007):

    S(a, b) = Σ IC(shared ancestors) / Σ IC(union of ancestors)

which is a single sparse matrix product and needs no per-pair term matching.
``IC(t) = -log(n_genes annotated to t or a descendant / n_annotated genes)``,
computed over the whole species' annotation set, not just the genes asked
about, so a term is rare or common on its own merits.

Files come from the GO Consortium (``go-basic.obo`` and the species GAF) and
live beside the gene-set caches; :mod:`gene_set_sources` downloads them as
the ``go`` source's one library, so the Gene set library modal is where a
user fetches them. Parsed structures are memoised in-process by file mtime
because the OBO is 30 MB and takes seconds to read.

Pure: paths and gene lists in, arrays out. No AnnData, no adaptor.
"""
from __future__ import annotations

import gzip
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

OBO_URL = 'https://current.geneontology.org/ontology/go-basic.obo'
GAF_URLS = {
    'mouse': 'https://current.geneontology.org/annotations/mgi.gaf.gz',
    'human': 'https://current.geneontology.org/annotations/goa_human.gaf.gz',
}
#: GAF column 1 values that carry gene annotations; ComplexPortal and
#: RNAcentral rows annotate complexes and ncRNAs under other identifiers.
GAF_DB = {'mouse': {'MGI'}, 'human': {'UniProtKB'}}

ASPECTS = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
ASPECT_LETTER = {'P': 'bp', 'F': 'mf', 'C': 'cc'}
#: Relationships along which annotations propagate. ``regulates`` and its
#: siblings do not: regulating a process is not taking part in it.
PROPAGATING = {'is_a', 'part_of'}


def go_dir() -> Path:
    from xcell import gene_set_sources as gss  # noqa: PLC0415

    return gss.cache_dir() / 'go'


def obo_path() -> Path:
    return go_dir() / 'go-basic.obo'


def gaf_path(species: str) -> Path:
    return go_dir() / f'{species}.gaf.gz'


# --- parsing -----------------------------------------------------------------

def parse_obo(text: str) -> dict[str, Any]:
    """``{'parents', 'namespace', 'name', 'alt', 'version'}`` from go-basic.obo.

    Obsolete terms are dropped; ``alt_id`` maps old ids onto the live term so
    a GAF that still uses one resolves. Only ``is_a`` and ``part_of`` become
    parents (see :data:`PROPAGATING`).
    """
    parents: dict[str, set[str]] = {}
    namespace: dict[str, str] = {}
    names: dict[str, str] = {}
    alt: dict[str, str] = {}
    version = ''
    cur: dict[str, Any] | None = None

    def flush() -> None:
        if cur is None or cur.get('obsolete') or not cur.get('id'):
            return
        tid = cur['id']
        parents[tid] = set(cur.get('parents', ()))
        namespace[tid] = cur.get('namespace', '')
        names[tid] = cur.get('name', '')
        for a in cur.get('alt', ()):
            alt[a] = tid

    for raw in text.splitlines():
        line = raw.strip()
        if cur is None and line.startswith('data-version:'):
            version = line.split(':', 1)[1].strip()
            continue
        if line == '[Term]':
            flush()
            cur = {'parents': [], 'alt': []}
            continue
        if line.startswith('['):  # [Typedef] and anything else ends a term
            flush()
            cur = None
            continue
        if cur is None or not line:
            continue
        key, _, value = line.partition(':')
        value = value.strip()
        if key == 'id':
            cur['id'] = value
        elif key == 'name':
            cur['name'] = value
        elif key == 'namespace':
            cur['namespace'] = value
        elif key == 'is_obsolete' and value.startswith('true'):
            cur['obsolete'] = True
        elif key == 'alt_id':
            cur['alt'].append(value)
        elif key == 'is_a':
            cur['parents'].append(value.split('!', 1)[0].strip())
        elif key == 'relationship':
            rel, _, target = value.partition(' ')
            if rel in PROPAGATING:
                cur['parents'].append(target.split('!', 1)[0].strip())
    flush()
    return {'parents': parents, 'namespace': namespace, 'name': names, 'alt': alt, 'version': version}


def parse_gaf(lines, species: str) -> dict[str, Any]:
    """``{'annotations': {symbol: {term}}, 'aspects': {term: 'bp'|…}, 'date'}``.

    Rows with a ``NOT`` qualifier are negative evidence and are skipped;
    rows from databases other than the species' gene database (complexes,
    ncRNAs) are skipped; evidence codes are all kept — IEA is most of what
    exists for many genes, and the similarity is meant to be broad.
    """
    dbs = GAF_DB.get(species, set())
    annotations: dict[str, set[str]] = {}
    aspects: dict[str, str] = {}
    date = ''
    for raw in lines:
        line = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
        if line.startswith('!'):
            if line.startswith('!date-generated:'):
                date = line.split(':', 1)[1].strip()
            continue
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 9:
            continue
        if dbs and cols[0] not in dbs:
            continue
        if cols[3].startswith('NOT'):
            continue
        sym, term, aspect = cols[2].strip(), cols[4].strip(), cols[8].strip()
        if not sym or not term.startswith('GO:'):
            continue
        annotations.setdefault(sym, set()).add(term)
        if aspect in ASPECT_LETTER:
            aspects.setdefault(term, ASPECT_LETTER[aspect])
    return {'annotations': annotations, 'aspects': aspects, 'date': date}


def ancestors(parents: dict[str, set[str]], term: str, cache: dict[str, frozenset[str]]) -> frozenset[str]:
    """The term and every ancestor along propagating relationships (memoised)."""
    hit = cache.get(term)
    if hit is not None:
        return hit
    seen: set[str] = {term}
    stack = list(parents.get(term, ()))
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        stack.extend(parents.get(t, ()))
    out = frozenset(seen)
    cache[term] = out
    return out


# --- the semantic space -------------------------------------------------------

class GOSpace:
    """One species, one aspect: closed annotations per gene and IC per term."""

    def __init__(self, ontology: dict[str, Any], gaf: dict[str, Any], aspect: str):
        if aspect not in ASPECTS:
            raise ValueError(f"GO aspect must be one of {', '.join(ASPECTS)}; got '{aspect}'")
        self.aspect = aspect
        ns = ASPECTS[aspect]
        parents, namespace, alt = ontology['parents'], ontology['namespace'], ontology['alt']
        self.names: dict[str, str] = ontology['name']
        cache: dict[str, frozenset[str]] = {}
        closed: dict[str, frozenset[str]] = {}
        counts: dict[str, int] = {}
        for sym, terms in gaf['annotations'].items():
            acc: set[str] = set()
            for t in terms:
                t = alt.get(t, t)
                if namespace.get(t) != ns:
                    continue
                acc |= ancestors(parents, t, cache)
            if not acc:
                continue
            fs = frozenset(acc)
            closed[sym] = fs
            for t in fs:
                counts[t] = counts.get(t, 0) + 1
        self.closed = closed
        self.n_genes = len(closed)
        self.ic: dict[str, float] = {
            t: -math.log(c / self.n_genes) for t, c in counts.items() if self.n_genes
        }
        # Case-insensitive lookup, since a dataset may spell Col1a1 as COL1A1.
        self._upper: dict[str, str] = {}
        for sym in closed:
            self._upper.setdefault(sym.upper(), sym)

    def terms_for(self, gene: str) -> frozenset[str]:
        sym = self._upper.get(str(gene).strip().upper())
        return self.closed.get(sym, frozenset()) if sym else frozenset()

    def similarity(self, genes: list[str]) -> tuple[np.ndarray, list[int]]:
        """SimGIC over the closed annotations; unit diagonal; unannotated genes
        are similar to nothing but themselves. Also each gene's closed term count."""
        from scipy import sparse  # noqa: PLC0415

        g = len(genes)
        term_index: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        n_terms: list[int] = []
        for i, gene in enumerate(genes):
            terms = self.terms_for(gene)
            n_terms.append(len(terms))
            for t in terms:
                j = term_index.setdefault(t, len(term_index))
                rows.append(i)
                cols.append(j)
                vals.append(self.ic.get(t, 0.0))
        n_t = max(1, len(term_index))
        AW = sparse.csr_matrix((vals, (rows, cols)), shape=(g, n_t), dtype=float)
        A = AW.copy()
        A.data[:] = 1.0
        shared = np.asarray((AW @ A.T).todense(), dtype=float)   # Σ IC over shared terms
        self_ic = np.asarray(AW.sum(axis=1)).ravel()              # Σ IC over own terms
        union = self_ic[:, None] + self_ic[None, :] - shared
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.where(union > 0, shared / union, 0.0)
        np.clip(S, 0.0, 1.0, out=S)
        np.fill_diagonal(S, 1.0)
        return S, n_terms


# --- loading (memoised by file mtime) ------------------------------------------

_ontology_cache: dict[tuple[str, float], dict[str, Any]] = {}
_gaf_cache: dict[tuple[str, float], dict[str, Any]] = {}
_space_cache: dict[tuple[str, float, str, float, str], GOSpace] = {}


def _mtime(path: Path) -> float:
    return os.stat(path).st_mtime


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else obo_path()
    if not p.is_file():
        raise ValueError("The Gene Ontology has not been fetched — fetch 'GO annotations' in the Gene set library first")
    key = (str(p), _mtime(p))
    hit = _ontology_cache.get(key)
    if hit is None:
        hit = parse_obo(p.read_text(encoding='utf-8', errors='replace'))
        _ontology_cache.clear()
        _ontology_cache[key] = hit
    return hit


def load_gaf(species: str, path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else gaf_path(species)
    if not p.is_file():
        raise ValueError(f"GO annotations for {species} have not been fetched — fetch 'GO annotations' in the Gene set library first")
    key = (str(p), _mtime(p))
    hit = _gaf_cache.get(key)
    if hit is None:
        opener = gzip.open if p.suffix == '.gz' else open
        with opener(p, 'rt', encoding='utf-8', errors='replace') as fh:
            hit = parse_gaf(fh, species)
        _gaf_cache.clear()
        _gaf_cache[key] = hit
    return hit


def space(species: str, aspect: str = 'bp') -> GOSpace:
    op, gp = obo_path(), gaf_path(species)
    onto = load_ontology(op)
    gaf = load_gaf(species, gp)
    key = (str(op), _mtime(op), str(gp), _mtime(gp), aspect)
    hit = _space_cache.get(key)
    if hit is None:
        hit = GOSpace(onto, gaf, aspect)
        if len(_space_cache) > 6:
            _space_cache.clear()
        _space_cache[key] = hit
    return hit


def similarity(genes: list[str], species: str, aspect: str = 'bp') -> dict[str, Any]:
    """SimGIC matrix for ``genes`` plus what the caller reports about it."""
    sp = space(species, aspect)
    S, n_terms = sp.similarity(genes)
    return {
        'similarity': S,
        'terms_per_gene': n_terms,
        'n_genes_annotated': int(sum(1 for n in n_terms if n > 0)),
        'n_terms': int(len({t for g in genes for t in sp.terms_for(g)})),
        'aspect': aspect,
        'species': species,
        'n_genes_in_space': sp.n_genes,
    }


def availability(species: str | None = None) -> dict[str, Any]:
    """Whether the files are on disk, without parsing them."""
    out: dict[str, Any] = {'obo': obo_path().is_file(), 'obo_path': str(obo_path()), 'gaf': {}}
    for sp in GAF_URLS:
        if species and sp != species:
            continue
        out['gaf'][sp] = gaf_path(sp).is_file()
    return out


# --- library sets from the closed annotations ---------------------------------

def term_sets(ontology: dict[str, Any], gaf: dict[str, Any], *, min_genes: int = 5,
              max_genes: int = 500) -> list[dict[str, Any]]:
    """One gene set per term whose closed membership fits the size bounds.

    Ancestors included, so "collagen fibril organization" carries the genes of
    its children too. The bounds drop the roots (every gene) and the leaves
    with one or two genes, which is the usual enrichment convention.
    """
    parents, namespace, names, alt = ontology['parents'], ontology['namespace'], ontology['name'], ontology['alt']
    cache: dict[str, frozenset[str]] = {}
    members: dict[str, set[str]] = {}
    for sym, terms in gaf['annotations'].items():
        acc: set[str] = set()
        for t in terms:
            t = alt.get(t, t)
            if t in namespace:
                acc |= ancestors(parents, t, cache)
        for t in acc:
            members.setdefault(t, set()).add(sym)
    short = {'biological_process': 'BP', 'molecular_function': 'MF', 'cellular_component': 'CC'}
    out: list[dict[str, Any]] = []
    for t, genes in members.items():
        if not (min_genes <= len(genes) <= max_genes):
            continue
        asp = short.get(namespace.get(t, ''), '')
        out.append({
            'name': f"{names.get(t, t)} ({t}, {asp})" if asp else f"{names.get(t, t)} ({t})",
            'description': f"{ASPECTS_LONG.get(asp, asp)} · {len(genes)} genes with ancestors included",
            'genes': sorted(genes),
            'url': f'https://www.ebi.ac.uk/QuickGO/term/{t}',
        })
    out.sort(key=lambda s: s['name'].lower())
    return out


ASPECTS_LONG = {'BP': 'biological process', 'MF': 'molecular function', 'CC': 'cellular component'}
