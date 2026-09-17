"""External gene-set sources — fetched once, cached locally, searched offline.

Three sources behind one interface:

* **MSigDB** — per-collection ``*.symbols.gmt`` files plus the matching
  ``.json`` metadata from the Broad's release directory, for human (``Hs``)
  and mouse (``Mm``). The newest release is discovered from the directory
  index and pinned to :data:`MSIGDB_PINNED_RELEASE` when the index is
  unreachable, so a cached library keeps its version even when offline.
* **Enrichr** — ``datasetStatistics`` for the catalogue (a few hundred
  libraries: GO, Reactome, KEGG, WikiPathways, PanglaoDB, CellMarker, Tabula
  Muris …) and ``geneSetLibrary?mode=text`` for one library, a GMT whose
  tokens may carry ``,1.0`` weights.
* **STRING** — not a library but a query service: interaction partners of
  seed genes, or the edges among a gene list. Used to expand a set and, by
  the gene map, as a similarity channel.

A fetched library is written as one JSON file under :func:`cache_dir`
(``$XDG_CACHE_HOME/xcell/gene_set_sources/<source>/<species>/<id>.json``,
overridable as ``gene_set_sources.cache_dir`` in ``config.yaml``); a source's
catalogue is cached beside it and refreshed after :data:`CATALOGUE_TTL_DAYS`.
Search, paging and overlap all run over the cached file, never the network.

Pure module: no adaptor import, no AnnData. Only the standard library is used
for HTTP — the fetches here are three GETs and the alternatives are present in
the environment only transitively. Every network failure is a ``ValueError``
naming the URL, so routes turn it into a 400 the UI can show.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

USER_AGENT = 'xcell (https://github.com/CahanLab/xcell)'
TIMEOUT = 60
CATALOGUE_TTL_DAYS = 7

MSIGDB_BASE = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/'
MSIGDB_PINNED_RELEASE = '2026.1'
MSIGDB_SPECIES_CODE = {'human': 'Hs', 'mouse': 'Mm'}
ENRICHR_BASE = 'https://maayanlab.cloud/Enrichr/'
STRING_BASE = 'https://string-db.org/api/json/'
SPECIES_TAXON = {'human': 9606, 'mouse': 10090}
#: STRING's documented per-request identifier cap.
MAX_STRING_IDENTIFIERS = 2000

SPECIES = ('human', 'mouse')

Report = Callable[..., None]

_cache_dir_override: Path | None = None


# --- HTTP (module-level so tests monkeypatch them) ---------------------------

def fetch_text(url: str, *, timeout: float = TIMEOUT) -> str:
    """GET ``url`` as text. Raises ``ValueError`` naming the URL on any failure."""
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:  # URLError, HTTPError, timeout, SSL — one message for all
        raise ValueError(f"Could not fetch {url}: {e}") from e
    return data.decode('utf-8', errors='replace')


def fetch_json(url: str, *, timeout: float = TIMEOUT) -> Any:
    text = fetch_text(url, timeout=timeout)
    try:
        return json.loads(text)
    except ValueError as e:
        raise ValueError(f"{url} did not return JSON: {e}") from e


# --- cache location ------------------------------------------------------------

def set_cache_dir(path: Path | str | None) -> None:
    """Override the cache directory (tests, or a one-off script)."""
    global _cache_dir_override
    _cache_dir_override = Path(path) if path is not None else None


def cache_dir() -> Path:
    if _cache_dir_override is not None:
        return _cache_dir_override
    from xcell import config as user_config  # noqa: PLC0415

    cfg = user_config.get_user_config().get('gene_set_sources')
    raw = cfg.get('cache_dir') if isinstance(cfg, dict) else None
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    base = os.environ.get('XDG_CACHE_HOME') or '~/.cache'
    return Path(base).expanduser() / 'xcell' / 'gene_set_sources'


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)


def _library_path(source: str, species: str, library_id: str) -> Path:
    return cache_dir() / source / species / f'{_safe(library_id)}.json'


def _catalogue_path(source: str, species: str) -> Path:
    return cache_dir() / source / f'catalogue.{species}.json'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def check_species(species: str) -> None:
    if species not in SPECIES:
        raise ValueError(f"Unsupported species '{species}'; expected one of {', '.join(SPECIES)}")


# --- GMT -----------------------------------------------------------------------

def parse_gmt(text: str) -> list[dict[str, Any]]:
    """Parse GMT text into ``[{name, description, genes, url}]``.

    Tolerates Enrichr's weighted tokens (``COL1A1,1.0``), trailing empty
    columns, and repeated symbols. A set with no genes is dropped.
    """
    sets: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        name = parts[0].strip()
        if not name:
            continue
        description = parts[1].strip() if len(parts) > 1 else ''
        genes: list[str] = []
        seen: set[str] = set()
        for tok in parts[2:]:
            sym = tok.strip().split(',', 1)[0].strip()
            if sym and sym not in seen:
                seen.add(sym)
                genes.append(sym)
        if genes:
            sets.append({'name': name, 'description': description, 'genes': genes, 'url': ''})
    return sets


# --- sources -------------------------------------------------------------------

def _entry(source: str, library_id: str, name: str, description: str, species: str, *,
           n_sets: int | None = None, version: str | None = None, url: str = '') -> dict[str, Any]:
    return {
        'source': source, 'id': library_id, 'name': name, 'description': description,
        'species': species, 'n_sets': n_sets, 'version': version, 'url': url,
        'cached': False, 'fetched_at': None,
    }


class Source:
    id = ''
    name = ''
    description = ''
    url = ''
    kind = 'library'  # or 'query'

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        """The catalogue entry for ``library_id``, or ``ValueError``. Called
        synchronously by the fetch route so a typo is a 400, not a failed task."""
        raise NotImplementedError

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        raise NotImplementedError


class MSigDBSource(Source):
    id = 'msigdb'
    name = 'MSigDB'
    description = 'Molecular Signatures Database (Broad Institute): hallmark, curated, GO, cell-type and more'
    url = 'https://www.gsea-msigdb.org/gsea/msigdb/'

    #: (file prefix, display name, description) per species. The prefixes are
    #: the Broad's own; the mouse collections are orthology-mapped versions of
    #: the human ones plus mouse-specific phenotype and cell-type sets.
    COLLECTIONS: dict[str, list[tuple[str, str, str]]] = {
        'human': [
            ('h.all', 'Hallmark', '50 well-defined biological states and processes'),
            ('c2.cgp', 'Chemical and genetic perturbations', 'Expression signatures from published studies; includes the NABA matrisome sets'),
            ('c2.cp.reactome', 'Reactome pathways', ''),
            ('c2.cp.kegg_medicus', 'KEGG MEDICUS pathways', ''),
            ('c2.cp.wikipathways', 'WikiPathways', ''),
            ('c2.cp.biocarta', 'BioCarta pathways', ''),
            ('c2.cp.pid', 'PID pathways', ''),
            ('c3.tft.gtrd', 'Transcription factor targets (GTRD)', ''),
            ('c5.go.bp', 'GO biological process', ''),
            ('c5.go.cc', 'GO cellular component', ''),
            ('c5.go.mf', 'GO molecular function', ''),
            ('c5.hpo', 'Human phenotype ontology', ''),
            ('c6.all', 'Oncogenic signatures', ''),
            ('c7.immunesigdb', 'ImmuneSigDB', ''),
            ('c8.all', 'Cell type signatures', 'Markers from single-cell atlases'),
        ],
        'mouse': [
            ('mh.all', 'Hallmark', '50 well-defined biological states and processes'),
            ('m2.cgp', 'Chemical and genetic perturbations', 'Expression signatures from published studies; includes the NABA matrisome sets'),
            ('m2.cp.reactome', 'Reactome pathways', ''),
            ('m2.cp.wikipathways', 'WikiPathways', ''),
            ('m2.cp.biocarta', 'BioCarta pathways', ''),
            ('m3.gtrd', 'Transcription factor targets (GTRD)', ''),
            ('m5.go.bp', 'GO biological process', ''),
            ('m5.go.cc', 'GO cellular component', ''),
            ('m5.go.mf', 'GO molecular function', ''),
            ('m5.mpt', 'Mouse phenotype ontology', ''),
            ('m8.all', 'Cell type signatures', 'Markers from single-cell atlases'),
        ],
    }

    def latest_release(self, species: str) -> str:
        """Newest ``YYYY.N`` release listed for the species, else the pinned one."""
        check_species(species)
        code = MSIGDB_SPECIES_CODE[species]
        try:
            html = fetch_text(MSIGDB_BASE)
        except ValueError:
            return MSIGDB_PINNED_RELEASE
        versions = re.findall(rf'href="(\d{{4}}\.\d+)\.{code}/"', html)
        if not versions:
            return MSIGDB_PINNED_RELEASE
        return max(versions, key=lambda v: tuple(int(x) for x in v.split('.')))

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        check_species(species)
        return [
            _entry(self.id, prefix, name, desc, species, url=self.url)
            for prefix, name, desc in self.COLLECTIONS[species]
        ]

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        check_species(species)
        known = {c[0]: c for c in self.COLLECTIONS[species]}
        if library_id not in known:
            raise ValueError(f"'{library_id}' is not a MSigDB collection for {species}")
        prefix, name, desc = known[library_id]
        return _entry(self.id, prefix, name, desc, species, url=self.url)

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        entry = self.validate_library(library_id, species)
        display, desc = entry['name'], entry['description']
        version = self.latest_release(species)
        code = MSIGDB_SPECIES_CODE[species]
        base = f'{MSIGDB_BASE}{version}.{code}/{library_id}.v{version}.{code}'

        report(0.05, f'Downloading {display} gene sets…')
        sets = parse_gmt(fetch_text(base + '.symbols.gmt'))
        report(0.6, 'Downloading set metadata…')
        try:
            meta = fetch_json(base + '.json')
        except ValueError:
            meta = {}  # metadata is a nicety; the sets are the point
        if not isinstance(meta, dict):
            meta = {}
        for s in sets:
            # MSigDB's GMT puts the set's card URL in the description column.
            url_col = s['description'] if s['description'].startswith('http') else ''
            rec = meta.get(s['name'])
            if isinstance(rec, dict):
                s['url'] = rec.get('msigdbURL') or url_col
                bits = [rec.get('exactSource') or '',
                        f"PMID {rec['pmid']}" if rec.get('pmid') else '',
                        rec.get('collection') or '']
                s['description'] = ' · '.join(b for b in bits if b)
            else:
                s['url'] = url_col
                if url_col:
                    s['description'] = ''
        report(1.0, 'Done')
        return {
            'source': self.id, 'id': library_id, 'name': display, 'description': desc,
            'species': species, 'version': version, 'url': self.url, 'sets': sets,
        }


class EnrichrSource(Source):
    id = 'enrichr'
    name = 'Enrichr'
    description = "Ma'ayan lab gene-set libraries: GO, Reactome, KEGG, WikiPathways, cell-type markers and more"
    url = 'https://maayanlab.cloud/Enrichr/#libraries'

    @staticmethod
    def _species_of(library_name: str) -> str:
        return 'mouse' if re.search(r'mouse|mus_musculus|_mm\b|muris', library_name, re.IGNORECASE) else 'human'

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        check_species(species)
        data = fetch_json(ENRICHR_BASE + 'datasetStatistics')
        stats = data.get('statistics') if isinstance(data, dict) else None
        if not isinstance(stats, list):
            raise ValueError(f'{ENRICHR_BASE}datasetStatistics returned no statistics')
        entries = []
        for rec in stats:
            if not isinstance(rec, dict) or not isinstance(rec.get('libraryName'), str):
                continue
            lib = rec['libraryName']
            desc_bits = []
            if rec.get('genesPerTerm'):
                desc_bits.append(f"{rec['genesPerTerm']} genes/term")
            if rec.get('geneCoverage'):
                desc_bits.append(f"{rec['geneCoverage']:,} genes covered")
            if rec.get('link'):
                desc_bits.append(str(rec['link']))
            entries.append(_entry(
                self.id, lib, lib.replace('_', ' '), ' · '.join(desc_bits),
                self._species_of(lib), n_sets=rec.get('numTerms'), url=self.url,
            ))
        entries.sort(key=lambda e: e['name'].lower())
        return entries

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        check_species(species)
        known = {e['id']: e for e in catalogue(self.id, species)}
        if library_id not in known:
            raise ValueError(f"'{library_id}' is not an Enrichr library")
        return known[library_id]

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        entry = self.validate_library(library_id, species)
        report(0.05, f"Downloading {entry['name']}…")
        text = fetch_text(f'{ENRICHR_BASE}geneSetLibrary?mode=text&libraryName={quote(library_id)}')
        sets = parse_gmt(text)
        report(1.0, 'Done')
        return {
            'source': self.id, 'id': library_id, 'name': entry['name'],
            'description': entry['description'], 'species': entry['species'],
            'version': None, 'url': self.url, 'sets': sets,
        }


class STRINGSource(Source):
    id = 'string'
    name = 'STRING'
    description = 'Protein–protein interaction network (v12): partners of seed genes, or edges among a list'
    url = 'https://string-db.org/'
    kind = 'query'

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        return []

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        raise ValueError('STRING is a query service, not a library — use the partners or network endpoints')

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        raise ValueError('STRING is a query service, not a library — use the partners or network endpoints')

    @staticmethod
    def _taxon(species: str) -> int:
        if species not in SPECIES_TAXON:
            raise ValueError(f"STRING species must be one of {', '.join(SPECIES_TAXON)}, got '{species}'")
        return SPECIES_TAXON[species]

    @staticmethod
    def _identifiers(genes: list[str]) -> str:
        cleaned = [str(g).strip() for g in genes if str(g).strip()]
        if not cleaned:
            raise ValueError('No genes given')
        if len(cleaned) > MAX_STRING_IDENTIFIERS:
            raise ValueError(f'STRING accepts at most {MAX_STRING_IDENTIFIERS} identifiers per request; got {len(cleaned)}')
        return quote('\r'.join(cleaned), safe='')

    def map_ids(self, genes: list[str], species: str) -> tuple[dict[str, str], list[str]]:
        """``(query -> preferred STRING name, unmapped queries)``."""
        taxon = self._taxon(species)
        url = f'{STRING_BASE}get_string_ids?identifiers={self._identifiers(genes)}&species={taxon}&caller_identity=xcell'
        recs = fetch_json(url)
        mapping: dict[str, str] = {}
        for r in recs if isinstance(recs, list) else []:
            if isinstance(r, dict) and r.get('queryItem') and r.get('preferredName'):
                mapping.setdefault(str(r['queryItem']), str(r['preferredName']))
        unmapped = [g for g in genes if g not in mapping]
        return mapping, unmapped

    def partners(self, genes: list[str], species: str, *, limit: int = 50,
                 required_score: int = 400) -> dict[str, Any]:
        taxon = self._taxon(species)
        mapping, unmapped = self.map_ids(genes, species)
        seeds = [g for g in genes if g in mapping]
        sets: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        threshold = required_score / 1000.0
        for seed in seeds:
            url = (f'{STRING_BASE}interaction_partners?identifiers={quote(mapping[seed], safe="")}'
                   f'&species={taxon}&limit={int(limit)}&required_score={int(required_score)}&caller_identity=xcell')
            recs = fetch_json(url)
            partners: list[tuple[str, float]] = []
            for r in recs if isinstance(recs, list) else []:
                if not isinstance(r, dict):
                    continue
                score = float(r.get('score') or 0.0)
                if score < threshold:
                    continue
                partner = str(r.get('preferredName_B') or '')
                if partner and partner != mapping[seed]:
                    partners.append((partner, score))
            partners.sort(key=lambda t: -t[1])
            for partner, score in partners:
                edges.append({'a': seed, 'b': partner, 'score': round(score, 3)})
            sets.append({
                'name': f'STRING partners of {seed}',
                'genes': [seed] + [p for p, _ in partners],
                'scores': [1.0] + [round(s, 3) for _, s in partners],
                'description': f'{len(partners)} partners with combined score ≥ {required_score / 1000:.2f}',
                'url': f'https://string-db.org/network/{taxon}.{mapping[seed]}' if partners else self.url,
            })
        union: list[str] = []
        seen: set[str] = set()
        for s in sets:
            for g in s['genes']:
                if g not in seen:
                    seen.add(g)
                    union.append(g)
        return {'sets': sets, 'union': union, 'edges': edges, 'unmapped': unmapped, 'species': species}

    def network(self, genes: list[str], species: str, *, required_score: int = 400) -> dict[str, Any]:
        taxon = self._taxon(species)
        url = (f'{STRING_BASE}network?identifiers={self._identifiers(genes)}&species={taxon}'
               f'&required_score={int(required_score)}&caller_identity=xcell')
        recs = fetch_json(url)
        threshold = required_score / 1000.0
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for r in recs if isinstance(recs, list) else []:
            if not isinstance(r, dict):
                continue
            a, b = str(r.get('preferredName_A') or ''), str(r.get('preferredName_B') or '')
            score = float(r.get('score') or 0.0)
            if not a or not b or a == b or score < threshold:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            edges.append({'a': a, 'b': b, 'score': round(score, 3)})
        return {'edges': edges, 'species': species, 'n_genes': len(genes)}


SOURCES: dict[str, Source] = {
    s.id: s for s in (MSigDBSource(), EnrichrSource(), STRINGSource())
}


def get_source(source_id: str) -> Source:
    try:
        return SOURCES[source_id]
    except KeyError:
        raise ValueError(f"Unknown gene-set source '{source_id}'; expected one of {', '.join(SOURCES)}") from None


# --- cache ---------------------------------------------------------------------

def save_library(lib: dict[str, Any]) -> Path:
    path = _library_path(lib['source'], lib['species'], lib['id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(lib))
    tmp.replace(path)  # never leave a half-written library behind
    return path


def load_library(source: str, species: str, library_id: str) -> dict[str, Any] | None:
    path = _library_path(source, species, library_id)
    if not path.is_file():
        return None
    try:
        lib = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f'[xcell.gene_set_sources] Skipping unreadable cache {path.name}: {e}')
        return None
    return lib if isinstance(lib, dict) and isinstance(lib.get('sets'), list) else None


def library_summary(lib: dict[str, Any]) -> dict[str, Any]:
    return {k: lib.get(k) for k in ('source', 'id', 'name', 'description', 'species', 'version', 'url', 'fetched_at', 'n_sets')}


def find_library(source: str, library_id: str, species: str | None = None) -> dict[str, Any] | None:
    """Load a cached library, trying ``species`` first, then the other species.

    An Enrichr library is stored under *its own* species (a human-symbol GO
    library fetched while browsing mouse data lives under ``human``), so a
    lookup by the species the user is browsing must be allowed to miss.
    """
    order = ([species] if species in SPECIES else []) + [sp for sp in SPECIES if sp != species]
    for sp in order:
        lib = load_library(source, sp, library_id)
        if lib is not None:
            return lib
    return None


def list_cached() -> list[dict[str, Any]]:
    root = cache_dir()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob('*/*/*.json')):
        try:
            lib = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(lib, dict) and isinstance(lib.get('sets'), list):
            out.append(library_summary(lib))
    out.sort(key=lambda s: (s['source'] or '', s['species'] or '', (s['name'] or '').lower()))
    return out


def _catalogue_ttl_days() -> float:
    from xcell import config as user_config  # noqa: PLC0415

    cfg = user_config.get_user_config().get('gene_set_sources')
    raw = cfg.get('catalogue_ttl_days') if isinstance(cfg, dict) else None
    try:
        return float(raw) if raw is not None else float(CATALOGUE_TTL_DAYS)
    except (TypeError, ValueError):
        return float(CATALOGUE_TTL_DAYS)


def catalogue(source_id: str, species: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    """A source's libraries for ``species``, from the cached catalogue when fresh.

    Every entry carries ``cached`` and, once fetched, the real ``n_sets`` /
    ``version`` / ``fetched_at`` of the local copy.
    """
    src = get_source(source_id)
    check_species(species)
    path = _catalogue_path(source_id, species)
    entries: list[dict[str, Any]] | None = None
    if not refresh and path.is_file():
        age = time.time() - path.stat().st_mtime
        if age < _catalogue_ttl_days() * 86400:
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict) and isinstance(data.get('entries'), list):
                    entries = data['entries']
            except (OSError, ValueError):
                entries = None
    if entries is None:
        entries = src.catalogue(species)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'fetched_at': _now(), 'entries': entries}))

    cached = {(c['species'], c['id']): c for c in list_cached() if c['source'] == source_id}
    for e in entries:
        c = cached.get((e.get('species'), e.get('id')))
        e['cached'] = c is not None
        if c is not None:
            e['n_sets'] = c.get('n_sets')
            e['fetched_at'] = c.get('fetched_at')
            e['version'] = c.get('version') or e.get('version')
    return entries


def fetch_library(source_id: str, library_id: str, species: str,
                  report: Report | None = None) -> dict[str, Any]:
    """Download one library, cache it, and return it (with its sets)."""
    src = get_source(source_id)
    check_species(species)
    lib = src.fetch(library_id, species, report or (lambda *a, **k: None))
    lib['fetched_at'] = _now()
    lib['n_sets'] = len(lib['sets'])
    save_library(lib)
    return lib


# --- search --------------------------------------------------------------------

def search_sets(library: dict[str, Any], *, q: str = '', gene: str = '',
                offset: int = 0, limit: int = 50) -> dict[str, Any]:
    """Filter a cached library's sets by name/description text and/or member gene.

    Name-prefix matches rank first, then name substrings, then description
    hits; within a rank the library's own order is kept. Matching is
    case-insensitive because the same symbol is ``COL1A1`` in one library and
    ``Col1a1`` in another.
    """
    ql = q.strip().lower()
    gl = gene.strip().lower()
    ranked: list[tuple[int, dict[str, Any]]] = []
    for s in library.get('sets', []):
        name_l = str(s.get('name', '')).lower()
        rank = 0
        if ql:
            if name_l.startswith(ql):
                rank = 0
            elif ql in name_l:
                rank = 1
            elif ql in str(s.get('description') or '').lower():
                rank = 2
            else:
                continue
        if gl and gl not in {str(g).lower() for g in s.get('genes', [])}:
            continue
        ranked.append((rank, s))
    ranked.sort(key=lambda t: t[0])
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    page = [
        {
            'name': s['name'], 'description': s.get('description') or '', 'url': s.get('url') or '',
            'n_genes': len(s.get('genes', [])), 'genes': list(s.get('genes', [])),
        }
        for _, s in ranked[offset:offset + limit]
    ]
    return {'total': len(ranked), 'offset': offset, 'limit': limit, 'sets': page}


# --- STRING convenience --------------------------------------------------------

def string_partners(genes: list[str], species: str, *, limit: int = 50,
                    required_score: int = 400) -> dict[str, Any]:
    src = SOURCES['string']
    assert isinstance(src, STRINGSource)
    return src.partners(genes, species, limit=limit, required_score=required_score)


def string_network(genes: list[str], species: str, *, required_score: int = 400) -> dict[str, Any]:
    src = SOURCES['string']
    assert isinstance(src, STRINGSource)
    return src.network(genes, species, required_score=required_score)


# --- availability --------------------------------------------------------------

def availability() -> dict[str, Any]:
    """What the UI needs to draw the source list without touching the network."""
    cached = list_cached()
    return {
        'available': True,
        'sources': {
            sid: {'name': s.name, 'description': s.description, 'url': s.url, 'kind': s.kind}
            for sid, s in SOURCES.items()
        },
        'species': list(SPECIES),
        'cache_dir': str(cache_dir()),
        'n_cached': len(cached),
        'cached': cached,
    }
