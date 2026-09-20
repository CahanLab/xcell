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
* **OmniPath** — CollecTRI regulons (one signed set per transcription
  factor: activated targets up, repressed targets down) and curated
  ligand–receptor pairs (the receptors of each ligand, the ligands of each
  receptor), for mouse and human from the OmniPath web service.
* **MGI GXD** — one library per Theiler stage: the genes the Gene Expression
  Database has seen detected in each anatomical structure at that stage
  (in situ, blot and RNA-seq assays). Mouse only.
* **GO** — the ontology and a species' GAF from the GO Consortium, kept as
  raw files for :mod:`go_semantic` (the gene map's GO channel) and also
  offered as sets, one per term with ancestors included.

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


def fetch_bytes(url: str, *, timeout: float = TIMEOUT) -> bytes:
    """GET ``url`` raw — for gzipped files that must land on disk as they are."""
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        raise ValueError(f"Could not fetch {url}: {e}") from e


def parse_tsv(text: str) -> list[dict[str, str]]:
    """Header-keyed rows of a tab-separated table; a trailing tab adds no column."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip() for h in lines[0].rstrip('\t').split('\t')]
    rows: list[dict[str, str]] = []
    for ln in lines[1:]:
        cells = ln.split('\t')
        rows.append({h: (cells[i].strip() if i < len(cells) else '') for i, h in enumerate(header)})
    return rows


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


class OmniPathSource(Source):
    id = 'omnipath'
    name = 'OmniPath'
    description = 'CollecTRI transcription-factor regulons (signed) and curated ligand–receptor pairs'
    url = 'https://omnipathdb.org/'

    BASE = 'https://omnipathdb.org/interactions'
    #: Ligand–receptor resources OmniPath aggregates; the pairs are split
    #: across its ``omnipath`` and ``ligrecextra`` datasets.
    LR_RESOURCES = 'CellPhoneDB,CellChatDB,CellTalkDB,ICELLNET,connectomeDB2020,Cellinker,Baccin2019'
    LIBRARIES = [
        ('collectri', 'CollecTRI regulons',
         'One set per transcription factor: its activated targets as the set, repressed targets as '
         'the down genes (a directional set, so UCell scores TF activity). Targets of unknown sign '
         'count as activated, CollecTRI\'s own convention.'),
        ('ligrec', 'Ligand–receptor pairs',
         'The receptors of each ligand and the ligands of each receptor, from CellPhoneDB, CellChatDB, '
         'CellTalkDB, ICELLNET, connectomeDB2020, Cellinker and Baccin2019 as curated by OmniPath. '
         'Complexes contribute every subunit.'),
    ]

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        check_species(species)
        return [_entry(self.id, lid, name, desc, species, url=self.url) for lid, name, desc in self.LIBRARIES]

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        check_species(species)
        known = {lid: (name, desc) for lid, name, desc in self.LIBRARIES}
        if library_id not in known:
            raise ValueError(f"'{library_id}' is not an OmniPath library; expected one of {', '.join(known)}")
        name, desc = known[library_id]
        return _entry(self.id, library_id, name, desc, species, url=self.url)

    def collectri_url(self, species: str) -> str:
        return (f'{self.BASE}?datasets=collectri&organisms={SPECIES_TAXON[species]}'
                f'&genesymbols=yes&fields=sources,references,curation_effort')

    def ligrec_url(self, species: str) -> str:
        return (f'{self.BASE}?datasets=omnipath,ligrecextra&organisms={SPECIES_TAXON[species]}'
                f'&genesymbols=yes&resources={self.LR_RESOURCES}'
                f'&fields=sources,references,curation_effort,entity_type')

    @staticmethod
    def _true(v: str) -> bool:
        return str(v).strip().lower() in ('true', '1', 'yes')

    #: UniProt accession shapes (P12345, Q9WUP1, A0A8Q0P8A2). OmniPath prints
    #: the accession in the gene-symbol column when an entry has no symbol —
    #: unreviewed TrEMBL records, mostly — and no dataset spells a gene that way.
    _ACCESSION = re.compile(r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$')

    @classmethod
    def _is_symbol(cls, name: str) -> bool:
        return bool(name) and not cls._ACCESSION.match(name)

    @classmethod
    def _members(cls, symbol: str) -> list[str]:
        """A complex is written ``ITGA10_ITGB1``; a plain gene has no underscore
        (mouse symbols never carry one, human symbols only in rare readthroughs
        and those are not receptors). Accessions standing in for symbols are dropped."""
        return [m for m in str(symbol).split('_') if cls._is_symbol(m)]

    @classmethod
    def parse_collectri(cls, text: str) -> list[dict[str, Any]]:
        regulons: dict[str, dict[str, list[str]]] = {}
        refs: dict[str, set[str]] = {}
        for row in parse_tsv(text):
            tf, target = row.get('source_genesymbol', ''), row.get('target_genesymbol', '')
            if not cls._is_symbol(tf) or not cls._is_symbol(target):
                continue
            r = regulons.setdefault(tf, {'up': [], 'down': []})
            side = 'down' if cls._true(row.get('consensus_inhibition', '')) and not cls._true(row.get('consensus_stimulation', '')) else 'up'
            if target not in r['up'] and target not in r['down']:
                r[side].append(target)
            for ref in str(row.get('references', '')).split(';'):
                if ref.strip():
                    refs.setdefault(tf, set()).add(ref.strip())
        sets: list[dict[str, Any]] = []
        for tf in sorted(regulons, key=str.lower):
            up, down = regulons[tf]['up'], regulons[tf]['down']
            n_ref = len(refs.get(tf, ()))
            sets.append({
                'name': f'{tf} regulon',
                'description': f'{len(up) + len(down)} targets: {len(up)} activated or unknown, {len(down)} repressed'
                               + (f' · {n_ref} references' if n_ref else '') + ' · CollecTRI',
                'genes': up,
                'genes_down': down,
                'url': f'https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&sources={quote(tf)}',
            })
        return sets

    @classmethod
    def parse_ligrec(cls, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_ligand: dict[str, dict[str, Any]] = {}
        by_receptor: dict[str, dict[str, Any]] = {}
        pairs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in parse_tsv(text):
            lig, rec = row.get('source_genesymbol', ''), row.get('target_genesymbol', '')
            if not cls._members(lig) or not cls._members(rec) or (lig, rec) in seen:
                continue
            seen.add((lig, rec))
            sources = [x for x in str(row.get('sources', '')).split(';') if x]
            pairs.append({'ligand': lig, 'receptor': rec, 'sources': sources,
                          'n_references': len([x for x in str(row.get('references', '')).split(';') if x])})
            L = by_ligand.setdefault(lig, {'genes': [], 'partners': [], 'sources': set()})
            for m in cls._members(rec):
                if m not in L['genes']:
                    L['genes'].append(m)
            L['partners'].append(rec)
            L['sources'].update(sources)
            R = by_receptor.setdefault(rec, {'genes': [], 'partners': [], 'sources': set()})
            for m in cls._members(lig):
                if m not in R['genes']:
                    R['genes'].append(m)
            R['partners'].append(lig)
            R['sources'].update(sources)

        def describe(kind: str, entry: dict[str, Any]) -> str:
            n = len(entry['partners'])
            src = ', '.join(sorted(entry['sources']))
            return f"{n} {kind}{'' if n == 1 else 's'}" + (f' · {src}' if src else '') + ' · OmniPath'

        sets: list[dict[str, Any]] = []
        for lig in sorted(by_ligand, key=str.lower):
            e = by_ligand[lig]
            sets.append({'name': f'{lig} receptors', 'description': describe('receptor', e),
                         'genes': e['genes'], 'url': f'https://omnipathdb.org/interactions?datasets=omnipath,ligrecextra&genesymbols=yes&sources={quote(lig)}'})
        for rec in sorted(by_receptor, key=str.lower):
            e = by_receptor[rec]
            sets.append({'name': f'{rec} ligands', 'description': describe('ligand', e),
                         'genes': e['genes'], 'url': f'https://omnipathdb.org/interactions?datasets=omnipath,ligrecextra&genesymbols=yes&targets={quote(rec)}'})
        return sets, pairs

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        entry = self.validate_library(library_id, species)
        pairs: list[dict[str, Any]] = []
        if library_id == 'collectri':
            report(0.05, f'Downloading CollecTRI ({species})…')
            sets = self.parse_collectri(fetch_text(self.collectri_url(species)))
        else:
            report(0.05, f'Downloading ligand–receptor pairs ({species})…')
            sets, pairs = self.parse_ligrec(fetch_text(self.ligrec_url(species)))
        report(1.0, 'Done')
        # OmniPath publishes no data version; the fetch date is the version.
        lib = {
            'source': self.id, 'id': library_id, 'name': entry['name'], 'description': entry['description'],
            'species': species, 'version': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'url': self.url, 'sets': sets,
        }
        if pairs:
            lib['pairs'] = pairs
        return lib


class MGISource(Source):
    id = 'mgi'
    name = 'MGI GXD'
    description = 'Mouse Gene Expression Database: genes detected in each anatomical structure, one library per Theiler stage'
    url = 'https://www.informatics.jax.org/expression.shtml'

    REPORT_URL = 'https://www.informatics.jax.org/gxd/report.txt'
    SUMMARY_URL = 'https://www.informatics.jax.org/gxd/summary'
    #: Theiler stage → embryonic day, the label an embryologist reads.
    STAGES = {
        1: 'E0–0.9', 2: 'E1', 3: 'E2', 4: 'E3', 5: 'E4', 6: 'E4.5', 7: 'E5', 8: 'E6', 9: 'E6.5',
        10: 'E7', 11: 'E7.5', 12: 'E8', 13: 'E8.5', 14: 'E9', 15: 'E9.5', 16: 'E10', 17: 'E10.5',
        18: 'E11', 19: 'E11.5', 20: 'E12', 21: 'E13', 22: 'E14', 23: 'E15', 24: 'E16', 25: 'E17',
        26: 'E18', 27: 'P0–P3 (newborn)', 28: 'P4 to adult',
    }

    @staticmethod
    def _stage_of(library_id: str) -> int:
        m = re.fullmatch(r'ts(\d{1,2})', str(library_id).lower())
        if not m or int(m.group(1)) not in MGISource.STAGES:
            raise ValueError(f"'{library_id}' is not a Theiler stage library; expected ts1 … ts28")
        return int(m.group(1))

    def _entry_for(self, stage: int, species: str) -> dict[str, Any]:
        return _entry(
            self.id, f'ts{stage}', f'TS{stage} · {self.STAGES[stage]}',
            f'Genes GXD has seen detected in each anatomical structure (EMAPA) at Theiler stage {stage}, '
            'from in situ, blot, immunohistochemistry and RNA-seq assays; one set per structure.',
            species, url=f'{self.SUMMARY_URL}?theilerStage={stage}&detected=Yes',
        )

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        check_species(species)
        if species != 'mouse':
            return []  # the resource is mouse; an empty list, not an error, so the browser just shows nothing
        return [self._entry_for(st, species) for st in sorted(self.STAGES)]

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        check_species(species)
        if species != 'mouse':
            raise ValueError('MGI GXD is a mouse resource — choose the mouse species')
        return self._entry_for(self._stage_of(library_id), species)

    def report_url(self, stage: int) -> str:
        return f'{self.REPORT_URL}?theilerStage={stage}&detected=Yes'

    @classmethod
    def parse_report(cls, text: str, stage: int) -> list[dict[str, Any]]:
        """One set per structure from the GXD summary export.

        Only ``Detected == Yes`` rows count (the export can carry No / Ambiguous
        when asked for them); the assay-type mix is kept in the description so a
        set built from one RNA-seq experiment reads differently from one built
        from a hundred in situs.
        """
        by_structure: dict[str, dict[str, Any]] = {}
        for row in parse_tsv(text):
            if row.get('Detected', '') != 'Yes':
                continue
            structure, gene = row.get('Structure', ''), row.get('Gene Symbol', '')
            if not structure or not gene:
                continue
            e = by_structure.setdefault(structure, {'genes': [], 'seen': set(), 'assays': {}})
            if gene not in e['seen']:
                e['seen'].add(gene)
                e['genes'].append(gene)
            assay = row.get('Assay Type', '') or 'unspecified'
            e['assays'][assay] = e['assays'].get(assay, 0) + 1
        sets: list[dict[str, Any]] = []
        for structure in sorted(by_structure, key=str.lower):
            e = by_structure[structure]
            assays = ', '.join(f'{k} {v}' for k, v in sorted(e['assays'].items(), key=lambda kv: -kv[1]))
            sets.append({
                'name': structure,
                'description': f"{len(e['genes'])} genes detected at TS{stage} · assays: {assays}",
                'genes': sorted(e['genes'], key=str.lower),
                'url': f'{cls.SUMMARY_URL}?structure={quote(structure)}&theilerStage={stage}&detected=Yes',
            })
        return sets

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        entry = self.validate_library(library_id, species)
        stage = self._stage_of(library_id)
        report(0.05, f'Downloading GXD detected-expression report for TS{stage} (tens of MB)…')
        text = fetch_text(self.report_url(stage), timeout=600)
        report(0.8, 'Grouping by structure…')
        sets = self.parse_report(text, stage)
        report(1.0, 'Done')
        return {
            'source': self.id, 'id': library_id, 'name': entry['name'], 'description': entry['description'],
            'species': species, 'version': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'url': entry['url'], 'sets': sets,
        }


class GOSource(Source):
    id = 'go'
    name = 'Gene Ontology'
    description = 'The ontology and a species\' annotations (GAF): GO terms as sets, and the files the gene map\'s GO channel needs'
    url = 'https://geneontology.org/'

    LIBRARY_ID = 'annotations'

    def _entry_for(self, species: str) -> dict[str, Any]:
        return _entry(
            self.id, self.LIBRARY_ID, f'GO annotations ({species})',
            'go-basic.obo plus the species GAF from the GO Consortium. Fetching this is what switches on the '
            'GO channel of the gene map (IC-weighted semantic similarity). Also browsable as sets: one per term '
            'with 5–500 genes, ancestors included.',
            species, url=self.url,
        )

    def catalogue(self, species: str) -> list[dict[str, Any]]:
        check_species(species)
        return [self._entry_for(species)]

    def validate_library(self, library_id: str, species: str) -> dict[str, Any]:
        check_species(species)
        if library_id != self.LIBRARY_ID:
            raise ValueError(f"'{library_id}' is not a GO library; the only one is '{self.LIBRARY_ID}'")
        return self._entry_for(species)

    def fetch(self, library_id: str, species: str, report: Report) -> dict[str, Any]:
        from xcell import go_semantic as gos  # noqa: PLC0415

        entry = self.validate_library(library_id, species)
        gos.go_dir().mkdir(parents=True, exist_ok=True)
        report(0.02, 'Downloading go-basic.obo (~30 MB)…')
        obo = fetch_bytes(gos.OBO_URL, timeout=600)
        tmp = gos.obo_path().with_suffix('.obo.tmp')
        tmp.write_bytes(obo)
        tmp.replace(gos.obo_path())
        report(0.35, f'Downloading the {species} GAF (~15 MB)…')
        gaf = fetch_bytes(gos.GAF_URLS[species], timeout=600)
        tmp = gos.gaf_path(species).with_suffix('.gz.tmp')
        tmp.write_bytes(gaf)
        tmp.replace(gos.gaf_path(species))
        report(0.6, 'Reading the ontology…')
        onto = gos.load_ontology()
        report(0.75, 'Reading the annotations…')
        ann = gos.load_gaf(species)
        report(0.85, 'Building term sets…')
        sets = gos.term_sets(onto, ann)
        report(1.0, 'Done')
        version = ' · '.join(v for v in (onto.get('version') or '', f"GAF {ann.get('date')}" if ann.get('date') else '') if v)
        return {
            'source': self.id, 'id': library_id, 'name': entry['name'], 'description': entry['description'],
            'species': species, 'version': version or None, 'url': self.url, 'sets': sets,
            'files': {'obo': str(gos.obo_path()), 'gaf': str(gos.gaf_path(species))},
        }


SOURCES: dict[str, Source] = {
    s.id: s for s in (MSigDBSource(), EnrichrSource(), STRINGSource(),
                      OmniPathSource(), MGISource(), GOSource())
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
        if gl and gl not in {str(g).lower() for g in list(s.get('genes', [])) + list(s.get('genes_down') or [])}:
            continue
        ranked.append((rank, s))
    ranked.sort(key=lambda t: t[0])
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    page = []
    for _, s in ranked[offset:offset + limit]:
        down = list(s.get('genes_down') or [])
        rec = {
            'name': s['name'], 'description': s.get('description') or '', 'url': s.get('url') or '',
            'n_genes': len(s.get('genes', [])) + len(down), 'genes': list(s.get('genes', [])),
        }
        if down:
            rec['genes_down'] = down
        page.append(rec)
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
