"""Per-gene annotations from MyGene.info, cached locally.

One batch query returns, for each symbol (or Ensembl id), the RefSeq
summary, GO terms with evidence codes, InterPro domains, pathway
membership, HomoloGene orthologs and the identifiers that make links
(NCBI, Ensembl, MGI / HGNC, UniProt). Mouse genes often have no RefSeq
summary of their own; when HomoloGene names a human ortholog, its summary
is used and marked as such, in one extra batch call.

Records are cached in a SQLite file beside the gene-set library cache
(``~/.cache/xcell/gene_annotations.sqlite``), keyed by species and the
dataset's own spelling of the gene, so a symbol is fetched once per
machine. A gene MyGene does not know is cached as ``notfound`` so it is
not re-queried on every hover; ``refresh`` re-fetches.

Pure with respect to AnnData; the only network calls are the two
module-level functions, which tests monkeypatch.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MYGENE = 'https://mygene.info/v3/'
TAXON = {'mouse': 10090, 'human': 9606}
ENSEMBL_PREFIX = {'mouse': 'ENSMUSG', 'human': 'ENSG'}
FIELDS = ('symbol,name,summary,alias,type_of_gene,entrezgene,ensembl.gene,taxid,'
          'go,interpro,pathway,homologene,MGI,HGNC,uniprot.Swiss-Prot')
#: MyGene's documented per-request cap for batch queries.
BATCH = 1000
TIMEOUT = 60

#: Strength of GO evidence, for keeping the best code when a term repeats
#: and for ordering on the frontend. Experimental > curated > phylogenetic >
#: computational.
EVIDENCE_RANK = {
    'EXP': 5, 'IDA': 5, 'IPI': 5, 'IMP': 5, 'IGI': 5, 'IEP': 5,
    'HTP': 4, 'HDA': 4, 'HMP': 4, 'HGI': 4, 'HEP': 4,
    'TAS': 3, 'NAS': 3, 'IC': 3,
    'IBA': 2, 'IBD': 2, 'IKR': 2, 'IRD': 2, 'ISS': 2, 'ISO': 2, 'ISA': 2, 'ISM': 2, 'IGC': 2, 'RCA': 2,
    'IEA': 1, 'ND': 0,
}

_cache_path_override: Path | None = None


# --- network (module-level so tests monkeypatch them) ---------------------------

def _post(url: str, data: dict[str, Any]) -> Any:
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={
        'User-Agent': 'xcell (https://github.com/CahanLab/xcell)',
        'Content-Type': 'application/x-www-form-urlencoded',
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
    except Exception as e:
        raise ValueError(f'Could not reach {url}: {e}') from e
    try:
        return json.loads(raw.decode('utf-8', errors='replace'))
    except ValueError as e:
        raise ValueError(f'{url} did not return JSON: {e}') from e


def post_query(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = _post(MYGENE + 'query', payload)
    return out if isinstance(out, list) else []


def post_gene(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = _post(MYGENE + 'gene', payload)
    return out if isinstance(out, list) else []


# --- cache ------------------------------------------------------------------------

def set_cache_path(path: Path | str | None) -> None:
    global _cache_path_override
    _cache_path_override = Path(path) if path is not None else None


def cache_path() -> Path:
    if _cache_path_override is not None:
        return _cache_path_override
    from xcell import gene_set_sources as gss  # noqa: PLC0415

    return gss.cache_dir().parent / 'gene_annotations.sqlite'


def _db() -> sqlite3.Connection:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute('CREATE TABLE IF NOT EXISTS annotations '
                '(species TEXT NOT NULL, key TEXT NOT NULL, json TEXT NOT NULL, fetched_at TEXT NOT NULL, '
                'PRIMARY KEY (species, key))')
    return con


def _load(con: sqlite3.Connection, species: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(keys), 900):  # SQLite's bound-variable limit
        chunk = keys[i:i + 900]
        marks = ','.join('?' for _ in chunk)
        for key, blob in con.execute(
                f'SELECT key, json FROM annotations WHERE species = ? AND key IN ({marks})', [species, *chunk]):
            try:
                out[key] = json.loads(blob)
            except ValueError:
                continue
    return out


def _save(con: sqlite3.Connection, species: str, records: dict[str, dict[str, Any]]) -> None:
    now = _now()
    con.executemany('INSERT OR REPLACE INTO annotations (species, key, json, fetched_at) VALUES (?, ?, ?, ?)',
                    [(species, key, json.dumps(rec), now) for key, rec in records.items()])
    con.commit()


def stats(species: str) -> dict[str, Any]:
    con = _db()
    try:
        (n,) = con.execute('SELECT COUNT(*) FROM annotations WHERE species = ?', [species]).fetchone()
    finally:
        con.close()
    return {'species': species, 'n_cached': int(n), 'path': str(cache_path())}


def count_cached(species: str, keys: list[str]) -> int:
    con = _db()
    try:
        return len(_load(con, species, list(dict.fromkeys(str(k) for k in keys))))
    finally:
        con.close()


# --- parsing ----------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _is_ensembl(symbol: str) -> bool:
    return re.match(r'^ENS[A-Z]*G\d+', symbol) is not None


def parse_hit(hit: dict[str, Any], species: str) -> dict[str, Any]:
    """One MyGene hit → the compact record the UI shows and the cache stores."""
    q = str(hit.get('query', ''))
    if hit.get('notfound'):
        return {'symbol': q, 'query': q, 'notfound': True, 'fetched_at': _now()}

    prefix = ENSEMBL_PREFIX.get(species, 'ENS')
    ens_ids = [str(e['gene']) for e in _as_list(hit.get('ensembl')) if isinstance(e, dict) and e.get('gene')]
    ensembl = next((e for e in ens_ids if e.startswith(prefix)), ens_ids[0] if ens_ids else None)
    raw_entrez = hit.get('entrezgene')
    entrez = int(raw_entrez) if raw_entrez not in (None, '') and str(raw_entrez).isdigit() else None

    go: dict[str, list[dict[str, str]]] = {}
    for cat in ('BP', 'CC', 'MF'):
        terms: dict[str, dict[str, str]] = {}
        for t in _as_list((hit.get('go') or {}).get(cat)):
            if not isinstance(t, dict) or not t.get('id'):
                continue
            ev = str(t.get('evidence') or '')
            cur = terms.get(t['id'])
            if cur is None:
                terms[t['id']] = {'id': str(t['id']), 'term': str(t.get('term') or ''), 'evidence': ev}
            elif EVIDENCE_RANK.get(ev, 0) > EVIDENCE_RANK.get(cur['evidence'], 0):
                cur['evidence'] = ev
        go[cat] = list(terms.values())

    interpro = [
        {'id': str(d['id']), 'name': str(d.get('desc') or d.get('short_desc') or ''), 'short': str(d.get('short_desc') or '')}
        for d in _as_list(hit.get('interpro')) if isinstance(d, dict) and d.get('id')
    ]
    pathways: dict[str, list[dict[str, str]]] = {}
    raw_pw = hit.get('pathway') or {}
    if isinstance(raw_pw, dict):
        for src, items in raw_pw.items():
            lst = [{'id': str(p.get('id')), 'name': str(p.get('name') or '')} for p in _as_list(items)
                   if isinstance(p, dict) and p.get('id')]
            if lst:
                pathways[str(src)] = lst
    homologene: dict[str, int] = {}
    hg = hit.get('homologene')
    if isinstance(hg, dict):
        for pair in _as_list(hg.get('genes')):
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[1]).isdigit():
                homologene[str(pair[0])] = int(pair[1])
    uni = hit.get('uniprot')
    uniprot = uni.get('Swiss-Prot') if isinstance(uni, dict) else None
    if isinstance(uniprot, list):
        uniprot = uniprot[0] if uniprot else None
    mgi = hit.get('MGI')
    hgnc = hit.get('HGNC')
    links: dict[str, str] = {}
    if entrez:
        links['ncbi'] = f'https://www.ncbi.nlm.nih.gov/gene/{entrez}'
    if ensembl:
        links['ensembl'] = f'https://www.ensembl.org/id/{ensembl}'
    if mgi:
        links['mgi'] = f'https://www.informatics.jax.org/marker/{mgi}'
    if hgnc:
        links['hgnc'] = f'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:{hgnc}'
    if uniprot:
        links['uniprot'] = f'https://www.uniprot.org/uniprotkb/{uniprot}'
    summary = str(hit.get('summary') or '').strip()
    return {
        'symbol': str(hit.get('symbol') or q), 'query': q, 'notfound': False,
        'name': str(hit.get('name') or ''),
        'summary': summary, 'summary_source': 'refseq' if summary else None,
        'type': hit.get('type_of_gene'),
        'aliases': [str(a) for a in _as_list(hit.get('alias'))],
        'entrez': entrez, 'ensembl': ensembl, 'mgi': mgi, 'hgnc': hgnc, 'uniprot': uniprot,
        'taxid': hit.get('taxid') or TAXON.get(species),
        'go': go, 'interpro': interpro, 'pathways': pathways,
        'homologene': homologene, 'orthologs': {}, 'links': links,
        'fetched_at': _now(),
    }


def pick_hits(hits: list[dict[str, Any]], species: str) -> dict[str, dict[str, Any]]:
    """Best hit per query: right species, has an Entrez id, symbol matches exactly."""
    taxid = TAXON.get(species)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for h in hits:
        if isinstance(h, dict) and 'query' in h:
            grouped.setdefault(str(h['query']), []).append(h)
    best: dict[str, dict[str, Any]] = {}
    for q, cands in grouped.items():
        real = [c for c in cands if not c.get('notfound')]
        if not real:
            best[q] = {'query': q, 'notfound': True}
            continue

        def score(c: dict[str, Any]) -> int:
            return (2 * int(c.get('taxid') == taxid)
                    + int(c.get('entrezgene') not in (None, ''))
                    + int(str(c.get('symbol', '')).upper() == q.upper()))
        best[q] = max(real, key=score)
    return best


def _ortholog_fill(records: dict[str, dict[str, Any]], species: str) -> None:
    """Attach the other species' ortholog symbol; borrow its summary when ours is empty."""
    other = 'human' if species == 'mouse' else 'mouse'
    other_tax = str(TAXON[other])
    wanted = sorted({r['homologene'][other_tax] for r in records.values()
                     if not r.get('notfound') and r.get('homologene', {}).get(other_tax)})
    if not wanted:
        return
    lookup: dict[int, dict[str, Any]] = {}
    for i in range(0, len(wanted), BATCH):
        chunk = wanted[i:i + BATCH]
        for h in post_gene({'ids': ','.join(str(x) for x in chunk), 'fields': 'symbol,name,summary,taxid'}):
            if isinstance(h, dict) and not h.get('notfound') and str(h.get('query', '')).isdigit():
                lookup[int(h['query'])] = h
    for r in records.values():
        if r.get('notfound'):
            continue
        oid = r.get('homologene', {}).get(other_tax)
        h = lookup.get(oid) if oid else None
        if not h:
            continue
        r['orthologs'][other] = {'symbol': str(h.get('symbol') or ''), 'entrez': int(oid)}
        if not r['summary'] and h.get('summary'):
            r['summary'] = str(h['summary']).strip()
            r['summary_source'] = f"ortholog:{h.get('symbol') or oid}"


# --- public -----------------------------------------------------------------------

def get_annotations(symbols: list[str], species: str, *, refresh: bool = False,
                    report: Callable[..., None] | None = None) -> dict[str, dict[str, Any]]:
    """Annotations for ``symbols`` (dataset spelling), from the cache or MyGene."""
    if species not in TAXON:
        raise ValueError(f"species must be one of {', '.join(TAXON)}; got '{species}'")
    keys = list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
    con = _db()
    try:
        cached = {} if refresh else _load(con, species, keys)
        missing = [k for k in keys if k not in cached]
        if missing:
            by_scope: dict[str, list[str]] = {'ensembl.gene': [], 'symbol,alias': []}
            for k in missing:
                by_scope['ensembl.gene' if _is_ensembl(k) else 'symbol,alias'].append(k)
            chunks = [(scope, items[i:i + BATCH]) for scope, items in by_scope.items()
                      for i in range(0, len(items), BATCH)]
            records: dict[str, dict[str, Any]] = {}
            for n, (scope, chunk) in enumerate(chunks):
                if report:
                    report(0.05 + 0.8 * n / max(1, len(chunks)), f'Fetching annotations {n + 1}/{len(chunks)}…')
                hits = post_query({'q': ','.join(c.replace(',', ' ') for c in chunk), 'scopes': scope,
                                   'species': str(TAXON[species]), 'fields': FIELDS})
                best = pick_hits(hits, species)
                for q in chunk:
                    records[q] = parse_hit(best.get(q, {'query': q, 'notfound': True}), species)
            if report:
                report(0.9, 'Looking up orthologs…')
            _ortholog_fill(records, species)
            _save(con, species, records)
            cached.update(records)
        if report:
            report(1.0, 'Done')
        return {k: cached[k] for k in keys if k in cached}
    finally:
        con.close()
