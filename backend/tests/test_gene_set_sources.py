"""External gene-set sources: parsing, release discovery, caching, search, STRING.

Everything runs offline: ``fetch_text`` / ``fetch_json`` are monkeypatched with
captured response shapes. A source that reaches the network in a test is a bug.
"""
import json
import time

import pytest

from xcell import gene_set_sources as gss


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    # Any un-stubbed network call is a test failure, not a slow test.
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(f'network: {url}')))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(f'network: {url}')))
    yield
    gss.set_cache_dir(None)


# --- GMT parsing --------------------------------------------------------------

def test_parse_gmt_strips_enrichr_weights_and_blank_tokens():
    text = "Set A\t\tCOL1A1,1.0\tCOL1A2,1.0\t\t\nSet B\thttps://x/y\tSox9\tRunx2\n\n"
    sets = gss.parse_gmt(text)
    assert [s['name'] for s in sets] == ['Set A', 'Set B']
    assert sets[0]['genes'] == ['COL1A1', 'COL1A2']
    assert sets[0]['description'] == ''
    assert sets[1]['description'] == 'https://x/y'
    assert sets[1]['genes'] == ['Sox9', 'Runx2']


def test_parse_gmt_deduplicates_genes_within_a_set():
    assert gss.parse_gmt("S\t\tA\tB\tA\n")[0]['genes'] == ['A', 'B']


# --- MSigDB -------------------------------------------------------------------

INDEX_HTML = '<a href="2024.1.Hs/">x</a> <a href="2026.1.Mm/">y</a> <a href="2026.1.Hs/">z</a> <a href="2025.1.Hs/">w</a>'


def test_msigdb_latest_release_picks_newest_for_species(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: INDEX_HTML)
    assert gss.MSigDBSource().latest_release('human') == '2026.1'
    assert gss.MSigDBSource().latest_release('mouse') == '2026.1'


def test_msigdb_latest_release_falls_back_to_pinned_when_index_unreachable(monkeypatch):
    def boom(url, **kw):
        raise ValueError('offline')
    monkeypatch.setattr(gss, 'fetch_text', boom)
    assert gss.MSigDBSource().latest_release('mouse') == gss.MSIGDB_PINNED_RELEASE


def test_msigdb_catalogue_lists_collections_per_species():
    cat = gss.MSigDBSource().catalogue('mouse')
    ids = [c['id'] for c in cat]
    assert 'mh.all' in ids and 'm2.cgp' in ids and 'm5.go.bp' in ids
    assert all(c['species'] == 'mouse' for c in cat)
    assert 'c2.cgp' not in ids  # human-only prefix
    human = [c['id'] for c in gss.MSigDBSource().catalogue('human')]
    assert 'h.all' in human and 'c2.cgp' in human and 'c8.all' in human


def test_msigdb_fetch_parses_gmt_merges_json_metadata_and_caches(monkeypatch):
    gmt = ("NABA_COLLAGENS\thttps://www.gsea-msigdb.org/gsea/msigdb/mouse/geneset/NABA_COLLAGENS\tCol1a1\tCol1a2\tCol2a1\n"
           "HALLMARK_APOPTOSIS\thttps://x\tBax\tBak1\n")
    meta = {"NABA_COLLAGENS": {"collection": "M2:CGP", "pmid": "22159717", "msigdbURL": "https://msig/NABA_COLLAGENS",
                               "exactSource": "Naba et al.", "geneSymbols": ["Col1a1", "Col1a2", "Col2a1"]}}
    calls = []

    def fake_text(url, **kw):
        calls.append(url)
        if url.endswith('/'):
            return INDEX_HTML
        assert url.endswith('m2.cgp.v2026.1.Mm.symbols.gmt'), url
        return gmt

    def fake_json(url, **kw):
        calls.append(url)
        assert url.endswith('m2.cgp.v2026.1.Mm.json'), url
        return meta

    monkeypatch.setattr(gss, 'fetch_text', fake_text)
    monkeypatch.setattr(gss, 'fetch_json', fake_json)

    progress = []
    lib = gss.fetch_library('msigdb', 'm2.cgp', 'mouse', report=lambda f, m=None: progress.append((f, m)))
    assert lib['source'] == 'msigdb' and lib['id'] == 'm2.cgp' and lib['species'] == 'mouse'
    assert lib['version'] == '2026.1'
    assert lib['n_sets'] == 2
    by_name = {s['name']: s for s in lib['sets']}
    assert by_name['NABA_COLLAGENS']['genes'] == ['Col1a1', 'Col1a2', 'Col2a1']
    assert by_name['NABA_COLLAGENS']['url'] == 'https://msig/NABA_COLLAGENS'
    assert 'Naba' in by_name['NABA_COLLAGENS']['description']
    assert by_name['HALLMARK_APOPTOSIS']['url'] == 'https://x'  # GMT column 2 when no JSON record
    assert progress and progress[-1][0] == 1.0

    # Cached on disk and reloadable without the network.
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError('network')))
    again = gss.load_library('msigdb', 'mouse', 'm2.cgp')
    assert again is not None and again['n_sets'] == 2 and again['fetched_at']
    cached = gss.list_cached()
    assert [(c['source'], c['species'], c['id'], c['n_sets']) for c in cached] == [('msigdb', 'mouse', 'm2.cgp', 2)]


def test_msigdb_fetch_survives_missing_json_metadata(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: INDEX_HTML if url.endswith('/') else "S1\thttps://u\tA\tB\n")

    def no_json(url, **kw):
        raise ValueError('404')
    monkeypatch.setattr(gss, 'fetch_json', no_json)
    lib = gss.fetch_library('msigdb', 'mh.all', 'mouse')
    assert lib['sets'][0]['url'] == 'https://u' and lib['sets'][0]['description'] == ''


# --- Enrichr ------------------------------------------------------------------

STATS = {"statistics": [
    {"libraryName": "GO_Biological_Process_2026", "numTerms": 5000, "geneCoverage": 14000, "genesPerTerm": 30, "link": "http://geneontology.org", "categoryId": 2},
    {"libraryName": "WikiPathways_2024_Mouse", "numTerms": 200, "geneCoverage": 3000, "genesPerTerm": 40, "link": "", "categoryId": 2},
]}


def test_enrichr_catalogue_tags_species_and_counts(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: STATS)
    cat = gss.catalogue('enrichr', 'mouse')
    by_id = {c['id']: c for c in cat}
    assert by_id['GO_Biological_Process_2026']['species'] == 'human'
    assert by_id['GO_Biological_Process_2026']['n_sets'] == 5000
    assert by_id['GO_Biological_Process_2026']['name'] == 'GO Biological Process 2026'
    assert by_id['WikiPathways_2024_Mouse']['species'] == 'mouse'
    assert by_id['GO_Biological_Process_2026']['cached'] is False


def test_enrichr_fetch_parses_text_library(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: STATS)
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: "collagen fibril organization (GO:0030199)\t\tCOL1A1,1.0\tCOL5A1,1.0\n")
    lib = gss.fetch_library('enrichr', 'GO_Biological_Process_2026', 'mouse')
    assert lib['species'] == 'human'  # the library's own species, not the requested one
    assert lib['sets'][0]['genes'] == ['COL1A1', 'COL5A1']
    assert lib['sets'][0]['name'].startswith('collagen fibril')
    assert gss.load_library('enrichr', 'human', 'GO_Biological_Process_2026') is not None


# --- catalogue caching --------------------------------------------------------

def test_catalogue_is_cached_and_refreshed_only_when_stale(monkeypatch, tmp_path):
    n = {'calls': 0}

    def stats(url, **kw):
        n['calls'] += 1
        return STATS
    monkeypatch.setattr(gss, 'fetch_json', stats)
    gss.catalogue('enrichr', 'human')
    gss.catalogue('enrichr', 'human')
    assert n['calls'] == 1
    gss.catalogue('enrichr', 'human', refresh=True)
    assert n['calls'] == 2
    # Age the file past the TTL: it is refetched.
    path = gss._catalogue_path('enrichr', 'human')
    old = time.time() - (gss.CATALOGUE_TTL_DAYS + 1) * 86400
    import os
    os.utime(path, (old, old))
    gss.catalogue('enrichr', 'human')
    assert n['calls'] == 3


def test_catalogue_marks_cached_libraries(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: STATS)
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: "T\t\tA\tB\n")
    gss.fetch_library('enrichr', 'WikiPathways_2024_Mouse', 'mouse')
    cat = {c['id']: c for c in gss.catalogue('enrichr', 'mouse')}
    assert cat['WikiPathways_2024_Mouse']['cached'] is True
    assert cat['WikiPathways_2024_Mouse']['n_sets'] == 1  # the real count once fetched
    assert cat['GO_Biological_Process_2026']['cached'] is False


def test_unknown_source_and_library_are_value_errors(monkeypatch):
    with pytest.raises(ValueError, match='Unknown gene-set source'):
        gss.catalogue('nope', 'mouse')
    with pytest.raises(ValueError, match='not a MSigDB collection'):
        gss.fetch_library('msigdb', 'zz.all', 'mouse')


# --- search -------------------------------------------------------------------

def _lib():
    return {'source': 'msigdb', 'id': 'x', 'species': 'mouse', 'sets': [
        {'name': 'NABA_COLLAGENS', 'description': 'collagen genes', 'genes': ['Col1a1', 'Col2a1'], 'url': ''},
        {'name': 'NABA_CORE_MATRISOME', 'description': 'ECM', 'genes': ['Col1a1', 'Acan', 'Fn1'], 'url': ''},
        {'name': 'HALLMARK_APOPTOSIS', 'description': 'death', 'genes': ['Bax'], 'url': ''},
    ]}


def test_search_by_name_is_case_insensitive_and_prefix_first():
    out = gss.search_sets(_lib(), q='naba')
    assert out['total'] == 2 and [s['name'] for s in out['sets']] == ['NABA_COLLAGENS', 'NABA_CORE_MATRISOME']
    out = gss.search_sets(_lib(), q='matrisome')
    assert [s['name'] for s in out['sets']] == ['NABA_CORE_MATRISOME']
    assert out['sets'][0]['n_genes'] == 3
    # description text is searched too
    assert gss.search_sets(_lib(), q='death')['total'] == 1


def test_search_by_member_gene_is_case_insensitive():
    out = gss.search_sets(_lib(), gene='col1a1')
    assert sorted(s['name'] for s in out['sets']) == ['NABA_COLLAGENS', 'NABA_CORE_MATRISOME']


def test_search_pages_and_reports_total():
    out = gss.search_sets(_lib(), offset=1, limit=1)
    assert out['total'] == 3 and len(out['sets']) == 1 and out['offset'] == 1


# --- STRING -------------------------------------------------------------------

PARTNERS = [
    {"stringId_A": "10090.A", "stringId_B": "10090.B", "preferredName_A": "Col1a1", "preferredName_B": "Col3a1", "ncbiTaxonId": 10090, "score": 0.999},
    {"stringId_A": "10090.A", "stringId_B": "10090.C", "preferredName_A": "Col1a1", "preferredName_B": "Itgb1", "ncbiTaxonId": 10090, "score": 0.988},
    {"stringId_A": "10090.A", "stringId_B": "10090.D", "preferredName_A": "Col1a1", "preferredName_B": "Weak", "ncbiTaxonId": 10090, "score": 0.30},
]
IDS = [{"queryItem": "Col1a1", "stringId": "10090.A", "preferredName": "Col1a1"}]


def test_string_partners_builds_a_set_and_edges_above_required_score(monkeypatch):
    urls = []

    def fake_json(url, **kw):
        urls.append(url)
        if 'get_string_ids' in url:
            return IDS
        assert 'interaction_partners' in url and 'species=10090' in url
        return PARTNERS
    monkeypatch.setattr(gss, 'fetch_json', fake_json)
    out = gss.string_partners(['Col1a1', 'NotAGene'], 'mouse', limit=10, required_score=400)
    assert out['unmapped'] == ['NotAGene']
    assert out['sets'][0]['name'] == 'STRING partners of Col1a1'
    assert out['sets'][0]['genes'] == ['Col1a1', 'Col3a1', 'Itgb1']  # seed first, Weak (0.30) dropped
    assert out['edges'] == [{'a': 'Col1a1', 'b': 'Col3a1', 'score': 0.999}, {'a': 'Col1a1', 'b': 'Itgb1', 'score': 0.988}]
    assert any('caller_identity=xcell' in u for u in urls)


def test_string_network_returns_edges_among_the_list(monkeypatch):
    net = [{"preferredName_A": "Col1a1", "preferredName_B": "Col1a2", "score": 0.99},
           {"preferredName_A": "Col1a1", "preferredName_B": "Col2a1", "score": 0.5}]
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: net)
    out = gss.string_network(['Col1a1', 'Col1a2', 'Col2a1'], 'mouse', required_score=700)
    assert out['edges'] == [{'a': 'Col1a1', 'b': 'Col1a2', 'score': 0.99}]


def test_string_rejects_unknown_species():
    with pytest.raises(ValueError, match='species'):
        gss.string_network(['A'], 'zebrafish')


# --- availability -------------------------------------------------------------

def test_availability_reports_sources_and_cache_dir(tmp_path):
    info = gss.availability()
    assert set(info['sources']) >= {'msigdb', 'enrichr', 'string'}
    assert info['cache_dir'].startswith(str(tmp_path))
