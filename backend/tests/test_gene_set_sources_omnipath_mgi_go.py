"""OmniPath, MGI GXD and GO as gene-set sources — parsers, catalogues, fetches, caching.

Offline: every network function is stubbed with captured response shapes
(the headers and rows are verbatim from the live services on 2026-09-19).
"""
import gzip
import json

import pytest

from xcell import gene_set_sources as gss
from xcell import go_semantic as gos


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    for fn in ('fetch_text', 'fetch_json', 'fetch_bytes'):
        monkeypatch.setattr(gss, fn, lambda url, **kw: (_ for _ in ()).throw(AssertionError(f'network: {url}')))
    yield
    gss.set_cache_dir(None)


# --- OmniPath -----------------------------------------------------------------

COLLECTRI = (
    "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\t"
    "consensus_direction\tconsensus_stimulation\tconsensus_inhibition\tsources\treferences\tcuration_effort\n"
    "P01108\tO70372\tMyc\tTert\tTrue\tTrue\tFalse\tTrue\tTrue\tFalse\tCollecTRI;ExTRI_CollecTRI\tCollecTRI:10022617;ExTRI:1\t2\n"
    "P01108\tQ1\tMyc\tCdkn1a\tTrue\tFalse\tTrue\tTrue\tFalse\tTrue\tCollecTRI\tCollecTRI:2\t1\n"
    "P01108\tQ2\tMyc\tOdc1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCollecTRI\t\t1\n"
    "P01108\tQ3\tMyc\tBoth\tTrue\tTrue\tTrue\tTrue\tTrue\tTrue\tCollecTRI\t\t1\n"
    "P17433\tP86547\tSpi1\tBglap2\tTrue\tTrue\tFalse\tTrue\tTrue\tFalse\tCollecTRI;ExTRI_CollecTRI\tCollecTRI:10022617\t1\n"
    "P01108\tD6RFS9\tMyc\tD6RFS9\tTrue\tTrue\tFalse\tTrue\tTrue\tFalse\tCollecTRI\t\t1\n"
    "A0A8Q0P8A2\tP86547\tA0A8Q0P8A2\tBglap2\tTrue\tTrue\tFalse\tTrue\tTrue\tFalse\tCollecTRI\t\t1\n"
)

LIGREC = (
    "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\t"
    "consensus_direction\tconsensus_stimulation\tconsensus_inhibition\tsources\treferences\tcuration_effort\t"
    "entity_type_source\tentity_type_target\n"
    "P1\tCOMPLEX:O75578_P05556\tCOL1A1\tITGA10_ITGB1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCellPhoneDB;CellChatDB\tCellPhoneDB:1\t1\tprotein\tcomplex\n"
    "P1\tP2\tCOL1A1\tDDR1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCellChatDB\t\t1\tprotein\tprotein\n"
    "P1\tP2\tCOL1A1\tDDR1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCellChatDB\t\t1\tprotein\tprotein\n"
    "P3\tP2\tCOL2A1\tDDR1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tICELLNET\t\t1\tprotein\tprotein\n"
    "P1\tQ9WUP1\tCOL1A1\tQ9WUP1\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCellChatDB\t\t1\tprotein\tprotein\n"
    "P1\tC1\tCOL1A1\tITGA1_A0A0N4SUL9\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\tCellChatDB\t\t1\tprotein\tcomplex\n"
)


def test_collectri_regulons_are_signed_sets_named_after_the_tf():
    sets = gss.OmniPathSource.parse_collectri(COLLECTRI)
    by = {s['name']: s for s in sets}
    assert list(by) == ['Myc regulon', 'Spi1 regulon']
    myc = by['Myc regulon']
    # Repressed targets go down; unknown sign and conflicting sign count as activated.
    assert myc['genes'] == ['Tert', 'Odc1', 'Both']
    assert myc['genes_down'] == ['Cdkn1a']
    assert '4 targets: 3 activated or unknown, 1 repressed' in myc['description']
    assert '3 references' in myc['description']
    assert by['Spi1 regulon']['genes_down'] == []
    # A UniProt accession standing in for a missing symbol is neither a TF nor a target.
    assert 'D6RFS9' not in myc['genes'] and 'A0A8Q0P8A2 regulon' not in by


def test_ligrec_gives_the_receptors_of_each_ligand_and_the_ligands_of_each_receptor():
    sets, pairs = gss.OmniPathSource.parse_ligrec(LIGREC)
    by = {s['name']: s for s in sets}
    # complex flattened, duplicate row ignored, accession-only receptor dropped, accession subunit dropped
    assert by['COL1A1 receptors']['genes'] == ['ITGA10', 'ITGB1', 'DDR1', 'ITGA1']
    assert 'Q9WUP1 ligands' not in by
    assert by['DDR1 ligands']['genes'] == ['COL1A1', 'COL2A1']
    assert by['ITGA10_ITGB1 ligands']['genes'] == ['COL1A1']
    assert '3 receptors' in by['COL1A1 receptors']['description']
    assert 'CellChatDB, CellPhoneDB' in by['COL1A1 receptors']['description']
    assert len(pairs) == 4 and pairs[0] == {'ligand': 'COL1A1', 'receptor': 'ITGA10_ITGB1',
                                            'sources': ['CellPhoneDB', 'CellChatDB'], 'n_references': 1}


def test_omnipath_catalogue_and_urls_per_species():
    src = gss.OmniPathSource()
    ids = [c['id'] for c in src.catalogue('mouse')]
    assert ids == ['collectri', 'ligrec']
    assert 'organisms=10090' in src.collectri_url('mouse') and 'organisms=9606' in src.collectri_url('human')
    assert 'datasets=collectri' in src.collectri_url('mouse')
    assert 'datasets=omnipath,ligrecextra' in src.ligrec_url('mouse') and 'resources=CellPhoneDB' in src.ligrec_url('mouse')
    with pytest.raises(ValueError):
        src.validate_library('nope', 'mouse')


def test_omnipath_fetch_caches_a_directional_library_that_search_and_overlap_can_use(monkeypatch):
    calls = []

    def fake_text(url, **kw):
        calls.append(url)
        return COLLECTRI
    monkeypatch.setattr(gss, 'fetch_text', fake_text)
    lib = gss.fetch_library('omnipath', 'collectri', 'mouse')
    assert calls == [gss.OmniPathSource().collectri_url('mouse')]
    assert lib['n_sets'] == 2 and lib['version']  # the fetch date
    cached = gss.load_library('omnipath', 'mouse', 'collectri')
    assert cached is not None and cached['sets'][0]['genes_down'] == ['Cdkn1a']
    hit = gss.search_sets(cached, q='myc')['sets'][0]
    assert hit['genes_down'] == ['Cdkn1a'] and hit['n_genes'] == 4
    # A set is found by one of its down genes too.
    assert gss.search_sets(cached, gene='cdkn1a')['total'] == 1
    # A set without down genes does not grow the key.
    assert 'genes_down' not in gss.search_sets(cached, q='spi1')['sets'][0]


def test_omnipath_ligrec_fetch_keeps_the_pairs(monkeypatch):
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: LIGREC)
    lib = gss.fetch_library('omnipath', 'ligrec', 'human')
    assert len(lib['pairs']) == 4
    assert gss.load_library('omnipath', 'human', 'ligrec')['pairs'][1]['receptor'] == 'DDR1'


# --- MGI GXD --------------------------------------------------------------------

GXD = (
    "MGI Gene ID\tGene Symbol\tGene Name\tMGI Assay ID\tAssay Type\tAge\tTheiler Stage\tStructure\tCell Type\tDetected\t"
    "TPM Level (RNA-Seq)\tBiological Replicates (RNA-Seq)\tImages\tMutant Allele(s)\tStrain\tSex\tNotes (RNA-Seq)\t"
    "TPM (avg_quantile normalization)\tMGI Reference ID\tPubMed ID\tCitation\t\n"
    "MGI:1913964\t4921524J17Rik\tRIKEN cDNA\tMGI:5823563\tRNA in situ\tE11.5\t19\tforelimb bud\t\tYes\t\t\t\t\tSwiss Webster\t\t\t\tJ:226028\t26238476\tLewandowski JP\n"
    "MGI:98297\tShh\tsonic hedgehog\tMGI:1\tRNA in situ\tE11.5\t19\tforelimb bud\t\tYes\t\t\t\t\t\t\t\t\tJ:1\t1\tX\n"
    "MGI:98297\tShh\tsonic hedgehog\tMGI:2\tRNA-Seq\tE11.5\t19\tforelimb bud\t\tYes\tHigh\t\t\t\t\t\t\t\tJ:2\t2\tY\n"
    "MGI:98297\tShh\tsonic hedgehog\tMGI:3\tRNA in situ\tE11.5\t19\tlimb\t\tNo\t\t\t\t\t\t\t\t\tJ:3\t3\tZ\n"
    "MGI:1\tSox9\tSRY-box 9\tMGI:4\tImmunohistochemistry\tE11.5\t19\tlimb\t\tYes\t\t\t\t\t\t\t\t\tJ:4\t4\tW\n"
)


def test_gxd_report_becomes_one_set_per_structure_of_detected_genes():
    sets = gss.MGISource.parse_report(GXD, 19)
    by = {s['name']: s for s in sets}
    assert list(by) == ['forelimb bud', 'limb']
    assert by['forelimb bud']['genes'] == ['4921524J17Rik', 'Shh']  # Shh once, though assayed twice
    assert by['limb']['genes'] == ['Sox9']  # the Detected=No row is not a detection
    assert 'RNA in situ 2, RNA-Seq 1' in by['forelimb bud']['description']
    assert 'theilerStage=19' in by['limb']['url'] and 'structure=limb' in by['limb']['url']


def test_mgi_catalogue_is_the_theiler_stages_for_mouse_and_nothing_for_human():
    src = gss.MGISource()
    cat = src.catalogue('mouse')
    assert [c['id'] for c in cat][:3] == ['ts1', 'ts2', 'ts3'] and len(cat) == 28
    assert cat[18]['name'].startswith('TS19 · E11.5')
    assert src.catalogue('human') == []
    with pytest.raises(ValueError):
        src.validate_library('ts19', 'human')
    with pytest.raises(ValueError):
        src.validate_library('ts99', 'mouse')
    assert src.validate_library('TS19', 'mouse')['id'] == 'ts19'


def test_mgi_fetch_asks_for_detected_rows_at_the_stage_and_caches(monkeypatch):
    calls = []

    def fake_text(url, **kw):
        calls.append((url, kw.get('timeout')))
        return GXD
    monkeypatch.setattr(gss, 'fetch_text', fake_text)
    lib = gss.fetch_library('mgi', 'ts19', 'mouse')
    assert calls[0][0] == 'https://www.informatics.jax.org/gxd/report.txt?theilerStage=19&detected=Yes'
    assert calls[0][1] == 600  # a full-stage report is tens of MB
    assert lib['n_sets'] == 2
    assert gss.load_library('mgi', 'mouse', 'ts19')['sets'][0]['name'] == 'forelimb bud'


# --- GO as a source ---------------------------------------------------------------

OBO = """format-version: 1.2
data-version: releases/2026-07-26

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0001501
name: skeletal system development
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0002062
name: chondrocyte differentiation
namespace: biological_process
alt_id: GO:9999999
is_a: GO:0001501 ! skeletal system development

[Term]
id: GO:0030199
name: collagen fibril organization
namespace: biological_process
relationship: part_of GO:0001501 ! skeletal system development
relationship: regulates GO:0008150 ! biological_process

[Term]
id: GO:0000001
name: obsolete thing
namespace: biological_process
is_obsolete: true

[Term]
id: GO:0005576
name: extracellular region
namespace: cellular_component

[Typedef]
id: part_of
name: part of
"""

GAF = """!gaf-version: 2.2
!date-generated: 2026-05-21
MGI\tMGI:1\tSox9\tinvolved_in\tGO:0002062\tPMID:1\tIMP\t\tP\tSRY-box 9\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:1
MGI\tMGI:2\tCol2a1\tinvolved_in\tGO:9999999\tPMID:2\tIDA\t\tP\tcollagen II\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:2
MGI\tMGI:3\tCol1a1\tinvolved_in\tGO:0030199\tPMID:3\tIDA\t\tP\tcollagen I\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:3
MGI\tMGI:4\tRunx2\tinvolved_in\tGO:0001501\tPMID:4\tIMP\t\tP\trunt\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:4
MGI\tMGI:5\tActb\tNOT|involved_in\tGO:0002062\tPMID:5\tIMP\t\tP\tactin\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:5
MGI\tMGI:5\tActb\tlocated_in\tGO:0005576\tPMID:5\tIDA\t\tC\tactin\t\tprotein\ttaxon:10090\t20130101\tMGI\t\tMGI:MGI:5
ComplexPortal\tCPX-1\tcollagen complex\tpart_of\tGO:0005576\tPMID:6\tIDA\t\tC\tx\t\tprotein_complex\ttaxon:10090\t20130101\tComplexPortal\t\t
"""


def test_go_fetch_downloads_both_files_and_builds_term_sets(monkeypatch):
    urls = []

    def fake_bytes(url, **kw):
        urls.append(url)
        if url.endswith('.obo'):
            return OBO.encode()
        return gzip.compress(GAF.encode())
    monkeypatch.setattr(gss, 'fetch_bytes', fake_bytes)
    lib = gss.fetch_library('go', 'annotations', 'mouse')
    assert urls == [gos.OBO_URL, gos.GAF_URLS['mouse']]
    assert gos.obo_path().is_file() and gos.gaf_path('mouse').is_file()
    assert lib['version'] == 'releases/2026-07-26 · GAF 2026-05-21'
    names = [s['name'] for s in lib['sets']]
    # With min 5 genes nothing qualifies on this toy; the size bounds are the point.
    assert names == []
    small = gos.term_sets(gos.load_ontology(), gos.load_gaf('mouse'), min_genes=2, max_genes=3)
    by = {s['name']: s for s in small}
    # chondrocyte differentiation carries Sox9 and Col2a1 (via the alt_id); Actb's NOT annotation is excluded.
    assert by['chondrocyte differentiation (GO:0002062, BP)']['genes'] == ['Col2a1', 'Sox9']
    # skeletal system development has 4 genes with ancestors included — above max 3, so absent.
    assert 'skeletal system development (GO:0001501, BP)' not in by
    assert gss.availability()['cached'][0]['source'] == 'go'


def test_go_catalogue_is_one_library_per_species():
    src = gss.GOSource()
    assert [c['id'] for c in src.catalogue('human')] == ['annotations']
    with pytest.raises(ValueError):
        src.validate_library('bp', 'mouse')


def test_new_sources_are_registered_and_reported():
    av = gss.availability()
    assert {'omnipath', 'mgi', 'go'} <= set(av['sources'])
    assert av['sources']['mgi']['kind'] == 'library'
