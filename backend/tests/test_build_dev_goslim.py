"""The developmental GO slim build: table validation, closure, bundle shape.

``xcell/data/build_dev_goslim.py`` turns ``dev_goslim.tsv`` (coarse terms and
the fine terms nested under them) into four gene-set bundles. A wrong GO id,
an obsolete one, or a fine term quietly placed under a coarse term it does
not descend from would ship as a plausible-looking set, so the validator is
the contract. A toy ontology and GAF keep the tests offline.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from xcell import gene_set_library
from xcell import go_semantic as gos

_SPEC = importlib.util.spec_from_file_location(
    'build_dev_goslim', Path(gene_set_library.LIBRARY_DIR).parent / 'build_dev_goslim.py')
bdg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bdg)

OBO = """format-version: 1.2
data-version: releases/2026-07-26

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0030154
name: cell differentiation
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0001501
name: skeletal system development
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0002062
name: chondrocyte differentiation
namespace: biological_process
is_a: GO:0001501 ! skeletal system development
is_a: GO:0030154 ! cell differentiation

[Term]
id: GO:0030199
name: collagen fibril organization
namespace: biological_process
relationship: part_of GO:0001501 ! skeletal system development

[Term]
id: GO:0001649
name: osteoblast differentiation
namespace: biological_process
is_a: GO:0030154 ! cell differentiation

[Term]
id: GO:0016055
name: Wnt signaling pathway
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0007219
name: Notch signaling pathway
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0005576
name: extracellular region
namespace: cellular_component

[Term]
id: GO:0000001
name: obsolete thing
namespace: biological_process
is_obsolete: true
"""

GAF_ROWS = [
    ('Sox9', 'GO:0002062'), ('Col2a1', 'GO:0002062'), ('Col1a1', 'GO:0030199'), ('Runx2', 'GO:0001649'),
    ('Ctnnb1', 'GO:0016055'), ('Notch1', 'GO:0007219'), ('Actb', 'GO:0005576'),
]

TABLE = """level\tparent\tgo_id\tname\tmouse_genes\tcross
coarse\t\tGO:0001501\tskeletal system development\t0\t
fine\tGO:0001501\tGO:0002062\tchondrocyte differentiation\t0\t
fine\tGO:0001501\tGO:0030199\tcollagen fibril organization\t0\t
fine\tGO:0001501\tGO:0001649\tosteoblast differentiation\t0\tx
coarse\t\tunion:signaling\tdevelopmental signaling pathways\t0\t
fine\tunion:signaling\tGO:0016055\tWnt signaling pathway\t0\t
fine\tunion:signaling\tGO:0007219\tNotch signaling pathway\t0\t
"""


def _gaf():
    lines = ['!gaf-version: 2.2', '!date-generated: 2026-05-21']
    for i, (sym, term) in enumerate(GAF_ROWS):
        lines.append('\t'.join(['MGI', f'MGI:{i}', sym, 'involved_in', term, 'PMID:1', 'IDA', '', 'P', 'x', '',
                                'protein', 'taxon:10090', '20130101', 'MGI', '', '']))
    return gos.parse_gaf(lines, 'mouse')


def _onto():
    return gos.parse_obo(OBO)


def _rows(text=TABLE, tmp_path=None):
    p = (tmp_path or Path('/tmp')) / 'dev_goslim.tsv'
    p.write_text(text)
    return bdg.read_table(p)


def test_a_valid_table_has_no_problems(tmp_path):
    assert bdg.validate(_rows(tmp_path=tmp_path), _onto()) == []


@pytest.mark.parametrize('bad_row, expect', [
    ('fine\tGO:0001501\tGO:9999999\tnot a term\t0\t', 'GO:9999999'),
    ('fine\tGO:0001501\tGO:0000001\tobsolete thing\t0\t', 'obsolete'),
    ('fine\tGO:0001501\tGO:0030199\tcollagen fibre organisation\t0\t', 'name'),
    ('fine\tGO:0001501\tGO:0005576\textracellular region\t0\tx', 'biological_process'),
    ('fine\tGO:0001501\tGO:0001649\tosteoblast differentiation\t0\t', 'descend'),
    ('fine\tGO:0001501\tGO:0002062\tchondrocyte differentiation\t0\t', 'twice'),
    ('fine\tGO:0099999\tGO:0007219\tNotch signaling pathway\t0\t', 'parent'),
], ids=['unknown-id', 'obsolete', 'name-mismatch', 'not-bp', 'cross-link-unflagged', 'duplicate', 'orphan'])
def test_each_table_fault_is_named(tmp_path, bad_row, expect):
    rows = _rows(TABLE + bad_row + '\n', tmp_path)
    problems = bdg.validate(rows, _onto())
    assert problems, 'the fault was accepted'
    assert any(expect in p for p in problems), problems


def test_coarse_sets_are_the_closure_or_the_union_and_cross_links_stay_out(tmp_path):
    rows = _rows(tmp_path=tmp_path)
    bundle = bdg.build(rows, _onto(), _gaf(), 'mouse', 'coarse')
    by = {s['name']: s for s in bundle['sets']}
    # Runx2 is under osteoblast differentiation, a cross-link: in the folder, not in the coarse set.
    assert by['skeletal system development (GO:0001501)']['genes'] == ['Col1a1', 'Col2a1', 'Sox9']
    assert by['developmental signaling pathways']['genes'] == ['Ctnnb1', 'Notch1']
    assert 'folder' not in by['skeletal system development (GO:0001501)']
    assert 'releases/2026-07-26' in bundle['description'] and '2026-05-21' in bundle['description']
    assert bundle['name'] == 'Developmental GO slim, coarse (mouse)'


def test_fine_bundle_has_one_folder_per_coarse_term_and_parses(tmp_path):
    rows = _rows(tmp_path=tmp_path)
    bundle = bdg.build(rows, _onto(), _gaf(), 'mouse', 'fine')
    folders: dict[str, list[str]] = {}
    for s in bundle['sets']:
        folders.setdefault(s['folder'], []).append(s['name'])
    assert folders == {
        'skeletal system development': ['chondrocyte differentiation (GO:0002062)',
                                        'collagen fibril organization (GO:0030199)',
                                        'osteoblast differentiation (GO:0001649)'],
        'developmental signaling pathways': ['Wnt signaling pathway (GO:0016055)', 'Notch signaling pathway (GO:0007219)'],
    }
    by = {s['name']: s for s in bundle['sets']}
    assert by['osteoblast differentiation (GO:0001649)']['genes'] == ['Runx2']
    assert 'GO:0001501' in by['chondrocyte differentiation (GO:0002062)']['description']
    path = tmp_path / 'dev_goslim_fine_mouse.json'
    path.write_text(json.dumps(bundle))
    parsed = gene_set_library._parse_bundle(path)
    assert parsed['count'] == 5 and parsed['sets'][0]['folder'] == 'skeletal system development'


def test_two_unions_keep_their_own_fine_terms(tmp_path):
    # The first build had one id 'union' for every union row, so each union took the others' terms.
    extra = ('coarse\t\tunion:cranio\tcraniofacial development\t0\t\n'
             'fine\tunion:cranio\tGO:0030154\tcell differentiation\t0\t\n')
    rows = _rows(TABLE + extra, tmp_path)
    assert bdg.validate(rows, _onto()) == []
    coarse = {s['name']: s['genes'] for s in bdg.build(rows, _onto(), _gaf(), 'mouse', 'coarse')['sets']}
    assert coarse['developmental signaling pathways'] == ['Ctnnb1', 'Notch1']
    assert coarse['craniofacial development'] == ['Col2a1', 'Runx2', 'Sox9']
    fine = bdg.build(rows, _onto(), _gaf(), 'mouse', 'fine')['sets']
    assert [s['name'] for s in fine if s['folder'] == 'craniofacial development'] == ['cell differentiation (GO:0030154)']
    assert len(fine) == 6
    dup = _rows(TABLE + 'coarse\t\tunion:signaling\tanother\t0\t\n', tmp_path)
    assert any('twice' in p for p in bdg.validate(dup, _onto()))


def test_a_term_with_no_annotated_gene_is_left_out_with_a_note(tmp_path):
    rows = _rows(tmp_path=tmp_path)
    gaf = _gaf()
    del gaf['annotations']['Notch1']
    bundle = bdg.build(rows, _onto(), gaf, 'mouse', 'fine')
    assert 'Notch signaling pathway (GO:0007219)' not in {s['name'] for s in bundle['sets']}
    assert bundle['empty'] == ['Notch signaling pathway (GO:0007219)']


def test_sibling_overlap_report_names_the_redundant_pair(tmp_path):
    rows = _rows(tmp_path=tmp_path)
    gaf = _gaf()
    gaf['annotations']['Col1a1'].add('GO:0002062')  # collagen fibril organization now ⊂ chondrocyte differentiation
    members = gos.closed_members(_onto(), gaf)
    pairs = bdg.sibling_overlaps(rows, members, threshold=0.3)
    assert [(a, b) for a, b, _ in pairs] == [('GO:0002062', 'GO:0030199')]
    assert bdg.sibling_overlaps(rows, gos.closed_members(_onto(), _gaf()), threshold=0.3) == []


def test_rewrite_sizes_updates_only_the_count_column(tmp_path):
    p = tmp_path / 'dev_goslim.tsv'
    p.write_text(TABLE)
    rows = bdg.read_table(p)
    bdg.rewrite_sizes(p, rows, gos.closed_members(_onto(), _gaf()))
    again = bdg.read_table(p)
    assert [(r['go_id'], r['mouse_genes']) for r in again][:4] == [
        ('GO:0001501', 3), ('GO:0002062', 2), ('GO:0030199', 1), ('GO:0001649', 1)]
    assert [r['cross'] for r in again] == [r['cross'] for r in rows]
    assert p.read_text().splitlines()[0] == 'level\tparent\tgo_id\tname\tmouse_genes\tcross'


# --- the committed table, offline ---------------------------------------------

def test_the_shipped_table_is_well_formed():
    rows = bdg.read_table(bdg.TABLE_PATH)
    coarse = [r for r in rows if r['level'] == 'coarse']
    fine = [r for r in rows if r['level'] == 'fine']
    assert len(coarse) >= 30 and len(fine) >= 150
    coarse_ids = {r['go_id'] for r in coarse}
    assert all(r['parent'] in coarse_ids for r in fine)
    for level in (coarse, fine):
        ids = [r['go_id'] for r in level]
        assert len(ids) == len(set(ids))
    assert all(r['go_id'].startswith(('GO:', 'union:')) for r in rows)
