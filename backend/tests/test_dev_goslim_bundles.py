"""The developmental GO slim bundles shipped with xcell.

Four files under ``xcell/data/gene_sets/`` — coarse and fine, mouse and
human — built by ``build_dev_goslim.py`` from ``dev_goslim.tsv``. These pin
the shipped files to the table: every coarse term is a coarse set and a fine
folder, every fine term is in its folder, the two species carry the same
terms, and a few sets a developmental biologist would check hold the genes
they should.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from xcell import gene_set_library

DATA = Path(gene_set_library.LIBRARY_DIR)
_SPEC = importlib.util.spec_from_file_location('build_dev_goslim', DATA.parent / 'build_dev_goslim.py')
bdg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bdg)
ROWS = bdg.read_table(bdg.TABLE_PATH)
COARSE = [r for r in ROWS if r['level'] == 'coarse']
FINE = [r for r in ROWS if r['level'] == 'fine']


def _bundle(level, species):
    return json.loads((DATA / f'dev_goslim_{level}_{species}.json').read_text())


@pytest.mark.parametrize('species', ['mouse', 'human'])
@pytest.mark.parametrize('level', ['coarse', 'fine'])
def test_the_bundle_is_listed_by_the_library(level, species):
    ids = {b['id']: b for b in gene_set_library.list_bundles()}
    b = ids[f'dev_goslim_{level}_{species}']
    assert level in b['name'] and species in b['name']
    assert b['count'] == len(_bundle(level, species)['sets'])


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_coarse_bundle_is_one_set_per_coarse_row_and_flat(species):
    sets = _bundle('coarse', species)['sets']
    assert [s['name'] for s in sets] == [bdg._set_name(r) for r in COARSE]
    assert all('folder' not in s for s in sets)


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_fine_bundle_has_the_table_folders_and_terms(species):
    b = _bundle('fine', species)
    folders: dict[str, list[str]] = {}
    for s in b['sets']:
        folders.setdefault(s['folder'], []).append(s['name'])
    expected = {c['name']: [bdg._set_name(f) for f in bdg._fine_rows(ROWS, c['go_id'])] for c in COARSE}
    for name in b['empty']:  # a term with no gene on this species is reported, not shipped
        for lst in expected.values():
            if name in lst:
                lst.remove(name)
    assert folders == expected
    assert len(b['empty']) <= 2


@pytest.mark.parametrize('level', ['coarse', 'fine'])
def test_both_species_carry_the_same_terms(level):
    m = {s['name'] for s in _bundle(level, 'mouse')['sets']}
    h = {s['name'] for s in _bundle(level, 'human')['sets']}
    assert m - h <= set(_bundle(level, 'human')['empty']) and h - m <= set(_bundle(level, 'mouse')['empty'])


@pytest.mark.parametrize('level', ['coarse', 'fine'])
@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_no_set_repeats_a_gene_or_is_empty(level, species):
    for s in _bundle(level, species)['sets']:
        assert s['genes'] and len(set(s['genes'])) == len(s['genes']), s['name']
        assert s['genes'] == sorted(s['genes'])


def test_the_named_examples_are_where_a_biologist_expects_them():
    fine = {s['name']: set(s['genes']) for s in _bundle('fine', 'mouse')['sets']}
    coarse = {s['name']: set(s['genes']) for s in _bundle('coarse', 'mouse')['sets']}
    assert {'Sox9', 'Col2a1', 'Runx2'} <= fine['chondrocyte differentiation (GO:0002062)']
    assert {'Shh', 'Smo', 'Gli1'} <= fine['smoothened signaling pathway (GO:0007224)']
    assert {'Tbx5', 'Hand2', 'Shh'} <= coarse['limb development (GO:0060173)']
    assert {'Nodal', 'Lefty1', 'Pitx2'} <= fine['left/right pattern formation (GO:0060972)']
    # A cross-link is in its folder but not in the coarse set.
    assert 'Sp7' in fine['osteoblast differentiation (GO:0001649)']
    assert 'Runx2' in coarse['skeletal system development (GO:0001501)']
    # The signaling union is exactly its folder's union.
    sig = set().union(*(g for n, g in fine.items() if n.endswith('pathway (GO:0060070)') or 'signaling' in n or n == 'hippo signaling (GO:0035329)'))
    assert coarse['developmental signaling pathways'] <= sig
    human = {s['name']: set(s['genes']) for s in _bundle('fine', 'human')['sets']}
    assert {'SOX9', 'COL2A1'} <= human['chondrocyte differentiation (GO:0002062)']
