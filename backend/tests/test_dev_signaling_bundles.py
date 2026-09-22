"""The developmental-signalling pathway bundles shipped with xcell.

Two files under ``xcell/data/gene_sets/`` — one per species — hold the major
developmental pathways as folders (Wnt, BMP, FGF, Hedgehog, Notch, …), each
folder carrying its genes split by role (ligands, receptors, modulators,
effectors, inhibitors, targets) in a fixed order. The heatmap groups rows by
those sets, so a set that is empty, a role out of order, or a symbol Ensembl
does not know would show up as a silent gap on the plot. These tests pin the
files against the bundled Ensembl symbol tables.
"""
import json
from pathlib import Path

import pytest

from xcell import gene_set_library, gene_symbols

DATA = Path(gene_set_library.LIBRARY_DIR)
ROLES = ['ligands', 'receptors', 'modulators', 'effectors', 'inhibitors', 'targets']
CORE = {'Wnt (canonical)', 'BMP', 'FGF', 'Hedgehog', 'Notch', 'TGFβ/Activin/Nodal', 'Retinoic acid'}


def _bundle(species):
    return json.loads((DATA / f'dev_signaling_{species}.json').read_text())


def _folders(bundle):
    out: dict[str, list[dict]] = {}
    for s in bundle['sets']:
        out.setdefault(s['folder'], []).append(s)
    return out


@pytest.fixture(scope='module')
def symbols():
    return {sp: set(gene_symbols.load_table(sp).values()) for sp in ('mouse', 'human')}


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_the_bundle_is_listed_by_the_library(species):
    ids = {b['id']: b for b in gene_set_library.list_bundles()}
    assert f'dev_signaling_{species}' in ids
    b = ids[f'dev_signaling_{species}']
    assert species in b['name'].lower() and b['count'] >= 100


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_every_symbol_is_known_to_ensembl(species, symbols):
    unknown = sorted({
        f"{s['folder']}/{s['name']}: {g}"
        for s in _bundle(species)['sets'] for g in s['genes'] if g not in symbols[species]
    })
    assert unknown == []


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_the_core_pathways_are_present_with_the_roles_in_order(species):
    folders = _folders(_bundle(species))
    assert CORE <= set(folders)
    for folder, sets in folders.items():
        names = [s['name'] for s in sets]
        assert len(set(names)) == len(names), folder
        assert all(n in ROLES for n in names), (folder, names)
        assert names == [r for r in ROLES if r in names], (folder, names)
        assert len(names) >= 3, folder


@pytest.mark.parametrize('species', ['mouse', 'human'])
def test_no_set_repeats_a_gene_or_is_empty(species):
    for s in _bundle(species)['sets']:
        assert s['genes'], (s['folder'], s['name'])
        assert len(set(s['genes'])) == len(s['genes']), (s['folder'], s['name'])


def test_both_species_have_the_same_pathways_and_roles():
    m, h = _folders(_bundle('mouse')), _folders(_bundle('human'))
    assert set(m) == set(h)
    for folder in m:
        assert [s['name'] for s in m[folder]] == [s['name'] for s in h[folder]], folder
        for ms, hs in zip(m[folder], h[folder]):
            assert len(ms['genes']) == len(hs['genes']), (folder, ms['name'])


def test_the_named_examples_are_where_a_biologist_expects_them():
    wnt = {s['name']: s['genes'] for s in _folders(_bundle('mouse'))['Wnt (canonical)']}
    assert {'Ctnnb1', 'Tcf7l2', 'Lef1'} <= set(wnt['effectors'])
    assert {'Fzd1', 'Fzd7', 'Lrp6'} <= set(wnt['receptors'])
    assert {'Wnt3a', 'Wnt1'} <= set(wnt['ligands'])
    assert {'Sfrp2', 'Cer1', 'Dkk1'} <= set(wnt['inhibitors'])
    assert {'Axin2', 'Lef1'} <= set(wnt['targets'])
    fgf_h = {s['name']: s['genes'] for s in _folders(_bundle('human'))['FGF']}
    assert 'FGF19' in fgf_h['ligands'] and 'FGF15' not in fgf_h['ligands']
