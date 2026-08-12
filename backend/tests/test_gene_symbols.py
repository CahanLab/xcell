import pytest

from xcell.gene_symbols import (
    detect_species, load_table, map_ids, strip_version,
)

TABLE = {'ENSMUSG00000002603': 'Tgfb1', 'ENSMUSG00000051951': 'Xkr4',
         'ENSMUSG00000000001': 'Gnai3', 'ENSMUSG00000000002': 'Gnai3'}


def test_strip_version_removes_only_the_suffix():
    assert strip_version('ENSG00000141510.15') == 'ENSG00000141510'
    assert strip_version('ENSG00000141510') == 'ENSG00000141510'
    # A symbol containing a dot is not an Ensembl id and must survive intact.
    assert strip_version('Tgfb1') == 'Tgfb1'
    assert strip_version('HLA-DRB1.2') == 'HLA-DRB1.2'


def test_detect_species_reads_the_prefix():
    assert detect_species(['ENSG00000141510', 'ENSG00000012048'])['species'] == 'human'
    assert detect_species(['ENSMUSG00000002603'])['species'] == 'mouse'


def test_detect_species_takes_the_majority_and_reports_the_split():
    out = detect_species(['ENSMUSG1', 'ENSMUSG2', 'ENSMUSG3', 'ENSG1'])
    assert out['species'] == 'mouse'
    assert out['counts']['mouse'] == 3 and out['counts']['human'] == 1


def test_detect_species_names_an_unsupported_species():
    # Naming the animal is the difference between a message a user can act on
    # and "unsupported".
    out = detect_species(['ENSDARG00000000001', 'ENSDARG00000000002'])
    assert out['species'] is None
    assert out['unsupported'] == 'zebrafish'


def test_detect_species_reports_symbols_as_not_ensembl():
    out = detect_species(['Tgfb1', 'Xkr4', 'Shh'])
    assert out['species'] is None
    assert out['n_recognized'] == 0
    assert out['unsupported'] is None


def test_map_ids_maps_and_keeps_unmapped_ids_as_themselves():
    out = map_ids(['ENSMUSG00000002603', 'ENSMUSG99999999999'], TABLE)
    assert out['symbols'] == ['Tgfb1', 'ENSMUSG99999999999']
    assert out['n_mapped'] == 1 and out['n_unmapped'] == 1


def test_map_ids_ignores_the_version_suffix():
    out = map_ids(['ENSMUSG00000002603.7'], TABLE)
    assert out['symbols'] == ['Tgfb1']


def test_map_ids_counts_duplicate_symbols_without_raising():
    # Several Ensembl ids legitimately share a symbol. That is a fact about the
    # annotation, not a mistake the user can fix, so it is counted not refused.
    out = map_ids(['ENSMUSG00000000001', 'ENSMUSG00000000002'], TABLE)
    assert out['symbols'] == ['Gnai3', 'Gnai3']
    assert out['n_duplicate_symbols'] == 1


def test_map_ids_never_produces_a_blank_symbol():
    out = map_ids(['ENSMUSG00000002603', 'weird', ''], TABLE)
    assert all(s is not None for s in out['symbols'])
    assert out['symbols'][1] == 'weird'


def test_load_table_refuses_a_species_it_does_not_ship():
    with pytest.raises(ValueError, match='human and mouse'):
        load_table('zebrafish')


def test_the_shipped_tables_load_and_carry_known_genes():
    for species, probe, expect in [('human', 'ENSG00000141510', 'TP53'),
                                   ('mouse', 'ENSMUSG00000002603', 'Tgfb1')]:
        table = load_table(species)
        assert len(table) > 40_000, f'{species} table looks truncated'
        assert table[probe] == expect
