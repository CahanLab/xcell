"""GO semantic similarity: ontology parsing, annotation closure, SimGIC.

A five-term toy ontology with a known DAG gives exact answers: two genes on
sibling leaves share only the ancestors, two on the same leaf share the leaf
too, and an unannotated gene is similar to nothing but itself.
"""
import gzip
import math

import numpy as np
import pytest

from xcell import gene_set_sources as gss
from xcell import go_semantic as gos

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

[Term]
id: GO:0006096
name: glycolytic process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Typedef]
id: part_of
name: part of
"""

GAF_ROWS = [
    ('Sox9', 'GO:0002062', 'P', 'involved_in'),
    ('Col2a1', 'GO:9999999', 'P', 'involved_in'),   # alt_id of chondrocyte differentiation
    ('Col1a1', 'GO:0030199', 'P', 'involved_in'),
    ('Runx2', 'GO:0001501', 'P', 'involved_in'),
    ('Actb', 'GO:0002062', 'P', 'NOT|involved_in'),
    ('Actb', 'GO:0005576', 'C', 'located_in'),
    ('Gapdh', 'GO:0005576', 'C', 'located_in'),
    ('Gapdh', 'GO:0006096', 'P', 'involved_in'),
]


def _gaf_text():
    lines = ['!gaf-version: 2.2', '!date-generated: 2026-05-21']
    for i, (sym, term, aspect, qual) in enumerate(GAF_ROWS):
        lines.append('\t'.join(['MGI', f'MGI:{i}', sym, qual, term, 'PMID:1', 'IDA', '', aspect, 'name', '', 'protein',
                                'taxon:10090', '20130101', 'MGI', '', f'MGI:MGI:{i}']))
    lines.append('\t'.join(['ComplexPortal', 'CPX-1', 'complex', 'part_of', 'GO:0005576', 'PMID:6', 'IDA', '', 'C',
                            'x', '', 'protein_complex', 'taxon:10090', '20130101', 'ComplexPortal', '', '']))
    return '\n'.join(lines) + '\n'


@pytest.fixture
def files(tmp_path):
    gss.set_cache_dir(tmp_path / 'cache')
    gos.go_dir().mkdir(parents=True)
    gos.obo_path().write_text(OBO)
    with gzip.open(gos.gaf_path('mouse'), 'wt') as fh:
        fh.write(_gaf_text())
    gos._ontology_cache.clear(); gos._gaf_cache.clear(); gos._space_cache.clear()
    yield
    gss.set_cache_dir(None)


def test_parse_obo_reads_parents_along_is_a_and_part_of_only_and_drops_obsolete():
    onto = gos.parse_obo(OBO)
    assert onto['version'] == 'releases/2026-07-26'
    assert onto['parents']['GO:0002062'] == {'GO:0001501'}
    assert onto['parents']['GO:0030199'] == {'GO:0001501'}   # regulates is not a parent
    assert 'GO:0000001' not in onto['parents']
    assert onto['alt']['GO:9999999'] == 'GO:0002062'
    assert onto['namespace']['GO:0005576'] == 'cellular_component'
    assert onto['name']['GO:0001501'] == 'skeletal system development'


def test_parse_gaf_skips_not_and_foreign_databases():
    gaf = gos.parse_gaf(_gaf_text().splitlines(), 'mouse')
    assert gaf['date'] == '2026-05-21'
    assert gaf['annotations']['Actb'] == {'GO:0005576'}          # the NOT row is gone
    assert 'complex' not in gaf['annotations']                    # ComplexPortal row skipped
    assert gaf['aspects']['GO:0002062'] == 'bp'


def test_ancestors_follow_the_dag_with_memoisation():
    onto = gos.parse_obo(OBO)
    cache = {}
    assert gos.ancestors(onto['parents'], 'GO:0002062', cache) == {'GO:0002062', 'GO:0001501', 'GO:0008150'}
    assert cache['GO:0002062'] is gos.ancestors(onto['parents'], 'GO:0002062', cache)


def test_simgic_ranks_shared_specific_terms_above_shared_ancestors(files):
    out = gos.similarity(['Sox9', 'Col2a1', 'Col1a1', 'Runx2', 'Gapdh', 'Nope'], 'mouse', 'bp')
    S = out['similarity']
    assert S.shape == (6, 6) and np.allclose(np.diag(S), 1.0)
    # Sox9 and Col2a1 share the leaf (via Col2a1's alt_id): identical closed sets.
    assert S[0, 1] == pytest.approx(1.0)
    # Sox9 and Col1a1 share skeletal system development + root, not the leaves.
    assert 0.0 < S[0, 2] < 1.0
    assert S[0, 2] < S[0, 1]
    # Runx2 sits on the shared ancestor: its closure is a subset of Sox9's, so
    # the shared IC is all of Runx2's, over Sox9's — between 0 and 1, above the sibling.
    assert S[0, 2] < S[0, 3] < 1.0
    # Gapdh shares only the root, whose IC is zero; an unknown gene has nothing at all.
    assert S[4, :4].max() == pytest.approx(0.0) and S[5].sum() == pytest.approx(1.0)
    assert out['terms_per_gene'] == [3, 3, 3, 2, 2, 0]
    assert out['n_genes_annotated'] == 5 and out['n_genes_in_space'] == 5


def test_ic_is_minus_log_frequency_over_the_whole_space(files):
    sp = gos.space('mouse', 'bp')
    # 5 BP-annotated genes; the root covers all 5, skeletal 4, the leaf 2, collagen fibril 1.
    assert sp.ic['GO:0008150'] == pytest.approx(0.0)
    assert sp.ic['GO:0001501'] == pytest.approx(-math.log(4 / 5))
    assert sp.ic['GO:0002062'] == pytest.approx(-math.log(2 / 5))
    assert sp.ic['GO:0030199'] == pytest.approx(-math.log(1 / 5))


def test_lookup_is_case_insensitive_and_aspect_is_validated(files):
    sp = gos.space('mouse', 'bp')
    assert sp.terms_for('SOX9') == sp.terms_for('Sox9')
    with pytest.raises(ValueError):
        gos.space('mouse', 'xx')


def test_missing_files_are_a_value_error_naming_the_library(tmp_path):
    gss.set_cache_dir(tmp_path / 'empty')
    try:
        with pytest.raises(ValueError, match='Gene set library'):
            gos.similarity(['Sox9'], 'mouse')
        assert gos.availability('mouse') == {'obo': False, 'obo_path': str(gos.obo_path()), 'gaf': {'mouse': False}}
    finally:
        gss.set_cache_dir(None)
