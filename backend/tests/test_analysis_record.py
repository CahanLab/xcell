"""The Analysis Record data model.

The record is what makes a GUI session reproducible, so the contracts that
matter are: nothing is silently dropped, the round-trip through .uns is
lossless, and nothing that reaches the API can be unserializable.
"""
import json
import math

import pytest

from xcell.analysis_record import (
    SELECTION_CAP,
    AnalysisRecord,
    Figure,
    Step,
    sanitize_for_json,
)


def _record():
    return AnalysisRecord(source={'path': '/data/cortex.h5ad', 'n_cells': 100, 'n_genes': 20})


# --- steps ---------------------------------------------------------------

def test_added_steps_are_indexed_in_order():
    r = _record()
    r.add_step('normalize_total', {'target_sum': 1e4}, {'status': 'completed'})
    r.add_step('log1p', {}, {'status': 'completed'})
    assert [s.index for s in r.steps] == [0, 1]
    assert [s.action for s in r.steps] == ['normalize_total', 'log1p']


def test_every_step_carries_a_timestamp():
    r = _record()
    step = r.add_step('log1p', {}, {})
    assert step.timestamp  # ISO-8601 string, set by the recorder not the caller


def test_step_note_can_be_set_and_cleared():
    r = _record()
    r.add_step('leiden', {'resolution': 1.0}, {'n_clusters': 12})
    r.set_note(0, 'resolution chosen from a silhouette sweep')
    assert r.steps[0].note == 'resolution chosen from a silhouette sweep'
    r.set_note(0, '')
    assert r.steps[0].note is None


def test_note_on_a_missing_step_is_an_error():
    r = _record()
    with pytest.raises(IndexError):
        r.set_note(3, 'nope')


# --- the report marker ---------------------------------------------------

def test_report_covers_every_step_by_default():
    r = _record()
    r.add_step('a', {}, {})
    r.add_step('b', {}, {})
    assert [s.action for s in r.steps_for_report()] == ['a', 'b']


def test_mark_start_trims_the_report_to_what_follows():
    """The user's "start recording here" — earlier exploration stays logged."""
    r = _record()
    r.add_step('a', {}, {})
    r.mark_start()
    r.add_step('b', {}, {})
    assert [s.action for s in r.steps_for_report()] == ['b']
    assert len(r.steps) == 2  # nothing was discarded


def test_clearing_keeps_the_load_step():
    """Without a source step the notebook has nothing to open."""
    r = _record()
    r.add_step('load_dataset', {'path': '/data/cortex.h5ad'}, {})
    r.add_step('log1p', {}, {})
    r.clear()
    assert [s.action for s in r.steps] == ['load_dataset']
    assert r.report_start == 0


# --- selections ----------------------------------------------------------

def test_a_subset_step_records_how_many_cells_it_ran_on():
    r = _record()
    r.add_step('leiden', {'resolution': 1.0}, {}, selection=[0, 1, 2, 3], n_total=100)
    step = r.steps[0]
    assert step.n_active == 4
    assert step.n_total == 100
    assert step.selection == [0, 1, 2, 3]


def test_a_whole_dataset_step_has_no_selection():
    r = _record()
    r.add_step('leiden', {'resolution': 1.0}, {}, selection=None, n_total=100)
    assert r.steps[0].n_active is None
    assert r.steps[0].selection is None


def test_huge_selections_keep_the_count_but_drop_the_indices():
    """Carrying a million indices per step would bloat .uns and the API."""
    r = _record()
    big = list(range(SELECTION_CAP + 1))
    r.add_step('leiden', {}, {}, selection=big, n_total=SELECTION_CAP + 10)
    step = r.steps[0]
    assert step.n_active == SELECTION_CAP + 1
    assert step.selection is None  # dropped, but the count survives


# --- figures -------------------------------------------------------------

def test_a_figure_attaches_to_the_most_recent_step():
    r = _record()
    r.add_step('umap', {}, {})
    fig = r.add_figure('iVBORw0KGgo=', caption='UMAP coloured by leiden')
    assert fig.step_index == 0
    assert r.steps[0].figure_ids == [fig.id]
    assert r.figures[fig.id].caption == 'UMAP coloured by leiden'


def test_a_figure_can_target_an_earlier_step():
    r = _record()
    r.add_step('pca', {}, {})
    r.add_step('umap', {}, {})
    fig = r.add_figure('iVBORw0KGgo=', step_index=0)
    assert r.steps[0].figure_ids == [fig.id]
    assert r.steps[1].figure_ids == []


def test_a_figure_captured_before_any_step_stands_alone():
    r = _record()
    fig = r.add_figure('iVBORw0KGgo=')
    assert fig.step_index is None
    assert r.standalone_figures() == [fig]


def test_removing_a_figure_detaches_it_from_its_step():
    r = _record()
    r.add_step('umap', {}, {})
    fig = r.add_figure('iVBORw0KGgo=')
    r.remove_figure(fig.id)
    assert fig.id not in r.figures
    assert r.steps[0].figure_ids == []


def test_figure_ids_are_unique():
    r = _record()
    ids = {r.add_figure('iVBORw0KGgo=').id for _ in range(50)}
    assert len(ids) == 50


# --- serialization -------------------------------------------------------

def test_round_trip_preserves_everything():
    r = _record()
    r.title = 'Cortex atlas'
    r.abstract = 'Clustering of 100 cells.'
    r.add_step('normalize_total', {'target_sum': 1e4}, {'status': 'completed'})
    r.set_note(0, 'library-size normalization')
    r.mark_start()
    r.add_step('leiden', {'resolution': 1.0}, {'n_clusters': 12}, selection=[1, 2], n_total=100)
    r.add_figure('iVBORw0KGgo=', caption='UMAP')

    back = AnalysisRecord.from_dict(r.to_dict())

    assert back.title == r.title
    assert back.abstract == r.abstract
    assert back.source == r.source
    assert back.report_start == r.report_start
    assert [s.action for s in back.steps] == ['normalize_total', 'leiden']
    assert back.steps[0].note == 'library-size normalization'
    assert back.steps[1].selection == [1, 2]
    assert back.steps[1].n_active == 2
    assert list(back.figures.values())[0].caption == 'UMAP'
    assert back.steps[1].figure_ids == r.steps[1].figure_ids


def test_the_serialized_form_is_json():
    """It is stored as a JSON string in .uns, so it has to survive dumps()."""
    r = _record()
    r.add_step('leiden', {'resolution': 1.0}, {'n_clusters': 12})
    text = json.dumps(r.to_dict())
    assert AnalysisRecord.from_dict(json.loads(text)).steps[0].action == 'leiden'


def test_from_dict_tolerates_an_older_record():
    """A record read from someone else's h5ad may predate any given field."""
    back = AnalysisRecord.from_dict({'steps': [{'action': 'log1p'}]})
    assert back.steps[0].action == 'log1p'
    assert back.steps[0].index == 0
    assert back.steps[0].params == {}
    assert back.title == ''


def test_from_dict_rejects_a_non_record():
    with pytest.raises(ValueError):
        AnalysisRecord.from_dict(['not', 'a', 'record'])


# --- JSON safety ---------------------------------------------------------

def test_nan_and_inf_in_a_result_become_none():
    """scipy stats reach the record through marker_genes and diffexp."""
    r = _record()
    r.add_step('marker_genes', {}, {'pval': float('nan'), 'score': math.inf})
    assert r.steps[0].result['pval'] is None
    assert r.steps[0].result['score'] is None


def test_sanitize_reaches_into_nested_structures():
    dirty = {'genes': [{'name': 'A', 'lfc': float('inf')}], 'meta': {'p': float('nan')}}
    clean = sanitize_for_json(dirty)
    assert clean['genes'][0]['lfc'] is None
    assert clean['meta']['p'] is None
    assert clean['genes'][0]['name'] == 'A'


def test_sanitize_leaves_ordinary_numbers_alone():
    assert sanitize_for_json({'a': 1, 'b': 2.5, 'c': 'x', 'd': None, 'e': True}) == {
        'a': 1, 'b': 2.5, 'c': 'x', 'd': None, 'e': True,
    }


def test_json_dump_is_strict_clean():
    """allow_nan=False is what an HTTP client effectively enforces."""
    r = _record()
    r.add_step('diffexp', {}, {'scores': [float('nan'), 1.0]})
    json.dumps(r.to_dict(), allow_nan=False)


def test_a_step_result_that_is_not_a_dict_is_kept_as_is():
    """_log_action is called with None for results in a few places."""
    r = _record()
    r.add_step('delete_pca_subset', {'obsm_key': 'X_pca_a'}, None)
    assert r.steps[0].result == {}


# --- Step / Figure basics -------------------------------------------------

def test_step_and_figure_survive_their_own_round_trip():
    s = Step(index=2, action='pca', params={'n_comps': 50}, result={}, timestamp='t')
    assert Step.from_dict(s.to_dict()) == s
    f = Figure(id='f1', png_b64='x', caption='c', step_index=2, timestamp='t')
    assert Figure.from_dict(f.to_dict()) == f
