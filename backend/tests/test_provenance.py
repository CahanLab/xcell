"""Which xcell produced a result.

0.1.0 covered eight months of main, so a version alone says little between
releases: a checkout adds its commit and whether it had uncommitted changes.
Each recorded step keeps the build that ran it (a record outlives the version
that started it), an exported h5ad says what wrote it, and a rendered notebook
says which builds recorded its steps.
"""
import subprocess
from pathlib import Path

import anndata
import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import xcell
from xcell import provenance
from xcell.adaptor import DataAdaptor
from xcell.analysis_record import AnalysisRecord, Step
from xcell.notebook_export import to_markdown, to_notebook

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_build_info():
    provenance.build_info.cache_clear()
    yield
    provenance.build_info.cache_clear()


def _info(**kw):
    return {'version': '0.2.0', 'commit': None, 'dirty': None, 'tag': None, **kw}


# --- the label --------------------------------------------------------------

def test_the_label_is_the_version_alone_at_a_clean_release_tag():
    assert provenance.build_label(_info(commit='abc1234', tag='v0.2.0', dirty=False)) == '0.2.0'


def test_the_label_adds_the_commit_between_releases():
    assert provenance.build_label(_info(commit='abc1234', dirty=False)) == '0.2.0+gabc1234'
    # A tag that is not this version's release says nothing about the code.
    assert provenance.build_label(_info(commit='abc1234', tag='v0.1.0', dirty=False)) == '0.2.0+gabc1234'


def test_the_label_says_when_the_code_had_uncommitted_changes():
    assert provenance.build_label(_info(commit='abc1234', tag='v0.2.0', dirty=True)) == '0.2.0+gabc1234.dirty'


def test_without_a_checkout_the_label_is_the_version():
    assert provenance.build_label(_info()) == '0.2.0'


# --- reading git ------------------------------------------------------------

@pytest.mark.skipif(not (REPO / '.git').exists(), reason='not a git checkout')
def test_a_checkout_reports_its_commit():
    info = provenance.build_info()
    head = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    assert info['version'] == xcell.__version__
    assert info['commit'] == head
    assert isinstance(info['dirty'], bool)
    assert provenance.build_label().startswith(xcell.__version__)


def test_no_git_means_the_version_alone(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(provenance.subprocess, 'run', missing)
    assert provenance.build_info() == {'version': xcell.__version__, 'commit': None,
                                       'dirty': None, 'tag': None}
    assert provenance.build_label() == xcell.__version__


def test_an_installed_copy_does_not_report_some_other_repository(monkeypatch, tmp_path):
    # site-packages can sit inside an unrelated repository; its commit would lie.
    monkeypatch.setattr(provenance, '_ROOT', tmp_path)
    assert provenance.build_info()['commit'] is None


# --- the record -------------------------------------------------------------

def test_each_step_records_the_build_that_ran_it(monkeypatch):
    monkeypatch.setattr(provenance, 'build_label', lambda info=None: '0.2.0+gabc1234')
    r = AnalysisRecord()
    step = r.add_step('log1p', {}, {'status': 'completed'})
    assert step.xcell == '0.2.0+gabc1234'
    again = AnalysisRecord.from_dict(r.to_dict())
    assert again.steps[0].xcell == '0.2.0+gabc1234'


def test_a_step_from_before_the_stamp_reads_as_unknown():
    old = {'index': 0, 'action': 'log1p', 'params': {}, 'result': {}, 'timestamp': ''}
    assert Step.from_dict(old).xcell is None


# --- exports ----------------------------------------------------------------

def _adaptor():
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((20, 10)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(10)]
    ad.obs_names = [f'c{i}' for i in range(20)]
    return DataAdaptor('x.h5ad', adata=ad)


def test_an_exported_h5ad_says_which_build_wrote_it(tmp_path):
    a = _adaptor()
    out = a.prepare_export_with_lines()
    stamp = out.uns['xcell_provenance']
    assert stamp['version'] == xcell.__version__
    assert stamp['build'] == provenance.build_label()
    assert stamp['exported_at']
    assert 'xcell_provenance' not in a.adata.uns  # the live data is untouched
    # h5ad cannot store None, so a missing commit must be left out, not written.
    path = tmp_path / 'out.h5ad'
    out.write_h5ad(path)
    with h5py.File(path) as f:
        assert f['uns']['xcell_provenance']['version'][()].decode() == xcell.__version__


def _record(labels):
    r = AnalysisRecord(source={'path': '/data/x.h5ad', 'n_cells': 20, 'n_genes': 10})
    for i, label in enumerate(labels):
        step = r.add_step('leiden', {'resolution': 1.0, 'key_added': f'leiden{i}'}, {'n_clusters': 3})
        step.xcell = label
    return r


def test_the_notebook_names_the_build_that_recorded_its_steps():
    nb = to_notebook(_record(['0.2.0', '0.2.0']))
    header = ''.join(nb['cells'][0]['source'])
    assert 'xcell](https://github.com/CahanLab/xcell) 0.2.0 on' in header
    assert nb['metadata']['xcell']['version'] == xcell.__version__
    assert nb['metadata']['xcell']['build'] == provenance.build_label()


def test_a_record_that_spans_builds_says_which_steps_each_ran():
    md = to_markdown(_record([None, '0.1.0+g9e16fcd', '0.1.0+g9e16fcd', '0.2.0']))
    assert ('an unrecorded version (step 1), 0.1.0+g9e16fcd (steps 2–3) and 0.2.0 (step 4)'
            in md)


def test_the_backend_reports_its_build():
    from fastapi.testclient import TestClient

    from xcell.main import app
    body = TestClient(app).get('/').json()
    assert body['version'] == xcell.__version__
    assert body['build'] == provenance.build_label()
