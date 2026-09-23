"""``top_n`` as an optional cap on line-association results.

It used to be mandatory — every call silently truncated to 50 genes per
direction (or per module), so a run that found 400 associated genes reported
50 and gave no sign of it. These tests pin down the three things that changed:
``None`` means "return everything significant", a cap keeps the *most
significant* genes rather than whichever happened to sort first, and a module's
``n_genes`` counts the genes actually returned.
"""

import anndata
import numpy as np
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


N_CELLS = 80
N_UP = 10
N_DOWN = 6
N_FLAT = 4
LINE = [[-2.0, -2.0], [2.0, 2.0]]


def _adata():
    """Ten rising genes of graded strength, six falling, four noise.

    Amplitude grows with gene index inside each direction, so "keep the top
    two" has an unambiguous right answer that differs from "keep the first
    two".
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N_CELLS)
    coords = np.column_stack([
        t * 4.0 - 2.0 + rng.normal(0, 0.02, N_CELLS),
        t * 4.0 - 2.0 + rng.normal(0, 0.02, N_CELLS),
    ])

    cols, names = [], []
    for i in range(N_UP):
        cols.append(3.0 + (1.0 + 0.5 * i) * t + rng.normal(0, 0.2, N_CELLS))
        names.append(f"up{i:02d}")
    for i in range(N_DOWN):
        cols.append(3.0 + (1.0 + 0.5 * i) * (1.0 - t) + rng.normal(0, 0.2, N_CELLS))
        names.append(f"dn{i:02d}")
    for i in range(N_FLAT):
        cols.append(rng.normal(3.0, 0.2, N_CELLS))
        names.append(f"fl{i:02d}")

    X = np.maximum(np.column_stack(cols), 0.0).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = names
    ad.obsm["X_pca"] = coords
    return ad


def _adaptor():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.set_lines([{"name": "L", "embeddingName": "X_pca", "points": LINE}])
    return a


def _run(a, **kwargs):
    return a.test_line_association("L", **kwargs)


# ---------- no cap ----------

def test_no_cap_returns_every_significant_gene():
    a = _adaptor()
    r = _run(a, top_n=None)

    assert len(r["positive"]) + len(r["negative"]) == r["n_significant"]
    assert len(r["positive"]) == r["n_positive"]
    assert len(r["negative"]) == r["n_negative"]


def test_no_cap_is_the_default():
    """The parameter is an optional filter, so leaving it out filters nothing."""
    a = _adaptor()
    assert _run(a) == _run(a, top_n=None)


def test_no_cap_returns_every_significant_gene_in_modules():
    a = _adaptor()
    r = _run(a, top_n=None, cluster_genes=True)

    returned = sum(len(m["genes"]) for m in r["modules"])
    assert returned == r["n_significant"]


# ---------- a cap keeps the strongest ----------

def test_cap_limits_each_direction():
    a = _adaptor()
    r = _run(a, top_n=2)

    assert len(r["positive"]) == 2
    assert len(r["negative"]) == 2
    # n_positive / n_negative report what was found, not what was returned.
    assert r["n_positive"] > 2


def test_cap_keeps_the_highest_scoring_genes_per_direction():
    a = _adaptor()
    full = _run(a, top_n=None)
    capped = _run(a, top_n=3)

    for side in ("positive", "negative"):
        expected = [g["gene"] for g in full[side][:3]]
        assert [g["gene"] for g in capped[side]] == expected


def test_cap_keeps_the_highest_scoring_genes_per_module():
    """Modules are listed by peak position; truncation must not follow that."""
    a = _adaptor()
    full = {m["module_id"]: m for m in _run(a, top_n=None, cluster_genes=True)["modules"]}
    capped = _run(a, top_n=3, cluster_genes=True)["modules"]

    assert capped, "expected at least one module"
    for mod in capped:
        members = full[mod["module_id"]]["genes"]
        if len(members) <= 3:
            continue
        score = {g["gene"]: -np.log10(g["fdr"] + 1e-300) * g["amplitude"] for g in members}
        best = sorted(score, key=lambda g: -score[g])[:3]
        assert set(g["gene"] for g in mod["genes"]) == set(best)


def test_capped_module_genes_stay_ordered_by_peak_position():
    a = _adaptor()
    for mod in _run(a, top_n=3, cluster_genes=True)["modules"]:
        peaks = [g["peak_position"] for g in mod["genes"]]
        assert peaks == sorted(peaks)


def test_module_n_genes_counts_what_was_returned():
    a = _adaptor()
    for mod in _run(a, top_n=2, cluster_genes=True)["modules"]:
        assert mod["n_genes"] == len(mod["genes"])


# ---------- through the route ----------

def _install(monkeypatch, a):
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    return TestClient(app)


def test_route_accepts_a_null_cap(monkeypatch):
    a = _adaptor()
    client = _install(monkeypatch, a)
    resp = client.post("/api/lines/association",
                       json={"line_name": "L", "top_n": None})

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    for _ in range(200):
        status = client.get(f"/api/tasks/{task_id}").json()
        if status["status"] in ("completed", "error"):
            break
    assert status["status"] == "completed", status
    result = status["result"]
    assert len(result["positive"]) + len(result["negative"]) == result["n_significant"]
