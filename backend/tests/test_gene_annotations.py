"""Per-gene annotations from MyGene.info: parsing, batching, cache, orthologs, routes.

Offline: the two network functions are monkeypatched with captured response
shapes (a mouse gene with a RefSeq summary, one without, a duplicate hit,
a notfound, and the /gene lookup used to fill summaries from orthologs).
"""
import time

import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell import gene_annotations as ga
from xcell.adaptor import DataAdaptor


COL1A1 = {"query": "Col1a1", "_id": "12842", "symbol": "Col1a1", "taxid": 10090, "entrezgene": 12842,
          "name": "collagen, type I, alpha 1", "type_of_gene": "protein-coding", "MGI": "MGI:88467",
          "uniprot": {"Swiss-Prot": "P11087"}, "alias": ["Col1a-1", "Mov13"],
          "ensembl": [{"gene": "ENSNVIG00000007125"}, {"gene": "ENSMUSG00000001506"}],
          "summary": "This gene encodes the alpha-1 subunit of type I collagen.",
          "go": {"BP": [{"evidence": "IEA", "id": "GO:0001501", "term": "skeletal system development"},
                        {"evidence": "IMP", "id": "GO:0001501", "term": "skeletal system development"},
                        {"evidence": "IDA", "id": "GO:0030199", "term": "collagen fibril organization"}],
                 "CC": {"evidence": "HDA", "id": "GO:0005576", "term": "extracellular region"},
                 "MF": [{"evidence": "IEA", "id": "GO:0002020", "term": "protease binding"}]},
          "interpro": [{"desc": "Fibrillar collagen, C-terminal", "id": "IPR000885", "short_desc": "Fib_collagen_C"}],
          "pathway": {"kegg": [{"id": "mmu04510", "name": "Focal adhesion"}], "reactome": {"id": "R-MMU-1474244", "name": "Extracellular matrix organization"}},
          "homologene": {"genes": [[9606, 1277], [10090, 12842], [10116, 29393]], "id": 73874}}
SOX9 = {"query": "Sox9", "_id": "20682", "symbol": "Sox9", "taxid": 10090, "entrezgene": 20682,
        "name": "SRY (sex determining region Y)-box 9", "type_of_gene": "protein-coding",
        "go": {"BP": [{"evidence": "IBA", "id": "GO:0000122", "term": "negative regulation of transcription"}]},
        "interpro": {"desc": "High mobility group box domain", "id": "IPR009071", "short_desc": "HMG_box_dom"},
        "homologene": {"genes": [[9606, 6662], [10090, 20682]], "id": 294}}
GM_A = {"query": "Gm1992", "_id": "ENSMUSG00000089699", "symbol": "Gm1992", "name": "predicted gene 1992", "ensembl": {"gene": "ENSMUSG00000089699"}}
GM_B = {"query": "Gm1992", "_id": "100038975", "symbol": "Gm1992", "taxid": 10090, "entrezgene": 100038975, "name": "predicted gene 1992", "type_of_gene": "protein-coding", "MGI": "MGI:3780162"}
NOTFOUND = {"query": "Fakegene9", "notfound": True}
HUMAN_GENES = [{"query": "1277", "_id": "1277", "symbol": "COL1A1", "taxid": 9606, "name": "collagen type I alpha 1 chain", "summary": "Human summary for COL1A1."},
               {"query": "6662", "_id": "6662", "symbol": "SOX9", "taxid": 9606, "name": "SRY-box transcription factor 9", "summary": "Human summary for SOX9."}]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    ga.set_cache_path(tmp_path / "ann.sqlite")
    calls = {"query": [], "gene": []}

    def fake_query(payload):
        calls["query"].append(payload)
        wanted = payload["q"].split(",")
        table = {"Col1a1": [COL1A1], "Sox9": [SOX9], "Gm1992": [GM_A, GM_B], "Fakegene9": [NOTFOUND]}
        out = []
        for q in wanted:
            out.extend(table.get(q, [{"query": q, "notfound": True}]))
        return out

    def fake_gene(payload):
        calls["gene"].append(payload)
        ids = payload["ids"].split(",")
        return [h for h in HUMAN_GENES if h["query"] in ids]

    monkeypatch.setattr(ga, "post_query", fake_query)
    monkeypatch.setattr(ga, "post_gene", fake_gene)
    yield calls
    ga.set_cache_path(None)


def test_parse_hit_compacts_go_interpro_pathways_and_links():
    rec = ga.parse_hit(COL1A1, "mouse")
    assert rec["symbol"] == "Col1a1" and rec["entrez"] == 12842 and rec["ensembl"] == "ENSMUSG00000001506"
    assert rec["mgi"] == "MGI:88467" and rec["uniprot"] == "P11087" and rec["aliases"] == ["Col1a-1", "Mov13"]
    assert rec["summary"].startswith("This gene encodes") and rec["summary_source"] == "refseq"
    # duplicate GO ids collapse, keeping the strongest evidence
    bp = rec["go"]["BP"]
    assert [t["id"] for t in bp] == ["GO:0001501", "GO:0030199"]
    assert bp[0]["evidence"] == "IMP"
    assert rec["go"]["CC"] == [{"id": "GO:0005576", "term": "extracellular region", "evidence": "HDA"}]
    assert rec["interpro"] == [{"id": "IPR000885", "name": "Fibrillar collagen, C-terminal", "short": "Fib_collagen_C"}]
    assert rec["pathways"]["reactome"] == [{"id": "R-MMU-1474244", "name": "Extracellular matrix organization"}]
    assert rec["pathways"]["kegg"][0]["name"] == "Focal adhesion"
    assert rec["homologene"] == {"9606": 1277, "10090": 12842, "10116": 29393}
    assert rec["links"]["ncbi"].endswith("12842") and "MGI:88467" in rec["links"]["mgi"] and rec["links"]["uniprot"].endswith("P11087")
    assert rec["notfound"] is False


def test_pick_best_hit_prefers_entrez_record_of_the_right_species():
    best = ga.pick_hits([GM_A, GM_B, NOTFOUND], "mouse")
    assert best["Gm1992"]["entrezgene"] == 100038975
    assert best["Fakegene9"].get("notfound") is True


def test_get_annotations_fills_summary_from_human_ortholog_and_caches(_isolated):
    out = ga.get_annotations(["Col1a1", "Sox9", "Fakegene9"], "mouse")
    assert set(out) == {"Col1a1", "Sox9", "Fakegene9"}
    assert out["Col1a1"]["summary_source"] == "refseq"
    assert out["Sox9"]["summary"] == "Human summary for SOX9." and out["Sox9"]["summary_source"] == "ortholog:SOX9"
    assert out["Sox9"]["orthologs"]["human"] == {"symbol": "SOX9", "entrez": 6662}
    assert out["Col1a1"]["orthologs"]["human"]["symbol"] == "COL1A1"
    assert out["Fakegene9"]["notfound"] is True
    assert len(_isolated["query"]) == 1 and len(_isolated["gene"]) == 1
    # second call is served from the cache
    again = ga.get_annotations(["Col1a1", "Sox9", "Fakegene9"], "mouse")
    assert again["Sox9"]["summary"] == "Human summary for SOX9." and len(_isolated["query"]) == 1
    assert ga.stats("mouse")["n_cached"] == 3
    # refresh re-queries
    ga.get_annotations(["Col1a1"], "mouse", refresh=True)
    assert len(_isolated["query"]) == 2


def test_batches_and_scopes(monkeypatch, _isolated):
    genes = [f"G{i}" for i in range(1500)] + ["ENSMUSG00000001506"]
    ga.get_annotations(genes, "mouse")
    payloads = _isolated["query"]
    scopes = [p["scopes"] for p in payloads]
    assert scopes.count("ensembl.gene") == 1 and scopes.count("symbol,alias") == 2
    assert all(int(p["species"]) == 10090 for p in payloads)
    assert max(len(p["q"].split(",")) for p in payloads) <= 1000


def test_species_validation():
    with pytest.raises(ValueError, match="species"):
        ga.get_annotations(["A"], "zebrafish")


def _adaptor():
    rng = np.random.default_rng(0)
    genes = ["Col1a1", "Sox9", "Gm1992", "Fakegene9"]
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, size=(5, len(genes))).astype(np.float32)))
    ad.var_names = genes
    return DataAdaptor("x.h5ad", adata=ad)


def test_adaptor_guesses_species_and_prefetches(_isolated):
    a = _adaptor()
    out = a.gene_annotations(["Col1a1", "nope"])
    assert out["species"] == "mouse" and out["annotations"]["Col1a1"]["symbol"] == "Col1a1"
    assert out["annotations"]["nope"]["notfound"] is True
    compute_fn, apply_fn = a.prepare_gene_annotation_prefetch()
    progress = []
    res = apply_fn(compute_fn(lambda f, m=None: progress.append(f)))
    assert res["n_genes"] == 4 and res["n_cached"] >= 4 and res["species"] == "mouse"
    assert progress and progress[-1] == 1.0
    st = a.gene_annotation_status()
    assert st["n_genes"] == 4 and st["n_cached"] >= 4


def _poll(client, task_id, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/tasks/{task_id}").json()
        if st["status"] in ("completed", "error", "cancelled"):
            return st
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_routes(monkeypatch, _isolated):
    from xcell.main import app
    from xcell.api import routes
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    r = client.get("/api/gene_annotations/Sox9")
    assert r.status_code == 200, r.text
    assert r.json()["annotation"]["summary_source"] == "ortholog:SOX9" and r.json()["species"] == "mouse"
    r = client.post("/api/gene_annotations", json={"genes": ["Col1a1", "Gm1992"]})
    assert r.status_code == 200 and set(r.json()["annotations"]) == {"Col1a1", "Gm1992"}
    r = client.post("/api/gene_annotations", json={"genes": [f"G{i}" for i in range(300)]})
    assert r.status_code == 400
    r = client.get("/api/gene_annotations/status")
    assert r.status_code == 200 and r.json()["n_genes"] == 4
    r = client.post("/api/gene_annotations/prefetch", json={})
    assert r.status_code == 202, r.text
    st = _poll(client, r.json()["task_id"])
    assert st["status"] == "completed" and st["result"]["n_cached"] >= 4
    r = client.post("/api/gene_annotations", json={"genes": ["Col1a1"], "species": "cat"})
    assert r.status_code == 400
