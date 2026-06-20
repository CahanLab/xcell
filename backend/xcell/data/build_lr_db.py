"""Build a compact ligand-receptor database for the spatial LR-score tool.

Downloads the CellPhoneDB (v5, ventolab fork) CSV release and distills it into a
single small table the backend can load without any R/RDS dependency. This is the
Python-friendly stand-in for CytoSignal's bundled ``db.diff`` / ``db.cont`` RDA
objects (CytoSignal itself is built on CellPhoneDB).

Output: ``lr_pairs.csv`` next to this script, with columns:

    interaction   unique id, "<ligand>->-<receptor>"
    ligand        '_'-joined subunit gene symbols (e.g. "TGFB1")
    receptor      '_'-joined subunit gene symbols (e.g. "TGFBR1_TGFBR2")
    type          "diffusion" (secreted ligand) or "contact" (membrane ligand)
    classification CellPhoneDB pathway/classification string
    source        provenance note

Diffusion-vs-contact mirrors CytoSignal's split: a secreted ligand diffuses
(Gaussian kernel over a neighborhood), a membrane ligand acts on direct contacts
(Delaunay neighbors). Symbols are human/uppercase; mouse data is matched by
uppercasing symbols, as CytoSignal does.

Run:  pixi run -e dev python -m xcell.data.build_lr_db
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data"
FILES = {
    "interaction": f"{BASE}/interaction_input.csv",
    "complex": f"{BASE}/complex_input.csv",
    "gene": f"{BASE}/gene_input.csv",
    "protein": f"{BASE}/protein_input.csv",
}

OUT = Path(__file__).resolve().parent / "lr_pairs.csv"


def _fetch(url: str) -> list[dict]:
    print(f"  downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _truthy(v: str | None) -> bool:
    return str(v).strip().lower() == "true"


def build(rows: dict[str, list[dict]]) -> list[dict]:
    # uniprot -> gene symbol (prefer HGNC symbol, fall back to gene_name)
    uni2sym: dict[str, str] = {}
    gene_names: set[str] = set()
    for g in rows["gene"]:
        sym = (g.get("hgnc_symbol") or g.get("gene_name") or "").strip()
        uni = (g.get("uniprot") or "").strip()
        if uni and sym:
            uni2sym.setdefault(uni, sym)
        if sym:
            gene_names.add(sym)
        if g.get("gene_name"):
            gene_names.add(g["gene_name"].strip())

    # complex_name -> (subunit uniprots, secreted)
    complexes: dict[str, tuple[list[str], bool]] = {}
    for c in rows["complex"]:
        subs = [c.get(f"uniprot_{i}", "").strip() for i in range(1, 6)]
        subs = [s for s in subs if s]
        complexes[c["complex_name"].strip()] = (subs, _truthy(c.get("secreted")))

    # single uniprot -> secreted
    protein_secreted: dict[str, bool] = {}
    for p in rows["protein"]:
        protein_secreted[p["uniprot"].strip()] = _truthy(p.get("secreted"))

    def resolve(partner: str) -> tuple[list[str], bool] | None:
        """Return (gene symbols, secreted?) for a partner, or None if unresolved."""
        partner = partner.strip()
        if partner in complexes:
            subs, secreted = complexes[partner]
            syms = [uni2sym[u] for u in subs if u in uni2sym]
            if len(syms) != len(subs) or not syms:
                return None
            return syms, secreted
        if partner in uni2sym:  # single-chain protein referenced by uniprot
            return [uni2sym[partner]], protein_secreted.get(partner, False)
        if partner in gene_names:  # already a gene symbol
            return [partner], False
        return None

    out: dict[str, dict] = {}
    skipped = 0
    for it in rows["interaction"]:
        if it.get("directionality", "").strip() != "Ligand-Receptor":
            continue
        lig = resolve(it.get("partner_a", ""))
        rec = resolve(it.get("partner_b", ""))
        if lig is None or rec is None:
            skipped += 1
            continue
        lig_syms, lig_secreted = lig
        rec_syms, _ = rec
        ligand = "_".join(lig_syms)
        receptor = "_".join(rec_syms)
        interaction = f"{ligand}->-{receptor}"
        if interaction in out:
            continue
        out[interaction] = {
            "interaction": interaction,
            "ligand": ligand,
            "receptor": receptor,
            "type": "diffusion" if lig_secreted else "contact",
            "classification": (it.get("classification") or "").strip(),
            "source": "CellPhoneDB-v5",
        }
    print(f"  resolved {len(out)} pairs, skipped {skipped} unresolved", file=sys.stderr)
    return sorted(out.values(), key=lambda r: r["interaction"])


def main() -> None:
    rows = {name: _fetch(url) for name, url in FILES.items()}
    pairs = build(rows)
    diff = sum(1 for p in pairs if p["type"] == "diffusion")
    cont = len(pairs) - diff
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["interaction", "ligand", "receptor", "type", "classification", "source"]
        )
        w.writeheader()
        w.writerows(pairs)
    print(f"wrote {OUT} : {len(pairs)} pairs ({diff} diffusion, {cont} contact)")


if __name__ == "__main__":
    main()
