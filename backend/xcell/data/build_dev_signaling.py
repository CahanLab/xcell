"""Build the developmental-signalling pathway bundles.

Run this to regenerate ``gene_sets/dev_signaling_{mouse,human}.json`` beside
it. The mouse table below is the source of truth; the human file is derived
by symbol case plus the explicit ortholog overrides in ``HUMAN_OVERRIDES``
(mouse *Fgf15* is human *FGF19*, rodents have two insulin genes, and so on).
Every symbol is checked against the bundled Ensembl tables for its species
before anything is written, so a typo or a renamed symbol fails the build
rather than silently missing from a heatmap.

Each pathway is a folder; each role is a set inside it, in the order the
heatmap should show them::

    ligands     the secreted / membrane signal and its agonists
    receptors   receptors and co-receptors that bind it
    modulators  extracellular and membrane factors that tune it (processing,
                presentation, decoys that act on the ligand rather than the
                cell), and the adaptors right under the receptor
    effectors   the intracellular transducers and the transcription factors
                that carry the signal
    inhibitors  antagonists at any level, extracellular or intracellular
    targets     transcriptional readouts — what goes up when the pathway is on

A gene that plays two roles (Axin2 is a Wnt target and an inhibitor) appears
in both sets; the heatmap labels it with the first. A role a pathway lacks is
simply absent.

Sources: the pathway chapters of Gilbert & Barresi *Developmental Biology*,
Reactome and KEGG pathway membership, and the primary literature on each
pathway's feedback targets (Wnt: Axin2/Lef1/Sp5; BMP: Id1-4/Msx; Hh:
Ptch1/Gli1/Hhip; Notch: Hes/Hey; FGF: Spry/Dusp6/Etv4-5; RA: Cyp26a1/Rarb).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / 'gene_sets'

ROLES = ['ligands', 'receptors', 'modulators', 'effectors', 'inhibitors', 'targets']

# Mouse symbols (MGI). Folder → role → genes.
MOUSE: dict[str, dict[str, list[str]]] = {
    'Wnt (canonical)': {
        'ligands': ['Wnt1', 'Wnt2', 'Wnt2b', 'Wnt3', 'Wnt3a', 'Wnt4', 'Wnt5a', 'Wnt5b', 'Wnt6',
                    'Wnt7a', 'Wnt7b', 'Wnt8a', 'Wnt8b', 'Wnt9a', 'Wnt9b', 'Wnt10a', 'Wnt10b',
                    'Wnt11', 'Wnt16', 'Rspo1', 'Rspo2', 'Rspo3', 'Rspo4', 'Ndp'],
        'receptors': ['Fzd1', 'Fzd2', 'Fzd3', 'Fzd4', 'Fzd5', 'Fzd6', 'Fzd7', 'Fzd8', 'Fzd9', 'Fzd10',
                      'Lrp5', 'Lrp6', 'Lgr4', 'Lgr5', 'Lgr6'],
        'modulators': ['Porcn', 'Wls', 'Gpc3', 'Gpc4', 'Tspan12', 'Ror2', 'Ryk', 'Csnk1e', 'Csnk1g1'],
        'effectors': ['Ctnnb1', 'Tcf7', 'Tcf7l1', 'Tcf7l2', 'Lef1', 'Dvl1', 'Dvl2', 'Dvl3',
                      'Bcl9', 'Bcl9l', 'Pygo1', 'Pygo2'],
        'inhibitors': ['Sfrp1', 'Sfrp2', 'Sfrp4', 'Sfrp5', 'Frzb', 'Dkk1', 'Dkk2', 'Dkk3', 'Dkk4',
                       'Wif1', 'Notum', 'Cer1', 'Sost', 'Sostdc1', 'Apcdd1', 'Shisa2', 'Shisa3',
                       'Znrf3', 'Rnf43', 'Nkd1', 'Nkd2', 'Axin1', 'Axin2', 'Apc', 'Gsk3b',
                       'Csnk1a1', 'Ctnnbip1', 'Tle1', 'Tle2', 'Tle3', 'Tle4', 'Dact1',
                       'Kremen1', 'Kremen2', 'Draxin'],
        'targets': ['Axin2', 'Lef1', 'Nkd1', 'Nkd2', 'Notum', 'Sp5', 'Lgr5', 'Ccnd1', 'Myc',
                    'Rnf43', 'Znrf3', 'Tcf7', 'Dkk1', 'Apcdd1', 'Cdx1', 'Cdx2', 'T', 'Msx1', 'Msx2'],
    },
    'Wnt/PCP': {
        'ligands': ['Wnt5a', 'Wnt5b', 'Wnt11'],
        'receptors': ['Fzd3', 'Fzd6', 'Fzd7', 'Ror1', 'Ror2', 'Ryk', 'Ptk7', 'Celsr1', 'Celsr2',
                      'Celsr3', 'Vangl1', 'Vangl2'],
        'modulators': ['Prickle1', 'Prickle2', 'Ankrd6', 'Scrib', 'Dact1'],
        'effectors': ['Dvl1', 'Dvl2', 'Dvl3', 'Daam1', 'Daam2', 'Rhoa', 'Rac1', 'Rock1', 'Rock2', 'Mapk8'],
    },
    'BMP': {
        'ligands': ['Bmp2', 'Bmp3', 'Bmp4', 'Bmp5', 'Bmp6', 'Bmp7', 'Bmp8a', 'Bmp8b', 'Bmp10',
                    'Bmp15', 'Gdf2', 'Gdf5', 'Gdf6', 'Gdf7', 'Amh'],
        'receptors': ['Bmpr1a', 'Bmpr1b', 'Bmpr2', 'Acvr1', 'Acvr2a', 'Acvr2b', 'Acvrl1', 'Amhr2'],
        'modulators': ['Rgma', 'Rgmb', 'Hjv', 'Bmper', 'Twsg1', 'Crim1', 'Eng'],
        'effectors': ['Smad1', 'Smad5', 'Smad9', 'Smad4'],
        'inhibitors': ['Nog', 'Chrd', 'Chrdl1', 'Chrdl2', 'Grem1', 'Grem2', 'Fst', 'Fstl1', 'Sost',
                       'Sostdc1', 'Cer1', 'Dand5', 'Nbl1', 'Smad6', 'Smad7', 'Smurf1', 'Bambi'],
        'targets': ['Id1', 'Id2', 'Id3', 'Id4', 'Msx1', 'Msx2', 'Smad6', 'Smad7', 'Bambi', 'Dlx5',
                    'Hey1', 'Gata2'],
    },
    'TGFβ/Activin/Nodal': {
        'ligands': ['Tgfb1', 'Tgfb2', 'Tgfb3', 'Inhba', 'Inhbb', 'Nodal', 'Gdf1', 'Gdf3', 'Gdf11', 'Mstn'],
        'receptors': ['Tgfbr1', 'Tgfbr2', 'Tgfbr3', 'Acvr1b', 'Acvr1c', 'Acvr2a', 'Acvr2b'],
        'modulators': ['Cripto', 'Cfc1', 'Ltbp1', 'Ltbp2', 'Ltbp3', 'Ltbp4', 'Thbs1', 'Eng', 'Bambi'],
        'effectors': ['Smad2', 'Smad3', 'Smad4', 'Foxh1'],
        'inhibitors': ['Lefty1', 'Lefty2', 'Fst', 'Fstl3', 'Inha', 'Smad7', 'Smurf1', 'Smurf2',
                       'Ski', 'Skil', 'Tgif1', 'Cer1', 'Bambi'],
        'targets': ['Serpine1', 'Lefty1', 'Lefty2', 'Nodal', 'Pitx2', 'Skil', 'Smad7', 'Tgfbi',
                    'Ccn2', 'Col1a1'],
    },
    'FGF': {
        'ligands': ['Fgf1', 'Fgf2', 'Fgf3', 'Fgf4', 'Fgf5', 'Fgf6', 'Fgf7', 'Fgf8', 'Fgf9', 'Fgf10',
                    'Fgf15', 'Fgf16', 'Fgf17', 'Fgf18', 'Fgf20', 'Fgf21', 'Fgf22', 'Fgf23'],
        'receptors': ['Fgfr1', 'Fgfr2', 'Fgfr3', 'Fgfr4', 'Fgfrl1'],
        'modulators': ['Kl', 'Klb', 'Fgfbp1', 'Fgfbp3', 'Frs2', 'Grb2', 'Sos1', 'Ptpn11'],
        'effectors': ['Etv4', 'Etv5', 'Etv1', 'Mapk1', 'Mapk3'],
        'inhibitors': ['Spry1', 'Spry2', 'Spry3', 'Spry4', 'Spred1', 'Spred2', 'Spred3', 'Dusp6', 'Il17rd'],
        'targets': ['Spry2', 'Spry4', 'Dusp6', 'Etv4', 'Etv5', 'Il17rd', 'Spred1'],
    },
    'Hedgehog': {
        'ligands': ['Shh', 'Ihh', 'Dhh'],
        'receptors': ['Ptch1', 'Ptch2', 'Smo'],
        'modulators': ['Gas1', 'Cdon', 'Boc', 'Lrp2', 'Hhat', 'Disp1', 'Scube2', 'Kif7'],
        'effectors': ['Gli1', 'Gli2', 'Gli3'],
        'inhibitors': ['Sufu', 'Hhip', 'Gpr161', 'Ptch1', 'Ptch2'],
        'targets': ['Ptch1', 'Gli1', 'Hhip', 'Ptch2', 'Foxf1', 'Foxa2', 'Nkx2-2', 'Olig2'],
    },
    'Notch': {
        'ligands': ['Dll1', 'Dll3', 'Dll4', 'Jag1', 'Jag2', 'Dlk1', 'Dlk2', 'Dner'],
        'receptors': ['Notch1', 'Notch2', 'Notch3', 'Notch4'],
        'modulators': ['Lfng', 'Mfng', 'Rfng', 'Adam10', 'Adam17', 'Psen1', 'Psen2', 'Ncstn',
                       'Aph1a', 'Psenen', 'Mib1', 'Mib2'],
        'effectors': ['Rbpj', 'Rbpjl', 'Maml1', 'Maml2', 'Maml3'],
        'inhibitors': ['Numb', 'Numbl', 'Nrarp', 'Fbxw7', 'Itch', 'Dlk1'],
        'targets': ['Hes1', 'Hes5', 'Hes7', 'Hey1', 'Hey2', 'Heyl', 'Nrarp'],
    },
    'Retinoic acid': {
        'ligands': ['Aldh1a1', 'Aldh1a2', 'Aldh1a3', 'Rdh10'],
        'receptors': ['Rara', 'Rarb', 'Rarg', 'Rxra', 'Rxrb', 'Rxrg'],
        'modulators': ['Crabp1', 'Crabp2', 'Rbp1', 'Rbp4', 'Stra6', 'Lrat'],
        'inhibitors': ['Cyp26a1', 'Cyp26b1', 'Cyp26c1', 'Dhrs3'],
        'targets': ['Cyp26a1', 'Rarb', 'Hoxa1', 'Hoxb1', 'Crabp2', 'Dhrs3', 'Stra6', 'Meis1', 'Meis2'],
    },
    'Hippo/YAP': {
        'receptors': ['Fat1', 'Fat4', 'Dchs1', 'Dchs2', 'Crb3'],
        'modulators': ['Nf2', 'Wwc1', 'Frmd6', 'Sav1', 'Mob1a', 'Mob1b', 'Amot', 'Amotl1', 'Amotl2'],
        'effectors': ['Yap1', 'Wwtr1', 'Tead1', 'Tead2', 'Tead3', 'Tead4'],
        'inhibitors': ['Lats1', 'Lats2', 'Stk3', 'Stk4', 'Vgll4', 'Ptpn14'],
        'targets': ['Ccn1', 'Ccn2', 'Ankrd1', 'Birc5', 'Axl', 'Amotl2', 'Lats2'],
    },
    'JAK-STAT': {
        'ligands': ['Il6', 'Lif', 'Osm', 'Cntf', 'Il11', 'Ctf1', 'Epo', 'Thpo', 'Gh', 'Prl', 'Ifng',
                    'Ifnb1', 'Il2', 'Il4', 'Il7', 'Il10', 'Il21'],
        'receptors': ['Il6ra', 'Il6st', 'Lifr', 'Osmr', 'Cntfr', 'Il11ra1', 'Epor', 'Mpl', 'Ghr',
                      'Prlr', 'Ifngr1', 'Ifngr2', 'Ifnar1', 'Ifnar2', 'Il2ra', 'Il2rb', 'Il2rg',
                      'Il4ra', 'Il7r', 'Il10ra', 'Il10rb'],
        'effectors': ['Jak1', 'Jak2', 'Jak3', 'Tyk2', 'Stat1', 'Stat2', 'Stat3', 'Stat4', 'Stat5a',
                      'Stat5b', 'Stat6'],
        'inhibitors': ['Socs1', 'Socs2', 'Socs3', 'Socs4', 'Socs5', 'Socs6', 'Socs7', 'Cish',
                       'Ptpn1', 'Ptpn2', 'Ptpn6', 'Pias1', 'Pias2', 'Pias3', 'Pias4'],
        'targets': ['Socs3', 'Socs1', 'Irf1', 'Bcl3', 'Junb', 'Cebpd', 'Pim1', 'Osmr'],
    },
    'EGF/ErbB': {
        'ligands': ['Egf', 'Tgfa', 'Areg', 'Ereg', 'Btc', 'Hbegf', 'Epgn', 'Nrg1', 'Nrg2', 'Nrg3', 'Nrg4'],
        'receptors': ['Egfr', 'Erbb2', 'Erbb3', 'Erbb4'],
        'modulators': ['Adam17', 'Adam10'],
        'effectors': ['Grb2', 'Sos1', 'Shc1', 'Ptpn11', 'Kras', 'Hras', 'Nras'],
        'inhibitors': ['Errfi1', 'Lrig1', 'Spry2', 'Cbl'],
        'targets': ['Egr1', 'Fos', 'Dusp6', 'Ier3', 'Areg'],
    },
    'PDGF': {
        'ligands': ['Pdgfa', 'Pdgfb', 'Pdgfc', 'Pdgfd'],
        'receptors': ['Pdgfra', 'Pdgfrb'],
        'effectors': ['Grb2', 'Sos1', 'Shc1', 'Ptpn11', 'Pik3r1', 'Plcg1'],
        'targets': ['Egr1', 'Fos', 'Ccnd1', 'Myc'],
    },
    'VEGF': {
        'ligands': ['Vegfa', 'Vegfb', 'Vegfc', 'Vegfd', 'Pgf'],
        'receptors': ['Kdr', 'Flt1', 'Flt4'],
        'modulators': ['Nrp1', 'Nrp2'],
        'effectors': ['Plcg1', 'Shc1', 'Grb2', 'Akt1'],
        'targets': ['Dll4', 'Esm1', 'Kdr', 'Angpt2', 'Apln', 'Nos3', 'Egr3'],
    },
    'IGF/insulin': {
        'ligands': ['Igf1', 'Igf2', 'Ins2'],
        'receptors': ['Igf1r', 'Insr', 'Igf2r'],
        'modulators': ['Igfbp1', 'Igfbp2', 'Igfbp3', 'Igfbp4', 'Igfbp5', 'Igfbp6', 'Igfbp7',
                       'Pappa', 'Pappa2', 'Htra1'],
        'effectors': ['Irs1', 'Irs2', 'Pik3ca', 'Pik3r1', 'Akt1', 'Akt2', 'Foxo1', 'Foxo3'],
        'inhibitors': ['Pten', 'Ptpn1', 'Grb10'],
    },
    'HGF/MET': {
        'ligands': ['Hgf', 'Mst1'],
        'receptors': ['Met', 'Mst1r'],
        'modulators': ['Hgfac', 'St14', 'Cd44'],
        'effectors': ['Gab1', 'Grb2', 'Ptpn11', 'Stat3'],
        'inhibitors': ['Spint1', 'Spint2', 'Cbl'],
    },
    'GDNF/RET': {
        'ligands': ['Gdnf', 'Nrtn', 'Artn', 'Pspn'],
        'receptors': ['Ret', 'Gfra1', 'Gfra2', 'Gfra3', 'Gfra4'],
        'effectors': ['Grb2', 'Shc1', 'Frs2', 'Gab1'],
        'inhibitors': ['Spry1', 'Cbl'],
        'targets': ['Etv4', 'Etv5', 'Wnt11', 'Spry1'],
    },
    'Eph/ephrin': {
        'ligands': ['Efna1', 'Efna2', 'Efna3', 'Efna4', 'Efna5', 'Efnb1', 'Efnb2', 'Efnb3'],
        'receptors': ['Epha1', 'Epha2', 'Epha3', 'Epha4', 'Epha5', 'Epha6', 'Epha7', 'Epha8', 'Epha10',
                      'Ephb1', 'Ephb2', 'Ephb3', 'Ephb4', 'Ephb6'],
        'modulators': ['Adam10'],
        'effectors': ['Ngef', 'Vav2', 'Vav3', 'Rac1', 'Rhoa', 'Nck1'],
    },
    'Semaphorin/plexin': {
        'ligands': ['Sema3a', 'Sema3b', 'Sema3c', 'Sema3d', 'Sema3e', 'Sema3f', 'Sema3g', 'Sema4a',
                    'Sema4b', 'Sema4c', 'Sema4d', 'Sema4f', 'Sema4g', 'Sema5a', 'Sema5b', 'Sema6a',
                    'Sema6b', 'Sema6c', 'Sema6d', 'Sema7a'],
        'receptors': ['Plxna1', 'Plxna2', 'Plxna3', 'Plxna4', 'Plxnb1', 'Plxnb2', 'Plxnb3', 'Plxnc1', 'Plxnd1'],
        'modulators': ['Nrp1', 'Nrp2'],
        'effectors': ['Dpysl2', 'Dpysl3', 'Rnd1', 'Mical1'],
    },
    'Slit/Robo': {
        'ligands': ['Slit1', 'Slit2', 'Slit3'],
        'receptors': ['Robo1', 'Robo2', 'Robo3', 'Robo4'],
        'modulators': ['Gpc1'],
        'effectors': ['Srgap1', 'Srgap2', 'Srgap3', 'Enah', 'Vasp', 'Evl'],
    },
    'Netrin': {
        'ligands': ['Ntn1', 'Ntn3', 'Ntn4', 'Ntn5', 'Ntng1', 'Ntng2'],
        'receptors': ['Dcc', 'Neo1', 'Unc5a', 'Unc5b', 'Unc5c', 'Unc5d', 'Dscam'],
        'effectors': ['Dock1', 'Nck1', 'Pak1'],
        'inhibitors': ['Draxin'],
    },
    'Endothelin': {
        'ligands': ['Edn1', 'Edn2', 'Edn3'],
        'receptors': ['Ednra', 'Ednrb'],
        'modulators': ['Ece1', 'Ece2'],
        'effectors': ['Gnaq', 'Gna11', 'Plcb1'],
        'targets': ['Hand2', 'Dlx5', 'Dlx6', 'Dlx3', 'Gata3', 'Msx1'],
    },
    'Prostaglandin': {
        'ligands': ['Ptgs1', 'Ptgs2', 'Ptges', 'Ptges2', 'Ptgds', 'Ptgis', 'Tbxas1', 'Pla2g4a'],
        'receptors': ['Ptger1', 'Ptger2', 'Ptger3', 'Ptger4', 'Ptgfr', 'Ptgdr', 'Ptgdr2', 'Ptgir', 'Tbxa2r'],
        'modulators': ['Slco2a1'],
        'effectors': ['Gnas', 'Creb1'],
        'inhibitors': ['Hpgd'],
        'targets': ['Ptgs2', 'Nr4a1', 'Fos', 'Il6'],
    },
    'Hypoxia/HIF': {
        'modulators': ['Egln1', 'Egln2', 'Egln3', 'Hif1an'],
        'effectors': ['Hif1a', 'Epas1', 'Hif3a', 'Arnt', 'Arnt2'],
        'inhibitors': ['Vhl', 'Egln1', 'Hif1an'],
        'targets': ['Vegfa', 'Slc2a1', 'Pgk1', 'Ldha', 'Bnip3', 'Car9', 'Egln3', 'Adm', 'Ndrg1',
                    'Pdk1', 'Ankrd37', 'P4ha1'],
    },
}

# Where the human symbol is not just the mouse symbol upper-cased.
HUMAN_OVERRIDES: dict[str, str] = {
    'Fgf15': 'FGF19',    # the mouse Fgf15 / human FGF19 ortholog pair
    'Ins2': 'INS',       # rodents carry two insulin genes; Ins2 is the ancestral one
    'Car9': 'CA9',       # carbonic anhydrase 9
    'Gh': 'GH1',         # growth hormone
    'Il11ra1': 'IL11RA',
    'Il6ra': 'IL6R',
    'Il4ra': 'IL4R',
    'T': 'TBXT',         # brachyury; the bundled mouse table still spells it T
}

DESCRIPTION = (
    'The major developmental signalling pathways, one folder per pathway, with '
    'each pathway\'s genes split by role in a fixed order: ligands, receptors, '
    'modulators, effectors, inhibitors, targets. Pick a whole folder in the '
    'heatmap to see a pathway laid out by role; the targets set is the '
    'pathway\'s transcriptional readout. A gene with two roles appears in both.'
)


def to_human(symbol: str) -> str:
    return HUMAN_OVERRIDES.get(symbol, symbol.upper())


def build(species: str) -> dict:
    convert = (lambda s: s) if species == 'mouse' else to_human
    sets = []
    for folder, roles in MOUSE.items():
        for role in ROLES:
            genes = roles.get(role)
            if not genes:
                continue
            converted: list[str] = []
            for g in genes:
                h = convert(g)
                if h not in converted:
                    converted.append(h)
            sets.append({'name': role, 'folder': folder, 'genes': converted})
    return {
        'name': f'Developmental signalling pathways ({species})',
        'description': DESCRIPTION,
        'sets': sets,
    }


def unknown_symbols(bundle: dict, species: str) -> list[str]:
    """Symbols the bundled Ensembl table for `species` does not know."""
    sys.path.insert(0, str(HERE.parent.parent))
    from xcell import gene_symbols
    known = set(gene_symbols.load_table(species).values())
    return sorted({
        f"{s['folder']}/{s['name']}: {g}"
        for s in bundle['sets'] for g in s['genes'] if g not in known
    })


def main() -> int:
    problems: list[str] = []
    for folder, roles in MOUSE.items():
        for role, genes in roles.items():
            if role not in ROLES:
                problems.append(f'{folder}: unknown role {role!r}')
            if len(set(genes)) != len(genes):
                problems.append(f'{folder}/{role}: duplicate gene')
    bundles = {sp: build(sp) for sp in ('mouse', 'human')}
    for sp, bundle in bundles.items():
        problems += [f'{sp}: {p}' for p in unknown_symbols(bundle, sp)]
    if problems:
        print('\n'.join(problems))
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    for sp, bundle in bundles.items():
        path = OUT_DIR / f'dev_signaling_{sp}.json'
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + '\n')
        print(f'wrote {path.name}: {len(bundle["sets"])} sets, '
              f'{sum(len(s["genes"]) for s in bundle["sets"])} genes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
