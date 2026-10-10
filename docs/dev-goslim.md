# The developmental GO slim: what it is and how it was derived

xcell ships a Gene Ontology (GO) slim for developmental biology as four
curated gene-set bundles (Genes → Sets → Import):

| Bundle file | Sets | Layout |
|---|---|---|
| `dev_goslim_coarse_mouse.json` | 32 | flat |
| `dev_goslim_fine_mouse.json` | 200 | one folder per coarse term |
| `dev_goslim_coarse_human.json` | 32 | flat |
| `dev_goslim_fine_human.json` | 199 | one folder per coarse term |

This page records exactly how the lists were made, so anyone using them can
state their provenance. The files that produced them are in the repository:

| File | Role |
|---|---|
| `backend/xcell/data/dev_goslim.tsv` | the slim itself: every term, its depth, its parent, and a flag for cross-links |
| `backend/xcell/data/build_dev_goslim.py` | validates the table against GO and writes the four bundles |
| `backend/xcell/data/carve_dev_goslim_candidates.py` | the mechanical first pass that proposed fine-term candidates |
| `backend/tests/test_build_dev_goslim.py`, `test_dev_goslim_bundles.py` | the contract: validation faults, closure, bundle shape, spot checks |
| `docs/superpowers/specs/2026-10-08-dev-goslim-design.md` | the design note with each decision and its reason |

## What a set is

Every set is one GO biological-process term, and its genes are the term's
**closed annotation**: every gene annotated to the term or to any descendant
reached along `is_a` and `part_of` relationships. This is the closure GO's own
tools (`map2slim`) compute and the one xcell's GO library uses, so a slim set
and the matching GO-library set are identical.

Two consequences to keep in mind:

- **Regulators are excluded unless annotated to the term itself.** GO does
  not propagate *regulation of X* into X. The Hedgehog set
  (`smoothened signaling pathway`) has Shh, Ihh, Smo, Gli1, Gli2, Gli3 and
  Sufu but not Ptch1 or Hhip; `left/right pattern formation` has Nodal and
  Lefty1 but not Lefty2; `canonical Wnt signaling pathway` has Axin2 and Apc
  but not Dkk1. This is GO's convention and was kept deliberately. The
  developmental signaling bundles (`dev_signaling_*.json`) carry inhibitors
  by role if you need them.
- **Human and mouse sets are independent.** Mouse genes come from the MGI
  GAF, human genes from the goa_human GAF (UniProtKB rows). No ortholog
  mapping is applied, so a human set is what GO says about human genes, which
  is often smaller: `blastocyst development` has 157 mouse and 20 human
  genes.

Annotations of every evidence code are used, including IEA, and rows with a
`NOT` qualifier are skipped, as in xcell's GO library.

## Source files and versions

| File | Version used | URL |
|---|---|---|
| `go-basic.obo` | `releases/2026-07-26` | https://current.geneontology.org/ontology/go-basic.obo |
| `mgi.gaf.gz` | generated 2026-01-05 | https://current.geneontology.org/annotations/mgi.gaf.gz |
| `goa_human.gaf.gz` | generated 2026-01-05 | https://current.geneontology.org/annotations/goa_human.gaf.gz |

Each bundle's description, and each set's description, states the GO release
and GAF date it was built from. Rebuilding with newer files changes
memberships; the term list does not change unless the table is edited.

## Structure: two axes, two depths

**Coarse** terms are the major processes and organ systems of development, on
two axes:

- *Mechanism*: pattern specification, cell fate commitment, stem cell
  population maintenance, gastrulation, blastocyst development, morphogenesis
  of an epithelium, mesenchyme development, cell migration, cell population
  proliferation, developmental growth, apoptotic process involved in
  development, developmental signaling pathways, extracellular matrix
  organization, cell adhesion, chromatin organization.
- *Anatomy*: nervous system, neural crest, circulatory system, hemopoiesis,
  skeletal system, muscle, renal system, respiratory system, digestive system
  (including the endoderm-derived organs liver, pancreas, thyroid, thymus),
  gland, epidermis, sensory organ, craniofacial, reproductive system, limb,
  placenta, connective tissue.

**Fine** terms are the narrower terms nested under each coarse term, 2 to 16
per folder. A coarse set is sized for enrichment of marker and differential
expression lists; a fine set for scoring cells.

Two coarse terms have no single GO term and are **unions** of their fine
terms: *developmental signaling pathways* (canonical Wnt, non-canonical Wnt,
BMP, TGF-β receptor, activin receptor, nodal, Notch, FGF receptor, smoothened,
retinoic acid receptor, hippo, ephrin receptor, EGF receptor, PDGF receptor,
VEGF receptor, IGF receptor) and *craniofacial development* (face, roof of
mouth, pharyngeal system, odontogenesis, embryonic cranial skeleton
morphogenesis, middle ear morphogenesis).

**Cross-links.** 29 fine terms sit in a folder whose coarse term they do not
descend from in GO, because that is where a developmental biologist looks for
them: osteoblast differentiation, intramembranous ossification and bone
mineralization under skeletal system development; liver, pancreas, thyroid
and thymus under digestive system development; stem cell proliferation under
stem cell population maintenance. The table marks each with `cross = x`. A
cross-linked term's genes are in its fine set and in its folder, but **not in
the coarse set**, which is always the coarse GO term's own closure.

### The 32 coarse terms

Sizes are closed memberships on the files above.

| Coarse term | GO id | Fine terms | Cross-links | Mouse genes | Human genes |
|---|---|---|---|---|---|
| pattern specification process | GO:0007389 | 8 | 1 | 486 | 266 |
| cell fate commitment | GO:0045165 | 6 | 0 | 285 | 156 |
| stem cell population maintenance | GO:0019827 | 3 | 2 | 152 | 52 |
| gastrulation | GO:0007369 | 9 | 5 | 172 | 94 |
| blastocyst development | GO:0001824 | 4 | 1 | 157 | 20 |
| morphogenesis of an epithelium | GO:0002009 | 5 | 1 | 578 | 275 |
| mesenchyme development | GO:0060485 | 4 | 1 | 257 | 186 |
| cell migration | GO:0016477 | 7 | 0 | 1032 | 753 |
| cell population proliferation | GO:0008283 | 6 | 0 | 678 | 325 |
| developmental growth | GO:0048589 | 5 | 0 | 553 | 195 |
| apoptotic process involved in development | GO:1902742 | 2 | 1 | 30 | 13 |
| developmental signaling pathways | union | 16 | 0 | 747 | 542 |
| extracellular matrix organization | GO:0030198 | 2 | 0 | 301 | 218 |
| cell adhesion | GO:0007155 | 3 | 1 | 813 | 807 |
| chromatin organization | GO:0006325 | 5 | 0 | 865 | 784 |
| nervous system development | GO:0007399 | 12 | 0 | 2296 | 1758 |
| neural crest cell development | GO:0014032 | 5 | 4 | 87 | 56 |
| circulatory system development | GO:0072359 | 11 | 0 | 1035 | 614 |
| hemopoiesis | GO:0030097 | 8 | 0 | 857 | 485 |
| skeletal system development | GO:0001501 | 9 | 4 | 560 | 309 |
| muscle structure development | GO:0061061 | 7 | 0 | 567 | 328 |
| renal system development | GO:0072001 | 7 | 0 | 383 | 213 |
| respiratory system development | GO:0060541 | 5 | 0 | 287 | 79 |
| digestive system development | GO:0055123 | 8 | 4 | 166 | 83 |
| gland development | GO:0048732 | 5 | 0 | 530 | 215 |
| epidermis development | GO:0008544 | 5 | 0 | 369 | 238 |
| sensory organ development | GO:0007423 | 7 | 0 | 732 | 372 |
| craniofacial development | union | 6 | 0 | 316 | 144 |
| reproductive system development | GO:0061458 | 6 | 2 | 370 | 157 |
| limb development | GO:0060173 | 5 | 0 | 225 | 91 |
| placenta development | GO:0001890 | 5 | 0 | 197 | 63 |
| connective tissue development | GO:0061448 | 4 | 2 | 276 | 123 |

The full list of fine terms, with the mouse size of each, is the table
`dev_goslim.tsv` itself.

## How the term list was derived

The slim was made on 2026-10-08 and 2026-10-09 in three steps. Step 1 is
mechanical and reproducible from the script; steps 2 and 3 are curation and
are recorded in the table and in the design note.

**Step 1: candidates by carving the ontology.** For each coarse term,
`carve_dev_goslim_candidates.py` listed its biological-process descendants
with a closed mouse membership of **40 to 350 genes**, largest first,
skipping any term whose Jaccard overlap with a term already listed under the
same root was **0.5 or more**, up to **14 per root**. The window was chosen
so that fine sets are large enough to score cells with UCell and small
enough to be specific; the overlap cap removes nested near-duplicates
(`neural tube development` versus `neurulation`) from the proposal, though
some were later added back on purpose.

Before that, an exploratory greedy cover of all genes under
`developmental process` (GO:0032502), with terms of 150 to 1200 genes and a
Jaccard cap of 0.3, produced 26 terms covering 5668 of the 6292 annotated
mouse genes. It picked organ systems almost exclusively (circulatory, renal,
ear, spermatogenesis) and no mechanism, which is why the slim has two axes
rather than one algorithmic list.

**Step 2: curation.** Terms were kept, dropped and added by hand:

- Coarse terms were chosen to cover the mechanisms and organ systems a
  developmental biologist expects, not to maximise gene coverage.
- Fine terms below the 40-gene window were added where the biology demands
  them (primitive streak formation, 12 mouse genes; limb bud formation, 9;
  zygotic genome activation, 1). All 25 fine terms with fewer than 20 mouse
  genes were kept deliberately: they are informative in enrichment, and a
  user scoring cells can see the size in the set's description.
- Three fine terms above 500 genes were kept (cell-cell adhesion 569, germ
  cell development 524, chromatin remodeling 694).
- Fine terms found under one coarse term in GO but belonging elsewhere by
  usage were placed where a biologist looks, and flagged as cross-links.
- The signaling pathways were included even though the signaling bundles
  overlap them, so the slim stands alone.
- A term appears in one folder only. `neural crest cell migration` is under
  neural crest, not also under cell migration.
- Terms whose names no longer exist in GO were resolved, not guessed:
  *neurulation* is now `neural tube formation` (GO:0001841, same id);
  *histone modification* and *DNA methylation* are obsolete in biological
  process and have no replacement there; *cell-cell adhesion via
  plasma-membrane adhesion molecules* was replaced by `cell-cell adhesion`
  (GO:0098609).

**Step 3: validation and build.** `build_dev_goslim.py` refuses the table if
any id is unknown, obsolete or an `alt_id`, if a name does not match the
ontology's, if a term is not a biological process, if a fine term's parent is
not a coarse row, if a fine term does not descend from its parent and is not
flagged `x` (or is flagged and does descend), or if a term appears twice at
one depth. It then computes the closures, writes the four bundles, rewrites
the `mouse_genes` column, and prints every term's mouse and human size and
every pair of sibling fine terms with Jaccard above 0.5.

### Known redundancies

Sibling fine terms with high overlap on mouse were kept where each name is
one a user searches for. The closest pairs:

| Pair | Jaccard (mouse) |
|---|---|
| mesonephros development ~ ureteric bud development | 0.93 |
| somitogenesis ~ somite development | 0.82 |
| telencephalon development ~ forebrain development | 0.68 |
| neural tube development ~ neural tube formation | 0.66 |
| epigenetic programming of gene expression ~ genomic imprinting | 0.64 |
| segmentation ~ somitogenesis | 0.64 |

In enrichment, such pairs will rise and fall together; that is redundancy in
GO, not independent evidence.

### Gaps

- `ectodermal cell fate commitment` (GO:0001712) has no human gene annotated
  and is absent from the human fine bundle.
- GO's nervous-system subtree is very large (911 descendant terms); 12 fine
  terms do not cover it. The sensory and neural crest folders take some of
  it.
- Hemopoiesis folders lean toward lineages (progenitor, primitive and
  definitive hemopoiesis, erythroid, megakaryocyte, myeloid, lymphoid,
  macrophage), not immune function.

## Regenerating

From `backend/`:

```
pixi run -e dev python xcell/data/build_dev_goslim.py
```

The script downloads go-basic.obo and both GAFs into xcell's gene-set-source
cache if they are not there (about 60 MB), validates the table, writes the
bundles and prints the report. Edit `dev_goslim.tsv`, never the JSON. To
propose candidates for a new or changed folder:

```
pixi run -e dev python xcell/data/carve_dev_goslim_candidates.py --root 'limb development' --min 20
```

## Citing

The term lists are xcell's; the memberships are the Gene Ontology
Consortium's annotations. Cite GO (Ashburner et al. 2000; The Gene Ontology
Consortium 2023) and state the release and GAF dates above, and MGI for the
mouse annotations (Baldarelli et al. 2024).
