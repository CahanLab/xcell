# Developmental GO slim — design

A curated GO slim for developmental biology, shipped as gene-set bundles at
two depths: **coarse** (one set per major developmental process or organ
system, for enrichment of markers and DE lists) and **fine** (3 to 11 terms
under each coarse term, sized for scoring cells). The three slims the GO
Consortium distributes carry two developmental terms each, so the gap is real.
Both depths were asked for; nesting them is what makes them one object.

Decisions marked **[decision]** were made in the brainstorm and can be
revisited; the term table is the part that needs Patrick's eye.

## The object

One table, `backend/xcell/data/dev_goslim.tsv`, is the source of truth:

| column | meaning |
|---|---|
| `level` | `coarse` or `fine` |
| `parent` | for a fine row, the id of its coarse row (a GO id or `union:<id>`) |
| `go_id` | the term, or `union:<id>` for a coarse row that has no single GO term (`union:signaling`, `union:craniofacial`) |
| `name` | the GO name, checked against go-basic.obo at build time so a wrong id fails loudly |
| `mouse_genes` | informational, rewritten by the build: closed membership on the MGI GAF |
| `cross` | `x` when a fine term is not a GO descendant of its coarse term; must be set deliberately |

Two axes, both as coarse rows **[decision]**:

- **Mechanism** (15): pattern specification, cell fate commitment, stem cell
  population maintenance, gastrulation, blastocyst development, morphogenesis
  of an epithelium, mesenchyme development, cell migration, cell population
  proliferation, developmental growth, apoptotic process involved in
  development, developmental signaling pathways (union), extracellular matrix
  organization, cell adhesion, chromatin organization.
- **Anatomy** (17): nervous, neural crest, circulatory, hemopoiesis, skeletal,
  muscle, renal, respiratory, digestive (with liver, pancreas, thyroid, thymus
  as endoderm organs), gland, epidermis, sensory organ, craniofacial (union),
  reproductive, limb, placenta, connective tissue.

A coarse set is the GO term's closed membership (ancestors included, as in
`go_semantic.closed_members`), or for a `union:` row the union of its fine sets.
A fine set is its term's closed membership. **No size bounds** at either level:
a slim is curated, and the build *reports* sizes rather than enforcing them.

**Cross-links.** A fine term may sit under a coarse term it does not descend
from in GO (osteoblast differentiation under skeletal system development;
liver development under digestive; the pathways under the signaling union).
Then the folder shows the term but the coarse set does not contain its genes.
The build refuses a non-descendant without the `x` flag, so every such
placement is a recorded choice. 29 of the 200 draft fine rows are cross-links.

**Regulators are not in a pathway set unless annotated to it.** The closure
follows `is_a` and `part_of`, as GO's own tools do; *regulation of smoothened
signaling pathway* does not propagate into *smoothened signaling pathway*. So
the Hedgehog set has Shh, Ihh, Smo, Gli1–3 and Sufu but not Ptch1 or Hhip, and
the canonical Wnt set has Axin2 and Apc but not Dkk1. **[decision]** Kept GO's
convention, because a user comparing the slim against the GO library expects
the same sets; the signaling bundles are where inhibitors live by role.

**Signaling pathways are in** (Patrick, 2026-10-08), as GO's pathway terms:
canonical and non-canonical Wnt, BMP, TGF-beta, activin, nodal, Notch, FGF,
Hedgehog (smoothened), retinoic acid, Hippo, ephrin, EGF, PDGF, VEGF, IGF.
These overlap the hand-curated developmental signaling bundles; the slim stands
alone and the bundles stay role-resolved.

## Draft numbers (mouse, MGI GAF 2026-01-05 closed over go-basic releases/2026-07-26)

- 32 coarse terms, 200 fine terms (the TSV as committed).
- Coarse sizes run from 30 (apoptotic process involved in development) to
  2296 (nervous system development); the signaling union is 747.
- 25 fine terms have fewer than 20 mouse genes (zygotic genome activation has
  1; ectodermal cell fate commitment 3; stomach development 4). Too small to
  score cells with, but informative in enrichment. Patrick decides which stay.
- 3 fine terms exceed 500: cell-cell adhesion 569, germ cell development 524,
  chromatin remodeling 694.
- Dropped because the name no longer exists in GO: *neurulation*, *DNA
  methylation*, *histone modification*, the two *cell-cell adhesion via
  plasma-membrane adhesion molecules* terms (obsoleted upstream).

## Build

`backend/xcell/data/build_dev_goslim.py`, beside `build_dev_signaling.py`,
with the same contract: run it to regenerate the JSON; the table is the truth.

1. Load go-basic.obo and the mouse and human GAFs through `go_semantic`'s
   loaders from the gene-set-source cache; download them through
   `GOSource._download_files` if absent (first run on a clean machine costs
   ~60 MB; the files are the ones the GO library uses).
2. Validate the table: every id exists and is not obsolete, is
   `biological_process`, its name matches the ontology, no id appears twice at
   one level, every fine row's parent is a coarse row, and a non-descendant
   fine term carries `x`. Any failure is a non-zero exit naming the row.
3. Per species, compute closed membership, build the sets, and write four
   bundles to `gene_sets/`:
   `dev_goslim_coarse_{mouse,human}.json` (one folder) and
   `dev_goslim_fine_{mouse,human}.json` (one folder per coarse term, named by
   the coarse term). Each set's description carries the GO id, the size and,
   for a fine set, its parent; the bundle description carries the obo release
   and GAF date, so a user can see which annotation release a set reflects.
4. Rewrite `mouse_genes` in the TSV and print a report: per coarse folder, each
   fine term's mouse and human size, and any sibling pair with Jaccard above
   0.5, so redundancy is a visible number before the table is accepted.

Human symbols come straight from goa_human (UniProtKB rows), not from the
mouse table by ortholog mapping **[decision]**, because GO annotates the two
species independently and the human closure is the honest human set.

## Shipping

The bundles go out through the existing curated-library route
(`GET /api/gene_sets/library`, the Import dialog), like the signaling bundles.
That gives folders, offline use and frozen membership with the release
recorded, with **no frontend change** **[decision]**. The alternative, a GO
source library built from the files on disk at fetch time, was not chosen:
the Library modal imports into one folder, so the nesting would be lost, and
membership would drift with whatever GAF the user last fetched.

Regenerating the bundles at each xcell release (part of the release
checklist, not automated) keeps them within a few months of current GO.

## Tests

`backend/tests/test_build_dev_goslim.py`, toy obo and GAF in the file:

- a bad id, an obsolete id, a name mismatch, a non-BP term, and a
  non-descendant without `x` each fail validation naming the row;
- a union coarse set is the union of its fine sets, and a term coarse set is
  the closure;
- the fine bundle has one folder per coarse term and parses through
  `gene_set_library._parse_bundle`;
- the report lists a sibling pair above the Jaccard threshold.

Plus the real-data check that is not a test: build on the cached files, open
the fine bundle on the E11.5 limb data and the Grosswendt gastrulation set,
score a dozen fine sets with UCell and confirm they separate the cell types
they should (chondrocyte differentiation on the condensations, left/right
pattern formation on the node). Terms that separate nothing anywhere are
candidates to drop.

## Out of scope

- A general slim-builder UI (pick roots, window, redundancy cap). The carve
  that proposed this table was a one-off script; if a second curated slim is
  wanted, that is the time to make it a tool.
- Custom term lists as GO source libraries.
- Ortholog-mapped human sets.
