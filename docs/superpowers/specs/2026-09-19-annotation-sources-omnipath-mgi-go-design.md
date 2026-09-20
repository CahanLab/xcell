# Annotation sources: OmniPath, MGI GXD by stage, and GO semantic similarity

Date: 2026-09-19. Status: built without review, as asked; decisions Patrick
did not make are marked **[decision]**. Extends
`2026-09-17-gene-annotation-resources-design.md`, whose `Source` adapter,
cache and Library modal are reused unchanged — every new source is one class
in `gene_set_sources.py` and needed no new frontend for browsing, fetching or
importing. ARCHS4 was explicitly left out.

## What each source gives

**OmniPath** (`omnipath`, mouse and human, from `omnipathdb.org/interactions`):

- `collectri` — one **signed** set per transcription factor, "`<TF> regulon`":
  activated targets as the set, repressed targets as `genes_down`. That is the
  directional format UCell already scores, so a regulon imported from here
  gives a TF-activity score in one step. **[decision]** Targets of unknown or
  conflicting sign count as activated, CollecTRI's own convention
  (`consensus_stimulation` / `consensus_inhibition` decide; ~5 % of rows
  carry both). Mouse: 867 regulons.
- `ligrec` — the receptors of each ligand ("`Wnt5a receptors`") and the
  ligands of each receptor ("`Fzd1 ligands`"), from CellPhoneDB, CellChatDB,
  CellTalkDB, ICELLNET, connectomeDB2020, Cellinker and Baccin2019 as
  OmniPath curates them (its `omnipath` and `ligrecextra` datasets, filtered
  by `resources=`). A complex (`ITGA10_ITGB1`) contributes every subunit; the
  raw pairs are kept in the library as `pairs` for later use. Mouse: 2,109
  sets.
- **[decision]** OmniPath prints a UniProt accession (`D6RFS9`,
  `A0A8Q0P8A2`) in the symbol column when an entry has no symbol. Those are
  dropped by shape; no dataset spells a gene that way. OmniPath publishes no
  data version, so the fetch date is recorded as the version.

**MGI GXD** (`mgi`, mouse only, one library per Theiler stage `ts1`…`ts28`):
the genes the Gene Expression Database has seen **detected** in each
anatomical structure (EMAPA) at that stage, one set per structure, with the
assay mix in the description ("RNA in situ 787, RT-PCR 133, …"). Fetched from
the undocumented but stable summary export
`gxd/report.txt?theilerStage=<n>&detected=Yes` — a whole stage is ~35 MB and
ten seconds, done as a task. TS19 (E11.5): 798 structures; "limb" carries
697 genes. **[decision]** Stage libraries rather than structure queries,
because a stage is what a dataset has; the structure is what the user
searches for inside it. MouseMine still answers but its data is a January
2025 snapshot, and the Alliance bulk file has no absent calls and no symbols,
so neither was used.

**Gene Ontology** (`go`, one library per species, `annotations`): fetches
`go-basic.obo` and the species GAF (`mgi.gaf.gz` / `goa_human.gaf.gz`) from
the GO Consortium into `<cache>/go/`, keeps them as files, and builds sets
from them — one per term with 5–500 genes, **ancestors included**, so
"collagen fibril organization" carries its children's genes too. The files
are what the gene map's GO channel reads.

## GO semantic similarity in the gene map

`go_semantic.py` (pure) parses the OBO into `is_a` / `part_of` parents
(**[decision]** `regulates` does not propagate: regulating a process is not
taking part in it), skips obsolete terms and resolves `alt_id`s; parses the
GAF skipping `NOT` qualifiers and rows from other databases (ComplexPortal,
RNAcentral); closes each gene's annotations over its ancestors; and weights
every term by information content, `IC = −log(fraction of the species'
annotated genes under the term)`, over the whole GAF rather than the genes
asked about.

**[decision]** The similarity is SimGIC (Pesquita 2007):
`Σ IC(shared ancestors) / Σ IC(union of ancestors)`. It is one sparse
matrix product, so 3,000 genes cost nothing, and it is exactly what the
annotation channel lacks — there, sharing "biological process" counts as
much as sharing "chondrocyte differentiation"; here the root has IC 0.
Parsed structures are memoised by file mtime because the OBO is 30 MB.

The gene map gains `go_weight`, `go_aspect` (bp | mf | cc) and `go_species`;
the modal shows a fourth channel, enabled only once the GO library for the
species is cached, and the summary reports how many genes were annotated
and over how many terms. Defaults in `config.yaml`: `gene_map.go_weight
1.0`, `go_aspect bp`.

Verified on a 36-gene mouse dataset with three planted programs: GO alone
groups Sox9/Col2a1/Comp/Ihh with Col1a1/Col3a1/Bglap (skeletal) apart from
Myc/Cdkn1a/Ccnd1/Actb (proliferation and housekeeping); expression plus GO
recovers the three programs exactly.

## Small things fixed on the way

- Library names sort numerically in the modal, so Theiler stages read TS1,
  TS2 … TS10 rather than TS1, TS10, TS11.
- A directional library set travels through search (`genes_down`), the
  overlap check (`genesDown`) and import (`genes_down_resolved`); the
  import helper already understood the last step.

## Out of scope

- An OmniPath ligand–receptor channel for the gene map (the pairs are cached
  for it).
- MGI stages beyond one-at-a-time fetches; a "detected anywhere at TSn"
  union set.
- Resnik / Lin / Wang term-pair similarities; SimGIC was chosen for being
  vectorisable and needing no best-match search.
- Human GO / OmniPath human were built and tested offline but only mouse was
  fetched live.
