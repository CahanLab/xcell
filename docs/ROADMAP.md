# Roadmap / future work

Notes on planned improvements. Add new items as they come up.

## Next up: core marker-gene set per cluster (adaptive)

Identify a **core set of genes that distinguishes each cluster from all others**
(per-cluster one-vs-rest markers), likely with **adaptive filtering** — i.e.
pick the minimal/robust gene set per cluster rather than a fixed top-N, adapting
thresholds (effect size, specificity, detection fraction) to each cluster's
separability. Open design questions to settle when we start: one-vs-rest vs
pairwise; ranking statistic (AUROC / log-fold-change / specificity score);
how "adaptive" chooses the cutoff (e.g. elbow on the ranked statistic, or a
target separability); de-duplhication across clusters; and how it surfaces in
the UI (a new Analyze → Genes tool writing a gene set per cluster). Build on the
existing diffexp / marker-gene plumbing.

## Cell typing (PySingleCellNet)

- **Intermediate vs Hybrid.** `pySingleCellNet.tl.categorize_classification`
  splits cells that clear more than one class threshold into *Intermediate*
  (the classes are close in a cell-type relatedness graph) and *Hybrid* (they
  are not). xcell has no such graph, so it reports those cells as **Ambiguous**.
  To do this properly we'd need the training-time graph — PySCN builds it with
  `paga_connectivities_to_igraph` on the reference — persisted alongside the
  classifier. Worth proposing upstream: `train_classifier` could stash the PAGA
  graph in the returned dict, which would make the full categorization portable
  to any consumer.
- **Gene symbols containing `_`.** PySCN encodes gene pairs as the string
  `geneA_geneB` and decodes with `split("_")`, so such symbols can't round-trip.
  xcell currently excludes them at training time and refuses clearly when too
  few remain. A delimiter that can't occur in a symbol (or storing pairs as
  tuples rather than joined strings) would fix this upstream; until then any
  dataset using underscores in gene names — including our own
  `toy_spatial.h5ad` — cannot train a classifier.
- **Held-out assessment in the UI.** `tl.create_classifier_report` +
  `pl.heatmap_classifier_report` give precision/recall/F1 per class against
  ground truth. The Train tab could hold out a fraction of cells, score the
  classifier on them, and show the confusion matrix — turning "it trained" into
  "here's how well it works" before anyone applies it to a query.
- **Cross-species classifiers.** Gene matching is exact, with an opt-in
  case-insensitive fallback for the `ACTB`/`Actb` case. A real ortholog table
  would let a mouse classifier be applied to human data (and vice versa) — the
  same missing piece as the L-R item below, so one table would serve both.

## Ligand-Receptor (Analyze → Spatial → Ligand-Receptor)

- **Proper mouse→human ortholog table.** The bundled L-R database (`backend/xcell/data/lr_pairs.csv`) uses human UPPERCASE HGNC symbols. Mouse data currently works via case-insensitive matching (`Pdgfb` → `PDGFB`) in `prepare_ligrec`, mirroring CytoSignal. This is a heuristic and fails for genes whose mouse/human symbols genuinely differ, or 1:many ortholog cases. **We need to add a real ortholog mapping** (e.g. MGI `HOM_MouseHumanSequence`, or Ensembl BioMart orthologs) as an option, selectable by species, so mouse (and other species) genes map to the human database correctly. Until then, document the limitation in the tool.
- Possible follow-ups: Ensembl-ID → symbol fallback when `.var_names` are Ensembl IDs; expose `p_thresh` re-thresholding from stored p-values without re-running; per-interaction spatial-variability ranking (SPARK-X / Moran's I).
