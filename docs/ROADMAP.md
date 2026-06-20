# Roadmap / future work

Notes on planned improvements. Add new items as they come up.

## Ligand-Receptor (Analyze → Spatial → Ligand-Receptor)

- **Proper mouse→human ortholog table.** The bundled L-R database (`backend/xcell/data/lr_pairs.csv`) uses human UPPERCASE HGNC symbols. Mouse data currently works via case-insensitive matching (`Pdgfb` → `PDGFB`) in `prepare_ligrec`, mirroring CytoSignal. This is a heuristic and fails for genes whose mouse/human symbols genuinely differ, or 1:many ortholog cases. **We need to add a real ortholog mapping** (e.g. MGI `HOM_MouseHumanSequence`, or Ensembl BioMart orthologs) as an option, selectable by species, so mouse (and other species) genes map to the human database correctly. Until then, document the limitation in the tool.
- Possible follow-ups: Ensembl-ID → symbol fallback when `.var_names` are Ensembl IDs; expose `p_thresh` re-thresholding from stored p-values without re-running; per-interaction spatial-variability ranking (SPARK-X / Moran's I).
