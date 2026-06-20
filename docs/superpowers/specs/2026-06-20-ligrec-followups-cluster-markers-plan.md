# Continuation plan — LR follow-ups + core cluster markers (2026-06-20)

A self-contained handoff so a fresh context can resume. Covers (0) current
state, (A) core marker-gene-set-per-cluster, (B) antagonist-aware LR scoring.

---

## 0. Current state

**Branch:** `ligrec-and-viz-enhancements`, **4 commits ahead of `main`, NOT pushed.**
```
8b70f3b Docs: roadmap note ...
796cc11 LR: section-aware analysis, gene-subset option, ~35x speedup
1b51e60 Frontend: LR tool, label-transfer modal, movable panels, embedding labels, palettes
6f2551b Backend: ligand-receptor spatial scoring, subcluster-label transfer, task progress
```
Outstanding decision for the user: **push the branch / open a PR / merge to main.**
(Repo convention is feature branches merged to `main`.) Nothing is pushed yet.

**Done this session (all committed on the branch):**
- Feature 1 — Refine a category with subcluster labels: `transfer_obs_labels`
  (adaptor) + `POST /obs/transfer_labels` + `TransferLabelsModal.tsx` + Cells
  "…" menu item.
- Feature 2 — Movable/minimizable `FloatingPanel` (legends + the Source/Embedding
  "View" box); categorical labels overlaid at cluster centroids (Cells "…" →
  Show labels on plot); continuous `.obs` coloring honors the color scale +
  new palettes (sunset/ocean/grape/mint); label renamed "Continuous Color Scale".
- Feature 3 — Ligand-receptor spatial scoring (CytoSignal-style, no velocity):
  engine `backend/xcell/ligrec.py`; `prepare_ligrec`/`finalize_ligrec`/
  `get_ligrec_result`/`suggest_ligrec_params` on the adaptor; routes
  `/scanpy/ligrec/{suggest,result,prepare,finalize}`; `LigRecModal.tsx`.
  Bundled DB `backend/xcell/data/lr_pairs.csv` (2508 pairs from CellPhoneDB v5,
  built by `build_lr_db.py`); synthetic `toy_spatial_ligrec.h5ad`
  (`generate_toy_ligrec.py`). Results persist to `adata.obsm['lrscore']` +
  `['lrscore_significant']` + `adata.uns['lrscore']`; re-opening the tool restores
  the ranked table for re-selection. Case-insensitive gene match (mouse works).
  Section-aware (graphs drop cross-section edges + within-section permutation via
  `ligrec.section_permutation`). Optional gene subset (boolean `.var` column).
  Speed: matrix reduced to used genes + receptor side precomputed (~35x on a
  2k-gene panel). Live progress bar (task_manager `report(frac,msg)`; `/tasks`
  returns `progress`+`message`).

**Status:** 92 backend tests pass; frontend `tsc` + `vite build` clean.

**How to run / test (verified this session):**
- Backend tests: `cd backend && pixi run -e dev python -m pytest tests/ -q`
- Frontend check: `cd frontend && npx tsc --noEmit && npm run build`
- App (the user usually has their own on :8000/:5173 — DO NOT kill those):
  isolated stack = backend `XCELL_DATA_PATH=<abs>/backend/xcell/data/toy_spatial_ligrec.h5ad pixi run -e dev uvicorn xcell.main:app --host 127.0.0.1 --port 8001`
  + frontend `XCELL_BACKEND=http://127.0.0.1:8001 npx vite --port 5180 --strictPort`
  (vite.config reads `XCELL_BACKEND`). Drive with Playwright MCP at the chosen port.
  Always clean up your own servers + screenshot PNGs afterward.

**Key conventions / gotchas learned:**
- `is_categorical_dtype` deprecation warnings are pre-existing; match the
  surrounding code style.
- Categorical `.obs` `colorBy.values` are **numeric codes**; map via
  `colorBy.categories[code]` for display (this bit the embedding-label feature).
- LR significance is sensitive to **library-size normalization**: test fixtures
  need broadly-expressed receptors + a housekeeping gene (so lib sizes are
  stable) AND a globally-rare-but-locally-dense ligand source for decisive
  permutation significance. See `_signaling_field` / the section test in
  `tests/test_ligrec.py`.
- The async background-task pattern: adaptor returns/saves results; route submits
  `compute_fn` (may take a `report` arg) + `apply_fn` to `task_manager.submit`;
  frontend polls `/tasks/{id}` (use `pollTask(id, slot, onProgress)`).

---

## A. Core marker-gene set per cluster (adaptive filtering)  [the "next" task]

**Goal:** for each group in a categorical `.obs` column, return a **minimal,
robust "core" set of genes that distinguishes it from all others** — adaptive
cutoff, not a fixed top-N.

**Existing plumbing to build on (don't reinvent):**
- `adaptor.run_marker_genes` (adaptor.py:6871) — one-vs-rest `rank_genes_groups`
  + `filter_rank_genes_groups`; returns per-group DataFrames (names, logFC,
  pvals_adj, pct in/out). This is the natural base layer.
- `adaptor.run_diffexp` (1821) / `diffexp.py::compute_diffexp` — pairwise/2-group.
- `adaptor._find_elbow_kneedle` (4702) — reuse for the adaptive cutoff.
- Frontend `MarkerGenesModal.tsx`, `DiffExpModal.tsx` to model the UI on.
- Gene-set store (`gene_set_store.py`) to optionally write a gene set per cluster.

**Proposed approach (start simple, one-vs-rest):**
1. For each cluster c (vs all others), get per-gene stats from `rank_genes_groups`
   (Wilcoxon, `use_raw=False`): logFC, pct.in (`pct_nz_group`), pct.out
   (`pct_nz_reference`), pval_adj.
2. **Composite "coreness" score** per gene = effect size × specificity ×
   detection, e.g. `logFC_clipped * (pct.in - pct.out) * detection_gate`, plus a
   **cross-cluster specificity penalty**: down-weight genes that rank highly for
   many clusters (a gene marking 6 clusters isn't "core"). Track how many clusters
   each gene is a top candidate for.
3. **Adaptive selection** (the crux — pick the cutoff per cluster, not top-N):
   - Option (a) **elbow**: rank genes by score, `_find_elbow_kneedle` on the
     sorted scores → keep genes above the knee.
   - Option (b) **greedy-to-separability** (preferred, more principled): add
     genes by score one at a time; after each, score how well a simple rule
     (e.g. mean of selected markers, or a tiny logistic/centroid classifier)
     separates cluster c vs rest (AUROC on held-in cells). Stop when the marginal
     AUROC gain < ε or a target AUROC (e.g. 0.9) is reached → the minimal core set.
   - Gate by hard specificity floors (pct.in ≥ τ_in, pct.out ≤ τ_out, pval_adj <
     0.05) so junk never enters.
4. Return per-cluster: ordered core genes + their stats + achieved separability.
   Optionally write each as a gene set (`<col>_core_<cluster>`), and/or a summary
   table for a UI.

**Backend plan:** new `adaptor.core_cluster_markers(cluster_col, *, method='wilcoxon',
target_auroc=0.9, max_genes=50, min_pct_in=..., max_pct_out=..., gene_subset=None)`
→ dict per cluster. Probably a sync call (fast) or task_manager if slow on big
data. New route `/scanpy/core_markers`. Reuse `_resolve_gene_mask` for gene_subset.

**Frontend:** new modal (model on `MarkerGenesModal`) under Analyze → Genes:
pick cluster column + params (target separability, max genes), show per-cluster
core sets in a table, button to save as gene sets / color by.

**Tests (TDD):** synthetic adata with K clusters each having a few planted
exclusive marker genes + shared/noise genes; assert each cluster's core set
recovers its planted markers and excludes shared/noise; assert adaptive cutoff
returns *small* sets when separation is easy and larger when it's hard.

**Open design questions to confirm with the user before building:**
- one-vs-rest (default) vs also pairwise "what separates A from its nearest
  neighbor B"; ranking statistic (stick with Wilcoxon logFC+pct, or add AUROC);
  exact adaptive rule (elbow vs greedy-to-AUROC); how to handle the
  cross-cluster specificity penalty; UI surface + whether to auto-write gene sets.

---

## B. Antagonist/inhibitor-aware LR scoring  [from the Noggin/BMP discussion]

**Motivation:** CytoSignal (and our port), CellPhoneDB, LIANA, squidpy, COMMOT do
**not** model secreted antagonists/decoys (e.g. **Noggin/Nog** sequestering BMP).
**CellChat is the exception** — its communication probability multiplies the L–R
Hill term by cofactor factors and `CellChatDB` annotates *soluble agonists,
antagonists, co-stimulatory and co-inhibitory* cofactors. Adding spatial
antagonist-aware scoring would differentiate our tool (no spatial tool does it).
Refs: CellChat Nat Commun 2021 (s41467-021-21246-9); CellChat v2 spatial
(bioRxiv 2023.11.05.565674).

**Design (fits our existing engine cleanly):**
1. **Cofactor table:** `pathway/interaction → antagonist genes`. Two options:
   - (a) Extract from **CellChatDB** (R package `sqjin/CellChat`; `CellChatDB.human$interaction`
     has `agonist`/`antagonist`/`co_A_receptor`/`co_I_receptor` columns that key
     into `CellChatDB$cofactor`). Needs R/RDS parsing → emit a CSV like lr_pairs.csv.
     Join to our `lr_pairs.csv` by pathway (`classification` col) or by L–R genes.
   - (b) Start with a small **curated CSV** for major pathways and expand:
     BMP → {NOG, GREM1, GREM2, CHRD, CHRDL1, CHRDL2, BMPER, ...};
     WNT → {DKK1, DKK2, SFRP1..5, WIF1, NKD1};
     Activin/Nodal/TGFβ → {LEFTY1, LEFTY2, CER1, FST, FSTL3, BAMBI};
     Notch → {DLK1, ...}. Store as `backend/xcell/data/lr_antagonists.csv`
     with columns `pathway, antagonist_genes` (and/or per-interaction).
   Recommend (b) to start (Python-friendly, no R), seeded from CellChatDB values,
   with (a) as a later "import full cofactor table" step.
2. **Scoring change** in `ligrec.compute_ligrec`: after the per-cell, per-pair
   LRscore `S[c,p]`, multiply by an antagonist factor
   `A[c,p] = 1 / (1 + kappa * imputed_antag[c,p])`, where `imputed_antag` is the
   antagonist gene(s) for pair p's pathway, **imputed over the same Gaussian
   (diffusion) neighborhood** and normalized the same way (antagonists are
   secreted → diffusion kernel). If a pair has no annotated/expressed antagonist,
   `A=1` (no change). `kappa` = user strength (default ~1).
3. **Null handling:** treat the antagonist field like the receptor — **fixed
   (not permuted)** across permutations, applied identically to real and null
   scores. This keeps the null testing "is ligand→receptor signaling, net of the
   local inhibitory environment, spatially non-random." (Decide: apply antagonist
   factor before or after the spatial score smoothing — apply before smoothing,
   consistent with the product.)
4. Reuses existing machinery entirely: gaussian imputation, the score loop,
   permutation. Mostly: build an antagonist-gene → imputed-field map once
   (precompute like the receptor side), and a per-pair antagonist index.

**Backend plan:** load `lr_antagonists.csv` in `ligrec`; `prepare_ligrec` gains
`account_for_antagonists: bool = False`, `antagonist_strength: float`; pass to
`compute_ligrec`. Store the flag in `uns['lrscore']['params']`. Per-pair: resolve
antagonist genes by pathway (`classification`) using the same case-insensitive
matching; skip if none present.

**Frontend:** in `LigRecModal` config, add a checkbox "Account for
antagonists/inhibitors" + a strength slider; in the results table, mark which
interactions had an active antagonist (so users see where it applied).

**Tests (TDD):** synthetic field where a ligand source + receptor co-localize
(would be significant), but an **antagonist source** overlaps the receptor region
→ with `account_for_antagonists=True` the score/significance in the antagonist
zone drops markedly vs. off; and a control region without antagonist is
unchanged. Also: pairs with no annotated antagonist are identical to the current
result (factor=1).

**Open questions to confirm:** curated-CSV vs full CellChatDB extraction for v1;
exact antagonist factor form + default kappa; whether co-inhibitory *receptors*
(membrane) are in scope too (same idea, contact kernel); pathway-join key
(`classification` is coarse — may need a better map from L–R to pathway).

---

## Suggested order when resuming
1. Decide push/PR/merge of the current branch with the user.
2. Brainstorm + confirm open questions for whichever of A / B the user wants first
   (user flagged A — cluster markers — as "next"; B — antagonists — came up after).
3. TDD backend → route → frontend → verify (tests + tsc/build + a Playwright
   smoke on the isolated stack) → commit on the branch.
