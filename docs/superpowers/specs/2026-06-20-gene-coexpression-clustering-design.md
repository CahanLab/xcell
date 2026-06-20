# Improve gene clustering from the genes panel (co-expression modules)

**Date:** 2026-06-20
**Branch:** `improve-gene-coexpression-clustering`
**Entry point:** Genes panel → gene set "…" → **Cluster genes** → produces new gene sets.
**Status:** Implemented 2026-06-20 (plan
`docs/superpowers/plans/2026-06-20-gene-coexpression-clustering.md`). New
`backend/xcell/gene_coexpression.py` + `method='auto'` in `cluster_gene_set`,
route params, and the modal's "Auto" default. 119 backend tests pass; frontend
tsc + build clean; verified live end-to-end (UI → API → backend produced module
gene sets on a real `.h5ad`).

---

## 1. Problem

The "Cluster genes" path (`adaptor.cluster_gene_set`) does not produce
pure-ish co-expression modules on real data. Root causes:

- **Inappropriate metrics.** `kmeans` runs Euclidean K-means on *raw*
  log-normalized expression vectors → dominated by gene magnitude, not
  co-expression pattern. `hierarchical`/`dbscan` use Pearson correlation
  distance, which is sensitive to dropouts/outliers in sparse single-cell data.
- **User must pre-specify K** (hierarchical, kmeans). DBSCAN avoids K but a
  single global `eps` can't cope with variable module density.
- **No post-clustering refinement** — raw partitions are returned as-is, so
  near-duplicate modules stay split, glued/bimodal modules stay impure, and
  tiny spurious modules survive.

Desired (from the user):
1. No need to pre-specify the number of clusters.
2. Post-clustering refinement: **merge** similar clusters, **split** impure
   ones, **eliminate** clusters with fewer than a minimum number of genes.

## 2. Current implementation (what we build on)

- `adaptor.cluster_gene_set(gene_names, method, k, cell_indices, eps,
  min_samples, layer, use_gene_mask) -> list[list[str]]`
  (`backend/xcell/adaptor.py:5443`). Builds `X_genes` = (n_genes, n_cells)
  from the normalized matrix (or a chosen layer), clusters by gene profile,
  returns clusters as gene-name lists; DBSCAN noise is a trailing group.
- Route `POST /cluster_gene_set` + `ClusterGeneSetRequest`
  (`backend/xcell/api/routes.py:1082`, `:2712`).
- `ClusterGeneSetModal.tsx` → `runClusterGeneSet()` → wraps each returned
  cluster into a new gene set under a `gene_clusters` folder. **Already renders
  a trailing noise/unassigned group**, so an "unassigned" bucket needs no
  data-flow change.
- Existing related-but-separate path (NOT touched): `run_cluster_genes`
  (Leiden on a gene–gene graph stored in `.var`) — different entry point.

Environment (verified): `sklearn 1.8` (`silhouette_score`), `scipy 1.17`
(`linkage`/`fcluster`/`squareform`), `numpy 2.4`. **No new dependencies.**

## 3. Design — new `method='auto'` (co-expression modules)

A new method, default in the modal. Existing `hierarchical`/`kmeans`/`dbscan`
branches are left intact for power users. Pipeline:

### 3.1 Robust similarity
- Source matrix as today: normalized `.X` (or chosen `layer`), optional
  `cell_indices`, optional gene mask. `X_genes` = (n_genes, n_cells).
- Compute a gene–gene correlation with a **`metric`** option:
  - `bicor` (**default**) — biweight midcorrelation (WGCNA's robust
    correlation; resistant to outlier cells).
  - `pearson`, `spearman` — alternatives.
- Distance = `1 − corr`, **signed** (anti-correlated genes are far → not
  co-clustered), clipped to `[0, 2]`.
- Vectorized: transform each gene's profile to its metric-specific standardized
  form (z-score for pearson; rank-then-z for spearman; biweight weights for
  bicor), then `corr = Zᵀ Z / n` via a single matmul. O(g²·n), fine for the
  tens–hundreds of genes in a gene set.
- Zero-variance / degenerate genes (constant across the chosen cells) are
  dropped up front with a clear error only if too few remain.

### 3.2 Auto-K base clustering
- **Average-linkage hierarchical clustering with a silhouette-selected cut.**
  Build `linkage(squareform(D), method='average')`, then sweep `K = 2..K_max`
  (`K_max = min(20, g-1)`), cut with `fcluster(criterion='maxclust')`, and keep
  the `K` with the best `silhouette_score(D, labels, metric='precomputed')`.
  No K to specify by the user. Deterministic, always yields a partition (every
  gene assigned), so the merge/prune steps always have module eigengenes to
  reassign against.
- *Why not HDBSCAN:* considered, but on small/weakly-structured gene sets it can
  label everything as noise (no surviving modules → nothing to reassign to). The
  silhouette cut may under- or over-segment, but that is exactly what the
  refinement pass (split / merge) is designed to correct — so a base that always
  partitions is the robust choice. Tiny clusters are removed by **prune**, which
  is one of the user's explicit refinement asks anyway.

### 3.3 Refinement pass (independently testable pure functions)
Applied in order: **split → merge → prune**.

- `_module_eigengene(profiles) -> vec` — PC1 of the standardized module
  profiles, sign-aligned to the mean profile (the module "eigengene").
- `_module_coherence(profiles) -> float` — fraction of variance explained by
  the eigengene (PVE). High for a coherent module, low for a glued/bimodal one.
- **Split** (`split_impure_modules`): for each module with coherence
  `< purity_threshold` and size `≥ 2·min_genes`, do a 2-way Ward split on its
  submatrix; recurse up to `max_split_depth`; keep the split only if both
  children are more coherent than the parent and meet `min_genes`.
- **Merge** (`merge_similar_modules`): compute each module's eigengene;
  iteratively merge the closest pair whose eigengene correlation
  `≥ merge_threshold`, recomputing eigengenes after each merge, until none
  qualify (WGCNA `mergeCloseModules`).
- **Prune** (`prune_small_modules`): drop modules with `< min_genes` genes;
  reassign each orphaned gene to the nearest module whose eigengene correlation
  `≥ reassign_floor`, else into a trailing **"unassigned"** bucket.

### 3.4 Output
`list[list[str]]`, modules ordered by size descending, trailing unassigned
bucket last (if any). **Unchanged return contract.**

### 3.5 Parameters (all defaulted; `auto` only)
| Param | Default | Meaning |
|---|---|---|
| `metric` | `bicor` | `bicor` \| `pearson` \| `spearman` |
| `min_genes` | `5` | min genes per surviving module (prune floor) |
| `merge_threshold` | `0.8` | eigengene-corr above which modules merge |
| `purity_threshold` | `0.5` | eigengene PVE below which a module is split |
| `max_split_depth` | `2` | recursion cap on splitting |
| `reassign_floor` | `0.5` | min eigengene-corr to reassign a pruned gene |

(Thresholds tuned against the synthetic tests; values above are the starting
point.)

## 4. Changes by layer

- **`backend/xcell/gene_coexpression.py`** (new): metric transforms (`bicor`/
  pearson/spearman → standardized profiles + corr/distance), eigengene,
  coherence, split/merge/prune primitives, and a top-level
  `auto_coexpression_modules(X_genes, gene_names, *, metric, min_genes,
  merge_threshold, purity_threshold, max_split_depth, reassign_floor)
  -> list[list[str]]`. Pure NumPy/scipy/sklearn — no adaptor/anndata
  coupling, so it is unit-testable in isolation.
- **`adaptor.cluster_gene_set`**: add `method='auto'` branch that builds
  `X_genes` (existing code) and delegates to `auto_coexpression_modules`; add
  the new keyword params (defaulted; ignored by other methods).
- **`routes.py`**: `ClusterGeneSetRequest` gains the new optional fields; route
  passes them through.
- **`ClusterGeneSetModal.tsx`** + `useData.ts`: add "Auto (recommended)" as the
  default method; when selected, hide K and show Metric + Min-genes +
  Merge/Split controls under a collapsible "Advanced" (defaults pre-filled);
  send the new fields in the payload.

## 5. Testing (TDD)

New `backend/tests/test_gene_coexpression.py`. Synthetic (n_cells × n_genes)
matrix with **planted ground-truth modules**:
- 3–4 modules of strongly co-expressed genes + uncorrelated noise genes.
- Two **near-duplicate** modules (shared latent factor) → must **merge**.
- One **glued bimodal** module (two anti-/un-correlated sub-programs) → must
  **split**.
- One **2-gene** spurious module → must be **pruned** to unassigned.

Assertions:
- `auto` recovers planted modules without K (high ARI vs ground truth).
- near-duplicates merged; glued module split; tiny module pruned.
- **bicor robustness:** inject outlier cells; bicor recovers modules where
  pearson degrades.
- Unit tests for each primitive (`_module_eigengene`, `_module_coherence`,
  `merge_similar_modules`, `split_impure_modules`, `prune_small_modules`) in
  isolation.
- Adaptor-level: `cluster_gene_set(method='auto')` returns a valid partition of
  the input genes (every gene appears exactly once across modules+unassigned).
- Existing `test_cluster_gene_set_mask.py` continues to pass (gene-mask path).

Frontend: `tsc --noEmit` + `vite build` clean; Playwright smoke on the isolated
stack (backend :8001 + vite :5180) — open a gene set, run Auto, confirm new
gene sets appear.

## 6. Out of scope
- Changing `run_cluster_genes` (the `.var` Leiden gene-graph path).
- Reworking the existing `hierarchical`/`kmeans`/`dbscan` branches beyond
  leaving them available (no behavior change).
- Importing the full WGCNA package (we reimplement only the needed primitives).

## 7. Build order
1. TDD `gene_coexpression.py` primitives (metric → distance; eigengene;
   coherence; split; merge; prune).
2. TDD `auto_coexpression_modules` end-to-end on synthetic planted modules.
3. Wire `cluster_gene_set(method='auto')` + adaptor params (+ test).
4. Route fields.
5. Frontend modal + payload (+ tsc/build).
6. Playwright smoke on isolated stack; clean up servers/artifacts.
7. Commit on the branch.
