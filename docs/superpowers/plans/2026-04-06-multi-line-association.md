# Multi-Line Combined Association Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to combine multiple drawn lines (one per tissue section/replicate) into a single pooled association analysis, increasing statistical power.

**Architecture:** Extract the core spline regression logic into a shared helper (`_run_spline_association`), add a new `test_multi_line_association` method that projects and pools cells across lines, expose it via a new API endpoint, and extend the frontend Lines panel with checkboxes and a multi-line Line Tools modal mode. Also add a cell selection highlight in the Cell Panel.

**Tech Stack:** Python (FastAPI, NumPy, SciPy, pandas, statsmodels), TypeScript (React, Zustand)

---

### Task 1: Extract Shared Spline Regression Helper (Backend)

**Files:**
- Modify: `xcell/backend/xcell/adaptor.py:1218-1609`

The current `test_line_association` method does line lookup, cell projection, AND the core spline regression. Extract the regression portion (lines 1332-1609) into a reusable `_run_spline_association` method, then refactor `test_line_association` to call it.

- [ ] **Step 1: Add `_run_spline_association` method**

Insert this method immediately before `test_line_association` (before line 1218). This is the core regression logic extracted verbatim from `test_line_association`, taking pre-computed test values and cell indices as input:

```python
    def _run_spline_association(
        self,
        test_values: np.ndarray,
        cell_indices: np.ndarray,
        gene_mask: np.ndarray,
        n_spline_knots: int = 5,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Core spline regression for gene-variable association testing.

        Fits cubic B-spline regression for each gene against a spatial variable,
        then tests significance via F-test. Used by both single-line and
        multi-line association methods.

        Args:
            test_values: Normalized [0,1] spatial variable per cell (position or distance).
            cell_indices: Indices into self.adata for each cell.
            gene_mask: Boolean mask over genes to test.
            n_spline_knots: Number of interior knots for the B-spline basis.
            fdr_threshold: FDR threshold for significance.
            top_n: Maximum genes to return per module.

        Returns:
            Dict with keys: positive, negative, modules, n_cells, n_significant,
            n_positive, n_negative, n_modules, fdr_threshold, diagnostics.
        """
        from scipy.interpolate import BSpline
        from scipy.stats import f as f_dist
        from statsmodels.stats.multitest import multipletests

        n_cells_used = len(cell_indices)

        # Get expression matrix for selected cells and genes
        X = self.adata.X[cell_indices][:, gene_mask]
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        n_genes = X.shape[1]
        gene_names = self.adata.var_names[gene_mask].tolist()

        # Build B-spline basis matrix
        pos = test_values
        degree = 3
        n_interior = n_spline_knots
        interior_knots = np.quantile(pos, np.linspace(0, 1, n_interior + 2)[1:-1])

        knots = np.concatenate([
            np.repeat(0.0, degree + 1),
            interior_knots,
            np.repeat(1.0, degree + 1),
        ])

        n_basis = len(knots) - degree - 1

        B = np.zeros((n_cells_used, n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            B[:, i] = spline(pos)

        design = np.column_stack([np.ones(n_cells_used), B])
        k = n_basis

        try:
            XtX = design.T @ design
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ design.T @ X
        except np.linalg.LinAlgError:
            raise ValueError("Singular design matrix. Try fewer spline knots.")

        predicted = design @ beta
        residuals = X - predicted
        rss_full = np.sum(residuals ** 2, axis=0)

        gene_means = X.mean(axis=0)
        rss_null = np.sum((X - gene_means) ** 2, axis=0)

        df1 = k
        df2 = n_cells_used - k - 1

        rss_full_safe = np.maximum(rss_full, 1e-10)
        f_stat = ((rss_null - rss_full) / df1) / (rss_full_safe / df2)
        f_stat = np.maximum(f_stat, 0)

        p_values = 1 - f_dist.cdf(f_stat, df1, df2)

        _, fdr, _, _ = multipletests(p_values, method='fdr_bh')

        r_squared = 1 - rss_full / np.maximum(rss_null, 1e-10)
        r_squared = np.clip(r_squared, 0, 1)

        amplitude = predicted.max(axis=0) - predicted.min(axis=0)

        pos_centered = pos - pos.mean()
        pred_centered = predicted - predicted.mean(axis=0)
        direction = np.zeros(n_genes)
        for g in range(n_genes):
            if np.std(pred_centered[:, g]) > 1e-10:
                corr = np.corrcoef(pos_centered, pred_centered[:, g])[0, 1]
                direction[g] = corr if not np.isnan(corr) else 0
            else:
                direction[g] = 0

        results = pd.DataFrame({
            'gene': gene_names,
            'f_stat': f_stat,
            'pval': p_values,
            'fdr': fdr,
            'r_squared': r_squared,
            'amplitude': amplitude,
            'direction': direction,
        })

        results['score'] = -np.log10(results['fdr'] + 1e-300) * results['amplitude']

        sig_mask = results['fdr'] < fdr_threshold
        n_significant = sig_mask.sum()

        pos_mask = sig_mask & (results['direction'] > 0)
        neg_mask = sig_mask & (results['direction'] < 0)

        positive_genes = (
            results[pos_mask]
            .nlargest(top_n, 'score')
            [['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']]
            .to_dict('records')
        )

        negative_genes = (
            results[neg_mask]
            .nlargest(top_n, 'score')
            [['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']]
            .to_dict('records')
        )

        # Module-based clustering
        n_profile_points = 50
        profile_positions = np.linspace(0.0, 1.0, n_profile_points)
        profile_design = np.zeros((n_profile_points, n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            profile_design[:, i] = spline(profile_positions)
        profile_design_full = np.column_stack([np.ones(n_profile_points), profile_design])

        sig_indices = np.where(sig_mask.values)[0]
        modules = []

        if len(sig_indices) > 0:
            sig_profiles = profile_design_full @ beta[:, sig_indices]

            prof_min = sig_profiles.min(axis=0)
            prof_max = sig_profiles.max(axis=0)
            prof_range = prof_max - prof_min
            prof_range[prof_range < 1e-10] = 1.0
            norm_profiles = (sig_profiles - prof_min) / prof_range

            if len(sig_indices) == 1:
                cluster_labels = np.array([0])
            else:
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist

                profile_matrix = norm_profiles.T
                dists = pdist(profile_matrix, metric='correlation')
                dists = np.clip(dists, 0, 2)

                Z = linkage(dists, method='average')
                cluster_labels = fcluster(Z, t=0.5, criterion='distance') - 1

            sig_results = results.iloc[sig_indices].reset_index(drop=True)
            n_modules = int(cluster_labels.max()) + 1

            gene_peak_positions = np.argmax(norm_profiles, axis=0) / max(n_profile_points - 1, 1)

            for mod_idx in range(n_modules):
                member_mask = cluster_labels == mod_idx
                member_genes = sig_results[member_mask]
                member_profiles = norm_profiles[:, member_mask]
                member_peak_positions = gene_peak_positions[member_mask]

                rep_profile = member_profiles.mean(axis=1)
                pattern = self._classify_profile_pattern(rep_profile, profile_positions)

                member_genes = member_genes.copy()
                member_genes['peak_position'] = member_peak_positions
                member_genes_sorted = member_genes.sort_values('peak_position')
                if len(member_genes_sorted) > top_n:
                    member_genes_sorted = member_genes_sorted.head(top_n)

                gene_records = []
                for row_idx, (orig_sig_idx, row) in enumerate(member_genes_sorted.iterrows()):
                    gene_profile = norm_profiles[:, orig_sig_idx].tolist()
                    gene_records.append({
                        'gene': row['gene'],
                        'f_stat': row['f_stat'],
                        'pval': row['pval'],
                        'fdr': row['fdr'],
                        'r_squared': row['r_squared'],
                        'amplitude': row['amplitude'],
                        'direction': row['direction'],
                        'profile': gene_profile,
                        'peak_position': float(row['peak_position']),
                    })

                modules.append({
                    'module_id': mod_idx,
                    'pattern': pattern,
                    'n_genes': int(member_mask.sum()),
                    'representative_profile': rep_profile.tolist(),
                    'profile_positions': profile_positions.tolist(),
                    'genes': gene_records,
                })

            pattern_order = {'increasing': 0, 'decreasing': 1, 'peak': 2, 'trough': 3, 'complex': 4}
            modules.sort(key=lambda m: (pattern_order.get(m['pattern'], 5), -m['n_genes']))

        # Diagnostics
        n_pval_below_05 = int((p_values < 0.05).sum())
        n_pval_below_01 = int((p_values < 0.01).sum())
        expr_min = float(X.min())
        expr_max = float(X.max())
        expr_mean = float(X.mean())
        n_zero_genes = int((X.sum(axis=0) == 0).sum())
        pos_min = float(pos.min())
        pos_max = float(pos.max())
        pos_std = float(pos.std())

        return {
            'positive': positive_genes,
            'negative': negative_genes,
            'modules': modules,
            'n_cells': n_cells_used,
            'n_significant': int(n_significant),
            'n_positive': int(pos_mask.sum()),
            'n_negative': int(neg_mask.sum()),
            'n_modules': len(modules),
            'fdr_threshold': fdr_threshold,
            'diagnostics': {
                'n_genes_tested': n_genes,
                'n_pval_below_05': n_pval_below_05,
                'n_pval_below_01': n_pval_below_01,
                'position_range': [pos_min, pos_max],
                'position_std': pos_std,
                'expression_range': [expr_min, expr_max],
                'expression_mean': expr_mean,
                'n_zero_genes': n_zero_genes,
                'spline_df': k,
            },
        }
```

- [ ] **Step 2: Refactor `test_line_association` to use the helper**

Replace the body of `test_line_association` (lines 1263-1609) with this shorter version that handles line lookup, cell projection, and delegates to the helper:

```python
        from scipy.interpolate import BSpline
        from scipy.stats import f as f_dist
        from statsmodels.stats.multitest import multipletests

        # Find the line
        line = None
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line = l
                break

        if line is None:
            raise ValueError(f"Line '{line_name}' not found")

        embedding_name = line.get('embeddingName', '')
        if embedding_name not in self.adata.obsm:
            raise ValueError(f"Embedding '{embedding_name}' not found")

        line_points = line.get('smoothedPoints') or line.get('points', [])
        if len(line_points) < 2:
            raise ValueError("Line must have at least 2 points")

        coords = self.adata.obsm[embedding_name][:, :2]
        positions, distances = self._project_cells_onto_line(line_points, coords)

        if cell_indices is not None:
            cell_mask = np.zeros(self.n_cells, dtype=bool)
            cell_mask[cell_indices] = True
        else:
            cell_mask = np.ones(self.n_cells, dtype=bool)

        selected_indices = np.where(cell_mask)[0]
        selected_positions = positions[cell_mask]
        selected_distances = distances[cell_mask]
        n_cells_used = len(selected_indices)

        if n_cells_used < min_cells:
            raise ValueError(
                f"Too few cells ({n_cells_used}). Need at least {min_cells}."
            )

        if test_variable == 'distance':
            d_min = selected_distances.min()
            d_max = selected_distances.max()
            if d_max - d_min < 1e-10:
                raise ValueError(
                    "All cells have the same distance from the line. "
                    "Cannot test distance association."
                )
            test_values = (selected_distances - d_min) / (d_max - d_min)
        else:
            test_values = selected_positions

        if gene_subset is not None:
            gene_mask, _, _ = self._resolve_gene_mask(gene_subset)
        else:
            gene_mask = np.ones(self.n_genes, dtype=bool)

        result = self._run_spline_association(
            test_values=test_values,
            cell_indices=selected_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
        )

        result['line_name'] = line_name
        result['test_variable'] = test_variable
        return result
```

- [ ] **Step 3: Verify the refactored single-line method works**

Run: `cd xcell/backend && python -c "from xcell.adaptor import DataAdaptor; print('Import OK')"`
Expected: `Import OK`

Then start the backend and manually test "Find Associated Genes" on a single line in the browser to confirm identical behavior.

- [ ] **Step 4: Commit**

```bash
git add xcell/backend/xcell/adaptor.py
git commit -m "refactor: extract _run_spline_association helper from test_line_association"
```

---

### Task 2: Add Multi-Line Association Method (Backend)

**Files:**
- Modify: `xcell/backend/xcell/adaptor.py` (add method after `test_line_association`)

- [ ] **Step 1: Add `test_multi_line_association` method**

Insert this method immediately after `test_line_association` (after the closing of its body, before `_classify_profile_pattern`):

```python
    def test_multi_line_association(
        self,
        lines: list[dict[str, Any]],
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Test genes for association by pooling cells across multiple lines.

        Each line represents the same biological axis in a different tissue
        section or replicate. Cells are projected onto their respective line,
        positions are normalized to [0,1] (optionally reversed), then pooled
        for a single spline regression.

        Args:
            lines: List of dicts, each with:
                - name (str): Line name (must exist in _drawn_lines)
                - cell_indices (list[int]): Cell indices associated with this line
                - reversed (bool): If True, flip positions (1 - pos) for this line
            gene_subset: Optional gene filter (same as test_line_association).
            test_variable: 'position' or 'distance'.
            n_spline_knots: Number of interior knots for the B-spline basis.
            min_cells: Minimum pooled cells required.
            fdr_threshold: FDR threshold for significance.
            top_n: Maximum genes per module.

        Returns:
            Same dict shape as test_line_association, plus:
            - n_lines: Number of lines used
            - lines_used: List of line names
        """
        if not lines:
            raise ValueError("No lines provided")

        all_test_values = []
        all_cell_indices = []
        lines_used = []

        for entry in lines:
            line_name = entry['name']
            entry_cell_indices = entry['cell_indices']
            reversed_dir = entry.get('reversed', False)

            # Find the line
            line = None
            for l in self._drawn_lines:
                if l.get('name') == line_name:
                    line = l
                    break

            if line is None:
                raise ValueError(f"Line '{line_name}' not found")

            embedding_name = line.get('embeddingName', '')
            if embedding_name not in self.adata.obsm:
                raise ValueError(f"Embedding '{embedding_name}' not found")

            line_points = line.get('smoothedPoints') or line.get('points', [])
            if len(line_points) < 2:
                raise ValueError(f"Line '{line_name}' must have at least 2 points")

            coords = self.adata.obsm[embedding_name][:, :2]
            positions, distances = self._project_cells_onto_line(line_points, coords)

            # Select only this line's cells
            idx_array = np.array(entry_cell_indices, dtype=int)
            line_positions = positions[idx_array]
            line_distances = distances[idx_array]

            if reversed_dir:
                line_positions = 1.0 - line_positions

            if test_variable == 'distance':
                d_min = line_distances.min()
                d_max = line_distances.max()
                if d_max - d_min < 1e-10:
                    # All same distance — use zeros, pooling may still work
                    line_test_values = np.zeros(len(idx_array))
                else:
                    line_test_values = (line_distances - d_min) / (d_max - d_min)
            else:
                line_test_values = line_positions

            all_test_values.append(line_test_values)
            all_cell_indices.append(idx_array)
            lines_used.append(line_name)

        pooled_test_values = np.concatenate(all_test_values)
        pooled_cell_indices = np.concatenate(all_cell_indices)
        n_pooled = len(pooled_cell_indices)

        if n_pooled < min_cells:
            raise ValueError(
                f"Too few pooled cells ({n_pooled}). Need at least {min_cells}."
            )

        # Resolve gene subset
        if gene_subset is not None:
            gene_mask, _, _ = self._resolve_gene_mask(gene_subset)
        else:
            gene_mask = np.ones(self.n_genes, dtype=bool)

        result = self._run_spline_association(
            test_values=pooled_test_values,
            cell_indices=pooled_cell_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
        )

        result['line_name'] = ' + '.join(lines_used)
        result['test_variable'] = test_variable
        result['n_lines'] = len(lines_used)
        result['lines_used'] = lines_used
        return result
```

- [ ] **Step 2: Verify import still works**

Run: `cd xcell/backend && python -c "from xcell.adaptor import DataAdaptor; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add xcell/backend/xcell/adaptor.py
git commit -m "feat: add test_multi_line_association for pooled multi-line analysis"
```

---

### Task 3: Add Multi-Line API Endpoint (Backend)

**Files:**
- Modify: `xcell/backend/xcell/api/routes.py:978-1077`

- [ ] **Step 1: Add response model fields and new request/endpoint**

Add `n_lines` and `lines_used` to `LineAssociationResponse`, then add new models and endpoint. Insert the new response fields in the existing `LineAssociationResponse` class (around line 1035), and the new models + endpoint after the existing `test_line_association` route (after line 1077):

First, modify `LineAssociationResponse` — add two fields after `fdr_threshold`:

```python
class LineAssociationResponse(BaseModel):
    """Response model for line association testing."""
    positive: list[LineAssociationGene]
    negative: list[LineAssociationGene]
    modules: list[LineAssociationModule] = []
    n_cells: int
    n_significant: int
    n_positive: int
    n_negative: int
    n_modules: int = 0
    line_name: str
    test_variable: str = 'position'
    fdr_threshold: float
    n_lines: int = 1
    lines_used: list[str] = []
    diagnostics: LineAssociationDiagnostics | None = None
```

Then, after the existing `test_line_association` route function (after line 1077), add:

```python
class MultiLineEntry(BaseModel):
    """A single line entry for multi-line association."""
    name: str
    cell_indices: list[int]
    reversed: bool = False


class MultiLineAssociationRequest(BaseModel):
    """Request model for multi-line association testing."""
    lines: list[MultiLineEntry]
    gene_subset: str | list[str] | None = None
    test_variable: str = 'position'
    n_spline_knots: int = 5
    min_cells: int = 20
    fdr_threshold: float = 0.05
    top_n: int = 50


@router.post("/lines/multi-association", response_model=LineAssociationResponse)
def test_multi_line_association(request: MultiLineAssociationRequest, dataset: str | None = Query(None)):
    """Test genes for association by pooling cells across multiple lines.

    Each line represents the same biological axis in a different section/replicate.
    Cells are projected onto their respective line, positions normalized and
    optionally reversed, then pooled for a single spline regression.
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.test_multi_line_association(
            lines=[{
                'name': entry.name,
                'cell_indices': entry.cell_indices,
                'reversed': entry.reversed,
            } for entry in request.lines],
            gene_subset=request.gene_subset,
            test_variable=request.test_variable,
            n_spline_knots=request.n_spline_knots,
            min_cells=request.min_cells,
            fdr_threshold=request.fdr_threshold,
            top_n=request.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 2: Verify backend starts**

Run: `cd xcell/backend && python -c "from xcell.api.routes import router; print('Routes OK')"`
Expected: `Routes OK`

- [ ] **Step 3: Commit**

```bash
git add xcell/backend/xcell/api/routes.py
git commit -m "feat: add POST /lines/multi-association endpoint"
```

---

### Task 4: Extend Store Types and Add Multi-Line API Function (Frontend)

**Files:**
- Modify: `xcell/frontend/src/store.ts:209-223`
- Modify: `xcell/frontend/src/hooks/useData.ts:787-870`

- [ ] **Step 1: Add optional fields to `LineAssociationResult` in `store.ts`**

In `xcell/frontend/src/store.ts`, add `n_lines` and `lines_used` to the `LineAssociationResult` interface (after the `diagnostics` field, around line 222):

```typescript
export interface LineAssociationResult {
  positive: LineAssociationGene[]
  negative: LineAssociationGene[]
  modules: LineAssociationModule[]
  n_cells: number
  n_significant: number
  n_positive: number
  n_negative: number
  n_modules: number
  line_name: string
  test_variable: string
  fdr_threshold: number
  n_lines?: number
  lines_used?: string[]
  diagnostics?: LineAssociationDiagnostics
}
```

- [ ] **Step 2: Add `MultiLineEntry` and `MultiLineAssociationParams` interfaces and `runMultiLineAssociation` function in `useData.ts`**

In `xcell/frontend/src/hooks/useData.ts`, add the following after the existing `runLineAssociation` function (after line 829):

```typescript
// Multi-line association types and API function
export interface MultiLineEntry {
  name: string
  cellIndices: number[]
  reversed: boolean
}

export interface MultiLineAssociationParams {
  lines: MultiLineEntry[]
  geneSubset?: string | string[] | { columns: string[]; operation: string } | null
  testVariable?: 'position' | 'distance'
  nSplineKnots?: number
  minCells?: number
  fdrThreshold?: number
  topN?: number
}

export async function runMultiLineAssociation(params: MultiLineAssociationParams, slot?: DatasetSlot): Promise<LineAssociationResult> {
  return fetchJson<LineAssociationResult>(appendDataset(`${API_BASE}/lines/multi-association`, slot), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      lines: params.lines.map((l) => ({
        name: l.name,
        cell_indices: l.cellIndices,
        reversed: l.reversed,
      })),
      gene_subset: params.geneSubset ?? null,
      test_variable: params.testVariable ?? 'position',
      n_spline_knots: params.nSplineKnots ?? 5,
      min_cells: params.minCells ?? 20,
      fdr_threshold: params.fdrThreshold ?? 0.05,
      top_n: params.topN ?? 50,
    }),
  })
}
```

- [ ] **Step 3: Add `runMultiAssociation` to the `useLineAssociation` hook**

In the same file, extend the `useLineAssociation` hook (around line 832) to include a `runMultiAssociation` method. Add this after the existing `runAssociation` callback definition and before the return statement:

```typescript
  const runMultiAssociation = useCallback(async (params: MultiLineAssociationParams) => {
    setLineAssociationLoading(true)
    try {
      await syncLinesToBackend(drawnLines)
      const result = await runMultiLineAssociation(params)
      setLineAssociationResult(result)
      setLineAssociationModalOpen(true)
      return result
    } catch (err) {
      setLineAssociationResult(null)
      throw err
    } finally {
      setLineAssociationLoading(false)
    }
  }, [drawnLines, setLineAssociationLoading, setLineAssociationResult, setLineAssociationModalOpen])
```

Update the return statement to include `runMultiAssociation`:

```typescript
  return {
    drawnLines,
    lineAssociationResult,
    isLineAssociationLoading,
    runAssociation,
    runMultiAssociation,
  }
```

- [ ] **Step 4: Verify build**

Run: `cd xcell/frontend && npm run build`
Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add xcell/frontend/src/store.ts xcell/frontend/src/hooks/useData.ts
git commit -m "feat: add multi-line association types and API function"
```

---

### Task 5: Add Line Selection Checkboxes to Lines Panel

**Files:**
- Modify: `xcell/frontend/src/components/ShapeManager.tsx:611-750` (the `ShapeManager` component)

- [ ] **Step 1: Add checked state and multi-line modal trigger**

In the `ShapeManager` component (starts at line 613), add local state for checked lines and a state to trigger multi-line modal. Add these after the existing state declarations (after line 629):

```typescript
  const [checkedLineIds, setCheckedLineIds] = useState<Set<string>>(new Set())
  const [multiLineModalOpen, setMultiLineModalOpen] = useState(false)
```

Add a derived value for lines with projections and total cell count (after `currentEmbeddingLines`):

```typescript
  const linesWithProjections = currentEmbeddingLines.filter((l) => l.projections.length > 0)
  const checkedLines = currentEmbeddingLines.filter((l) => checkedLineIds.has(l.id))
  const checkedTotalCells = checkedLines.reduce((sum, l) => sum + l.projections.length, 0)
```

Add a toggle handler:

```typescript
  const handleToggleCheck = useCallback((lineId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setCheckedLineIds((prev) => {
      const next = new Set(prev)
      if (next.has(lineId)) {
        next.delete(lineId)
      } else {
        next.add(lineId)
      }
      return next
    })
  }, [])
```

- [ ] **Step 2: Add checkbox to line rows**

In the line row JSX (the `currentEmbeddingLines.map(...)` block, around line 684), add a checkbox as the first element inside the row `<div>`, before the line name. Only render it for lines with projections:

```tsx
{line.projections.length > 0 && (
  <input
    type="checkbox"
    checked={checkedLineIds.has(line.id)}
    onChange={() => {/* handled by onClick */}}
    onClick={(e) => handleToggleCheck(line.id, e)}
    style={{ marginRight: '4px', cursor: 'pointer', flexShrink: 0 }}
    title={`Include in multi-line analysis (${line.projections.length} cells)`}
  />
)}
```

- [ ] **Step 3: Add action bar below line list**

After the line list `</div>` (after the `currentEmbeddingLines.map` block ends, around line 780), but still inside the `!collapsed &&` conditional and before the closing `)}` of the content div, add the action bar:

```tsx
{checkedLineIds.size > 0 && checkedLines.length > 0 && (
  <div style={{
    padding: '8px 16px',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    fontSize: '11px',
  }}>
    <span style={{ color: '#888' }}>
      {checkedLines.length} line{checkedLines.length !== 1 ? 's' : ''} ({checkedTotalCells.toLocaleString()} cells)
    </span>
    <button
      style={{
        padding: '4px 10px',
        fontSize: '11px',
        backgroundColor: '#4ecdc4',
        color: '#000',
        border: '1px solid #4ecdc4',
        borderRadius: '4px',
        cursor: 'pointer',
        fontWeight: 500,
      }}
      onClick={() => setMultiLineModalOpen(true)}
    >
      Find Associated Genes
    </button>
  </div>
)}
```

- [ ] **Step 4: Render multi-line modal**

In the fragment at the bottom of the return (where `{toolsLine && <LineToolsModal .../>}` is rendered), add the multi-line modal render:

```tsx
{multiLineModalOpen && checkedLines.length > 0 && (
  <MultiLineToolsModal
    lines={checkedLines}
    onClose={() => setMultiLineModalOpen(false)}
  />
)}
```

(The `MultiLineToolsModal` component will be created in the next task.)

- [ ] **Step 5: Verify build**

Run: `cd xcell/frontend && npm run build`
Expected: Build fails because `MultiLineToolsModal` doesn't exist yet. That's expected — we'll add it in Task 6. To verify the rest compiles, temporarily comment out the `MultiLineToolsModal` render, build, then uncomment.

- [ ] **Step 6: Commit**

```bash
git add xcell/frontend/src/components/ShapeManager.tsx
git commit -m "feat: add line selection checkboxes and action bar to Lines panel"
```

---

### Task 6: Create Multi-Line Tools Modal

**Files:**
- Modify: `xcell/frontend/src/components/ShapeManager.tsx` (add `MultiLineToolsModal` component)

- [ ] **Step 1: Add `MultiLineToolsModal` component**

Add this new component before the `ShapeManager` component (after the existing `LineToolsModal`, around line 609). It reuses the same styles object already defined at the top of the file.

```tsx
function MultiLineToolsModal({
  lines,
  onClose,
}: {
  lines: ReturnType<typeof useStore.getState>['drawnLines']
  onClose: () => void
}) {
  const { scanpyActionHistory } = useStore()
  const { runMultiAssociation, isLineAssociationLoading } = useLineAssociation()

  const [reversals, setReversals] = useState<Record<string, boolean>>({})
  const [associationError, setAssociationError] = useState<string | null>(null)
  const [geneSubsetColumns, setGeneSubsetColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')
  const [testVariable, setTestVariable] = useState<'position' | 'distance'>('position')
  const [nSplineKnots, setNSplineKnots] = useState(5)
  const [fdrThreshold, setFdrThreshold] = useState(0.05)
  const [topN, setTopN] = useState(50)

  const totalCells = lines.reduce((sum, l) => sum + l.projections.length, 0)

  useEffect(() => {
    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((res) => res.json())
      .then(setGeneSubsetColumns)
      .catch(() => setGeneSubsetColumns([]))
  }, [scanpyActionHistory])

  const toggleReversal = useCallback((lineId: string) => {
    setReversals((prev) => ({ ...prev, [lineId]: !prev[lineId] }))
  }, [])

  const handleRun = useCallback(async () => {
    setAssociationError(null)
    try {
      let geneSubset: string | { columns: string[]; operation: string } | null = null
      if (selectedGeneColumns.length === 1) {
        geneSubset = selectedGeneColumns[0]
      } else if (selectedGeneColumns.length > 1) {
        geneSubset = { columns: selectedGeneColumns, operation: geneSubsetOperation }
      }

      await runMultiAssociation({
        lines: lines.map((l) => ({
          name: l.name,
          cellIndices: l.projections.map((p) => p.cellIndex),
          reversed: !!reversals[l.id],
        })),
        geneSubset,
        testVariable,
        nSplineKnots,
        fdrThreshold,
        topN,
      })
      onClose()
    } catch (err) {
      setAssociationError((err as Error).message)
    }
  }, [runMultiAssociation, lines, reversals, selectedGeneColumns, geneSubsetOperation, testVariable, nSplineKnots, fdrThreshold, topN, onClose])

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.modalHeader}>
          <h2 style={styles.modalTitle}>
            Line Association: {lines.length} lines ({totalCells.toLocaleString()} cells)
          </h2>
          <button style={styles.modalClose} onClick={onClose}>&times;</button>
        </div>

        <div style={styles.modalContent}>
          {/* Line list with direction toggles */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Lines</div>
            {lines.map((line) => (
              <div key={line.id} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 0',
                fontSize: '12px',
                color: '#ccc',
                borderBottom: '1px solid #0a0f1a',
              }}>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {line.name}
                </span>
                <span style={{ fontSize: '11px', color: '#888' }}>
                  {line.projections.length} cells
                </span>
                <button
                  onClick={() => toggleReversal(line.id)}
                  style={{
                    padding: '2px 8px',
                    fontSize: '11px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    border: '1px solid #1a1a2e',
                    backgroundColor: reversals[line.id] ? '#e94560' : '#0f3460',
                    color: reversals[line.id] ? '#fff' : '#aaa',
                    fontWeight: reversals[line.id] ? 600 : 400,
                  }}
                  title={reversals[line.id] ? 'Direction: reversed (click to reset)' : 'Direction: as drawn (click to reverse)'}
                >
                  {reversals[line.id] ? '\u2190' : '\u2192'}
                </button>
              </div>
            ))}
          </div>

          {/* Gene subset selector (same as single-line mode) */}
          {geneSubsetColumns.length > 0 && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>
                Genes {selectedGeneColumns.length === 0 ? '(all)' : ''}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {geneSubsetColumns.map((col) => {
                  const isSelected = selectedGeneColumns.includes(col.name)
                  return (
                    <button
                      key={col.name}
                      onClick={() => {
                        if (isSelected) {
                          setSelectedGeneColumns(selectedGeneColumns.filter((c) => c !== col.name))
                        } else {
                          setSelectedGeneColumns([...selectedGeneColumns, col.name])
                        }
                      }}
                      style={{
                        ...styles.pillButton,
                        backgroundColor: isSelected ? '#4ecdc4' : '#0f3460',
                        color: isSelected ? '#000' : '#aaa',
                        borderColor: isSelected ? '#4ecdc4' : '#1a1a2e',
                        fontWeight: isSelected ? 600 : 400,
                      }}
                      title={`${col.n_true.toLocaleString()} of ${col.n_total.toLocaleString()} genes`}
                    >
                      {col.name} ({col.n_true.toLocaleString()})
                    </button>
                  )
                })}
              </div>
              {selectedGeneColumns.length >= 2 && (
                <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
                  <span style={{ color: '#888' }}>Combine:</span>
                  <button
                    onClick={() => setGeneSubsetOperation('intersection')}
                    style={{
                      ...styles.toggleButton,
                      backgroundColor: geneSubsetOperation === 'intersection' ? '#4ecdc4' : '#0f3460',
                      color: geneSubsetOperation === 'intersection' ? '#000' : '#aaa',
                    }}
                  >
                    AND
                  </button>
                  <button
                    onClick={() => setGeneSubsetOperation('union')}
                    style={{
                      ...styles.toggleButton,
                      backgroundColor: geneSubsetOperation === 'union' ? '#4ecdc4' : '#0f3460',
                      color: geneSubsetOperation === 'union' ? '#000' : '#aaa',
                    }}
                  >
                    OR
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Test variable */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Test against</div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                onClick={() => setTestVariable('position')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: testVariable === 'position' ? '#4ecdc4' : '#0f3460',
                  color: testVariable === 'position' ? '#000' : '#aaa',
                  borderColor: testVariable === 'position' ? '#4ecdc4' : '#1a1a2e',
                  fontWeight: testVariable === 'position' ? 600 : 400,
                }}
              >
                Position along line
              </button>
              <button
                onClick={() => setTestVariable('distance')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: testVariable === 'distance' ? '#4ecdc4' : '#0f3460',
                  color: testVariable === 'distance' ? '#000' : '#aaa',
                  borderColor: testVariable === 'distance' ? '#4ecdc4' : '#1a1a2e',
                  fontWeight: testVariable === 'distance' ? 600 : 400,
                }}
              >
                Distance from line
              </button>
            </div>
          </div>

          {/* Parameters */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Parameters</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', fontSize: '11px', color: '#ccc' }}>
              <span>Spline knots</span>
              <input
                type="number"
                min="2"
                max="20"
                value={nSplineKnots}
                onChange={(e) => setNSplineKnots(Math.max(2, Math.min(20, parseInt(e.target.value) || 5)))}
                style={styles.smallInput}
              />
              <span>FDR</span>
              <input
                type="number"
                min="0.001"
                max="0.5"
                step="0.01"
                value={fdrThreshold}
                onChange={(e) => setFdrThreshold(Math.max(0.001, Math.min(0.5, parseFloat(e.target.value) || 0.05)))}
                style={{ ...styles.smallInput, width: '56px' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#ccc' }}>
              <span>Max genes/module</span>
              <input
                type="number"
                min="10"
                max="500"
                step="10"
                value={topN}
                onChange={(e) => setTopN(Math.max(10, Math.min(500, parseInt(e.target.value) || 50)))}
                style={styles.smallInput}
              />
            </div>
          </div>

          {/* Run button */}
          <button
            style={{
              ...styles.primaryActionButton,
              opacity: isLineAssociationLoading ? 0.6 : 1,
            }}
            onClick={handleRun}
            disabled={isLineAssociationLoading}
          >
            {isLineAssociationLoading ? 'Analyzing...' : 'Find Associated Genes'}
          </button>
          {associationError && (
            <div style={{ marginTop: '6px', fontSize: '11px', color: '#e94560' }}>
              {associationError}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Update imports at top of ShapeManager.tsx**

The file already imports `useLineAssociation` and `appendDataset` from `useData.ts`. Add the `MultiLineAssociationParams` import if the `runMultiAssociation` method is accessed through the hook (which it is — no additional import needed since it's returned from the hook).

Verify the import line at the top of the file includes what's needed:

```typescript
import { useLineAssociation, createLineEmbedding, appendDataset } from '../hooks/useData'
```

This is already sufficient — `runMultiAssociation` comes from the `useLineAssociation()` hook return value.

- [ ] **Step 3: Verify build**

Run: `cd xcell/frontend && npm run build`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add xcell/frontend/src/components/ShapeManager.tsx
git commit -m "feat: add MultiLineToolsModal for combined line association analysis"
```

---

### Task 7: Update Results Modal for Multi-Line Display

**Files:**
- Modify: `xcell/frontend/src/components/LineAssociationModal.tsx:633-750`

- [ ] **Step 1: Update header to show combined line names**

In the results modal component, find the header title (the `<h2>` element around line 715). Replace the current title logic:

Find:
```tsx
          <h2 style={styles.title}>
            Line Association: {line_name}
            {test_variable === 'distance' && (
              <span style={{ fontSize: '12px', color: '#888', fontWeight: 400, marginLeft: '8px' }}>
                (distance from line)
              </span>
            )}
          </h2>
```

Replace with:
```tsx
          <h2 style={styles.title}>
            Line Association: {lines_used && lines_used.length > 1 ? lines_used.join(' + ') : line_name}
            {test_variable === 'distance' && (
              <span style={{ fontSize: '12px', color: '#888', fontWeight: 400, marginLeft: '8px' }}>
                (distance from line)
              </span>
            )}
          </h2>
```

- [ ] **Step 2: Add destructuring for new fields and add Lines count to summary**

Update the destructuring line to include the new fields. Find:
```typescript
  const { n_cells, n_significant, line_name, test_variable, fdr_threshold, diagnostics, modules } = lineAssociationResult
```

Replace with:
```typescript
  const { n_cells, n_significant, line_name, test_variable, fdr_threshold, diagnostics, modules, n_lines, lines_used } = lineAssociationResult
```

In the summary stats section (the `<div style={styles.summary}>` block), add a "Lines" item when `n_lines > 1`. Add this after the "Cells Tested" summary item:

```tsx
            {n_lines != null && n_lines > 1 && (
              <div style={styles.summaryItem}>
                <div style={styles.summaryLabel}>Lines</div>
                <div style={styles.summaryValue}>{n_lines}</div>
              </div>
            )}
```

- [ ] **Step 3: Verify build**

Run: `cd xcell/frontend && npm run build`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add xcell/frontend/src/components/LineAssociationModal.tsx
git commit -m "feat: show multi-line info in association results modal"
```

---

### Task 8: Cell Panel Selection Highlight

**Files:**
- Modify: `xcell/frontend/src/components/CellPanel.tsx`

- [ ] **Step 1: Add local state for tracking category selection source**

In the main `CellPanel` component (find where `handleSelectCellsByCategory` is defined, around line 799), add local state to track which category was clicked. Add this near the other state declarations at the top of the component:

```typescript
  const [selectedCategorySource, setSelectedCategorySource] = useState<{ column: string; value: string } | null>(null)
```

- [ ] **Step 2: Set the source when a category is clicked**

In `handleSelectCellsByCategory`, after `setSelectedCellIndices(indices)` (line 822), add:

```typescript
          setSelectedCategorySource({ column: columnName, value: categoryValue })
```

- [ ] **Step 3: Clear the source when selection changes from other sources**

The `selectedCellIndices` can change from lasso selection or other sources. We need to clear the highlight when that happens. Add a `useEffect` that watches `selectedCellIndices` — but only clears if the change didn't come from our category click. The simplest approach: clear `selectedCategorySource` whenever `selectedCellIndices` becomes empty. Also pass the source down so the `CategoryColumn` component can apply it.

Add this effect near the other effects in CellPanel:

```typescript
  useEffect(() => {
    if (selectedCellIndices.length === 0) {
      setSelectedCategorySource(null)
    }
  }, [selectedCellIndices])
```

- [ ] **Step 4: Pass selection source to CategoryColumn and apply highlight**

Pass `selectedCategorySource` to the `CategoryColumn` component. Update the `CategoryColumnProps` interface to add:

```typescript
  selectedCategorySource: { column: string; value: string } | null
```

Update the `CategoryColumn` function signature to include the new prop.

In the category value rows (around line 548), apply a highlight when the category matches the selection source. Find the `<div key={cat.value} style={styles.categoryItem}>` line and conditionally add a left border:

```tsx
<div key={cat.value} style={{
  ...styles.categoryItem,
  ...(selectedCategorySource?.column === summary.name && selectedCategorySource?.value === cat.value
    ? { borderLeft: '3px solid #4ecdc4', paddingLeft: '5px' }
    : {}),
}}>
```

Where `summary.name` is the column name available from the parent scope — the `CategoryColumn` component receives `summary` as a prop which has `summary.name`.

Update all call sites of `<CategoryColumn>` (there should be one, around line 983) to pass the new prop:

```tsx
selectedCategorySource={selectedCategorySource}
```

- [ ] **Step 5: Verify build**

Run: `cd xcell/frontend && npm run build`
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add xcell/frontend/src/components/CellPanel.tsx
git commit -m "feat: highlight category value row when cells are selected from it"
```

---

### Task 9: Update Documentation

**Files:**
- Modify: `xcell/CLAUDE.md`
- Modify: `xcell/CHANGELOG.md`
- Modify: `xcell/README.md`

- [ ] **Step 1: Update CLAUDE.md**

Add the new API endpoint to the API table:

In the Lines group row, add `POST /lines/multi-association` alongside the existing entries.

Update the DataAdaptor Key Methods table — add `test_multi_line_association(lines, ...)` and `_run_spline_association(test_values, cell_indices, gene_mask, ...)` to the Lines/trajectory group.

Update the Key Behaviors section — add a bullet about multi-line association:

```
- **Multi-line association**: When 2+ lines have projected cells, checkboxes appear on those line rows in the Lines panel. Checking lines reveals an action bar with "Find Associated Genes". This opens a multi-line Line Tools modal with per-line direction toggles, gene subset, test variable, and spline parameters. Backend pools cells across lines (normalizing positions per-line, respecting reversals) and runs a single spline regression. Results display in the same LineAssociationModal with combined line names in the header.
```

- [ ] **Step 2: Update CHANGELOG.md**

Add under `## [Unreleased]` → `### Added`:

```markdown
- Multi-line combined association analysis: check multiple lines with projected cells in the Lines panel, then run a pooled "Find Associated Genes" across tissue sections or replicates. Per-line direction reversal ensures consistent biological axis alignment. Results appear in the same association modal.
- Cell Panel: clicking a category value to select cells now highlights that row with a teal left border, making the selection source visible.
```

- [ ] **Step 3: Update README.md**

Update the Trajectory Analysis section (section 10) to mention multi-line:

```markdown
### 10. Trajectory Analysis

- Draw lines on the scatter plot
- Click the gear icon on a line in the **Lines** panel to open **Line Tools**
- Under **Gene Association**, configure:
  - **Test against**: position along line or distance from line
  - **Gene subset**: filter to highly variable genes or other boolean columns
  - **Spline knots**: number of interior knots for the B-spline model (default 5; higher = more flexible fit)
  - **FDR**: significance threshold (default 0.05)
  - **Max genes/module**: cap on genes returned per expression module
- Click **Find Associated Genes** to run the analysis
- In the results modal, use the **Filters** bar to refine results interactively: adjust min R², min amplitude, max FDR, or toggle pattern types (increasing, decreasing, peak, trough, complex)

#### Multi-section / replicate analysis

- Draw a line on each tissue section representing the same biological axis
- For each line, select cells (via lasso or clicking a category value in the **Cells** panel) and click **+** to associate them with the line
- Check the lines to include using the checkboxes that appear on lines with projected cells
- Click **Find Associated Genes** in the action bar
- In the multi-line modal, toggle direction per line if needed (arrow button) and set analysis parameters
- Results pool cells across all lines for a single, higher-powered analysis
```

- [ ] **Step 4: Commit**

```bash
git add xcell/CLAUDE.md xcell/CHANGELOG.md xcell/README.md
git commit -m "docs: document multi-line association and cell panel selection highlight"
```

---

### Task 10: Final Verification

- [ ] **Step 1: Full build check**

Run: `cd xcell/frontend && npm run build`
Expected: Build succeeds with no errors.

- [ ] **Step 2: Backend import check**

Run: `cd xcell/backend && python -c "from xcell.adaptor import DataAdaptor; from xcell.api.routes import router; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Manual browser testing**

Start backend and frontend:
```bash
cd xcell/backend && uvicorn xcell.main:app --reload &
cd xcell/frontend && npm run dev &
```

Test the following:
1. Single-line "Find Associated Genes" still works (via gear icon → Line Tools)
2. Draw 2+ lines, project cells onto each, checkboxes appear
3. Check 2 lines → action bar appears with cell count
4. Click "Find Associated Genes" → multi-line modal opens
5. Toggle direction on one line, run analysis
6. Results modal shows combined line names and line count
7. In Cell Panel, click a category value → row highlights with teal border
8. Lasso select different cells → highlight clears
