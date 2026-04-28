# XCell

Interactive web application for exploring and analyzing scRNA-seq and spatial transcriptomics data. Load an h5ad, 10x Genomics h5, Seurat .rds file, 10x CellRanger matrix folder, or prefixed 10x file trio from GEO, visualize cells on a scatter plot, run Scanpy analysis pipelines, and explore results — all from your browser.

 ![Screenshot](docs/images/xcell_screenshot.jpg)   

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- R with Seurat and SeuratDisk packages (optional, required for loading `.rds` files)

### Backend Setup

```bash
cd xcell/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Frontend Setup

```bash
cd xcell/frontend
npm install
```

### Launch

```bash
# Terminal 1: Start the backend (from xcell/backend/)
uvicorn xcell.main:app --reload

# Terminal 2: Start the frontend (from xcell/frontend/)
npm run dev
```

Open http://localhost:5173 in your browser.

A bundled toy dataset (`toy_spatial.h5ad`) loads automatically if no data path is specified. To load your own data, set the `XCELL_DATA_PATH` environment variable:

```bash
XCELL_DATA_PATH=/path/to/your/data.h5ad uvicorn xcell.main:app --reload  # also supports .h5 and .rds
```

## Getting Started with Toy Data

The included `test_data/toy_spatial.h5ad` dataset is a small spatial transcriptomics dataset for exploring XCell's features. Here's a step-by-step walkthrough:

### 1. Explore the Scatter Plot

- Pan by clicking and dragging
- Zoom with scroll wheel
- Cells are rendered as points at their spatial coordinates

### 2. Color by Metadata

- Open **Cell Manager** (left panel)
- Select a metadata column to color cells by that annotation

### 3. Select Cells

- Click the **Select** button in the toolbar (use the dropdown arrow to choose between Lasso and Polygon tools)
  - **Lasso**: click and drag to draw a freehand selection
  - **Polygon**: click to add vertices, double-click to close and select cells inside
- Hold **Shift** while selecting to add to the existing selection
- Checkboxes in the Cell Manager also select/deselect cells by category
- **Rename a category label** by double-clicking the label in the expanded category list. Press Enter to commit (or Escape to cancel). Works on Leiden clusters, Contourize results, user annotations — any categorical metadata.
- **Merge two or more labels** by clicking the `⋯` menu in a column header and choosing **Merge labels…**. Pick the labels to merge, type a new name (or reuse an existing one to fold them in), then click Merge.
- Selected cells can be masked or deleted

### 4. Run Preprocessing

- Open the **Scanpy** modal (top toolbar)
- Go to **Preprocessing** and run in order:
  1. **Normalize Total** — normalize counts per cell
  2. **Log1p** — log-transform the data
  3. **Highly Variable Genes** — identify informative genes

### 5. Run Cell Analysis

- In the **Scanpy** modal, go to **Cell Analysis** and run in order:
  1. **PCA** — reduce dimensionality
  2. **PCA Loadings** (optional) — scan the top-loading genes on each side of every PC (hover a gene to see its exact loading). If you spot PCs dominated by technical signal (cell cycle, mitochondrial genes, etc.), check them and click **Create PC subset** to persist a derived embedding (e.g. `X_pca_noPC2_5`).
  3. **Neighbors** — build cell neighborhood graph (requires PCA). If you created derived subsets in step 2, pick one from the **PC source** dropdown — UMAP and Leiden inherit the choice automatically through the neighbors graph.
  4. **UMAP** — compute 2D embedding (requires Neighbors)
  5. **Leiden** — cluster cells (requires Neighbors)

  Re-running PCA clears all derived PC subsets (with a toast) since their column indices refer to the previous eigenvectors.

### 6. View Clustering Results

- In **Cell Manager**, select the `leiden` column to color by cluster
- Switch the embedding to `X_umap` to see the UMAP layout

### 7. Color by Gene Expression

- Open **Gene Manager** (right panel)
- If the dataset has alternative gene identifier columns (e.g., gene symbols alongside Ensembl IDs), use the **Gene IDs** dropdown at the top of the panel to switch
- Search or browse genes
- Click a gene to color cells by its expression

### Gene Mask

To scope the Gene Panel to a relevant gene universe, click the `⋯` button in the Genes panel header and choose `Gene mask…`. The modal lists all boolean columns in your dataset's `.var` (for example, `highly_variable` after running Highly Variable Genes, or `spatially_variable` after spatial autocorrelation). For each column, choose:

- **Off** — ignore this column
- **Keep** — include genes where this column is True
- **Hide** — exclude genes where this column is True

When you have multiple Keep columns, choose whether to match **ANY** (union) or **ALL** (intersection). Hide columns always combine as a union.

The mask applies to the gene browse list, gene search, expanded gene set rows, and gene set score aggregation used for display coloring. It does **not** apply to analysis operations (Diff Exp, Marker Genes, Gene PCA, etc.) — those have their own gene subset dropdowns. The mask is per-dataset and session-only; reloading the page clears it.

### 8. Gene Sets

- Create gene sets manually in Gene Manager
- Import gene lists from files

### Curating gene sets into folders

The Manual category at the top of the Gene Panel is the home for gene sets
you create by hand. Click `+ 📁` to create a named folder (e.g. "Fig 3 markers").
Inside a folder, click `+` to add a new empty set, or drag an existing
top-level set onto the folder row to move it in. Drag a set back onto the thin
strip above the first folder to move it out. Drag sets within the same container
to reorder them.

Each gene set and folder row has a `⋯` button with secondary actions.
On a gene set row, that's where you find Pin and Cluster genes.
On a manual folder row, that's where you find Pin and Export (JSON/GMT/CSV).

Use the `Pin/Unpin` option in the `⋯` menu on any set or folder to float it to
the top of its container. Pinning works in every category — including
auto-generated ones — and survives moving a set between folders.

The `Export ▸` option in the `⋯` menu on any manual folder lets you export just
that folder's gene sets to JSON, GMT, or CSV. Filename defaults to the sanitized
folder name. JSON round-trips via the existing Import modal.

Use the 👁 button on a category header to hide a whole category from view
(useful when an analysis has filled `Gene Clusters` or `Differential Expression`
with results you're done with). A `N hidden ▸` footer appears at the bottom of
the Gene Panel — click it and then Unhide to bring a category back.

Tip: double-click any gene set name or manual folder name to rename it inline.

### Sub-clustering a gene set

Any gene set with at least 4 genes can be sub-clustered by expression
pattern. Click the `⋯` button on a gene set row and choose `Cluster genes…`.
Pick a method (Hierarchical or K-means), a number of clusters K (default 3),
and a cell context ("All cells", "Current selection" if you've lasso-picked
some cells, or "Annotation category" to restrict to specific categorical
values in a `.obs` column). Clicking Run creates a new folder in
`Gene Clusters` named after the source set, containing one gene set per
cluster. Re-running with different K or a different cell context appends
another folder so you can compare runs side by side.

### Selecting cells by expression threshold

You can select cells based on a gene's expression or a gene set score without needing to eyeball the scatter plot:

1. In the Gene Panel, click the `⋯` menu on any gene row or gene set row and choose `Select cells…`.
2. The modal opens and the scatter plot switches to expression coloring for that source. An interactive histogram of the values is shown.
3. Pick a threshold mode (`Above`, `Below`, or `Between`) and drag the red cutoff line(s). The match counter updates live.
4. Choose an action:
   - **Update selection** replaces, adds to, or intersects with your current lasso selection.
   - **Label cells** creates a new annotation column with `high`/`low` labels for the cells in the chosen context (current selection or all cells). On success, click `Open Diff Exp ▸` to immediately run differential expression between the two groups.

Typical workflow for "find DEGs by expression state in a region": lasso a region → `⋯ → Select cells…` on a gene → drag the threshold → Label cells → Open Diff Exp.

### 9. Compare Cell Groups

- Open the **Analyze** modal (top toolbar) → **Cell Analysis** → **Compare Cells**
- Select an .obs column (e.g., `leiden`) from the dropdown
- Check 2 or more groups to compare:
  - **2 checked** → pairwise differential expression
  - **3+ checked** → one-vs-rest marker gene analysis
- Set **Top N genes** and click **Run**
- You can also use lasso selection: select cells → **Set as Group 1** / **Set as Group 2** → click **Compare** in the comparison bar

### 10. Trajectory Analysis

- Draw lines on the scatter plot
- Click the gear icon on a shape in the **Shapes** panel to open **Line Tools**
- Under **Gene Association**, configure:
  - **Test against**: position along line or distance from line
  - **Gene subset**: filter to highly variable genes or other boolean columns
  - **Spline knots**: number of interior knots for the B-spline model (default 5; higher = more flexible fit)
  - **FDR**: significance threshold (default 0.05)
  - **Max genes/direction** (or **/module** when clustering is on): cap on genes returned
  - **Cluster genes into modules** (default off): when checked, significant genes are grouped by expression profile shape (increasing, decreasing, peak, trough, complex); when unchecked, only positive/negative lists are returned
- Click **Find Associated Genes** to run the analysis
- In the results modal, use the **Filters** bar to refine results interactively: adjust min R², min amplitude, max FDR, or toggle pattern types (increasing, decreasing, peak, trough, complex)
- Click **Add to Gene Sets** in the results modal to save the genes — each run creates its own folder in the **Line Association** category of the Gene Panel (one set per module if clustering is on, or a single combined `Associated genes` set if clustering is off)
- Click **Download CSV** in the results modal to export stats (gene, f_stat, pval, fdr, r_squared, amplitude, direction) for every gene tested — a ranked-list suitable for GSEA or other external analyses

#### Multi-section / replicate analysis

- Draw a line on each tissue section representing the same biological axis
- For each line, select cells (via lasso or clicking a category value in the **Cells** panel) and click **+** to associate them with the line
- Check the lines to include using the checkboxes that appear on lines with projected cells
- Click **Find Associated Genes** in the action bar
- In the multi-line modal, toggle direction per line if needed (arrow button) and set analysis parameters
- Results pool cells across all lines for a single, higher-powered analysis

#### Combine neighbor graphs for spatially-aware clustering

- After computing both **Neighbors** (Cell Analysis) and **Spatial Neighbors** (Spatial Analysis), open **Analyze** → **Cell Analysis** → **Combine Neighbors**
- Select two or more graphs and set their weights (default: equal weights; weights are normalized to sum to 1)
- Click **Combine graphs** — the combined graph becomes the default `connectivities` slot
- Run **Leiden** (or **UMAP**) afterward and clustering/embedding will reflect both graphs, encouraging spatially neighboring cells to cluster together when the spatial graph is weighted in

### 11. Run Gene Analysis

- In the **Scanpy** modal, go to **Gene Analysis**:
  1. **Build Gene Graph** — compute gene-gene similarity
  2. **Cluster Genes** — group genes by expression pattern

### 12. Spatial Contouring

- Select genes in the **Gene Panel** (click individual genes or use a gene set)
- Open the **Scanpy** modal, go to **Spatial Analysis** > **Contourize**
- Adjust smoothing sigma, contour levels, and grid resolution as needed
- Click **Run** — a new categorical column appears in the Cell Panel
- Color cells by the contour column to visualize spatial expression zones

### 13. Load a Second Dataset

- Click **Load** in the toolbar — the modal shows a sidebar with quick-access locations (Home, Desktop, Documents, Downloads) and recently loaded files, plus breadcrumb path navigation for clicking any ancestor directory
- Choose **Secondary** from the "Load into" dropdown
- Browse or enter the path to a second h5ad, h5, rds file, 10x matrix folder, or prefixed 10x file trio and click **Load**
- A dataset switcher dropdown appears in the header — switch between Primary and Secondary to compare datasets
- Click the **Split** button to view both datasets side by side
- Click on either plot to make it the active dataset — the Cell and Gene panels update accordingly
- Each plot has its own embedding selector, legend, and independent pan/zoom

### 14. Export Results

- Click **Export** in the toolbar to download annotations and results

## Customizing default parameters

xcell ships with hardcoded defaults for every form in the Scanpy modal, the Line Association dialog, and the Display Settings panel (e.g. `filter_cells` → min genes = 25, point size = 3). To change these without touching code, drop a YAML (or JSON) file at **`~/.xcell/config.yaml`** — or set `XCELL_CONFIG_PATH` to point somewhere else. A sample is included at `docs/config.example.yaml`.

Shape is a nested mapping matching the form namespace — only include keys you want to override, everything else falls back to the built-in default:

```yaml
scanpy:
  filter_cells:
    min_genes: 15       # was 25
  neighbors:
    n_neighbors: 20     # was 15

line_association:
  fdr_threshold: 0.1    # was 0.05
  cluster_genes: true   # was false

display:
  point_size: 4               # was 3
  point_opacity: 0.7          # was 0.85
  background_color: '#000000' # was '#1a1a2e'
  color_scale: magma          # was viridis
  clip_percentile: 0.5        # was 1.0
  gene_set_aggregation: median # was mean
```

A backend restart is required to pick up edits. Verify what was loaded by hitting `GET /api/config/defaults`; unknown keys are silently ignored. Display defaults are applied to every dataset slot at startup and re-applied on each fresh dataset load — you can still tweak any value in the Display Settings panel for the current session.

## Session persistence

Most changes you make in a session survive on the backend process: deleted cells, transformed embeddings, computed PCA / neighbors / UMAP / Leiden, drawn lines, and — as of this version — your **gene sets** (categories, folders, individual sets). If the browser tab accidentally reloads, the gene panel is rehydrated from the server. Restarting the backend still clears everything; persist important sets via the Gene Panel export controls before shutting down.

## Features

- **Interactive scatter plot** — deck.gl-powered visualization with pan, zoom, lasso selection
- **Cell Manager** — browse/color by metadata, mask/delete cells
- **Gene Manager** — search genes, create gene sets, import gene lists
- **Scanpy integration** — run preprocessing, cell analysis (PCA, Neighbors, UMAP, Leiden), gene analysis, spatial analysis (contourize), and differential expression directly in the browser. Long-running operations (gene neighbors, spatial neighbors, spatial autocorrelation, contourize, line gene association) can be cancelled mid-run without corrupting session data.
- **Trajectory analysis** — draw lines and associate genes with spatial trajectories
- **Quilt mode** — lasso and rearrange tissue pieces: drag to translate, shift+drag to rotate, flip to reflect selected cell subsets
- **Display settings** — adjust point size, opacity, colormaps, bivariate coloring
- **Multi-dataset support** — load two datasets (h5ad, h5, rds, 10x matrix folders, or prefixed 10x file trios from GEO), switch between them, or view side by side in split mode
- **Export** — download annotations and analysis results

## Project Structure

```
xcell/
├── backend/
│   ├── xcell/
│   │   ├── main.py          # FastAPI app entry point
│   │   ├── adaptor.py       # DataAdaptor class (wraps AnnData)
│   │   ├── diffexp.py       # Differential expression
│   │   ├── data/
│   │   │   └── toy_spatial.h5ad  # Bundled toy dataset
│   │   └── api/
│   │       └── routes.py    # REST API endpoints
│   └── pyproject.toml       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main app component
│   │   ├── store.ts          # Zustand state management
│   │   ├── main.tsx          # Entry point
│   │   ├── components/
│   │   │   ├── ScatterPlot.tsx        # deck.gl scatter plot
│   │   │   ├── CellPanel.tsx          # Cell metadata manager
│   │   │   ├── GenePanel.tsx          # Gene browser / gene sets
│   │   │   ├── ScanpyModal.tsx        # Scanpy analysis pipeline UI
│   │   │   ├── DiffExpModal.tsx       # Differential expression
│   │   │   ├── LineAssociationModal.tsx # Trajectory analysis
│   │   │   ├── DisplaySettings.tsx    # Visualization settings
│   │   │   ├── ShapeManager.tsx       # Shape/selection tools
│   │   │   └── ImportModal.tsx        # Gene list import
│   │   └── hooks/
│   │       └── useData.ts    # Data fetching hooks
│   ├── package.json          # Node dependencies
│   └── vite.config.ts        # Vite configuration
├── README.md
test_data/
├── toy_spatial.h5ad          # Toy dataset for testing
└── generate_toy.py           # Script to regenerate toy data
```

## Architecture

- **Backend**: FastAPI + AnnData + Scanpy, serving data and running analysis via REST API
- **Frontend**: React + TypeScript + Vite + deck.gl + Zustand for state management
- **Data flow**: h5ad file → DataAdaptor → REST API → React hooks → deck.gl visualization
- **API docs**: Available at http://localhost:8000/docs when the backend is running
