# XCell

Modern web application for exploring scRNA-seq and spatial transcriptomics data.

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- An h5ad file (AnnData format) with embeddings in `.obsm`

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Run with auto-reload (replace with your h5ad file path)
XCELL_DATA_PATH=/path/to/your/data.h5ad uvicorn xcell.main:app --reload --port 8000
```

The backend will:
- Auto-reload when you edit Python files
- Provide API docs at http://localhost:8000/docs
- Expose API at http://localhost:8000/api/

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run dev server with hot module replacement
npm run dev
```

The frontend will:
- Run at http://localhost:5173
- Instantly update when you edit React/TypeScript files
- Proxy API requests to the backend

### Example with Sample Data

If you have the excellxgene example dataset:

```bash
# Terminal 1: Backend
cd backend
XCELL_DATA_PATH=../../excellxgene/example-dataset/pbmc3k.h5ad uvicorn xcell.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Project Structure

```
xcell/
├── backend/
│   ├── xcell/
│   │   ├── main.py       # FastAPI app entry point
│   │   ├── adaptor.py    # DataAdaptor class (wraps AnnData)
│   │   └── api/
│   │       └── routes.py # REST API endpoints
│   └── pyproject.toml    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main app component
│   │   ├── store.ts             # Zustand state management
│   │   ├── components/
│   │   │   └── ScatterPlot.tsx  # deck.gl scatter plot
│   │   └── hooks/
│   │       └── useData.ts       # Data fetching hooks
│   ├── package.json             # Node dependencies
│   └── vite.config.ts           # Vite configuration
└── README.md
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/schema` | Dataset info: cell/gene counts, embedding names, metadata columns |
| `GET /api/embedding/{name}` | Embedding coordinates (e.g., X_umap, X_pca) |
| `GET /api/obs/{column}` | Cell metadata values for coloring |
| `GET /api/health` | Health check |

## Development Workflow

The setup prioritizes fast iteration:

1. **Edit Python code** → Backend auto-reloads (uvicorn --reload)
2. **Edit React/TypeScript** → Browser updates instantly (Vite HMR)
3. **No manual restart needed** for most changes

## Architecture

- **Backend**: FastAPI + AnnData + Scanpy-ready adaptor pattern
- **Frontend**: React + Vite + TypeScript + deck.gl + Zustand
- **Data flow**: h5ad → DataAdaptor → REST API → React hooks → deck.gl visualization

## Future Additions

The `DataAdaptor` class is designed for easy scanpy integration:

```python
# Planned methods
adaptor.run_pca(n_comps=50)
adaptor.run_umap()
adaptor.run_leiden(resolution=1.0)
adaptor.run_diffexp(groupby='cluster')
```
