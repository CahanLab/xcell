"""XCell FastAPI application entry point."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from xcell.adaptor import DataAdaptor
from xcell.api.routes import router, set_adaptor

app = FastAPI(
    title="XCell",
    description="Modern web app for scRNA-seq and spatial transcriptomics exploration",
    version="0.1.0",
)

# Configure CORS for development (frontend runs on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Load data on application startup."""
    # Get data file path from environment variable
    data_path = os.environ.get("XCELL_DATA_PATH")

    if not data_path:
        # Fall back to bundled toy dataset
        bundled = Path(__file__).parent / "data" / "toy_spatial.h5ad"
        if bundled.exists():
            data_path = str(bundled)
            print("No XCELL_DATA_PATH set — using bundled toy_spatial.h5ad")
        else:
            print("Warning: XCELL_DATA_PATH not set. Set it to load an h5ad or h5 file.")
            print("Example: XCELL_DATA_PATH=/path/to/data.h5ad uvicorn xcell.main:app --reload")
            return

    path = Path(data_path)
    if path.exists():
        print(f"Loading data from: {path}")
        adaptor = DataAdaptor(path)
        set_adaptor(adaptor, slot="primary")
        print(f"Loaded {adaptor.n_cells} cells, {adaptor.n_genes} genes")
    else:
        print(f"Warning: Data file not found: {path}")


@app.get("/")
def root():
    """Root endpoint with basic info."""
    return {
        "name": "XCell",
        "version": "0.1.0",
        "docs": "/docs",
        "api": "/api/schema",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
