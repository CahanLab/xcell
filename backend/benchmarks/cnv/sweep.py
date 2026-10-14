"""Parameter sweep of xcell prepare_cnv over the benchmark datasets.

python -I sweep.py <bench_dir> <sweep_name> '<json list of param dicts>' [datasets...]
Each run is saved as runs/<dataset>__<sweep_name>-<i>.npz (same layout as run_cnv.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cnv import run_xcell  # noqa: E402

BENCH = Path(sys.argv[1])
name = sys.argv[2]
grid = json.loads(sys.argv[3])
datasets = sys.argv[4:] or ['snu601', 'nugc4', 'MM199', 'BCC06', 'PBMC']
out = BENCH / 'runs'
for d in datasets:
    a = ad.read_h5ad(BENCH / 'h5ad' / f'{d}.h5ad')
    for i, params in enumerate(grid):
        try:
            res = run_xcell(a.copy(), params)
        except Exception as e:  # keep sweeping; record the failure
            print(f'{d} {name}-{i} FAILED {params}: {e}')
            continue
        extra = dict(is_tumor=a.obs['is_tumor'].values.astype(bool), is_ref=a.obs['is_ref'].values.astype(bool),
                     group=a.obs['group'].astype(str).values, patient=a.obs['patient'].astype(str).values if 'patient' in a.obs else np.array(['-'] * a.n_obs),
                     params=json.dumps(params), tool='xcell')
        np.savez_compressed(out / f'{d}__{name}-{i}.npz', **res, **extra)
        print(f'{d} {name}-{i} {params} windows={res["M"].shape[1]} {res["seconds"]:.1f}s')
