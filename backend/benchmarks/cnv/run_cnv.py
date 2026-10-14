"""Run xcell's prepare_cnv (and optionally infercnvpy) on a benchmark h5ad.

python -I run_cnv.py <dataset.h5ad> <out_dir> <tag> [--tool xcell|infercnvpy] [--param k=v ...]

Writes <out_dir>/<dataset>__<tag>.npz with:
  M (cells x windows float32, NaN outside the mask), win_chrom, win_start, win_end,
  score, clone (str), is_tumor, is_ref, group, patient (if any), params (json).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

BACKEND = Path(__file__).resolve().parents[2]          # backend/
PYLIB = Path(__file__).resolve().parent / 'pylib'      # optional: infercnvpy installed with pip --target
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PYLIB))


def parse(argv):
    h5, out, tag = argv[1:4]
    tool = 'xcell'
    params = {}
    it = iter(argv[4:])
    for a in it:
        if a == '--tool':
            tool = next(it)
        elif a == '--param':
            k, v = next(it).split('=', 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            params[k] = v
    return Path(h5), Path(out), tag, tool, params


def run_xcell(a: ad.AnnData, params: dict):
    from xcell.adaptor import DataAdaptor
    ref_cats = sorted(a.obs.loc[a.obs['is_ref'], 'group'].astype(str).unique())
    kw = dict(key='cnv', species='human', reference_column='group', reference_categories=ref_cats)
    kw.update(params)
    if kw.get('reference_column') is None:
        kw.pop('reference_categories', None)
    adaptor = DataAdaptor('bench.h5ad', adata=a)
    compute, apply = adaptor.prepare_cnv(**kw)
    t0 = time.time()
    res = compute(lambda f, m: None)
    out = apply(res)
    dt = time.time() - t0
    rec = json.loads(adaptor.adata.uns['xcell_cnv']['cnv'])
    wins = rec['windows']
    M = np.asarray(adaptor.adata.obsm['X_cnv'], dtype=np.float32)
    clone = adaptor.adata.obs['cnv_clone'].astype(str).values if 'cnv_clone' in adaptor.adata.obs else np.array(['none'] * a.n_obs)
    return dict(M=M, win_chrom=np.array([w['chromosome'] for w in wins]), win_start=np.array([w['start'] for w in wins]),
                win_end=np.array([w['end'] for w in wins]), score=adaptor.adata.obs['cnv_score'].values.astype(np.float32),
                clone=clone, seconds=dt, info=json.dumps({k: v for k, v in rec.items() if k not in ('windows', 'clones')}, default=str))


def run_infercnvpy(a: ad.AnnData, params: dict):
    import scanpy as sc
    import infercnvpy as cnv
    from xcell import cnv as xc
    table = xc.load_positions('human')
    ids = a.var['gene_id'].values if 'gene_id' in a.var else a.var_names.values
    # match by id, then symbol, then upper symbol, like xcell
    chrom, start, end = [], [], []
    for gid, sym in zip(ids, a.var_names):
        p = table.get(str(gid).split('.')[0]) or table.get(sym) or table.get(sym.upper())
        if p is None:
            chrom.append(None); start.append(np.nan); end.append(np.nan)
        else:
            chrom.append(p[0]); start.append(p[1]); end.append(p[2])
    a.var['chromosome'] = chrom
    a.var['start'] = start
    a.var['end'] = end
    a = a[:, a.var['chromosome'].notna()].copy()
    a.var['chromosome'] = a.var['chromosome'].astype(str)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    ref_cats = sorted(a.obs.loc[a.obs['is_ref'], 'group'].astype(str).unique())
    kw = dict(reference_key='group', reference_cat=ref_cats, window_size=100, step=10, lfc_clip=3, dynamic_threshold=1.5, n_jobs=4)
    kw.update(params)
    t0 = time.time()
    cnv.tl.infercnv(a, **kw)
    cnv.tl.pca(a)
    cnv.pp.neighbors(a)
    cnv.tl.leiden(a, resolution=1.0)
    dt = time.time() - t0
    M = np.asarray(a.obsm['X_cnv'].todense(), dtype=np.float32)
    chr_pos = a.uns['cnv']['chr_pos']  # {chrom: first column}
    order = sorted(chr_pos.items(), key=lambda kv: kv[1])
    win_chrom = np.empty(M.shape[1], dtype=object)
    bounds = [c for _, c in order] + [M.shape[1]]
    for (name, c0), c1 in zip(order, bounds[1:]):
        win_chrom[c0:c1] = name
    # window genomic spans: infercnvpy keeps genes sorted by start within chromosome; reconstruct
    win_start = np.full(M.shape[1], np.nan); win_end = np.full(M.shape[1], np.nan)
    var = a.var.sort_values(['chromosome', 'start'])
    for name, c0 in order:
        c1 = bounds[bounds.index(c0) + 1]
        g = var[var['chromosome'] == name]
        starts = g['start'].values
        n = len(starts)
        for j, col in enumerate(range(c0, c1)):
            lo = j * kw['step']
            hi = min(lo + kw['window_size'], n)
            if lo >= n:
                lo = max(n - 1, 0)
            win_start[col] = starts[lo]; win_end[col] = starts[hi - 1]
    score = np.abs(M).mean(axis=1)
    return dict(M=M, win_chrom=win_chrom.astype(str), win_start=win_start, win_end=win_end, score=score.astype(np.float32),
                clone=a.obs['cnv_leiden'].astype(str).values, seconds=dt, info=json.dumps({'n_windows': M.shape[1], 'n_genes': a.n_vars}))


if __name__ == '__main__':
    h5, out, tag, tool, params = parse(sys.argv)
    a = ad.read_h5ad(h5)
    res = run_xcell(a, params) if tool == 'xcell' else run_infercnvpy(a, params)
    out.mkdir(parents=True, exist_ok=True)
    extra = dict(is_tumor=a.obs['is_tumor'].values.astype(bool), is_ref=a.obs['is_ref'].values.astype(bool),
                 group=a.obs['group'].astype(str).values, patient=a.obs['patient'].astype(str).values if 'patient' in a.obs else np.array(['-'] * a.n_obs),
                 params=json.dumps(params), tool=tool)
    np.savez_compressed(out / f'{h5.stem}__{tag}.npz', **res, **extra)
    print(f'{h5.stem} {tag} {tool}: windows={res["M"].shape[1]} seconds={res["seconds"]:.1f} info={res["info"][:300]}')
