"""Assemble the benchmark datasets (Schmid et al. 2025 subset) into h5ad files.

Each output has raw counts in X, obs['group'] (reference category or tumor
label), obs['is_tumor'] (bool), obs['is_ref'] (bool) and, where known,
obs['patient'].  Run with `python -I load_data.py <dataset> <bench_dir>`.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
import anndata as ad

BENCH = Path(sys.argv[2])
RAW = BENCH / 'raw'
OUT = BENCH / 'h5ad'


def read_10x_mtx(d: Path, prefix: str) -> ad.AnnData:
    m = scipy.io.mmread(gzip.open(d / f'{prefix}_matrix.mtx.gz')).T.tocsr().astype(np.float32)
    genes = pd.read_csv(d / f'{prefix}_genes.tsv.gz', sep='\t', header=None)
    bcs = pd.read_csv(d / f'{prefix}_barcodes.tsv.gz', sep='\t', header=None)[0].values
    # Andor et al. deposited a one-column gene file (symbols only, 14,377 pre-filtered genes)
    sym = genes[genes.shape[1] - 1].values
    a = ad.AnnData(m, obs=pd.DataFrame(index=bcs), var=pd.DataFrame({'gene_id': genes[0].values, 'symbol': sym}, index=sym))
    a.var_names_make_unique()
    return a


def read_dense_txt(path: Path, index_col=0) -> ad.AnnData:
    df = pd.read_csv(path, sep='\t', index_col=index_col)
    df = df.groupby(level=0).sum()  # duplicate symbols
    a = ad.AnnData(sp.csr_matrix(df.values.T.astype(np.float32)), obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
    return a


def gastric_reference() -> ad.AnnData:
    parts = []
    for p in range(25, 30):
        g = p + 22
        a = read_dense_txt(RAW / 'gastric_ref' / f'GSM45463{g}_Pat{p}-A.txt.gz')
        a.obs['group'] = f'Patient{p}'
        parts.append(a)
    ref = ad.concat(parts, join='inner')
    ref.obs['is_tumor'] = False
    ref.obs['is_ref'] = True
    return ref


def basic_qc(a: ad.AnnData, min_counts=1000, max_mt=20.0) -> ad.AnnData:
    a.var['mt'] = a.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], inplace=True, percent_top=None)
    keep = (a.obs['total_counts'] >= min_counts) & (a.obs['pct_counts_mt'] <= max_mt)
    return a[keep].copy()


def cell_line(name: str, prefix: str, truth_label: str):
    tum = read_10x_mtx(RAW / name, prefix)
    tum = basic_qc(tum)
    tum.obs['group'] = truth_label
    tum.obs['is_tumor'] = True
    tum.obs['is_ref'] = False
    ref = gastric_reference()
    ref = basic_qc(ref)
    # intersect on symbol; keep the Ensembl id from the 10x side for position matching
    common = tum.var_names.intersection(ref.var_names)
    tum = tum[:, common].copy()
    ref = ref[:, common].copy()
    both = ad.concat([tum, ref], join='inner', merge='first')
    both.var['gene_id'] = tum.var.loc[common, 'gene_id'].values
    both.obs_names_make_unique()
    for c in ('group',):
        both.obs[c] = both.obs[c].astype('category')
    both.write_h5ad(OUT / f'{name}.h5ad')
    print(name, both.shape, both.obs['group'].value_counts().to_dict())


def annotate_clusters(a: ad.AnnData, markers: dict[str, list[str]], resolution=1.0) -> pd.Series:
    """Coarse cell-type labels: Leiden clusters named by their top marker-set score."""
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    sc.pp.pca(b, n_comps=30, mask_var='highly_variable')
    sc.pp.neighbors(b, n_neighbors=15)
    sc.tl.leiden(b, resolution=resolution, flavor='igraph', n_iterations=2, random_state=0)
    scores = {}
    for ct, genes in markers.items():
        genes = [g for g in genes if g in b.var_names]
        sc.tl.score_genes(b, genes, score_name=ct)
        scores[ct] = b.obs.groupby('leiden', observed=True)[ct].mean()
    S = pd.DataFrame(scores)
    S = (S - S.mean()) / S.std()
    label = S.idxmax(axis=1)
    print(S.round(2).assign(label=label, n=b.obs['leiden'].value_counts()))
    return b.obs['leiden'].map(label).astype(str)


def mm():
    counts = read_dense_txt(RAW / 'mm' / 'MM199_All_Celltypes_RawData.txt.gz')
    mito = read_dense_txt(RAW / 'mm' / 'MM199_Mitochondrial_Expression.txt.gz')
    a = ad.concat([counts, mito], axis=1, join='outer', merge='first')
    a = basic_qc(a, min_counts=1000, max_mt=10.0)
    markers = {
        'Plasma': ['SDC1', 'TNFRSF17', 'MZB1', 'XBP1', 'JCHAIN', 'IGKC', 'IGHG1', 'PRDM1'],
        'T': ['CD3D', 'CD3E', 'CD2', 'IL7R', 'TRAC'],
        'NK': ['NKG7', 'GNLY', 'KLRD1', 'KLRF1'],
        'Mono': ['LYZ', 'S100A8', 'S100A9', 'CD14', 'FCGR3A', 'CST3'],
        'B': ['MS4A1', 'CD79A', 'CD79B', 'CD19'],
        'Erythroid': ['HBB', 'HBA1', 'HBA2', 'GYPA'],
        'Progenitor': ['CD34', 'SPINK2', 'MPO', 'ELANE'],
    }
    lab = annotate_clusters(a, markers)
    a.obs['group'] = lab.where(lab != 'Plasma', 'MM').astype('category')
    a.obs['is_tumor'] = (a.obs['group'] == 'MM').values
    a.obs['is_ref'] = a.obs['group'].isin(['T', 'NK', 'Mono', 'B']).values
    a.write_h5ad(OUT / 'MM199.h5ad')
    print('MM199', a.shape, a.obs['group'].value_counts().to_dict())


def bcc():
    meta = pd.read_csv(RAW / 'bcc' / 'GSE123813_bcc_all_metadata.txt.gz', sep='\t', index_col=0)
    meta = meta[meta['treatment'] == 'pre']
    meta['cluster'] = meta['cluster'].replace({'Tumor_1': 'Tumor', 'Tumor_2': 'Tumor'})
    meta = meta[meta['cluster'].isin(['Endothelial', 'Melanocytes', 'Myofibroblasts', 'Tumor'])]
    n_tum = meta[meta['cluster'] == 'Tumor'].groupby('patient').size()
    meta = meta[meta['patient'].isin(n_tum[n_tum >= 20].index)]
    print('BCC cells wanted', meta.shape, meta.groupby(['patient', 'cluster']).size().to_dict())
    # R-style table: header has one field fewer than the rows (gene names unnamed).
    want = set(meta.index)
    with gzip.open(RAW / 'bcc' / 'GSE123813_bcc_scRNA_counts.txt.gz', 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        idx = np.array([i + 1 for i, c in enumerate(header) if c in want])
        cols = [header[i - 1] for i in idx]
        genes, rows = [], []
        for line in fh:
            f = line.rstrip('\n').split('\t')
            genes.append(f[0])
            rows.append(np.array(f, dtype=object)[idx].astype(np.float32))
    df = pd.DataFrame(np.vstack(rows), index=genes, columns=cols)
    df = df.groupby(level=0).sum()
    a = ad.AnnData(sp.csr_matrix(df.values.T.astype(np.float32)), obs=meta.loc[df.columns, ['patient', 'cluster']].copy(), var=pd.DataFrame(index=df.index))
    a.obs['is_tumor'] = (a.obs['cluster'] == 'Tumor').values
    a.obs['is_ref'] = ~a.obs['is_tumor'].values
    # per-patient dataset: su006 (BCC06 in Schmid et al.)
    s6 = a[a.obs['patient'] == 'su006'].copy()
    s6.obs['group'] = s6.obs['cluster'].where(s6.obs['cluster'] != 'Tumor', 'BCC06').astype('category')
    s6.write_h5ad(OUT / 'BCC06.h5ad')
    print('BCC06', s6.shape, s6.obs['group'].value_counts().to_dict())
    # mixed dataset: tumor cells of all qualifying patients plus all normals
    a.obs['group'] = a.obs['cluster'].where(a.obs['cluster'] != 'Tumor', 'BCC').astype('category')
    a.obs['patient'] = a.obs['patient'].astype('category')
    a.write_h5ad(OUT / 'BCCmix.h5ad')
    print('BCCmix', a.shape, a.obs['group'].value_counts().to_dict())


def pbmc():
    a = sc.read_10x_h5(RAW / 'pbmc' / 'pbmc_10k_v3_filtered_feature_bc_matrix.h5')
    a.var_names_make_unique()
    a.var['gene_id'] = a.var['gene_ids'].values
    a = basic_qc(a, min_counts=1000, max_mt=15.0)
    markers = {
        'CD4T': ['CD3D', 'CD3E', 'IL7R', 'CD4', 'CCR7'],
        'CD8T': ['CD3D', 'CD8A', 'CD8B', 'GZMK'],
        'NK': ['NKG7', 'GNLY', 'KLRD1', 'KLRF1'],
        'Mono': ['LYZ', 'S100A8', 'S100A9', 'CD14', 'FCGR3A'],
        'B': ['MS4A1', 'CD79A', 'CD79B'],
        'DC': ['FCER1A', 'CLEC10A', 'LILRA4', 'IRF8'],
        'Platelet': ['PPBP', 'PF4'],
    }
    lab = annotate_clusters(a, markers, resolution=0.5)
    a.obs['group'] = lab.astype('category')
    a.obs['is_tumor'] = False
    a.obs['is_ref'] = (a.obs['group'] == 'CD4T').values
    a.write_h5ad(OUT / 'PBMC.h5ad')
    print('PBMC', a.shape, a.obs['group'].value_counts().to_dict())


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'SNU601':
        cell_line('snu601', 'GSM4238687_SNU-601', 'SNU601')
    elif which == 'NUGC4':
        cell_line('nugc4', 'GSM4238684_NUGC-4', 'NUGC4')
    elif which == 'MM':
        mm()
    elif which == 'BCC':
        bcc()
    elif which == 'PBMC':
        pbmc()
