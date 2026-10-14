"""Score CNV runs against the Schmid et al. 2025 ground truth.

python -I evaluate.py <bench_dir> [glob]   -> prints a table and writes results.tsv

Profile metrics follow the paper: pseudobulk (mean over tumor cells) of the
window matrix, spread onto 100 kb bins (a bin takes the mean of the windows
overlapping it), truth = scWGS cell fractions (gain/loss called when > 50 % of
cells) or WES GATK calls (+/-/0 -> 3/1/2, bin mean), binarized at 1.5 / 2.5.
Max F1 = macro F1 over {loss, base, gain} maximized over a grid of loss and
gain thresholds on the method's own scale (baseline 0).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, adjusted_rand_score, homogeneity_score, f1_score

BENCH = Path(sys.argv[1])
TRUTH = {
    'snu601': ('scwgs', BENCH / 'truth' / 'SNU601_wgs_results_formated.csv'),
    'nugc4': ('scwgs', BENCH / 'truth' / 'NUGC4_wgs_results_formated.csv'),
    'MM199': ('wes', BENCH / 'truth' / 'MM199Tumor_clean.called.seg'),
    'BCC06': ('wes', BENCH / 'truth' / 'BCC06_tumor_clean.called.seg'),
    'BCCmix': (None, None),
    'PBMC': ('diploid', None),
}
BIN = 100_000


def truth_bins(kind, path) -> pd.DataFrame:
    """DataFrame chr, start, end, wgs_mean (1 loss, 2 base, 3 gain), gain_frac, loss_frac."""
    if kind == 'scwgs':
        t = pd.read_csv(path, index_col=0)
        t['chr'] = 'chr' + t['chr'].astype(str)
        t['wgs_mean'] = 1 * t['loss_wgs'] + 2 * t['base_wgs'] + 3 * t['gain_wgs']
        # the paper's rule for scWGS: a state is called when > 50 % of cells carry it
        t['call'] = np.where(t['gain_wgs'] > 0.5, 3, np.where(t['loss_wgs'] > 0.5, 1, 2))
        return t.rename(columns={'gain_wgs': 'gain_frac', 'loss_wgs': 'loss_frac'})[['chr', 'start', 'end', 'wgs_mean', 'call', 'gain_frac', 'loss_frac']]
    seg = pd.read_csv(path, sep='\t', comment='@')
    seg['num'] = seg['CALL'].map({'-': 1, '+': 3, '0': 2}).astype(float)
    rows = []
    for chrom, g in seg.groupby('CONTIG'):
        lo = (g['START'].min() // BIN) * BIN
        hi = int(np.ceil(g['END'].max() / BIN)) * BIN
        starts = np.arange(lo, hi, BIN)
        s0, e0 = g['START'].values, g['END'].values
        for b in starts:
            ov = (e0 >= b) & (s0 <= b + BIN - 1)
            if ov.any():
                m = g['num'].values[ov].mean()
                rows.append((chrom, b, b + BIN - 1, m))
    t = pd.DataFrame(rows, columns=['chr', 'start', 'end', 'wgs_mean'])
    t['call'] = np.where(t['wgs_mean'] > 2.5, 3, np.where(t['wgs_mean'] < 1.5, 1, 2))
    t['gain_frac'] = np.nan; t['loss_frac'] = np.nan
    return t


def windows_to_bins(values: np.ndarray, chrom: np.ndarray, start: np.ndarray, end: np.ndarray, truth: pd.DataFrame) -> np.ndarray:
    """Mean of the windows overlapping each truth bin; NaN when none does."""
    out = np.full(len(truth), np.nan)
    for c in np.unique(chrom):
        w = chrom == c
        tb = (truth['chr'] == c).values
        if not tb.any():
            continue
        ws, we, wv = start[w], end[w], values[w]
        bs, be = truth.loc[tb, 'start'].values, truth.loc[tb, 'end'].values
        acc = np.zeros(tb.sum()); n = np.zeros(tb.sum())
        for s, e, v in zip(ws, we, wv):
            ov = (be >= s) & (bs <= e)
            acc[ov] += v; n[ov] += 1
        res = np.where(n > 0, acc / np.maximum(n, 1), np.nan)
        out[tb] = res
    return out


def _macro_f1(lab: np.ndarray, call: np.ndarray) -> float:
    f = 0.0
    for k in (1, 2, 3):
        tp = np.sum((lab == k) & (call == k)); fp = np.sum((lab == k) & (call != k)); fn = np.sum((lab != k) & (call == k))
        f += 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / 3


def max_f1(pred: np.ndarray, call: np.ndarray):
    """Macro F1 over {1,2,3} maximized over loss (< t_l) and gain (> t_g) thresholds."""
    ok = ~np.isnan(pred)
    pred, call = pred[ok], call[ok]
    losses = np.quantile(pred[pred < 0], np.linspace(0, 1, 60)) if (pred < 0).any() else np.array([0.0])
    gains = np.quantile(pred[pred > 0], np.linspace(0, 1, 60)) if (pred > 0).any() else np.array([0.0])
    best = (-1, 0, 0)
    for tl in np.unique(np.append(losses, 0.0)):
        low = pred < tl
        for tg in np.unique(np.append(gains, 0.0)):
            if tl > tg:
                continue
            lab = np.where(low, 1, np.where(pred > tg, 3, 2))
            f = _macro_f1(lab, call)
            if f > best[0]:
                best = (f, tl, tg)
    f, tl, tg = best
    lab = np.where(pred < tl, 1, np.where(pred > tg, 3, 2))
    def sp(k):
        tp = ((lab == k) & (call == k)).sum()
        return tp / max((call == k).sum(), 1), tp / max((lab == k).sum(), 1)
    sg, pg = sp(3); sl, pl = sp(1)
    return dict(max_f1=f, t_loss=tl, t_gain=tg, sens_gain=sg, prec_gain=pg, sens_loss=sl, prec_loss=pl)


def evaluate(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    name, tag = npz_path.stem.split('__', 1)
    kind, path = TRUTH[name]
    M = z['M']; is_tumor = z['is_tumor']; is_ref = z['is_ref']; score = z['score']; clone = z['clone']
    row = dict(dataset=name, run=tag, tool=str(z['tool']), n_windows=M.shape[1], seconds=float(z['seconds']), params=str(z['params']))
    tum = M[is_tumor]
    ref = M[is_ref]
    row['sd_ref'] = float(np.nanstd(ref)) if len(ref) else np.nan
    row['sd_tumor'] = float(np.nanstd(tum)) if len(tum) else np.nan
    # tumor vs reference separation by the per-cell score
    if is_tumor.any() and is_ref.any():
        y = np.r_[np.ones(is_tumor.sum()), np.zeros(is_ref.sum())]
        s = np.r_[score[is_tumor], score[is_ref]]
        row['auroc_score_tumor_vs_ref'] = float(roc_auc_score(y, s))
        # unsupervised rule: a clone is normal when its mean score is below 2x the reference mean (per-cell accuracy)
        ref_mean = float(np.mean(score[is_ref]))
        cl = pd.Series(score).groupby(clone).mean()
        normal_clones = set(cl[cl < 2 * ref_mean].index)
        pred_normal = np.isin(clone, list(normal_clones))
        acc = ((~pred_normal) == is_tumor)[is_tumor | is_ref].mean()
        row['acc_clone_rule'] = float(acc)
    if kind in ('scwgs', 'wes'):
        truth = truth_bins(kind, path)
        pb = np.nanmean(tum, axis=0)
        binned = windows_to_bins(pb, z['win_chrom'], z['win_start'].astype(float), z['win_end'].astype(float), truth)
        ok = ~np.isnan(binned)
        row['n_bins'] = int(ok.sum())
        row['pearson'] = float(np.corrcoef(binned[ok], truth['wgs_mean'].values[ok])[0, 1])
        call = truth['call'].values
        if (call[ok] == 3).any():
            row['auc_gain'] = float(roc_auc_score(call[ok] == 3, binned[ok]))
        if (call[ok] == 1).any():
            row['auc_loss'] = float(roc_auc_score(call[ok] == 1, -binned[ok]))
        row.update(max_f1(binned, call))
        if kind == 'scwgs':
            # subclonal test: bins where 20-80 % of scWGS cells carry a gain (or loss) --
            # does the fraction of RNA cells called gained track the DNA fraction?
            tl, tg = row['t_loss'], row['t_gain']
            cell_bins_gain = np.zeros(len(truth)); cell_bins_loss = np.zeros(len(truth))
            # per-cell binarization is expensive on 28k bins x 5k cells; do it per window then spread
            wg = (tum > tg).mean(axis=0); wl = (tum < tl).mean(axis=0)
            fg = windows_to_bins(wg, z['win_chrom'], z['win_start'].astype(float), z['win_end'].astype(float), truth)
            fl = windows_to_bins(wl, z['win_chrom'], z['win_start'].astype(float), z['win_end'].astype(float), truth)
            g = truth['gain_frac'].values; l = truth['loss_frac'].values
            sub = ok & (((g > 0.2) & (g < 0.8)) | ((l > 0.2) & (l < 0.8)))
            row['n_subclonal_bins'] = int(sub.sum())
            if sub.sum() > 10:
                row['r_subclonal_gain_frac'] = float(np.corrcoef(fg[sub], g[sub])[0, 1])
                row['r_subclonal_loss_frac'] = float(np.corrcoef(fl[sub], l[sub])[0, 1])
            allok = ok
            row['r_gain_frac_all'] = float(np.corrcoef(fg[allok], g[allok])[0, 1])
            row['r_loss_frac_all'] = float(np.corrcoef(fl[allok], l[allok])[0, 1])
    if name.startswith('BCCmix'):  # BCCmix and any variant of it
        pat = z['patient'][is_tumor]
        row['ari_patient'] = float(adjusted_rand_score(pat, clone[is_tumor]))
        row['homogeneity_patient'] = float(homogeneity_score(pat, clone[is_tumor]))
        row['n_clones_tumor'] = int(len(np.unique(clone[is_tumor])))
    if kind == 'diploid':
        other = M[~is_ref]
        row['rmse_nonref_from_zero'] = float(np.sqrt(np.nanmean(other ** 2)))
        row['mean_abs_nonref'] = float(np.nanmean(np.abs(other)))
        row['frac_nonref_nonzero'] = float(np.nanmean(other != 0))
        row['n_clones'] = int(len(np.unique(clone)))
    row['n_clones_all'] = int(len(np.unique(clone)))
    return row


if __name__ == '__main__':
    pattern = sys.argv[2] if len(sys.argv) > 2 else '*'
    files = sorted(glob.glob(str(BENCH / 'runs' / f'{pattern}.npz')))
    rows = [evaluate(Path(f)) for f in files]
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['dataset', 'run', 'n_windows', 'seconds', 'pearson', 'auc_gain', 'auc_loss', 'max_f1', 'sens_gain', 'prec_gain', 'sens_loss', 'prec_loss',
            'auroc_score_tumor_vs_ref', 'acc_clone_rule', 'n_clones_all', 'r_subclonal_gain_frac', 'r_subclonal_loss_frac', 'ari_patient', 'homogeneity_patient', 'rmse_nonref_from_zero', 'frac_nonref_nonzero', 'sd_ref', 'sd_tumor']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].round(3).to_string(index=False))
    out = BENCH / 'results.tsv'
    if out.exists():
        old = pd.read_csv(out, sep='\t')
        df = pd.concat([old[~old.set_index(['dataset', 'run']).index.isin(df.set_index(['dataset', 'run']).index)], df], ignore_index=True)
    df.to_csv(out, sep='\t', index=False)
