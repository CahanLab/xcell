"""Infer what scale an expression matrix is on — counts, normalized, or logged.

Datasets arrive from collaborators and public repositories with no reliable
record of what was done to them. ``adata.X`` might be raw UMIs, might be
CPM, might be ``log1p(normalize_total(...))``, might be z-scored. Guessing
wrong is not cosmetic: UCell ranks and top-scoring-pair classifiers assume a
per-cell monotone scale, ``normalize_total`` on already-normalized data is a
no-op that silently looks fine, and z-scored input breaks anything that
assumes non-negativity.

This module answers the question from the numbers themselves. The signals,
in the order they discriminate:

  * **negatives** — only centering/scaling produces them.
  * **integrality** — counts are integers; nothing downstream of
    ``normalize_total`` is.
  * **row-sum constancy** — ``normalize_total`` makes every library sum to
    the same target. Surviving that check is strong evidence.
  * **expm1 row-sum constancy** — the same check after undoing ``log1p``,
    which identifies the scanpy-default log-normalized scale.
  * **dynamic range** — log-scale values live in roughly [0, 25]; linear
    normalized values run to the hundreds or thousands.

Row sums are computed over the *full* gene axis of a sampled set of cells, so
they stay exact per sampled cell. Only the cell axis is subsampled.

Nothing here mutates the AnnData. Every verdict carries the evidence that
produced it so a user can disagree with the classifier and still be informed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Human-readable name per verdict. The frontend renders these verbatim, so
# they are short enough to sit inside a dropdown option label.
SCALE_LABELS: dict[str, str] = {
    'raw_counts': 'raw counts',
    'normalized_linear': 'normalized (linear)',
    'log_normalized': 'log-normalized',
    'log_transformed': 'log-transformed',
    'z_scored': 'scaled / z-scored',
    'binary': 'binary',
    'empty': 'empty',
    'unknown': 'unknown scale',
}

# One-line explanation of what each verdict means for downstream analysis.
SCALE_NOTES: dict[str, str] = {
    'raw_counts': 'Integer UMI/read counts. Safe input for normalize_total, '
                  'HVG (seurat_v3), and count-based models.',
    'normalized_linear': 'Library-size normalized but not log-transformed. '
                         'Usually wants a log1p before clustering or DE.',
    'log_normalized': 'log1p of library-size normalized values — the scanpy '
                      'default expression scale. Ready for PCA / DE / plotting.',
    'log_transformed': 'On a log scale, but library size was never equalized '
                       '(or genes were subset afterwards). Check whether '
                       'normalize_total was run.',
    'z_scored': 'Per-gene centered and scaled. Do not feed to methods that '
                'assume non-negative values (UCell, NMF, count models).',
    'binary': 'Values are only 0 and 1 — a presence/absence matrix.',
    'empty': 'No non-zero values to judge.',
    'unknown': 'Does not match a recognized scale.',
}

# --- thresholds ------------------------------------------------------------
# A coefficient of variation this small across library sizes only happens by
# construction, not by chance — normalize_total is the only common cause.
_CONSTANT_ROWSUM_CV = 0.01
# log1p of a normalized count matrix tops out well under this; linear
# normalized values blow past it immediately.
_LOG_RANGE_MAX = 25.0
# Below this the "integers" could be a rounded/downsampled matrix, so the
# raw-counts call gets an explicit caveat instead of silent confidence.
_MIN_COUNT_MAX_FOR_HIGH_CONF = 2.0


def _sample_rows(M, max_cells: int, seed: int):
    """Return up to ``max_cells`` full rows of ``M`` as a dense float array.

    Deterministic: the row indices come from a seeded generator, so repeated
    calls on the same matrix produce identical statistics.
    """
    from scipy.sparse import issparse

    n_obs = M.shape[0]
    if n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_obs, size=max_cells, replace=False))
        sub = M[idx]
    else:
        sub = M

    if issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float64)


def _cv(values: np.ndarray) -> float | None:
    """Coefficient of variation, or None when it isn't defined.

    None rather than ``inf`` because these stats go out over JSON, which has
    no encoding for infinity. Callers must treat None as "no evidence".
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return None
    return float(np.std(values) / abs(mean))


def _below(cv: float | None, threshold: float) -> bool:
    """True only when a coefficient of variation exists and is under threshold."""
    return cv is not None and cv < threshold


def assess_matrix_scale(
    M,
    *,
    max_cells: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Classify the scale of an expression matrix from its values alone.

    Args:
        M: Dense array or scipy sparse matrix, cells x genes.
        max_cells: Cap on how many cell rows to read. Rows are sampled whole,
            so per-cell row sums stay exact.
        seed: Seed for the row sample, so results are reproducible.

    Returns:
        Dict with ``verdict`` (a key of :data:`SCALE_LABELS`), ``label``,
        ``note``, ``confidence`` ('high' | 'medium' | 'low'), ``reasons``
        (list of short human strings), and ``stats`` (the evidence).
        Never raises — an unusable matrix comes back as ``unknown``.
    """
    stats: dict[str, Any] = {
        'n_cells_sampled': 0,
        'n_genes': int(M.shape[1]) if hasattr(M, 'shape') and len(M.shape) > 1 else 0,
        'min': 0.0,
        'max': 0.0,
        'integer_valued': False,
        'has_negative': False,
        'nonzero_frac': 0.0,
        'nonzero_mean': 0.0,
        'row_sum_median': 0.0,
        'row_sum_cv': None,
        'expm1_row_sum_median': 0.0,
        'expm1_row_sum_cv': None,
    }

    def _out(verdict: str, confidence: str, reasons: list[str]) -> dict[str, Any]:
        return {
            'verdict': verdict,
            'label': SCALE_LABELS[verdict],
            'note': SCALE_NOTES[verdict],
            'confidence': confidence,
            'reasons': reasons,
            'stats': stats,
            'provenance': [],
        }

    try:
        A = _sample_rows(M, max_cells, seed)
    except Exception:
        return _out('unknown', 'low', ['matrix could not be read'])

    if A.ndim != 2 or A.size == 0:
        return _out('empty', 'low', ['matrix has no values'])

    stats['n_cells_sampled'] = int(A.shape[0])
    stats['n_genes'] = int(A.shape[1])

    finite = np.isfinite(A)
    if not finite.all():
        n_bad = int((~finite).sum())
        if not finite.any():
            return _out('unknown', 'low', ['all values are NaN or infinite'])
        # Zero out non-finite entries so the summaries below stay meaningful,
        # and say so in the reasons.
        A = np.where(finite, A, 0.0)
        nonfinite_reason = [f'{n_bad:,} non-finite values were ignored']
    else:
        nonfinite_reason = []

    vmin = float(A.min())
    vmax = float(A.max())
    stats['min'] = vmin
    stats['max'] = vmax

    nz = A[A != 0]
    stats['nonzero_frac'] = float(nz.size / A.size)
    stats['nonzero_mean'] = float(nz.mean()) if nz.size else 0.0

    if nz.size == 0:
        return _out('empty', 'low', nonfinite_reason + ['every sampled value is zero'])

    has_negative = bool(vmin < 0)
    stats['has_negative'] = has_negative
    integer_valued = bool(np.all(np.equal(np.mod(A, 1.0), 0.0)))
    stats['integer_valued'] = integer_valued

    row_sums = A.sum(axis=1)
    stats['row_sum_median'] = float(np.median(row_sums))
    stats['row_sum_cv'] = _cv(row_sums)

    # expm1 only makes sense on non-negative data, and overflows on large
    # values — both cases just leave the log-scale check unavailable.
    if not has_negative and vmax < 700:
        expm1_sums = np.expm1(A).sum(axis=1)
        stats['expm1_row_sum_median'] = float(np.median(expm1_sums))
        stats['expm1_row_sum_cv'] = _cv(expm1_sums)

    reasons = list(nonfinite_reason)

    # 1. Negatives — only centering produces them.
    if has_negative:
        reasons.append(f'contains negative values (min {vmin:.2f})')
        col_mean = float(np.abs(A.mean(axis=0)).mean())
        col_sd = float(np.median(A.std(axis=0)))
        centered = col_mean < 0.1
        unit_sd = 0.5 < col_sd < 2.0
        if centered:
            reasons.append('per-gene means are ~0 — genes were centered')
        if unit_sd:
            reasons.append(f'per-gene SD ~{col_sd:.2f} — genes were scaled to unit variance')
        conf = 'high' if (centered and unit_sd) else 'medium'
        return _out('z_scored', conf, reasons)

    # 2. Integers — counts, or a binary/presence matrix.
    if integer_valued:
        if vmax <= 1.0:
            reasons.append('every value is 0 or 1')
            return _out('binary', 'high', reasons)
        reasons.append(f'all values are non-negative integers (max {vmax:.0f})')
        if _below(stats['row_sum_cv'], _CONSTANT_ROWSUM_CV):
            # Integers *and* a fixed library size means someone downsampled to
            # a common depth. Still counts, but worth flagging.
            reasons.append(
                f'library sizes are all ~{stats["row_sum_median"]:,.0f} — '
                'counts appear to have been downsampled to a common depth'
            )
            return _out('raw_counts', 'medium', reasons)
        reasons.append(
            f'library sizes vary (median {stats["row_sum_median"]:,.0f}, '
            f'CV {stats["row_sum_cv"]:.2f})'
        )
        conf = 'high' if vmax > _MIN_COUNT_MAX_FOR_HIGH_CONF else 'medium'
        return _out('raw_counts', conf, reasons)

    # 3. Non-integer, non-negative. Constant row sums are the giveaway.
    reasons.append(f'values are non-integer (max {vmax:.3g})')

    if _below(stats['row_sum_cv'], _CONSTANT_ROWSUM_CV):
        reasons.append(
            f'every cell sums to ~{stats["row_sum_median"]:,.0f} — '
            'library sizes were normalized to a fixed target'
        )
        reasons.append('values are not on a log scale (no log1p applied)')
        return _out('normalized_linear', 'high', reasons)

    if _below(stats['expm1_row_sum_cv'], _CONSTANT_ROWSUM_CV):
        reasons.append(
            f'after undoing log1p, every cell sums to ~'
            f'{stats["expm1_row_sum_median"]:,.0f} — normalized, then log1p'
        )
        return _out('log_normalized', 'high', reasons)

    # 4. No constant-sum signal. Fall back to dynamic range. Gene subsetting
    #    after normalization lands here, which is common enough to deserve a
    #    real answer rather than 'unknown'.
    if vmax <= _LOG_RANGE_MAX:
        reasons.append(
            f'values span [{vmin:.3g}, {vmax:.3g}] — a log-scale range, but '
            'library sizes are not constant (genes may have been subset after '
            'normalizing, or log1p was applied to raw counts)'
        )
        return _out('log_transformed', 'medium', reasons)

    reasons.append(
        f'values reach {vmax:,.1f} with variable library sizes '
        f'(CV {stats["row_sum_cv"]:.2f}) — linear scale, normalization unclear'
    )
    return _out('unknown', 'low', reasons)


def provenance_from_adata(adata, layer_name: str) -> list[str]:
    """Recover recorded provenance for one matrix from the AnnData itself.

    These are facts, not inferences — scanpy and xcell both leave breadcrumbs
    in ``.uns``. Only ``.X`` inherits the scanpy ones, since that is the
    matrix scanpy's preprocessing functions operate on.
    """
    out: list[str] = []
    uns = getattr(adata, 'uns', {}) or {}

    if layer_name == 'X' and 'log1p' in uns:
        base = None
        try:
            base = (uns['log1p'] or {}).get('base')
        except Exception:
            pass
        out.append(
            "adata.uns['log1p'] is set — scanpy's log1p was applied to .X"
            + (f' (base {base})' if base else '')
        )

    xcell_uns = uns.get('xcell') if isinstance(uns.get('xcell'), dict) else {}
    if layer_name == 'counts' and xcell_uns.get('counts_inferred'):
        out.append(
            "xcell created this layer at load time by copying .X, because .X "
            "looked like integer counts — it was not in the original file"
        )
    return out


def provenance_from_history(history: list[dict], layer_name: str) -> list[str]:
    """Summarize xcell preprocessing actions that wrote to this matrix.

    ``history`` is the adaptor's ``_action_history``. Only actions that change
    the scale of a matrix are reported; filtering and clustering do not.
    """
    out: list[str] = []
    for entry in history or []:
        action = entry.get('action')
        params = entry.get('params') or {}
        result = entry.get('result') or {}
        if action == 'normalize_total' and layer_name == 'X':
            target = params.get('target_sum') or result.get('target_sum')
            out.append(
                'xcell ran normalize_total on .X'
                + (f' (target_sum {target:,.0f})' if isinstance(target, (int, float)) else '')
            )
        elif action == 'log1p' and layer_name == 'X':
            out.append('xcell ran log1p on .X')
        elif action == 'scale' and layer_name == 'X':
            out.append('xcell ran scale on .X (genes centered/scaled)')
        elif action == 'smooth' and result.get('output_layer') == layer_name:
            src = result.get('source_layer') or params.get('source_layer') or 'X'
            out.append(
                f'xcell produced this layer by smoothing {src} over '
                f'{result.get("graph_key", "a kNN graph")}'
            )
    return out
