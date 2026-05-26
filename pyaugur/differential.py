"""Differential prioritization test ported from R Augur."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_differential_prioritization(
    augur1,
    augur2,
    permuted1,
    permuted2,
    n_subsamples=50,
    n_permutations=1000,
):
    """Statistical test for differential prioritization.

    Port of R Augur's calculate_differential_prioritization.

    Parameters
    ----------
    augur1 : dict
        Augur results from condition 1 (calculate_auc output).
    augur2 : dict
        Augur results from condition 2.
    permuted1 : dict
        Permuted Augur results from condition 1 (augur_mode='permute').
    permuted2 : dict
        Permuted Augur results from condition 2.
    n_subsamples : int
        Number of subsamples to pool for mean AUC.
    n_permutations : int
        Number of permutations for null distribution.

    Returns
    -------
    pd.DataFrame
        Results with columns: cell_type, auc.x, auc.y, delta_auc, b, m, z, pval, padj.
    """
    obs1 = augur1["AUC"].copy()
    obs2 = augur2["AUC"].copy()

    permuted_res1 = permuted1["results"].copy()
    permuted_res2 = permuted2["results"].copy()

    # Calculate interval size
    n_intervals = permuted_res1["subsample_idx"].max() // 50

    # Average across folds per subsample
    perm_auc1 = (
        permuted_res1[permuted_res1["metric"] == "roc_auc"]
        .groupby(["cell_type", "subsample_idx"])["estimate"]
        .mean()
        .reset_index()
    )
    perm_auc2 = (
        permuted_res2[permuted_res2["metric"] == "roc_auc"]
        .groupby(["cell_type", "subsample_idx"])["estimate"]
        .mean()
        .reset_index()
    )

    # Draw mean AUCs for permutations
    rnd1 = _draw_mean_aucs(perm_auc1, n_permutations, n_intervals)
    rnd2 = _draw_mean_aucs(perm_auc2, n_permutations, n_intervals)

    # Observed delta-AUC
    delta = obs1.merge(obs2, on="cell_type", suffixes=(".x", ".y"))
    delta["delta_auc"] = delta["auc.y"] - delta["auc.x"]

    # Permuted delta-AUCs
    rnd = rnd1.merge(rnd2, on=["cell_type", "permutation_idx"], suffixes=(".x", ".y"))
    rnd["delta_rnd"] = rnd["mean.y"] - rnd["mean.x"]

    # Calculate p-values
    pvals_list = []
    for ct in delta["cell_type"].unique():
        ct_delta = delta[delta["cell_type"] == ct]
        ct_rnd = rnd[rnd["cell_type"] == ct]

        delta_auc = ct_delta["delta_auc"].values[0]
        delta_rnd = ct_rnd["delta_rnd"].values

        b = np.sum(delta_rnd >= delta_auc)
        m = len(delta_rnd)

        z = (delta_auc - np.mean(delta_rnd)) / np.std(delta_rnd) if np.std(delta_rnd) > 0 else 0.0

        pval = min(
            2 * (b + 1) / (m + 1),
            2 * (m - b + 1) / (m + 1),
        )

        pvals_list.append({
            "cell_type": ct,
            "b": b,
            "m": m,
            "z": z,
            "pval": pval,
        })

    pvals = pd.DataFrame(pvals_list)

    # BH correction
    pvals["padj"] = _bh_correction(pvals["pval"].values)

    # Merge results
    result = delta[["cell_type", "auc.x", "auc.y", "delta_auc"]].merge(
        pvals, on="cell_type"
    )
    result = result.dropna(subset=["pval"]).reset_index(drop=True)

    return result


def _draw_mean_aucs(permuted_aucs, n_permutations, n_intervals):
    """Draw mean AUCs from permuted results."""
    rng = np.random.default_rng(42)
    results = []

    for perm_idx in range(1, n_permutations + 1):
        rng_inner = np.random.default_rng(perm_idx)
        perm_copy = permuted_aucs.copy()
        perm_copy["bin"] = pd.cut(
            perm_copy["subsample_idx"], bins=n_intervals, labels=False
        ) + 1

        # For each cell type, shuffle bins and take bin 1
        for ct in perm_copy["cell_type"].unique():
            ct_mask = perm_copy["cell_type"] == ct
            ct_data = perm_copy[ct_mask].copy()
            bins = ct_data["bin"].values.copy()
            rng_inner.shuffle(bins)
            ct_data["bin"] = bins
            ct_bin1 = ct_data[ct_data["bin"] == 1]

            if len(ct_bin1) > 0:
                results.append({
                    "cell_type": ct,
                    "permutation_idx": perm_idx,
                    "mean": ct_bin1["estimate"].mean(),
                    "sd": ct_bin1["estimate"].std(),
                })

    return pd.DataFrame(results)


def _bh_correction(pvalues):
    """Benjamini-Hochberg p-value correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])

    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)

    adjusted = pvalues * n / ranked
    # Ensure monotonicity
    adjusted_sorted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = adjusted_sorted
    return np.clip(result, 0, 1)
