"""Parity test: Python AUC must match R reference across multiple metrics."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(TESTS_DIR, "reference_output.json")
EXPR_FILE = os.path.join(TESTS_DIR, "sc_sim_expr.csv")
META_FILE = os.path.join(TESTS_DIR, "sc_sim_meta.csv")

# Parity gates
PEARSON_THRESHOLD = 0.95
SPEARMAN_THRESHOLD = 1.0


def _load_reference():
    """Load R reference AUC values."""
    with open(REFERENCE_FILE, "r") as f:
        ref = json.load(f)
    return ref


def _load_test_data():
    """Load the sc_sim expression matrix and metadata."""
    expr = pd.read_csv(EXPR_FILE, index_col=0)
    meta = pd.read_csv(META_FILE, index_col=0)
    expr_mat = expr.values.astype(np.float64)
    return expr_mat, meta


@pytest.mark.skipif(
    not os.path.exists(REFERENCE_FILE),
    reason="R reference output not found. Run r_reference_driver.R first.",
)
class TestParity:
    """Parity tests against R reference output."""

    def test_auc_parity(self):
        """Parity: Pearson r + Spearman rho on AUC values."""
        ref = _load_reference()
        ref_aucs = {r["cell_type"]: r["auc"] for r in ref["AUC"]}
        expr_mat, meta = _load_test_data()

        from pyaugur import calculate_auc

        start = time.time()
        result = calculate_auc(
            expr_mat, meta=meta, n_subsamples=50, subsample_size=20,
            folds=3, var_quantile=0.5, feature_perc=0.5, seed=42,
        )
        elapsed = time.time() - start

        py_aucs = result["AUC"]
        ct = sorted(set(ref_aucs.keys()) & set(py_aucs["cell_type"].tolist()))
        assert len(ct) >= 2, f"Need >= 2 common cell types, got {ct}"

        rv = np.array([ref_aucs[c] for c in ct])
        pv = py_aucs.set_index("cell_type").loc[ct, "auc"].values

        r_p, _ = pearsonr(rv, pv)
        r_s, _ = spearmanr(rv, pv)

        print(f"\n{'='*50}")
        print(f"{'Metric':<20} {'Value':>10} {'Gate':>10} {'Pass?':>6}")
        print(f"{'='*50}")
        print(f"{'Pearson r':<20} {r_p:>10.4f} {'>='+str(PEARSON_THRESHOLD):>10} {'YES' if r_p >= PEARSON_THRESHOLD else 'NO':>6}")
        print(f"{'Spearman rho':<20} {r_s:>10.4f} {'='+str(SPEARMAN_THRESHOLD):>10} {'YES' if r_s >= SPEARMAN_THRESHOLD else 'NO':>6}")
        print(f"{'='*50}")
        print(f"{'Time':<20} {elapsed:>9.1f}s")
        print(f"\nPer cell type:")
        print(f"{'Cell Type':<12} {'R':>8} {'Python':>8} {'Diff':>8}")
        print(f"{'-'*40}")
        for c, r, p in zip(ct, rv, pv):
            print(f"{c:<12} {r:>8.4f} {p:>8.4f} {p-r:>+8.4f}")

        assert r_p >= PEARSON_THRESHOLD, f"Pearson r={r_p:.4f} < {PEARSON_THRESHOLD}"
        assert r_s >= SPEARMAN_THRESHOLD, f"Spearman rho={r_s:.4f} < {SPEARMAN_THRESHOLD}"

    def test_auc_ranking_preserved(self):
        """Cell type ranking must be identical between R and Python."""
        ref = _load_reference()
        ref_order = [r["cell_type"] for r in ref["AUC"]]
        expr_mat, meta = _load_test_data()

        from pyaugur import calculate_auc
        result = calculate_auc(
            expr_mat, meta=meta, n_subsamples=50, subsample_size=20,
            folds=3, var_quantile=0.5, feature_perc=0.5, seed=42,
        )

        py_order = result["AUC"]["cell_type"].tolist()
        assert ref_order == py_order, f"Ranking mismatch. R: {ref_order}, Python: {py_order}"

    def test_timing_vs_r(self):
        """Python must be faster than R."""
        ref_timing_file = os.path.join(TESTS_DIR, "r_timing.json")
        if not os.path.exists(ref_timing_file):
            pytest.skip("R timing not available")

        with open(ref_timing_file) as f:
            r_timing = json.load(f)

        expr_mat, meta = _load_test_data()
        from pyaugur import calculate_auc

        start = time.time()
        calculate_auc(
            expr_mat, meta=meta, n_subsamples=50, subsample_size=20,
            folds=3, var_quantile=0.5, feature_perc=0.5, seed=42,
        )
        elapsed = time.time() - start

        r_time = r_timing["elapsed_seconds"]
        print(f"\nR: {r_time:.1f}s | Python: {elapsed:.1f}s | Speedup: {r_time/elapsed:.1f}x")
        assert elapsed < r_time, f"Python ({elapsed:.1f}s) slower than R ({r_time:.1f}s)"
