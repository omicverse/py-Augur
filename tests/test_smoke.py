"""Smoke tests for pyaugur - verify imports and basic functionality."""

import numpy as np
import pandas as pd
import pytest


def test_import():
    import pyaugur
    assert hasattr(pyaugur, "calculate_auc")
    assert hasattr(pyaugur, "calculate_differential_prioritization")
    assert hasattr(pyaugur, "select_variance")
    assert hasattr(pyaugur, "select_random")


def test_select_random():
    from pyaugur.feature_selection import select_random

    mat = np.random.randn(100, 50)
    result = select_random(mat, feature_perc=0.5, rng=np.random.default_rng(42))
    assert result.shape[0] == 50
    assert result.shape[1] == 50


def test_select_random_full():
    from pyaugur.feature_selection import select_random

    mat = np.random.randn(100, 50)
    result = select_random(mat, feature_perc=1.0)
    assert result.shape == mat.shape


def test_select_variance():
    from pyaugur.feature_selection import select_variance

    # Create a matrix with varying variance
    rng = np.random.default_rng(42)
    mat = rng.normal(0, 1, (200, 50))
    # Make some rows have higher variance
    mat[0:50, :] *= 5

    result = select_variance(mat, var_quantile=0.5)
    assert result.shape[0] < mat.shape[0]  # Should filter some features
    assert result.shape[1] == mat.shape[1]


def test_calculate_auc_toy():
    """Quick smoke test with a tiny synthetic dataset."""
    from pyaugur import calculate_auc

    rng = np.random.default_rng(42)
    n_genes, n_cells = 100, 60
    expr = rng.normal(0, 1, (n_genes, n_cells))
    meta = pd.DataFrame({
        "cell_type": ["A"] * 30 + ["B"] * 30,
        "label": ["ctrl"] * 15 + ["treat"] * 15 + ["ctrl"] * 15 + ["treat"] * 15,
    })

    result = calculate_auc(
        expr,
        meta=meta,
        n_subsamples=3,
        subsample_size=10,
        folds=3,
        var_quantile=1.0,
        feature_perc=1.0,
        seed=42,
    )

    assert "AUC" in result
    assert "cell_types" in result
    assert len(result["AUC"]) == 2
    assert all(0 <= auc <= 1 for auc in result["AUC"]["auc"])
