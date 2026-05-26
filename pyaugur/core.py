"""Core calculate_auc function ported from R Augur."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
import warnings

from .feature_selection import select_variance, select_random


class _FastRandomForest:
    """Lightweight RF that bypasses sklearn's parameter validation overhead."""

    def __init__(self, n_estimators=100, max_features=2, min_samples_split=2,
                 random_state=1, mode="classification"):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = check_random_state(random_state)
        self.mode = mode
        self.trees = []
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            n = X.shape[0]
            indices = self.rng.choice(n, size=n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            if self.mode == "classification":
                tree = DecisionTreeClassifier(
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    random_state=self.rng,
                )
            else:
                tree = DecisionTreeRegressor(
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    random_state=self.rng,
                )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X):
        if self.mode == "classification":
            votes = np.array([tree.predict(X) for tree in self.trees])
            # Majority vote
            result = np.empty(X.shape[0], dtype=self.classes_.dtype)
            for i in range(X.shape[0]):
                counts = np.bincount([np.where(self.classes_ == v)[0][0] for v in votes[:, i]])
                result[i] = self.classes_[np.argmax(counts)]
            return result
        else:
            preds = np.array([tree.predict(X) for tree in self.trees])
            return np.mean(preds, axis=0)

    def predict_proba(self, X):
        # Each tree may have different classes; align to self.classes_
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            tree_classes = tree.classes_
            for i, cls in enumerate(tree_classes):
                j = np.where(self.classes_ == cls)[0][0]
                proba[:, j] += tree_proba[:, i]
        proba /= self.n_estimators
        return proba


def calculate_auc(
    input,
    meta=None,
    label_col="label",
    cell_type_col="cell_type",
    n_subsamples=50,
    subsample_size=20,
    folds=3,
    min_cells=None,
    var_quantile=0.5,
    feature_perc=0.5,
    n_threads=4,
    augur_mode="default",
    classifier="rf",
    rf_params=None,
    lr_params=None,
    seed=42,
):
    """Prioritize cell types involved in a biological process.

    Port of R Augur's calculate_auc. Trains a machine learning classifier
    (random forest or logistic regression) to predict condition labels per
    cell type, evaluates AUC in cross-validation.

    Parameters
    ----------
    input : AnnData, pd.DataFrame, np.ndarray, or sparse matrix
        Gene expression matrix (genes x cells) or AnnData object.
    meta : pd.DataFrame, optional
        Metadata with cell_type and label columns. Required if input is matrix.
    label_col : str
        Column in meta containing condition labels.
    cell_type_col : str
        Column in meta containing cell type labels.
    n_subsamples : int
        Number of random subsamples per cell type.
    subsample_size : int
        Cells per condition per subsample.
    folds : int
        Number of CV folds.
    min_cells : int, optional
        Minimum cells per cell type. Defaults to subsample_size.
    var_quantile : float
        Quantile for variance-based feature selection.
    feature_perc : float
        Proportion of features randomly selected.
    n_threads : int
        Number of threads (unused, kept for API compatibility).
    augur_mode : str
        'default', 'velocity', or 'permute'.
    classifier : str
        'rf' (random forest) or 'lr' (logistic regression).
    rf_params : dict, optional
        Random forest parameters: trees, mtry, min_n, importance.
    lr_params : dict, optional
        Logistic regression parameters: mixture, penalty.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Augur results with keys: 'X', 'y', 'cell_types', 'parameters',
        'results', 'feature_importance', 'AUC' (or 'CCC' for regression).
    """
    # Default parameters
    if rf_params is None:
        rf_params = {"trees": 100, "mtry": 2, "min_n": None, "importance": "accuracy"}
    if lr_params is None:
        lr_params = {"mixture": 1.0, "penalty": "auto"}
    if min_cells is None:
        min_cells = subsample_size

    # Handle augur_mode
    if augur_mode == "velocity":
        feature_perc = 1.0
        var_quantile = 1.0
    elif augur_mode == "permute" and n_subsamples < 100:
        n_subsamples = 500

    # Extract expression matrix and metadata
    expr, cell_types, labels = _extract_input(input, meta, label_col, cell_type_col)

    # Validate
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Need at least 2 labels, got {len(np.unique(labels))}")
    if np.any(pd.isna(labels)):
        raise ValueError("Labels contain missing values")
    if np.any(pd.isna(cell_types)):
        raise ValueError("Cell types contain missing values")

    # Detect mode
    is_numeric_labels = _is_numeric(labels)
    if is_numeric_labels:
        mode = "regression"
        multiclass = False
    else:
        mode = "classification"
        multiclass = len(np.unique(labels)) > 2
        labels = labels.astype(str)

    n_iter = max(n_subsamples, 1)
    unique_types = np.unique(cell_types)

    # Iterate over cell types
    all_results = []
    all_importances = []

    for ct in unique_types:
        ct_mask = cell_types == ct
        y_ct = labels[ct_mask]
        X_ct = expr[:, ct_mask]

        # Check minimum cells
        if mode == "classification":
            min_count = min(np.sum(y_ct == lab) for lab in np.unique(y_ct))
            if min_count < min_cells:
                warnings.warn(
                    f"Skipping cell type {ct}: min cells ({min_count}) < {min_cells}"
                )
                continue
        else:
            if len(y_ct) < min_cells:
                warnings.warn(
                    f"Skipping cell type {ct}: total cells ({len(y_ct)}) < {min_cells}"
                )
                continue

        # Variance-based feature selection
        if X_ct.shape[0] >= 1000 and var_quantile < 1.0:
            X_ct = select_variance(X_ct, var_quantile, filter_negative_residuals=False)

        # Per-subsample iteration
        for subsample_idx in range(1, n_iter + 1):
            rng = np.random.default_rng(seed)

            # Optionally permute labels
            if augur_mode == "permute":
                perm_rng = np.random.default_rng(subsample_idx)
                y_ct_iter = perm_rng.permutation(y_ct)
            else:
                y_ct_iter = y_ct.copy()

            # Subsample cells
            if n_subsamples < 1:
                # No subsampling
                if X_ct.shape[0] >= 1000 and feature_perc < 1.0:
                    X_sub = select_random(X_ct, feature_perc, rng=rng)
                else:
                    X_sub = X_ct
                y_sub = y_ct_iter
            else:
                # Stratified subsample
                subsample_idxs = _stratified_subsample(
                    y_ct_iter, subsample_size, mode, rng
                )
                y_sub = y_ct_iter[subsample_idxs]

                # Random feature selection
                if X_ct.shape[0] >= 1000 and feature_perc < 1.0:
                    X_feat = select_random(X_ct, feature_perc, rng=rng)
                else:
                    X_feat = X_ct

                # Subset cells and remove zero-variance features
                X_sub = X_feat[:, subsample_idxs]
                if sparse.issparse(X_sub):
                    X_sub = X_sub.toarray()

                # Remove zero-variance features (colVars > 0)
                col_vars = np.var(X_sub, axis=0, ddof=1)
                nonzero_var = col_vars > 0
                X_sub = X_sub[:, nonzero_var]

            # Transpose: R uses genes x cells, sklearn uses cells x genes
            if sparse.issparse(X_sub):
                X_sub = X_sub.toarray()
            X_df = X_sub.T  # cells x genes

            # K-fold CV
            if mode == "classification":
                skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                      random_state=subsample_idx)
                fold_splits = list(skf.split(X_df, y_sub))
            else:
                kf = KFold(n_splits=folds, shuffle=True,
                           random_state=subsample_idx)
                fold_splits = list(kf.split(X_df))

            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                X_train = X_df[train_idx]
                X_test = X_df[test_idx]
                y_train = y_sub[train_idx]
                y_test = y_sub[test_idx]

                # Train model
                np.random.seed(1)  # Match R's set.seed(1) in seeded_rf
                if classifier == "rf":
                    model = _FastRandomForest(
                        n_estimators=rf_params["trees"],
                        max_features=rf_params["mtry"],
                        min_samples_split=rf_params.get("min_n", 2) or 2,
                        random_state=1,
                        mode=mode,
                    )
                    model.fit(X_train, y_train)
                elif classifier == "lr":
                    penalty_val = lr_params.get("penalty", 1.0)
                    if isinstance(penalty_val, str) or penalty_val is None:
                        penalty_val = 1.0  # default
                    model = LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        C=1.0 / penalty_val,
                        max_iter=1000,
                        random_state=1,
                    )
                    model.fit(X_train, y_train)
                else:
                    raise ValueError(f"Invalid classifier: {classifier}")

                # Predict
                y_pred = model.predict(X_test)

                if mode == "classification":
                    y_prob = model.predict_proba(X_test)
                    classes = model.classes_

                    # Compute metrics per fold
                    result = _compute_classification_metrics(
                        y_test, y_pred, y_prob, classes, multiclass
                    )
                    for metric_name, estimate in result.items():
                        all_results.append({
                            "cell_type": ct,
                            "subsample_idx": subsample_idx,
                            "fold": fold_idx + 1,
                            "metric": metric_name,
                            "estimator": "binary" if not multiclass else "macro",
                            "estimate": estimate,
                        })
                else:
                    # Regression metrics
                    from scipy.stats import pearsonr
                    r_val = pearsonr(y_test.astype(float), y_pred.astype(float))[0]
                    all_results.append({
                        "cell_type": ct,
                        "subsample_idx": subsample_idx,
                        "fold": fold_idx + 1,
                        "metric": "ccc",
                        "estimator": "standard",
                        "estimate": r_val,
                    })

                # Feature importance
                if classifier == "rf":
                    importances = model.feature_importances_
                    # Need original gene names - use indices
                    n_genes = X_df.shape[1]
                    for g_idx in range(n_genes):
                        all_importances.append({
                            "cell_type": ct,
                            "subsample_idx": subsample_idx,
                            "fold": fold_idx + 1,
                            "gene": f"gene_{g_idx}",
                            "importance": importances[g_idx],
                        })

    # Build results DataFrames
    results_df = pd.DataFrame(all_results)
    importances_df = pd.DataFrame(all_importances)

    if len(results_df) == 0:
        raise ValueError(f"No cell type had at least {min_cells} cells in all conditions")

    # Summarize AUC per cell type
    if mode == "classification":
        auc_rows = results_df[results_df["metric"] == "roc_auc"]
        # First average across folds per subsample
        auc_by_sub = auc_rows.groupby(["cell_type", "subsample_idx"])["estimate"].mean()
        # Then average across subsamples
        auc_summary = auc_by_sub.groupby("cell_type").mean().reset_index()
        auc_summary.columns = ["cell_type", "auc"]
        auc_summary = auc_summary.sort_values("auc", ascending=False).reset_index(drop=True)
    else:
        auc_rows = results_df[results_df["metric"] == "ccc"]
        auc_by_sub = auc_rows.groupby(["cell_type", "subsample_idx"])["estimate"].mean()
        auc_summary = auc_by_sub.groupby("cell_type").mean().reset_index()
        auc_summary.columns = ["cell_type", "ccc"]
        auc_summary = auc_summary.sort_values("ccc", ascending=False).reset_index(drop=True)

    # Build output
    obj = {
        "X": expr,
        "y": labels,
        "cell_types": cell_types,
        "parameters": {
            "n_subsamples": n_subsamples,
            "subsample_size": subsample_size,
            "folds": folds,
            "min_cells": min_cells,
            "var_quantile": var_quantile,
            "feature_perc": feature_perc,
            "n_threads": n_threads,
            "classifier": classifier,
            "rf_params": rf_params if classifier == "rf" else None,
            "lr_params": lr_params if classifier == "lr" else None,
        },
        "results": results_df,
        "feature_importance": importances_df,
    }

    if mode == "classification":
        obj["AUC"] = auc_summary
    else:
        obj["CCC"] = auc_summary

    return obj


def _extract_input(input, meta, label_col, cell_type_col):
    """Extract expression matrix, cell types, and labels from various input types."""
    # Check for AnnData by class name to avoid importing anndata (and torch)
    if hasattr(input, 'obs') and hasattr(input, 'X'):
        meta = input.obs
        expr = input.X.T  # Transpose to genes x cells
        if sparse.issparse(expr):
            expr = expr.tocsr()
        cell_types = meta[cell_type_col].values
        labels = meta[label_col].values
        return expr, cell_types, labels

    if isinstance(input, pd.DataFrame):
        # DataFrame: rows = genes, columns = cells
        expr = input.values
    elif sparse.issparse(input):
        expr = input
    elif isinstance(input, np.ndarray):
        expr = input
    else:
        raise ValueError("Unsupported input type")

    if meta is None:
        raise ValueError("Must provide metadata if not supplying AnnData")

    cell_types = meta[cell_type_col].values
    labels = meta[label_col].values
    return expr, cell_types, labels


def _is_numeric(arr):
    """Check if array contains numeric values."""
    try:
        np.asarray(arr, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def _stratified_subsample(y, subsample_size, mode, rng):
    """Stratified subsampling matching R's behavior.

    For classification: sample subsample_size cells per label group.
    For regression: sample subsample_size cells total.
    """
    indices = np.arange(len(y))

    if mode == "classification":
        unique_labels = np.unique(y)
        selected = []
        for lab in unique_labels:
            lab_mask = y == lab
            lab_indices = indices[lab_mask]
            n = min(subsample_size, len(lab_indices))
            chosen = rng.choice(lab_indices, size=n, replace=False)
            selected.append(chosen)
        return np.concatenate(selected)
    else:
        n = min(subsample_size, len(indices))
        return rng.choice(indices, size=n, replace=False)


def _compute_classification_metrics(y_true, y_pred, y_prob, classes, multiclass):
    """Compute classification metrics matching R yardstick's metric_set."""
    metrics = {}

    # ROC AUC
    try:
        if multiclass:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=classes
            )
        else:
            # Find index of positive class
            pos_idx = np.where(classes == np.unique(y_true)[1])[0][0]
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob[:, pos_idx], labels=classes
            )
    except ValueError:
        metrics["roc_auc"] = np.nan

    # Accuracy
    metrics["accuracy"] = np.mean(y_true == y_pred)

    return metrics
