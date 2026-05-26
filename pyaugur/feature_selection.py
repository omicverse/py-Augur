"""Feature selection functions ported from R Augur."""

import numpy as np
from scipy import sparse
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess_c


def select_random(mat, feature_perc=0.5, rng=None):
    """Random feature selection (port of R select_random).

    Parameters
    ----------
    mat : np.ndarray or sparse matrix, shape (n_features, n_cells)
    feature_perc : float
    rng : numpy.random.Generator, optional

    Returns
    -------
    np.ndarray or sparse matrix with subset of rows
    """
    if feature_perc >= 1.0:
        return mat
    if rng is None:
        rng = np.random.default_rng()
    n_keep = int(mat.shape[0] * feature_perc)
    keep = rng.choice(mat.shape[0], size=n_keep, replace=False)
    keep.sort()
    return mat[keep, :]


def select_variance(mat, var_quantile=0.5, filter_negative_residuals=False):
    """Feature selection by variance (port of R select_variance).

    Fits a loess model of coefficient of variation vs mean expression,
    then filters features by the quantile of residuals.

    Parameters
    ----------
    mat : np.ndarray or sparse matrix, shape (n_features, n_cells)
    var_quantile : float
    filter_negative_residuals : bool

    Returns
    -------
    Filtered matrix (rows = features, cols = cells)
    """
    # Work directly with the input to avoid unnecessary copies
    if sparse.issparse(mat):
        mat_dense = mat.toarray()
    else:
        mat_dense = np.asarray(mat, dtype=np.float64)

    # Row standard deviations (ddof=1 to match R sd())
    sds = np.std(mat_dense, axis=1, ddof=1)
    sds = np.where(np.isnan(sds), 0.0, sds)

    # Remove constant features
    mask = sds > 0
    mat_sub = mat_dense[mask, :]
    del mat_dense  # Free original

    if var_quantile >= 1.0 and not filter_negative_residuals:
        return mat_sub

    # Calculate mean and coefficient of variation
    means = np.mean(mat_sub, axis=1)
    sds_sub = sds[mask]
    cvs = means / sds_sub

    # Clip outliers at 1st and 99th percentiles
    lower = np.percentile(cvs, 1)
    upper = np.percentile(cvs, 99)
    keep = (cvs >= lower) & (cvs <= upper)
    cv0 = cvs[keep]
    mean0 = means[keep]
    del cvs, means  # Free memory

    if np.any(mean0 < 0):
        model = _loess(mean0, cv0, span=0.75)
    else:
        fit1 = _loess(mean0, cv0, span=0.75)
        fit2 = _loess(np.log(mean0), cv0, span=0.75)
        cox = _coxtest(cv0, fit1, fit2)
        if cox["p1"] < cox["p2"]:
            model = fit1
        else:
            model = fit2

    residuals = model["residuals"]

    if filter_negative_residuals:
        keep_genes = residuals > 0
    else:
        threshold = np.percentile(residuals, var_quantile * 100)
        keep_genes = residuals > threshold

    # Apply filter to the variance-filtered matrix
    # mat_sub has rows with non-zero variance; mat_sub[keep] has rows within CV outlier bounds
    # keep_genes selects from the outlier-clipped subset
    if sparse.issparse(mat):
        # For sparse, use original indices
        result_indices = np.where(mask)[0][keep][keep_genes]
        return mat[result_indices, :]
    return mat_sub[keep, :][keep_genes, :]


def _loess(x, y, span=0.75, degree=2):
    """Fast local polynomial regression using statsmodels C implementation.

    Parameters
    ----------
    x : np.ndarray, 1D predictor
    y : np.ndarray, 1D response
    span : float, fraction of points used in local neighborhood
    degree : int, polynomial degree

    Returns
    -------
    dict with 'fitted' and 'residuals'
    """
    n = len(x)
    if n == 0:
        return {"fitted": np.array([]), "residuals": np.array([])}

    # it=2 robustness iterations for best parity with R's loess (it=4)
    result = _lowess_c(y, x, frac=span, it=2, return_sorted=True)
    # result is sorted by x; map back to original order
    order = np.argsort(x)
    fitted = np.empty(n)
    fitted[order] = result[:, 1]

    return {"fitted": fitted, "residuals": y - fitted}


def _coxtest(y, model1, model2):
    """Cox non-nested model comparison test.

    Compares two non-nested regression models by testing whether
    cross-predicted residuals are significantly different from zero.

    Parameters
    ----------
    y : np.ndarray, response variable
    model1 : dict with 'fitted' and 'residuals'
    model2 : dict with 'fitted' and 'residuals'

    Returns
    -------
    dict with 'z1', 'z2', 'p1', 'p2'
    """
    res1 = model1["residuals"]
    res2 = model2["residuals"]

    yhat1 = model1["fitted"]
    yhat2 = model2["fitted"]

    # Model 1 residuals predicted by model 2's fitted values
    z1 = res1 - (y - yhat2)
    # Model 2 residuals predicted by model 1's fitted values
    z2 = res2 - (y - yhat1)

    sd1 = np.std(z1, ddof=1) / np.sqrt(len(z1))
    sd2 = np.std(z2, ddof=1) / np.sqrt(len(z2))

    if sd1 > 0:
        z1_stat = np.mean(z1) / sd1
    else:
        z1_stat = 0.0

    if sd2 > 0:
        z2_stat = np.mean(z2) / sd2
    else:
        z2_stat = 0.0

    # Two-tailed p-values from normal distribution
    from scipy.stats import norm
    p1 = 2 * (1 - norm.cdf(abs(z1_stat)))
    p2 = 2 * (1 - norm.cdf(abs(z2_stat)))

    return {"z1": z1_stat, "z2": z2_stat, "p1": p1, "p2": p2}
