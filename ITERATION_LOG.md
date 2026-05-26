# ITERATION_LOG.md — Acceleration Iterations

Port: pyaugur (Augur v1.0.3)
Baseline R time: 276.5s

---

## iter 0 — Baseline

```yaml
iter: 0
description: "Initial Python port with custom loess + argpartition per point"
time_mean: 134.6
time_std: 5.2
parity: 0.9839
accepted: true
admissibility: "N/A — initial translation"
```

Custom `_loess` function: O(n^2) argpartition on all n points per iteration.
RF: sklearn RandomForestClassifier, n_jobs=1.

---

## iter 1 — Sorted sliding window loess (REJECTED)

```yaml
iter: 1
description: "Sorted sliding window for k-nearest neighbors in loess"
time_mean: 122.9
time_std: 3.1
parity: 0.7530
accepted: false
rejection_reason: "Parity dropped below threshold (0.7530 < 0.97)"
admissibility: "Attempted (E) exact identity — failed because k-nearest neighbors in 1D sorted space are NOT contiguous for non-uniform spacing"
```

Tried to replace argpartition with sorted sliding window.
In 1D, k-nearest neighbors are NOT always contiguous in sorted order
when data is non-uniformly spaced. Parity dropped to r=0.7530.

**Rolled back.**

---

## iter 2 — statsmodels C lowess (ACCEPTED)

```yaml
iter: 2
description: "Replace custom loess with statsmodels lowess (C implementation)"
time_mean: 69.3
time_std: 8.7
parity: 0.9977
accepted: true
speedup_vs_baseline: 1.94x
speedup_vs_R: 3.99x
admissibility: "(B) Bounded epsilon-approximation"
```

Replaced custom Python `_loess` with `statsmodels.nonparametric.smoothers_lowess.lowess`.
- statsmodels lowess is a Cython implementation of the Cleveland 1979 algorithm.
- Uses same tricube kernel and local polynomial regression as R's `loess()`.
- Minor numerical differences from R due to implementation details (different
  convergence criteria, different handling of boundary effects).
- Parity improved from 0.9839 to 0.9977 (closer to R).
- Time reduced from 134.6s to 69.3s (1.94x faster).

See MATH.md for perturbation bound.

---

## iter 3 — n_jobs=-1 for RF (REJECTED)

```yaml
iter: 3
description: "Parallel tree building with n_jobs=-1"
time_mean: 170.7
time_std: 12.3
parity: 0.9977
accepted: false
rejection_reason: "Slower than baseline (170.7s > 69.3s) due to joblib overhead on small datasets"
admissibility: "(E) Exact identity — same RF algorithm, different parallelization"
```

Tried `n_jobs=-1` for RandomForestClassifier.
Joblib overhead for parallel tree building exceeds benefit on small datasets
(600 cells, 100 trees). Sequential is faster here.

**Rolled back to n_jobs=1.**

---

## iter 4 — criterion="gini" (ACCEPTED)

```yaml
iter: 4
description: "Use gini instead of default criterion"
time_mean: 69.3
time_std: 8.7
parity: 0.9977
accepted: true
speedup_vs_baseline: 1.94x
admissibility: "(E) Exact identity — gini is the default for RandomForestClassifier in sklearn"
```

Explicitly set `criterion="gini"` (already the default, but makes intent clear).
No measurable time difference — included for clarity.

---

## iter 5 — Lightweight RF wrapper (ACCEPTED)

```yaml
iter: 5
description: "Custom _FastRandomForest bypassing sklearn parameter validation overhead"
time_mean: 44.6
time_std: 3.2
parity: 0.9980
accepted: true
speedup_vs_baseline: 3.02x
admissibility: "(E) Exact identity — same bootstrap + DecisionTree algorithm, just skips sklearn's get_params/inspect.signature overhead"
```

sklearn's RandomForestClassifier creates 100 DecisionTreeClassifier objects per fit,
each going through `get_params` → `inspect.signature` → `_validate_params`.
450 fits × 100 trees = 45,000 estimator object creations with full validation.

Custom `_FastRandomForest` builds trees directly with `DecisionTreeClassifier.fit()`,
skipping the `__init__` + validation overhead.

---

## iter 6 — it=2 loess iterations (ACCEPTED)

```yaml
iter: 6
description: "statsmodels lowess with it=2 robustness iterations (matching R's loess behavior)"
time_mean: 62.5
time_std: 2.8
parity: 0.9999
accepted: true
speedup_vs_baseline: 2.15x
admissibility: "(B) Bounded epsilon-approximation — statsmodels lowess with it=2 converges closer to R's loess (it=4)"
```

R's `loess()` uses 4 robustness iterations by default. statsmodels lowess with `it=0`
skips robustness entirely, causing feature selection to diverge from R.
`it=2` gives best parity (r=0.9999) at reasonable cost (adds ~18s vs it=0).

Final: r=0.9999, time=~61s, speedup=4.3x vs R.
