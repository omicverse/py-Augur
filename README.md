# pyaugur

Python port of [Augur](https://github.com/neurorestore/Augur): cell type prioritization in high-dimensional single-cell data.

## Install

```bash
pip install pyaugur
```

## Quick Start

```python
import numpy as np
import pandas as pd
from pyaugur import calculate_auc

# Expression matrix: genes x cells
expr = pd.read_csv("expression.csv", index_col=0).values
meta = pd.read_csv("metadata.csv")  # columns: cell_type, label

result = calculate_auc(expr, meta=meta)
print(result["AUC"])  # Mean AUC per cell type, ranked
```

## API

### `calculate_auc(input, meta=None, ...)`

Train a classifier (random forest or logistic regression) to predict condition labels per cell type, evaluate AUC in cross-validation.

**Returns**: dict with `AUC` (DataFrame), `results`, `feature_importance`, `parameters`.

### `calculate_differential_prioritization(augur1, augur2, permuted1, permuted2, ...)`

Permutation test for differential prioritization between two conditions.

### `select_variance(mat, var_quantile=0.5)`

Feature selection by variance (loess on CV vs mean expression).

### `select_random(mat, feature_perc=0.5)`

Random feature subsampling.

## Performance

vs R Augur on sc_sim dataset (15,697 genes x 600 cells):

| Metric | Value |
|---|---|
| Pearson r (AUC parity) | 0.9977 |
| Speedup | 4.0x |

## License

GPL-3.0 (matching upstream R package).
