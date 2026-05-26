# DISCOVERY.md — Dependency Reuse Audit

## Target Package
- **R Package**: Augur v1.0.3
- **Description**: Cell type prioritization in high-dimensional single-cell data
- **License**: GPL-3
- **Repository**: https://github.com/neurorestore/Augur

## Existing py- Mirror Check
- **py-augur exists in omicverse?**: No — this is a new port.
- **Checked via**: Manual search (no `gh` CLI configured for org query).

## R Dependency Audit

| R Dependency | Python Equivalent | Reuse Decision | Notes |
|---|---|---|---|
| randomForest | scikit-learn (RandomForestClassifier) | **hard dep** | Direct API mapping |
| glmnet | scikit-learn (LogisticRegression with L1) | **hard dep** | Simplified: no CV for penalty auto-tuning |
| rsample | scikit-learn (StratifiedKFold) | **hard dep** | Built-in cross-validation |
| yardstick | scikit-learn (metrics) | **hard dep** | roc_auc_score, accuracy |
| recipes | (not needed) | **skip** | sklearn handles preprocessing natively |
| parsnip | (not needed) | **skip** | sklearn IS the model API |
| dplyr | pandas | **hard dep** | Data manipulation |
| tidyr | pandas | **hard dep** | Data reshaping |
| tibble | pandas | **hard dep** | DataFrame |
| magrittr | (not needed) | **skip** | Python has native chaining |
| purrr | (not needed) | **skip** | Python list comprehensions |
| Matrix | scipy.sparse | **hard dep** | Sparse matrix support |
| sparseMatrixStats | numpy | **hard dep** | Row/column statistics |
| lmtest | statsmodels | **hard dep** | Cox test for loess comparison |
| pbmcapply | (sequential loop) | **skip** | Single-threaded for stability |
| ggplot2 | matplotlib | **optional dep** | Plotting |
| ggrepel | matplotlib | **optional dep** | Label repulsion |
| pals | matplotlib | **optional dep** | Color palettes |
| scales | matplotlib | **optional dep** | Axis scaling |
| viridis | matplotlib | **optional dep** | Color maps |
| Seurat | anndata | **optional dep** | Input format support |
| monocle3 | (not supported) | **skip** | Future work |
| SingleCellExperiment | anndata | **optional dep** | Input format support |

## Ecosystem Savings
- No existing py- mirrors to reuse (new port).
- scikit-learn, pandas, scipy, statsmodels cover all R dependencies.
- Estimated LOC saved by reusing scikit-learn RF/LR: ~500 lines vs implementing from scratch.
