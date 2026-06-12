"""Regenerate all test plot images with the updated plotting module."""

import numpy as np
import pandas as pd
import anndata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyaugur import (
    calculate_auc,
    plot_lollipop, plot_umap, plot_important_features,
    plot_differential_prioritization,
)

np.random.seed(42)

# ── synthetic data ──
n_cells = 300
n_genes = 500
X = np.random.randn(n_genes, n_cells)

cell_types = ['CellTypeA', 'CellTypeB', 'CellTypeC']
cell_type_labels = np.repeat(cell_types, n_cells // 3)
labels = np.random.choice(['control', 'stimulated'], size=n_cells)

umap_coords = np.column_stack([
    np.random.randn(n_cells) * 2,
    np.random.randn(n_cells) * 2,
])
umap_coords[:100, 0] += 3
umap_coords[100:200, 1] += 3
umap_coords[200:, 0] -= 3
umap_coords[200:, 1] -= 3

adata = anndata.AnnData(X=X.T)
adata.obs['cell_type'] = cell_type_labels
adata.obs['label'] = labels
adata.obsm['X_umap'] = umap_coords

# ── run augur ──
print("Running calculate_auc ...")
result = calculate_auc(adata, n_subsamples=10)
print(result['AUC'])

# ── 1. lollipop ──
print("plot_lollipop ...")
fig, ax = plot_lollipop(result)
fig.savefig("examples/test_lollipop.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 2. important features ──
print("plot_important_features ...")
fig, ax = plot_important_features(result)
fig.savefig("examples/test_important_features.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 3. umap default (frameon='small', omicverse style) ──
print("plot_umap default ...")
fig, ax = plot_umap(adata, result, mode="default", top_n=3)
fig.savefig("examples/test_umap_default.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 4. umap rank ──
print("plot_umap rank ...")
fig, ax = plot_umap(adata, result, mode="rank", top_n=3, palette="viridis")
fig.savefig("examples/test_umap_rank.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 5. umap plasma ──
print("plot_umap plasma ...")
fig, ax = plot_umap(adata, result, mode="default", top_n=3, palette="plasma")
fig.savefig("examples/test_umap_plasma.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 6. differential prioritization ──
print("plot_differential_prioritization ...")
dp_df = pd.DataFrame({
    'cell_type': cell_types,
    'auc.x': [0.62, 0.58, 0.71],
    'auc.y': [0.75, 0.55, 0.69],
    'delta_auc': [0.13, -0.03, -0.02],
    'b': [3, 48, 45],
    'm': [50, 50, 50],
    'z': [2.8, -1.5, -0.9],
    'pval': [0.005, 0.12, 0.35],
    'padj': [0.015, 0.18, 0.35],
})
fig, ax = plot_differential_prioritization(dp_df, top_n=2)
fig.savefig("examples/test_differential_prioritization.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("\nDone - all plots saved to examples/")
