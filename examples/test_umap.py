"""Test UMAP visualization for py-Augur."""

import numpy as np
import pandas as pd
from pyaugur import calculate_auc, plot_umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create synthetic data with UMAP coordinates
np.random.seed(42)
n_cells = 300
n_genes = 500

# Generate expression matrix
X = np.random.randn(n_genes, n_cells)

# Create cell type labels
cell_types = ['CellTypeA', 'CellTypeB', 'CellTypeC']
cell_type_labels = np.repeat(cell_types, n_cells // 3)

# Create condition labels (perturbation vs control)
labels = np.random.choice(['control', 'stimulated'], size=n_cells)

# Create UMAP coordinates
umap_coords = np.column_stack([
    np.random.randn(n_cells) * 2,
    np.random.randn(n_cells) * 2
])

# Add some structure to UMAP based on cell types
umap_coords[:100, 0] += 3
umap_coords[100:200, 1] += 3
umap_coords[200:, 0] -= 3
umap_coords[200:, 1] -= 3

# Create AnnData object
import anndata
adata = anndata.AnnData(X=X.T)
adata.obs['cell_type'] = cell_type_labels
adata.obs['label'] = labels
adata.obsm['X_umap'] = umap_coords

# Run Augur
print("Running calculate_auc...")
result = calculate_auc(adata, n_subsamples=10)

# Print AUC results
print("\nAUC results:")
print(result['AUC'])

# Generate UMAP plots
print("\nGenerating UMAP plots...")

# Plot 1: Default mode (raw AUC values)
fig1, ax1 = plt.subplots(figsize=(10, 8))
plot_umap(adata, result, mode="default", top_n=3, ax=ax1)
fig1.savefig("examples/test_umap_default.png", dpi=150, bbox_inches='tight')
print("Saved: examples/test_umap_default.png")

# Plot 2: Rank mode (relative rank percentage)
fig2, ax2 = plt.subplots(figsize=(10, 8))
plot_umap(adata, result, mode="rank", top_n=3, palette="viridis", ax=ax2)
fig2.savefig("examples/test_umap_rank.png", dpi=150, bbox_inches='tight')
print("Saved: examples/test_umap_rank.png")

# Plot 3: Different palette (plasma)
fig3, ax3 = plt.subplots(figsize=(10, 8))
plot_umap(adata, result, mode="default", top_n=3, palette="plasma", ax=ax3)
fig3.savefig("examples/test_umap_plasma.png", dpi=150, bbox_inches='tight')
print("Saved: examples/test_umap_plasma.png")

print("\nDone! Generated 3 UMAP visualizations.")
