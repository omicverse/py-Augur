#!/usr/bin/env Rscript
# R reference driver for Augur parity testing
# Produces reference_output.json from sc_sim.rda

library(jsonlite)
library(dplyr)
library(purrr)
library(tibble)
library(magrittr)
library(Matrix)
library(sparseMatrixStats)
library(parsnip)
library(recipes)
library(rsample)
library(yardstick)
library(pbmcapply)
library(lmtest)
library(randomForest)
library(tidyselect)
library(rlang)
library(scales)
library(tidyr)
library(glmnet)

# Fix masking: ensure magrittr::extract is available for sparse matrices
extract <- magrittr::extract

# Source Augur R code
augur_dir <- "D:/test/Augur-master/Augur-master"

# Source all R files
r_files <- list.files(file.path(augur_dir, "R"), pattern = "\\.R$", full.names = TRUE)
for (f in r_files) {
  tryCatch(source(f), error = function(e) message("Skipping ", f, ": ", e$message))
}

# Load the sc_sim data
load(file.path(augur_dir, "data", "sc_sim.rda"))

# Extract expression matrix and metadata from Seurat object
cat("sc_sim class:", class(sc_sim), "\n")

# Get expression matrix (genes x cells) and metadata
library(Matrix)
# Try different ways to get the expression data
tryCatch({
  expr <- Seurat::GetAssayData(sc_sim)
}, error = function(e) {
  # Fallback: try to access the data slot directly
  tryCatch({
    expr <- sc_sim@assays$RNA@data
  }, error = function(e2) {
    expr <- sc_sim@assays[[1]]@data
  })
})
meta <- sc_sim@meta.data

cat("Expression dims:", dim(expr), "\n")
cat("Meta dims:", dim(meta), "\n")
cat("Cell types:", unique(meta$cell_type), "\n")
cat("Labels:", unique(meta$label), "\n")

# Run calculate_auc with default parameters
set.seed(42)
result <- calculate_auc(
  expr,
  meta = meta,
  label_col = "label",
  cell_type_col = "cell_type",
  n_subsamples = 50,
  subsample_size = 20,
  folds = 3,
  var_quantile = 0.5,
  feature_perc = 0.5,
  n_threads = 1,
  show_progress = FALSE,
  classifier = "rf"
)

cat("\n=== AUC Results ===\n")
print(result$AUC)

# Save results as JSON
output <- list(
  AUC = result$AUC,
  parameters = result$parameters
)

json_path <- file.path(getwd(), "tests", "reference_output.json")
write_json(output, json_path, auto_unbox = TRUE, pretty = TRUE)
cat("\nReference output saved to:", json_path, "\n")

# Also save timing info
timing <- system.time({
  calculate_auc(
    expr,
    meta = meta,
    label_col = "label",
    cell_type_col = "cell_type",
    n_subsamples = 50,
    subsample_size = 20,
    folds = 3,
    var_quantile = 0.5,
    feature_perc = 0.5,
    n_threads = 1,
    show_progress = FALSE,
    classifier = "rf"
  )
})

# Export expression matrix and metadata for Python
cat("\n=== Exporting data for Python ===\n")
# Save as CSV for Python
# Expression matrix: genes x cells (rows x cols)
expr_dense <- as.matrix(expr)
write.csv(expr_dense, file.path(getwd(), "tests", "sc_sim_expr.csv"))
# Metadata
write.csv(meta, file.path(getwd(), "tests", "sc_sim_meta.csv"), row.names = TRUE)
cat("Data saved to tests/sc_sim_expr.csv and tests/sc_sim_meta.csv\n")

# Export feature selection results for each cell type
cat("\n=== Feature selection comparison ===\n")
for (ct in unique(meta$cell_type)) {
  ct_mask <- meta$cell_type == ct
  X_ct <- expr_dense[, ct_mask]
  X_filtered <- select_variance(X_ct, var_quantile = 0.5, filter_negative_residuals = FALSE)
  selected_genes <- rownames(X_filtered)
  writeLines(selected_genes, file.path(getwd(), "tests", paste0("r_selected_genes_", ct, ".txt")))
  cat(ct, ":", length(selected_genes), "genes selected\n")
}

cat("\n=== Timing ===\n")
cat("Elapsed:", timing["elapsed"], "seconds\n")

timing_json <- list(elapsed_seconds = unname(timing["elapsed"]))
write_json(timing_json, file.path(getwd(), "tests", "r_timing.json"), auto_unbox = TRUE)
cat("Timing saved to:", file.path(getwd(), "tests", "r_timing.json"), "\n")
