#!/usr/bin/env Rscript
# R per-function dump for Notebook 3 (function_by_function_R_parity)
# Runs each R function with the same inputs and dumps results as JSON

library(jsonlite)
library(dplyr)
library(purrr)
library(tibble)
library(magrittr)
library(Matrix)
library(sparseMatrixStats)
library(randomForest)
library(scales)
library(tidyr)

# Source Augur R code
augur_dir <- "D:/test/Augur-master/Augur-master"
r_files <- list.files(file.path(augur_dir, "R"), pattern = "\\.R$", full.names = TRUE)
for (f in r_files) {
  tryCatch(source(f), error = function(e) message("Skipping ", f, ": ", e$message))
}

# Load data
load(file.path(augur_dir, "data", "sc_sim.rda"))
tryCatch({
  expr <- Seurat::GetAssayData(sc_sim)
}, error = function(e) {
  tryCatch({
    expr <- sc_sim@assays$RNA@data
  }, error = function(e2) {
    expr <- sc_sim@assays[[1]]@data
  })
})
meta <- sc_sim@meta.data

results <- list()

# 1. select_variance
cat("Running select_variance...\n")
expr_dense <- as.matrix(expr)
sd_before <- nrow(expr_dense)
expr_filtered <- select_variance(expr_dense, var_quantile = 0.5, filter_negative_residuals = FALSE)
sd_after <- nrow(expr_filtered)
results$select_variance <- list(
  input_genes = sd_before,
  output_genes = sd_after,
  filtered = sd_before - sd_after
)

# 2. select_random
cat("Running select_random...\n")
set.seed(42)
expr_random <- select_random(expr_filtered, feature_perc = 0.5)
results$select_random <- list(
  input_genes = nrow(expr_filtered),
  output_genes = nrow(expr_random)
)

# 3. calculate_auc (small run)
cat("Running calculate_auc (10 subsamples)...\n")
set.seed(42)
result_auc <- calculate_auc(
  expr,
  meta = meta,
  label_col = "label",
  cell_type_col = "cell_type",
  n_subsamples = 10,
  subsample_size = 20,
  folds = 3,
  var_quantile = 0.5,
  feature_perc = 0.5,
  n_threads = 1,
  show_progress = FALSE,
  classifier = "rf"
)

results$calculate_auc <- list(
  AUC = result_auc$AUC,
  n_results = nrow(result_auc$results),
  n_features = nrow(result_auc$feature_importance)
)

# Save
output_path <- file.path(getwd(), "examples", "r_per_function_output.json")
write_json(results, output_path, auto_unbox = TRUE, pretty = TRUE)
cat("Saved to:", output_path, "\n")
