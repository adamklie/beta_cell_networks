#!/usr/bin/env Rscript
# =============================================================================
# Seurat Integration Pipeline for HPC
# =============================================================================
# This script performs batch correction using multiple Seurat integration methods.
# Designed for SLURM execution with checkpointing for long-running jobs.
#
# Usage:
#   Rscript seurat_integration.R --config config.yaml
#   Rscript seurat_integration.R --input merged.h5ad --batch-key sample_id --output ./results
#
# =============================================================================

# -----------------------------------------------------------------------------
# Setup and Libraries
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(SeuratDisk)
  library(harmony)
  library(SeuratWrappers)
  library(batchelor)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(optparse)
  library(yaml)
})

# -----------------------------------------------------------------------------
# Command Line Arguments
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-c", "--config"), type = "character", default = NULL,
              help = "Path to YAML config file"),
  make_option(c("-i", "--input"), type = "character", default = NULL,
              help = "Input h5ad file path"),
  make_option(c("-o", "--output"), type = "character", default = NULL,
              help = "Output directory"),
  make_option(c("--batch-key"), type = "character", default = NULL,
              help = "Metadata column for batch (visualization)"),
  make_option(c("--integration-key"), type = "character", default = NULL,
              help = "Metadata column for integration (defaults to batch-key)"),
  make_option(c("--n-hvgs"), type = "integer", default = NULL,
              help = "Number of highly variable genes"),
  make_option(c("--n-pcs"), type = "integer", default = NULL,
              help = "Number of PCs to compute"),
  make_option(c("--n-dims"), type = "integer", default = NULL,
              help = "Number of dims for UMAP/neighbors"),
  make_option(c("--resolutions"), type = "character", default = NULL,
              help = "Comma-separated clustering resolutions"),
  make_option(c("--methods"), type = "character", default = NULL,
              help = "Comma-separated integration methods: cca,rpca,harmony,fastmnn"),
  make_option(c("--checkpoint"), type = "character", default = NULL,
              help = "Resume from checkpoint RDS file"),
  make_option(c("--skip-to"), type = "character", default = NULL,
              help = "Skip to step: load, preprocess, integrate, umap, cluster, save"),
  make_option(c("--threads"), type = "integer", default = 1,
              help = "Number of threads for parallel operations [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
load_config <- function(config_path) {
  yaml::read_yaml(config_path)
}

# Defaults
defaults <- list(
  input_path = NULL,
  output_dir = "./results/seurat/",
  batch_key = "dataset",
  integration_key = NULL,  # will default to batch_key
  n_hvgs = 3000,
  n_pcs = 50,
  n_dims = 30,
  resolutions = c(0.2, 0.5, 0.8, 1.0),
  methods = c("cca", "rpca", "harmony", "fastmnn"),
  checkpoint = NULL,
  skip_to = NULL,
  threads = 1
)

# Load from config if provided
if (!is.null(opt$config)) {
  config <- load_config(opt$config)

  defaults$input_path <- config$input$h5ad_path
  defaults$output_dir <- file.path(config$output$dir, "seurat")
  defaults$batch_key <- config$input$batch_key
  defaults$integration_key <- config$integration$key
  defaults$n_hvgs <- config$preprocessing$n_top_genes
  defaults$n_pcs <- config$preprocessing$n_pcs
  defaults$n_dims <- config$clustering$n_neighbors  # use same as neighbors
  defaults$resolutions <- config$clustering$resolutions
  defaults$methods <- config$integration$seurat$methods
}

# Override with command line arguments
INPUT_PATH <- opt$input %||% defaults$input_path
OUTPUT_DIR <- opt$output %||% defaults$output_dir
BATCH_KEY <- opt$`batch-key` %||% defaults$batch_key
INTEGRATION_KEY <- opt$`integration-key` %||% defaults$integration_key %||% BATCH_KEY
N_HVGS <- opt$`n-hvgs` %||% defaults$n_hvgs
N_PCS <- opt$`n-pcs` %||% defaults$n_pcs
N_DIMS <- opt$`n-dims` %||% defaults$n_dims
CHECKPOINT_FILE <- opt$checkpoint %||% defaults$checkpoint
SKIP_TO <- opt$`skip-to` %||% defaults$skip_to
N_THREADS <- opt$threads %||% defaults$threads

# Parse resolutions
if (!is.null(opt$resolutions)) {
  CLUSTER_RESOLUTIONS <- as.numeric(strsplit(opt$resolutions, ",")[[1]])
} else {
  CLUSTER_RESOLUTIONS <- defaults$resolutions
}

# Parse methods
if (!is.null(opt$methods)) {
  METHODS_TO_RUN <- strsplit(opt$methods, ",")[[1]]
} else {
  METHODS_TO_RUN <- defaults$methods
}

# Validate required inputs
if (is.null(INPUT_PATH)) {
  stop("Input path required. Provide --config or --input")
}

# Set parallelization
if (N_THREADS > 1) {
  options(future.globals.maxSize = 50 * 1024^3)  # 50 GB
  library(future)
  plan("multicore", workers = N_THREADS)
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
log_msg <- function(...) {
  timestamp <- format(Sys.time(), "[%Y-%m-%d %H:%M:%S]")
  message(timestamp, " ", ...)
}

save_checkpoint <- function(obj, name, output_dir) {
  checkpoint_dir <- file.path(output_dir, "checkpoints")
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(checkpoint_dir, paste0(name, ".rds"))
  log_msg("Saving checkpoint: ", path)
  saveRDS(obj, path)
  return(path)
}

save_figure <- function(plot, name, output_dir, width = 12, height = 8) {
  fig_dir <- file.path(output_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = height, dpi = 150)
  log_msg("Saved figure: ", path)
}

# -----------------------------------------------------------------------------
# Create Output Directories
# -----------------------------------------------------------------------------
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "figures"), showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "checkpoints"), showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "embeddings"), showWarnings = FALSE)

# Log configuration
log_msg("=" %>% rep(70) %>% paste(collapse = ""))
log_msg("Seurat Integration Pipeline")
log_msg("=" %>% rep(70) %>% paste(collapse = ""))
log_msg("Input: ", INPUT_PATH)
log_msg("Output: ", OUTPUT_DIR)
log_msg("Batch key: ", BATCH_KEY)
log_msg("Integration key: ", INTEGRATION_KEY)
log_msg("Methods: ", paste(METHODS_TO_RUN, collapse = ", "))
log_msg("HVGs: ", N_HVGS, " | PCs: ", N_PCS, " | Dims: ", N_DIMS)
log_msg("Resolutions: ", paste(CLUSTER_RESOLUTIONS, collapse = ", "))
if (!is.null(CHECKPOINT_FILE)) log_msg("Resuming from: ", CHECKPOINT_FILE)
if (!is.null(SKIP_TO)) log_msg("Skipping to: ", SKIP_TO)
log_msg("=" %>% rep(70) %>% paste(collapse = ""))

# =============================================================================
# STEP 1: Load Data and Create Seurat Object
# =============================================================================
step_order <- c("load", "preprocess", "integrate", "umap", "cluster", "save")
current_step <- if (!is.null(SKIP_TO)) SKIP_TO else "load"
step_idx <- which(step_order == current_step)

if (!is.null(CHECKPOINT_FILE) && file.exists(CHECKPOINT_FILE)) {
  log_msg("Loading checkpoint...")
  seurat_obj <- readRDS(CHECKPOINT_FILE)
  log_msg("Loaded object with ", ncol(seurat_obj), " cells")
} else if (step_idx <= 1) {
  log_msg("STEP 1: Loading h5ad and creating Seurat object...")

  adata_sce <- readH5AD(INPUT_PATH)

  counts <- assay(adata_sce, 'X')
  col_data <- as.data.frame(colData(adata_sce))

  seurat_obj <- CreateSeuratObject(
    counts = counts,
    project = "beta_cell_networks",
    meta.data = col_data,
    assay = "RNA"
  )

  rm(adata_sce, counts, col_data)
  gc()

  log_msg("Created Seurat object:")
  log_msg("  Cells: ", ncol(seurat_obj))
  log_msg("  Genes: ", nrow(seurat_obj))
  log_msg("  Batches: ", length(unique(seurat_obj[[INTEGRATION_KEY, drop = TRUE]])))

  # Save checkpoint after loading
  save_checkpoint(seurat_obj, "01_loaded", OUTPUT_DIR)
}

# =============================================================================
# STEP 2: Preprocessing
# =============================================================================
if (step_idx <= 2) {
  log_msg("STEP 2: Preprocessing...")

  log_msg("  Normalizing...")
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)

  log_msg("  Finding ", N_HVGS, " variable features...")
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = N_HVGS, verbose = FALSE)

  log_msg("  Scaling...")
  seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)

  log_msg("  Running PCA (", N_PCS, " components)...")
  seurat_obj <- RunPCA(seurat_obj, npcs = N_PCS, verbose = FALSE)

  # Elbow plot
  p_elbow <- ElbowPlot(seurat_obj, ndims = N_PCS)
  save_figure(p_elbow, "elbow_plot", OUTPUT_DIR, width = 8, height = 6)

  log_msg("  Computing uncorrected UMAP...")
  seurat_obj <- FindNeighbors(seurat_obj, reduction = "pca", dims = 1:N_DIMS, verbose = FALSE)
  seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:N_DIMS,
                        reduction.name = "umap_uncorrected", verbose = FALSE)

  # Plot uncorrected
  p1 <- DimPlot(seurat_obj, reduction = "umap_uncorrected", group.by = INTEGRATION_KEY,
                pt.size = 0.1) + ggtitle(paste("Uncorrected -", INTEGRATION_KEY))
  p2 <- DimPlot(seurat_obj, reduction = "umap_uncorrected", group.by = BATCH_KEY,
                pt.size = 0.1) + ggtitle(paste("Uncorrected -", BATCH_KEY))
  save_figure(p1 + p2, "umap_uncorrected", OUTPUT_DIR, width = 14, height = 6)

  # Save checkpoint after preprocessing
  save_checkpoint(seurat_obj, "02_preprocessed", OUTPUT_DIR)
  log_msg("Preprocessing complete")
}

# =============================================================================
# STEP 3: Integration
# =============================================================================
if (step_idx <= 3) {
  log_msg("STEP 3: Running integration methods...")

  # Split layers for integration
  log_msg("  Splitting layers by ", INTEGRATION_KEY, "...")
  seurat_obj[[INTEGRATION_KEY]] <- as.factor(seurat_obj[[INTEGRATION_KEY, drop = TRUE]])
  seurat_obj[["RNA"]] <- split(seurat_obj[["RNA"]], f = seurat_obj[[INTEGRATION_KEY, drop = TRUE]])

  # Re-run preprocessing on split object
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = N_HVGS, verbose = FALSE)
  seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
  seurat_obj <- RunPCA(seurat_obj, npcs = N_PCS, reduction.name = "pca", verbose = FALSE)

  # --- CCA Integration ---
  if ("cca" %in% METHODS_TO_RUN) {
    log_msg("  Running CCA integration...")
    tryCatch({
      seurat_obj <- IntegrateLayers(
        object = seurat_obj,
        method = CCAIntegration,
        orig.reduction = "pca",
        new.reduction = "integrated_cca",
        verbose = FALSE
      )
      log_msg("  CCA complete")
    }, error = function(e) {
      log_msg("  CCA failed: ", e$message)
    })
  }

  # --- RPCA Integration ---
  if ("rpca" %in% METHODS_TO_RUN) {
    log_msg("  Running RPCA integration...")
    tryCatch({
      seurat_obj <- IntegrateLayers(
        object = seurat_obj,
        method = RPCAIntegration,
        orig.reduction = "pca",
        new.reduction = "integrated_rpca",
        verbose = FALSE
      )
      log_msg("  RPCA complete")
    }, error = function(e) {
      log_msg("  RPCA failed: ", e$message)
    })
  }

  # --- Harmony Integration ---
  if ("harmony" %in% METHODS_TO_RUN) {
    log_msg("  Running Harmony integration...")
    tryCatch({
      seurat_obj <- IntegrateLayers(
        object = seurat_obj,
        method = HarmonyIntegration,
        orig.reduction = "pca",
        new.reduction = "integrated_harmony",
        verbose = FALSE
      )
      log_msg("  Harmony complete")
    }, error = function(e) {
      log_msg("  Harmony failed: ", e$message)
    })
  }

  # --- FastMNN Integration ---
  if ("fastmnn" %in% METHODS_TO_RUN) {
    log_msg("  Running FastMNN integration...")
    tryCatch({
      seurat_obj <- IntegrateLayers(
        object = seurat_obj,
        method = FastMNNIntegration,
        orig.reduction = "pca",
        new.reduction = "integrated_mnn",
        verbose = FALSE
      )
      log_msg("  FastMNN complete")
    }, error = function(e) {
      log_msg("  FastMNN failed: ", e$message)
    })
  }

  # Join layers back
  log_msg("  Joining layers...")
  seurat_obj <- JoinLayers(seurat_obj)

  # Save checkpoint after integration
  save_checkpoint(seurat_obj, "03_integrated", OUTPUT_DIR)
  log_msg("Integration complete")
}

# =============================================================================
# STEP 4: Compute UMAPs for Each Method
# =============================================================================
if (step_idx <= 4) {
  log_msg("STEP 4: Computing UMAPs for integrated reductions...")

  reductions_to_process <- c(
    "integrated_cca" = "umap_cca",
    "integrated_rpca" = "umap_rpca",
    "integrated_harmony" = "umap_harmony",
    "integrated_mnn" = "umap_mnn"
  )

  for (red_name in names(reductions_to_process)) {
    if (red_name %in% names(seurat_obj@reductions)) {
      umap_name <- reductions_to_process[[red_name]]
      log_msg("  Computing UMAP: ", red_name, " -> ", umap_name)

      seurat_obj <- FindNeighbors(seurat_obj, reduction = red_name, dims = 1:N_DIMS, verbose = FALSE)
      seurat_obj <- RunUMAP(seurat_obj, reduction = red_name, dims = 1:N_DIMS,
                            reduction.name = umap_name, verbose = FALSE)
    }
  }

  # Create comparison plots
  available_umaps <- c("umap_uncorrected")
  umap_labels <- c("Uncorrected")

  if ("umap_cca" %in% names(seurat_obj@reductions)) {
    available_umaps <- c(available_umaps, "umap_cca"); umap_labels <- c(umap_labels, "CCA")
  }
  if ("umap_rpca" %in% names(seurat_obj@reductions)) {
    available_umaps <- c(available_umaps, "umap_rpca"); umap_labels <- c(umap_labels, "RPCA")
  }
  if ("umap_harmony" %in% names(seurat_obj@reductions)) {
    available_umaps <- c(available_umaps, "umap_harmony"); umap_labels <- c(umap_labels, "Harmony")
  }
  if ("umap_mnn" %in% names(seurat_obj@reductions)) {
    available_umaps <- c(available_umaps, "umap_mnn"); umap_labels <- c(umap_labels, "FastMNN")
  }

  # Comparison by batch
  plots_batch <- lapply(seq_along(available_umaps), function(i) {
    DimPlot(seurat_obj, reduction = available_umaps[i], group.by = INTEGRATION_KEY,
            pt.size = 0.1) +
      ggtitle(umap_labels[i]) +
      theme(legend.position = "none")
  })
  comparison_batch <- wrap_plots(plots_batch, ncol = 3)
  save_figure(comparison_batch, "method_comparison_batch", OUTPUT_DIR, width = 16, height = 12)

  # Comparison by dataset
  plots_dataset <- lapply(seq_along(available_umaps), function(i) {
    DimPlot(seurat_obj, reduction = available_umaps[i], group.by = BATCH_KEY,
            pt.size = 0.1) +
      ggtitle(umap_labels[i]) +
      theme(legend.position = "none")
  })
  comparison_dataset <- wrap_plots(plots_dataset, ncol = 3)
  save_figure(comparison_dataset, "method_comparison_dataset", OUTPUT_DIR, width = 16, height = 12)

  log_msg("UMAPs complete")
}

# =============================================================================
# STEP 5: Clustering
# =============================================================================
if (step_idx <= 5) {
  log_msg("STEP 5: Clustering on each integrated reduction...")

  # Build available_umaps if not already defined
  if (!exists("available_umaps")) {
    available_umaps <- c("umap_uncorrected")
    if ("umap_cca" %in% names(seurat_obj@reductions)) available_umaps <- c(available_umaps, "umap_cca")
    if ("umap_rpca" %in% names(seurat_obj@reductions)) available_umaps <- c(available_umaps, "umap_rpca")
    if ("umap_harmony" %in% names(seurat_obj@reductions)) available_umaps <- c(available_umaps, "umap_harmony")
    if ("umap_mnn" %in% names(seurat_obj@reductions)) available_umaps <- c(available_umaps, "umap_mnn")

    umap_labels <- c("Uncorrected")
    if ("umap_cca" %in% names(seurat_obj@reductions)) umap_labels <- c(umap_labels, "CCA")
    if ("umap_rpca" %in% names(seurat_obj@reductions)) umap_labels <- c(umap_labels, "RPCA")
    if ("umap_harmony" %in% names(seurat_obj@reductions)) umap_labels <- c(umap_labels, "Harmony")
    if ("umap_mnn" %in% names(seurat_obj@reductions)) umap_labels <- c(umap_labels, "FastMNN")
  }

  for (i in seq_along(available_umaps)) {
    red_name <- available_umaps[i]
    label <- tolower(gsub("umap_", "", red_name))

    # Determine which reduction to use for neighbors
    if (red_name == "umap_uncorrected") {
      neighbors_red <- "pca"
    } else {
      neighbors_red <- paste0("integrated_", label)
      if (label == "mnn") neighbors_red <- "integrated_mnn"
    }

    if (neighbors_red %in% names(seurat_obj@reductions)) {
      log_msg("  Clustering for ", label, "...")
      seurat_obj <- FindNeighbors(seurat_obj, reduction = neighbors_red, dims = 1:N_DIMS, verbose = FALSE)

      # Cluster at each resolution
      for (res in CLUSTER_RESOLUTIONS) {
        seurat_obj <- FindClusters(seurat_obj, resolution = res, verbose = FALSE)
        cluster_col <- paste0("clusters_", label, "_", res)
        seurat_obj[[cluster_col]] <- seurat_obj$seurat_clusters
      }

      # Keep a default cluster column (first resolution)
      default_col <- paste0("clusters_", label, "_", CLUSTER_RESOLUTIONS[1])
      seurat_obj[[paste0("clusters_", label)]] <- seurat_obj[[default_col]]
    }
  }

  # Plot clusters (using first resolution)
  cluster_cols <- grep("^clusters_[a-z]+$", colnames(seurat_obj@meta.data), value = TRUE)

  if (length(cluster_cols) > 0) {
    plots_clusters <- lapply(seq_along(available_umaps), function(i) {
      label <- tolower(gsub("umap_", "", available_umaps[i]))
      cluster_col <- paste0("clusters_", label)

      if (cluster_col %in% colnames(seurat_obj@meta.data)) {
        DimPlot(seurat_obj, reduction = available_umaps[i], group.by = cluster_col,
                pt.size = 0.1, label = TRUE) +
          ggtitle(paste(umap_labels[i], "- Clusters")) +
          theme(legend.position = "none")
      } else {
        ggplot() + theme_void() + ggtitle(paste(umap_labels[i], "- No clusters"))
      }
    })

    cluster_comparison <- wrap_plots(plots_clusters, ncol = 3)
    save_figure(cluster_comparison, "method_comparison_clusters", OUTPUT_DIR, width = 16, height = 12)
  }

  log_msg("Clustering complete")
}

# =============================================================================
# STEP 6: Save Final Results
# =============================================================================
log_msg("STEP 6: Saving final results...")

# Save as RDS (main output)
rds_path <- file.path(OUTPUT_DIR, "integrated_seurat.rds")
log_msg("  Saving Seurat object: ", rds_path)
saveRDS(seurat_obj, rds_path)

# Save as h5seurat
h5seurat_path <- file.path(OUTPUT_DIR, "integrated_seurat.h5seurat")
log_msg("  Saving h5seurat: ", h5seurat_path)
tryCatch({
  SaveH5Seurat(seurat_obj, filename = h5seurat_path, overwrite = TRUE)
}, error = function(e) {
  log_msg("  Warning: h5seurat save failed: ", e$message)
})

# Save cell metadata
metadata_path <- file.path(OUTPUT_DIR, "cell_metadata.tsv")
log_msg("  Saving metadata: ", metadata_path)
write.table(seurat_obj@meta.data, metadata_path, sep = "\t", quote = FALSE)

# Save embeddings
embeddings_dir <- file.path(OUTPUT_DIR, "embeddings")
for (red_name in names(seurat_obj@reductions)) {
  embed_path <- file.path(embeddings_dir, paste0(red_name, ".csv"))
  write.csv(Embeddings(seurat_obj, reduction = red_name), embed_path)
  log_msg("  Saved embedding: ", red_name)
}

# =============================================================================
# Summary
# =============================================================================
log_msg("=" %>% rep(70) %>% paste(collapse = ""))
log_msg("Pipeline complete!")
log_msg("=" %>% rep(70) %>% paste(collapse = ""))
log_msg("Output directory: ", OUTPUT_DIR)
log_msg("Final object: ", rds_path)
log_msg("Available reductions: ", paste(names(seurat_obj@reductions), collapse = ", "))
log_msg("Cells: ", ncol(seurat_obj))
log_msg("Cluster columns: ", paste(grep("^clusters_", colnames(seurat_obj@meta.data), value = TRUE), collapse = ", "))

# Session info
sink(file.path(OUTPUT_DIR, "session_info.txt"))
sessionInfo()
sink()
log_msg("Session info saved")

log_msg("Done!")
