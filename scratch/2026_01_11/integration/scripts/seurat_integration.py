#!/usr/bin/env python
"""
Seurat integration methods for scRNA-seq data.

Runs CCA, RPCA, Harmony, and/or FastMNN via Seurat in R.

Usage:
    python seurat_integration.py --config config.yaml
    python seurat_integration.py --input merged.h5ad --batch-key sample_id --output results/
"""

import argparse
import yaml
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import store_raw_counts
from utils.evaluation import compare_integration_methods


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_seurat_integration(
    input_path: str,
    output_dir: str,
    batch_key: str,
    methods: list = None,
    n_hvgs: int = 3000,
    n_pcs: int = 50,
):
    """
    Run Seurat integration methods via R/rpy2.

    Parameters
    ----------
    input_path : str
        Path to merged h5ad file.
    output_dir : str
        Output directory.
    batch_key : str
        Column in obs for batch correction.
    methods : list
        Integration methods to run: ['cca', 'rpca', 'harmony', 'fastmnn'].
    n_hvgs : int
        Number of highly variable genes.
    n_pcs : int
        Number of principal components.
    """
    if methods is None:
        methods = ["cca", "rpca", "harmony", "fastmnn"]

    # Setup rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    numpy2ri.activate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    (output_path / "metrics").mkdir(exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Shape: {adata.shape}")
    print(f"Batches ({batch_key}): {adata.obs[batch_key].nunique()}")

    # Store raw counts
    if "counts" not in adata.layers:
        store_raw_counts(adata, layer_name="counts")

    # Extract data for R
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts_t = counts.T

    cell_names = adata.obs_names.tolist()
    gene_names = adata.var_names.tolist()
    metadata = adata.obs.copy()

    # Push to R
    ro.globalenv["counts_matrix"] = counts_t
    ro.globalenv["cell_names"] = ro.StrVector(cell_names)
    ro.globalenv["gene_names"] = ro.StrVector(gene_names)
    ro.globalenv["metadata"] = ro.conversion.py2rpy(metadata)
    ro.globalenv["integration_key"] = batch_key
    ro.globalenv["n_hvgs"] = n_hvgs
    ro.globalenv["n_pcs"] = n_pcs
    ro.globalenv["methods_to_run"] = ro.StrVector(methods)

    # Run R code
    r_code = """
    library(Seurat)
    library(SeuratObject)
    library(harmony)
    library(SeuratWrappers)
    library(batchelor)

    # Create Seurat object
    rownames(counts_matrix) <- gene_names
    colnames(counts_matrix) <- cell_names
    rownames(metadata) <- cell_names

    seurat_obj <- CreateSeuratObject(
        counts = counts_matrix,
        meta.data = metadata
    )

    # Preprocess
    seurat_obj <- NormalizeData(seurat_obj)
    seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = n_hvgs)
    seurat_obj <- ScaleData(seurat_obj)
    seurat_obj <- RunPCA(seurat_obj, npcs = n_pcs)

    # Split for integration
    seurat_obj[[integration_key]] <- as.factor(seurat_obj[[integration_key, drop=TRUE]])
    seurat_obj <- SplitObject(seurat_obj, split.by = integration_key)
    seurat_obj <- merge(seurat_obj[[1]], seurat_obj[-1])
    seurat_obj[['RNA']] <- split(seurat_obj[['RNA']], f = seurat_obj[[integration_key]])

    # Re-preprocess
    seurat_obj <- NormalizeData(seurat_obj)
    seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = n_hvgs)
    seurat_obj <- ScaleData(seurat_obj)
    seurat_obj <- RunPCA(seurat_obj, npcs = n_pcs, reduction.name = "pca")

    # Run integration methods
    if ("cca" %in% methods_to_run) {
        message("Running CCA...")
        seurat_obj <- IntegrateLayers(seurat_obj, method = CCAIntegration,
                                       orig.reduction = "pca", new.reduction = "integrated_cca")
    }
    if ("rpca" %in% methods_to_run) {
        message("Running RPCA...")
        seurat_obj <- IntegrateLayers(seurat_obj, method = RPCAIntegration,
                                       orig.reduction = "pca", new.reduction = "integrated_rpca")
    }
    if ("harmony" %in% methods_to_run) {
        message("Running Harmony...")
        seurat_obj <- IntegrateLayers(seurat_obj, method = HarmonyIntegration,
                                       orig.reduction = "pca", new.reduction = "integrated_harmony")
    }
    if ("fastmnn" %in% methods_to_run) {
        message("Running FastMNN...")
        seurat_obj <- IntegrateLayers(seurat_obj, method = FastMNNIntegration,
                                       orig.reduction = "pca", new.reduction = "integrated_mnn")
    }

    # Join and compute UMAPs
    seurat_obj <- JoinLayers(seurat_obj)

    reductions <- c("pca", "integrated_cca", "integrated_rpca", "integrated_harmony", "integrated_mnn")
    for (red in reductions) {
        if (red %in% names(seurat_obj@reductions)) {
            umap_name <- ifelse(red == "pca", "umap_uncorrected", gsub("integrated_", "umap_", red))
            seurat_obj <- FindNeighbors(seurat_obj, reduction = red, dims = 1:30)
            seurat_obj <- RunUMAP(seurat_obj, reduction = red, dims = 1:30, reduction.name = umap_name)
        }
    }

    # Extract embeddings
    embeddings_dict <- list()
    cell_order <- colnames(seurat_obj)
    for (red_name in names(seurat_obj@reductions)) {
        embeddings_dict[[red_name]] <- as.matrix(Embeddings(seurat_obj, reduction = red_name))
    }

    message("Integration complete")
    """

    print("Running Seurat integration in R...")
    ro.r(r_code)

    # Get results back
    embeddings_dict = dict(ro.globalenv["embeddings_dict"])
    cell_order = list(ro.globalenv["cell_order"])

    # Reorder adata
    adata = adata[cell_order].copy()

    # Store embeddings
    name_map = {
        "pca": "X_pca",
        "umap_uncorrected": "X_umap_uncorrected",
        "integrated_cca": "X_seurat_cca",
        "integrated_rpca": "X_seurat_rpca",
        "integrated_harmony": "X_seurat_harmony",
        "integrated_mnn": "X_seurat_mnn",
        "umap_cca": "X_umap_cca",
        "umap_rpca": "X_umap_rpca",
        "umap_harmony": "X_umap_harmony",
        "umap_mnn": "X_umap_mnn",
    }

    for r_name, py_name in name_map.items():
        if r_name in embeddings_dict:
            adata.obsm[py_name] = np.array(embeddings_dict[r_name])
            print(f"Stored {py_name}: {adata.obsm[py_name].shape}")

    # Compute metrics
    print("\nComputing integration metrics...")
    integrated_embeddings = {
        "Uncorrected": "X_pca",
        "CCA": "X_seurat_cca",
        "RPCA": "X_seurat_rpca",
        "Harmony": "X_seurat_harmony",
        "FastMNN": "X_seurat_mnn",
    }
    integrated_embeddings = {k: v for k, v in integrated_embeddings.items() if v in adata.obsm}

    metrics_df = compare_integration_methods(
        adata,
        batch_key=batch_key,
        embeddings=integrated_embeddings,
    )
    print(metrics_df)
    metrics_df.to_csv(output_path / "metrics" / "integration_metrics.csv")

    # Save
    print(f"\nSaving to {output_path}...")
    adata.write_h5ad(output_path / "integrated_seurat.h5ad")

    print("\nDone!")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Seurat integration for scRNA-seq")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, help="Input h5ad file")
    parser.add_argument("--batch-key", type=str, help="Batch key for integration")
    parser.add_argument("--output", type=str, default="./results/seurat/", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=["cca", "rpca", "harmony", "fastmnn"],
                        choices=["cca", "rpca", "harmony", "fastmnn"])
    parser.add_argument("--n-hvgs", type=int, default=3000, help="Number of HVGs")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        run_seurat_integration(
            input_path=config["input"]["h5ad_path"],
            output_dir=config["output"]["dir"],
            batch_key=config["integration"]["key"],
            methods=config["integration"]["seurat"].get("methods", ["cca", "rpca", "harmony", "fastmnn"]),
            n_hvgs=config["preprocessing"].get("n_top_genes", 3000),
            n_pcs=config["preprocessing"].get("n_pcs", 50),
        )
    else:
        if not args.input or not args.batch_key:
            parser.error("Either --config or both --input and --batch-key required")
        run_seurat_integration(
            input_path=args.input,
            output_dir=args.output,
            batch_key=args.batch_key,
            methods=args.methods,
            n_hvgs=args.n_hvgs,
            n_pcs=args.n_pcs,
        )


if __name__ == "__main__":
    main()
