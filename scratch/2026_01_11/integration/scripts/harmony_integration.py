#!/usr/bin/env python
"""
Harmony integration for scRNA-seq data.

Usage:
    python harmony_integration.py --config config.yaml
    python harmony_integration.py --input merged.h5ad --batch-key sample_id --output results/
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

from utils.preprocessing import standard_preprocess, store_raw_counts
from utils.integration import run_harmony_rpy2
from utils.evaluation import summarize_integration_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_harmony_pipeline(
    input_path: str,
    output_dir: str,
    batch_key: str,
    n_hvgs: int = 3000,
    regress_vars: list = None,
    n_pcs: int = 50,
    n_neighbors: int = 30,
    resolutions: list = None,
    theta: float = None,
    save_slim: bool = True,
    save_markers: bool = True,
):
    """
    Run full Harmony integration pipeline.

    Parameters
    ----------
    input_path : str
        Path to merged h5ad file.
    output_dir : str
        Output directory.
    batch_key : str
        Column in obs for batch correction.
    n_hvgs : int
        Number of highly variable genes.
    regress_vars : list
        Variables to regress out.
    n_pcs : int
        Number of PCs.
    n_neighbors : int
        Number of neighbors for graph.
    resolutions : list
        Leiden clustering resolutions.
    theta : float
        Harmony diversity penalty.
    save_slim : bool
        Whether to save slim version.
    save_markers : bool
        Whether to compute and save markers.
    """
    if resolutions is None:
        resolutions = [0.2, 0.5, 0.8, 1.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    (output_path / "markers").mkdir(exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Shape: {adata.shape}")
    print(f"Batches ({batch_key}): {adata.obs[batch_key].nunique()}")

    # Store raw counts
    if "counts" not in adata.layers:
        print("Storing raw counts...")
        store_raw_counts(adata, layer_name="counts")

    # Preprocessing
    print("\nPreprocessing...")
    standard_preprocess(
        adata,
        n_hvgs=n_hvgs,
        regress_vars=regress_vars,
        n_pcs=n_pcs,
    )
    print(f"HVGs: {adata.var['highly_variable'].sum()}")

    # Run Harmony
    print(f"\nRunning Harmony on {batch_key}...")
    harmony_key = f"X_harmony_{batch_key}"
    run_harmony_rpy2(
        adata,
        batch_key=batch_key,
        use_rep="X_pca",
        theta=theta,
        key_added=harmony_key,
    )

    # Neighbors and clustering
    print("\nComputing neighbors and clustering...")
    sc.pp.neighbors(
        adata,
        use_rep=harmony_key,
        n_neighbors=n_neighbors,
        metric="cosine",
    )

    for res in resolutions:
        key = f"leiden_harmony_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_clusters} clusters")

    # UMAP
    print("\nComputing UMAP...")
    sc.tl.umap(adata)
    adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"].copy()

    # Metrics
    print("\nIntegration metrics:")
    metrics = summarize_integration_metrics(adata, batch_key, use_rep=harmony_key)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Marker genes
    if save_markers:
        default_cluster = f"leiden_harmony_{resolutions[1]}"
        print(f"\nFinding markers for {default_cluster}...")
        sc.tl.rank_genes_groups(adata, groupby=default_cluster, method="wilcoxon")

        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        for group in groups:
            markers_df = pd.DataFrame({
                "gene": result["names"][group],
                "score": result["scores"][group],
                "logfoldchange": result["logfoldchanges"][group],
                "pval": result["pvals"][group],
                "pval_adj": result["pvals_adj"][group],
            })
            markers_df.to_csv(
                output_path / "markers" / f"cluster_{group}_markers.tsv",
                sep="\t",
                index=False,
            )

    # Save
    print(f"\nSaving to {output_path}...")
    adata.write_h5ad(output_path / "integrated_harmony.h5ad")

    if save_slim:
        adata_slim = sc.AnnData(
            X=adata.layers["counts"] if "counts" in adata.layers else adata.X,
            obs=adata.obs,
            var=adata.var[["highly_variable"]],
            obsm=adata.obsm,
        )
        adata_slim.write_h5ad(output_path / "integrated_harmony_slim.h5ad")

    adata.obs.to_csv(output_path / "cell_metadata.tsv", sep="\t")

    print("\nDone!")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Harmony integration for scRNA-seq")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, help="Input h5ad file")
    parser.add_argument("--batch-key", type=str, help="Batch key for integration")
    parser.add_argument("--output", type=str, default="./results/harmony/", help="Output directory")
    parser.add_argument("--n-hvgs", type=int, default=3000, help="Number of HVGs")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors")
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--theta", type=float, default=None, help="Harmony theta parameter")
    parser.add_argument("--no-slim", action="store_true", help="Don't save slim version")
    parser.add_argument("--no-markers", action="store_true", help="Don't compute markers")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        run_harmony_pipeline(
            input_path=config["input"]["h5ad_path"],
            output_dir=config["output"]["dir"],
            batch_key=config["integration"]["key"],
            n_hvgs=config["preprocessing"].get("n_top_genes", 3000),
            regress_vars=config["preprocessing"].get("regress_out"),
            n_pcs=config["preprocessing"].get("n_pcs", 50),
            n_neighbors=config["clustering"].get("n_neighbors", 30),
            resolutions=config["clustering"].get("resolutions", [0.2, 0.5, 0.8, 1.0]),
            theta=config["integration"]["harmony"].get("theta"),
            save_slim=config["output"].get("save_slim", True),
            save_markers=config["output"].get("save_markers", True),
        )
    else:
        if not args.input or not args.batch_key:
            parser.error("Either --config or both --input and --batch-key required")
        run_harmony_pipeline(
            input_path=args.input,
            output_dir=args.output,
            batch_key=args.batch_key,
            n_hvgs=args.n_hvgs,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            resolutions=args.resolutions,
            theta=args.theta,
            save_slim=not args.no_slim,
            save_markers=not args.no_markers,
        )


if __name__ == "__main__":
    main()
