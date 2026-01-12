#!/usr/bin/env python
"""
Python-native integration methods (scVI and Scanorama) for scRNA-seq data.

Usage:
    python python_integration.py --config config.yaml
    python python_integration.py --input merged.h5ad --batch-key sample_id --output results/
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

from utils.preprocessing import store_raw_counts, normalize_and_log, find_hvgs
from utils.evaluation import compare_integration_methods


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_python_integration(
    input_path: str,
    output_dir: str,
    batch_key: str,
    methods: list = None,
    n_hvgs: int = 3000,
    n_neighbors: int = 30,
    resolutions: list = None,
    scvi_n_latent: int = 30,
    scvi_n_layers: int = 2,
    scvi_n_epochs: int = None,
    scanorama_knn: int = 20,
    save_model: bool = True,
):
    """
    Run Python integration methods.

    Parameters
    ----------
    input_path : str
        Path to merged h5ad file.
    output_dir : str
        Output directory.
    batch_key : str
        Column in obs for batch correction.
    methods : list
        Methods to run: ['scvi', 'scanorama'].
    n_hvgs : int
        Number of highly variable genes.
    n_neighbors : int
        Number of neighbors for graph.
    resolutions : list
        Leiden clustering resolutions.
    scvi_n_latent : int
        scVI latent dimensions.
    scvi_n_layers : int
        scVI encoder/decoder layers.
    scvi_n_epochs : int
        scVI training epochs.
    scanorama_knn : int
        Scanorama k-nearest neighbors.
    save_model : bool
        Whether to save scVI model.
    """
    if methods is None:
        methods = ["scvi", "scanorama"]
    if resolutions is None:
        resolutions = [0.2, 0.5, 0.8, 1.0]

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
        print("Storing raw counts...")
        store_raw_counts(adata, layer_name="counts")

    # Preprocessing
    print("\nPreprocessing...")
    normalize_and_log(adata)
    find_hvgs(adata, n_top_genes=n_hvgs)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    # Uncorrected UMAP
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.obsm["X_umap_uncorrected"] = adata.obsm["X_umap"].copy()

    embeddings = {"Uncorrected": "X_pca"}

    # scVI
    if "scvi" in methods:
        try:
            from utils.integration import run_scvi

            print(f"\nRunning scVI...")
            model_dir = output_path / "scvi_model" if save_model else None

            run_scvi(
                adata,
                batch_key=batch_key,
                n_latent=scvi_n_latent,
                n_layers=scvi_n_layers,
                n_epochs=scvi_n_epochs,
                early_stopping=True,
                key_added="X_scVI",
                model_dir=str(model_dir) if model_dir else None,
            )

            # UMAP on scVI
            sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=n_neighbors)
            sc.tl.umap(adata)
            adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()

            # Clustering
            for res in resolutions:
                sc.tl.leiden(adata, resolution=res, key_added=f"leiden_scvi_{res}")

            embeddings["scVI"] = "X_scVI"
            print("scVI complete")

        except ImportError:
            print("scvi-tools not installed, skipping scVI")

    # Scanorama
    if "scanorama" in methods:
        try:
            import scanorama

            print(f"\nRunning Scanorama...")

            # Subset to HVGs
            adata_hvg = adata[:, adata.var["highly_variable"]].copy()

            # Split by batch
            batches = adata_hvg.obs[batch_key].unique()
            adatas_batch = [
                adata_hvg[adata_hvg.obs[batch_key] == b].copy() for b in batches
            ]

            # Run Scanorama
            scanorama.integrate_scanpy(adatas_batch, knn=scanorama_knn)

            # Concatenate
            adata_integrated = adatas_batch[0].concatenate(
                adatas_batch[1:],
                batch_key="_scanorama_batch",
                index_unique=None,
            )
            adata_integrated = adata_integrated[adata.obs_names]
            adata.obsm["X_scanorama"] = adata_integrated.obsm["X_scanorama"]

            # PCA on Scanorama
            from sklearn.decomposition import PCA

            pca = PCA(n_components=50)
            adata.obsm["X_pca_scanorama"] = pca.fit_transform(adata.obsm["X_scanorama"])

            # UMAP on Scanorama
            sc.pp.neighbors(adata, use_rep="X_pca_scanorama", n_neighbors=n_neighbors)
            sc.tl.umap(adata)
            adata.obsm["X_umap_scanorama"] = adata.obsm["X_umap"].copy()

            # Clustering
            for res in resolutions:
                sc.tl.leiden(adata, resolution=res, key_added=f"leiden_scanorama_{res}")

            embeddings["Scanorama"] = "X_pca_scanorama"
            print("Scanorama complete")

        except ImportError:
            print("scanorama not installed, skipping Scanorama")

    # Metrics
    print("\nComputing integration metrics...")
    metrics_df = compare_integration_methods(
        adata,
        batch_key=batch_key,
        embeddings=embeddings,
    )
    print(metrics_df)
    metrics_df.to_csv(output_path / "metrics" / "integration_metrics.csv")

    # Save
    print(f"\nSaving to {output_path}...")
    adata.write_h5ad(output_path / "integrated_python.h5ad")
    adata.obs.to_csv(output_path / "cell_metadata.tsv", sep="\t")

    print("\nDone!")
    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Python integration (scVI/Scanorama) for scRNA-seq"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, help="Input h5ad file")
    parser.add_argument("--batch-key", type=str, help="Batch key for integration")
    parser.add_argument(
        "--output", type=str, default="./results/python/", help="Output directory"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["scvi", "scanorama"],
        choices=["scvi", "scanorama"],
    )
    parser.add_argument("--n-hvgs", type=int, default=3000, help="Number of HVGs")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors")
    parser.add_argument(
        "--resolutions", nargs="+", type=float, default=[0.2, 0.5, 0.8, 1.0]
    )
    parser.add_argument("--scvi-n-latent", type=int, default=30)
    parser.add_argument("--scvi-n-layers", type=int, default=2)
    parser.add_argument("--scvi-n-epochs", type=int, default=None)
    parser.add_argument("--scanorama-knn", type=int, default=20)
    parser.add_argument("--no-save-model", action="store_true")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        run_python_integration(
            input_path=config["input"]["h5ad_path"],
            output_dir=config["output"]["dir"],
            batch_key=config["integration"]["key"],
            methods=["scvi", "scanorama"],
            n_hvgs=config["preprocessing"].get("n_top_genes", 3000),
            n_neighbors=config["clustering"].get("n_neighbors", 30),
            resolutions=config["clustering"].get("resolutions", [0.2, 0.5, 0.8, 1.0]),
            scvi_n_latent=config["integration"]["scvi"].get("n_latent", 30),
            scvi_n_layers=config["integration"]["scvi"].get("n_layers", 2),
            scvi_n_epochs=config["integration"]["scvi"].get("n_epochs"),
            scanorama_knn=config["integration"]["scanorama"].get("knn", 20),
            save_model=config["output"].get("save_model", True),
        )
    else:
        if not args.input or not args.batch_key:
            parser.error("Either --config or both --input and --batch-key required")
        run_python_integration(
            input_path=args.input,
            output_dir=args.output,
            batch_key=args.batch_key,
            methods=args.methods,
            n_hvgs=args.n_hvgs,
            n_neighbors=args.n_neighbors,
            resolutions=args.resolutions,
            scvi_n_latent=args.scvi_n_latent,
            scvi_n_layers=args.scvi_n_layers,
            scvi_n_epochs=args.scvi_n_epochs,
            scanorama_knn=args.scanorama_knn,
            save_model=not args.no_save_model,
        )


if __name__ == "__main__":
    main()
