#!/usr/bin/env python
"""
Merge multiple h5ad files into a single AnnData object.

Usage:
    python merge_datasets.py --config config.yaml
    python merge_datasets.py --files data1.h5ad data2.h5ad --names dataset1 dataset2 --output merged.h5ad
"""

import argparse
import yaml
import scanpy as sc
import anndata as ad
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_datasets(
    files: list,
    names: list,
    batch_key: str = "dataset",
    join: str = "inner",
) -> ad.AnnData:
    """
    Merge multiple h5ad files.

    Parameters
    ----------
    files : list
        List of h5ad file paths.
    names : list
        List of dataset names (same length as files).
    batch_key : str
        Column name for batch labels.
    join : str
        Join type ('inner' or 'outer').

    Returns
    -------
    AnnData
        Merged dataset.
    """
    if len(files) != len(names):
        raise ValueError("Number of files must match number of names")

    adatas = {}
    for path, name in zip(files, names):
        print(f"Loading {name} from {path}...")
        adata = sc.read_h5ad(path)
        adata.obs[batch_key] = name
        adatas[name] = adata
        print(f"  Shape: {adata.shape}")

    print(f"\nMerging with join={join}...")
    adata_merged = ad.concat(
        list(adatas.values()),
        join=join,
        label=batch_key,
        keys=list(adatas.keys()),
        index_unique="_",
    )

    # Ensure batch_key is categorical
    adata_merged.obs[batch_key] = adata_merged.obs[batch_key].astype("category")

    print(f"\nMerged shape: {adata_merged.shape}")
    print(f"Cells per batch:")
    for batch, count in adata_merged.obs[batch_key].value_counts().items():
        print(f"  {batch}: {count}")

    return adata_merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple h5ad files for integration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Input h5ad file paths",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help="Dataset names (same order as files)",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="dataset",
        help="Column name for batch labels (default: dataset)",
    )
    parser.add_argument(
        "--join",
        type=str,
        default="inner",
        choices=["inner", "outer"],
        help="Join type for genes (default: inner)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged.h5ad",
        help="Output file path (default: merged.h5ad)",
    )

    args = parser.parse_args()

    # Load config or use command line args
    if args.config:
        config = load_config(args.config)
        files = [f["path"] for f in config["input"]["files"]]
        names = [f["name"] for f in config["input"]["files"]]
        batch_key = config["input"].get("batch_key", "dataset")
        output_dir = Path(config["output"]["dir"])
        output_path = output_dir / "merged.h5ad"
    else:
        if not args.files or not args.names:
            parser.error("Either --config or both --files and --names required")
        files = args.files
        names = args.names
        batch_key = args.batch_key
        output_path = Path(args.output)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge datasets
    adata_merged = merge_datasets(
        files=files,
        names=names,
        batch_key=batch_key,
        join=args.join if not args.config else "inner",
    )

    # Save
    print(f"\nSaving to {output_path}...")
    adata_merged.write_h5ad(output_path)

    # Save summary
    summary_path = output_path.with_suffix(".summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Merge Summary ===\n\n")
        f.write(f"Total cells: {adata_merged.n_obs}\n")
        f.write(f"Total genes: {adata_merged.n_vars}\n")
        f.write(f"Batch key: {batch_key}\n\n")
        f.write("Cells per batch:\n")
        for batch, count in adata_merged.obs[batch_key].value_counts().items():
            f.write(f"  {batch}: {count}\n")
        f.write(f"\nOutput: {output_path}\n")

    print(f"Summary saved to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
