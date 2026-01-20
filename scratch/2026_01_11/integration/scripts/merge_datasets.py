#!/usr/bin/env python
"""
Merge multiple h5ad files into a single AnnData object.

Supports optional downsampling:
  - Random: Downsample to a target cell count
  - Cell type aware: Match cell types between datasets

Usage:
    python merge_datasets.py --config config.yaml
    python merge_datasets.py --files data1.h5ad data2.h5ad --names dataset1 dataset2 --output merged.h5ad
    python merge_datasets.py --config config.yaml --downsample random --n-cells 50000
    python merge_datasets.py --config config.yaml --downsample celltype_aware --celltype-column cell_type
"""

import argparse
import yaml
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def downsample_random(
    adatas: dict,
    n_cells: int,
    per_dataset: Optional[dict] = None,
    seed: int = 42,
) -> dict:
    """
    Randomly downsample datasets to a target cell count.

    Parameters
    ----------
    adatas : dict
        Dictionary of AnnData objects keyed by dataset name.
    n_cells : int
        Target total cell count.
    per_dataset : dict, optional
        Specific cell counts per dataset. If None, proportionally split n_cells.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary of downsampled AnnData objects.
    """
    np.random.seed(seed)

    if per_dataset is None:
        # Proportionally split n_cells based on dataset sizes
        total = sum(adata.n_obs for adata in adatas.values())
        per_dataset = {
            name: int(n_cells * adata.n_obs / total)
            for name, adata in adatas.items()
        }

    downsampled = {}
    for name, adata in adatas.items():
        target = per_dataset.get(name, adata.n_obs)
        target = min(target, adata.n_obs)  # Can't sample more than available

        if target < adata.n_obs:
            indices = np.random.choice(adata.n_obs, size=target, replace=False)
            indices = np.sort(indices)
            downsampled[name] = adata[indices].copy()
            print(f"  {name}: {adata.n_obs} -> {target} cells")
        else:
            downsampled[name] = adata.copy()
            print(f"  {name}: {adata.n_obs} cells (no downsampling needed)")

    return downsampled


def downsample_celltype_aware(
    adatas: dict,
    celltype_column: str,
    n_cells_per_type: int = 1000,
    keep_unshared: bool = False,
    min_cells_per_type: int = 10,
    seed: int = 42,
) -> dict:
    """
    Downsample datasets with cell type awareness.

    Matches the number of cells per cell type across datasets and optionally
    removes cell types that are not present in all datasets.

    Parameters
    ----------
    adatas : dict
        Dictionary of AnnData objects keyed by dataset name.
    celltype_column : str
        Column name containing cell type annotations.
    n_cells_per_type : int
        Maximum number of cells to keep per cell type per dataset.
    keep_unshared : bool
        If False, remove cell types not present in all datasets.
    min_cells_per_type : int
        Minimum cells required to keep a cell type.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary of downsampled AnnData objects.
    """
    np.random.seed(seed)

    # Find cell types in each dataset
    celltype_sets = {
        name: set(adata.obs[celltype_column].dropna().unique())
        for name, adata in adatas.items()
    }

    # Find shared cell types
    shared_celltypes = set.intersection(*celltype_sets.values())
    all_celltypes = set.union(*celltype_sets.values())

    print(f"\n  Cell types per dataset:")
    for name, ct_set in celltype_sets.items():
        print(f"    {name}: {len(ct_set)} types")
    print(f"  Shared cell types: {len(shared_celltypes)}")
    print(f"  Total unique cell types: {len(all_celltypes)}")

    if not keep_unshared:
        print(f"  Keeping only shared cell types: {sorted(shared_celltypes)}")
        celltypes_to_keep = shared_celltypes
    else:
        celltypes_to_keep = all_celltypes

    downsampled = {}
    for name, adata in adatas.items():
        indices_to_keep = []

        for celltype in celltypes_to_keep:
            # Get cells of this type
            mask = adata.obs[celltype_column] == celltype
            ct_indices = np.where(mask)[0]

            if len(ct_indices) < min_cells_per_type:
                continue

            # Downsample if needed
            if len(ct_indices) > n_cells_per_type:
                sampled = np.random.choice(ct_indices, size=n_cells_per_type, replace=False)
            else:
                sampled = ct_indices

            indices_to_keep.extend(sampled)

        indices_to_keep = np.sort(indices_to_keep)
        downsampled[name] = adata[indices_to_keep].copy()

        # Report
        orig_types = adata.obs[celltype_column].nunique()
        new_types = downsampled[name].obs[celltype_column].nunique()
        print(f"  {name}: {adata.n_obs} -> {len(indices_to_keep)} cells "
              f"({orig_types} -> {new_types} cell types)")

    # Summary of cell types in downsampled data
    print("\n  Cell type counts after downsampling:")
    for name, adata in downsampled.items():
        counts = adata.obs[celltype_column].value_counts()
        print(f"    {name}:")
        for ct, count in counts.items():
            print(f"      {ct}: {count}")

    return downsampled


def load_datasets(
    files: list,
    names: list,
    batch_key: str = "dataset",
) -> dict:
    """
    Load multiple h5ad files into a dictionary.

    Parameters
    ----------
    files : list
        List of h5ad file paths.
    names : list
        List of dataset names (same length as files).
    batch_key : str
        Column name for batch labels.

    Returns
    -------
    dict
        Dictionary of AnnData objects keyed by name.
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

    return adatas


def merge_adatas(
    adatas: dict,
    batch_key: str = "dataset",
    join: str = "inner",
) -> ad.AnnData:
    """
    Merge a dictionary of AnnData objects.

    Parameters
    ----------
    adatas : dict
        Dictionary of AnnData objects keyed by name.
    batch_key : str
        Column name for batch labels.
    join : str
        Join type ('inner' or 'outer').

    Returns
    -------
    AnnData
        Merged dataset.
    """
    print(f"Merging with join={join}...")
    adata_merged = ad.concat(
        list(adatas.values()),
        join=join,
        label=batch_key,
        keys=list(adatas.keys()),
        index_unique="_",
    )

    # Ensure batch_key is categorical
    adata_merged.obs[batch_key] = adata_merged.obs[batch_key].astype("category")

    print(f"  Merged shape: {adata_merged.shape}")
    print(f"  Cells per batch:")
    for batch, count in adata_merged.obs[batch_key].value_counts().items():
        print(f"    {batch}: {count}")

    return adata_merged


def save_adata_with_summary(
    adata: ad.AnnData,
    output_path: Path,
    batch_key: str,
    downsample_info: Optional[dict] = None,
):
    """Save AnnData and write summary file."""
    print(f"Saving to {output_path}...")
    adata.write_h5ad(output_path)

    summary_path = output_path.with_suffix(".summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Merge Summary ===\n\n")
        f.write(f"Total cells: {adata.n_obs}\n")
        f.write(f"Total genes: {adata.n_vars}\n")
        f.write(f"Batch key: {batch_key}\n\n")

        if downsample_info:
            f.write(f"Downsampling: {downsample_info.get('method', 'unknown')}\n")
            for k, v in downsample_info.items():
                if k != "method":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("Cells per batch:\n")
        for batch, count in adata.obs[batch_key].value_counts().items():
            f.write(f"  {batch}: {count}\n")

        # Cell type summary if available
        ct_col = downsample_info.get("celltype_column") if downsample_info else None
        if ct_col and ct_col in adata.obs.columns:
            f.write(f"\nCell types ({ct_col}):\n")
            for ct, count in adata.obs[ct_col].value_counts().items():
                f.write(f"  {ct}: {count}\n")

        f.write(f"\nOutput: {output_path}\n")

    print(f"  Summary: {summary_path}")


def ensure_list(value):
    """Convert a single value to a list if needed."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple h5ad files for integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge (full dataset only)
  python merge_datasets.py --config config.yaml

  # With random downsampling (multiple sizes)
  python merge_datasets.py --config config.yaml --downsample-random 10000 25000 50000

  # With cell type aware downsampling
  python merge_datasets.py --config config.yaml --downsample-celltype 500 1000

  # Both methods
  python merge_datasets.py --config config.yaml --downsample-random 50000 --downsample-celltype 1000
        """,
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
        help="Output file path for full dataset (default: merged.h5ad)",
    )
    # Downsampling arguments
    parser.add_argument(
        "--downsample-random",
        type=int,
        nargs="+",
        help="Target cell counts for random downsampling (can specify multiple)",
    )
    parser.add_argument(
        "--downsample-celltype",
        type=int,
        nargs="+",
        help="Cells per type for celltype_aware downsampling (can specify multiple)",
    )
    parser.add_argument(
        "--celltype-column",
        type=str,
        default="cell_type",
        help="Column for cell type annotations (default: cell_type)",
    )
    parser.add_argument(
        "--keep-unshared",
        action="store_true",
        help="Keep cell types not shared across all datasets",
    )
    parser.add_argument(
        "--min-cells-per-type",
        type=int,
        default=10,
        help="Min cells to keep a cell type (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip saving the full merged dataset (only save downsampled)",
    )

    args = parser.parse_args()

    # Initialize from config or CLI
    random_n_cells = []
    celltype_n_cells_per_type = []
    celltype_column = args.celltype_column
    keep_unshared = args.keep_unshared
    min_cells_per_type = args.min_cells_per_type
    seed = args.seed

    if args.config:
        config = load_config(args.config)
        files = [f["path"] for f in config["input"]["files"]]
        names = [f["name"] for f in config["input"]["files"]]
        batch_key = config["input"].get("batch_key", "dataset")
        output_dir = Path(config["output"]["dir"])
        output_path = output_dir / "merged.h5ad"
        join_type = "inner"

        # Get downsampling config
        ds_config = config.get("downsampling", {})
        seed = ds_config.get("seed", seed)

        # Random downsampling settings
        random_config = ds_config.get("random", {})
        random_n_cells = ensure_list(random_config.get("n_cells", []))

        # Cell type aware settings
        ct_config = ds_config.get("celltype_aware", {})
        celltype_n_cells_per_type = ensure_list(ct_config.get("n_cells_per_type", []))
        celltype_column = ct_config.get("celltype_column", celltype_column)
        keep_unshared = ct_config.get("keep_unshared", keep_unshared)
        min_cells_per_type = ct_config.get("min_cells_per_type", min_cells_per_type)
    else:
        if not args.files or not args.names:
            parser.error("Either --config or both --files and --names required")
        files = args.files
        names = args.names
        batch_key = args.batch_key
        output_path = Path(args.output)
        join_type = args.join

    # CLI args override config for downsampling
    if args.downsample_random:
        random_n_cells = args.downsample_random
    if args.downsample_celltype:
        celltype_n_cells_per_type = args.downsample_celltype

    # Create output directory
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load datasets
    # =========================================================================
    print("=" * 60)
    print("Loading datasets")
    print("=" * 60)
    adatas = load_datasets(files, names, batch_key)

    # =========================================================================
    # Save full merged dataset (always, unless --skip-full)
    # =========================================================================
    if not args.skip_full:
        print("\n" + "=" * 60)
        print("Saving full merged dataset")
        print("=" * 60)
        adata_full = merge_adatas(adatas, batch_key, join_type)
        save_adata_with_summary(adata_full, output_path, batch_key)
        del adata_full

    # =========================================================================
    # Random downsampling
    # =========================================================================
    for n_cells in random_n_cells:
        print("\n" + "=" * 60)
        print(f"Random downsampling: {n_cells} cells")
        print("=" * 60)

        adatas_ds = downsample_random(adatas, n_cells=n_cells, seed=seed)
        adata_ds = merge_adatas(adatas_ds, batch_key, join_type)

        # Output filename: merged_random_50000.h5ad
        ds_path = output_dir / f"merged_random_{n_cells}.h5ad"
        save_adata_with_summary(
            adata_ds,
            ds_path,
            batch_key,
            downsample_info={"method": "random", "n_cells": n_cells, "seed": seed},
        )
        del adatas_ds, adata_ds

    # =========================================================================
    # Cell type aware downsampling
    # =========================================================================
    for n_per_type in celltype_n_cells_per_type:
        print("\n" + "=" * 60)
        print(f"Cell type aware downsampling: {n_per_type} cells per type")
        print("=" * 60)

        adatas_ds = downsample_celltype_aware(
            adatas,
            celltype_column=celltype_column,
            n_cells_per_type=n_per_type,
            keep_unshared=keep_unshared,
            min_cells_per_type=min_cells_per_type,
            seed=seed,
        )
        adata_ds = merge_adatas(adatas_ds, batch_key, join_type)

        # Output filename: merged_celltype_1000pertype.h5ad
        ds_path = output_dir / f"merged_celltype_{n_per_type}pertype.h5ad"
        save_adata_with_summary(
            adata_ds,
            ds_path,
            batch_key,
            downsample_info={
                "method": "celltype_aware",
                "celltype_column": celltype_column,
                "n_cells_per_type": n_per_type,
                "keep_unshared": keep_unshared,
                "min_cells_per_type": min_cells_per_type,
                "seed": seed,
            },
        )
        del adatas_ds, adata_ds

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Done! Output files:")
    print("=" * 60)
    if not args.skip_full:
        print(f"  Full:     {output_path}")
    for n_cells in random_n_cells:
        print(f"  Random:   {output_dir / f'merged_random_{n_cells}.h5ad'}")
    for n_per_type in celltype_n_cells_per_type:
        print(f"  Celltype: {output_dir / f'merged_celltype_{n_per_type}pertype.h5ad'}")


if __name__ == "__main__":
    main()
