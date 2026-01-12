"""
Visualization utilities for scRNA-seq integration.

Provides functions for UMAP plots, method comparisons, and batch distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path


def plot_umap_grid(
    adata: AnnData,
    color_keys: List[str],
    basis: str = "X_umap",
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot UMAP colored by multiple variables in a grid.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with UMAP in .obsm.
    color_keys : list of str
        Columns in .obs to color by.
    basis : str
        Key in .obsm for UMAP coordinates.
    ncols : int
        Number of columns in grid.
    figsize : tuple, optional
        Figure size. If None, auto-calculated.
    title_prefix : str
        Prefix for subplot titles.
    save_path : str, optional
        Path to save figure.
    palette : dict, optional
        Color palette mapping values to colors.
    **kwargs
        Additional arguments passed to sc.pl.embedding.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_plots = len(color_keys)
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, color in enumerate(color_keys):
        ax = axes[i]
        sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            ax=ax,
            show=False,
            palette=palette,
            title=f"{title_prefix}{color}" if title_prefix else color,
            **kwargs,
        )

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_method_comparison(
    adata: AnnData,
    embeddings: Dict[str, str],
    color_by: str,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None,
    point_size: float = 1,
    **kwargs,
) -> plt.Figure:
    """
    Side-by-side comparison of multiple integration methods.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with multiple UMAP embeddings.
    embeddings : dict
        Mapping from method name to obsm key for UMAP
        (e.g., {'Harmony': 'X_umap_harmony', 'scVI': 'X_umap_scvi'}).
    color_by : str
        Column in .obs to color points by.
    ncols : int
        Number of columns.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.
    palette : dict, optional
        Color palette.
    point_size : float
        Size of points.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = list(embeddings.keys())
    n_methods = len(methods)
    nrows = int(np.ceil(n_methods / ncols))

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        embed_key = embeddings[method]

        if embed_key not in adata.obsm:
            ax.set_title(f"{method}\n(embedding not found)")
            ax.axis("off")
            continue

        coords = adata.obsm[embed_key]
        colors = adata.obs[color_by]

        # Handle categorical vs continuous
        if pd.api.types.is_categorical_dtype(colors) or colors.dtype == object:
            unique_vals = colors.unique()
            if palette:
                color_map = [palette.get(v, "#999999") for v in colors]
            else:
                cmap = plt.cm.get_cmap("tab20", len(unique_vals))
                val_to_color = {v: cmap(j) for j, v in enumerate(unique_vals)}
                color_map = [val_to_color[v] for v in colors]

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_map,
                s=point_size,
                alpha=0.6,
                **kwargs,
            )
        else:
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                s=point_size,
                alpha=0.6,
                cmap="viridis",
                **kwargs,
            )
            plt.colorbar(scatter, ax=ax)

        ax.set_title(method)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n_methods, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_batch_distribution(
    adata: AnnData,
    batch_key: str,
    cluster_key: str,
    normalize: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Plot batch composition per cluster as stacked bar chart.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        Column in .obs with batch labels.
    cluster_key : str
        Column in .obs with cluster labels.
    normalize : bool
        If True, show proportions; else show counts.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    palette : dict, optional
        Color palette for batches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Create crosstab
    ct = pd.crosstab(
        adata.obs[cluster_key],
        adata.obs[batch_key],
        normalize="index" if normalize else False,
    )

    # Sort clusters
    try:
        ct = ct.loc[sorted(ct.index, key=lambda x: int(x))]
    except (ValueError, TypeError):
        ct = ct.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if palette:
        colors = [palette.get(col, "#999999") for col in ct.columns]
        ct.plot(kind="bar", stacked=True, ax=ax, color=colors)
    else:
        ct.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel(cluster_key)
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(f"Batch composition per {cluster_key}")
    ax.legend(title=batch_key, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing integration metrics across methods.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with methods as index and metrics as columns.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_metrics = len(metrics_df.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_df.columns):
        ax = axes[i]
        metrics_df[metric].plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_umap_by_category(
    adata: AnnData,
    category_key: str,
    basis: str = "X_umap",
    output_dir: str = "figures",
    figsize: Tuple[float, float] = (6, 6),
    point_size: float = 1,
    palette: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save individual UMAP plots for each category value.

    Useful for highlighting specific cell types or conditions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    category_key : str
        Column in .obs to split by.
    basis : str
        Key in .obsm for UMAP coordinates.
    output_dir : str
        Directory to save figures.
    figsize : tuple
        Figure size.
    point_size : float
        Point size.
    palette : dict, optional
        Color palette.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coords = adata.obsm[basis]
    categories = adata.obs[category_key].unique()

    for cat in categories:
        fig, ax = plt.subplots(figsize=figsize)

        # Background points (gray)
        mask = adata.obs[category_key] != cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c="#EEEEEE",
            s=point_size,
            alpha=0.5,
        )

        # Highlighted points
        mask = adata.obs[category_key] == cat
        color = palette.get(cat, "#E41A1C") if palette else "#E41A1C"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=point_size * 2,
            alpha=0.8,
            label=cat,
        )

        ax.set_title(f"{category_key}: {cat}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

        # Clean filename
        clean_name = str(cat).replace("/", "_").replace(" ", "_")
        save_path = output_path / f"umap_{category_key}_{clean_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
