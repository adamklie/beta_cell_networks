"""
Preprocessing utilities for scRNA-seq integration.

Standard preprocessing pipeline following scanpy best practices.
"""

import scanpy as sc
import numpy as np
from anndata import AnnData
from typing import Optional, List


def normalize_and_log(
    adata: AnnData,
    target_sum: float = 1e4,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Normalize counts and apply log1p transformation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts.
    target_sum : float
        Target sum for normalization (default: 10,000).
    copy : bool
        If True, return a copy instead of modifying in place.

    Returns
    -------
    AnnData or None
        If copy=True, returns the modified AnnData object.
    """
    if copy:
        adata = adata.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if copy:
        return adata


def find_hvgs(
    adata: AnnData,
    n_top_genes: int = 3000,
    flavor: str = "seurat",
    batch_key: Optional[str] = None,
    subset: bool = False,
) -> None:
    """
    Identify highly variable genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized).
    n_top_genes : int
        Number of highly variable genes to select.
    flavor : str
        Method for HVG selection ('seurat', 'cell_ranger', 'seurat_v3').
    batch_key : str, optional
        If provided, HVGs are selected per batch and combined.
    subset : bool
        If True, subset adata to only HVGs.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key,
        subset=subset,
    )


def regress_and_scale(
    adata: AnnData,
    regress_vars: Optional[List[str]] = None,
    max_value: float = 10,
) -> None:
    """
    Regress out unwanted variation and scale data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized, HVGs identified).
    regress_vars : list of str, optional
        Variables to regress out (e.g., ['total_counts', 'pct_counts_mt']).
        If None, skips regression.
    max_value : float
        Max value after scaling (clip to this value).
    """
    if regress_vars is not None:
        # Check that all regress_vars exist in adata.obs
        missing = [v for v in regress_vars if v not in adata.obs.columns]
        if missing:
            raise ValueError(f"Variables not found in adata.obs: {missing}")
        sc.pp.regress_out(adata, regress_vars)

    sc.pp.scale(adata, max_value=max_value)


def run_pca(
    adata: AnnData,
    n_comps: int = 50,
    use_highly_variable: bool = True,
    svd_solver: str = "arpack",
    random_state: int = 0,
) -> None:
    """
    Run PCA on the data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (scaled).
    n_comps : int
        Number of principal components to compute.
    use_highly_variable : bool
        Whether to use only highly variable genes.
    svd_solver : str
        SVD solver to use ('arpack', 'randomized', 'auto').
    random_state : int
        Random seed for reproducibility.
    """
    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable,
        svd_solver=svd_solver,
        random_state=random_state,
    )


def standard_preprocess(
    adata: AnnData,
    n_hvgs: int = 3000,
    regress_vars: Optional[List[str]] = None,
    n_pcs: int = 50,
    target_sum: float = 1e4,
    max_scale_value: float = 10,
    copy: bool = False,
    random_state: int = 0,
) -> Optional[AnnData]:
    """
    Full preprocessing pipeline: normalize, HVGs, regress, scale, PCA.

    This follows the standard scanpy workflow used for integration methods.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts.
    n_hvgs : int
        Number of highly variable genes to select.
    regress_vars : list of str, optional
        Variables to regress out. If None, skips regression.
    n_pcs : int
        Number of principal components.
    target_sum : float
        Target sum for normalization.
    max_scale_value : float
        Max value after scaling.
    copy : bool
        If True, return a copy instead of modifying in place.
    random_state : int
        Random seed for PCA.

    Returns
    -------
    AnnData or None
        If copy=True, returns the modified AnnData object.

    Example
    -------
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> standard_preprocess(
    ...     adata,
    ...     n_hvgs=3000,
    ...     regress_vars=["total_counts", "pct_counts_mt"],
    ...     n_pcs=50,
    ... )
    >>> # adata now has .obsm['X_pca'] ready for integration
    """
    if copy:
        adata = adata.copy()

    # Step 1: Normalize and log-transform
    normalize_and_log(adata, target_sum=target_sum)

    # Step 2: Find highly variable genes
    find_hvgs(adata, n_top_genes=n_hvgs)

    # Step 3: Regress out covariates and scale
    regress_and_scale(adata, regress_vars=regress_vars, max_value=max_scale_value)

    # Step 4: PCA
    run_pca(adata, n_comps=n_pcs, random_state=random_state)

    if copy:
        return adata


def subset_to_hvgs(adata: AnnData, copy: bool = True) -> AnnData:
    """
    Subset AnnData to only highly variable genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with highly_variable column in .var.
    copy : bool
        If True (default), return a copy.

    Returns
    -------
    AnnData
        Subsetted AnnData object.
    """
    if "highly_variable" not in adata.var.columns:
        raise ValueError("Run find_hvgs() first to identify highly variable genes.")

    if copy:
        return adata[:, adata.var["highly_variable"]].copy()
    else:
        return adata[:, adata.var["highly_variable"]]


def store_raw_counts(adata: AnnData, layer_name: str = "counts") -> None:
    """
    Store raw counts in a layer before normalization.

    Useful for methods like scVI that need raw counts.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts in .X.
    layer_name : str
        Name of the layer to store counts in.
    """
    adata.layers[layer_name] = adata.X.copy()
