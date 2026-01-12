"""
Integration method wrappers for scRNA-seq batch correction.

Includes wrappers for:
- Harmony (via R/rpy2)
- scVI (variational autoencoder)
- Scanorama (panoramic stitching)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Optional, List, Union


def run_harmony_rpy2(
    adata: AnnData,
    batch_key: str,
    use_rep: str = "X_pca",
    theta: Optional[float] = None,
    key_added: Optional[str] = None,
) -> np.ndarray:
    """
    Run Harmony batch correction via rpy2.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA computed in .obsm[use_rep].
    batch_key : str
        Column in .obs containing batch labels.
    use_rep : str
        Key in .obsm for input embedding (default: 'X_pca').
    theta : float, optional
        Diversity clustering penalty parameter for Harmony.
        Higher values = more aggressive correction.
    key_added : str, optional
        Key to store result in .obsm. Default: 'X_harmony_{batch_key}'.

    Returns
    -------
    np.ndarray
        Harmonized embedding matrix (n_cells x n_dims).

    Notes
    -----
    Requires R with harmony package installed and rpy2 configured.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError:
        raise ImportError(
            "rpy2 is required for Harmony. Install with: pip install rpy2"
        )

    # Import R harmony package
    try:
        harmony = importr("harmony")
    except Exception:
        raise ImportError(
            "R harmony package not found. Install in R with: "
            "install.packages('harmony')"
        )

    if use_rep not in adata.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found in adata.obsm")

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    # Extract PCA embedding and batch labels
    pca_embed = adata.obsm[use_rep]
    n_pcs = pca_embed.shape[1]

    # Create column names for R
    pc_names = [f"PC{i}" for i in range(1, n_pcs + 1)]
    pc_df = pd.DataFrame(pca_embed, columns=pc_names, index=adata.obs_names)

    # Get batch labels
    batch_labels = adata.obs[batch_key].values

    # Convert to R objects
    r_pc = ro.conversion.py2rpy(pc_df)
    r_batch = ro.StrVector(batch_labels.astype(str))

    # Build harmony arguments
    harmony_args = {
        "data_mat": r_pc,
        "meta_data": r_batch,
        "vars_use": batch_key,
        "do_pca": False,
    }
    if theta is not None:
        harmony_args["theta"] = theta

    # Run Harmony
    r_harmony = ro.r["HarmonyMatrix"]
    harmonized = r_harmony(**harmony_args)

    # Convert back to numpy
    harmonized_np = np.array(harmonized)

    # Store in adata
    if key_added is None:
        key_added = f"X_harmony_{batch_key}"
    adata.obsm[key_added] = harmonized_np

    return harmonized_np


def run_scvi(
    adata: AnnData,
    batch_key: str,
    n_latent: int = 30,
    n_layers: int = 2,
    n_epochs: Optional[int] = None,
    early_stopping: bool = True,
    use_gpu: bool = True,
    key_added: str = "X_scVI",
    model_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Run scVI integration.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts in .X or .layers['counts'].
    batch_key : str
        Column in .obs containing batch labels.
    n_latent : int
        Dimensionality of latent space.
    n_layers : int
        Number of hidden layers in encoder/decoder.
    n_epochs : int, optional
        Max training epochs. If None, uses scVI defaults with early stopping.
    early_stopping : bool
        Whether to use early stopping.
    use_gpu : bool
        Whether to use GPU if available.
    key_added : str
        Key to store latent representation in .obsm.
    model_dir : str, optional
        Directory to save the trained model.

    Returns
    -------
    np.ndarray
        Latent representation (n_cells x n_latent).

    Notes
    -----
    Requires scvi-tools: pip install scvi-tools
    """
    try:
        import scvi
    except ImportError:
        raise ImportError(
            "scvi-tools is required. Install with: pip install scvi-tools"
        )

    # Ensure we have raw counts
    if "counts" in adata.layers:
        adata_scvi = adata.copy()
        adata_scvi.X = adata_scvi.layers["counts"]
    else:
        adata_scvi = adata.copy()

    # Setup AnnData for scVI
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        batch_key=batch_key,
    )

    # Create and train model
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_layers=n_layers,
    )

    train_kwargs = {}
    if n_epochs is not None:
        train_kwargs["max_epochs"] = n_epochs
    if early_stopping:
        train_kwargs["early_stopping"] = True

    # Set accelerator
    if use_gpu:
        try:
            model.train(accelerator="gpu", **train_kwargs)
        except Exception:
            print("GPU not available, falling back to CPU")
            model.train(accelerator="cpu", **train_kwargs)
    else:
        model.train(accelerator="cpu", **train_kwargs)

    # Get latent representation
    latent = model.get_latent_representation()

    # Store in original adata
    adata.obsm[key_added] = latent

    # Save model if requested
    if model_dir is not None:
        model.save(model_dir, overwrite=True)

    return latent


def run_scanorama(
    adata: AnnData,
    batch_key: str,
    key_added: str = "X_scanorama",
    knn: int = 20,
    sigma: float = 15,
    approx: bool = True,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Run Scanorama integration.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized, HVGs computed).
    batch_key : str
        Column in .obs containing batch labels.
    key_added : str
        Key to store corrected embedding in .obsm.
    knn : int
        Number of nearest neighbors for matching.
    sigma : float
        Correction smoothing parameter.
    approx : bool
        Use approximate nearest neighbors (faster).
    alpha : float
        Alignment score minimum cutoff.

    Returns
    -------
    np.ndarray
        Corrected embedding (n_cells x n_genes or n_cells x n_dims).

    Notes
    -----
    Requires scanorama: pip install scanorama
    """
    try:
        import scanorama
    except ImportError:
        raise ImportError(
            "scanorama is required. Install with: pip install scanorama"
        )

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    # Split by batch
    batches = adata.obs[batch_key].unique()
    adatas = [adata[adata.obs[batch_key] == b].copy() for b in batches]

    # Run Scanorama integration
    scanorama.integrate_scanpy(
        adatas,
        knn=knn,
        sigma=sigma,
        approx=approx,
        alpha=alpha,
    )

    # Concatenate results back
    adata_integrated = adatas[0].concatenate(
        adatas[1:],
        batch_key="_scanorama_batch",
        index_unique=None,
    )

    # Reorder to match original
    adata_integrated = adata_integrated[adata.obs_names]

    # Store corrected embedding
    adata.obsm[key_added] = adata_integrated.obsm["X_scanorama"]

    return adata.obsm[key_added]


def compute_neighbors_and_umap(
    adata: AnnData,
    use_rep: str,
    n_neighbors: int = 30,
    metric: str = "cosine",
    n_pcs: Optional[int] = None,
    random_state: int = 0,
    key_added_suffix: Optional[str] = None,
) -> None:
    """
    Compute neighbors graph and UMAP embedding on a given representation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_rep : str
        Key in .obsm to use for neighbor computation.
    n_neighbors : int
        Number of neighbors for the graph.
    metric : str
        Distance metric ('euclidean', 'cosine', etc.).
    n_pcs : int, optional
        Number of dimensions to use from the representation.
    random_state : int
        Random seed.
    key_added_suffix : str, optional
        Suffix for storing results. If provided, stores neighbors in
        'neighbors_{suffix}' and UMAP in 'X_umap_{suffix}'.
    """
    # Compute neighbors
    sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        metric=metric,
        n_pcs=n_pcs,
        random_state=random_state,
        key_added=f"neighbors_{key_added_suffix}" if key_added_suffix else "neighbors",
    )

    # Compute UMAP
    umap_key = f"X_umap_{key_added_suffix}" if key_added_suffix else "X_umap"
    neighbors_key = (
        f"neighbors_{key_added_suffix}" if key_added_suffix else "neighbors"
    )

    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        random_state=random_state,
    )

    # Rename UMAP if suffix provided
    if key_added_suffix:
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()


def run_leiden_clustering(
    adata: AnnData,
    resolutions: List[float] = [0.2, 0.5, 0.8, 1.0],
    neighbors_key: Optional[str] = None,
    key_prefix: str = "leiden",
    random_state: int = 0,
) -> None:
    """
    Run Leiden clustering at multiple resolutions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors computed.
    resolutions : list of float
        Resolution parameters to test.
    neighbors_key : str, optional
        Key for neighbors graph. If None, uses default.
    key_prefix : str
        Prefix for cluster column names in .obs.
    random_state : int
        Random seed.
    """
    for res in resolutions:
        key_added = f"{key_prefix}_{res}"
        sc.tl.leiden(
            adata,
            resolution=res,
            neighbors_key=neighbors_key,
            key_added=key_added,
            random_state=random_state,
        )
