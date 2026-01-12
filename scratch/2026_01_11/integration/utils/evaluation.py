"""
Evaluation metrics for scRNA-seq integration quality.

Includes metrics for:
- Batch mixing (how well batches are integrated)
- Biological conservation (how well cell types are preserved)
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Optional, Dict, List
from sklearn.metrics import silhouette_score, adjusted_rand_score


def compute_batch_mixing_silhouette(
    adata: AnnData,
    batch_key: str,
    use_rep: str,
    sample_size: Optional[int] = None,
    random_state: int = 0,
) -> float:
    """
    Compute silhouette score for batch mixing.

    Lower (more negative) scores indicate better batch mixing,
    as cells from different batches are more intermingled.
    We return 1 - silhouette to make higher = better mixing.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        Column in .obs with batch labels.
    use_rep : str
        Key in .obsm with embedding to evaluate.
    sample_size : int, optional
        If provided, subsample cells for faster computation.
    random_state : int
        Random seed for subsampling.

    Returns
    -------
    float
        Batch mixing score (higher = better mixing, range -1 to 1).
    """
    if use_rep not in adata.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found in adata.obsm")

    embedding = adata.obsm[use_rep]
    batch_labels = adata.obs[batch_key].values

    # Subsample if requested
    if sample_size is not None and sample_size < len(adata):
        np.random.seed(random_state)
        idx = np.random.choice(len(adata), sample_size, replace=False)
        embedding = embedding[idx]
        batch_labels = batch_labels[idx]

    # Compute silhouette score
    # Silhouette ranges from -1 to 1, where:
    # - High positive: well-separated batches (bad for integration)
    # - Near 0: overlapping batches (good for integration)
    # - Negative: mixed batches (very good for integration)
    sil_score = silhouette_score(embedding, batch_labels)

    # Convert so higher = better mixing
    batch_mixing_score = 1 - sil_score

    return batch_mixing_score


def compute_cluster_purity(
    adata: AnnData,
    cluster_key: str,
    label_key: str,
) -> float:
    """
    Compute cluster purity (how well clusters match biological labels).

    Higher purity indicates better biological conservation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_key : str
        Column in .obs with cluster assignments.
    label_key : str
        Column in .obs with ground truth labels (e.g., cell type).

    Returns
    -------
    float
        Purity score (range 0 to 1, higher = better).
    """
    clusters = adata.obs[cluster_key].values
    labels = adata.obs[label_key].values

    # For each cluster, find the most common label
    cluster_ids = np.unique(clusters)
    total_correct = 0

    for cluster in cluster_ids:
        mask = clusters == cluster
        cluster_labels = labels[mask]

        # Most common label in this cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        most_common_count = counts.max()
        total_correct += most_common_count

    purity = total_correct / len(labels)
    return purity


def compute_ari(
    adata: AnnData,
    cluster_key: str,
    label_key: str,
) -> float:
    """
    Compute Adjusted Rand Index between clusters and labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_key : str
        Column in .obs with cluster assignments.
    label_key : str
        Column in .obs with ground truth labels.

    Returns
    -------
    float
        ARI score (range -1 to 1, higher = better agreement).
    """
    return adjusted_rand_score(
        adata.obs[label_key].values,
        adata.obs[cluster_key].values,
    )


def compute_batch_entropy(
    adata: AnnData,
    batch_key: str,
    use_rep: str,
    n_neighbors: int = 50,
    sample_size: Optional[int] = 5000,
    random_state: int = 0,
) -> float:
    """
    Compute batch entropy in local neighborhoods.

    Higher entropy indicates better batch mixing.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        Column in .obs with batch labels.
    use_rep : str
        Key in .obsm with embedding.
    n_neighbors : int
        Number of neighbors to consider for each cell.
    sample_size : int, optional
        Subsample cells for faster computation.
    random_state : int
        Random seed.

    Returns
    -------
    float
        Mean batch entropy (range 0 to log(n_batches), higher = better).
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import entropy

    embedding = adata.obsm[use_rep]
    batch_labels = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    # Subsample
    if sample_size is not None and sample_size < len(adata):
        np.random.seed(random_state)
        idx = np.random.choice(len(adata), sample_size, replace=False)
        embedding_sample = embedding[idx]
        batch_labels_sample = batch_labels[idx]
    else:
        embedding_sample = embedding
        batch_labels_sample = batch_labels

    # Fit nearest neighbors on full data
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(embedding)

    # Find neighbors for sampled points
    _, indices = nn.kneighbors(embedding_sample)

    # Compute entropy for each cell's neighborhood
    entropies = []
    for i, neighbor_idx in enumerate(indices):
        # Skip self (first neighbor)
        neighbor_idx = neighbor_idx[1:]
        neighbor_batches = batch_labels[neighbor_idx]

        # Count batches
        counts = np.array(
            [(neighbor_batches == b).sum() for b in unique_batches]
        )
        probs = counts / counts.sum()
        ent = entropy(probs, base=2)
        entropies.append(ent)

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log2(n_batches)
    mean_entropy = np.mean(entropies) / max_entropy if max_entropy > 0 else 0

    return mean_entropy


def summarize_integration_metrics(
    adata: AnnData,
    batch_key: str,
    label_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    use_rep: str = "X_pca",
    sample_size: int = 5000,
) -> Dict[str, float]:
    """
    Compute a summary of integration quality metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with integration results.
    batch_key : str
        Column in .obs with batch labels.
    label_key : str, optional
        Column in .obs with cell type labels.
    cluster_key : str, optional
        Column in .obs with cluster assignments.
    use_rep : str
        Key in .obsm with embedding to evaluate.
    sample_size : int
        Sample size for expensive computations.

    Returns
    -------
    dict
        Dictionary with metric names and values.
    """
    metrics = {}

    # Batch mixing metrics
    metrics["batch_silhouette"] = compute_batch_mixing_silhouette(
        adata, batch_key, use_rep, sample_size=sample_size
    )
    metrics["batch_entropy"] = compute_batch_entropy(
        adata, batch_key, use_rep, sample_size=sample_size
    )

    # Biological conservation metrics (if labels available)
    if label_key is not None and label_key in adata.obs.columns:
        if cluster_key is not None and cluster_key in adata.obs.columns:
            metrics["cluster_purity"] = compute_cluster_purity(
                adata, cluster_key, label_key
            )
            metrics["ari"] = compute_ari(adata, cluster_key, label_key)

    return metrics


def compare_integration_methods(
    adata: AnnData,
    batch_key: str,
    embeddings: Dict[str, str],
    label_key: Optional[str] = None,
    cluster_keys: Optional[Dict[str, str]] = None,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Compare multiple integration methods.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with multiple integration embeddings.
    batch_key : str
        Column in .obs with batch labels.
    embeddings : dict
        Mapping from method name to obsm key (e.g., {'Harmony': 'X_harmony'}).
    label_key : str, optional
        Column in .obs with cell type labels.
    cluster_keys : dict, optional
        Mapping from method name to cluster column in obs.
    sample_size : int
        Sample size for expensive computations.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each method.
    """
    results = []

    for method_name, embed_key in embeddings.items():
        if embed_key not in adata.obsm:
            print(f"Warning: {embed_key} not found, skipping {method_name}")
            continue

        cluster_key = None
        if cluster_keys is not None:
            cluster_key = cluster_keys.get(method_name)

        metrics = summarize_integration_metrics(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            cluster_key=cluster_key,
            use_rep=embed_key,
            sample_size=sample_size,
        )
        metrics["method"] = method_name
        results.append(metrics)

    df = pd.DataFrame(results)
    if "method" in df.columns:
        df = df.set_index("method")

    return df
