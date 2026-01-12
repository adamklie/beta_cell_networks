"""
scRNA-seq Integration Utilities

Reusable functions for preprocessing, integration, evaluation, and visualization
of single-cell RNA-seq data integration workflows.
"""

from .preprocessing import (
    standard_preprocess,
    normalize_and_log,
    find_hvgs,
    regress_and_scale,
    run_pca,
)

from .integration import (
    run_harmony_rpy2,
    run_scvi,
    run_scanorama,
)

from .evaluation import (
    compute_batch_mixing_silhouette,
    compute_cluster_purity,
    summarize_integration_metrics,
)

from .visualization import (
    plot_umap_grid,
    plot_method_comparison,
    plot_batch_distribution,
)

__all__ = [
    # Preprocessing
    "standard_preprocess",
    "normalize_and_log",
    "find_hvgs",
    "regress_and_scale",
    "run_pca",
    # Integration
    "run_harmony_rpy2",
    "run_scvi",
    "run_scanorama",
    # Evaluation
    "compute_batch_mixing_silhouette",
    "compute_cluster_purity",
    "summarize_integration_metrics",
    # Visualization
    "plot_umap_grid",
    "plot_method_comparison",
    "plot_batch_distribution",
]
