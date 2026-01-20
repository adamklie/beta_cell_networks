# scRNA-seq Integration Pipeline

This pipeline merges and integrates scRNA-seq datasets from multiple sources using various batch correction methods.

## Quick Start (Test Run)

Run the full pipeline on a small test dataset (~5000 cells) to verify everything works:

```bash
cd scratch/2026_01_11/integration

# Step 1: Merge datasets and create downsampled test data
python scripts/merge_datasets.py --config config/sc-islet_integration_test.yaml

# Step 2: Run Seurat integration on the test data
Rscript scripts/seurat_integration.R \
    --input results/test/merged_celltype_500pertype.h5ad \
    --output results/test/seurat \
    --integration-key dataset \
    --plot-vars cell_type \
    --methods cca,rpca,harmony,fastmnn \
    --resolutions 0.2,0.5,0.8
```

## Directory Structure

```
integration/
├── config/
│   ├── example_config.yaml          # Template config with all options
│   └── sc-islet_integration_test.yaml  # Test config for sc-islet data
├── notebooks/
│   ├── 01_merge_datasets.ipynb      # Interactive merging + downsampling
│   └── 04_seurat_integrationR.ipynb # Interactive Seurat integration (R)
├── scripts/
│   ├── merge_datasets.py            # CLI for merging datasets
│   ├── seurat_integration.R         # CLI for Seurat integration (R)
│   └── submit_seurat_integration.sh # SLURM submission script
└── results/                         # Output directory
```

## Step-by-Step Guide

### 1. Merge Datasets

The merge step combines multiple h5ad files and optionally creates downsampled versions for testing.

**Using the notebook (interactive):**
```bash
jupyter notebook notebooks/01_merge_datasets.ipynb
```

**Using the CLI:**
```bash
# Basic merge (full dataset only)
python scripts/merge_datasets.py --config config/sc-islet_integration_test.yaml

# With downsampling via CLI
python scripts/merge_datasets.py \
    --config config/sc-islet_integration_test.yaml \
    --downsample-celltype 500 1000 \
    --celltype-column cell_type

# Direct file specification (no config)
python scripts/merge_datasets.py \
    --files /path/to/data1.h5ad /path/to/data2.h5ad \
    --names dataset1 dataset2 \
    --batch-key dataset \
    --output results/merged.h5ad \
    --downsample-celltype 500
```

**Outputs:**
- `merged.h5ad` - Full merged dataset (always created)
- `merged_random_{n}.h5ad` - Random downsampled versions
- `merged_celltype_{n}pertype.h5ad` - Cell type aware downsampled versions
- `*.summary.txt` - Summary for each output

### 2. Seurat Integration (R)

Run multiple Seurat integration methods (CCA, RPCA, Harmony, FastMNN) on the merged data.

**Using the notebook (interactive):**
```bash
# In R/RStudio
# Open notebooks/04_seurat_integrationR.ipynb
```

**Using the CLI (local):**
```bash
Rscript scripts/seurat_integration.R \
    --config config/sc-islet_integration_test.yaml

# Or with direct arguments
Rscript scripts/seurat_integration.R \
    --input results/test/merged_celltype_500pertype.h5ad \
    --output results/test/seurat \
    --integration-key dataset \
    --plot-vars cell_type,sample \
    --methods cca,rpca,harmony,fastmnn \
    --resolutions 0.2,0.5,0.8,1.0 \
    --n-hvgs 3000 \
    --n-pcs 50 \
    --threads 4
```

**Using SLURM (HPC):**
```bash
# Submit job
sbatch scripts/submit_seurat_integration.sh \
    --config config/sc-islet_integration_test.yaml

# With overrides
sbatch scripts/submit_seurat_integration.sh \
    --config config/sc-islet_integration_test.yaml \
    --input results/test/merged_celltype_500pertype.h5ad \
    --methods harmony,fastmnn

# Resume from checkpoint after failure
sbatch scripts/submit_seurat_integration.sh \
    --config config/sc-islet_integration_test.yaml \
    --checkpoint results/test/seurat/checkpoints/02_preprocessed.rds \
    --skip-to integrate
```

**Outputs:**
- `integrated_seurat.rds` - Full Seurat object with all reductions
- `integrated_seurat.h5seurat` - H5Seurat format (convertible to h5ad)
- `cell_metadata.tsv` - Cell metadata with cluster assignments
- `embeddings/` - CSV files with embeddings for each reduction
- `figures/` - Comparison plots (UMAPs, clusters)
- `checkpoints/` - Intermediate saves for resuming
- `session_info.txt` - R session info

### 3. Checkpoints and Resuming

The Seurat integration script saves checkpoints at key stages:

| Checkpoint | Contents | Resume with `--skip-to` |
|------------|----------|-------------------------|
| `01_loaded.rds` | Seurat object after h5ad load | `preprocess` |
| `02_preprocessed.rds` | After normalize/HVG/PCA/uncorrected UMAP | `integrate` |
| `03_integrated.rds` | After all integration methods | `umap` |

**Resume example:**
```bash
Rscript scripts/seurat_integration.R \
    --checkpoint results/seurat/checkpoints/02_preprocessed.rds \
    --skip-to integrate \
    --output results/seurat
```

## Configuration Reference

See `config/example_config.yaml` for all available options. Key sections:

```yaml
input:
  files:                    # List of h5ad files to merge
  h5ad_path:               # Path to merged h5ad for integration
  batch_key: "dataset"     # Column for batch labels

downsampling:
  seed: 42
  random:
    n_cells: [10000, 50000]  # Target cell counts
  celltype_aware:
    celltype_column: "cell_type"
    n_cells_per_type: [500, 1000]
    keep_unshared: false

preprocessing:
  n_top_genes: 3000
  n_pcs: 50

integration:
  key: "dataset"           # Column for batch correction
  plot_vars:               # Additional variables to plot
    - "cell_type"
  seurat:
    methods: [cca, rpca, harmony, fastmnn]

clustering:
  resolutions: [0.2, 0.5, 0.8, 1.0]
  n_neighbors: 30

output:
  dir: "./results/"
```

## Tips

1. **Test first**: Always run on downsampled data first to catch errors early
2. **Memory**: Full dataset integration requires significant RAM (256GB+ for 300k cells)
3. **Time**: CCA/RPCA are slowest; Harmony is fastest
4. **Checkpoints**: Use `--checkpoint` to resume if jobs fail on HPC
5. **Methods**: Start with Harmony for quick results, then compare with CCA/RPCA

## Troubleshooting

**Out of memory:**
- Reduce `--n-pcs` or `--n-hvgs`
- Use downsampled data for testing
- Request more memory in SLURM (`--mem=512G`)

**Integration fails:**
- Check that `batch_key` exists in the data
- Ensure at least 2 batches are present
- Try running methods individually with `--methods harmony`

**h5ad loading errors:**
- Ensure `zellkonverter` R package is installed
- Check h5ad was saved with compatible version
