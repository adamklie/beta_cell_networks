#!/bin/bash
#SBATCH --job-name=seurat_integration
#SBATCH --output=seurat_integration_%j.out
#SBATCH --error=seurat_integration_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# =============================================================================
# SLURM Submission Script for Seurat Integration
# =============================================================================
# Usage:
#   sbatch submit_seurat_integration.sh --config ../config/example_config.yaml
#
# To resume from a checkpoint:
#   sbatch submit_seurat_integration.sh --config ../config/example_config.yaml \
#       --checkpoint /path/to/checkpoints/03_integrated.rds --skip-to umap
#
# Override config values:
#   sbatch submit_seurat_integration.sh --config ../config/example_config.yaml \
#       --methods harmony,fastmnn
# =============================================================================

# Load modules (adjust for your HPC)
module load R/4.3.0  # or your R version
# module load hdf5    # if needed

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values (override via command line or config)
CONFIG=""
INPUT=""
OUTPUT=""
INTEGRATION_KEY=""
PLOT_VARS=""
N_HVGS=""
N_PCS=""
N_DIMS=""
RESOLUTIONS=""
METHODS=""
CHECKPOINT=""
SKIP_TO=""
THREADS=${SLURM_CPUS_PER_TASK:-8}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --integration-key)
      INTEGRATION_KEY="$2"
      shift 2
      ;;
    --plot-vars)
      PLOT_VARS="$2"
      shift 2
      ;;
    --n-hvgs)
      N_HVGS="$2"
      shift 2
      ;;
    --n-pcs)
      N_PCS="$2"
      shift 2
      ;;
    --n-dims)
      N_DIMS="$2"
      shift 2
      ;;
    --resolutions)
      RESOLUTIONS="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --skip-to)
      SKIP_TO="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Validation
if [[ -z "${CONFIG}" && -z "${INPUT}" ]]; then
  echo "ERROR: Must provide --config or --input"
  exit 1
fi

# Print job info
echo "========================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================"
if [[ -n "${CONFIG}" ]]; then echo "Config: ${CONFIG}"; fi
if [[ -n "${INPUT}" ]]; then echo "Input: ${INPUT}"; fi
if [[ -n "${OUTPUT}" ]]; then echo "Output: ${OUTPUT}"; fi
if [[ -n "${METHODS}" ]]; then echo "Methods: ${METHODS}"; fi
if [[ -n "${CHECKPOINT}" ]]; then echo "Checkpoint: ${CHECKPOINT}"; fi
if [[ -n "${SKIP_TO}" ]]; then echo "Skip to: ${SKIP_TO}"; fi
echo "Threads: ${THREADS}"
echo "========================================"

# Build command
CMD="Rscript ${SCRIPT_DIR}/seurat_integration.R"

# Add arguments
[[ -n "${CONFIG}" ]] && CMD="${CMD} --config ${CONFIG}"
[[ -n "${INPUT}" ]] && CMD="${CMD} --input ${INPUT}"
[[ -n "${OUTPUT}" ]] && CMD="${CMD} --output ${OUTPUT}"
[[ -n "${INTEGRATION_KEY}" ]] && CMD="${CMD} --integration-key ${INTEGRATION_KEY}"
[[ -n "${PLOT_VARS}" ]] && CMD="${CMD} --plot-vars ${PLOT_VARS}"
[[ -n "${N_HVGS}" ]] && CMD="${CMD} --n-hvgs ${N_HVGS}"
[[ -n "${N_PCS}" ]] && CMD="${CMD} --n-pcs ${N_PCS}"
[[ -n "${N_DIMS}" ]] && CMD="${CMD} --n-dims ${N_DIMS}"
[[ -n "${RESOLUTIONS}" ]] && CMD="${CMD} --resolutions ${RESOLUTIONS}"
[[ -n "${METHODS}" ]] && CMD="${CMD} --methods ${METHODS}"
[[ -n "${CHECKPOINT}" ]] && CMD="${CMD} --checkpoint ${CHECKPOINT}"
[[ -n "${SKIP_TO}" ]] && CMD="${CMD} --skip-to ${SKIP_TO}"
CMD="${CMD} --threads ${THREADS}"

# Run
echo "Running: ${CMD}"
echo "========================================"

${CMD}

EXIT_CODE=$?

echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
