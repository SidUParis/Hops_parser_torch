#!/bin/bash
#
# Setup script for DepVer on Jean Zay
# Run once on a login node to install dependencies and download data.
#
# Usage:
#   bash scripts/jeanzay/setup.sh
#

set -euo pipefail

FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
ENV_DIR="$FSWORK/conda_related/envs/depver"
PROJECT_DIR="$FSWORK/Projects/Hops_parser_torch"

echo "=== DepVer Setup on Jean Zay ==="
echo "Environment: $ENV_DIR"
echo "Project:     $PROJECT_DIR"
echo ""

# Load modules
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

# Activate env (should already exist from conda create)
conda activate "$ENV_DIR"

cd "$PROJECT_DIR"

# Install hopsparser
echo ">>> Installing hopsparser..."
pip install -e "./hopsparser"

# Install depver with all extras
echo ">>> Installing depver..."
pip install -e "./depver[all]"

# Download NLTK data for WordNet
echo ">>> Downloading NLTK data..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Prepare data
DATA_DIR="$FSWORK/depver_data"
mkdir -p "$DATA_DIR"
echo ">>> Downloading datasets to $DATA_DIR ..."
python experiments/prepare_data.py --output-dir "$DATA_DIR" --dataset all

# Model directory
MODEL_DIR="$FSWORK/depver_models"
mkdir -p "$MODEL_DIR"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next: download a hopsparser English model:"
echo "  python -c \""
echo "  from huggingface_hub import snapshot_download"
echo "  snapshot_download('hopsparser/hopsparser-en_ewt-camembert', local_dir='$MODEL_DIR/en_ewt')"
echo "  \""
echo ""
echo "Then submit:"
echo "  sbatch scripts/jeanzay/run_all_a100.slurm"
