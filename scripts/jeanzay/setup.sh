#!/bin/bash
#
# Setup script for DepVer on Jean Zay
# Run once to create the conda environment and install dependencies.
#
# Usage:
#   bash scripts/jeanzay/setup.sh
#

set -euo pipefail

echo "=== DepVer Setup on Jean Zay ==="

# Load modules
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.4.0  # adjust to latest available: module avail pytorch-gpu
module load git

# Create conda env (if not exists)
ENV_NAME="depver"
ENV_DIR="$WORK/envs/$ENV_NAME"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment at $ENV_DIR ..."
    conda create --prefix "$ENV_DIR" python=3.11 -y
fi

# Activate
conda activate "$ENV_DIR"

# Go to project directory
PROJECT_DIR="$WORK/hops_parser"
cd "$PROJECT_DIR"

# Install hopsparser
echo "Installing hopsparser..."
pip install -e "./hopsparser[tests]"

# Install depver with all extras
echo "Installing depver..."
pip install -e "./depver[all]"

# Download NLTK data for WordNet
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Prepare data directory
DATA_DIR="$WORK/depver_data"
mkdir -p "$DATA_DIR"

echo "Downloading datasets..."
python experiments/prepare_data.py --output-dir "$DATA_DIR" --dataset all

# Download hopsparser English model (if not already present)
MODEL_DIR="$WORK/depver_models"
mkdir -p "$MODEL_DIR"
echo ""
echo "=== Setup complete ==="
echo ""
echo "NOTE: You need to download or train a hopsparser model for English."
echo "  Option 1: Download a pretrained model from HuggingFace"
echo "    python -c \"from huggingface_hub import snapshot_download; snapshot_download('hopsparser/hopsparser-en_ewt-camembert', local_dir='$MODEL_DIR/en_ewt')\""
echo "  Option 2: Train your own model on UD English-EWT"
echo "    See hopsparser documentation for training instructions."
echo ""
echo "Data directory:  $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Environment:     $ENV_DIR"
