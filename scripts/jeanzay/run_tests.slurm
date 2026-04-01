#!/bin/bash
#SBATCH --job-name=depver-tests
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost           # CPU-only partition for tests
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_PROJECT_ID     # <-- CHANGE THIS
#
# Run depver unit tests (no GPU needed)
#

set -euo pipefail

PROJECT_DIR="$WORK/hops_parser"
ENV_DIR="$WORK/envs/depver"

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

cd "$PROJECT_DIR/depver"

echo "Running depver tests..."
python -m pytest tests/ -v --tb=short

echo "Done."
