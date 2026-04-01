#!/bin/bash
#SBATCH --job-name=depver-tests
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --time=00:30:00
#SBATCH --account=jsl@cpu
#
# Unit tests (CPU only, no GPU needed)
#

set -euo pipefail

FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
PROJECT_DIR="$FSWORK/Projects/hops_parser"
ENV_DIR="$FSWORK/conda_related/envs/depver"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

cd "$PROJECT_DIR/depver"
echo "Running depver tests..."
python -m pytest tests/ -v --tb=short
echo "Done."
