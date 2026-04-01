#!/bin/bash
#SBATCH --job-name=depver-quick
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --time=00:30:00
#SBATCH --account=jsl@h100
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=h100
#
# Quick smoke test: 50 examples to check everything works.
#

set -euo pipefail

FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
PROJECT_DIR="$FSWORK/Projects/Hops_parser_torch"
DATA_DIR="$FSWORK/depver_data"
MODEL_DIR="$FSWORK/depver_models/en_ewt_roberta/model"
RESULTS_DIR="$FSWORK/depver_results/quick_$SLURM_JOB_ID"
ENV_DIR="$FSWORK/conda_related/envs/depver"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

echo "=== Quick smoke test (50 examples) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

mkdir -p "$RESULTS_DIR"

# Tests first
cd "$PROJECT_DIR/depver"
python -m pytest tests/ -v --tb=short
cd "$PROJECT_DIR"

# Run with --max-examples 50
if [ -d "$DATA_DIR/frank" ]; then
    python experiments/eval_frank.py \
        --model-path "$MODEL_DIR" \
        --data-dir "$DATA_DIR/frank" \
        --output-dir "$RESULTS_DIR/frank_depver" \
        --device cuda \
        --max-examples 50
fi

echo ""
echo "=== Quick test done ==="
echo "Results: $RESULTS_DIR"
find "$RESULTS_DIR" -name "metrics.json" -exec cat {} \;
