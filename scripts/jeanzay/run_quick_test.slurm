#!/bin/bash
#SBATCH --job-name=depver-quick
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_PROJECT_ID     # <-- CHANGE THIS
#SBATCH --qos=qos_gpu-dev            # dev QOS: max 2h, higher priority
#
# Quick smoke test: run DepVer on 50 examples to check everything works.
#

set -euo pipefail

PROJECT_DIR="$WORK/hops_parser"
DATA_DIR="$WORK/depver_data"
MODEL_DIR="$WORK/depver_models/en_ewt"
RESULTS_DIR="$WORK/depver_results/quick_test_$SLURM_JOB_ID"
ENV_DIR="$WORK/envs/depver"

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

cd "$PROJECT_DIR"

echo "=== Quick smoke test (50 examples) ==="

mkdir -p "$RESULTS_DIR"

python experiments/run_all.py \
    --model-path "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda \
    --max-examples 50

echo ""
echo "=== Quick test done ==="
echo "Results at: $RESULTS_DIR"
