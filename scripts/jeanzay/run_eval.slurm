#!/bin/bash
#SBATCH --job-name=depver-eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13           # V100 32GB; use gpu_p5 for A100
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_PROJECT_ID     # <-- CHANGE THIS to your IDRIS project
#SBATCH --qos=qos_gpu-t3             # up to 20h; use qos_gpu-t4 for longer
#
# Run DepVer evaluation on FRANK + AggreFact
#
# Usage:
#   sbatch scripts/jeanzay/run_eval.slurm
#
# Make sure to:
#   1. Replace YOUR_PROJECT_ID with your actual IDRIS project ID
#   2. Run scripts/jeanzay/setup.sh first
#   3. Have a hopsparser model at $WORK/depver_models/en_ewt
#

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="$WORK/hops_parser"
DATA_DIR="$WORK/depver_data"
MODEL_DIR="$WORK/depver_models/en_ewt"
RESULTS_DIR="$WORK/depver_results/$SLURM_JOB_ID"
ENV_DIR="$WORK/envs/depver"

# --- Load modules ---
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

# --- Run ---
cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:  $MODEL_DIR"
echo "Data:   $DATA_DIR"
echo "Output: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

python experiments/run_all.py \
    --model-path "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda

echo ""
echo "=== Done ==="
echo "Results at: $RESULTS_DIR"
