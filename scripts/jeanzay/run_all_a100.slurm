#!/bin/bash
#SBATCH --job-name=depver-all
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --time=04:00:00
#SBATCH --account=jsl@h100
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=h100
#
# DepVer: Full evaluation pipeline on a single H100
#

set -euo pipefail

# --- Configuration ---
FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
PROJECT_DIR="$FSWORK/Projects/Hops_parser_torch"
DATA_DIR="$FSWORK/depver_data"
MODEL_DIR="$FSWORK/depver_models/en_ewt"
RESULTS_DIR="$FSWORK/depver_results/$SLURM_JOB_ID"
ENV_DIR="$FSWORK/conda_related/envs/depver"

# --- Setup ---
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

echo "============================================"
echo "  DepVer Full Evaluation"
echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Model:     $MODEL_DIR"
echo "Data:      $DATA_DIR"
echo "Results:   $RESULTS_DIR"
echo "Started:   $(date)"
echo "============================================"
echo ""

mkdir -p "$RESULTS_DIR"

# --- Step 1: Unit tests ---
echo ""
echo ">>> Step 1/5: Running unit tests..."
cd "$PROJECT_DIR/depver"
python -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_output.txt"
cd "$PROJECT_DIR"
echo "    Tests done."

# --- Step 2: DepVer on FRANK ---
echo ""
echo ">>> Step 2/5: DepVer on FRANK..."
if [ -d "$DATA_DIR/frank" ]; then
    python experiments/eval_frank.py \
        --model-path "$MODEL_DIR" \
        --data-dir "$DATA_DIR/frank" \
        --output-dir "$RESULTS_DIR/frank_depver" \
        --device cuda
else
    echo "    SKIP: $DATA_DIR/frank not found"
fi

# --- Step 3: DepVer on AggreFact ---
echo ""
echo ">>> Step 3/5: DepVer on AggreFact..."
if [ -d "$DATA_DIR/aggrefact" ]; then
    python experiments/eval_aggrefact.py \
        --model-path "$MODEL_DIR" \
        --data-dir "$DATA_DIR/aggrefact" \
        --output-dir "$RESULTS_DIR/aggrefact_depver" \
        --device cuda
else
    echo "    SKIP: $DATA_DIR/aggrefact not found"
fi

# --- Step 4: BERTScore baselines ---
echo ""
echo ">>> Step 4/5: BERTScore baselines..."
if [ -d "$DATA_DIR/frank" ]; then
    python experiments/baselines/run_bertscore.py \
        --data-dir "$DATA_DIR/frank" \
        --output-dir "$RESULTS_DIR/frank_bertscore"
fi
if [ -d "$DATA_DIR/aggrefact" ]; then
    python experiments/baselines/run_bertscore.py \
        --data-dir "$DATA_DIR/aggrefact" \
        --output-dir "$RESULTS_DIR/aggrefact_bertscore"
fi

# --- Step 5: Comparison ---
echo ""
echo ">>> Step 5/5: Comparison summary..."

echo ""
echo "============================================"
echo "  DONE"
echo "  Results: $RESULTS_DIR"
echo "  Finished: $(date)"
echo "============================================"

# Print all metrics
echo ""
echo ">>> All metrics:"
find "$RESULTS_DIR" -name "metrics.json" -exec echo "--- {} ---" \; -exec cat {} \; -exec echo "" \;
