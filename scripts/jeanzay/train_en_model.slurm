#!/bin/bash
#SBATCH --job-name=depver-train-en
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --time=06:00:00
#SBATCH --account=jsl@h100
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=h100
#
# Train hopsparser English model on UD English-EWT with roberta-base
# Estimated time: 2-4 hours on H100
#

set -euo pipefail

FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
PROJECT_DIR="$FSWORK/Projects/Hops_parser_torch"
ENV_DIR="$FSWORK/conda_related/envs/depver"
UD_DIR="$PROJECT_DIR/data/ud"
MODEL_OUTPUT="$PROJECT_DIR/models/en_ewt_roberta"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

echo "============================================"
echo "  Training English hopsparser model"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

cd "$PROJECT_DIR/hopsparser"

# Check treebank exists
TRAIN_FILE="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-train.conllu"
DEV_FILE="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-dev.conllu"
TEST_FILE="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-test.conllu"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Train file not found at $TRAIN_FILE"
    echo "Did you run the UD download step? See GUIDE_TO_RUN.md"
    exit 1
fi

echo "Train: $TRAIN_FILE"
echo "Dev:   $DEV_FILE"
echo "Test:  $TEST_FILE"
echo "Config: $PROJECT_DIR/scripts/jeanzay/config_en_roberta.yaml"
echo "Output: $MODEL_OUTPUT"
echo ""

mkdir -p "$MODEL_OUTPUT"

hopsparser train \
    "$PROJECT_DIR/scripts/jeanzay/config_en_roberta.yaml" \
    "$TRAIN_FILE" \
    "$MODEL_OUTPUT" \
    --dev-file "$DEV_FILE" \
    --test-file "$TEST_FILE" \
    --device cuda

echo ""
echo "============================================"
echo "  Training complete"
echo "  Model saved to: $MODEL_OUTPUT/model"
echo "  Finished: $(date)"
echo "============================================"
