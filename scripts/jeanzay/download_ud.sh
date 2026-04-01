#!/bin/bash
#
# Download UD English-EWT treebank on Jean Zay login node
#
# Usage:
#   bash scripts/jeanzay/download_ud.sh
#

set -euo pipefail

FSWORK="/lustre/fswork/projects/rech/jsl/ucc66vf"
PROJECT_DIR="$FSWORK/Projects/Hops_parser_torch"
UD_DIR="$FSWORK/depver_data/ud"
ENV_DIR="$FSWORK/conda_related/envs/depver"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate "$ENV_DIR"

cd "$PROJECT_DIR/hopsparser"

echo "Downloading UD treebanks to $UD_DIR ..."
mkdir -p "$UD_DIR"

python scripts/get_UD_data.py download "$UD_DIR"

echo ""
echo "Checking English-EWT..."
TRAIN="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-train.conllu"
DEV="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-dev.conllu"
TEST="$UD_DIR/ud-treebanks-v2.16/all_corpora/UD_English-EWT/en_ewt-ud-test.conllu"

for f in "$TRAIN" "$DEV" "$TEST"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "Done. UD data at: $UD_DIR"
echo ""
echo "Next: submit training job:"
echo "  sbatch scripts/jeanzay/train_en_model.slurm"
