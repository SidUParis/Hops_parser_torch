# DepVer: Guide to Run

## Quick Overview

```
Local:      tests, development
Jean Zay:   dataset download, model loading, GPU evaluation
```

---

## 1. Local Development (No GPU needed)

### Install

```bash
cd hops_parser

# Install hopsparser (dependency)
pip install -e "./hopsparser"

# Install depver with test extras
pip install -e "./depver[tests]"
```

### Run Tests

```bash
cd depver
python -m pytest tests/ -v
```

Tests use hand-built `DepGraph` objects — no parser model or GPU needed.

### Quick Sanity Check (with parser model)

If you have a hopsparser model locally:

```bash
# Extract triples from a text file
echo "Macron signed the climate bill in March." > /tmp/test.txt
depver extract --model /path/to/model --input /tmp/test.txt

# Verify generated text against source
echo "The company reported revenue growth." > /tmp/source.txt
echo "The company denied revenue growth." > /tmp/generated.txt
depver verify --model /path/to/model --source /tmp/source.txt --generated /tmp/generated.txt
```

---

## 2. Jean Zay Setup

### 2.1 Upload Code

```bash
# From your local machine
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    hops_parser/ jean-zay:$WORK/hops_parser/
```

Or if you use git:

```bash
# On Jean Zay
cd $WORK
git clone <your-repo-url> hops_parser
```

### 2.2 Run Setup Script

```bash
# On Jean Zay
cd $WORK/hops_parser
bash scripts/jeanzay/setup.sh
```

This will:
- Create a conda environment at `$WORK/envs/depver`
- Install hopsparser + depver + all dependencies
- Download NLTK data (WordNet)
- Download FRANK + AggreFact + SummEval datasets to `$WORK/depver_data/`

### 2.3 Download a Parser Model

You need an English hopsparser model. Options:

**Option A: Download pretrained from HuggingFace** (recommended)

```bash
module load pytorch-gpu/py3/2.4.0
conda activate $WORK/envs/depver

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'hopsparser/hopsparser-en_ewt-camembert',
    local_dir='$WORK/depver_models/en_ewt'
)
"
```

Check available models:
```bash
python -c "
from huggingface_hub import list_models
for m in list_models(author='hopsparser'):
    print(m.modelId)
"
```

**Option B: Train your own** (if you need a specific treebank)

See hopsparser documentation. Training needs a UD treebank in CoNLL-U format.

### 2.4 Edit SLURM Scripts

Before submitting jobs, edit the account in all SLURM scripts:

```bash
# Replace YOUR_PROJECT_ID with your actual IDRIS project
sed -i 's/YOUR_PROJECT_ID/abc@v100/g' scripts/jeanzay/run_*.slurm
```

Replace `abc@v100` with your project allocation (e.g., `xyz@a100` for A100).

---

## 3. Running Experiments on Jean Zay

### 3.1 Smoke Test First (30 min, 50 examples)

```bash
sbatch scripts/jeanzay/run_quick_test.slurm
```

Check output:
```bash
# Check job status
squeue -u $USER

# View output when done
cat depver-quick_*.out
```

### 3.2 Full Evaluation (4 hours)

```bash
sbatch scripts/jeanzay/run_eval.slurm
```

This runs:
1. DepVer on FRANK
2. DepVer on AggreFact
3. BERTScore baseline on both
4. Prints comparison table

Results go to `$WORK/depver_results/<job_id>/`.

### 3.3 Unit Tests (CPU only, 30 min)

```bash
sbatch scripts/jeanzay/run_tests.slurm
```

### 3.4 Check Results

```bash
# Find latest results
ls -lt $WORK/depver_results/ | head

# View metrics
cat $WORK/depver_results/<job_id>/frank_depver/metrics.json
cat $WORK/depver_results/<job_id>/comparison.json

# Copy results to local machine
rsync -avz jean-zay:$WORK/depver_results/ results/
```

---

## 4. File Layout on Jean Zay

```
$WORK/
├── hops_parser/                # Code (this repo)
│   ├── hopsparser/             # Parser source
│   ├── depver/                 # DepVer source
│   ├── experiments/            # Evaluation scripts
│   └── scripts/jeanzay/        # SLURM scripts
│
├── envs/depver/                # Conda environment
│
├── depver_data/                # Downloaded datasets
│   ├── frank/
│   ├── aggrefact/
│   └── summeval/
│
├── depver_models/              # Parser models
│   └── en_ewt/                 # English UD-EWT model
│
└── depver_results/             # Experiment outputs
    ├── <job_id>/
    │   ├── frank_depver/
    │   ├── frank_bertscore/
    │   ├── aggrefact_depver/
    │   ├── aggrefact_bertscore/
    │   └── comparison.json
    └── quick_test_<job_id>/
```

---

## 5. Common Issues

### "No module named hopsparser"
Make sure both packages are installed:
```bash
pip install -e "./hopsparser" && pip install -e "./depver"
```

### "CUDA out of memory"
Reduce batch size in the parser. Edit the model's `config.json` and lower `default_batch_size`, or pass `--batch-size 8` if supported.

### "Requested model not loaded" on server
You're using the CLI, not the server. Use `depver verify --model <path>`.

### Datasets not downloading
Jean Zay compute nodes don't have internet. Download on a login node:
```bash
# Run on login node (not in a SLURM job)
python experiments/prepare_data.py --output-dir $WORK/depver_data
```

### Module version conflicts
Check available modules:
```bash
module avail pytorch-gpu
```
Pick the latest available and update `setup.sh` and SLURM scripts.

### Job stays PENDING
Check your allocation:
```bash
idrcontract  # shows your available hours
squeue -u $USER --start  # shows estimated start time
```
Use `qos_gpu-dev` for quick tests (higher priority, max 2h).

---

## 6. Running Custom Verification

### Single pair (interactive)

```bash
# On a GPU node (srun for interactive)
srun --gres=gpu:1 --partition=gpu_p13 --account=YOUR_PROJECT --qos=qos_gpu-dev --time=01:00:00 --pty bash

module load pytorch-gpu/py3/2.4.0
conda activate $WORK/envs/depver
cd $WORK/hops_parser

depver verify \
    --model $WORK/depver_models/en_ewt \
    --source /path/to/source.txt \
    --generated /path/to/generated.txt \
    --device cuda \
    --format text
```

### Batch verification (JSONL)

Prepare a JSONL file where each line has `{"source": "...", "generated": "..."}`:

```bash
depver verify-batch \
    --model $WORK/depver_models/en_ewt \
    --input my_pairs.jsonl \
    --output my_results.jsonl \
    --device cuda
```

### From Python

```python
from depver.pipeline import DepVerifier
from depver.scoring.report import format_report

verifier = DepVerifier.from_pretrained("$WORK/depver_models/en_ewt", device="cuda")

result = verifier.verify(
    source_text="The company reported a 5% revenue increase.",
    generated_text="The company denied a 5% revenue increase.",
)

print(format_report(result))
# Shows: VERB_SUBSTITUTION (reported -> denied), severity=HIGH
```

### Without a parser (pre-parsed CoNLL-U)

```python
from depver.pipeline import DepVerifierWithoutParser

verifier = DepVerifierWithoutParser()
result = verifier.verify_from_conllu(source_conllu, generated_conllu)
```

---

## 7. Next Steps After Initial Results

1. **Analyze per-type accuracy** — which divergence types does DepVer catch best?
   ```bash
   cat results/frank_depver/divergence_types.json
   ```

2. **Tune threshold** — try `--threshold 0.3` and `--threshold 0.5` to see impact on precision/recall

3. **Add embeddings backend** — install `sentence-transformers` and re-run:
   ```bash
   pip install sentence-transformers
   # Re-run the same evaluation — similarity.py auto-detects the backend
   ```

4. **Cross-lingual (French)** — download a French hopsparser model and prepare French evaluation data

5. **Ensemble with BERTScore** — combine scores for a hybrid approach
