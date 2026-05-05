#!/bin/bash
#SBATCH --job-name=eval_judge_baseline
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --account=plgroomagine-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=outputs/logs/eval_judge_baseline_%j.out
#SBATCH --error=outputs/logs/eval_judge_baseline_%j.err

# NOTE: Gemini is a remote API — no GPU is actually used.
# We request 1 GPU only because plgrid-gpu-a100 is the only allocation
# available on this account.

cd /net/pr2/projects/plgrid/plggrecontext/joanna/vigil/vigil

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p outputs/logs evaluations

module purge
module load GCCcore/13.2.0 Python/3.11.5

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "ERROR: venv not found at $(pwd)/venv. Aborting."
  exit 1
fi

export PIP_CACHE_DIR=$PWD/.pip_cache
export TMPDIR=$PWD/.tmp
export XDG_CACHE_HOME=$PWD/.xdg_cache
export XDG_DATA_HOME=$PWD/.xdg_data

if [ -f ".env" ]; then
  set -a; source .env; set +a
fi
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) not set."
  echo "       Add it to $(pwd)/.env (GEMINI_API_KEY=...)."
  exit 1
fi

# Judge concurrency / rate-limit / retry tuning.
#   JUDGE_MODEL          : which Gemini model to use as judge
#   JUDGE_MAX_WORKERS    : parallel API calls in flight (set to 1 to disable parallelism)
#   JUDGE_MIN_INTERVAL_S : min seconds between request *starts* across all workers (0 = no throttle, rely on backoff)
#                          Set to ~6.5 if you hit 10 RPM free-tier limits.
#   JUDGE_MAX_RETRIES    : retry attempts on 429 / 503 / timeouts (exponential backoff capped at 30s)
export JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
export JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
export JUDGE_MIN_INTERVAL_S="${JUDGE_MIN_INTERVAL_S:-0.0}"
export JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-5}"

# --baseline: 'all' (default) iterates qwen8b/gemini/gemma12b
# --batch_size: how many (datapoint, hallucination_type) cases per Gemini request
BASELINE="${BASELINE:-all}"
BATCH_SIZE="${BATCH_SIZE:-15}"

echo ""
echo "Starting LLM-as-a-Judge evaluation..."
echo "  Judge model: $JUDGE_MODEL"
echo "  Baseline:    $BASELINE"
echo "  Batch size:  $BATCH_SIZE"
echo "  Workers:     $JUDGE_MAX_WORKERS"
echo ""

python3 create_evaluations/eval_llm_as_a_judge_baseline.py --baseline "$BASELINE" --batch_size "$BATCH_SIZE"

echo ""
echo "Finished judge evaluation. Report in: evaluations/"
