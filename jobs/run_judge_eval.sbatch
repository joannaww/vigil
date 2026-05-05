#!/bin/bash
#SBATCH --job-name=vigil_judge_eval
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=plgroomagine-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/judge_eval_%j.out
#SBATCH --error=outputs/logs/judge_eval_%j.err

# CPU/network-bound: judges the specified grid run(s) via Gemini Batch API.
# Caches per-run verdicts to evaluations/judge_cache/<run>.jsonl and writes a
# single-row summary at evaluations/judge_summaries/PIPELINE_JUDGE_<run>.csv
# for each requested run.
#
# Re-running a cached run is a no-op (cache hit). Use --force to recompute.
#
# Usage:
#   sbatch jobs/run_judge_eval.sbatch --run run_12_th0.2_mg0.2_boxes_pad1.0 --yes
#   sbatch jobs/run_judge_eval.sbatch --run run_12_...,run_8_... --yes

cd /net/pr2/projects/plgrid/plggrecontext/joanna/vigil/vigil

echo "Working directory: $(pwd)"
echo "Job: $SLURM_JOB_ID"
echo "Args: $@"

mkdir -p outputs/logs

module purge
module load GCCcore/13.2.0 Python/3.11.5

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "ERROR: venv not found at $(pwd)/venv. Aborting."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python3 -u create_evaluations/eval_llm_as_a_judge_pipeline.py "$@"
