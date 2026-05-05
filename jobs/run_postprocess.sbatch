#!/bin/bash
#SBATCH --job-name=vigil_postprocess
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=plgroomagine-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/postprocess_%j.out
#SBATCH --error=outputs/logs/postprocess_%j.err

# CPU/network-bound: parses final_results.json from outputs/grid_search/*,
# submits all Gemini Batch API jobs in parallel, polls until done, writes
# per-category baseline-format JSONs to data/<cat>/pipeline_<run>.json.
#
# Pass --use-gemini-judge to enable the Gemini-as-judge filter on top of the
# heuristic. Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in .env or env.
#
# Usage:
#   sbatch jobs/run_postprocess.sbatch                    # heuristic only
#   sbatch jobs/run_postprocess.sbatch --use-gemini-judge # heuristic + judge

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

python3 evaluation/postprocess_pipeline_to_baseline.py "$@"
