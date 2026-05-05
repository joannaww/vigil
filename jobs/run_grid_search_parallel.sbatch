#!/bin/bash
#SBATCH --job-name=vigil_grid_search
#SBATCH --array=1-48
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH --account=plgroomagine-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=outputs/logs/grid_search_%A_%a.out
#SBATCH --error=outputs/logs/grid_search_%A_%a.err

cd /net/pr2/projects/plgrid/plggrecontext/joanna/vigil/vigil

echo "Working directory: $(pwd)"
echo "Array task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX (Job: $SLURM_ARRAY_JOB_ID)"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p outputs/logs

module purge
module load GCCcore/13.2.0 Python/3.11.5
module load CUDA/12.1.1

export VLLM_USE_PRECOMPILED=1
export TORCH_CUDA_ARCH_LIST="8.0"

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "ERROR: venv not found at $(pwd)/venv. Aborting."
  exit 1
fi

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_CACHE_DIR=$PWD/.pip_cache
export TMPDIR=$PWD/.tmp
export XDG_CACHE_HOME=$PWD/.xdg_cache
export XDG_DATA_HOME=$PWD/.xdg_data
export HF_HOME=$PWD/.hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"

GRID_DIR="configs/grid_search"

if [ ! -d "$GRID_DIR" ] || [ -z "$(ls -A $GRID_DIR 2>/dev/null)" ]; then
  echo "ERROR: No config files found in $GRID_DIR"
  echo "Run: bash jobs/create_grid_configs.sh"
  exit 1
fi

# Iterate the same grid order as create_grid_configs.sh and pick the cell
# matching SLURM_ARRAY_TASK_ID.
pairing_thresholds=(0.2 0.3 0.4 0.5)
mask_margins=(0.0 0.1 0.2)
abs_diff_settings=(0 1)
physical_padding_ratios=(0.5 1.0)

counter=0
for threshold in "${pairing_thresholds[@]}"; do
  for margin in "${mask_margins[@]}"; do
    for abs_diff in "${abs_diff_settings[@]}"; do
      for padding in "${physical_padding_ratios[@]}"; do
        counter=$((counter + 1))

        if [ $counter -eq $SLURM_ARRAY_TASK_ID ]; then
          abs_diff_label=$([ "$abs_diff" -eq 1 ] && echo "boxes" || echo "noboxes")
          config_name="grid_${counter}_th${threshold}_mg${margin}_${abs_diff_label}_pad${padding}.yaml"
          config_file="$GRID_DIR/$config_name"

          if [ ! -f "$config_file" ]; then
            echo "ERROR: Config file not found: $config_file"
            exit 1
          fi

          echo "Running cell $counter: $config_name"
          echo "  threshold=$threshold, margin=$margin, boxes=$abs_diff_label, padding_ratio=$padding"
          echo

          python3 main.py --config "$config_file"
          exit $?
        fi
      done
    done
  done
done

echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID did not match any cell (max counter=$counter)"
exit 1
