#!/bin/bash
#SBATCH --job-name=mycorrhiza
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Template Slurm job for running basin stability experiments on HPC.
# Adapt the module names and paths to your cluster.
#
# Usage:
#   mkdir -p logs
#   sbatch slurm_job.sh
#   sbatch --gres=gpu:a100:1 slurm_job.sh    # request specific GPU

# --- Adapt these to your cluster ---
# module load cuda/12.4
# module load python/3.11
# -----------------------------------

cd $SLURM_SUBMIT_DIR

# Setup (only runs pip install if .venv doesn't exist yet)
if [ ! -d ".venv" ]; then
    bash setup.sh --cuda
else
    source .venv/bin/activate
fi

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Run experiment
OUTPUT_DIR="results/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"

python -m experiments.basin_stability.run_experiment \
    --vmap \
    --seeds 500 \
    --plot \
    --output-dir "$OUTPUT_DIR"

echo "Results saved to $OUTPUT_DIR"
