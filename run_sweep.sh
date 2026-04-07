#!/bin/bash
# Run basin stability experiment sweep
# Usage: bash run_sweep.sh [--vmap] [--seeds 500] [--quick] [--plot] [--output-dir DIR]
#
# Examples:
#   bash run_sweep.sh --quick                  # 10 seeds, sequential, fast check
#   bash run_sweep.sh --vmap --seeds 100       # 100 seeds, GPU-batched
#   bash run_sweep.sh --vmap --seeds 500 --plot  # full sweep with plots
set -e

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Devices: $(python -c 'import jax; print(jax.devices())' 2>/dev/null)"
echo ""

python -m experiments.basin_stability.run_experiment "$@"
