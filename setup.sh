#!/bin/bash
# Mycorrhiza — portable setup for any Linux machine
# Usage: bash setup.sh [--cuda]
#
# Works on: HPC nodes, cloud VMs, WSL2, any Linux with Python 3.10+
set -e

CUDA=false
for arg in "$@"; do
    case $arg in
        --cuda) CUDA=true ;;
    esac
done

echo "=== Mycorrhiza Setup ==="

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Python: $(python --version)"

# Install project
if [ "$CUDA" = true ]; then
    echo "Installing with CUDA support..."
    pip install -e ".[cuda]" --quiet
else
    echo "Installing (CPU only)..."
    pip install -e . --quiet
fi

# Verify JAX
echo ""
echo "=== JAX Device Check ==="
python -c "
import jax
devices = jax.devices()
print(f'Devices: {devices}')
backend = devices[0].platform
if backend == 'gpu':
    print('GPU detected — vmap sweeps will use GPU acceleration')
else:
    print('CPU only — use --cuda flag for GPU support')
    print('  Or on HPC: module load cuda/12 && bash setup.sh --cuda')
"

# Quick smoke test
echo ""
echo "=== Smoke Test (10 steps, 2 seeds) ==="
python -c "
import sys; sys.path.insert(0, '.')
from experiments.basin_stability.environment import run_batched
from metrics import ECONOMIC_METRICS, GOVERNANCE_METRICS
import jax.random as jr
import numpy as np

metrics = {**ECONOMIC_METRICS, **GOVERNANCE_METRICS}
batch = run_batched('pdd', 20, 4, jr.PRNGKey(0), 2, T=10, metrics=metrics)
r = np.array(batch.global_attrs['metric_resource_level'])
print(f'vmap shape: {r.shape} (seeds, steps)')
print(f'Final resources: {r[:, -1]}')
print('Smoke test passed.')
"

echo ""
echo "=== Ready ==="
echo "Run experiments:"
echo "  source .venv/bin/activate"
echo "  python -m experiments.basin_stability.run_experiment --quick --vmap"
echo "  bash run_sweep.sh --vmap --seeds 500"
