"""
CSV export for metric arrays extracted from GraphState.

Two granularities:
- trajectory CSV: one row per (condition × seed × timestep)
- summary CSV: one row per (condition), aggregated across seeds

All export happens post-run in Python-land, not inside JIT.
"""
import csv
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


def _wilson_ci(successes: int, total: int, z: float = 1.96):
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return 0.0, 1.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def extract_metric_arrays(final_state, metric_names: List[str]) -> Dict[str, np.ndarray]:
    """Extract metric arrays from a final GraphState into numpy arrays.

    Args:
        final_state: GraphState after run_scan (contains filled metric arrays)
        metric_names: list of metric names (without 'metric_' prefix)

    Returns:
        dict mapping metric name to numpy array of shape (T,)
    """
    arrays = {}
    for name in metric_names:
        key = f"metric_{name}"
        arr = final_state.global_attrs[key]
        arrays[name] = np.array(arr)
    return arrays


def write_trajectory_csv(
    path: Path,
    final_state,
    metric_names: List[str],
    run_meta: Dict[str, Any],
    append: bool = True,
) -> None:
    """Write one row per timestep from the metric arrays in global_attrs.

    Args:
        path: output CSV path
        final_state: GraphState after run_scan
        metric_names: metric names to export
        run_meta: dict with keys like mechanism, adversarial_fraction, seed
        append: if True, append to existing file (write header only if new)
    """
    path = Path(path)
    arrays = extract_metric_arrays(final_state, metric_names)
    T = len(next(iter(arrays.values())))

    meta_keys = sorted(run_meta.keys())
    header = meta_keys + ["step"] + sorted(metric_names)

    write_header = not path.exists() or not append
    mode = "a" if append else "w"

    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for t in range(T):
            row = [run_meta[k] for k in meta_keys]
            row.append(t)
            for name in sorted(metric_names):
                row.append(float(arrays[name][t]))
            writer.writerow(row)


def write_summary_csv(
    path: Path,
    condition_meta: Dict[str, Any],
    final_states: list,
    metric_names: List[str],
    extra_cols: Optional[Dict[str, float]] = None,
    append: bool = True,
) -> None:
    """Write one row per condition, aggregated across seeds.

    Computes mean and std of final metric values across seeds.

    Args:
        path: output CSV path
        condition_meta: dict with keys like mechanism, adversarial_fraction
        final_states: list of final GraphStates (one per seed)
        metric_names: metric names to aggregate
        extra_cols: additional columns (e.g., basin_stability, ci bounds)
        append: if True, append to existing file
    """
    path = Path(path)
    extra_cols = extra_cols or {}

    # Extract final values for each metric across seeds
    final_values = {name: [] for name in metric_names}
    for state in final_states:
        arrays = extract_metric_arrays(state, metric_names)
        for name in metric_names:
            final_values[name].append(float(arrays[name][-1]))

    meta_keys = sorted(condition_meta.keys())
    metric_cols = []
    for name in sorted(metric_names):
        metric_cols.extend([f"mean_{name}", f"std_{name}"])
    extra_keys = sorted(extra_cols.keys())

    header = meta_keys + ["n_seeds"] + extra_keys + metric_cols

    write_header = not path.exists() or not append
    mode = "a" if append else "w"

    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        row = [condition_meta[k] for k in meta_keys]
        row.append(len(final_states))
        for k in extra_keys:
            row.append(extra_cols[k])
        for name in sorted(metric_names):
            vals = np.array(final_values[name])
            row.append(float(np.mean(vals)))
            row.append(float(np.std(vals)))
        writer.writerow(row)
