"""A dial grid as a Python loop of ``make_env`` cells vs one ``EnvSpec.run_grid``.

Two things from one run, on the same grid:

1. Time. The loop is how every sweep driver works today (``gd_bundles``, the WP
   runners): a fresh spec per cell, so a compile per cell. ``run_grid`` passes
   the dial values as runtime data and compiles once for the whole grid.
2. The gap between the paths' numbers (invariant 3 in docs/run-grid-design.md).
   The loop cell is the reference, since its config holds the value. With
   ``batch=1`` (cells one at a time) a grid cell should match it bit for bit;
   with ``batch=G`` (all cells vectorized) XLA may sum in a different order.

``ledger_society`` at its defaults, swept over two dials it already reads from
``global_attrs`` (``churn_now``, ``repair_rate_now``). The dial values are
arbitrary; this measures the engine, not the model. Run outside pytest, so the
persistent compile cache stays off (an on-disk cache would hide the loop's
compiles on a second run).

    python -m experiments.perf_baseline.grid_calls [--seeds 8] [--steps 400] [--side 5]
"""
import argparse
import itertools
import platform
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.environments import make_env
from cilib.environments.ledger_society.dynamics import _human_shares
from cilib.metrics.reducers import late_mean

DIALS = {"churn": "churn_now", "repair_rate": "repair_rate_now"}
SHARES = ("human_income_share", "human_attention_share", "human_power_share")


def reducers():
    return {k: late_mean(lambda s, k=k: _human_shares(s)[k]) for k in SHARES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--side", type=int, default=5, help="values per dial; G = side**2")
    a = p.parse_args()
    churn = np.linspace(0.0, 0.2, a.side, dtype=np.float32)
    repair = np.linspace(0.0, 0.04, a.side, dtype=np.float32)
    cells = list(itertools.product(churn, repair))
    key, red = jr.PRNGKey(0), reducers()
    print(f"jax {jax.__version__}, {jax.devices()[0]}, {platform.processor() or platform.machine()}; "
          f"ledger_society, G={len(cells)} cells, S={a.seeds} seeds, T={a.steps}")

    t0 = time.perf_counter()
    loop = []
    for c, r in cells:
        env = make_env("ledger_society", churn=float(c), repair_rate=float(r))
        _, _, out = env.run_reduced_batch(key, a.seeds, a.steps, red)
        loop.append({k: np.asarray(v) for k, v in out.items()})
    t_loop = time.perf_counter() - t0

    grid = {"churn_now": jnp.array([c for c, _ in cells]),
            "repair_rate_now": jnp.array([r for _, r in cells])}
    print(f"python loop of make_env + run_reduced_batch: {t_loop:7.2f} s  "
          f"({len(cells)} programs)")
    for batch in (1, len(cells)):
        env = make_env("ledger_society")
        t0 = time.perf_counter()
        _, _, out = env.run_grid(key, a.seeds, a.steps, grid, red, batch=batch)
        jax.block_until_ready(out)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        jax.block_until_ready(env.run_grid(jr.PRNGKey(1), a.seeds, a.steps, grid, red,
                                           batch=batch)[2])
        t_warm = time.perf_counter() - t0
        diffs = np.stack([np.abs(np.asarray(out[k])[g] - loop[g][k])
                          for g in range(len(cells)) for k in SHARES])
        exact = np.mean([np.all(np.stack([np.asarray(out[k])[g] == loop[g][k]
                                          for k in SHARES])) for g in range(len(cells))])
        print(f"run_grid batch={batch:<3d} first call {t_first:6.2f} s, repeat {t_warm:6.2f} s "
              f"({len(env._programs)} program); vs loop: max |diff| {diffs.max():.1e}, "
              f"cells bit-identical {100 * exact:.0f}%")
    print("(diffs are late-window human shares, values in [0, 1], over all cells and seeds)")


if __name__ == "__main__":
    main()
