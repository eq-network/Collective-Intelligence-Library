"""perf_baseline — where the time goes, per environment, before any kernel is touched.

For every registered environment at its default config this measures, on the
machine it runs on:

- eager calls: ``env.run`` as the tests and studies call it today (a bare
  ``lax.scan``, no ``jax.jit``); first call and two repeats, to see whether the
  eager path re-traces and re-compiles every time;
- jitted calls: the same run under ``jax.jit`` with static T — first call is
  trace + compile + run, the warm best-of-k is the true run cost;
- compile time against T at two scan lengths, to confirm compile is a fixed
  cost independent of run length;
- seed batches: ``run_batch`` under jit for 1 / 8 / 32 seeds, so the parallel
  efficiency of ``vmap`` on this CPU is a number rather than a hope;
- trace bytes of the default trace, the memory ceiling CLAUDE.md names;
- with ``--scaling``: population multiplied 1x / 2x / 4x / 8x on each
  environment's own size field, warm cost and the fitted growth exponent.

Numbers are machine-specific. ``baseline.json`` carries provenance (git rev,
jax version, CPU, date) and the README quotes them only with it. Compile and
run are reported separately because they scale differently and only the
latter is what a long run pays (examples/07 makes the same point).

    python -m experiments.perf_baseline.run              # baseline, all envs
    python -m experiments.perf_baseline.run --scaling    # + population sweep
    python -m experiments.perf_baseline.run --envs ledger_society capital_economy
    python -m experiments.perf_baseline.run --smoke
    python -m experiments.perf_baseline.run --envs ledger_society --scaling --mults 1 8 32 64 --out experiments/perf_baseline/scaling-large.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import time
import traceback

import jax
import jax.random as jr

from cilib import runs
from cilib.environments import REGISTRY, make_env

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "baseline.json")   # scalars + provenance: small, committed

# Each environment's own population field. The other size fields (AI blocks,
# sectors, grid) stay at their defaults so one axis moves at a time.
SIZE_FIELD = {
    "commons_harvest": "n_agents",
    "governed_commons": "n_households",
    "compute_economy": "n_households",
    "io_economy": "n_households",
    "capital_economy": "n_households",
    "task_economy": "n_households",
    "value_contagion": "n_agents",
    "influence_exchange": "n_citizens",
    "delegative_polity": "n_citizens",
    "coupled_society": "n_humans",
    "ledger_society": "n_humans",
}


def _block(x):
    jax.block_until_ready(x)
    return x


def _timed(fn):
    t0 = time.perf_counter()
    out = _block(fn())
    return time.perf_counter() - t0, out


def _best_of(fn, k):
    return min(_timed(fn)[0] for _ in range(k))


def _leaf_bytes(tree) -> int:
    return int(sum(getattr(x, "nbytes", 0) for x in jax.tree_util.tree_leaves(tree)))


def _n_nodes(env, key) -> int:
    return int(env.init_fn(key).node_types.shape[0])


def provenance() -> dict:
    """The shared engine block plus the machine facts a timing depends on."""
    return {
        **runs.engine_info(),
        "date": _dt.date.today().isoformat(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "compilation_cache_dir": jax.config.jax_compilation_cache_dir,
    }


def profile_env(name: str, T: int, T_short: int, seeds: tuple, repeats: int) -> dict:
    key = jr.PRNGKey(0)
    r: dict = {"T": T}

    t_build, env = _timed(lambda: make_env(name))
    r["build_s"] = t_build
    r["n_nodes"] = _n_nodes(env, key)

    # 1. eager, as the tests call it: first call, then two repeats.
    t1, (_, trace) = _timed(lambda: env.run(key, T))
    r["eager_call1_s"] = t1
    r["eager_call2_s"] = _timed(lambda: env.run(key, T))[0]
    r["eager_call3_s"] = _timed(lambda: env.run(key, T))[0]
    r["trace_bytes"] = _leaf_bytes(trace)

    # 2. jitted: first call = trace + compile + run; warm = the run itself.
    f = jax.jit(lambda k: env.run(k, T))
    r["jit_call1_s"] = _timed(lambda: f(key))[0]
    r["jit_warm_s"] = _best_of(lambda: f(key), repeats)
    r["compile_est_s"] = r["jit_call1_s"] - r["jit_warm_s"]
    r["step_us"] = 1e6 * r["jit_warm_s"] / T
    r["step_node_us"] = r["step_us"] / r["n_nodes"]

    # 3. compile vs T: a second scan length; compile should not move with T.
    g = jax.jit(lambda k: env.run(k, T_short))
    c1 = _timed(lambda: g(key))[0]
    w1 = _best_of(lambda: g(key), repeats)
    r["T_short"] = T_short
    r["compile_est_short_s"] = c1 - w1
    r["jit_warm_short_s"] = w1

    # 4. seed batches under jit.
    r["seeds"] = {}
    for S in seeds:
        fb = jax.jit(lambda k, S=S: env.run_batch(k, S, T))
        c = _timed(lambda: fb(key))[0]
        w = _best_of(lambda: fb(key), repeats)
        r["seeds"][str(S)] = {"call1_s": c, "warm_s": w}
    w1 = r["seeds"][str(seeds[0])]["warm_s"]
    for S in seeds:
        e = r["seeds"][str(S)]
        e["efficiency"] = (w1 * S / seeds[0]) / e["warm_s"] if e["warm_s"] > 0 else None
    return r


def scaling_env(name: str, T: int, mults: tuple, repeats: int) -> dict:
    field = SIZE_FIELD[name]
    base = getattr(make_env(name).config, field)
    key = jr.PRNGKey(0)
    rows = []
    for m in mults:
        n = int(base * m)
        env = make_env(name, **{field: n})
        f = jax.jit(lambda k, env=env: env.run(k, T))
        c = _timed(lambda: f(key))[0]
        w = _best_of(lambda: f(key), repeats)
        rows.append({"mult": m, field: n, "n_nodes": _n_nodes(env, key),
                     "call1_s": c, "warm_s": w, "step_us": 1e6 * w / T})
    import math
    lo, hi = rows[0], rows[-1]
    p = (math.log(hi["warm_s"] / lo["warm_s"]) / math.log(hi["n_nodes"] / lo["n_nodes"])
         if hi["n_nodes"] > lo["n_nodes"] and lo["warm_s"] > 0 else None)
    return {"field": field, "base": base, "rows": rows, "exponent": p}


def _fmt(x, w=9, d=2):
    return f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) else f"{str(x):>{w}}"


def main(envs, T, T_short, seeds, mults, repeats, scaling, smoke, out=OUT):
    results = {"provenance": provenance(), "params": {
        "T": T, "T_short": T_short, "seeds": list(seeds), "mults": list(mults),
        "repeats": repeats, "smoke": smoke}, "envs": {}, "scaling": {}}
    print(f"perf_baseline  jax {jax.__version__}  {results['provenance']['backend']}  "
          f"cpu={os.cpu_count()}  T={T}\n")
    hdr = (f"{'env':<20}{'N':>5}{'eager1':>9}{'eager2':>9}{'jit1':>9}{'warm':>9}"
           f"{'compile':>9}{'us/step':>9}{'trace MB':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name in envs:
        try:
            r = profile_env(name, T, T_short, seeds, repeats)
            results["envs"][name] = r
            print(f"{name:<20}{r['n_nodes']:>5}{_fmt(r['eager_call1_s'])}{_fmt(r['eager_call2_s'])}"
                  f"{_fmt(r['jit_call1_s'])}{_fmt(r['jit_warm_s'], d=3)}{_fmt(r['compile_est_s'])}"
                  f"{_fmt(r['step_us'], d=0)}{_fmt(r['trace_bytes'] / 1e6, w=10)}", flush=True)
        except Exception as e:  # keep going; one broken env must not hide the rest
            results["envs"][name] = {"error": f"{type(e).__name__}: {e}",
                                     "traceback": traceback.format_exc()}
            print(f"{name:<20} ERROR {type(e).__name__}: {e}", flush=True)
        _dump(results, out)

    print("\nseed batches (jit, warm best-of): seconds and parallel efficiency")
    print(f"{'env':<20}" + "".join(f"{'S=' + str(S):>14}" for S in seeds))
    for name, r in results["envs"].items():
        if "seeds" not in r:
            continue
        cells = "".join(f"{r['seeds'][str(S)]['warm_s']:>8.3f}s "
                        f"{100 * (r['seeds'][str(S)]['efficiency'] or 0):>4.0f}%"
                        for S in seeds)
        print(f"{name:<20}{cells}")

    if scaling:
        print(f"\npopulation scaling (jit warm, T={T}): cost ~ N^p")
        for name in envs:
            if name not in SIZE_FIELD or "error" in results["envs"].get(name, {}):
                continue
            try:
                s = scaling_env(name, T, mults, repeats)
                results["scaling"][name] = s
                cells = "  ".join(f"N={row['n_nodes']}:{row['warm_s']:.3f}s" for row in s["rows"])
                pexp = f"{s['exponent']:.2f}" if s["exponent"] is not None else "n/a"
                print(f"{name:<20} p={pexp:<6} {cells}", flush=True)
            except Exception as e:
                results["scaling"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"{name:<20} ERROR {type(e).__name__}: {e}", flush=True)
            _dump(results, out)

    _dump(results, out)
    print(f"\nwrote {out}")
    return results


def _dump(results, out=OUT):
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--T-short", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=OUT, help="where to write the json")
    a = ap.parse_args()
    if a.smoke:
        main(envs=a.envs[:2], T=5, T_short=3, seeds=(1, 2), mults=(1, 2),
             repeats=1, scaling=True, smoke=True, out=a.out)
    else:
        main(envs=a.envs, T=a.T, T_short=a.T_short, seeds=tuple(a.seeds),
             mults=tuple(a.mults), repeats=a.repeats, scaling=a.scaling, smoke=False,
             out=a.out)
