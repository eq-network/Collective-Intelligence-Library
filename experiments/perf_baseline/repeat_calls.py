"""First vs repeat calls of ``EnvSpec.run`` / ``run_batch`` against the eager scan.

The README's "Done since" numbers. ``old`` is the pre-2026-09-30 code path
(``core.scan.run_scan`` / ``run_scan_batch``, which recompile every call), best
of two; ``new`` is the spec's cached program, first call and best repeat with a
fresh key. Run outside pytest so the persistent compile cache stays off.

    python -m experiments.perf_baseline.repeat_calls
"""
import time

import jax
import jax.random as jr

from cilib.core.scan import run_scan, run_scan_batch
from cilib.environments import list_envs, make_env

T, S = 200, 8


def _time(f):
    t0 = time.perf_counter()
    jax.block_until_ready(f())
    return time.perf_counter() - t0


def main():
    print(f"jax {jax.__version__}, {jax.devices()[0]}, T={T}, batch S={S}")
    print(f"{'env':20s} {'old run':>8s} {'new 1st':>8s} {'new rep':>8s}"
          f" {'old batch':>10s} {'new 1st':>8s} {'new rep':>8s}")
    for name in list_envs():
        env = make_env(name)

        def eager(k):
            k_init, k_run = jr.split(k)
            return run_scan(env.round_fn, env.init_fn(k_init), T, k_run, trace_fn=env.trace_fn)

        def eager_batch(k):
            return run_scan_batch(env.round_fn, env.init_fn, T, jr.split(k, S),
                                  trace_fn=env.trace_fn)

        old = min(_time(lambda: eager(jr.PRNGKey(i))) for i in range(2))
        first = _time(lambda: env.run(jr.PRNGKey(0), T))
        rep = min(_time(lambda: env.run(jr.PRNGKey(i), T)) for i in (1, 2, 3))
        old_b = min(_time(lambda: eager_batch(jr.PRNGKey(i))) for i in range(2))
        first_b = _time(lambda: env.run_batch(jr.PRNGKey(0), S, T))
        rep_b = min(_time(lambda: env.run_batch(jr.PRNGKey(i), S, T)) for i in (1, 2, 3))
        print(f"{name:20s} {old:7.3f}s {first:7.3f}s {rep * 1e3:6.1f}ms"
              f" {old_b:9.3f}s {first_b:7.3f}s {rep_b * 1e3:6.1f}ms", flush=True)


if __name__ == "__main__":
    main()
