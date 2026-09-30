"""Test-suite setup: JAX's persistent compilation cache.

The suite is compile-bound (experiments/perf_baseline: about one to three
compiles per test, 0.3 to 0.9 s each), and most tests build a fresh environment,
so no in-process cache can help them. The persistent cache is keyed on the
compiled program, not on Python object identity, so a rerun of the suite reuses
the previous run's compiles. Set through the environment, not ``jax.config``,
so the example smoke tests (subprocesses) inherit it; it must run before jax
is imported, which a root conftest does. The minimum compile time is zeroed
because JAX's default (1 s) sits above nearly every compile in this repo.

To opt out or relocate, set ``JAX_COMPILATION_CACHE_DIR`` yourself; delete
``.jax_cache/`` to clear it. It grows without bound.
"""
import os
import pathlib

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      str(pathlib.Path(__file__).parent / ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
