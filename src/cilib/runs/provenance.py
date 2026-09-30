"""What produced a record: package version, git state of the code that ran, JAX setup."""
from __future__ import annotations

import os
import platform
import subprocess

import jax

import cilib

PACKAGE = "collective-intelligence-library"
SOURCE = os.path.dirname(os.path.abspath(cilib.__file__))


def _git(*args: str) -> str:
    # Resolved against the cilib package that actually ran, not the caller's
    # checkout: with a worktree and an editable install the two can differ.
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             cwd=SOURCE, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def engine_info() -> dict:
    """One block, written by every manifest.

    ``source`` says which checkout was imported, so a record made with the wrong
    ``PYTHONPATH`` is visible. ``git_rev`` is ``"unknown"`` outside a checkout;
    ``git_dirty`` covers the whole repository that checkout belongs to, not only
    the package, since the harness that defines a run's address lives outside it.
    """
    rev = _git("rev-parse", "--short", "HEAD")
    top = _git("rev-parse", "--show-toplevel") if rev else ""
    dirty = bool(_git("-C", top, "status", "--porcelain")) if top else False
    return {
        "package": PACKAGE,
        "version": cilib.__version__,
        "source": SOURCE,
        "git_rev": rev or "unknown",
        "git_dirty": dirty,
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }
