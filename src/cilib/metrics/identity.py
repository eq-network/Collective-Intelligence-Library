"""Metric identity (observability O0): a stable id, a kind, and a definition hash.

A metric in this repo is a plain function, either ``GraphState -> scalar`` (the
family catalogs) or ``trace -> value`` (an environment's ``make_metrics``). Both
carry their identity the same way: the ``@kind`` decorator declares what sort of
number the function produces, and ``metric_id`` reads that back together with a
hash of the function's definition.

Kinds are a closed vocabulary, one renderer per kind downstream:

- ``share``          a fraction of a whole, in [0, 1]
- ``concentration``  an inequality or concentration index (gini, hhi, top-k, centralization)
- ``rate``           a per-step quantity or a fraction of events per step
- ``level``          a raw quantity in model units (a stock, an output, a wage, a time)
- ``index``          a bounded dimensionless score that is not a share of a whole
                     (fidelity, alignment, entropy, correlation, quality)

Widening the vocabulary is a one-line change to ``KINDS``.

The definition hash covers the function's own source and, transitively, every
function and module constant it reaches by name in this codebase: ``harvest_gini``
is ``gini_of(...)``, so its hash moves when ``gini_of`` changes. Values captured by
a closure (the config, counts derived from it) are not part of the definition;
they belong to the record's address. The walk is by name, so a function reached
only through a container or an argument is not seen.

The version is authoritative and the hash is advisory: a hash alone false-alarms
on a comment edit, a version alone gets forgotten. The hash's job is to warn that
a bump was missed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import sys
import textwrap
import types
from typing import Callable, Dict, List

KINDS = frozenset({"share", "concentration", "rate", "level", "index"})

_CONSTANT_TYPES = (int, float, str, bytes, tuple, frozenset)


def kind(name: str, version: int = 1) -> Callable[[Callable], Callable]:
    """Declare a metric's kind (and version) at its definition. Returns the same function."""
    if name not in KINDS:
        raise ValueError(f"unknown metric kind {name!r}; one of {sorted(KINDS)}")

    def declare(fn: Callable) -> Callable:
        fn.kind = name
        fn.version = version
        return fn
    return declare


def _source(fn: Callable) -> str:
    """Dedented source from the ``def`` line on: decorators (where the version
    lives) are not part of the definition, so a version bump leaves the hash alone."""
    lines = textwrap.dedent(inspect.getsource(fn)).splitlines(keepends=True)
    body = [ln for ln in lines if not ln.lstrip().startswith("@")]
    return "".join(body)


def _in_scope(obj, root_pkg: str) -> bool:
    mod = getattr(obj, "__module__", None) or ""
    return mod.split(".")[0] in ("cilib", root_pkg)


def _walk(fn: Callable, root_pkg: str, seen: Dict[tuple, str]) -> None:
    """Collect ``fn`` and everything it reaches by global or closure name."""
    key = (fn.__module__, fn.__qualname__)
    if key in seen:
        return
    seen[key] = _source(fn)
    module_globals = getattr(sys.modules.get(fn.__module__), "__dict__", {})
    reached = []
    codes = [fn.__code__]
    while codes:
        code = codes.pop()
        codes.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        reached.extend((n, module_globals.get(n)) for n in code.co_names)
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            reached.append((name, cell.cell_contents))
        except ValueError:            # empty cell
            pass
    reached.extend(("<default>", d) for d in (fn.__defaults__ or ()))
    for name, obj in reached:
        if isinstance(obj, types.FunctionType) and _in_scope(obj, root_pkg):
            _walk(obj, root_pkg, seen)
        elif isinstance(obj, _CONSTANT_TYPES) and not isinstance(obj, bool) \
                and (name == "<default>" or module_globals.get(name) is obj):
            seen[(fn.__module__, fn.__qualname__, name, repr(obj))] = repr(obj)


def definition_hash(fn: Callable) -> str:
    """sha256 over the dedented source of ``fn`` and of every function or module
    constant it reaches by name (recursively, within this codebase)."""
    seen: Dict[tuple, str] = {}
    _walk(fn, (fn.__module__ or "").split(".")[0], seen)
    # sources only: where a definition lives (module, nesting) is not what it is
    h = hashlib.sha256()
    for text in sorted(set(seen.values())):
        h.update(text.encode("utf-8"))
        h.update(b"\n--\n")
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class MetricId:
    id: str
    kind: str
    version: int
    definition_hash: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def metric_id(name: str, fn: Callable) -> MetricId:
    """Identity of one metric; the function must carry a ``@kind`` declaration."""
    if not hasattr(fn, "kind"):
        raise ValueError(f"metric {name!r} has no @kind declaration")
    return MetricId(name, fn.kind, fn.version, definition_hash(fn))


def metric_ids(metrics: Dict[str, Callable]) -> List[dict]:
    """Manifest entries for a metric dict, in the dict's order."""
    return [metric_id(name, fn).as_dict() for name, fn in metrics.items()]
