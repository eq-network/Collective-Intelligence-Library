"""Metric identity: every catalog metric has one, and the hash tracks the definition."""
import importlib.util
import sys
import textwrap

import pytest

from cilib.environments import REGISTRY, make_env
from cilib.metrics import (
    ECONOMIC_METRICS, GOVERNANCE_METRICS, GRAPH_METRICS, CONCENTRATION_METRICS,
    SPECTRAL_METRICS,
)
from cilib.metrics.identity import KINDS, definition_hash, kind, metric_id, metric_ids

FAMILIES = (ECONOMIC_METRICS, GOVERNANCE_METRICS, GRAPH_METRICS,
            CONCENTRATION_METRICS, SPECTRAL_METRICS)


@pytest.mark.parametrize("env_name", sorted(REGISTRY))
def test_every_environment_metric_has_an_identity(env_name):
    ids = metric_ids(make_env(env_name).metrics)
    assert ids, env_name
    for entry in ids:
        assert entry["kind"] in KINDS
        assert entry["version"] >= 1
        assert len(entry["definition_hash"]) == 64


def test_every_family_metric_has_an_identity():
    for fam in FAMILIES:
        for entry in metric_ids(fam):
            assert entry["kind"] in KINDS


def test_undeclared_metric_is_rejected():
    def nameless(trace):
        return trace
    with pytest.raises(ValueError, match="no @kind"):
        metric_id("nameless", nameless)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown metric kind"):
        kind("vibe")


def test_hash_ignores_indentation_and_version_but_not_the_body():
    def outer():
        @kind("level")
        def m(trace):
            return trace["x"][-1]
        return m

    @kind("level", version=7)
    def m(trace):
        return trace["x"][-1]

    @kind("level")
    def m2(trace):
        return trace["x"][0]

    assert definition_hash(outer()) == definition_hash(m)
    assert definition_hash(m) != definition_hash(m2)


def test_closure_hash_is_config_independent():
    a = make_env("governed_commons").metrics["stock_pct"]
    b = make_env("governed_commons", K_cap=250.0).metrics["stock_pct"]
    assert metric_id("stock_pct", a) == metric_id("stock_pct", b)


def _load(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MODULE = """
    from cilib.metrics.identity import kind
    LATE = {late}

    def helper(x):
        return {helper_body}

    def make(cfg):
        def window(s, frac=LATE):
            return s[int(len(s) * frac):]

        @kind("level")
        def m(trace):
            return helper(window(trace["x"]))

        return {{"m": m}}
"""


def test_hash_follows_helpers_constants_and_defaults(tmp_path):
    """The walk reaches a module helper by name, a helper through a default
    argument, and a module constant; changing any of them moves the hash."""
    base = _load(tmp_path, "ident_a", MODULE.format(late=0.75, helper_body="x.sum()"))
    same = _load(tmp_path, "ident_b", MODULE.format(late=0.75, helper_body="x.sum()"))
    helper_changed = _load(tmp_path, "ident_c", MODULE.format(late=0.75, helper_body="x.sum() * 0.5"))
    constant_changed = _load(tmp_path, "ident_d", MODULE.format(late=0.5, helper_body="x.sum()"))

    h = lambda mod: definition_hash(mod.make(None)["m"])
    assert h(base) == h(same)
    assert h(base) != h(helper_changed)
    assert h(base) != h(constant_changed)
