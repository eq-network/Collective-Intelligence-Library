"""
Metrics for the Coupled Society — per-domain human shares, their joint decline,
and the defense-transfer-gap instrument (the leaderboard's headline number).

Per-domain vital signs (each late-run mean, the ``_late`` idiom):

- ``human_income_share`` economy: who the income actually goes to (sees rents;
  the composite's economy leg). ``labor_share`` (wage·L/Y) is kept as the
  classical factor-share readout, blind to rule-capture by construction.
- ``human_culture_share`` culture: humans still holding human-origin culture
- ``human_influence_share`` politics: citizens' total consensus weight
- ``composite_human_share``: their mean — the single "how are the humans
  doing across the whole society" number.
- ``correlated_decline``: mean pairwise Pearson correlation of the three
  human-share time series — the alpha plan's correlated-decline index (near 0
  when domains move independently; near 1 under joint lock-in).

``defense_transfer_gap`` is NOT a trace metric — it is the paired-twin
instrument (counterfactual.py's same-key pattern): run the SAME defenses in
the coupled env and its ``kappa = 0`` twin with the same key, difference the
composite. At kappa = 0 the twin is bit-identical, so the gap is exactly 0 by
construction — the instrument's null is built in.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from cilib.metrics.identity import kind


def _late(series):
    return series[3 * series.shape[0] // 4:]


def _domain_series(trace, n_humans):
    """The three human-share series over the full run, each shape (T,). The
    economy leg is the human INCOME share, not the factor share — rule-capture
    moves income without moving wage·L/Y, and GD §2/§5's economic
    disempowerment is about who the income goes to."""
    total = jnp.sum(trace["last_reward"], axis=1)
    human = jnp.sum(trace["last_reward"][:, :n_humans], axis=1)
    income = jnp.clip(human / jnp.maximum(total, 1e-6), 0.0, 1.0)
    culture = jnp.mean(1.0 - trace["culture"][:, :n_humans], axis=1)
    influence = jnp.sum(trace["influence"][:, :n_humans], axis=1)
    return income, culture, influence


def make_metrics(cfg):
    H = cfg.n_humans

    @kind("share")
    def labor_share(trace):
        """Factor share wage·L/Y — kept as the classical readout; blind to
        rents by construction (see _domain_series)."""
        labor_income = trace["wage"] * jnp.sum(trace["labor_supply"], axis=1)
        return jnp.mean(_late(labor_income / jnp.maximum(trace["output"], 1e-6)))

    @kind("share")
    def human_income_share(trace):
        return jnp.mean(_late(_domain_series(trace, H)[0]))

    @kind("share")
    def human_culture_share(trace):
        return jnp.mean(_late(_domain_series(trace, H)[1]))

    @kind("share")
    def human_influence_share(trace):
        return jnp.mean(_late(_domain_series(trace, H)[2]))

    @kind("share")
    def composite_human_share(trace):
        l, c, i = _domain_series(trace, H)
        return (jnp.mean(_late(l)) + jnp.mean(_late(c)) + jnp.mean(_late(i))) / 3.0

    @kind("index")
    def correlated_decline(trace):
        l, c, i = _domain_series(trace, H)

        def corr(a, b):
            a = a - jnp.mean(a)
            b = b - jnp.mean(b)
            denom = jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b))
            return jnp.where(denom > 1e-9, jnp.sum(a * b) / denom, 0.0)

        return (corr(l, c) + corr(c, i) + corr(l, i)) / 3.0

    return {
        "labor_share": labor_share,
        "human_income_share": human_income_share,
        "human_culture_share": human_culture_share,
        "human_influence_share": human_influence_share,
        "composite_human_share": composite_human_share,
        "correlated_decline": correlated_decline,
    }


def defense_transfer_gap(coupled_env, uncoupled_env, key, n_seeds: int,
                         n_steps: int):
    """``composite(uncoupled) − composite(coupled)`` over paired same-key
    batches — how much of a defense portfolio's protection evaporates once the
    domains feed each other. Both envs must carry the same defenses; the
    uncoupled twin is the same builder at ``kappa = 0``."""
    _, tr_c = coupled_env.run_batch(key, n_seeds, n_steps)
    _, tr_u = uncoupled_env.run_batch(key, n_seeds, n_steps)
    comp_c = jax.vmap(coupled_env.metrics["composite_human_share"])(tr_c)
    comp_u = jax.vmap(uncoupled_env.metrics["composite_human_share"])(tr_u)
    return jnp.mean(comp_u - comp_c)
