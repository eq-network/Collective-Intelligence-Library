"""
Analysis functions for the basin stability experiment.

Computes paper-ready metrics:
- Basin stability curves with Wilson CIs
- Breakdown thresholds (min p_adv where BS < 0.5)
- Delegation Gini coefficient (PLD weight concentration)
- Capture rate (PRD adversarial representative fraction)
- Fisher exact pairwise comparisons with Holm-Bonferroni correction
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import jax.numpy as jnp


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def basin_stability_curve(results: Dict) -> Dict[str, List]:
    """Extract BS +/- Wilson CI per mechanism across adversarial fractions.

    Args:
        results: output from run_sweep()

    Returns:
        Dict mapping mechanism -> {fractions, bs, ci_lower, ci_upper}
    """
    curves = {}
    for mechanism in results:
        if mechanism == "R_coop":
            continue
        fractions = []
        bs_values = []
        ci_lowers = []
        ci_uppers = []

        for key in sorted(results[mechanism].keys()):
            if key == "R_coop":
                continue
            try:
                frac = float(key)
            except ValueError:
                continue
            data = results[mechanism][key]
            fractions.append(frac)
            bs_values.append(data["basin_stability"])
            ci_lowers.append(data["ci_lower"])
            ci_uppers.append(data["ci_upper"])

        curves[mechanism] = {
            "fractions": fractions,
            "bs": bs_values,
            "ci_lower": ci_lowers,
            "ci_upper": ci_uppers,
        }

    return curves


def breakdown_threshold(results: Dict) -> Dict[str, Optional[float]]:
    """Find minimum adversarial fraction where BS drops below 0.5.

    This is the key robustness metric: higher threshold = more resilient.

    Args:
        results: output from run_sweep()

    Returns:
        Dict mapping mechanism -> threshold (or None if never breaks down)
    """
    thresholds = {}
    for mechanism in results:
        if not isinstance(results[mechanism], dict):
            continue

        threshold = None
        for key in sorted(results[mechanism].keys()):
            if key == "R_coop":
                continue
            try:
                frac = float(key)
            except ValueError:
                continue
            data = results[mechanism][key]
            if data["basin_stability"] < 0.5:
                threshold = frac
                break

        thresholds[mechanism] = threshold

    return thresholds


def delegation_gini(trust_scores: jnp.ndarray) -> float:
    """Compute Gini coefficient of PLD delegation weights.

    Measures concentration of voting power. Gini=0 means perfectly equal,
    Gini=1 means one agent holds all influence.

    Args:
        trust_scores: (N, N) trust matrix

    Returns:
        Gini coefficient in [0, 1]
    """
    # Agent weights = column sums (how much others trust this agent)
    weights = jnp.sum(trust_scores, axis=0)
    weights = weights / (jnp.sum(weights) + 1e-8)

    n = weights.shape[0]
    sorted_w = jnp.sort(weights)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    return float((2 * jnp.sum(index * sorted_w) / (n * jnp.sum(sorted_w))) - (n + 1) / n)


def capture_rate(rep_mask: jnp.ndarray, node_types: jnp.ndarray) -> float:
    """Fraction of PRD representatives that are adversarial.

    Args:
        rep_mask: (N,) binary mask of current representatives
        node_types: (N,) 0=cooperative, 1=adversarial

    Returns:
        Fraction of reps that are adversarial
    """
    n_reps = jnp.sum(rep_mask)
    n_adv_reps = jnp.sum(rep_mask * node_types)
    return float(n_adv_reps / (n_reps + 1e-8))


def fisher_exact_test(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test p-value for 2x2 contingency table.

    Tests whether mechanism A has higher basin stability than mechanism B.

        |        | Stable | Unstable |
        |--------|--------|----------|
        | Mech A |   a    |    b     |
        | Mech B |   c    |    d     |

    Uses log-space computation to handle large factorials.
    """
    n = a + b + c + d

    def log_factorial(x):
        return sum(math.log(i) for i in range(1, x + 1))

    log_p_cutoff = (
        log_factorial(a + b) + log_factorial(c + d) +
        log_factorial(a + c) + log_factorial(b + d) -
        log_factorial(n) -
        log_factorial(a) - log_factorial(b) -
        log_factorial(c) - log_factorial(d)
    )
    p_cutoff = math.exp(log_p_cutoff)

    # Sum probabilities for all tables as extreme or more extreme
    p_value = 0.0
    for i in range(min(a + b, a + c) + 1):
        j = (a + b) - i
        k = (a + c) - i
        l = (c + d) - k
        if j < 0 or k < 0 or l < 0:
            continue
        log_p = (
            log_factorial(a + b) + log_factorial(c + d) +
            log_factorial(a + c) + log_factorial(b + d) -
            log_factorial(n) -
            log_factorial(i) - log_factorial(j) -
            log_factorial(k) - log_factorial(l)
        )
        p = math.exp(log_p)
        if p <= p_cutoff + 1e-12:
            p_value += p

    return p_value


def fisher_exact_pairwise(results: Dict, adversarial_fraction: str = "0.3") -> Dict:
    """Pairwise Fisher exact tests with Holm-Bonferroni correction.

    Compares all pairs of mechanisms at a given adversarial fraction.

    Args:
        results: output from run_sweep()
        adversarial_fraction: which fraction to compare at

    Returns:
        Dict with pairwise comparisons and corrected p-values
    """
    mechanisms = [m for m in results if isinstance(results[m], dict) and adversarial_fraction in results[m]]

    comparisons = []
    for i in range(len(mechanisms)):
        for j in range(i + 1, len(mechanisms)):
            m1, m2 = mechanisms[i], mechanisms[j]
            d1 = results[m1][adversarial_fraction]
            d2 = results[m2][adversarial_fraction]

            a = d1["n_stable"]
            b = d1["n_total"] - d1["n_stable"]
            c = d2["n_stable"]
            d_val = d2["n_total"] - d2["n_stable"]

            p = fisher_exact_test(a, b, c, d_val)
            comparisons.append({
                "mechanism_1": m1,
                "mechanism_2": m2,
                "bs_1": d1["basin_stability"],
                "bs_2": d2["basin_stability"],
                "p_value": p,
            })

    # Holm-Bonferroni correction
    comparisons.sort(key=lambda x: x["p_value"])
    n_comparisons = len(comparisons)
    for rank, comp in enumerate(comparisons):
        comp["corrected_p"] = min(1.0, comp["p_value"] * (n_comparisons - rank))
        comp["significant"] = comp["corrected_p"] < 0.05

    return {
        "adversarial_fraction": adversarial_fraction,
        "comparisons": comparisons,
    }


def load_and_analyze(results_path: str) -> Dict:
    """Load results JSON and compute all analysis metrics.

    Args:
        results_path: path to basin_stability_*.json

    Returns:
        Dict with curves, thresholds, and pairwise comparisons
    """
    with open(results_path) as f:
        results = json.load(f)

    analysis = {
        "curves": basin_stability_curve(results),
        "thresholds": breakdown_threshold(results),
    }

    # Pairwise comparisons at multiple adversarial fractions
    for frac in ["0.2", "0.3", "0.4"]:
        analysis[f"pairwise_{frac}"] = fisher_exact_pairwise(results, frac)

    return analysis
