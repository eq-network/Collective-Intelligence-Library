"""Render the README tables from the committed json — never typed by hand.

    python -m experiments.perf_baseline.table                       # baseline.json
    python -m experiments.perf_baseline.table scaling-large.json    # a sweep file
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def baseline_table(d):
    p = d["provenance"]
    jax_version = p.get("jax_version", p.get("jax"))   # files before 2026-09-11 carry "jax"
    print(f"Measured {p['date']} at `{p['git_rev']}`, jax {jax_version}, {p['backend'][0]}, "
          f"{p['cpu_count']} cores, T={d['params']['T']}, best of {d['params']['repeats']}.\n")
    print("| env | N | eager call 1 | eager call 2 | jit call 1 | jit warm | compile | us/step | trace MB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in d["envs"].items():
        if "error" in r:
            print(f"| `{name}` | | error: {r['error']} |")
            continue
        print(f"| `{name}` | {r['n_nodes']} | {r['eager_call1_s']:.2f} s | {r['eager_call2_s']:.2f} s | "
              f"{r['jit_call1_s']:.2f} s | {r['jit_warm_s'] * 1e3:.1f} ms | {r['compile_est_s']:.2f} s | "
              f"{r['step_us']:.0f} | {r['trace_bytes'] / 1e6:.2f} |")


def seeds_table(d):
    seeds = d["params"]["seeds"]
    print("| env | " + " | ".join(f"S={S}" for S in seeds) + " |")
    print("|---|" + "---:|" * len(seeds))
    for name, r in d["envs"].items():
        if "seeds" not in r:
            continue
        cells = " | ".join(f"{r['seeds'][str(S)]['warm_s'] * 1e3:.0f} ms ({100 * (r['seeds'][str(S)]['efficiency'] or 0):.0f}%)"
                           for S in seeds)
        print(f"| `{name}` | {cells} |")


def scaling_table(d):
    if not d.get("scaling"):
        return
    mults = d["params"]["mults"]
    print("| env | " + " | ".join(f"{m}x" for m in mults) + " | p |")
    print("|---|" + "---:|" * (len(mults) + 1))
    for name, s in d["scaling"].items():
        if "error" in s:
            print(f"| `{name}` | error: {s['error']} |")
            continue
        cells = " | ".join(f"N={row['n_nodes']}: {row['warm_s'] * 1e3:.0f} ms" for row in s["rows"])
        pexp = f"{s['exponent']:.2f}" if s["exponent"] is not None else "n/a"
        print(f"| `{name}` | {cells} | {pexp} |")


if __name__ == "__main__":
    d = load(sys.argv[1] if len(sys.argv) > 1 else "baseline.json")
    print("## per environment\n"); baseline_table(d)
    print("\n## seed batches (jit warm; efficiency = S x single-seed time / batch time)\n"); seeds_table(d)
    print("\n## population scaling (jit warm; p from the first and last column)\n"); scaling_table(d)
