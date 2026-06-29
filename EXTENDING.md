# Extending CI Lib

Every extension is the same move: **write a factory that satisfies a catalog's type
function, then add one line to that catalog's `REGISTRY`.** Open the catalog's
`__init__.py` to see its current entries and the `README.md` beside it for the
contract. Background: [ARCHITECTURE.md](ARCHITECTURE.md).

Contracts (the "type functions") all live in
[`src/cilib/core/protocols.py`](src/cilib/core/protocols.py).

## Add an agent → `src/cilib/agents/`

A `Policy` is `(obs, key) -> action`; a `PureAgent` exposes `round_fn() -> (state,t,key)->state`.

```python
# agents/greedy.py
class GreedyPolicy:
    def __init__(self, cfg): ...
    def __call__(self, obs, key): return obs.argmax(...)
# agents/__init__.py
from .greedy import GreedyPolicy
REGISTRY = { ..., "greedy": GreedyPolicy }
```

## Add a transformation → `src/cilib/transformations/`

An atomic, pure, vectorized `Config -> Transform`.

```python
# transformations/decay.py
from cilib.core import transform
def make_decay(cfg):
    @transform(reads=["score"], writes=["score"])
    def decay(state):
        return state.update_node_attrs("score", state.node_attrs["score"] * cfg.rate)
    return decay
# transformations/__init__.py  ->  REGISTRY["decay"] = make_decay
```

## Add a mechanism → `src/cilib/mechanisms/`

A *composition* of transformations, with declared writes disjoint from its family
siblings (so the **swap test** holds: another family member drops into the same
pipeline slot). Families: `market`, `network`, `democracy`.

```python
# mechanisms/democracy.py  (family: democracy, writes: harvest_target, penalty)
def make_direct_democracy(cfg):
    @transform(reads=["vote_value"], writes=["harvest_target", "penalty"])
    def pdd(state): ...        # median vote -> quota + penalty
    return pdd
# mechanisms/__init__.py  ->  REGISTRY["direct_democracy"] = make_direct_democracy
```

## Add an environment → `src/cilib/environments/`

A `(**cfg) -> EnvSpec` builder. Mirror `environments/commons_harvest/` (config /
state / dynamics / metrics), then:

```python
# environments/__init__.py
from .my_env import build_my_env
REGISTRY = { ..., "my_env": build_my_env }
```

## Add a paradigm → `src/cilib/paradigms/<name>/`

A self-contained model. Follow the 6-part contract in `paradigms/README.md` (schema /
primitives / transforms / agents / environments / tests). `polycentric/` is the
worked example. Compose its round with `compile_pipeline`, run via `run_scan_batch`.

## Add an experiment → `experiments/<name>/`

Copy `experiments/_template/`: `config.py` (frozen config + sweep axes), `run.py`
(build state → `compile_pipeline` → `run_scan_batch` → save), `figures.py`,
`README.md` (hypothesis + sweep). Experiments import `cilib`; they are not part of the
installed package.

---

**Checklist for any addition:** factory satisfies the type function · one `REGISTRY`
line · a behavioral test (assert the mechanism, not bit-exact numbers) · `pytest` green.
