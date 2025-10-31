# core/scheduler.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np
import jax.random as jrandom

from core.graph import GraphState
from core.category import Transform

__all__ = ["MechanismSchedule", "MultiMechanismRunner"]


@dataclass(frozen=True)
class MechanismSchedule:
    name: str
    transform: Transform
    frequency: int
    phase: int = 0
    mask_key: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class MultiMechanismRunner:
    def __init__(
        self,
        initial_state: GraphState,
        mechanisms: List[MechanismSchedule],
        watch_keys: Optional[List[str]] = None,
        delta_eps: float = 1e-9,
    ):
        self.state = initial_state
        self.mechanisms = sorted(
            mechanisms,
            key=lambda m: (m.phase % max(1, m.frequency), m.frequency, m.name),
        )
        self.timestep = int(getattr(initial_state, "global_attrs", {}).get("tick", 0))
        self.execution_log: List[Dict[str, Any]] = []

        self.watch_keys = list(watch_keys or [])
        self.delta_eps = float(delta_eps)

    # --- helpers ---
    def _read_numpy(self, st: GraphState) -> Dict[str, np.ndarray]:
        out = {}
        for k in self.watch_keys:
            if k in st.node_attrs:
                out[k] = np.asarray(st.node_attrs[k])
        return out

    def _sparse_delta(
        self, before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, list]]:
        sparse: Dict[str, Dict[str, list]] = {}
        for k in self.watch_keys:
            if k not in before or k not in after:
                continue
            b, a = before[k], after[k]
            if b.shape != a.shape:
                continue
            if b.ndim == 1:
                d = a - b
                idx = np.flatnonzero(np.abs(d) > self.delta_eps)
                if idx.size:
                    sparse[k] = {
                        "idx": idx.astype(int).tolist(),
                        "delta": d[idx].astype(float).tolist(),
                        "before": b[idx].astype(float).tolist(),
                        "after": a[idx].astype(float).tolist(),
                    }
            elif b.ndim == 2:
                for j in range(b.shape[1]):
                    d = a[:, j] - b[:, j]
                    idx = np.flatnonzero(np.abs(d) > self.delta_eps)
                    if idx.size:
                        sparse[f"{k}:{j}"] = {
                            "idx": idx.astype(int).tolist(),
                            "delta": d[idx].astype(float).tolist(),
                            "before": b[idx, j].astype(float).tolist(),
                            "after": a[idx, j].astype(float).tolist(),
                        }
        return sparse

    def step(self) -> GraphState:
        st = self.state.update_global_attr("tick", self.timestep)

        metrics_before = self.monitor.compute_metrics(st) if self.monitor else {}
        st, tick_key = st.split_rng()
        before = self._read_numpy(st)

        active: List[Dict[str, Any]] = []
        for mech in self.mechanisms:
            if self.timestep - mech.phase >= 0 and (
                (self.timestep - mech.phase) % max(1, mech.frequency) == 0
            ):
                if mech.mask_key:
                    st = st.update_global_attr("active_mask", mech.mask_key)
                if tick_key is not None:
                    tick_key, mech_key = jrandom.split(tick_key)
                    st = st.update_global_attr("rng_key_for_mech", mech_key)
                if not callable(mech.transform):
                    raise TypeError(
                        f"Schedule '{mech.name}' has non-callable transform: {mech.transform!r}"
                    )
                st = mech.transform(st)
                active.append(
                    {"name": mech.name, "params": mech.params, "mask": mech.mask_key}
                )

        after = self._read_numpy(st)
        delta_sparse = self._sparse_delta(before, after)

        # capture per-tick events (e.g., from market/democracy)
        events = {
            "trades_edges": st.global_attrs.get("trades_edges", []),
            "winner_policy": st.global_attrs.get("policy", None),
        }

        metrics_after = self.monitor.compute_metrics(st) if self.monitor else {}

        self.execution_log.append(
            {
                "tick": self.timestep,
                "active_mechanisms": active,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "delta_sparse": delta_sparse,
                "events": events,
            }
        )

        self.state = st
        self.timestep += 1
        return self.state

    def run(self, num_steps: int):
        for _ in range(num_steps):
            self.step()
        return [self.state]

    def save_log(self, path: Optional[str] = None):
        path = path or f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(self.execution_log, f, indent=2)
        return path
