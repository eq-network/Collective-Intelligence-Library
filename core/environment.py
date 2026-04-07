# core/environment.py
"""
Simulation environment built on transform-based stepping via jax.lax.scan.

The environment holds an initial GraphState and a composed Transform.
Running the simulation applies the transform T times via lax.scan,
producing a final GraphState with filled metric arrays.

Termination is handled via an 'alive' flag in global_attrs — the scan
always runs T iterations, but transforms become no-ops when alive=0.
"""
import jax.lax as lax

from core.graph import GraphState
from core.category import Transform


class Environment:
    """Transform-based simulation environment with lax.scan execution.

    The environment is fully defined by:
    1. An initial GraphState (immutable pytree)
    2. A composed step Transform (GraphState -> GraphState)

    Usage:
        env = Environment(initial_state, step_transform)
        final_state = env.run(T=200)
        # final_state.global_attrs contains filled metric arrays
    """

    def __init__(self, initial_state: GraphState, step_transform: Transform):
        self.state = initial_state
        self._initial_state = initial_state
        self._step_transform = step_transform

    def run(self, num_rounds: int) -> GraphState:
        """Run the simulation via jax.lax.scan.

        Applies step_transform num_rounds times. All state evolution,
        metric recording, and termination handling happen inside the
        transform pipeline (via the alive flag pattern).

        Args:
            num_rounds: Number of steps to execute.

        Returns:
            Final GraphState with filled metric arrays and terminal state.
        """
        self.reset()

        def scan_body(state, _):
            return self._step_transform(state), None

        self.state, _ = lax.scan(scan_body, self.state, None, length=num_rounds)
        return self.state

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = self._initial_state
