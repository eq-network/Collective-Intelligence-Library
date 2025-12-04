"""
EditSession: Mutable UI coordinator for immutable GraphState editing.

This module implements the EditSession pattern - a mutable wrapper that enables
interactive graph editing while maintaining functional purity of GraphState.

Key Insight: EditSession is the ONLY mutable component, isolated in UI layer.
All graph operations remain pure functions (GraphState → GraphState).
"""
from typing import Dict, Set, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass, field
from core.graph import GraphState
from core.category import Transform


@dataclass
class EditMetadata:
    """
    UI-only state that doesn't belong in GraphState.

    This includes visual layout information and transient UI selections.
    """
    # Custom node positions (overrides VizConfig auto-layout)
    node_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Current UI selection state
    selected_nodes: Set[int] = field(default_factory=set)
    selected_edges: Set[Tuple[int, int]] = field(default_factory=set)

    # Edge directionality per relation type
    edge_directionality: Dict[str, bool] = field(default_factory=dict)

    def copy(self) -> 'EditMetadata':
        """Create a deep copy of metadata."""
        return EditMetadata(
            node_positions=dict(self.node_positions),
            selected_nodes=set(self.selected_nodes),
            selected_edges=set(self.selected_edges),
            edge_directionality=dict(self.edge_directionality)
        )


class EditSession:
    """
    Mutable coordinator for graph editing with undo/redo support.

    Manages:
    - Current GraphState (immutable snapshot)
    - History for undo (stack of previous states)
    - Future for redo (stack of undone states)
    - UI metadata (positions, selections)

    Usage:
        session = EditSession(initial_state, config)
        session.apply_edit(lambda s: add_node(s, node_type=0))
        session.undo()
        final_state = session.export_final()
    """

    def __init__(
        self,
        initial_state: GraphState,
        max_history: int = 100
    ):
        """
        Initialize edit session.

        Args:
            initial_state: Starting graph state
            max_history: Maximum undo history size (default 100)
        """
        self.current_state: GraphState = initial_state
        self.metadata: EditMetadata = EditMetadata()

        # History management
        self.history: List[Tuple[GraphState, EditMetadata]] = []
        self.future: List[Tuple[GraphState, EditMetadata]] = []
        self.max_history = max_history

        # Callbacks for UI updates
        self.on_state_changed: Optional[Callable[[GraphState], None]] = None
        self.on_metadata_changed: Optional[Callable[[EditMetadata], None]] = None

    def apply_edit(self, edit_fn: Transform, description: str = "") -> GraphState:
        """
        Apply a pure edit function to current state.

        Saves current state to history before applying edit.
        Clears redo stack (branching from new timeline).

        Args:
            edit_fn: Pure function GraphState → GraphState
            description: Optional description for debugging

        Returns:
            New current state after edit

        Example:
            session.apply_edit(
                lambda s: add_node(s, node_type=0, initial_attrs={"resources": 100}),
                "Add resource node"
            )
        """
        # Save current state to history
        self._push_to_history()

        # Apply pure edit function
        new_state = edit_fn(self.current_state)

        # Update current state
        self.current_state = new_state

        # Clear future (branching timeline)
        self.future.clear()

        # Notify listeners
        if self.on_state_changed:
            self.on_state_changed(new_state)

        return new_state

    def undo(self) -> Optional[GraphState]:
        """
        Restore previous state from history.

        Returns:
            Previous state if history exists, None otherwise
        """
        if not self.history:
            return None

        # Save current to future (for redo)
        self.future.append((self.current_state, self.metadata.copy()))

        # Restore previous state
        prev_state, prev_metadata = self.history.pop()
        self.current_state = prev_state
        self.metadata = prev_metadata

        # Notify listeners
        if self.on_state_changed:
            self.on_state_changed(prev_state)
        if self.on_metadata_changed:
            self.on_metadata_changed(prev_metadata)

        return prev_state

    def redo(self) -> Optional[GraphState]:
        """
        Restore undone state from future.

        Returns:
            Next state if future exists, None otherwise
        """
        if not self.future:
            return None

        # Save current to history
        self._push_to_history()

        # Restore future state
        next_state, next_metadata = self.future.pop()
        self.current_state = next_state
        self.metadata = next_metadata

        # Notify listeners
        if self.on_state_changed:
            self.on_state_changed(next_state)
        if self.on_metadata_changed:
            self.on_metadata_changed(next_metadata)

        return next_state

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.history) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.future) > 0

    def update_metadata(self, **updates) -> None:
        """
        Update UI metadata fields.

        Args:
            **updates: Keyword arguments matching EditMetadata fields

        Example:
            session.update_metadata(selected_nodes={0, 1, 2})
        """
        for key, value in updates.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

        if self.on_metadata_changed:
            self.on_metadata_changed(self.metadata)

    def set_node_position(self, node_id: int, x: float, y: float) -> None:
        """
        Set custom position for a node.

        Args:
            node_id: Node to position
            x: X coordinate [0, 1]
            y: Y coordinate [0, 1]
        """
        self.metadata.node_positions[node_id] = (x, y)

        if self.on_metadata_changed:
            self.on_metadata_changed(self.metadata)

    def get_node_position(self, node_id: int) -> Optional[Tuple[float, float]]:
        """
        Get custom position for a node.

        Returns:
            (x, y) if custom position exists, None otherwise
        """
        return self.metadata.node_positions.get(node_id)

    def select_node(self, node_id: int, multi_select: bool = False) -> None:
        """
        Select a node.

        Args:
            node_id: Node to select
            multi_select: If True, add to selection. If False, replace selection.
        """
        if not multi_select:
            self.metadata.selected_nodes.clear()
            self.metadata.selected_edges.clear()

        self.metadata.selected_nodes.add(node_id)

        if self.on_metadata_changed:
            self.on_metadata_changed(self.metadata)

    def deselect_all(self) -> None:
        """Clear all selections."""
        self.metadata.selected_nodes.clear()
        self.metadata.selected_edges.clear()

        if self.on_metadata_changed:
            self.on_metadata_changed(self.metadata)

    def is_node_selected(self, node_id: int) -> bool:
        """Check if node is selected."""
        return node_id in self.metadata.selected_nodes

    def get_selected_nodes(self) -> Set[int]:
        """Get set of selected node IDs."""
        return set(self.metadata.selected_nodes)

    def export_final(self) -> GraphState:
        """
        Export immutable GraphState for simulation.

        Returns final state without UI metadata.
        This is what gets passed to the simulation engine.

        Returns:
            Current GraphState (immutable)
        """
        return self.current_state

    def clear_history(self) -> None:
        """Clear undo/redo history."""
        self.history.clear()
        self.future.clear()

    def _push_to_history(self) -> None:
        """Push current state to history stack."""
        self.history.append((self.current_state, self.metadata.copy()))

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
