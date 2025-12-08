"""
Toolbar: Tkinter UI component for editing tools.

Provides visual tool selection with buttons:
[Select] [Add Node] [Add Edge] [Delete] | [Undo] [Redo]

Highlights active tool and updates on tool changes.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional

from studio.edit_mode import EditMode, Tool


class EditToolbar:
    """
    Tkinter toolbar for graph editing tools.

    Provides buttons for tool selection and undo/redo operations.
    Automatically highlights the active tool.
    """

    def __init__(self, parent: tk.Widget, edit_mode: EditMode):
        """
        Create toolbar.

        Args:
            parent: Parent Tkinter widget
            edit_mode: EditMode instance to coordinate with
        """
        self.edit_mode = edit_mode
        self.active_button: Optional[tk.Button] = None

        # Create frame for toolbar
        self.frame = ttk.Frame(parent, padding="5")

        # Tool buttons
        self.tool_buttons = {}

        # Select tool
        self.tool_buttons[Tool.SELECT] = self._create_tool_button(
            "Select (S)",
            Tool.SELECT,
            "Click to select, drag to move nodes"
        )

        # Add Node tool
        self.tool_buttons[Tool.ADD_NODE] = self._create_tool_button(
            "Add Agent (N)",
            Tool.ADD_NODE,
            "Click empty space to create agent node"
        )

        # Add Market tool
        self.tool_buttons[Tool.ADD_MARKET] = self._create_tool_button(
            "Add Market (M)",
            Tool.ADD_MARKET,
            "Click to add market mechanism"
        )

        # Add Resource tool
        self.tool_buttons[Tool.ADD_RESOURCE] = self._create_tool_button(
            "Add Resource (R)",
            Tool.ADD_RESOURCE,
            "Click to add resource depot"
        )

        # Add Edge tool
        self.tool_buttons[Tool.ADD_EDGE] = self._create_tool_button(
            "Add Edge (E)",
            Tool.ADD_EDGE,
            "Drag from node to node to create edge"
        )

        # Delete tool
        self.tool_buttons[Tool.DELETE] = self._create_tool_button(
            "Delete (D)",
            Tool.DELETE,
            "Click node to delete"
        )

        # Separator
        ttk.Separator(self.frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=5
        )

        # Undo button
        self.undo_button = ttk.Button(
            self.frame,
            text="Undo (Ctrl+Z)",
            command=self._on_undo
        )
        self.undo_button.pack(side=tk.LEFT, padx=2)

        # Redo button
        self.redo_button = ttk.Button(
            self.frame,
            text="Redo (Ctrl+Y)",
            command=self._on_redo
        )
        self.redo_button.pack(side=tk.LEFT, padx=2)

        # Status label
        ttk.Separator(self.frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=5
        )
        self.status_label = ttk.Label(self.frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Register callbacks
        self.edit_mode.on_tool_changed = self._on_tool_changed

        # Set initial tool (Select)
        self._highlight_tool(Tool.SELECT)

        # Update undo/redo button states
        self._update_undo_redo_buttons()

    def _create_tool_button(
        self,
        text: str,
        tool: Tool,
        tooltip: str = ""
    ) -> tk.Button:
        """
        Create a tool selection button.

        Args:
            text: Button text
            tool: Tool this button activates
            tooltip: Tooltip text (not implemented yet)

        Returns:
            Created button
        """
        button = ttk.Button(
            self.frame,
            text=text,
            command=lambda: self._on_tool_select(tool)
        )
        button.pack(side=tk.LEFT, padx=2)

        # TODO: Add tooltip support

        return button

    def _on_tool_select(self, tool: Tool) -> None:
        """Handle tool button click."""
        self.edit_mode.set_tool(tool)

    def _on_tool_changed(self, tool: Tool) -> None:
        """Handle tool change from EditMode."""
        self._highlight_tool(tool)
        self.set_status(f"Tool: {tool.value.replace('_', ' ').title()}")

    def _highlight_tool(self, tool: Tool) -> None:
        """
        Highlight the active tool button.

        Args:
            tool: Tool to highlight
        """
        # Remove highlight from all buttons
        for t, button in self.tool_buttons.items():
            button.state(['!pressed'])

        # Highlight active tool
        if tool in self.tool_buttons:
            self.tool_buttons[tool].state(['pressed'])
            self.active_button = self.tool_buttons[tool]

    def _on_undo(self) -> None:
        """Handle undo button click."""
        self.edit_mode.session.undo()
        self._update_undo_redo_buttons()
        self.set_status("Undo")

    def _on_redo(self) -> None:
        """Handle redo button click."""
        self.edit_mode.session.redo()
        self._update_undo_redo_buttons()
        self.set_status("Redo")

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button enabled states."""
        session = self.edit_mode.session

        # Enable/disable based on history
        if session.can_undo():
            self.undo_button.state(['!disabled'])
        else:
            self.undo_button.state(['disabled'])

        if session.can_redo():
            self.redo_button.state(['!disabled'])
        else:
            self.redo_button.state(['disabled'])

    def set_status(self, message: str) -> None:
        """
        Update status message.

        Args:
            message: Status message to display
        """
        self.status_label.config(text=message)

    def update(self) -> None:
        """Update toolbar state (call after edits)."""
        self._update_undo_redo_buttons()
