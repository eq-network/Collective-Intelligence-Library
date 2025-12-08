"""
WelcomeScreen: Main menu with Start, Tutorial, and Documentation options.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional

from studio.screens.base import Screen


class WelcomeScreen(Screen):
    """
    Welcome screen with main menu options.

    Options:
        - Start Simulation: Navigate to scenario selection
        - Tutorial: Placeholder (not implemented)
        - Documentation: Placeholder (not implemented)
    """

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()

        # Title
        title = ttk.Label(
            self.frame,
            text="Mycorrhiza Studio",
            font=("Arial", 32, "bold")
        )
        title.pack(pady=(80, 10))

        subtitle = ttk.Label(
            self.frame,
            text="A Programming Language for Collective Intelligence",
            font=("Arial", 14)
        )
        subtitle.pack(pady=(0, 60))

        # Menu buttons (centered)
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(expand=True)

        # Start Simulation button
        start_btn = ttk.Button(
            button_frame,
            text="Start Simulation",
            command=lambda: self.navigate_to("scenario"),
            width=25
        )
        start_btn.pack(pady=10)

        # Tutorial button (placeholder)
        tutorial_btn = ttk.Button(
            button_frame,
            text="Tutorial",
            command=self._show_tutorial,
            width=25
        )
        tutorial_btn.pack(pady=10)

        # Documentation button (placeholder)
        docs_btn = ttk.Button(
            button_frame,
            text="Documentation",
            command=self._open_docs,
            width=25
        )
        docs_btn.pack(pady=10)

        # Version info at bottom
        version = ttk.Label(
            self.frame,
            text="v0.1.0 - Visualiser Branch",
            font=("Arial", 10)
        )
        version.pack(side=tk.BOTTOM, pady=20)

    def _show_tutorial(self) -> None:
        """Placeholder for tutorial."""
        print("Tutorial not yet implemented")

    def _open_docs(self) -> None:
        """Placeholder for documentation."""
        print("Documentation not yet implemented")
