"""
ScreenManager: Navigation controller for multi-screen application.

Manages screen lifecycle, navigation stack, and non-blocking update loop.
"""
import tkinter as tk
from tkinter import ttk
from typing import Dict, Type, Optional, List

from studio.screens.base import Screen
from studio.app_state import AppState


class ScreenManager:
    """
    Manages screen navigation and lifecycle.

    Features:
        - Single root window with swappable screen frames
        - Navigation stack for back navigation
        - Non-blocking update loop for animations
        - Shared AppState passed to all screens
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        title: str = "Mycorrhiza Studio"
    ):
        """
        Initialize the screen manager.

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        # Create root window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(800, 600)

        # Shared application state
        self.app_state = AppState()

        # Screen registry: name -> Screen class
        self._screen_registry: Dict[str, Type[Screen]] = {}

        # Navigation stack (for back navigation)
        self._screen_stack: List[Screen] = []

        # Currently active screen
        self._current_screen: Optional[Screen] = None

        # Update loop state
        self._update_interval = 16  # ~60 FPS
        self._is_running = False

        # Create main container frame
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Bind window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def register_screen(self, name: str, screen_class: Type[Screen]) -> None:
        """
        Register a screen class with a name.

        Args:
            name: Identifier for navigation
            screen_class: Screen subclass to instantiate
        """
        self._screen_registry[name] = screen_class

    def navigate_to(self, screen_name: str, push: bool = True) -> None:
        """
        Navigate to a screen by name.

        Args:
            screen_name: Registered screen name
            push: If True, push current screen to stack (enables back)
        """
        if screen_name not in self._screen_registry:
            raise ValueError(f"Unknown screen: {screen_name}")

        # Exit current screen
        prev_screen = self._current_screen
        if prev_screen:
            prev_screen.on_exit()
            if prev_screen.frame:
                prev_screen.frame.pack_forget()

            # Push to stack if requested
            if push:
                self._screen_stack.append(prev_screen)

        # Create new screen instance
        screen_class = self._screen_registry[screen_name]
        new_screen = screen_class(self, self.app_state)

        # Enter new screen
        new_screen.on_enter(prev_screen)
        if new_screen.frame:
            new_screen.frame.pack(fill=tk.BOTH, expand=True)

        self._current_screen = new_screen

    def navigate_back(self) -> bool:
        """
        Return to previous screen in stack.

        Returns:
            True if navigation occurred, False if stack empty
        """
        if not self._screen_stack:
            return False

        # Exit current screen
        current = self._current_screen
        if current:
            current.on_exit()
            current.destroy()
            if current.frame:
                current.frame.pack_forget()

        # Pop and enter previous screen
        prev_screen = self._screen_stack.pop()

        # Re-enter the previous screen (recreate UI)
        prev_screen.on_enter(current)
        if prev_screen.frame:
            prev_screen.frame.pack(fill=tk.BOTH, expand=True)

        self._current_screen = prev_screen
        return True

    def replace_screen(self, screen_name: str) -> None:
        """
        Replace current screen without pushing to stack.

        Use for transitions that shouldn't allow "back".
        """
        self.navigate_to(screen_name, push=False)

    def clear_stack(self) -> None:
        """Clear navigation history."""
        for screen in self._screen_stack:
            screen.destroy()
        self._screen_stack.clear()

    def run(self) -> None:
        """
        Start the application with non-blocking update loop.

        Uses root.after() for periodic updates instead of blocking mainloop.
        """
        self._is_running = True
        self._update_loop()
        self.root.mainloop()

    def _update_loop(self) -> None:
        """Internal update loop using Tk's after()."""
        if not self._is_running:
            return

        # Update current screen
        if self._current_screen:
            self._current_screen.on_update(self._update_interval / 1000.0)

        # Schedule next update
        self.root.after(self._update_interval, self._update_loop)

    def _on_close(self) -> None:
        """Handle window close."""
        self._is_running = False

        # Cleanup all screens
        if self._current_screen:
            self._current_screen.on_exit()
            self._current_screen.destroy()

        for screen in self._screen_stack:
            screen.destroy()

        self.root.destroy()
