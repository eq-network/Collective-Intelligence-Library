"""
Screen: Base class for all application screens.

Implements game-style screen lifecycle:
- on_enter: Create UI when becoming active
- on_update: Called periodically for animations
- on_exit: Save state when leaving
"""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    from studio.screen_manager import ScreenManager
    from studio.app_state import AppState


class Screen(ABC):
    """
    Base class for all application screens.

    Lifecycle:
        1. __init__: Store references, don't create UI yet
        2. on_enter: Create UI, called when becoming active
        3. on_update: Called periodically for animations/simulation
        4. on_exit: Cleanup, called when leaving
        5. destroy: Final cleanup when screen removed from memory
    """

    def __init__(self, manager: 'ScreenManager', app_state: 'AppState'):
        """
        Initialize screen with references to manager and shared state.

        Args:
            manager: ScreenManager for navigation
            app_state: Shared application state
        """
        self.manager = manager
        self.app_state = app_state
        self.frame: Optional[tk.Frame] = None

    @property
    def name(self) -> str:
        """Screen identifier for navigation."""
        return self.__class__.__name__

    @abstractmethod
    def on_enter(self, prev_screen: Optional['Screen'] = None) -> None:
        """
        Called when screen becomes active. Create all UI elements here.

        Args:
            prev_screen: The screen we're coming from (for context)
        """
        pass

    def on_update(self, dt: float = 0.016) -> None:
        """
        Called periodically for animations, simulation stepping.

        Args:
            dt: Delta time since last update (seconds)
        """
        pass

    def on_exit(self, next_screen: Optional['Screen'] = None) -> None:
        """
        Called when leaving this screen. Save any state before leaving.

        Args:
            next_screen: The screen we're going to
        """
        pass

    def destroy(self) -> None:
        """Final cleanup when screen is removed from memory."""
        if self.frame:
            self.frame.destroy()
            self.frame = None

    # Navigation helpers

    def navigate_to(self, screen_name: str) -> None:
        """Navigate to another screen by name."""
        self.manager.navigate_to(screen_name)

    def navigate_back(self) -> None:
        """Return to previous screen."""
        self.manager.navigate_back()

    # UI helpers

    def _create_base_frame(self) -> tk.Frame:
        """Create the base frame for this screen's content."""
        self.frame = ttk.Frame(self.manager.container, padding="20")
        return self.frame

    def _create_nav_bar(self, title: str = "", show_back: bool = True) -> ttk.Frame:
        """
        Create standard navigation bar with optional back button.

        Args:
            title: Title to display in nav bar
            show_back: Whether to show back button

        Returns:
            The nav bar frame
        """
        nav_frame = ttk.Frame(self.frame)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))

        if show_back:
            back_btn = ttk.Button(
                nav_frame,
                text="< Back",
                command=self.navigate_back
            )
            back_btn.pack(side=tk.LEFT)

        if title:
            title_label = ttk.Label(
                nav_frame,
                text=title,
                font=("Arial", 16, "bold")
            )
            title_label.pack(side=tk.LEFT, padx=20)

        return nav_frame
