"""
Mycorrhiza Studio - Main Entry Point

Launch the visual simulation studio with multi-screen navigation.

Usage:
    python -m studio.main      (preferred - uses package imports)
    python studio/main.py      (works but requires path setup below)
"""
import sys
from pathlib import Path

# Add parent directory to path ONLY when running directly (not as module)
# This allows `python studio/main.py` to work without installation
_project_root = str(Path(__file__).parent.parent)
if __name__ == "__main__" and _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from studio.screen_manager import ScreenManager
from studio.screens.welcome import WelcomeScreen
from studio.screens.scenario import ScenarioSelectionScreen
from studio.screens.configuration import ConfigurationScreen
from studio.screens.editor import GraphEditorScreen
from studio.screens.simulation import SimulationScreen
from studio.screens.export import ExportScreen


def main():
    """Launch Mycorrhiza Studio."""
    print("=" * 50)
    print("  Mycorrhiza Studio")
    print("  A Programming Language for Collective Intelligence")
    print("=" * 50)
    print()
    print("Starting application...")
    print()

    # Create screen manager
    manager = ScreenManager(
        width=1024,
        height=768,
        title="Mycorrhiza Studio"
    )

    # Register all screens
    manager.register_screen("welcome", WelcomeScreen)
    manager.register_screen("scenario", ScenarioSelectionScreen)
    manager.register_screen("configuration", ConfigurationScreen)
    manager.register_screen("editor", GraphEditorScreen)
    manager.register_screen("simulation", SimulationScreen)
    manager.register_screen("export", ExportScreen)

    # Start at welcome screen
    manager.navigate_to("welcome")

    # Run application (blocks until window closed)
    manager.run()

    print()
    print("Application closed.")


if __name__ == "__main__":
    main()
