# graph_transformation/__init__.py
import sys
import os
from pathlib import Path

# Get the directory containing this file (graph_transformation)
package_dir = Path(__file__).resolve().parent

# Add the parent directory (Mycorrhiza) to Python path
project_root = package_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))