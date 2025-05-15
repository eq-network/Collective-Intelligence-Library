# __init__.py in the MYCORRHIZA root directory
import sys
import os
from pathlib import Path

# Add the current directory to sys.path if not already present
package_dir = Path(__file__).resolve().parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))