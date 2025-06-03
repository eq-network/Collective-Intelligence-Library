import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


# Implement relational databases, look at hash tables and locally stored information for memory optimisation?
# Similar to proceduraly generated games, zip memory files that are accessed when doing node check-ins.