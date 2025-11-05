"""
Quick runner for Farmer's Market.

python run_farmers_market.py
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine.environments.farmers_market.example import run_simulation

if __name__ == "__main__":
    run_simulation(num_farmers=10, num_rounds=50, seed=42)
