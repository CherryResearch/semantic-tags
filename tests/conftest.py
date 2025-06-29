import sys
from pathlib import Path

# Ensure package root is on sys.path for test imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
