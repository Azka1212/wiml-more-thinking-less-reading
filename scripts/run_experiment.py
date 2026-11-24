"""Run from a source checkout; requires explicit GPU/model dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiml.experiment import main

if __name__ == "__main__":
    main()
