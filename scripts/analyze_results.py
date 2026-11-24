"""Run from a source checkout; no editable installation is necessary."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiml.analysis import main

if __name__ == "__main__":
    main()
