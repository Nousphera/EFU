#!/usr/bin/env python3
"""Prepare one named dataset or the complete paper dataset suite."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from efu.dataset_io import main

if __name__ == "__main__":
    main()
