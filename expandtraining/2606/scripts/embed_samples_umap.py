from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sample_umap import main


if __name__ == "__main__":
    main()
