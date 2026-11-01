"""Log each frozen-LM method's final metrics to SwanLab, next to the trained runs (run on the Mac after pulling).

Frozen methods have no training curves, so each gets one experiment holding its final/* metrics, grouped by
tag (frozen_<tag>) with the method as its arm. Runs already logged (a swanlab_id file) are skipped.

SWANLAB_API_KEY=... uv run python expandtraining/2606/scripts/frozen_swanlab.py --tag a --config <eval.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from config import load_config, workspace_dir
from tracking import ID_FILE
from validate import log_final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for run_dir in sorted((workspace_dir() / "outputs" / "runs").glob(f"frozen_{args.tag}__*__s0")):
        if (run_dir / "validation" / "retrieval_probe.json").exists() and not (run_dir / ID_FILE).exists():
            log_final_metrics(config, run_dir.name, run_dir)
            print(f"logged {run_dir.name}", flush=True)


if __name__ == "__main__":
    main()
