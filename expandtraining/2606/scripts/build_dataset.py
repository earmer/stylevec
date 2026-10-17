from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_triplets import build_training_triplets
from config import load_config
from prebuilt import data_key, dataset_dir, default_name, read_manifest, write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build triplets once; train.py/validate.py --dataset load them.")
    parser.add_argument("--config", required=True, nargs="+", help="one or more configs; equal data keys build once")
    parser.add_argument("--run-name", default=None, help="dataset name (default d-<data key>); only with one config")
    parser.add_argument("--force", action="store_true", help="rebuild even if a matching dataset exists")
    args = parser.parse_args()
    if args.run_name and len(args.config) > 1:
        parser.error("--run-name names one dataset; pass a single --config")
    for config_path in args.config:
        config = load_config(config_path)
        name = args.run_name or default_name(config)
        out_dir = dataset_dir(name)
        if not args.force and (out_dir / "manifest.json").is_file():
            if read_manifest(name).get("data_key") == data_key(config):
                print(f"dataset up to date: {out_dir} ({config_path})")
                continue
            if args.run_name:
                sys.exit(f"{out_dir} holds a different data key; pick another --run-name or pass --force")
        triplets, manifest = build_training_triplets(config)
        write_dataset(config, out_dir, triplets, manifest)
        print(f"dataset outputs: {out_dir} ({len(triplets)} triplets, {config_path})")


if __name__ == "__main__":
    main()
