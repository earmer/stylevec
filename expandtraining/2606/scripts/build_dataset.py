from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_triplets import build_training_triplets
from config import load_config, workspace_dir
from records import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    triplets, manifest = build_training_triplets(config)
    out_dir = workspace_dir() / "outputs" / "datasets" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_dir / "triplets.jsonl"), [row.to_json_dict() for row in triplets])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"dataset outputs: {out_dir}")


if __name__ == "__main__":
    main()
