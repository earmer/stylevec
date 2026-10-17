from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ablation import SpecError, generated_dir, plan
from preflight import instance_checks, local_checks, preflight_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 checks for an ablation spec.")
    parser.add_argument("where", choices=("local", "instance"), help="local = 0a on the Mac; instance = 0b on the box")
    parser.add_argument("spec")
    parser.add_argument("--no-tests", action="store_true", help="0a: skip pytest")
    args = parser.parse_args()
    try:
        # local re-plans (and rewrites the generated configs); instance only reads what sync delivered.
        result = plan(args.spec, write=args.where == "local")
    except SpecError as error:
        raise SystemExit(f"spec: {error}") from None
    generated = generated_dir(result["name"])
    if args.where == "local":
        problems = local_checks(result, generated, run_tests=not args.no_tests)
    else:
        problems = instance_checks(result, generated)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if problems:
        raise SystemExit(1)
    if args.where == "local":
        print(f"0a ok: {len(result['runs'])} runs, datasets {list(result['datasets'])}; recorded {preflight_path(generated)}")
        print(f"next: sync_to_remote.sh <remote> {' '.join(result['datasets'])}")
    else:
        print(f"0b ok: {len(result['runs'])} runs ready")


if __name__ == "__main__":
    main()
