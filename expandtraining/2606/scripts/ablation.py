from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import yaml

from ablation import SpecError, compare, generated_dir, load_per_rule, measured_tiers, output_dir, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, compare, and derive tiers for 2606 ablations.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="check a spec and write one config per (arm, seed)")
    plan_parser.add_argument("spec")
    plan_parser.add_argument("--manifest", help="data_manifest.json or dataset manifest.json for the exposure audit")
    compare_parser = commands.add_parser("compare", help="aggregate validation outputs of a planned spec")
    compare_parser.add_argument("spec")
    compare_parser.add_argument("--runs-root", type=Path, default=None)
    tiers_parser = commands.add_parser("tiers", help="curriculum.tiers measured from a prior validation")
    tiers_parser.add_argument("validation", help="per_rule.csv or a run's validation/ directory")
    tiers_parser.add_argument("--easy-below", type=float, default=0.15)
    tiers_parser.add_argument("--hard-above", type=float, default=0.35)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan(args.spec, args.manifest)
            print(f"{len(result['runs'])} run configs: {generated_dir(result['name'])}")
            print(f"plan: {output_dir(result['name']) / 'plan.md'}")
        elif args.command == "compare":
            result = compare(args.spec, args.runs_root)
            print(f"missing runs: {len(result['missing_runs'])}; report: {output_dir(result['name']) / 'compare.md'}")
        else:
            tiers = measured_tiers(load_per_rule(args.validation), args.easy_below, args.hard_above)
            print(yaml.safe_dump({"curriculum.tiers": tiers}, sort_keys=True, allow_unicode=True), end="")
    except SpecError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
