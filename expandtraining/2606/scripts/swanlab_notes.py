"""Push run_notes.yaml to SwanLab as each experiment's description.

    uv run python expandtraining/2606/scripts/swanlab_notes.py [--dry-run] [--abort RUN ...]

SWANLAB_API_KEY must be set (from TOK= in swanlab.env, via env only). Runs without a SwanLab experiment are
listed and skipped; experiments without a note are listed so they can be added. --abort marks a run
killed by hand (still RUNNING on SwanLab) as ABORTED.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking import load_notes

PROJECT = "earmercarey/stylevec-2606"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--abort", nargs="*", default=[], help="run names to mark ABORTED")
    args = parser.parse_args()
    from swanlab import Api

    api = Api(api_key=os.environ["SWANLAB_API_KEY"])
    client = api._ctx.client  # no public setter for descriptions or state
    # A rerun of a crashed run shares its name; the note belongs to the newest.
    experiments = {e.name: e for e in sorted(api.runs(PROJECT), key=lambda e: e.created_at_ts)}
    notes = load_notes()
    for name, note in notes.items():
        experiment = experiments.get(name)
        if experiment is None:
            print(f"no experiment yet: {name}")
            continue
        if experiment.description == note:
            continue
        print(f"note: {name}")
        if not args.dry_run:
            client.put(f"/project/{PROJECT}/runs/{experiment.run_id}", {"description": note})
    for name in sorted(set(experiments) - set(notes)):
        print(f"no note for experiment: {name}")
    for name in args.abort:
        experiment = experiments[name]
        print(f"abort: {name} ({experiment.state})")
        if not args.dry_run:
            finished = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            client.put(f"/project/{PROJECT}/runs/{experiment.run_id}/state",
                       {"state": "ABORTED", "finishedAt": finished, "from": "sdk"})


if __name__ == "__main__":
    main()
