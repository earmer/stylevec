"""Run every (arm, seed) of a planned spec on this instance, resumably. Start it inside tmux:

    tmux new -s queue 'bash expandtraining/2606/scripts/run_queue.sh <spec> [--prune] [--keep-up]'

The box powers off when the queue ends (billing stops); pass --keep-up to leave it running.

Stage 0b runs once, 0c before each run. After the final weights, each train.keep_weights_at snapshot
(checkpoints/step_NNNNN) is validated into validation_step_NNNNN/ (metrics only, a few minutes each; also for runs already
done, so re-running a finished spec fills them in), so
compare can show the trajectory; a failed snapshot validation is logged and does not fail the run. A run is skipped when its validation is complete, resumed from
checkpoints/latest when training was cut off, otherwise started fresh. Per-run logs are written to
outputs/runs/<run>/queue_{train,validate}.log and the queue's own status to outputs/queue/<spec>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ablation import SpecError, generated_dir, plan
from config import repo_root, workspace_dir
from preflight import instance_checks, run_checks

SCRIPTS = workspace_dir() / "scripts"
TRAIN_DONE = ".queue_train_done"
VALIDATION_DONE = ("validation/per_rule.csv", "validation/retrieval_probe.json")


def run_dir(name: str) -> Path:
    return workspace_dir() / "outputs" / "runs" / name


def state_of(name: str) -> str:
    directory = run_dir(name)
    if all((directory / f).is_file() for f in VALIDATION_DONE):
        return "done"
    if (directory / TRAIN_DONE).is_file():
        return "validate"
    if (directory / "checkpoints" / "latest" / "training_state.pt").is_file():
        return "resume"
    return "fresh"


def step(command: list[str], env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%F %T')} {' '.join(command)}\n")
        handle.flush()
        # No stdin: a tool that would prompt (swanlab without a login) fails fast instead of hanging.
        return subprocess.run(command, cwd=repo_root(), env=env, stdin=subprocess.DEVNULL,
                              stdout=handle, stderr=subprocess.STDOUT).returncode


def pending_snapshots(name: str) -> list[Path]:
    """Kept step_NNNNN checkpoints whose validation_step_NNNNN/ is not complete yet."""
    checkpoints = run_dir(name) / "checkpoints"
    return [step for step in sorted(checkpoints.glob("step_*"))
            if (step / "weights.pt").is_file()
            and not all((run_dir(name) / f"validation_{step.name}" / Path(f).name).is_file() for f in VALIDATION_DONE)]


def prune(name: str) -> None:
    """Drop optimizer/scheduler state once validated. Only when weights.pt (LoRA plus projection head)
    sits beside it: adapter/ alone cannot rebuild the model, so without it nothing is deleted."""
    latest = run_dir(name) / "checkpoints" / "latest"
    state = latest / "training_state.pt"
    if state.is_file() and (latest / "weights.pt").is_file():
        state.unlink()
    elif state.is_file():
        print(f"prune skipped for {name}: no weights.pt beside training_state.pt", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    parser.add_argument("--only", nargs="*", default=None, help="run names (or arm names) to restrict to")
    parser.add_argument("--prune", action="store_true", help="delete training_state.pt after validation")
    parser.add_argument("--keep-up", action="store_true",
                        help="do not power off when the queue ends (by default it does, to stop billing)")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, then exit")
    args = parser.parse_args()
    try:
        result = plan(args.spec, write=False)
    except SpecError as error:
        raise SystemExit(f"spec: {error}") from None
    runs = [r for r in result["runs"] if not args.only or r["run"] in args.only or r["arm"] in args.only]
    status_path = workspace_dir() / "outputs" / "queue" / f"{result['name']}.json"
    status: dict[str, dict] = {r["run"]: {"state": state_of(r["run"])} for r in runs}

    def save() -> None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"{result['name']}: {len(runs)} runs; " +
          ", ".join(f"{s} {sum(v['state'] == s for v in status.values())}" for s in ("done", "validate", "resume", "fresh")),
          flush=True)
    problems = instance_checks(result, generated_dir(result["name"]))
    if problems:
        for problem in problems:
            print(f"0b FAIL {problem}", file=sys.stderr)
        raise SystemExit(1)
    if args.dry_run:
        for r in runs:
            print(f"{status[r['run']]['state']:>8}  {r['run']}")
        return

    env = {**os.environ, "NUM_PROCESSES": str(result["world_size"])}
    env["PATH"] = os.pathsep.join(p for p in ("/root/miniconda3/bin", env.get("PATH", "")) if p)
    failures = 0
    interrupted = False

    def validate_snapshots(name: str, r: dict, entry: dict) -> None:
        """Validate kept step_NNNNN snapshots not validated yet (also for runs finished before this
        existed). A failure is recorded as a warning and does not fail the run."""
        for snapshot in pending_snapshots(name):
            print(f"{time.strftime('%T')} validate {snapshot.name}: {name}", flush=True)
            code = step(["bash", str(SCRIPTS / "validate.sh"), name, str(snapshot)],
                        {**env, "CONFIG": r["eval_config"], "DATASET": r["eval_dataset"],
                         "OUT_NAME": f"validation_{snapshot.name}"},
                        run_dir(name) / "queue_validate.log")
            if code != 0:
                entry.setdefault("warnings", []).append(f"{snapshot.name} validate exit {code}")
                print(f"snapshot validation failed for {name} {snapshot.name}; see queue_validate.log", flush=True)
    try:
        for r in runs:
            name, entry = r["run"], status[r["run"]]
            entry["state"] = state_of(name)
            if entry["state"] == "done":
                validate_snapshots(name, r, entry)
                save()
                continue
            problems = run_checks(r)
            if problems:
                entry.update(state="failed", error="0c: " + "; ".join(problems))
                failures += 1
                save()
                continue
            free_gb = shutil.disk_usage(workspace_dir()).free / 1e9
            if free_gb < 2.0:
                entry.update(state="failed", error=f"disk: {free_gb:.1f} GB free")
                save()
                print("stopping: disk nearly full", file=sys.stderr, flush=True)
                break
            started = time.time()
            if entry["state"] in ("fresh", "resume"):
                command = ["bash", str(SCRIPTS / "remote_train.sh"), name]
                if entry["state"] == "resume":
                    command += ["--resume", str(run_dir(name) / "checkpoints" / "latest")]
                print(f"{time.strftime('%T')} train {entry['state']}: {name}", flush=True)
                entry["state"] = "training"
                save()
                code = step(command, {**env, "CONFIG": r["config"], "DATASET": r["dataset"]},
                            run_dir(name) / "queue_train.log")
                if code != 0:
                    entry.update(state="failed", error=f"train exit {code}; see queue_train.log")
                    failures += 1
                    save()
                    continue
                (run_dir(name) / TRAIN_DONE).write_text(time.strftime("%F %T"), encoding="utf-8")
            print(f"{time.strftime('%T')} validate: {name}", flush=True)
            entry["state"] = "validating"
            save()
            code = step(["bash", str(SCRIPTS / "validate.sh"), name],
                        {**env, "CONFIG": r["eval_config"], "DATASET": r["eval_dataset"]},
                        run_dir(name) / "queue_validate.log")
            if code != 0 or state_of(name) != "done":
                entry.update(state="failed", error=f"validate exit {code}; see queue_validate.log")
                failures += 1
                save()
                continue
            validate_snapshots(name, r, entry)
            if args.prune:
                prune(name)
            entry.update(state="done", minutes=round((time.time() - started) / 60, 1))
            save()
    except KeyboardInterrupt:
        interrupted = True  # a hand on the keyboard keeps the box up
    finally:
        save()
        done = sum(v["state"] == "done" for v in status.values())
        print(f"{time.strftime('%T')} queue ended: {done}/{len(runs)} done, {failures} failed; {status_path}", flush=True)
        if not args.keep_up and not args.dry_run and not interrupted:
            print("shutting down", flush=True)
            # AutoDL's /usr/bin/shutdown is a shell script without a #! line: exec'ing it directly fails
            # with ENOEXEC (Exec format error) and the box stays up, billing. Run it through bash.
            code = subprocess.run(["bash", "/usr/bin/shutdown"], check=False).returncode
            if code != 0:
                print(f"shutdown exited {code}; the box may still be running (billing)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
