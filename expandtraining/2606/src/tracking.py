"""SwanLab mirror of a run: TensorBoard scalars live, validation headline metrics at the end.

Configured by `logging.swanlab` ({enabled, project, mode}); SWANLAB_MODE overrides the mode. The API key
comes from `swanlab login` on the box or SWANLAB_API_KEY, never from a config. Tracking can never stop a
run: if online init fails (no login, no network) it falls back to offline logs under the run directory,
which `swanlab sync <dir>` uploads later.

One SwanLab experiment per run name, with a stable id, so a resumed run and its later validation land in
the same experiment. Ablation runs (`<spec>__<arm>__s<seed>`) are grouped by spec and tagged arm/seed.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

from records import stable_id

RUN_RE = re.compile(r"^(?P<spec>.+)__(?P<arm>.+)__s(?P<seed>\d+)$")


def settings(config: dict[str, Any]) -> dict[str, Any] | None:
    section = config.get("logging", {}).get("swanlab", {}) or {}
    mode = os.environ.get("SWANLAB_MODE") or section.get("mode", "online")
    if not section.get("enabled", False) or mode == "disabled" or config["runtime"].get("mode") == "dryrun":
        return None
    return {"project": section.get("project", "stylevec-2606"), "mode": mode}


def start(config: dict[str, Any], run_name: str, run_dir: Path, job_type: str) -> Any | None:
    """Begin or resume this run's experiment; returns the swanlab module, or None when tracking is off."""
    chosen = settings(config)
    if chosen is None:
        return None
    try:
        import swanlab
    except ImportError:
        warnings.warn("logging.swanlab.enabled but swanlab is not installed; continuing without it")
        return None
    match = RUN_RE.match(run_name)
    kwargs = {
        "project": chosen["project"],
        "name": run_name,
        "id": stable_id("swanlab", run_name, size=21),
        "resume": "allow",
        "job_type": job_type,
        "log_dir": str(run_dir / "swanlog"),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    if match:
        kwargs["group"] = match["spec"]
        kwargs["tags"] = [match["spec"], f"arm:{match['arm']}", f"seed:{match['seed']}"]
    for mode in dict.fromkeys((chosen["mode"], "offline")):
        try:
            swanlab.init(mode=mode, reinit=True, **kwargs)
            if mode != chosen["mode"]:
                print(f"swanlab: {chosen['mode']} init failed; logging offline to {run_dir / 'swanlog'}", flush=True)
            return swanlab
        except Exception as error:  # noqa: BLE001 - tracking must never stop training
            warnings.warn(f"swanlab init ({mode}) failed: {error}")
    return None


def mirror_tensorboard(tracker: Any | None) -> None:
    """Call before creating the SummaryWriter: every add_scalar is copied to SwanLab."""
    if tracker is not None:
        try:
            tracker.sync_tensorboard_torch()
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"swanlab tensorboard mirror failed: {error}")


def log(tracker: Any | None, metrics: dict[str, float], step: int | None = None) -> None:
    if tracker is not None and metrics:
        try:
            tracker.log({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"swanlab log failed: {error}")


def finish(tracker: Any | None, ok: bool = True) -> None:
    if tracker is not None:
        try:
            tracker.finish(state="success" if ok else "crashed")
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"swanlab finish failed: {error}")
