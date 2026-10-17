"""Stage 0: checks that fail in seconds instead of hours into a queue.

0a `local_checks` runs on the Mac before sync: the spec plans with real paths, every dataset it needs
is built with a matching data key, the tests pass, and the code fingerprint is recorded next to the
generated configs (which sync carries to the instance).
0b `instance_checks` runs once on the instance before the queue: same code as 0a recorded, GPUs and
bf16 match the spec's world size, environment, base model, datasets and disk headroom.
0c `run_checks` runs before each run: its config and datasets, and no resume escape hatch.
Each check returns a list of problems; empty means go.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from config import load_config, repo_root, resolve_repo_path, workspace_dir
from prebuilt import data_key, dataset_dir, read_manifest

# Mirrors the code rsync in scripts/sync_to_remote.sh: what it skips is not part of the fingerprint.
SKIP_DIRS = {"outputs", "tf-logs", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache", ".rsync-partial"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".pyd")
PREFLIGHT_FILE = "preflight.json"
MIN_FREE_GB = 5.0
MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")


def code_fingerprint() -> str:
    """sha256 over the synced workspace plus the root uv project, independent of mtimes."""
    root = workspace_dir()
    digest = hashlib.sha256()
    files = [p for p in root.rglob("*") if p.is_file() and not _skipped(p.relative_to(root))]
    files += [repo_root() / "pyproject.toml", repo_root() / "uv.lock"]
    for path in sorted(files, key=lambda p: str(p.relative_to(repo_root()))):
        digest.update(str(path.relative_to(repo_root())).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:24]


def _skipped(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts) or rel.name.startswith("._") or rel.name == ".DS_Store":
        return True
    return rel.name.endswith(SKIP_SUFFIXES) or rel.name == PREFLIGHT_FILE


def preflight_path(generated: Path) -> Path:
    return generated / PREFLIGHT_FILE


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True).stdout.strip()

    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(workspace_dir())))}


def dataset_problems(name: str, config: dict[str, Any]) -> list[str]:
    if not (dataset_dir(name) / "manifest.json").is_file():
        return [f"dataset {name} is not built (or incomplete): {dataset_dir(name)}"]
    if read_manifest(name).get("data_key") != data_key(config):
        return [f"dataset {name} was built from other data settings or builder code"]
    return []


def local_checks(result: dict[str, Any], generated: Path, run_tests: bool = True) -> list[str]:
    """0a. `result` is ablation.plan() output written with check_paths=True."""
    problems = []
    for name, entry in result["datasets"].items():
        if not entry["built"]:
            problems.append(f"dataset {name} not built; run: uv run python expandtraining/2606/scripts/"
                            f"build_dataset.py --config {entry['config']}")
    if run_tests:
        tests = subprocess.run(["uv", "run", "pytest", "-q", str(workspace_dir() / "tests")],
                               cwd=repo_root(), capture_output=True, text=True)
        if tests.returncode != 0:
            tail = "\n".join((tests.stdout + tests.stderr).strip().splitlines()[-15:])
            problems.append(f"tests failed:\n{tail}")
    if not problems:
        record = {
            "spec": result["name"],
            "code": code_fingerprint(),
            "git": git_state(),
            "world_size": result["world_size"],
            "datasets": {name: entry["key"] for name, entry in result["datasets"].items()},
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        preflight_path(generated).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return problems


def gpu_state() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"count": 0}
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    major, minor = torch.cuda.get_device_capability(0)
    return {"count": len(names), "names": names, "arch": f"sm_{major}{minor}",
            "arch_supported": f"sm_{major}{minor}" in torch.cuda.get_arch_list(),
            "bf16": bool(torch.cuda.is_bf16_supported())}


def instance_checks(result: dict[str, Any], generated: Path, gpus: dict[str, Any] | None = None) -> list[str]:
    """0b. Run on the instance with the environment remote_setup.sh built."""
    problems = []
    record_path = preflight_path(generated)
    if not record_path.is_file():
        return [f"no {PREFLIGHT_FILE} in {generated}: run scripts/preflight.py on the Mac, then sync"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["code"] != code_fingerprint():
        problems.append("code here differs from the code preflight checked; re-run preflight and sync")
    gpus = gpu_state() if gpus is None else gpus
    if gpus["count"] != result["world_size"]:
        problems.append(f"spec world_size {result['world_size']} but {gpus['count']} GPU(s) visible; "
                        f"row budgets would differ from the plan")
    elif not gpus.get("arch_supported", True) or not gpus.get("bf16", True):
        problems.append(f"GPU {gpus.get('names')} ({gpus.get('arch')}): torch kernels or bf16 unavailable")
    base = load_config(result["runs"][0]["config"])
    model_dir = resolve_repo_path(str(base["paths"]["base_model"]))
    problems += [f"base model file missing: {model_dir / f}" for f in MODEL_FILES if not (model_dir / f).is_file()]
    for name, key in record["datasets"].items():
        if not (dataset_dir(name) / "manifest.json").is_file():
            problems.append(f"dataset {name} missing here; sync it: sync_to_remote.sh <remote> {name}")
        elif read_manifest(name).get("data_key") != key:
            problems.append(f"dataset {name} here differs from the one preflight checked")
    free_gb = shutil.disk_usage(workspace_dir()).free / 1e9
    if free_gb < MIN_FREE_GB:
        problems.append(f"only {free_gb:.1f} GB free under {workspace_dir()}; need {MIN_FREE_GB}")
    return problems


def run_checks(run: dict[str, Any]) -> list[str]:
    """0c. Config files exist, datasets match them, resume cannot silently mix data."""
    problems = []
    for config_key, dataset_key in (("config", "dataset"), ("eval_config", "eval_dataset")):
        path = repo_root() / run[config_key]
        if not path.is_file():
            problems.append(f"{run[config_key]} missing")
            continue
        config = load_config(path)
        problems += dataset_problems(run[dataset_key], config)
        if bool(config["train"].get("allow_manifest_mismatch_on_resume", False)):
            problems.append(f"{run[config_key]} sets train.allow_manifest_mismatch_on_resume")
    return problems
