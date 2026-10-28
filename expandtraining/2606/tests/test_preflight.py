from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import preflight
from preflight import code_fingerprint, instance_checks

# scripts/ shares module names with src/ (preflight, train), so load the runner by path.
_spec = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "run_queue.py")
run_queue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_queue)

GPUS_1 = {"count": 1, "names": ["RTX 5090"], "arch": "sm_120", "arch_supported": True, "bf16": True}


def result(world_size: int = 1) -> dict:
    return {"name": "t", "world_size": world_size,
            "runs": [{"config": "expandtraining/2606/configs/remote_5090x2.yaml"}], "datasets": {}}


def record(tmp_path: Path, **overrides) -> Path:
    data = {"code": code_fingerprint(), "datasets": {}, **overrides}
    (tmp_path / "preflight.json").write_text(json.dumps(data))
    return tmp_path


def test_fingerprint_ignores_outputs_and_caches_but_not_code() -> None:
    assert preflight._skipped(Path("outputs/runs/x/config.yaml"))
    assert preflight._skipped(Path("src/__pycache__/train.cpython-313.pyc"))
    assert preflight._skipped(Path("configs/ablations/generated/t/preflight.json"))
    assert preflight._skipped(Path("src/._train.py"))
    assert not preflight._skipped(Path("configs/ablations/generated/t/staged__s42.yaml"))


def test_instance_checks_pass_on_a_matching_box(tmp_path: Path) -> None:
    assert instance_checks(result(), record(tmp_path), gpus=GPUS_1) == []


@pytest.mark.parametrize(
    ("world", "gpus", "rec", "message"),
    [
        (2, GPUS_1, {}, "world_size 2 but 1 GPU"),
        (1, {**GPUS_1, "arch_supported": False}, {}, "kernels or bf16"),
        (1, GPUS_1, {"code": "stale"}, "code here differs"),
        (1, GPUS_1, {"datasets": {"d-nope": "k"}}, "d-nope missing"),
    ],
)
def test_instance_checks_refuse(tmp_path: Path, world, gpus, rec, message) -> None:
    problems = instance_checks(result(world), record(tmp_path, **rec), gpus=gpus)
    assert any(message in p for p in problems), problems


def test_instance_checks_need_the_local_record(tmp_path: Path) -> None:
    assert "preflight.py on the Mac" in instance_checks(result(), tmp_path, gpus=GPUS_1)[0]


def test_queue_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_queue, "run_dir", lambda name: tmp_path / name)
    run = tmp_path / "r"
    assert run_queue.state_of("r") == "fresh"
    (run / "checkpoints" / "latest").mkdir(parents=True)
    (run / "checkpoints" / "latest" / "training_state.pt").write_bytes(b"x")
    assert run_queue.state_of("r") == "resume"
    (run / run_queue.TRAIN_DONE).write_text("t")
    assert run_queue.state_of("r") == "validate"
    (run / "validation").mkdir()
    for name in run_queue.VALIDATION_DONE:
        (run / name).write_text("x")
    assert run_queue.state_of("r") == "done"
    latest = run / "checkpoints" / "latest"
    run_queue.prune("r")
    assert (latest / "training_state.pt").exists(), "no weights.pt: pruning would lose the projection head"
    (latest / "weights.pt").write_bytes(b"w")
    run_queue.prune("r")
    assert not (latest / "training_state.pt").exists()
    assert (latest / "weights.pt").exists()


def test_step_kills_the_whole_process_group_past_the_deadline(tmp_path: Path) -> None:
    import time

    marker = tmp_path / "child_still_running"
    # The grandchild (sleep, then write) stands in for accelerate -> python under remote_train.sh's bash.
    command = ["bash", "-c", f"(sleep 3; touch {marker}) & wait"]
    started = time.time()
    code = run_queue.step(command, {}, tmp_path / "log.txt", deadline=time.time() + 1)
    assert code == run_queue.TIMED_OUT and time.time() - started < 3
    time.sleep(3)
    assert not marker.exists(), "the grandchild survived the kill"
    assert "killed: run budget spent" in (tmp_path / "log.txt").read_text()
    assert run_queue.step(["true"], {}, tmp_path / "log.txt", deadline=time.time() + 30) == 0
    assert run_queue.cost(60, 2.88) == "60.0 min, CNY 2.88"


def test_local_checks_refuse_runs_without_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "load_notes", lambda: {"spec__a__s42": "tests a"})
    result = {"name": "spec", "world_size": 1, "datasets": {},
              "runs": [{"run": "spec__a__s42"}, {"run": "spec__b__s42"}]}
    problems = preflight.local_checks(result, tmp_path, run_tests=False)
    assert len(problems) == 1 and "spec__b__s42" in problems[0] and "spec__a__s42" not in problems[0]


def test_every_note_is_one_line_of_text() -> None:
    from tracking import load_notes

    notes = load_notes()
    assert notes and all(isinstance(note, str) and note and "\n" not in note for note in notes.values())
