from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tracking


def config(**swanlab) -> dict:
    return {"runtime": {"mode": "train"}, "logging": {"swanlab": swanlab}, "_config_path": "x"}


class FakeSwanlab(types.ModuleType):
    def __init__(self, fail_modes=()):
        super().__init__("swanlab")
        self.fail_modes, self.inits, self.logged = set(fail_modes), [], []

    def init(self, mode, **kwargs):
        if mode in self.fail_modes:
            raise RuntimeError("not logged in")
        self.inits.append({"mode": mode, **kwargs})

    def log(self, data, step=None):
        self.logged.append(data)

    def finish(self, state="success"):
        pass


def test_off_unless_enabled_and_never_in_dryrun(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    assert tracking.settings(config()) is None
    assert tracking.settings({**config(enabled=True), "runtime": {"mode": "dryrun"}}) is None
    monkeypatch.setenv("SWANLAB_MODE", "disabled")
    assert tracking.settings(config(enabled=True)) is None


def test_ablation_runs_are_grouped_and_ids_are_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    for _ in range(2):
        assert tracking.start(config(enabled=True), "curriculum_a__staged__s42", tmp_path, "train") is fake
    first, second = fake.inits
    assert first["id"] == second["id"] and first["resume"] == "allow"
    assert first["group"] == "curriculum_a" and "arm:staged" in first["tags"]
    assert "_config_path" not in first["config"]


def test_fresh_run_dir_under_a_reused_name_gets_a_new_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    name = "infonce_b__hinge__s42"
    tracking.start(config(enabled=True), name, tmp_path / "old", "train")
    (tmp_path / "old").rename(tmp_path / "old.stopped")
    tracking.start(config(enabled=True), name, tmp_path / "old", "train")
    tracking.start(config(enabled=True), name, tmp_path / "old", "validate")
    first, fresh, validate = (init["id"] for init in fake.inits)
    assert fresh != first and validate == fresh


def test_run_dir_from_before_the_id_file_keeps_its_stable_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    (tmp_path / "swanlog").mkdir()
    tracking.start(config(enabled=True), "legacy__arm__s42", tmp_path, "validate")
    assert fake.inits[0]["id"] == tracking.stable_id("swanlab", "legacy__arm__s42", size=21)


def test_online_failure_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeSwanlab(fail_modes={"online"})
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    with pytest.warns(UserWarning, match="online"):
        assert tracking.start(config(enabled=True), "plain_run", tmp_path, "train") is fake
    assert fake.inits[0]["mode"] == "offline" and "group" not in fake.inits[0]
    tracking.log(fake, {"final/heldout_top1": 0.3})
    assert fake.logged == [{"final/heldout_top1": 0.3}]


def test_total_failure_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "swanlab", FakeSwanlab(fail_modes={"online", "offline"}))
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    with pytest.warns(UserWarning):
        assert tracking.start(config(enabled=True), "r", tmp_path, "train") is None
