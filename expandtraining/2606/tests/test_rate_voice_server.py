import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

# Loaded by path: putting scripts/ on sys.path shadows src/validate.py for the tests that run after this one.
spec = importlib.util.spec_from_file_location(
    "rate_voice_server", Path(__file__).resolve().parents[1] / "scripts" / "rate_voice_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_order_keeps_the_edit_size_mix_in_any_prefix():
    rows = [{"bin": b, "batch": x, "id": f"{b}{x}{i}"}
            for b in server.BIN_WEIGHTS for x in ("old", "new") for i in range(100)]
    ordered = server.stratified_order(rows)
    assert ordered == server.stratified_order(rows)
    counts = Counter(row["bin"] for row in ordered[:55])
    assert counts["small"] == pytest.approx(20, abs=2) and counts["rewrite"] == pytest.approx(10, abs=2)
    assert Counter(row["batch"] for row in ordered[:55])["new"] == pytest.approx(27, abs=2)


def test_latest_rating_wins():
    assert server.latest_ratings([{"id": 1, "voice": 0}, {"id": 1, "voice": 3}])[1]["voice"] == 3


@pytest.mark.parametrize("body", [{"id": 9}, {"id": 1, "voice": 4}, {"id": 1, "voice": True}, {"id": 1, "meaning": "x"},
                                  {"id": 1, "note": "n" * 501}])
def test_bad_ratings_are_refused(body):
    with pytest.raises(ValueError):
        server.clean_rating(body, {1})


def test_llm_answers_are_labelled_by_file_and_run(tmp_path):
    runs, careful = tmp_path / "runs.jsonl", tmp_path / "careful.jsonl"
    runs.write_text(json.dumps({"run": 2, "id": 1, "answer": "0y"}) + "\n", encoding="utf-8")
    careful.write_text(json.dumps({"run": 1, "id": 1, "answer": "2p"}) + "\n", encoding="utf-8")
    assert server.llm_answers({"run": runs, "careful": careful})[1] == [
        {"label": "run 2", "answer": "0y"}, {"label": "careful", "answer": "2p"}]


def test_rating_keeps_only_known_fields():
    row = server.clean_rating({"id": 1, "voice": 0, "meaning": "y", "views": {"talk": True}, "extra": 1}, {1})
    assert row == {"id": 1, "voice": 0, "meaning": "y", "broken": False, "note": "",
                   "views": {"talk": True, "diff": False, "translations": False, "llm": False}}


def test_page_data_never_carries_scores_or_strata(tmp_path):
    full = server.RATINGS_DIR / "voice_change_full_key.jsonl"
    if not (full.exists() and server.DB.exists()):
        pytest.skip("local data only")
    store = server.Store(tmp_path / "log.jsonl", 50)
    payload = json.dumps(store.set_payload(0), ensure_ascii=False)
    for field in ('"batch"', '"changed"', '"similarity"', '"bin"', '"talk_id"'):
        assert field not in payload
    assert sum(s["size"] for s in store.summary()[:1]) == 100


def test_fresh_pick_fills_each_bin_under_the_speaker_cap():
    pool = [{"id": f"{b}{n}", "bin": b, "speaker": f"s{n % 6}"} for b in ("small", "mid") for n in range(30)]
    picked = server.pick_fresh(pool, {"small": 6, "mid": 3}, 2)
    assert Counter(row["bin"] for row in picked) == {"small": 6, "mid": 3}
    assert max(Counter(row["speaker"] for row in picked).values()) <= 2
    with pytest.raises(AssertionError):
        server.pick_fresh(pool, {"small": 40}, 2)


def test_retests_skip_broken_and_unrated_pairs_and_keep_the_counts():
    full = [{"id": n, "changed": 0.1 if n < 10 else 0.2} for n in range(20)]
    ratings = {n: {"voice": n % 4, "broken": False} for n in range(20)}
    ratings[0] = {"voice": None, "broken": True}
    ratings[10] = {"voice": 1, "broken": True}
    del ratings[1]
    picked = server.pick_retests(full, ratings, {"small": 8, "other": 9})
    ids = {row["id"] for row in picked}
    assert len(ids) == 17 and not ids & {0, 1, 10}
