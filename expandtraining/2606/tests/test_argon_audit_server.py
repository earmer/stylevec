import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("argon_audit_server", Path(__file__).resolve().parents[1] / "scripts" / "argon_audit_server.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_rounds_oppose_the_two_scorers_at_similar_edit_size_and_never_reuse_a_pair():
    scores = pd.DataFrame({"id": range(8), audit.ARM: [8, 7, 6, 5, 4, 3, 2, 1], "scan": [1, 2, 3, 4, 5, 6, 7, 8],
                           "changed": [.10, .05, .08, .08, .08, .08, .05, .10]})
    rounds = audit.match_rounds(scores, rounds=2, shortlist=2)
    assert rounds == [(0, 7), (1, 6)]


def test_tally_follows_the_hidden_side():
    rows = [{"round": 1, "model_left": True}, {"round": 2, "model_left": False}, {"round": 3, "model_left": True}, {"round": 4, "model_left": True}]
    result = audit.tally(rows, {1: "left", 2: "left", 3: "neither"})
    assert (result["sided_with_model"], result["sided_with_argon"], result["neither"]) == (1, 1, 1)


def test_segments_mark_changes_and_deletions():
    assert audit.segments("abc", "axc") == [["a", False], ["x", True], ["c", False]]
    assert audit.segments("abc", "ac") == [["a", False], ["∅", True], ["c", False]]
