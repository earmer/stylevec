from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from train import accumulation_window


def test_accumulation_window_full_windows() -> None:
    assert accumulation_window(0, 10, 4) == 4
    assert accumulation_window(1, 10, 4) == 4
    assert accumulation_window(3, 10, 4) == 4


def test_accumulation_window_partial_tail() -> None:
    assert accumulation_window(8, 10, 4) == 2
    assert accumulation_window(9, 10, 4) == 2


def test_accumulation_window_smaller_than_epoch() -> None:
    assert accumulation_window(0, 2, 8) == 2
    assert accumulation_window(1, 2, 8) == 2


def test_accumulation_window_disabled() -> None:
    assert accumulation_window(0, 10, 1) == 1
    assert accumulation_window(7, 10, 0) == 1
