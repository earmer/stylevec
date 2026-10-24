"""Blend two checkpoints of one run into a new one: (1 - alpha) * early + alpha * late, tensor by tensor.

    uv run python expandtraining/2606/scripts/blend_weights.py <run> <early> <late> <alpha> [<alpha> ...]

<early>/<late> are checkpoint names under outputs/runs/<run>/checkpoints/ (e.g. step_01050, latest). Writes
checkpoints/blend_<early>_<late>_aNN/weights.pt per alpha, which validate.sh reads like any snapshot.

LoRA A and B are blended separately, not their product; checkpoints from one trajectory share a basin, so
the cross terms stay small. Integer buffers (counters) are taken from the late checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import save_weights

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "runs"


def blend(early: dict[str, torch.Tensor], late: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    if early.keys() != late.keys():
        raise SystemExit(f"checkpoints hold different tensors: {sorted(early.keys() ^ late.keys())[:5]}")
    return {
        name: torch.lerp(early[name].float(), late[name].float(), alpha).to(late[name].dtype)
        if late[name].is_floating_point() else late[name]
        for name in late
    }


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    run, early_name, late_name, *alphas = sys.argv[1:]
    checkpoints = RUNS / run / "checkpoints"
    early = torch.load(checkpoints / early_name / "weights.pt", map_location="cpu")
    late = torch.load(checkpoints / late_name / "weights.pt", map_location="cpu")
    if early["manifest_hash"] != late["manifest_hash"]:
        raise SystemExit("checkpoints were trained on different data")
    for alpha in map(float, alphas):
        out = checkpoints / f"blend_{early_name}_{late_name}_a{round(alpha * 100):02d}"
        save_weights(blend(early["model_trainable"], late["model_trainable"], alpha), out, late["global_step"],
                     late["manifest_hash"])
        print(out)


if __name__ == "__main__":
    main()
