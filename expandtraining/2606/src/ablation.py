"""Ablation specs: one base config, named arms of dotted-key overrides, explicit comparisons.

`plan` refuses a spec before GPU time is spent if a comparison differs in more than the keys it declares
(exactly one when it declares none), arms spend different row budgets, a wall-clock cap could truncate a
schedule, or arms would be scored on different evaluation sets. `compare` reads each run's validation
outputs and pairs arms by seed. CURRICULUM.md holds the catalogue of specs and how to read them.
"""
from __future__ import annotations

import copy
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config import repo_root, validate_config, workspace_dir
from curriculum import DEFAULT_STAGES, expected_exposure
from prebuilt import data_key, dataset_dir, default_name, read_manifest
from report import per_rule_table

NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Changing these moves identities between training and the shared held-out evaluation.
SPLIT_KEYS = ("data.speakers", "data.heldout_speakers")
DEFAULT_HEADLINE = ["heldout_top1_lift", "provenance_acc", "gap:same_speaker_topic_matched"]
DEFAULT_FOCUS_RULES = [
    "same_speaker_topic_matched",
    "same_speaker_real",
    "origin_vs_para",
    "generated_to_target",
    "bookraw_same_author_cross_work",
]
LOWER_IS_BETTER = ("provenance_", "violation:", "macro_violation", "memory_gap", "top_component_ratio", "length_r2")
DEFAULT_KILL = {"retrieval_lift_max": 0.05, "provenance_min": 0.85}


class SpecError(ValueError):
    pass


def generated_dir(name: str) -> Path:
    return workspace_dir() / "configs" / "ablations" / "generated" / name


def output_dir(name: str) -> Path:
    return workspace_dir() / "outputs" / "ablations" / name


def run_name(spec_name: str, arm: str, seed: int) -> str:
    return f"{spec_name}__{arm}__s{seed}"


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise SpecError(f"spec must be a mapping: {path}")
    for key in ("name", "base", "arms", "comparisons"):
        if key not in spec:
            raise SpecError(f"spec missing `{key}`: {path}")
    for name in [spec["name"], *spec["arms"]]:
        if not NAME_RE.match(str(name)):
            raise SpecError(f"names must match {NAME_RE.pattern}: {name!r}")
    spec.setdefault("seeds", [42])
    spec.setdefault("world_size", 1)
    spec.setdefault("common", {})
    spec.setdefault("new_keys", [])
    spec.setdefault("interactions", [])
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise SpecError("seeds must be a non-empty list of distinct ints")
    names = set()
    for comparison in spec["comparisons"]:
        for key in ("arm", "against"):
            if comparison.get(key) not in spec["arms"]:
                raise SpecError(f"comparison references unknown arm: {comparison}")
        comparison.setdefault("name", f"{comparison['arm']}-vs-{comparison['against']}")
        names.add(comparison["name"])
    for interaction in spec["interactions"]:
        if len(interaction.get("of", [])) != 2 or not set(interaction["of"]) <= names:
            raise SpecError(f"interaction needs `of: [comparison, comparison]` naming two comparisons: {interaction}")
    return spec


def load_base(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["base"])
    path = path if path.is_absolute() else workspace_dir() / path
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise SpecError(f"base config must be a mapping: {path}")
    return base


def set_dotted(config: dict[str, Any], key: str, value: Any, new_keys: list[str]) -> None:
    """Set `a.b.c`. Unknown keys are typos until proven otherwise, except under `curriculum.`."""
    allowed_new = key.startswith("curriculum.") or key in new_keys
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node:
            if not allowed_new:
                raise SpecError(f"override `{key}`: base has no `{part}`; list it in new_keys if intended")
            node[part] = {}
        node = node[part]
        if not isinstance(node, dict):
            raise SpecError(f"override `{key}`: `{part}` is not a mapping")
    if parts[-1] not in node and not allowed_new:
        raise SpecError(f"override `{key}` is not in the base config (typo?); list it in new_keys if intended")
    node[parts[-1]] = copy.deepcopy(value)


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            leaves.update(flatten(value, path + "."))
        else:
            leaves[path] = value
    return leaves


def comparable(config: dict[str, Any]) -> dict[str, Any]:
    """Flattened config with curriculum defaults filled in and the seed removed."""
    clean = copy.deepcopy(config)
    clean.get("runtime", {}).pop("seed", None)
    curriculum = clean.setdefault("curriculum", {})
    curriculum.setdefault("enabled", False)
    curriculum.setdefault("mode", "staged")
    curriculum.setdefault("pacing", "step")
    if not curriculum.get("stages"):
        curriculum["stages"] = DEFAULT_STAGES
    return flatten(clean)


def materialize(spec: dict[str, Any], base: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    shared = copy.deepcopy(base)
    for key, value in spec["common"].items():
        set_dotted(shared, key, value, spec["new_keys"])
    arms: dict[str, dict[str, Any]] = {}
    for arm, overrides in spec["arms"].items():
        config = copy.deepcopy(shared)
        for key, value in (overrides or {}).items():
            set_dotted(config, key, value, spec["new_keys"])
        arms[arm] = config
    return shared, arms


def row_budget(config: dict[str, Any], world_size: int) -> int:
    train = config["train"]
    return int(train["max_steps"]) * int(train["batch_size"]) * int(train.get("grad_accum", 1)) * int(world_size)


def eval_config(shared: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Score every arm on the shared data/validation set; only the model-shaping keys follow the arm."""
    config = copy.deepcopy(shared)
    config["model"] = copy.deepcopy(arm["model"])
    config.setdefault("regularization", {})["matryoshka_dims"] = copy.deepcopy(
        arm.get("regularization", {}).get("matryoshka_dims", [])
    )
    config["train"]["combine_triplet_forward"] = arm["train"].get("combine_triplet_forward", False)
    return config


def check_spec(spec: dict[str, Any], shared: dict[str, Any], arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[str] = []
    flat = {arm: comparable(config) for arm, config in arms.items()}
    shared_flat = comparable(shared)
    budgets = {arm: (int(c["train"]["max_steps"]), row_budget(c, spec["world_size"])) for arm, c in arms.items()}
    for arm, config in arms.items():
        if not bool(config.get("curriculum", {}).get("enabled", False)):
            errors.append(f"{arm}: curriculum.enabled must be true so every arm shares one budget accounting")
        if float(config["train"].get("max_seconds", 0) or 0) > 0:
            errors.append(f"{arm}: train.max_seconds > 0 can stop a run before its late stages; use max_steps only")
        for key in SPLIT_KEYS:
            if flat[arm].get(key) != shared_flat.get(key):
                errors.append(f"{arm}: changes {key}, which would leak identities into the shared held-out set")
        changed_validation = sorted(k for k in set(flat[arm]) | set(shared_flat)
                                    if k.startswith("validation.") and flat[arm].get(k) != shared_flat.get(k))
        if changed_validation:
            errors.append(f"{arm}: changes {changed_validation}; arms must be measured identically")
    if len(set(budgets.values())) > 1 and not spec.get("allow_budget_change", False):
        errors.append(f"arms spend different budgets (max_steps, rows): {budgets}")
    resolved: list[dict[str, Any]] = []
    for comparison in spec["comparisons"]:
        a, b = flat[comparison["arm"]], flat[comparison["against"]]
        diff = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
        declared = sorted(comparison.get("keys") or [])
        if not diff:
            errors.append(f"{comparison['name']}: arms are identical")
        elif declared and diff != declared:
            errors.append(f"{comparison['name']}: differs in {diff}, declared {declared}")
        elif not declared and len(diff) != 1:
            errors.append(f"{comparison['name']}: differs in {diff}; declare `keys` or split the comparison")
        resolved.append({**comparison, "keys": diff, "values": {k: [a.get(k), b.get(k)] for k in diff}})
    if errors:
        raise SpecError("ablation spec rejected:\n  - " + "\n  - ".join(errors))
    return resolved


def rule_counts_from_manifest(path: str | Path) -> dict[str, int]:
    counts = json.loads(Path(path).read_text(encoding="utf-8"))["counts"]
    out: dict[str, int] = {}
    for key, value in counts.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "triplets" and parts[2] == "train" and parts[1] != "all":
            out[parts[1]] = int(value)
    return out


def _dump(config: dict[str, Any], path: Path) -> None:
    clean = {key: value for key, value in config.items() if not str(key).startswith("_")}
    path.write_text(yaml.safe_dump(clean, allow_unicode=True, sort_keys=False), encoding="utf-8")


def plan(
    spec_path: str | Path,
    manifest: str | Path | None = None,
    check_paths: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    base = load_base(spec)
    shared, arms = materialize(spec, base)
    comparisons = check_spec(spec, shared, arms)
    if check_paths:
        for arm, config in arms.items():
            try:
                validate_config(copy.deepcopy(config))
            except (ValueError, FileNotFoundError) as error:
                raise SpecError(f"{arm}: {error}") from error
    repo = repo_root()
    target = generated_dir(spec["name"])
    shared_eval = eval_config(shared, shared)
    evals = {json.dumps(comparable(shared_eval), sort_keys=True, default=str): "eval.yaml"}
    files: dict[Path, dict[str, Any]] = {target / "eval.yaml": shared_eval}
    runs: list[dict[str, Any]] = []
    datasets: dict[str, dict[str, Any]] = {}
    for arm, config in arms.items():
        evaluation = eval_config(shared, config)
        key = json.dumps(comparable(evaluation), sort_keys=True, default=str)
        if key not in evals:
            evals[key] = f"eval__{arm}.yaml"
            files[target / evals[key]] = evaluation
        train_data, eval_data = default_name(config), default_name(evaluation)
        for name, source, use in ((train_data, config, "train"), (eval_data, evaluation, "eval")):
            entry = datasets.setdefault(name, {"key": data_key(source), "config": None, "used_by": {}})
            entry["used_by"].setdefault(use, []).append(arm)
        for seed in spec["seeds"]:
            seeded = copy.deepcopy(config)
            seeded["runtime"]["seed"] = int(seed)
            config_path = target / f"{arm}__s{seed}.yaml"
            files[config_path] = seeded
            runs.append({
                "run": run_name(spec["name"], arm, int(seed)), "arm": arm, "seed": int(seed),
                "config": str(config_path.relative_to(repo)),
                "eval_config": str((target / evals[key]).relative_to(repo)),
                "dataset": train_data, "eval_dataset": eval_data,
            })
            for name in (train_data, eval_data):
                datasets[name]["config"] = datasets[name]["config"] or runs[-1][
                    "config" if name == train_data else "eval_config"]
    for name, entry in datasets.items():
        built = (dataset_dir(name) / "manifest.json").is_file()
        entry["built"] = built and read_manifest(name).get("data_key") == entry["key"]
    # Exposure from each arm's own prebuilt dataset when every one exists, else from one shared manifest.
    arm_data = {run["arm"]: run["dataset"] for run in runs}
    exposure, source = None, None
    if manifest:
        counts = rule_counts_from_manifest(manifest)
        exposure = {arm: expected_exposure(counts, config, row_budget(config, spec["world_size"]))
                    for arm, config in arms.items()}
        source = str(manifest)
    elif all(datasets[name]["built"] for name in arm_data.values()):
        exposure = {arm: expected_exposure(rule_counts_from_manifest(dataset_dir(arm_data[arm]) / "manifest.json"),
                                           config, row_budget(config, spec["world_size"]))
                    for arm, config in arms.items()}
        source = "each arm's prebuilt dataset"
    shared_flat = comparable(shared)
    data_arms = sorted(arm for arm, config in arms.items()
                       if any(k.startswith(("data.", "paths.")) and v != shared_flat.get(k)
                              for k, v in comparable(config).items()))
    result = {
        "name": spec["name"],
        "description": str(spec.get("description", "")).strip(),
        "base": spec["base"],
        "seeds": spec["seeds"],
        "world_size": spec["world_size"],
        "budget": {arm: {"max_steps": int(c["train"]["max_steps"]), "rows": row_budget(c, spec["world_size"])}
                   for arm, c in arms.items()},
        "comparisons": comparisons,
        "interactions": spec["interactions"],
        "runs": runs,
        "datasets": datasets,
        "exposure": exposure,
        "exposure_source": source,
        "data_changing_arms": data_arms if manifest else [],
    }
    if write:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for path, config in files.items():
            _dump(config, path)
        out = output_dir(spec["name"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "plan.md").write_text(render_plan(result), encoding="utf-8")
    return result


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def render_plan(result: dict[str, Any]) -> str:
    lines = [f"# Ablation plan: {result['name']}", ""]
    if result["description"]:
        lines += [result["description"], ""]
    budget = next(iter(result["budget"].values()))
    lines += [
        f"- base: `{result['base']}`; seeds {result['seeds']}; world size {result['world_size']}",
        f"- budget per run: {budget['max_steps']} steps, {budget['rows']:,} rows",
        f"- runs: {len(result['runs'])}",
        "",
        "## Comparisons",
        "",
        "| comparison | isolates | differing key | arm | against |",
        "|---|---|---|---|---|",
    ]
    for c in result["comparisons"]:
        for key in c["keys"]:
            arm_value, against_value = c["values"][key]
            lines.append(f"| {c['name']} | {c.get('isolates', '')} | `{key}` | {_short(arm_value)} | {_short(against_value)} |")
    for interaction in result["interactions"]:
        lines.append(f"\nInteraction `{interaction['name']}` = ({interaction['of'][0]}) - ({interaction['of'][1]}), paired by seed.")
    datasets = result["datasets"]
    lines += ["", "## Datasets", "", "Build locally, then push with `sync_to_remote.sh <remote> <name>...`:", "",
              "| dataset | built | used by |", "|---|---|---|"]
    for name, entry in datasets.items():
        uses = "; ".join(f"{use}: {', '.join(arms)}" for use, arms in entry["used_by"].items())
        lines.append(f"| `{name}` | {'yes' if entry['built'] else 'no'} | {uses} |")
    missing = [entry["config"] for entry in datasets.values() if not entry["built"]]
    if missing:
        lines += ["", "```bash", f"uv run python expandtraining/2606/scripts/build_dataset.py --config {' '.join(missing)}", "```"]
    lines += ["", "## Runs", "", "Train, then validate on the shared evaluation config:", "", "```bash"]
    for run in result["runs"]:
        lines.append(f"DATASET={run['dataset']} CONFIG={run['config']} "
                     f"bash expandtraining/2606/scripts/remote_train.sh {run['run']}")
        lines.append(f"DATASET={run['eval_dataset']} CONFIG={run['eval_config']} "
                     f"bash expandtraining/2606/scripts/validate.sh {run['run']}")
    lines += ["```", ""]
    if result["exposure"]:
        lines += render_exposure(result)
    else:
        lines += ["## Exposure audit", "", "Not computed: build the datasets above and re-plan, or pass `--manifest`.", ""]
    return "\n".join(lines) + "\n"


def render_exposure(result: dict[str, Any]) -> list[str]:
    exposure = result["exposure"]
    arms = list(exposure)
    lines = ["## Exposure audit", "", f"Expected draws from `{result['exposure_source']}`."]
    if result["data_changing_arms"]:
        lines.append(f"Arms {result['data_changing_arms']} change data; their rows come from the base counts, so rebuild for exact numbers.")
    lines += ["", "Tier share by third of training:", "", "| arm | first | middle | last |", "|---|---|---|---|"]
    for arm in arms:
        cells = [" ".join(f"{t} {v:.2f}" for t, v in phase.items()) for phase in exposure[arm]["tier_share_by_phase"]]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "Repeats per distinct triplet (draws / available; `-` = dropped):", "",
              "| rule | tier | available | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    rules = sorted({rule for arm in arms for rule in exposure[arm]["rules"]})
    for rule in rules:
        first = next(exposure[arm]["rules"][rule] for arm in arms if rule in exposure[arm]["rules"])
        cells = []
        for arm in arms:
            row = exposure[arm]["rules"].get(rule)
            cells.append("-" if not row or row["tier"] == "drop" or not row["draws"] else f"{row['repeats']:.1f}")
        lines.append(f"| {rule} | {first['tier']} | {first['available']:,} | " + " | ".join(cells) + " |")
    return lines + [""]


def _rule_rows(table: pd.DataFrame, rule: str) -> pd.DataFrame:
    sub = table[table["rule"] == rule]
    held = sub[sub["heldout"].astype(str).str.lower() == "true"]
    return held if len(held) else sub


def _weighted(sub: pd.DataFrame, column: str) -> float:
    return float((sub[column] * sub["n"]).sum() / sub["n"].sum())


def run_metrics(run_dir: Path, focus_rules: list[str]) -> dict[str, float] | None:
    validation = run_dir / "validation"
    per_rule, probe_path = validation / "per_rule.csv", validation / "retrieval_probe.json"
    if not per_rule.exists() or not probe_path.exists():
        return None
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    heldout = probe["retrieval"].get("heldout speakers", {})
    seen = probe["retrieval"].get("seen speakers", {})
    if "top1" in heldout:
        metrics["heldout_top1"] = heldout["top1"]
        metrics["heldout_top1_lift"] = heldout["top1"] - heldout["chance_top1"]
        metrics["heldout_top5"] = heldout["top5"]
    if "top1" in seen:
        metrics["seen_top1"] = seen["top1"]
        if "top1" in heldout:
            metrics["memory_gap"] = seen["top1"] - heldout["top1"]
    provenance = probe.get("provenance", {})
    if "accuracy" in provenance:
        metrics["provenance_acc"] = provenance["accuracy"]
        metrics["provenance_lift"] = provenance["accuracy"] - provenance["chance"]
    summary_path = validation / "cosine_summary.json"
    if summary_path.exists():
        # Geometry of the validation embeddings: how many directions they use, and how much length leaks.
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for source, key in (("vector_effective_rank", "effective_rank"),
                            ("pca_top_component_ratio", "top_component_ratio"),
                            ("length_linear_r2", "length_r2"),
                            ("length_r2_within_group", "length_r2_group"),
                            ("length_r2_within_speaker", "length_r2_speaker")):
            if source in summary:
                metrics[key] = float(summary[source])
    table = pd.read_csv(per_rule)
    if not table.empty:
        metrics["macro_violation"] = float(
            sum(_weighted(table[table["rule"] == rule], "violation") for rule in table["rule"].unique())
            / table["rule"].nunique()
        )
        for rule in focus_rules:
            sub = _rule_rows(table, rule)
            if len(sub):
                metrics[f"gap:{rule}"] = _weighted(sub, "gap")
                metrics[f"violation:{rule}"] = _weighted(sub, "violation")
    return metrics


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n}


def lower_is_better(metric: str) -> bool:
    return metric.startswith(LOWER_IS_BETTER)


def verdict(metric: str, stats: dict[str, float]) -> str:
    """Paired-delta call: a difference counts only beyond two standard errors across seeds."""
    if stats["n"] < 2:
        return "n<2"
    if abs(stats["mean"]) <= 2 * stats["sd"] / math.sqrt(stats["n"]):
        return "within noise"
    return "worse" if (stats["mean"] > 0) == lower_is_better(metric) else "better"


def compare(spec_path: str | Path, runs_root: Path | None = None, write: bool = True) -> dict[str, Any]:
    spec = load_spec(spec_path)
    runs_root = runs_root or workspace_dir() / "outputs" / "runs"
    focus = spec.get("focus_rules", DEFAULT_FOCUS_RULES)
    headline = spec.get("headline", DEFAULT_HEADLINE)
    kill = {**DEFAULT_KILL, **spec.get("kill", {})}
    per_run: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in spec["arms"]}
    missing: list[str] = []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            name = run_name(spec["name"], arm, int(seed))
            metrics = run_metrics(runs_root / name, focus)
            if metrics is None:
                missing.append(name)
            else:
                per_run[arm][int(seed)] = metrics
    arms: dict[str, dict[str, dict[str, float]]] = {}
    for arm, by_seed in per_run.items():
        names = sorted({m for metrics in by_seed.values() for m in metrics})
        arms[arm] = {m: _stats([s[m] for s in by_seed.values() if m in s]) for m in names}

    def paired(arm: str, against: str) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for seed in sorted(set(per_run[arm]) & set(per_run[against])):
            a, b = per_run[arm][seed], per_run[against][seed]
            out[seed] = {m: a[m] - b[m] for m in set(a) & set(b)}
        return out

    deltas = {c["name"]: paired(c["arm"], c["against"]) for c in spec["comparisons"]}
    comparisons = []
    for c in spec["comparisons"]:
        by_seed = deltas[c["name"]]
        names = sorted({m for d in by_seed.values() for m in d})
        effects = {m: _stats([d[m] for d in by_seed.values() if m in d]) for m in names}
        comparisons.append({**c, "effects": {m: {**s, "verdict": verdict(m, s)} for m, s in effects.items()}})
    interactions = []
    for interaction in spec["interactions"]:
        first, second = (deltas[name] for name in interaction["of"])
        by_seed = {seed: {m: first[seed][m] - second[seed][m] for m in set(first[seed]) & set(second[seed])}
                   for seed in set(first) & set(second)}
        names = sorted({m for d in by_seed.values() for m in d})
        effects = {m: _stats([d[m] for d in by_seed.values() if m in d]) for m in names}
        interactions.append({**interaction, "effects": {m: {**s, "verdict": verdict(m, s)} for m, s in effects.items()}})
    flagged = sorted(
        arm for arm, stats in arms.items()
        if "heldout_top1_lift" in stats and "provenance_acc" in stats
        and stats["heldout_top1_lift"]["mean"] < kill["retrieval_lift_max"]
        and stats["provenance_acc"]["mean"] > kill["provenance_min"]
    )
    result = {"name": spec["name"], "headline": headline, "missing_runs": missing, "arms": arms,
              "comparisons": comparisons, "interactions": interactions, "kill": kill, "kill_flagged": flagged}
    if write:
        out = output_dir(spec["name"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "compare.md").write_text(render_compare(result), encoding="utf-8")
    return result


def _cell(stats: dict[str, float] | None) -> str:
    if not stats:
        return "-"
    return f"{stats['mean']:+.3f} ± {stats['sd']:.3f} (n={stats['n']})"


def render_compare(result: dict[str, Any]) -> str:
    headline = result["headline"]
    lines = [f"# Ablation results: {result['name']}", ""]
    if result["missing_runs"]:
        lines += [f"Missing validation outputs ({len(result['missing_runs'])}): "
                  + ", ".join(f"`{name}`" for name in result["missing_runs"]), ""]
    lines += ["## Arms (mean ± sd over seeds)", "", "| arm | " + " | ".join(headline) + " |",
              "|---|" + "---|" * len(headline)]
    for arm, stats in result["arms"].items():
        lines.append(f"| {arm} | " + " | ".join(_cell(stats.get(m)) for m in headline) + " |")
    lines += ["", "## Comparisons (arm − against, paired by seed; call needs |Δ| > 2 SE)", "",
              "| comparison | isolates | metric | Δ | verdict |", "|---|---|---|---|---|"]
    for c in result["comparisons"]:
        for metric in headline:
            effect = c["effects"].get(metric)
            lines.append(f"| {c['name']} | {c.get('isolates', '')} | {metric} | {_cell(effect)} | "
                         f"{effect['verdict'] if effect else '-'} |")
    if result["interactions"]:
        lines += ["", "## Interactions (difference of paired deltas)", "", "| interaction | metric | Δ | verdict |",
                  "|---|---|---|---|"]
        for i in result["interactions"]:
            for metric in headline:
                effect = i["effects"].get(metric)
                lines.append(f"| {i['name']} | {metric} | {_cell(effect)} | {effect['verdict'] if effect else '-'} |")
    lines += ["", "## Kill criterion (FINDINGS.md §6)", ""]
    kill = result["kill"]
    if result["kill_flagged"]:
        lines.append(f"Held-out top-1 lift < {kill['retrieval_lift_max']} with provenance accuracy > "
                     f"{kill['provenance_min']} in: {result['kill_flagged']}. The limit there is the representation "
                     "or objective, not the sampling schedule.")
    else:
        lines.append("No arm meets it.")
    lines += ["", "Lower is better for provenance, violation, memory_gap, top_component_ratio and length_r2; higher for",
              "retrieval, gap and effective_rank.",
              "All metrics per arm are in `compare.json`."]
    return "\n".join(lines) + "\n"


def load_per_rule(path: str | Path) -> pd.DataFrame:
    """A validation `per_rule.csv`, or a validation dir; older dirs are rebuilt from rows + embeddings."""
    path = Path(path)
    if path.is_file():
        return pd.read_csv(path)
    if (path / "per_rule.csv").exists():
        return pd.read_csv(path / "per_rule.csv")
    import numpy as np

    rows = pd.read_parquet(path / "rows.parquet")
    if "heldout" not in rows:
        rows["heldout"] = False
    return per_rule_table(rows, np.load(path / "embeddings.npy"))


def measured_tiers(table: pd.DataFrame, easy_below: float = 0.15, hard_above: float = 0.35) -> dict[str, str]:
    """Tier each rule by a prior checkpoint's violation rate, held-out rows preferred.

    Difficulty measured, not assumed. Label noise also scores as hard, so audit before training on it.
    """
    if not easy_below < hard_above:
        raise ValueError("easy_below must be below hard_above")
    tiers: dict[str, str] = {}
    for rule in sorted(table["rule"].unique()):
        violation = _weighted(_rule_rows(table, rule), "violation")
        tiers[rule] = "easy" if violation < easy_below else "hard" if violation > hard_above else "mid"
    return tiers
