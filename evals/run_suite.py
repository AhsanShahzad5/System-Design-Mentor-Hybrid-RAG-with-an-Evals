"""run_suite.py -- run the FULL eval suite against ONE app and write a snapshot.

Builds the app once and injects it into every eval, so the snapshot provably
measures a single configuration. Every metric is flattened to a dotted id
("<layer>.<metric>.<stat>"), the run is stamped with provenance (git sha,
prompt hash, golden-set hashes, non-secret settings), and written as JSON.

With --repeats N the suite runs N times; the snapshot stores each metric's
mean plus its standard deviation, which compare.py turns into a 2xSD noise band.

    uv run python -m evals.run_suite --baseline --repeats 2 --label "baseline"
    uv run python -m evals.run_suite --label "cleaned data, books-v2"
    uv run python -m evals.run_suite --limit 2 --out baselines/smoke.json   # smoke
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import (
    eval_application,
    eval_generator,
    eval_ops,
    eval_rag_pipeline,
    eval_retriever,
    eval_safety,
)
from evals.harness import GOLDENS_DIRECTORY, JUDGE_MODEL
from evals.metric_registry import rule_for
from settings import PROJECT_ROOT

BASELINES_DIRECTORY = PROJECT_ROOT / "baselines"
BASELINE_PATH = BASELINES_DIRECTORY / "baseline.json"
CANDIDATE_PATH = BASELINES_DIRECTORY / "candidate.json"

# Summary stats kept per judge metric (n and max are rarely useful for diffs).
KEPT_STATS = ("avg_score", "pass_rate", "min_score", "errors")


# ---------------------------------------------------------------- flattening
def _slug(name: str) -> str:
    return name.replace("[GEval]", "").strip().lower().replace(" ", "_")


def flatten_summary(layer: str, summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """{metric: {stat: value}} -> {"layer.metric_slug.stat": value}."""
    return {
        f"{layer}.{_slug(metric)}.{stat}": value
        for metric, stats in summary.items()
        for stat, value in stats.items()
        if stat in KEPT_STATS
    }


def run_once(app, limit: int | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    steps = [
        ("retriever", lambda: eval_retriever.run(app.retriever, limit)),
        ("generator", lambda: eval_generator.run(app.generator, limit)),
        ("pipeline", lambda: eval_rag_pipeline.run(app, limit)),
        ("application", lambda: eval_application.run(app, limit)),
        ("safety", lambda: eval_safety.run_safety(app, limit)),
    ]
    for i, (layer, step) in enumerate(steps, start=1):
        print(f"\n[{i}/6] {layer} eval...")
        metrics.update(flatten_summary(layer, step()))

    print("\n[6/6] ops eval...")
    ops = eval_ops.run_ops(app, limit, verbose=False)
    metrics.update({f"ops.{key}": value for key, value in ops.items()})
    return metrics


# ---------------------------------------------------------------- provenance
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _prompt_hash() -> str:
    from generation.prompts import SYSTEM_PROMPT, USER_PROMPT
    from rag import prompts as rag_prompts

    rag_text = "".join(
        str(value) for name, value in sorted(vars(rag_prompts).items()) if name.endswith("_prompt")
    )
    return _sha(SYSTEM_PROMPT + USER_PROMPT + rag_text)


def _goldens_hash() -> dict[str, str]:
    return {
        path.name: _sha(path.read_text(encoding="utf-8"))
        for path in sorted(GOLDENS_DIRECTORY.glob("*_goldens.json"))
    }


def build_metadata(app, label: str, limit: int | None, repeats: int) -> dict[str, Any]:
    settings = app.settings.model_dump(mode="json")
    public = {
        key: value
        for key, value in settings.items()
        if "api_key" not in key and not key.endswith(("_path", "_directory"))
    }
    return {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "prompt_hash": _prompt_hash(),
        "judge_model": JUDGE_MODEL,
        "goldens": _goldens_hash(),
        "settings": public,
        "limit": limit,
        "repeats": repeats,
    }


# ---------------------------------------------------------------- aggregation
def aggregate(runs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    """Mean of each metric across runs, plus its SD when there are 2+ runs."""
    means: dict[str, Any] = {}
    noise: dict[str, float] = {}
    for metric_id in sorted(set().union(*runs)):
        values = [run[metric_id] for run in runs if metric_id in run]
        if all(isinstance(v, bool) for v in values):
            means[metric_id] = all(values)  # a flag passes only if it passed every run
            continue
        numbers = [float(v) for v in values if not math.isnan(float(v))]
        means[metric_id] = statistics.fmean(numbers) if numbers else math.nan
        if len(numbers) > 1:
            noise[metric_id] = statistics.stdev(numbers)
    return means, noise


def run_suite(label: str = "", limit: int | None = None, repeats: int = 1) -> dict[str, Any]:
    from app import build_app

    started = time.perf_counter()
    print("Building app (once)...")
    app = build_app()

    runs = []
    for repeat in range(1, repeats + 1):
        print(f"\n######## suite run {repeat}/{repeats}")
        runs.append(run_once(app, limit))

    metrics, noise = aggregate(runs)
    metadata = build_metadata(app, label, limit, repeats)
    metadata["suite_seconds"] = round(time.perf_counter() - started, 1)
    return {"metadata": metadata, "metrics": metrics, "noise": noise, "runs": runs}


def write_snapshot(snapshot: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_snapshot(snapshot: dict[str, Any]) -> None:
    meta, metrics, noise = snapshot["metadata"], snapshot["metrics"], snapshot["noise"]
    print("\n" + "=" * 84)
    print(f"SNAPSHOT  {meta['label'] or '(no label)'}   git {meta['git_sha']}"
          f"{' (dirty)' if meta['git_dirty'] else ''}   prompt {meta['prompt_hash']}   "
          f"{meta['suite_seconds']}s")
    print("-" * 84)
    for metric_id in sorted(metrics):
        if rule_for(metric_id)["kind"] == "info":
            continue
        value = metrics[metric_id]
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        sd = f"  ±{noise[metric_id]:.4f}" if metric_id in noise else ""
        print(f"  {metric_id:<52} {shown}{sd}")
    print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full eval suite and write a snapshot.")
    parser.add_argument("--baseline", action="store_true", help=f"write {BASELINE_PATH.name}")
    parser.add_argument("--out", type=Path, default=None, help="custom output path")
    parser.add_argument("--label", default="", help="what change this snapshot represents")
    parser.add_argument("--repeats", type=int, default=1, help="suite runs to average (noise)")
    parser.add_argument("--limit", type=int, default=None, help="first N goldens only (smoke)")
    args = parser.parse_args()

    out = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    snapshot = run_suite(label=args.label, limit=args.limit, repeats=args.repeats)
    print_snapshot(snapshot)
    print(f"\nwrote snapshot -> {write_snapshot(snapshot, out)}")


if __name__ == "__main__":
    main()
