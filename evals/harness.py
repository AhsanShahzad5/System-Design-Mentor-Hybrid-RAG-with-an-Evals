"""Shared harness for the eval suite.

  load_goldens(name, limit)      read a golden file from golden_datasets/
  map_concurrently(fn, items)    run app calls in parallel, preserving order
  run_evaluation(...)            DeepEval evaluate() with the suite's settings
  summarize_by_metric(result)    per-metric summary: n, pass_rate, avg/min/max score
  print_summary(title, summary)  readable recap
  cli_limit()                    --limit N for quick smoke runs

Summaries stay per metric so a regression in one metric can't hide behind
another. Every metric in this suite is higher-is-better (DeepEval 4.x scores
Toxicity and PII Leakage as 1 = pass), and pass/fail comes from each metric's
own `success` flag.
"""

import argparse
import json
import math
import os

# DeepEval's default per-test-case deadline is too short for multi-metric cases
# with long retrieval context (the RAG triad); must be set before importing it.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any, TypeVar

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

from settings import PROJECT_ROOT, get_settings

GOLDENS_DIRECTORY = PROJECT_ROOT / "golden_datasets"

settings = get_settings()
JUDGE_MODEL = settings.eval_judge_model
THRESHOLD = settings.eval_threshold
MAX_WORKERS = settings.eval_max_workers

T = TypeVar("T")
R = TypeVar("R")


def load_goldens(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = GOLDENS_DIRECTORY / name
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def map_concurrently(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = MAX_WORKERS,
) -> list[R]:
    """Apply fn to every item in parallel threads; results keep input order.

    Each call runs in a copy of the current context so LangSmith tracing still
    nests correctly. Any exception is raised: a broken app call should fail the
    eval loudly, not be scored as a bad answer.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_context().run, fn, item) for item in items]
        return [future.result() for future in futures]


def run_evaluation(
    test_cases: list[LLMTestCase],
    metrics: list[BaseMetric],
    hyperparameters: dict[str, Any] | None = None,
):
    """Run DeepEval with async judging and the suite's run metadata."""
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL,
            "generation_model": settings.generation_model,
            **(hyperparameters or {}),
        },
        async_config=AsyncConfig(run_async=True, max_concurrent=5),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
        # A judge failure (e.g. a degenerate, length-capped response) marks that
        # metric as errored -- counted as a failure and reported -- instead of
        # aborting the whole run.
        error_config=ErrorConfig(ignore_errors=True),
    )


def summarize_by_metric(result) -> dict[str, dict[str, float]]:
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets: dict[str, dict[str, Any]] = {}
    for test_result in test_results:
        metrics = (
            getattr(test_result, "metrics_data", None)
            or getattr(test_result, "metrics", None)
            or []
        )
        for metric in metrics:
            name = getattr(metric, "name", "unknown")
            bucket = buckets.setdefault(
                name, {"scores": [], "passed": 0, "total": 0, "errors": 0}
            )
            bucket["total"] += 1
            if getattr(metric, "error", None):
                bucket["errors"] += 1
            score = getattr(metric, "score", None)
            if score is not None:
                bucket["scores"].append(score)
            if getattr(metric, "success", False):
                bucket["passed"] += 1

    return {
        name: _stats(b["scores"], b["passed"], b["total"], b["errors"])
        for name, b in buckets.items()
    }


def _stats(
    scores: list[float], passed: int, total: int, errors: int = 0
) -> dict[str, float]:
    return {
        "n": total,
        "errors": errors,
        "pass_rate": (100 * passed / total) if total else 0.0,
        "avg_score": (sum(scores) / len(scores)) if scores else math.nan,
        "min_score": min(scores) if scores else math.nan,
        "max_score": max(scores) if scores else math.nan,
    }


def failing_cases(result, limit: int = 5) -> list[dict[str, Any]]:
    """The lowest-scoring (input, metric, score, reason) rows, for quick triage."""
    rows = []
    for test_result in getattr(result, "test_results", []) or []:
        for metric in getattr(test_result, "metrics_data", None) or []:
            if not getattr(metric, "success", True):
                rows.append(
                    {
                        "input": str(getattr(test_result, "input", ""))[:90],
                        "metric": metric.name,
                        "score": metric.score,
                        "reason": (metric.reason or "")[:220],
                    }
                )
    rows.sort(key=lambda row: (row["score"] is None, row["score"] or 0))
    return rows[:limit]


def print_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print(f"{title}  (per-metric summary, judge={JUDGE_MODEL})")
    print("=" * 72)
    for name, stats in summary.items():
        avg = stats["avg_score"]
        avg_text = "nan" if math.isnan(avg) else f"{avg:.3f}"
        errors = f"  errors={stats['errors']}" if stats.get("errors") else ""
        print(f"  {name:<40} pass_rate={stats['pass_rate']:5.1f}%  avg={avg_text}  n={stats['n']}{errors}")
    print("=" * 72)


def print_failures(result, limit: int = 5) -> None:
    rows = failing_cases(result, limit)
    if not rows:
        return
    print(f"Lowest-scoring failures (up to {limit}):")
    for row in rows:
        print(f"  - [{row['metric']}] {row['score']}  {row['input']!r}\n      {row['reason']}")


def cli_limit(description: str) -> int | None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=None, help="only the first N goldens")
    return parser.parse_args().limit


def take(items: Iterable[T], limit: int | None) -> list[T]:
    items = list(items)
    return items[:limit] if limit else items
