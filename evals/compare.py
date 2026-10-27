"""compare.py -- the decision: is the candidate safe to promote?

Reads a baseline and a candidate snapshot, classifies every metric with its
registry rule, and returns a verdict:

  PASS   -- no gate blocked, no guardrail regressed. Safe to promote.
  REVIEW -- a guardrail regressed beyond its noise band. A human decides.
  FAIL   -- a gate (safety) regressed. Blocked.

Snapshots are comparable only if they used the same golden sets, judge model
and --limit; otherwise compare refuses (exit 3) unless --force.

Exit codes make it a CI gate: PASS=0, FAIL=1, REVIEW=2, INCOMPARABLE=3.

    uv run python -m evals.compare
    uv run python -m evals.compare --baseline a.json --candidate b.json --all
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evals.metric_registry import rule_for
from evals.run_suite import BASELINE_PATH, CANDIDATE_PATH

EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2, "INCOMPARABLE": 3}
STATUS_ORDER = ["blocked", "regressed", "dropped", "new", "improved", "flat", "info"]


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not (isinstance(value, float) and math.isnan(value))
    )


def comparability_problems(baseline: dict, candidate: dict) -> list[str]:
    b, c = baseline["metadata"], candidate["metadata"]
    problems = []
    for key in ("goldens", "judge_model", "limit"):
        if b.get(key) != c.get(key):
            problems.append(f"{key} differs: {b.get(key)} vs {c.get(key)}")
    return problems


def classify(metric_id: str, base: Any, cand: Any, noise_sd: float) -> dict[str, Any]:
    rule = rule_for(metric_id)
    row = {"id": metric_id, "baseline": base, "candidate": cand, "kind": rule["kind"],
           "delta": None, "band": None, "status": None}

    if base is None:
        row["status"] = "new"
        return row
    if cand is None:
        row["status"] = "dropped"
        return row

    if rule.get("bool") or isinstance(base, bool) or isinstance(cand, bool):
        if rule["kind"] == "info" or base == cand:
            row["status"] = "info" if rule["kind"] == "info" else "flat"
        elif cand and not base:
            row["status"] = "improved"
        else:
            row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
        return row

    if not (_is_number(base) and _is_number(cand)):
        row["status"] = "info"
        return row

    delta = cand - base
    row["delta"] = delta
    if rule["kind"] == "info":
        row["status"] = "info"
        return row

    improved = delta > 0 if rule["direction"] == "higher" else delta < 0
    band = max(rule["tol"], rule["rel_tol"] * abs(base), 2 * noise_sd)
    row["band"] = band
    if improved:
        row["status"] = "improved"
    elif abs(delta) <= band:
        row["status"] = "flat"
    else:
        row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
    return row


def compare(baseline: dict, candidate: dict) -> tuple[str, list[dict[str, Any]]]:
    b, c = baseline["metrics"], candidate["metrics"]
    noise = baseline.get("noise", {})
    rows = [
        classify(metric_id, b.get(metric_id), c.get(metric_id), noise.get(metric_id, 0.0))
        for metric_id in sorted(set(b) | set(c))
    ]
    statuses = {row["status"] for row in rows}
    if "blocked" in statuses:
        verdict = "FAIL"
    elif "regressed" in statuses:
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    return verdict, rows


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def print_report(verdict: str, rows: list[dict[str, Any]], show_all: bool = False) -> None:
    shown = [r for r in rows if show_all or r["kind"] != "info"]
    shown.sort(key=lambda r: (STATUS_ORDER.index(r["status"]), r["id"]))

    print("=" * 104)
    print(f"{'metric':<50} {'baseline':>10} {'candidate':>10} {'delta':>10} {'band':>9}  status")
    print("-" * 104)
    for r in shown:
        delta = f"{r['delta']:+.4g}" if r["delta"] is not None else ""
        band = f"±{r['band']:.3g}" if r["band"] is not None else ""
        marker = "  <<<" if r["status"] in ("blocked", "regressed") else ""
        print(f"{r['id']:<50} {_fmt(r['baseline']):>10} {_fmt(r['candidate']):>10} "
              f"{delta:>10} {band:>9}  {r['status']}{marker}")

    counts = Counter(r["status"] for r in rows)
    print("-" * 104)
    print("  ".join(f"{s}={counts[s]}" for s in STATUS_ORDER if counts[s]))
    if not show_all and counts["info"]:
        print(f"({counts['info']} info metrics hidden; pass --all to show them)")
    banner = {
        "PASS": "PASS   -- no gate blocked, no guardrail regressed. Safe to promote.",
        "REVIEW": "REVIEW -- a guardrail regressed beyond its noise band. A human decides.",
        "FAIL": "FAIL   -- a safety gate regressed. Blocked.",
    }[verdict]
    print("=" * 104)
    print(f"VERDICT: {banner}")
    print("=" * 104)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval snapshots.")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--all", action="store_true", help="also show info metrics")
    parser.add_argument("--force", action="store_true", help="compare even if not comparable")
    args = parser.parse_args()

    baseline, candidate = load(args.baseline), load(args.candidate)
    for name, snap in (("baseline", baseline), ("candidate", candidate)):
        m = snap["metadata"]
        print(f"{name:<9}: {m.get('label') or '(no label)'}  git {m['git_sha']}"
              f"{' (dirty)' if m.get('git_dirty') else ''}  prompt {m['prompt_hash']}  "
              f"namespace {m['settings'].get('pinecone_namespace')}  repeats {m.get('repeats')}")

    problems = comparability_problems(baseline, candidate)
    if problems and not args.force:
        print("\nINCOMPARABLE snapshots:\n  " + "\n  ".join(problems))
        sys.exit(EXIT_CODE["INCOMPARABLE"])

    verdict, rows = compare(baseline, candidate)
    print_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()
