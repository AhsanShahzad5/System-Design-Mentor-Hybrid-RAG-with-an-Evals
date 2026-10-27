"""Operational evals: LATENCY, COST, RELIABILITY.

Measurements, not judgements: no golden set and no LLM judge. One benchmark
loop streams the full app over a fixed question list; every run feeds all
three reports:

  LATENCY     -- percentiles (p50/p95/p99, not the mean) of end-to-end time,
                 time to first token (what a streaming user feels), and the
                 per-stage split: query analysis, retrieval, generation.
                 Warmup runs are discarded. Single user only; load testing is
                 a separate job.
  COST        -- derived, not measured: tokens x price, from the usage the app
                 records per model call, plus the query embedding and the
                 Pinecone rerank request.
  RELIABILITY -- success / error / retry rates, with exponential-backoff retries.

    uv run python -m evals.eval_ops [--limit N]
"""

import math
import time
import warnings
from typing import Any

import tiktoken

from evals.harness import cli_limit
from settings import get_settings

# LangChain warns when serializing structured output while streaming; harmless.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Short, medium and long-answer questions: answer length drives latency and cost.
QUESTIONS = [
    "What is the CAP theorem?",
    "How does the sliding window log rate limiter work?",
    "Explain how LSM-trees work and why their writes are fast.",
    "How should an ML system detect and respond to data distribution shifts?",
    "Compare fanout on write and fanout on read for a news feed, with their tradeoffs.",
]

REPEATS = 2  # measured runs per question
WARMUP_RUNS = 1  # discarded (cold connections)
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5

# SLOs: a latency number means nothing without a target. Product decisions;
# tune them to what your users accept.
SLO_E2E_P95_MS = 15_000
SLO_TTFT_P95_MS = 5_000

# Prices in USD, checked 2026-10-27. Re-check before trusting a budget.
# gpt-5.6-luna, Standard tier after the 2026-07-30 price cut
#   https://openrouter.ai/openai/gpt-5.6-luna
PRICE_INPUT_PER_1M = 0.20
PRICE_CACHED_INPUT_PER_1M = 0.02
PRICE_OUTPUT_PER_1M = 1.20
# text-embedding-3-large  https://developers.openai.com/api/docs/models/text-embedding-3-large
PRICE_EMBEDDING_PER_1M = 0.13
# Pinecone bge-reranker-v2-m3, Standard plan: $2 per 1k requests  https://www.pinecone.io/pricing/
PRICE_RERANK_PER_REQUEST = 0.002
# Not included: Pinecone read units for the dense and BM25 searches.

QUERIES_PER_DAY = 1_000
COST_BUDGET_PER_QUERY_USD = 0.005

_ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------- helpers
def percentile(values: list[float], p: float) -> float:
    values = sorted(v for v in values if not math.isnan(v))
    if not values:
        return math.nan
    k = (len(values) - 1) * (p / 100)
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo] * (hi - k) + values[hi] * (k - lo)


def distribution(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean) if clean else math.nan,
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
    }


# ---------------------------------------------------------------- one run
def timed_run(app, question: str) -> dict[str, Any]:
    """Stream one question; return stage timings (ms) and the result."""
    start = time.perf_counter()
    marks: dict[str, float] = {}
    first_token = math.nan
    result = None

    for kind, value in app.graph.stream(question):
        now = (time.perf_counter() - start) * 1000
        if kind == "token" and math.isnan(first_token):
            first_token = now
        elif kind == "node":
            marks[value] = now
        elif kind == "result":
            result = value

    total = (time.perf_counter() - start) * 1000
    analysis = marks.get("analyze_query", math.nan)
    retrieved = marks.get("retrieve", math.nan)
    return {
        "total": total,
        "ttft": first_token,
        "analysis": analysis,
        "retrieval": retrieved - analysis,
        "generation": total - retrieved,
        "result": result,
    }


def call_with_retries(fn, counters: dict[str, int]):
    counters["requests"] += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            value = fn()
            counters["successes"] += 1
            return value
        except Exception as error:  # noqa: BLE001 -- counting failures is the point
            if attempt < MAX_RETRIES:
                counters["retries"] += 1
                time.sleep(BACKOFF_BASE_S * 2**attempt)
            else:
                counters["failures"] += 1
                print(f"[reliability] FAILED after {MAX_RETRIES} retries: {error}")
    return None


# ---------------------------------------------------------------- cost
def query_cost(result: dict[str, Any]) -> dict[str, float]:
    input_tokens = output_tokens = cached_tokens = 0
    for usage in result["usage"].values():
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        details = usage.get("input_token_details") or {}
        cached_tokens += int(details.get("cache_read", 0) or 0)

    llm = (
        max(input_tokens - cached_tokens, 0) / 1e6 * PRICE_INPUT_PER_1M
        + cached_tokens / 1e6 * PRICE_CACHED_INPUT_PER_1M
        + output_tokens / 1e6 * PRICE_OUTPUT_PER_1M
    )
    retrieved = result["route"] == "rag_retrieval"
    embedding = (
        len(_ENCODER.encode(result["final_query"])) / 1e6 * PRICE_EMBEDDING_PER_1M
        if retrieved
        else 0.0
    )
    rerank = PRICE_RERANK_PER_REQUEST if retrieved else 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "llm": llm,
        "embedding": embedding,
        "rerank": rerank,
        "total": llm + embedding + rerank,
    }


# ---------------------------------------------------------------- suite
def run_ops(app, limit: int | None = None, verbose: bool = True) -> dict[str, float]:
    questions = QUESTIONS[:limit] if limit else QUESTIONS

    for i in range(WARMUP_RUNS):
        app.invoke(questions[i % len(questions)])

    counters = {"requests": 0, "successes": 0, "failures": 0, "retries": 0}
    runs = []
    for question in questions:
        for _ in range(REPEATS):
            run = call_with_retries(lambda q=question: timed_run(app, q), counters)
            if run is not None:
                runs.append(run)

    latency = {
        stage: distribution([r[stage] for r in runs])
        for stage in ("total", "ttft", "analysis", "retrieval", "generation")
    }
    costs = [query_cost(r["result"]) for r in runs]
    avg = {key: sum(c[key] for c in costs) / len(costs) for key in costs[0]} if costs else {}
    answer_chars = sum(len(r["result"]["answer"]) for r in runs) / max(len(runs), 1)

    requests = counters["requests"] or 1
    metrics = {
        "latency.e2e_p50_ms": latency["total"]["p50"],
        "latency.e2e_p95_ms": latency["total"]["p95"],
        "latency.e2e_p99_ms": latency["total"]["p99"],
        "latency.ttft_p50_ms": latency["ttft"]["p50"],
        "latency.ttft_p95_ms": latency["ttft"]["p95"],
        "latency.analysis_p95_ms": latency["analysis"]["p95"],
        "latency.retrieval_p95_ms": latency["retrieval"]["p95"],
        "latency.generation_p95_ms": latency["generation"]["p95"],
        "latency.avg_answer_chars": answer_chars,
        "latency.slo_e2e_pass": latency["total"]["p95"] <= SLO_E2E_P95_MS,
        "latency.slo_ttft_pass": latency["ttft"]["p95"] <= SLO_TTFT_P95_MS,
        "cost.per_query_usd": avg.get("total", math.nan),
        "cost.llm_per_query_usd": avg.get("llm", math.nan),
        "cost.rerank_per_query_usd": avg.get("rerank", math.nan),
        "cost.embedding_per_query_usd": avg.get("embedding", math.nan),
        "cost.avg_input_tokens": avg.get("input_tokens", math.nan),
        "cost.avg_output_tokens": avg.get("output_tokens", math.nan),
        "cost.avg_cached_tokens": avg.get("cached_tokens", math.nan),
        "cost.monthly_usd": avg.get("total", math.nan) * QUERIES_PER_DAY * 30,
        "cost.budget_pass": avg.get("total", math.inf) <= COST_BUDGET_PER_QUERY_USD,
        "reliability.requests": counters["requests"],
        "reliability.success_rate": 100 * counters["successes"] / requests,
        "reliability.error_rate": 100 * counters["failures"] / requests,
        "reliability.retry_rate": 100 * counters["retries"] / requests,
    }
    if verbose:
        report(latency, avg, metrics)
    return metrics


def report(latency: dict, avg: dict, metrics: dict) -> None:
    print("\n" + "=" * 78)
    print("LATENCY (ms)")
    print("-" * 78)
    for stage, d in latency.items():
        print(f"  {stage:<11} n={d['n']:<3} mean={d['mean']:8.0f}  p50={d['p50']:8.0f}  "
              f"p95={d['p95']:8.0f}  p99={d['p99']:8.0f}")
    print(f"  avg answer length: {metrics['latency.avg_answer_chars']:.0f} chars")
    print(f"  SLO e2e p95 <= {SLO_E2E_P95_MS} ms: {'PASS' if metrics['latency.slo_e2e_pass'] else 'FAIL'}"
          f"   SLO ttft p95 <= {SLO_TTFT_P95_MS} ms: {'PASS' if metrics['latency.slo_ttft_pass'] else 'FAIL'}")
    if avg:
        print("-" * 78)
        print("COST per query (USD)")
        print(f"  llm ${avg['llm']:.6f} + rerank ${avg['rerank']:.6f} + embedding ${avg['embedding']:.8f}"
              f" = ${avg['total']:.6f}")
        print(f"  tokens: {avg['input_tokens']:.0f} in ({avg['cached_tokens']:.0f} cached), "
              f"{avg['output_tokens']:.0f} out")
        print(f"  @ {QUERIES_PER_DAY}/day: ${metrics['cost.monthly_usd']:.2f}/month"
              f"   budget <= ${COST_BUDGET_PER_QUERY_USD}: {'PASS' if metrics['cost.budget_pass'] else 'FAIL'}")
    print("-" * 78)
    print(f"RELIABILITY  requests={metrics['reliability.requests']}  "
          f"success={metrics['reliability.success_rate']:.1f}%  "
          f"errors={metrics['reliability.error_rate']:.1f}%  retries={metrics['reliability.retry_rate']:.1f}%")
    print("=" * 78)


run = run_ops


if __name__ == "__main__":
    from app import build_app

    limit = cli_limit("Ops evals: latency, cost, reliability")
    run_ops(build_app(), limit)
