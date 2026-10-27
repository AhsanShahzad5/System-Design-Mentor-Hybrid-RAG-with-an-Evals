"""Pipeline eval: the RAG TRIAD on the full app (retriever + generator together).

All three are reference-free -- the judge only needs query, retrieved context
and answer:
  Contextual Relevancy -- how much of the retrieved context is relevant to the query
  Faithfulness         -- is the answer grounded in the retrieved context?
  Answer Relevancy     -- does the answer address the query?

Reuses the faithfulness queries; ideal_context is ignored, context comes from
the live retriever. Also reports how often these knowledge questions were
routed to retrieval: a misroute to the catalogue gives empty context.

    uv run python -m evals.eval_rag_pipeline [--limit N]
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evals.eval_generator import GOLDENS, ensure_goldens
from evals.harness import (
    JUDGE_MODEL,
    THRESHOLD,
    _stats,
    cli_limit,
    load_goldens,
    map_concurrently,
    print_failures,
    print_summary,
    run_evaluation,
    summarize_by_metric,
)


def run(app, limit: int | None = None) -> dict:
    ensure_goldens()
    goldens = load_goldens(GOLDENS, limit)
    results = map_concurrently(lambda g: app.invoke(g["query"]), goldens)

    routed = [r["route"] == "rag_retrieval" for r in results]
    test_cases = [
        LLMTestCase(
            input=g["query"],
            actual_output=r["answer"],
            retrieval_context=r["context"] or ["(no context retrieved)"],
        )
        for g, r in zip(goldens, results, strict=True)
    ]

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = run_evaluation(test_cases, metrics, hyperparameters={"eval": "rag_pipeline"})
    print_failures(result)
    summary = summarize_by_metric(result)
    summary["Routed To Retrieval"] = _stats(
        [float(r) for r in routed], sum(routed), len(routed)
    )
    return summary


if __name__ == "__main__":
    from app import build_app

    limit = cli_limit("RAG triad: contextual relevancy, faithfulness, answer relevancy")
    print_summary("rag_pipeline", run(build_app(), limit))
