"""Component eval: the RETRIEVER, in isolation.

Two failure modes, two metrics (both reference-based, need ideal_answer):
  Contextual Recall    -- missed context: does the retrieved context support
                          every sentence of the ideal answer?
  Contextual Precision -- noisy context: are the relevant chunks ranked first?

Runs the real retriever (dense + BM25 -> RRF -> rerank -> top 5).

    uv run python -m evals.eval_retriever [--limit N]
"""

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    JUDGE_MODEL,
    THRESHOLD,
    cli_limit,
    load_goldens,
    map_concurrently,
    print_failures,
    print_summary,
    run_evaluation,
    summarize_by_metric,
)
from retrieval import RetrievalPipeline
from settings import get_settings

GOLDENS = "retriever_goldens.json"


def run(retriever: RetrievalPipeline, limit: int | None = None) -> dict:
    goldens = load_goldens(GOLDENS, limit)
    retrieved = map_concurrently(lambda g: retriever.retrieve(g["query"]), goldens)

    test_cases = [
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=[doc.page_content for doc in docs],
            # component-level: no generator in this eval
            actual_output="(generator not evaluated in this run)",
        )
        for g, docs in zip(goldens, retrieved, strict=True)
    ]

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    settings = get_settings()
    result = run_evaluation(
        test_cases,
        metrics,
        hyperparameters={
            "eval": "retriever",
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "dense_top_k": settings.dense_top_k,
            "bm25_top_k": settings.bm25_top_k,
            "final_top_k": settings.final_top_k,
            "rerank_model": settings.rerank_model,
            "namespace": settings.pinecone_namespace,
        },
    )
    print_failures(result)
    return summarize_by_metric(result)


if __name__ == "__main__":
    from app import build_retriever

    limit = cli_limit("Retriever eval: contextual recall + precision")
    print_summary("retriever", run(build_retriever(get_settings()), limit))
