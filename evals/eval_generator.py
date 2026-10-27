"""Component eval: the GENERATOR, in isolation.

The generator is fed the GOLDEN context (exact book passages from
faithfulness_goldens.json), not the retriever's output, so a low score is the
generator's fault alone.

  Faithfulness     -- are the answer's claims supported by the given context?
  Answer Relevancy -- do the answer's statements address the question?

    uv run python -m evals.eval_generator [--limit N]
"""

import runpy

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    GOLDENS_DIRECTORY,
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
from generation import Generator

GOLDENS = "faithfulness_goldens.json"
BUILD_SCRIPT = GOLDENS_DIRECTORY / "build_faithfulness_goldens.py"


def ensure_goldens() -> None:
    """The faithfulness set is gitignored (verbatim book text); build if absent."""
    if not (GOLDENS_DIRECTORY / GOLDENS).is_file():
        runpy.run_path(str(BUILD_SCRIPT), run_name="__main__")


def run(generator: Generator, limit: int | None = None) -> dict:
    ensure_goldens()
    goldens = load_goldens(GOLDENS, limit)
    answers = map_concurrently(
        lambda g: generator.generate(g["query"], g["ideal_context"]).answer, goldens
    )

    test_cases = [
        LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=g["ideal_context"],
        )
        for g, answer in zip(goldens, answers, strict=True)
    ]

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = run_evaluation(test_cases, metrics, hyperparameters={"eval": "generator"})
    print_failures(result)
    return summarize_by_metric(result)


if __name__ == "__main__":
    from app import build_chat_model
    from document_processing import BOOK_TITLES
    from settings import get_settings

    limit = cli_limit("Generator eval: faithfulness + answer relevancy on golden context")
    generator = Generator(build_chat_model(get_settings()), BOOK_TITLES)
    print_summary("generator", run(generator, limit))
