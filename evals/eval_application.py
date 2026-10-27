"""Application quality eval: CORRECTNESS, COMPLETENESS, STYLE (G-Eval).

Judgement-based metrics, made close to deterministic with fixed evaluation
steps and a scoring rubric:
  Correctness  -- reference-based, judges TRUTH only (not coverage or length)
  Completeness -- reference-based, judges COVERAGE of the ideal answer's key points
  Style        -- reference-free, judges the mentor voice and that no internal
                  source labels or citations leak into the answer

The ideal answers are an expert answer key: the app's answer must be correct,
not necessarily drawn only from retrieved text. For `not_covered` questions
the correct answer is to say the books don't cover the topic.

    uv run python -m evals.eval_application [--limit N]
"""

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

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

GOLDENS = "correctness_goldens.json"

REFERENCE_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
]


def build_metrics() -> list[GEval]:
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "If the expected output says the books do not cover the topic, the actual output is correct only if it says the available material does not cover it and does not invent an answer.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points; omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent with the expected output. Brevity is fine."),
            Rubric(score_range=(5, 8), expected_outcome="Mostly correct but contains one minor inaccuracy."),
            Rubric(score_range=(0, 4), expected_outcome="Contains a clear factual error, contradicts the expected output, or invents an answer the expected output says is not covered."),
        ],
        evaluation_params=REFERENCE_PARAMS,
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly; correctness is judged separately.",
            "Do NOT penalize the actual output for adding information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8), expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4), expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=REFERENCE_PARAMS,
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and presentation of the actual output, not whether it is correct or complete.",
            "Reward a patient mentor voice: the direct answer comes first, then the explanation; intuition comes before jargon; technical terms are briefly unpacked.",
            "Reward explanations that are organized around mechanisms, tradeoffs and examples, and that end with a concise recap when the answer is long.",
            "Penalize padding, repetition, a stiff or robotic tone, or a bare list with no explanation.",
            "Penalize heavily any internal source labels or citations shown to the learner, such as 'Source 1', 'Source 2', or bracketed references, unless the learner explicitly asked for sources.",
            "A short, direct answer to a simple factual or catalogue question is fully acceptable and must not be penalized for brevity.",
            "If the actual output says the available material does not cover the question, judge only whether that message is clear, polite and helpful (for example offering a related topic); do not penalize it for not teaching the topic.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clear mentor voice: direct answer first, intuitive explanation, well organized, no leaked source labels."),
            Rubric(score_range=(7, 8), expected_outcome="Clear and well explained with minor stylistic issues; no leaked source labels."),
            Rubric(score_range=(4, 6), expected_outcome="Understandable but flat, padded, or list-heavy in places."),
            Rubric(score_range=(0, 3), expected_outcome="Dry, robotic, disorganized, or shows internal source labels or citations to the learner."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )
    return [correctness, completeness, style]


def run(app, limit: int | None = None) -> dict:
    goldens = load_goldens(GOLDENS, limit)
    results = map_concurrently(lambda g: app.invoke(g["question"]), goldens)

    test_cases = [
        LLMTestCase(
            input=g["question"],
            actual_output=r["answer"],
            expected_output=g["ideal_answer"],
            additional_metadata={"type": g["type"], "route": r["route"]},
        )
        for g, r in zip(goldens, results, strict=True)
    ]

    result = run_evaluation(test_cases, build_metrics(), hyperparameters={"eval": "application"})
    print_failures(result)
    return summarize_by_metric(result)


if __name__ == "__main__":
    from app import build_app

    limit = cli_limit("Application quality: correctness, completeness, style")
    print_summary("application", run(build_app(), limit))
