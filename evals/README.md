# Eval suite

Offline evals for the book mentor, layer by layer. Golden sets and the safety policy live in
[`golden_datasets/`](../golden_datasets/README.md).

| Level | File | Metrics | Golden set |
|---|---|---|---|
| Component | `eval_retriever.py` | Contextual Recall, Contextual Precision | `retriever_goldens.json` |
| Component | `eval_generator.py` | Faithfulness, Answer Relevancy (on golden context) | `faithfulness_goldens.json` |
| Pipeline | `eval_rag_pipeline.py` | RAG triad: Contextual Relevancy, Faithfulness, Answer Relevancy; plus routing rate | faithfulness queries |
| Application | `eval_application.py` | G-Eval: Correctness, Completeness, Style | `correctness_goldens.json` |
| Application | `eval_safety.py` | Scope, Protected Information Leakage, PII Leakage, Toxicity | scope / leakage / toxicity |
| Application | `eval_ops.py` | Latency (e2e, TTFT, per stage), Cost, Reliability | fixed question list |

Run any layer on its own; `--limit N` uses only the first N goldens (smoke test):

```powershell
uv run python -m evals.eval_retriever --limit 2
uv run python -m evals.eval_safety
```

Every eval exposes `run(...)` returning a per-metric summary
(`n`, `pass_rate`, `avg_score`, `min_score`, `max_score`), ready for a regression runner.

## Regression testing

Any change (prompt, model, chunking, retrieval settings) must not regress the
**whole** suite. Three files:

- `run_suite.py` builds the app once, runs every eval, flattens the results to
  dotted ids (`application.correctness.avg_score`, `ops.latency.e2e_p95_ms`, ...),
  and writes a snapshot. The snapshot is stamped with git sha, prompt hash,
  golden-set hashes, judge model and all non-secret settings.
- `metric_registry.py` gives each metric a direction, a kind (safety **gates** block;
  quality and ops **guardrails** need review; everything else is **info**), and a
  minimum tolerance.
- `compare.py` diffs two snapshots and returns PASS (0), FAIL (1), REVIEW (2) or
  INCOMPARABLE (3, when golden sets, judge or `--limit` differ).

Noise threshold: record the baseline with `--repeats N`. Each metric's tolerance
becomes `max(minimum, 2 × SD across the baseline runs)`, so LLM-judge jitter
isn't reported as a regression.

```powershell
uv run python -m evals.run_suite --baseline --repeats 2 --label "baseline"
# make a change, then:
uv run python -m evals.run_suite --label "describe the change"
uv run python -m evals.compare
```

To evaluate a different index without editing code, override settings through
environment variables, for example `$env:PINECONE_NAMESPACE = "books-v2"`.

## Conventions

- **One app, built once.** Evals take the app, retriever or generator from `app.py`, so they
  measure exactly what runs in production.
- **Judge model:** `settings.eval_judge_model` (default `gpt-4.1-mini`), kept separate from the
  generation model. Threshold: `settings.eval_threshold` (0.7).
- **Higher is better for every metric.** DeepEval 4.x scores Toxicity and PII Leakage as 1 = pass.
  Older versions scored Toxicity as the share of toxic opinions, so a perfect run read as 1.0
  "toxic".
- **Concurrency:** app calls run in `settings.eval_max_workers` threads; DeepEval judges
  asynchronously.
- **Prices** for the cost eval are constants at the top of `eval_ops.py`, with sources and the
  date checked.
- Run logs go to `eval_results/` (gitignored).
