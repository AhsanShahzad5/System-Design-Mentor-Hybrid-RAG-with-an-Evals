# RAG and LangGraph conventions

These instructions apply to Python code inside `rag/`.

## Scope and architecture

- Use LangGraph to orchestrate the runtime RAG workflow.
- Keep retrieval implementation in `retrieval/`; graph nodes should call the
  existing `RetrievalPipeline` rather than duplicate dense search, BM25, RRF,
  or reranking logic.
- Begin with an explicit deterministic workflow such as
  `retrieve -> generate`. Add query rewriting, document grading, CRAG routing,
  or fallback search as explicit nodes only when required.
- Do not introduce a ReAct agent unless the application genuinely needs dynamic
  selection between multiple tools.
- Keep the root executable separate from reusable graph code. Use
  `run_rag_pipeline.py` as the composition entry point.

## Package layout

Prefer the following structure as the workflow grows:

```text
rag/
├── __init__.py
├── graph.py
├── state.py
├── nodes.py
├── schemas.py
├── chains.py
└── prompts/
```

- `graph.py`: graph construction, edges, conditional routing, and compilation.
- `state.py`: explicit typed graph state.
- `nodes.py`: node implementations or a small node container with injected
  dependencies.
- `schemas.py`: Pydantic models for structured model outputs.
- `chains.py`: prompt/model composition and structured-output configuration.
- `prompts/`: prompt templates grouped by task.
- Do not create modules preemptively when their responsibility does not yet
  exist. A smaller package is preferred for the initial graph.

## Graph construction

- Put orchestration in a `RAGPipeline` or `RAGGraph` class when a class helps
  hold shared dependencies such as the retrieval pipeline and chat model.
- Initialize dependencies in `__init__` and build the graph in
  `_build_graph()`.
- Expose one stable public method such as `invoke(query: str)`.
- Keep node functions private methods or focused module-level callables.
- Give nodes stable, descriptive identifiers such as `retrieve`,
  `grade_documents`, `rewrite_query`, and `generate_answer`.
- Conditional routers must return a valid next-node identifier or `Send`
  objects for deliberate fan-out.
- Do not hide deterministic application steps inside an agent loop.

## State

- Keep state fields explicit and typed.
- Prefer `TypedDict` for ordinary LangGraph state. Use a Pydantic `BaseModel`
  when runtime validation provides a concrete benefit.
- Mark fields that are produced later as optional or otherwise model their
  absence explicitly.
- Use `Annotated[..., operator.add]` only for fields intentionally aggregated
  across parallel nodes.
- Nodes must not mutate state in place.
- Each node returns the smallest state patch containing only keys it updated.
- Validate required state values at the node boundary and raise a clear error
  when they are absent.

## Prompts and model chains

- Store prompt templates under `rag/prompts/`; do not place large prompt strings
  in graph or node modules.
- Use explicit prompt names such as `answer_system_prompt` and
  `answer_user_prompt`.
- The system prompt contains behavioral instructions. The user prompt contains
  only request inputs and context placeholders.
- Centralize prompt/model composition in `chains.py`. A keyed `model_chain()`
  registry is optional and should be introduced only when several nodes need
  different chains.
- Prefer `model.with_structured_output(OutputModel)` for structured decisions
  such as relevance grading or routing.
- If a node returns free-form text, type and document it as free-form text; do
  not present it as structured output.
- Prompt changes must be deliberate and scoped to the node whose behavior needs
  improvement.

## Retrieval and generation

- Retrieval nodes return LangChain `Document` objects with source metadata
  intact.
- Generation must use only the retrieved context for factual book-grounded
  claims and should preserve enough metadata for later citations.
- Keep retrieval candidate counts, final context size, model names, and other
  tuning values configurable at the composition boundary.
- Do not add query rewriting or corrective retrieval by default. Add them only
  after manual testing or evaluation demonstrates a failure mode.

## Tracing and errors

- Trace the overall graph and major nodes with LangSmith.
- Do not silently swallow retrieval, parsing, routing, or model failures.
- Optional steps may return an empty state patch `{}` when no update is needed.
- Raise a clear error when the graph finishes without its required final
  answer.

## Compatibility

- Preserve public `invoke()` inputs and outputs unless all callers and related
  schemas are updated together.
- Keep changes scoped to the relevant node or chain.
- Reuse existing project helpers and dependencies before adding abstractions or
  new packages.
