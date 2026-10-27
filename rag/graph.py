"""Deterministic LangGraph workflow for catalogue and knowledge questions.

    analyze_query ─┬─ catalogue     → answer_catalogue ──────────→ END
                   └─ rag_retrieval → retrieve → generate ───────→ END
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from generation import Generator
from rag.chains import build_catalogue_answer_chain, build_query_analysis_chain
from rag.nodes import RAGNodes, Retriever
from rag.state import ChatMessage, RAGResult, RAGState


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


def _trace_outputs(outputs: Any) -> dict[str, Any]:
    """Keep context strings for online evals; log only metadata for documents."""
    if not isinstance(outputs, dict):
        return {"output": outputs}
    traced = {key: value for key, value in outputs.items() if key != "documents"}
    traced["sources"] = [document.metadata for document in outputs.get("documents", [])]
    return traced


class RAGGraph:
    """Route catalogue questions or retrieve book context and generate."""

    def __init__(
        self,
        model: BaseChatModel,
        retriever: Retriever,
        generator: Generator,
        catalogue_path: Path,
    ) -> None:
        self._nodes = RAGNodes(
            query_analysis_chain=build_query_analysis_chain(model),
            catalogue_answer_chain=build_catalogue_answer_chain(model),
            catalogue=self._load_catalogue(catalogue_path),
            retriever=retriever,
            generator=generator,
        )
        self._graph = self._build_graph()

    @staticmethod
    def _load_catalogue(catalogue_path: Path) -> str:
        if not catalogue_path.is_file():
            raise FileNotFoundError(f"Book catalogue not found: {catalogue_path}")

        catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
        return json.dumps(catalogue, ensure_ascii=False)

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("analyze_query", self._nodes.analyze_query)
        graph.add_node("answer_catalogue", self._nodes.answer_catalogue)
        graph.add_node("retrieve", self._nodes.retrieve)
        graph.add_node("generate", self._nodes.generate)

        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self._nodes.route_query,
            {
                "catalogue": "answer_catalogue",
                "rag_retrieval": "retrieve",
            },
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("answer_catalogue", END)
        graph.add_edge("generate", END)
        return graph.compile()

    @traceable(
        name="rag-graph",
        run_type="chain",
        process_inputs=_trace_inputs,
        process_outputs=_trace_outputs,
    )
    def invoke(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> RAGResult:
        state = self._graph.invoke(
            {"query": query, "history": history or [], "usage": {}}
        )

        answer = state.get("answer")
        if not answer:
            raise ValueError("The graph finished without producing an answer.")

        documents = state.get("documents", [])
        return {
            "query": query,
            "final_query": state["final_query"],
            "route": state["route"],
            "answer": answer,
            "documents": documents,
            "context": [document.page_content for document in documents],
            "usage": state.get("usage", {}),
        }
