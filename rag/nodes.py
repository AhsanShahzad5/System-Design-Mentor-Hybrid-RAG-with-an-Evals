"""Graph node implementations with injected dependencies."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from generation import NO_CONTEXT_ANSWER, Generator
from generation.helpers import message_text, usage_from_message
from rag.schemas import QueryAnalysis
from rag.state import RAGState, Route

Retriever = Callable[[str], list[Document]]


def _require_final_query(state: RAGState) -> str:
    final_query = state.get("final_query", "").strip()
    if not final_query:
        raise ValueError("A final query is required after query analysis.")
    return final_query


class RAGNodes:
    """Nodes for: analyze_query -> (answer_catalogue | retrieve -> generate)."""

    def __init__(
        self,
        query_analysis_chain: Runnable,
        catalogue_answer_chain: Runnable,
        catalogue: str,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._query_analysis_chain = query_analysis_chain
        self._catalogue_answer_chain = catalogue_answer_chain
        self._catalogue = catalogue
        self._retriever = retriever
        self._generator = generator

    def analyze_query(self, state: RAGState) -> dict[str, Any]:
        query = state.get("query", "").strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        output = self._query_analysis_chain.invoke(
            {
                "query": query,
                "history": json.dumps(state.get("history", []), ensure_ascii=False),
            }
        )
        if output.get("parsing_error") is not None:
            raise ValueError("Query analysis output could not be parsed.") from output[
                "parsing_error"
            ]

        analysis = output.get("parsed")
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("Query analysis did not return QueryAnalysis.")

        final_query = analysis.final_query.strip()
        if not final_query:
            raise ValueError("Query analysis returned an empty final query.")

        return {
            "final_query": final_query,
            "route": analysis.route,
            "usage": {"query_analysis": usage_from_message(output["raw"])},
        }

    @staticmethod
    def route_query(state: RAGState) -> Route:
        route = state.get("route")
        if route not in {"catalogue", "rag_retrieval"}:
            raise ValueError(f"Invalid or missing route: {route}")
        return route

    def answer_catalogue(self, state: RAGState) -> dict[str, Any]:
        message = self._catalogue_answer_chain.invoke(
            {"query": _require_final_query(state), "catalogue": self._catalogue}
        )
        answer = message_text(message).strip()
        if not answer:
            raise ValueError("Catalogue generation returned an empty answer.")
        return {
            "answer": answer,
            "documents": [],
            "usage": {"catalogue_answer": usage_from_message(message)},
        }

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        return {"documents": self._retriever(_require_final_query(state))}

    def generate(self, state: RAGState) -> dict[str, Any]:
        final_query = _require_final_query(state)
        documents = state.get("documents")
        if documents is None:
            raise ValueError("Retrieval must run before generation.")

        if not documents:
            return {"answer": NO_CONTEXT_ANSWER}

        result = self._generator.generate(final_query, documents)
        return {"answer": result.answer, "usage": {"generation": result.usage}}
