"""Deterministic LangGraph workflow for catalogue and knowledge questions."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from generation import GenerationPipeline
from rag.chains import build_catalogue_answer_chain, build_query_analysis_chain
from rag.schemas import QueryAnalysis
from rag.state import ChatMessage, RAGState


class RAGGraph:
    """Route catalogue questions or run the existing book RAG pipeline."""

    def __init__(
        self,
        model: BaseChatModel,
        retriever: Callable[[str], list[Document]],
        catalogue_path: Path,
    ) -> None:
        self._catalogue = self._load_catalogue(catalogue_path)
        self._query_analysis_chain = build_query_analysis_chain(model)
        self._catalogue_answer_chain = build_catalogue_answer_chain(model)
        self._generation_pipeline = GenerationPipeline(
            retriever=retriever,
            model=model,
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
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("answer_catalogue", self._answer_catalogue)
        graph.add_node("answer_with_rag", self._answer_with_rag)

        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "catalogue": "answer_catalogue",
                "rag_retrieval": "answer_with_rag",
            },
        )
        graph.add_edge("answer_catalogue", END)
        graph.add_edge("answer_with_rag", END)
        return graph.compile()

    def _analyze_query(self, state: RAGState) -> dict[str, object]:
        query = state.get("query", "").strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        analysis = self._query_analysis_chain.invoke(
            {
                "query": query,
                "history": json.dumps(
                    state.get("history", []),
                    ensure_ascii=False,
                ),
            }
        )
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("Query analysis did not return QueryAnalysis.")

        final_query = analysis.final_query.strip()
        if not final_query:
            raise ValueError("Query analysis returned an empty final query.")

        return {
            "final_query": final_query,
            "route": analysis.route,
        }

    @staticmethod
    def _route_query(
        state: RAGState,
    ) -> Literal["catalogue", "rag_retrieval"]:
        route = state.get("route")
        if route not in {"catalogue", "rag_retrieval"}:
            raise ValueError(f"Invalid or missing route: {route}")
        return route

    def _answer_catalogue(self, state: RAGState) -> dict[str, str]:
        final_query = state.get("final_query", "").strip()
        if not final_query:
            raise ValueError("A final query is required for catalogue generation.")

        answer = self._catalogue_answer_chain.invoke(
            {
                "query": final_query,
                "catalogue": self._catalogue,
            }
        )
        if not answer.strip():
            raise ValueError("Catalogue generation returned an empty answer.")
        return {"answer": answer}

    def _answer_with_rag(self, state: RAGState) -> dict[str, str]:
        final_query = state.get("final_query", "").strip()
        if not final_query:
            raise ValueError("A final query is required for RAG generation.")

        answer = self._generation_pipeline.invoke(final_query)
        if not answer.strip():
            raise ValueError("RAG generation returned an empty answer.")
        return {"answer": answer}

    def invoke(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        result = self._graph.invoke(
            {
                "query": query,
                "history": history or [],
            }
        )
        answer = result.get("answer")
        if not answer:
            raise ValueError("The graph finished without producing an answer.")
        return answer
