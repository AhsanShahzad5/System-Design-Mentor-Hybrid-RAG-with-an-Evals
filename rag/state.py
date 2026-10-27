"""State carried through the RAG graph and the result returned to callers."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.documents import Document

Route = Literal["catalogue", "rag_retrieval"]

# Token usage per model call, keyed by node (e.g. "query_analysis", "generation").
Usage = dict[str, dict[str, object]]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def merge_usage(left: Usage, right: Usage) -> Usage:
    """Combine usage reported by different nodes into one mapping."""
    return {**left, **right}


class RAGState(TypedDict):
    query: str
    history: list[ChatMessage]
    usage: Annotated[Usage, merge_usage]
    final_query: NotRequired[str]
    route: NotRequired[Route]
    documents: NotRequired[list[Document]]
    answer: NotRequired[str]


class RAGResult(TypedDict):
    """Everything evals need: the triad (query, context, answer) plus routing."""

    query: str
    final_query: str
    route: Route
    answer: str
    documents: list[Document]
    context: list[str]
    usage: Usage
