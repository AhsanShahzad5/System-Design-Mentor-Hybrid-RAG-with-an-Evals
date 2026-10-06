"""State carried through the RAG graph."""

from typing import Literal, NotRequired, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RAGState(TypedDict):
    query: str
    history: list[ChatMessage]
    final_query: NotRequired[str]
    route: NotRequired[Literal["catalogue", "rag_retrieval"]]
    answer: NotRequired[str]
