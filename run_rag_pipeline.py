"""Run the LangGraph book mentor using query and history from input.json."""

import json

from langchain_openai import ChatOpenAI

from rag import RAGGraph
from rag.state import ChatMessage
from run_retrieval_pipeline import retrieve
from settings import get_settings

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
GENERATION_MODEL = settings.generation_model
BOOK_CATALOGUE_PATH = settings.book_catalogue_path
RAG_INPUT_PATH = settings.rag_input_path


def load_input() -> tuple[str, list[ChatMessage]]:
    if not RAG_INPUT_PATH.is_file():
        raise FileNotFoundError(f"RAG input file not found: {RAG_INPUT_PATH}")

    payload = json.loads(RAG_INPUT_PATH.read_text(encoding="utf-8"))
    query = str(payload.get("query", "")).strip()
    history = payload.get("history", [])

    if not query:
        raise ValueError("input.json must contain a non-empty 'query'.")
    if not isinstance(history, list):
        raise TypeError("input.json 'history' must be a list.")

    return query, history


def main() -> None:
    query, history = load_input()
    model = ChatOpenAI(
        model=GENERATION_MODEL,
        verbosity="low",
        api_key=OPENAI_API_KEY,  # type: ignore[arg-type]
    )
    graph = RAGGraph(
        model=model,
        retriever=retrieve,
        catalogue_path=BOOK_CATALOGUE_PATH,
    )
    print(graph.invoke(query=query, history=history))


if __name__ == "__main__":
    main()
