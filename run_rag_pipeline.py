"""Run the LangGraph book mentor using query and history from input.json."""

import json

from app import build_app
from rag import ChatMessage
from settings import get_settings


def load_input() -> tuple[str, list[ChatMessage]]:
    input_path = get_settings().rag_input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"RAG input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    query = str(payload.get("query", "")).strip()
    history = payload.get("history", [])

    if not query:
        raise ValueError("input.json must contain a non-empty 'query'.")
    if not isinstance(history, list):
        raise TypeError("input.json 'history' must be a list.")

    return query, history


def main() -> None:
    query, history = load_input()
    result = build_app().invoke(query=query, history=history)

    print(f"route       : {result['route']}")
    print(f"final query : {result['final_query']}")
    print(f"\n{result['answer']}")

    if result["documents"]:
        print("\nSources:")
        for rank, document in enumerate(result["documents"], start=1):
            meta = document.metadata
            print(
                f"  [{rank}] {meta.get('book_id')} | {meta.get('chapter_title')} | "
                f"{meta.get('section_title')} | p.{meta.get('page_number')}"
            )


if __name__ == "__main__":
    main()
