"""Run hybrid retrieval + reranking for one query and print the results."""

import logging

from app import build_retriever
from settings import get_settings

QUERY = "How should an ML system detect and respond to model drift?"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    retriever = build_retriever(get_settings())
    documents = retriever.retrieve(QUERY)
    for rank, document in enumerate(documents, start=1):
        print(f"\n--- Result {rank} ---")
        print(document.metadata)
        print(document.page_content)


if __name__ == "__main__":
    main()
