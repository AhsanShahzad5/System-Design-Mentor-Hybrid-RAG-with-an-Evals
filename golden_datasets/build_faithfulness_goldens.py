"""Build faithfulness_goldens.json from the evidence in retriever_goldens.json.

Each retriever golden points at the book section that answers it
(book_id, chapter_number, section_title, pages). This script copies those
exact passages out of processed_documents/ to use as the golden context for
generator-only (faithfulness) evals.

The output contains verbatim book text, so it is gitignored: rebuild it with

    uv run python golden_datasets/build_faithfulness_goldens.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from document_processing.document_store import document_path, load_documents  # noqa: E402
from settings import PROCESSED_DOCUMENTS_DIRECTORY  # noqa: E402

GOLDENS_DIRECTORY = Path(__file__).resolve().parent
RETRIEVER_GOLDENS = GOLDENS_DIRECTORY / "retriever_goldens.json"
OUTPUT = GOLDENS_DIRECTORY / "faithfulness_goldens.json"


def resolve_evidence(evidence: dict, documents_by_book: dict) -> list[str]:
    """Return the passages in the evidence section on the evidence pages."""
    book_id = evidence["book_id"]
    if book_id not in documents_by_book:
        path = document_path(PROCESSED_DOCUMENTS_DIRECTORY, book_id)
        documents_by_book[book_id] = load_documents(path)

    pages = set(evidence["pages"])
    return [
        document.page_content.strip()
        for document in documents_by_book[book_id]
        if document.metadata.get("chapter_number") == evidence["chapter_number"]
        and document.metadata.get("section_title") == evidence["section_title"]
        and document.metadata.get("page_number") in pages
    ]


def main() -> None:
    goldens = json.loads(RETRIEVER_GOLDENS.read_text(encoding="utf-8"))
    documents_by_book: dict = {}
    rows, missing = [], []

    for golden in goldens:
        passages = resolve_evidence(golden["evidence"], documents_by_book)
        if not passages:
            missing.append(golden["id"])
            continue
        rows.append(
            {
                "id": golden["id"].replace("r", "f", 1),
                "query": golden["query"],
                "ideal_context": passages,
                "evidence": golden["evidence"],
            }
        )

    if missing:
        raise SystemExit(f"Evidence matched no passages for: {missing}")

    OUTPUT.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(len(row["ideal_context"]) for row in rows)
    print(f"Wrote {len(rows)} goldens ({total} passages) -> {OUTPUT}")


if __name__ == "__main__":
    main()
