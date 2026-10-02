"""Formatting helpers for retrieved LangChain documents."""

from langchain_core.documents import Document


def _source_label(document: Document, source_number: int) -> str:
    metadata = document.metadata
    parts = [f"Source {source_number}"]

    book = metadata.get("book_title") or metadata.get("book_id")
    chapter_number = metadata.get("chapter_number")
    chapter_title = metadata.get("chapter_title")
    section_title = metadata.get("section_title")

    if book:
        parts.append(str(book))

    if chapter_number and chapter_title:
        parts.append(f"Chapter {chapter_number}: {chapter_title}")
    elif chapter_title:
        parts.append(str(chapter_title))
    elif chapter_number:
        parts.append(f"Chapter {chapter_number}")

    if section_title and section_title != chapter_title:
        parts.append(str(section_title))

    return " — ".join(parts)


def format_retrieved_context(documents: list[Document]) -> str:
    """Create clearly delimited, citable context for the generation model."""
    sources = []

    for source_number, document in enumerate(documents, start=1):
        label = _source_label(document, source_number)
        sources.append(
            f"<source label=\"{label}\">\n"
            f"{document.page_content.strip()}\n"
            "</source>"
        )

    return "\n\n".join(sources)
