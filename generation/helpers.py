"""Formatting helpers for generation inputs and model outputs."""

from collections.abc import Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def _source_label(
    source_number: int,
    metadata: Mapping[str, object],
    book_titles: Mapping[str, str],
) -> str:
    parts = [f"Source {source_number}"]

    book_id = metadata.get("book_id")
    if book_id:
        parts.append(book_titles.get(str(book_id), str(book_id)))

    # chapter_title is stored as "Chapter N. Title", so it is used as-is.
    chapter_title = metadata.get("chapter_title")
    chapter_number = metadata.get("chapter_number")
    if chapter_title:
        parts.append(str(chapter_title))
    elif chapter_number is not None:
        parts.append(f"Chapter {int(float(str(chapter_number)))}")

    section_title = metadata.get("section_title")
    if section_title and section_title != chapter_title:
        parts.append(str(section_title))

    return " — ".join(parts).replace('"', "'")


def format_retrieved_context(
    context: Sequence[Document | str],
    book_titles: Mapping[str, str] | None = None,
) -> str:
    """Create clearly delimited, citable context for the generation model.

    Plain strings are accepted so evals can pass golden context passages.
    """
    titles = book_titles or {}
    sources = []

    for source_number, item in enumerate(context, start=1):
        document = item if isinstance(item, Document) else Document(page_content=item)
        label = _source_label(source_number, document.metadata, titles)
        sources.append(
            f'<source label="{label}">\n{document.page_content.strip()}\n</source>'
        )

    return "\n\n".join(sources)


def message_text(message: BaseMessage) -> str:
    """Return the text of a chat message or streamed chunk."""
    return message.text


def usage_from_message(message: BaseMessage) -> dict[str, object]:
    """Extract token usage and the served model name from a model response."""
    usage: dict[str, object] = dict(getattr(message, "usage_metadata", None) or {})
    model_name = message.response_metadata.get("model_name")
    if model_name:
        usage["model"] = model_name
    return usage
