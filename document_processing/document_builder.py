import re
from typing import Any

import pymupdf
from langchain_core.documents import Document

from document_processing.pdf_parser import (
    chapter_for_page,
    extract_chapters,
    extract_markdown_pages,
)
from document_processing.schemas import BookConfig, Chapter

HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def clean_heading(title: str) -> str:
    """Remove Markdown emphasis characters surrounding a heading."""
    return title.strip().strip("*_` ")


def append_document(
    documents: list[Document],
    lines: list[str],
    book_id: str,
    page_number: int,
    chapter: Chapter | None,
    section_title: str | None,
) -> None:
    """Create a LangChain document when accumulated content is non-empty."""
    content = "\n".join(lines).strip()
    if not content:
        return

    metadata: dict[str, str | int] = {
        "book_id": book_id,
        "page_number": page_number,
    }

    if chapter is not None:
        metadata["chapter_number"] = chapter.number
        metadata["chapter_title"] = f"Chapter {chapter.number}. {chapter.title}"

    if section_title is not None:
        metadata["section_title"] = section_title

    documents.append(Document(page_content=content, metadata=metadata))


def markdown_pages_to_documents(
    markdown_pages: list[dict[str, Any]],
    chapters: list[Chapter],
    config: BookConfig,
) -> list[Document]:
    """Convert Markdown pages into page-bounded, heading-aware documents."""
    documents: list[Document] = []
    active_section: str | None = None
    previous_chapter_number: int | None = None

    for page_data in markdown_pages:
        page_number = int(page_data["metadata"]["page_number"])
        markdown = str(page_data["text"]).strip()

        if not markdown:
            continue

        chapter = chapter_for_page(chapters, page_number)

        if chapter is not None and chapter.number != previous_chapter_number:
            active_section = None
            previous_chapter_number = chapter.number

        section_lines: list[str] = []

        for line in markdown.splitlines():
            heading_match = HEADING_PATTERN.match(line.strip())

            if heading_match is None:
                section_lines.append(line)
                continue

            append_document(
                documents=documents,
                lines=section_lines,
                book_id=config.book_id,
                page_number=page_number,
                chapter=chapter,
                section_title=active_section,
            )
            section_lines = []

            heading_title = clean_heading(heading_match.group(1))
            is_chapter_heading = chapter is not None and (
                heading_title == chapter.title or heading_title.endswith(chapter.title)
            )

            if is_chapter_heading:
                active_section = None
                assert chapter is not None
                section_lines.append(f"# Chapter {chapter.number}. {chapter.title}")
                continue

            override_key = (
                chapter.number if chapter is not None else 0,
                heading_title,
            )
            active_section = config.section_title_overrides.get(
                override_key,
                heading_title,
            )
            section_lines.append(f"## {active_section}")

        append_document(
            documents=documents,
            lines=section_lines,
            book_id=config.book_id,
            page_number=page_number,
            chapter=chapter,
            section_title=active_section,
        )

    return documents


def load_book_documents(config: BookConfig) -> list[Document]:
    """Extract one book into page-bounded, heading-aware LangChain documents."""
    if not config.pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {config.pdf_path}")

    with pymupdf.open(config.pdf_path) as pdf:
        chapters = extract_chapters(pdf)
        markdown_pages = extract_markdown_pages(pdf)

    return markdown_pages_to_documents(markdown_pages, chapters, config)
