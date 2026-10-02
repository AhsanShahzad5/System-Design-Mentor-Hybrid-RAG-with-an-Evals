import re
from typing import Any

import pymupdf
import pymupdf4llm

from document_processing.schemas import Chapter

CHAPTER_PATTERNS = (
    re.compile(r"^Chapter\s+(\d+)[.:]\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(\d+)\.\s+(.+)$"),
)


def extract_chapters(pdf: pymupdf.Document) -> list[Chapter]:
    """Read numbered chapters from the PDF's embedded table of contents."""
    chapters: list[Chapter] = []

    for _level, title, start_page in pdf.get_toc():
        match = None
        for pattern in CHAPTER_PATTERNS:
            match = pattern.match(title)
            if match:
                break

        if not match:
            continue

        chapters.append(
            Chapter(
                number=int(match.group(1)),
                title=match.group(2).strip(),
                start_page=start_page,
            )
        )

    return chapters


def extract_markdown_pages(pdf: pymupdf.Document) -> list[dict[str, Any]]:
    """Extract layout-aware Markdown while preserving PDF page boundaries."""
    markdown_pages = pymupdf4llm.to_markdown(
        pdf,
        page_chunks=True,
        header=False,
        footer=False,
        use_ocr=False,
        show_progress=True,
    )

    if not isinstance(markdown_pages, list):
        raise TypeError("Expected page_chunks=True to return a list of pages")

    return markdown_pages


def chapter_for_page(chapters: list[Chapter], page_number: int) -> Chapter | None:
    """Return the chapter active on a 1-based PDF page."""
    current_chapter = None

    for chapter in chapters:
        if chapter.start_page > page_number:
            break
        current_chapter = chapter

    return current_chapter
