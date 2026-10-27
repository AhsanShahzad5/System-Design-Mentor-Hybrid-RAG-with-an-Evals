import re
from typing import Any

import pymupdf
import pymupdf4llm

from document_processing.schemas import Chapter

CHAPTER_PATTERNS = (
    re.compile(r"^Chapter\s+(\d+)[.:]\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(\d+)\.\s+(.+)$"),
)

# Front and back matter that adds noise to retrieval (keyword-dense index pages,
# legal text, author bios). Prefaces, glossaries and epilogues are kept.
EXCLUDED_TOC_TITLES = re.compile(
    r"^(copyright|table of contents|contents|index|about the authors?|colophon|"
    r"o[’']reilly safari|how to contact us)\b",
    re.IGNORECASE,
)


def _match_chapter(title: str) -> re.Match[str] | None:
    for pattern in CHAPTER_PATTERNS:
        match = pattern.match(title)
        if match:
            return match
    return None


def _chapter_level(toc: list[list[Any]]) -> int | None:
    levels = [level for level, title, _page in toc if _match_chapter(title)]
    return min(levels) if levels else None


def extract_chapters(pdf: pymupdf.Document) -> list[Chapter]:
    """Read numbered chapters and their page ranges from the embedded TOC.

    A chapter ends where the next TOC entry at the same or a higher level
    begins (the next chapter, a new part, an epilogue, the index...), so back
    matter is not mistaken for the last chapter.
    """
    toc = pdf.get_toc()
    chapters: list[Chapter] = []

    for index, (level, title, start_page) in enumerate(toc):
        match = _match_chapter(title)
        if not match:
            continue

        end_page = next(
            (page for next_level, _t, page in toc[index + 1 :] if next_level <= level),
            None,
        )
        chapters.append(
            Chapter(
                number=int(match.group(1)),
                title=match.group(2).strip(),
                start_page=start_page,
                end_page=end_page,
            )
        )

    return chapters


def excluded_pages(pdf: pymupdf.Document) -> set[int]:
    """1-based pages to skip: before the first TOC entry, or under excluded entries."""
    toc = pdf.get_toc()
    level = _chapter_level(toc)
    if level is None:
        return set()

    regions = [(title, page) for entry_level, title, page in toc if entry_level <= level]
    if not regions:
        return set()

    excluded = set(range(1, regions[0][1]))
    for index, (title, start_page) in enumerate(regions):
        if not EXCLUDED_TOC_TITLES.match(title.strip()):
            continue
        end_page = regions[index + 1][1] if index + 1 < len(regions) else pdf.page_count + 1
        excluded.update(range(start_page, max(end_page, start_page + 1)))
    return excluded


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
    """Return the chapter covering a 1-based PDF page, if any."""
    for chapter in chapters:
        in_range = chapter.start_page <= page_number and (
            chapter.end_page is None or page_number < chapter.end_page
        )
        if in_range:
            return chapter
    return None
