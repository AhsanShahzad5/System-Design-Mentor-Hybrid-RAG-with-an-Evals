"""Apply the current cleaning rules to an existing JSONL artifact, without
re-parsing the PDF.

PDF parsing is the expensive stage (image-heavy books can take hours). The
rules added after parsing work on documents just as well, so an old artifact
can be upgraded in seconds:
  - skip excluded front/back matter pages (TOC-based)
  - recompute chapter metadata from TOC chapter ranges (fixes back matter
    tagged as the last chapter)
  - drop reference/bibliography sections
  - clean extraction artifacts from text and section titles

    uv run python -m document_processing.migrate SRC_DIR DST_DIR [book_id ...]
"""

import sys
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from document_processing.book_configs import BOOKS
from document_processing.document_builder import REFERENCE_SECTION
from document_processing.document_store import document_path, load_documents, save_documents
from document_processing.pdf_parser import chapter_for_page, excluded_pages, extract_chapters
from document_processing.schemas import BookConfig
from document_processing.text_cleaning import clean_markdown, clean_title


def migrate_documents(documents: list[Document], config: BookConfig) -> list[Document]:
    with pymupdf.open(config.pdf_path) as pdf:
        chapters = extract_chapters(pdf)
        skip_pages = excluded_pages(pdf)

    migrated: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        page_number = int(metadata["page_number"])
        if page_number in skip_pages:
            continue

        section_title = metadata.get("section_title")
        if section_title is not None:
            section_title = clean_title(str(section_title))
            if not section_title or REFERENCE_SECTION.match(section_title):
                continue
            metadata["section_title"] = section_title

        metadata.pop("chapter_number", None)
        metadata.pop("chapter_title", None)
        chapter = chapter_for_page(chapters, page_number)
        if chapter is not None:
            metadata["chapter_number"] = chapter.number
            metadata["chapter_title"] = f"Chapter {chapter.number}. {chapter.title}"

        content = clean_markdown(document.page_content)
        if content:
            migrated.append(Document(page_content=content, metadata=metadata))
    return migrated


def main() -> None:
    source, destination, *book_ids = sys.argv[1:]
    books = [b for b in BOOKS if not book_ids or b.book_id in book_ids]
    for book in books:
        documents = load_documents(document_path(Path(source), book.book_id))
        migrated = migrate_documents(documents, book)
        save_documents(document_path(Path(destination), book.book_id), migrated)
        print(f"{book.book_id}: {len(documents)} -> {len(migrated)} documents")


if __name__ == "__main__":
    main()
