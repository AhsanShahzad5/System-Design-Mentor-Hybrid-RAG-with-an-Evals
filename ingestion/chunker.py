import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Merged sections shorter than this are fragments (lone headings, stray captions).
MIN_SECTION_CHARS = 100

_SECTION_KEYS = ("book_id", "chapter_number", "section_title")


def build_text_splitter(
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    """Create the default splitter for heading-aware book documents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size.")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def merge_section_documents(
    documents: Sequence[Document],
    min_chars: int = MIN_SECTION_CHARS,
) -> list[Document]:
    """Join consecutive page documents of the same section into one document.

    Page-level documents split a section at every page break, which leaves
    small fragments. Merging first lets the splitter cut by size across page
    breaks. Metadata keeps the first page as page_number plus page_end.
    Sections still shorter than min_chars after merging are dropped.
    """
    merged: list[Document] = []
    current: Document | None = None

    def key(document: Document) -> tuple:
        return tuple(document.metadata.get(k) for k in _SECTION_KEYS)

    for document in documents:
        if current is not None and key(document) == key(current):
            current.page_content += "\n\n" + document.page_content
            current.metadata["page_end"] = document.metadata["page_number"]
            continue
        if current is not None:
            merged.append(current)
        metadata = dict(document.metadata)
        metadata["page_end"] = metadata["page_number"]
        current = Document(page_content=document.page_content, metadata=metadata)

    if current is not None:
        merged.append(current)

    return [d for d in merged if len(d.page_content.strip()) >= min_chars]


def context_header(metadata: Mapping[str, object], book_titles: Mapping[str, str]) -> str:
    """'Book › Chapter › Section' so search and rerank see where a chunk lives."""
    book_id = str(metadata.get("book_id", ""))
    parts = [book_titles.get(book_id, book_id)]
    for key in ("chapter_title", "section_title"):
        value = metadata.get(key)
        if value and value not in parts:
            parts.append(str(value))
    return " › ".join(part for part in parts if part)


def chunk_documents(
    documents: Sequence[Document],
    text_splitter: RecursiveCharacterTextSplitter,
    book_titles: Mapping[str, str] | None = None,
) -> list[Document]:
    """Merge sections, split, prepend a context header, and attach stable IDs.

    The chunk ID hashes the chunk's location and text (not its position in
    the corpus), so a change elsewhere in a book does not renumber every
    later chunk. Identical chunks in one book get an occurrence suffix.
    """
    titles = book_titles or {}
    sections = merge_section_documents(documents)
    chunks = text_splitter.split_documents(sections)

    seen: Counter[str] = Counter()
    per_book_index: Counter[str] = Counter()

    for chunk in chunks:
        metadata = chunk.metadata
        book_id = str(metadata["book_id"])
        header = context_header(metadata, titles)
        if header:
            chunk.page_content = f"{header}\n\n{chunk.page_content}"

        identity = "|".join(
            str(metadata.get(k, "")) for k in (*_SECTION_KEYS, "page_number")
        ) + "|" + chunk.page_content
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        seen[digest] += 1
        chunk_id = digest if seen[digest] == 1 else f"{digest}-{seen[digest]}"

        metadata["chunk_id"] = chunk_id
        metadata["chunk_index"] = per_book_index[book_id]
        per_book_index[book_id] += 1

    return chunks
