import hashlib
from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def chunk_documents(
    documents: Sequence[Document],
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split documents and attach deterministic IDs and per-book indexes."""
    chunks = text_splitter.split_documents(list(documents))
    next_index_by_book: defaultdict[str, int] = defaultdict(int)

    for chunk in chunks:
        book_id = str(chunk.metadata["book_id"])
        chunk_index = next_index_by_book[book_id]
        next_index_by_book[book_id] += 1

        identity = "|".join(
            (
                book_id,
                str(chunk_index),
                chunk.page_content,
            )
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = chunk_id

    return chunks
