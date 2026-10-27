import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processing.document_store import load_documents
from ingestion.chunker import chunk_documents
from ingestion.vector_store import upsert_documents

logger = logging.getLogger(__name__)

DocumentLoader = Callable[[Path], list[Document]]


class IngestionPipeline:
    """Load processed documents, chunk them, and store vectors in Pinecone."""

    def __init__(
        self,
        document_paths: Iterable[Path],
        text_splitter: RecursiveCharacterTextSplitter,
        embeddings: Embeddings,
        pinecone_index: object,
        namespace: str,
        embedding_batch_size: int = 100,
        upsert_batch_size: int = 20,
        document_loader: DocumentLoader = load_documents,
        book_titles: Mapping[str, str] | None = None,
    ) -> None:
        self._document_paths = tuple(document_paths)
        self._text_splitter = text_splitter
        self._embeddings = embeddings
        self._pinecone_index = pinecone_index
        self._namespace = namespace
        self._embedding_batch_size = embedding_batch_size
        self._upsert_batch_size = upsert_batch_size
        self._document_loader = document_loader
        self._book_titles = dict(book_titles or {})

    def validate(self) -> None:
        """Fail before processing if configuration or input files are invalid."""
        if not self._document_paths:
            raise ValueError("At least one processed document file is required.")

        missing_files = [
            str(path) for path in self._document_paths if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing processed document files: {missing_files}"
            )

    def load_book_documents(self, path: Path) -> list[Document]:
        logger.info("Loading processed documents: %s", path)
        documents = self._document_loader(path)
        logger.info("Loaded %d documents from %s", len(documents), path)
        return documents

    @staticmethod
    def merge_documents(
        documents_by_book: Mapping[str, list[Document]],
    ) -> list[Document]:
        """Combine per-book collections while preserving configured book order."""
        return [
            document
            for book_documents in documents_by_book.values()
            for document in book_documents
        ]

    def run(self) -> list[Document]:
        """Load, merge, chunk, embed, and store all processed books."""
        self.validate()

        documents_by_book = {
            path.stem: self.load_book_documents(path)
            for path in self._document_paths
        }
        all_documents = self.merge_documents(documents_by_book)
        chunks = chunk_documents(
            all_documents, self._text_splitter, self._book_titles
        )

        logger.info(
            "Built %d documents and %d chunks from %d books",
            len(all_documents),
            len(chunks),
            len(documents_by_book),
        )
        upsert_documents(
            documents=chunks,
            embeddings=self._embeddings,
            pinecone_index=self._pinecone_index,
            namespace=self._namespace,
            embedding_batch_size=self._embedding_batch_size,
            upsert_batch_size=self._upsert_batch_size,
        )
        logger.info("Ingestion complete")
        return chunks
