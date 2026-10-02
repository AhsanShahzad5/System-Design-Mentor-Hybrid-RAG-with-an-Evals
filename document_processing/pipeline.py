import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.documents import Document

from document_processing.document_builder import load_book_documents
from document_processing.document_store import document_path, save_documents
from document_processing.schemas import BookConfig

logger = logging.getLogger(__name__)

DocumentBuilder = Callable[[BookConfig], list[Document]]


class DocumentProcessingPipeline:
    """Convert source PDFs into reusable per-book document artifacts."""

    def __init__(
        self,
        books: Iterable[BookConfig],
        output_directory: Path,
        document_builder: DocumentBuilder = load_book_documents,
    ) -> None:
        self._books = tuple(books)
        self._output_directory = output_directory
        self._document_builder = document_builder

    def validate(self) -> None:
        if not self._books:
            raise ValueError("At least one book must be configured.")

        book_ids = [book.book_id for book in self._books]
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("Every book must have a unique book_id.")

        missing_files = [
            str(book.pdf_path) for book in self._books if not book.pdf_path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"Missing PDF files: {missing_files}")

    def run(self, overwrite: bool = False) -> dict[str, Path]:
        """Process books and save one JSONL artifact per book."""
        self.validate()
        artifacts: dict[str, Path] = {}

        for book in self._books:
            output_path = document_path(self._output_directory, book.book_id)
            artifacts[book.book_id] = output_path

            if output_path.exists() and not overwrite:
                logger.info("Skipping existing artifact: %s", output_path)
                continue

            logger.info("Processing book: %s", book.book_id)
            documents = self._document_builder(book)
            save_documents(output_path, documents)
            logger.info("Saved %d documents to %s", len(documents), output_path)

        return artifacts
