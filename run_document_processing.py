import logging
from pathlib import Path

from document_processing import BOOKS, DocumentProcessingPipeline

PROCESSED_DOCUMENTS_DIRECTORY = Path("processed_documents")


def main() -> dict[str, Path]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pipeline = DocumentProcessingPipeline(
        books=BOOKS,
        output_directory=PROCESSED_DOCUMENTS_DIRECTORY,
    )
    return pipeline.run()


if __name__ == "__main__":
    main()
