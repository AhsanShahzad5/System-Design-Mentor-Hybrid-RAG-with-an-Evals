import logging

from document_processing import BOOKS, DocumentProcessingPipeline
from settings import get_settings

settings = get_settings()

PROCESSED_DOCUMENTS_DIRECTORY = settings.processed_documents_directory


def main():
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
