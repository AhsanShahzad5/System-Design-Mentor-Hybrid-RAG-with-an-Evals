import logging

from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pinecone import Pinecone

from document_processing import BOOKS
from document_processing.document_store import document_path
from ingestion import IngestionPipeline, build_text_splitter
from ingestion.helpers import ensure_hybrid_index
from settings import get_settings

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
PINECONE_API_KEY = settings.pinecone_api_key.get_secret_value()
PINECONE_INDEX_NAME = settings.pinecone_index_name
PINECONE_NAMESPACE = settings.pinecone_namespace
PINECONE_CLOUD = settings.pinecone_cloud
PINECONE_REGION = settings.pinecone_region
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIMENSIONS = settings.embedding_dimensions
EMBEDDING_BATCH_SIZE = settings.embedding_batch_size
PINECONE_UPSERT_BATCH_SIZE = settings.pinecone_upsert_batch_size
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
PROCESSED_DOCUMENTS_DIRECTORY = settings.processed_documents_directory


@traceable(name="book-ingestion-pipeline", run_type="chain")
def main():
    """Run document ingestion for every configured book."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pinecone_api_key = settings.pinecone_api_key.get_secret_value()
    pinecone_index_name = settings.pinecone_index_name

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=settings.openai_api_key,
    )
    pinecone = Pinecone(api_key=pinecone_api_key)
    ensure_hybrid_index(
        pinecone=pinecone,
        index_name=pinecone_index_name,
        embedding_dimensions=settings.embedding_dimensions,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
    )

    pipeline = IngestionPipeline(
        document_paths=[
            document_path(PROCESSED_DOCUMENTS_DIRECTORY, book.book_id) for book in BOOKS
        ],
        text_splitter=build_text_splitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ),
        embeddings=embeddings,
        pinecone_index=pinecone.index(name=pinecone_index_name),
        namespace=PINECONE_NAMESPACE,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        upsert_batch_size=PINECONE_UPSERT_BATCH_SIZE,
    )

    # for langsmith tracing
    chunks = pipeline.run()
    return {
        "index_name": pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "chunks_ingested": len(chunks),
    }


if __name__ == "__main__":
    main()
