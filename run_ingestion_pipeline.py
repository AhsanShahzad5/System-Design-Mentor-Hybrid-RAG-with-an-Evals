import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pinecone import Pinecone

from document_processing import BOOKS
from document_processing.document_store import document_path
from ingestion import IngestionPipeline, build_text_splitter
from ingestion.helpers import ensure_hybrid_index, require_environment_variable

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 100
PINECONE_UPSERT_BATCH_SIZE = 20
PINECONE_NAMESPACE = "books"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PROCESSED_DOCUMENTS_DIRECTORY = Path("processed_documents")

load_dotenv()


@traceable(name="book-ingestion-pipeline", run_type="chain")
def main():
    """Run document ingestion for every configured book."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pinecone_api_key = require_environment_variable("PINECONE_API_KEY")
    pinecone_index_name = require_environment_variable("PINECONE_INDEX_NAME")
    require_environment_variable("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    pinecone = Pinecone(api_key=pinecone_api_key)
    ensure_hybrid_index(
        pinecone=pinecone,
        index_name=pinecone_index_name,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
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
        "namespace": PINECONE_NAMESPACE,
        "chunks_ingested": len(chunks),
    }


if __name__ == "__main__":
    main()
