from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent

# Export .env to os.environ so libraries that read it directly (LangSmith
# tracing, OpenAI/Pinecone SDK defaults) work from every entry point.
load_dotenv(PROJECT_ROOT / ".env")
BOOKS_DIRECTORY = PROJECT_ROOT / "books_data"
PROCESSED_DOCUMENTS_DIRECTORY = PROJECT_ROOT / "processed_documents"
BOOK_CATALOGUE_PATH = PROJECT_ROOT / "rag" / "data" / "book_catalogue.json"
RAG_INPUT_PATH = PROJECT_ROOT / "input.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # External services
    openai_api_key: SecretStr
    pinecone_api_key: SecretStr
    pinecone_index_name: str

    # Project paths
    books_directory: Path = BOOKS_DIRECTORY
    processed_documents_directory: Path = PROCESSED_DOCUMENTS_DIRECTORY
    book_catalogue_path: Path = BOOK_CATALOGUE_PATH
    rag_input_path: Path = RAG_INPUT_PATH

    # Pinecone
    pinecone_namespace: str = "books"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings and ingestion
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 100
    pinecone_upsert_batch_size: int = 20

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # Retrieval
    dense_top_k: int = 20
    bm25_top_k: int = 20
    fused_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60

    # Reranking
    rerank_model: str = "bge-reranker-v2-m3"
    generation_model: str = "gpt-5.6-luna"

    # Evals: LLM-as-judge model (kept separate from the generation model)
    eval_judge_model: str = "gpt-4.1-mini"
    eval_threshold: float = 0.7
    eval_max_workers: int = 4  # concurrent app calls while building test cases


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
