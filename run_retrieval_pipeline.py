import logging

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pinecone import Pinecone

from ingestion.helpers import require_environment_variable
from retrieval import RetrievalPipeline

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
RERANK_MODEL = "bge-reranker-v2-m3"
PINECONE_NAMESPACE = "books"
DENSE_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 20
FINAL_TOP_K = 5

load_dotenv()


@traceable(name="hybrid-retrieval-pipeline", run_type="retriever")
def retrieve(query: str):
    pinecone_api_key = require_environment_variable("PINECONE_API_KEY")
    pinecone_index_name = require_environment_variable("PINECONE_INDEX_NAME")
    require_environment_variable("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    pinecone = Pinecone(api_key=pinecone_api_key)
    pipeline = RetrievalPipeline(
        embeddings=embeddings,
        pinecone=pinecone,
        pinecone_index=pinecone.index(name=pinecone_index_name),
        namespace=PINECONE_NAMESPACE,
        rerank_model=RERANK_MODEL,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        fused_top_k=FUSED_TOP_K,
        final_top_k=FINAL_TOP_K,
    )
    return pipeline.retrieve(query)


QUERY = "How should an ML system detect and respond to model drift?"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    documents = retrieve(QUERY)
    for rank, document in enumerate(documents, start=1):
        print(f"\n--- Result {rank} ---")
        print(document.metadata)
        print(document.page_content)


if __name__ == "__main__":
    main()
