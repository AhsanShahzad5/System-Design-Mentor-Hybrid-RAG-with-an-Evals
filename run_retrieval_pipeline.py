import logging

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pinecone import Pinecone

from retrieval import RetrievalPipeline
from settings import get_settings

load_dotenv()

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
PINECONE_API_KEY = settings.pinecone_api_key.get_secret_value()
PINECONE_INDEX_NAME = settings.pinecone_index_name
PINECONE_NAMESPACE = settings.pinecone_namespace
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIMENSIONS = settings.embedding_dimensions
DENSE_TOP_K = settings.dense_top_k
BM25_TOP_K = settings.bm25_top_k
FUSED_TOP_K = settings.fused_top_k
FINAL_TOP_K = settings.final_top_k
RRF_K = settings.rrf_k
RERANK_MODEL = settings.rerank_model


@traceable(name="hybrid-retrieval-pipeline", run_type="retriever")
def retrieve(query: str):
    pinecone_api_key = settings.pinecone_api_key.get_secret_value()
    pinecone_index_name = settings.pinecone_index_name

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=settings.openai_api_key,
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
        rrf_k=RRF_K,
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
