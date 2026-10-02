from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import traceable
from pinecone import Pinecone

from retrieval.bm25_retriever import retrieve_bm25
from retrieval.dense_retriever import retrieve_dense
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank_documents


class RetrievalPipeline:
    """Run dense and BM25 retrieval, RRF, and final reranking."""

    def __init__(
        self,
        embeddings: Embeddings,
        pinecone: Pinecone,
        pinecone_index: Any,
        namespace: str,
        rerank_model: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        fused_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._embeddings = embeddings
        self._pinecone = pinecone
        self._pinecone_index = pinecone_index
        self._namespace = namespace
        self._rerank_model = rerank_model
        self._dense_top_k = dense_top_k
        self._bm25_top_k = bm25_top_k
        self._fused_top_k = fused_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k

    @traceable(name="retrieval-pipeline", run_type="chain")
    def retrieve(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("The retrieval query cannot be empty.")

        dense_retrieval = partial(
            retrieve_dense,
            query=query,
            embeddings=self._embeddings,
            pinecone_index=self._pinecone_index,
            namespace=self._namespace,
            top_k=self._dense_top_k,
        )
        bm25_retrieval = partial(
            retrieve_bm25,
            query=query,
            pinecone_index=self._pinecone_index,
            namespace=self._namespace,
            top_k=self._bm25_top_k,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(copy_context().run, dense_retrieval)
            bm25_future = executor.submit(copy_context().run, bm25_retrieval)

            dense_documents = dense_future.result()
            bm25_documents = bm25_future.result()

        fused_documents = reciprocal_rank_fusion(
            ranked_lists=[dense_documents, bm25_documents],
            top_k=self._fused_top_k,
            rrf_k=self._rrf_k,
        )
        return rerank_documents(
            query=query,
            documents=fused_documents,
            pinecone=self._pinecone,
            model=self._rerank_model,
            top_n=self._final_top_k,
        )
