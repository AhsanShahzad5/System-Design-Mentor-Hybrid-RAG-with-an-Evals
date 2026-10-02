from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pinecone import DenseVectorQuery, TextQuery

RETURN_FIELDS = [
    "text",
    "book_id",
    "page_number",
    "chapter_number",
    "chapter_title",
    "section_title",
    "chunk_id",
    "chunk_index",
]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse Pinecone result lists without comparing incompatible raw scores."""
    fused: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, match in enumerate(ranked_list, start=1):
            record = match.to_dict()
            record_id = str(record["_id"])
            entry = fused.setdefault(
                record_id,
                {"score": 0.0, "record": record},
            )
            entry["score"] += 1 / (rrf_k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    documents: list[Document] = []
    for item in ordered:
        record = dict(item["record"])
        page_content = str(record.pop("text"))
        record_id = str(record.pop("_id"))
        record.pop("_score", None)
        record["pinecone_id"] = record_id
        record["rrf_score"] = item["score"]
        documents.append(Document(page_content=page_content, metadata=record))

    return documents


def hybrid_search(
    query: str,
    embeddings: Embeddings,
    pinecone_index: Any,
    namespace: str,
    top_k: int = 10,
    candidates_per_method: int = 20,
) -> list[Document]:
    """Run dense and BM25 searches, then combine their rankings with RRF."""
    query_embedding = embeddings.embed_query(query)

    dense_results = pinecone_index.documents.search(
        namespace=namespace,
        top_k=candidates_per_method,
        score_by=[
            DenseVectorQuery(field="embedding", values=query_embedding),
        ],
        include_fields=RETURN_FIELDS,
    )
    bm25_results = pinecone_index.documents.search(
        namespace=namespace,
        top_k=candidates_per_method,
        score_by=[
            TextQuery(query=query, fields=["text"]),
        ],
        include_fields=RETURN_FIELDS,
    )

    return reciprocal_rank_fusion(
        ranked_lists=[dense_results.matches, bm25_results.matches],
        top_k=top_k,
    )
