from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import traceable
from pinecone import DenseVectorQuery

RETURN_FIELDS = [
    "text",
    "book_id",
    "page_number",
    "page_end",
    "chapter_number",
    "chapter_title",
    "section_title",
    "chunk_id",
    "chunk_index",
]


@traceable(name="dense-retrieval", run_type="retriever")
def retrieve_dense(
    query: str,
    embeddings: Embeddings,
    pinecone_index: Any,
    namespace: str,
    top_k: int,
) -> list[Document]:
    query_embedding = embeddings.embed_query(query)
    response = pinecone_index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            DenseVectorQuery(field="embedding", values=query_embedding),
        ],
        include_fields=RETURN_FIELDS,
    )

    documents: list[Document] = []
    for match in response.matches:
        record = match.to_dict()
        page_content = str(record.pop("text"))
        record["pinecone_id"] = str(record.pop("_id"))
        record["dense_score"] = float(record.pop("_score"))
        documents.append(Document(page_content=page_content, metadata=record))

    return documents
