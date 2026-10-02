from typing import Any

from langchain_core.documents import Document
from langsmith import traceable
from pinecone import TextQuery

from retrieval.dense_retriever import RETURN_FIELDS


@traceable(name="bm25-retrieval", run_type="retriever")
def retrieve_bm25(
    query: str,
    pinecone_index: Any,
    namespace: str,
    top_k: int,
) -> list[Document]:
    response = pinecone_index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            TextQuery(query=query, fields=["text"]),
        ],
        include_fields=RETURN_FIELDS,
    )

    documents: list[Document] = []
    for match in response.matches:
        record = match.to_dict()
        page_content = str(record.pop("text"))
        record["pinecone_id"] = str(record.pop("_id"))
        record["bm25_score"] = float(record.pop("_score"))
        documents.append(Document(page_content=page_content, metadata=record))

    return documents
