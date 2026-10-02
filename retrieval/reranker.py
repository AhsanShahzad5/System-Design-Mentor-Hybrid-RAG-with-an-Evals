from langchain_core.documents import Document
from langsmith import traceable
from pinecone import Pinecone


@traceable(name="rerank-documents", run_type="retriever")
def rerank_documents(
    query: str,
    documents: list[Document],
    pinecone: Pinecone,
    model: str,
    top_n: int,
) -> list[Document]:
    if not documents:
        return []

    result = pinecone.inference.rerank(
        model=model,
        query=query,
        documents=[document.page_content for document in documents],
        rank_fields=["text"],
        top_n=min(top_n, len(documents)),
        return_documents=False,
    )

    reranked: list[Document] = []
    for ranked_document in result.data:
        document = documents[ranked_document.index]
        reranked.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rerank_score": ranked_document.score,
                    "rerank_model": result.model,
                },
            )
        )

    return reranked
