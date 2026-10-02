import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def upsert_documents(
    documents: Sequence[Document],
    embeddings: Embeddings,
    pinecone_index: Any,
    namespace: str,
    embedding_batch_size: int = 100,
    upsert_batch_size: int = 20,
) -> None:
    """Embed documents in batches and upsert them into Pinecone."""
    if embedding_batch_size <= 0 or upsert_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero.")

    total = len(documents)
    upserted = 0

    for start in range(0, total, embedding_batch_size):
        batch = documents[start : start + embedding_batch_size]
        vectors = embeddings.embed_documents(
            [document.page_content for document in batch]
        )

        records = [
            {
                "_id": str(document.metadata["chunk_id"]),
                "embedding": vector,
                "text": document.page_content,
                **document.metadata,
            }
            for document, vector in zip(batch, vectors, strict=True)
        ]

        for record_start in range(0, len(records), upsert_batch_size):
            record_batch = records[record_start : record_start + upsert_batch_size]
            pinecone_index.documents.upsert(
                namespace=namespace,
                documents=record_batch,
            )
            upserted += len(record_batch)
            logger.info("Upserted %d/%d chunks", upserted, total)
