from collections.abc import Sequence

from langchain_core.documents import Document
from langsmith import traceable


@traceable(name="reciprocal-rank-fusion", run_type="chain")
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse ranked lists without comparing their incompatible raw scores."""
    fused: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            document_id = str(document.metadata["pinecone_id"])
            entry = fused.setdefault(
                document_id,
                {
                    "document": document,
                    "score": 0.0,
                    "metadata": dict(document.metadata),
                },
            )
            entry["score"] += 1 / (rrf_k + rank)
            entry["metadata"].update(document.metadata)

    ordered = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    return [
        Document(
            page_content=item["document"].page_content,
            metadata={**item["metadata"], "rrf_score": item["score"]},
        )
        for item in ordered
    ]
