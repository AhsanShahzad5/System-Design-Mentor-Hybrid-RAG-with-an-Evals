import os

from pinecone import Pinecone


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_hybrid_index(
    pinecone: Pinecone,
    index_name: str,
    embedding_dimensions: int,
    cloud: str,
    region: str,
) -> None:
    existing_indexes = {index.name for index in pinecone.indexes.list().to_list()}

    if index_name not in existing_indexes:
        pinecone.indexes.create(
            name=index_name,
            schema={
                "fields": {
                    "embedding": {
                        "type": "dense_vector",
                        "dimension": embedding_dimensions,
                        "metric": "cosine",
                    },
                    "text": {
                        "type": "string",
                        "full_text_search": {"language": "en"},
                    },
                }
            },
            deployment={
                "deployment_type": "managed",
                "cloud": cloud,
                "region": region,
            },
        )
        return

    description = pinecone.indexes.describe(index_name)
    fields = description.schema.fields
    embedding_field = fields.get("embedding")
    text_field = fields.get("text")

    if (
        embedding_field is None
        or getattr(embedding_field, "dimension", None) != embedding_dimensions
        or getattr(embedding_field, "metric", None) != "cosine"
        or text_field is None
        or getattr(text_field, "full_text_search", None) is None
    ):
        raise ValueError(
            f"Pinecone index '{index_name}' is not compatible with the required "
            "dense + full-text-search schema. Use a new index name or recreate it."
        )
