"""Structured outputs used by graph nodes."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    final_query: str = Field(
        description=(
            "The original query unchanged when it is self-contained, otherwise a "
            "standalone query clarified using conversation history."
        )
    )
    route: Literal["catalogue", "rag_retrieval"]
