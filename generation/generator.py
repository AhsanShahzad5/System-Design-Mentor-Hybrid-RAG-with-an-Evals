"""Grounded answer generation from already-retrieved context.

The generator never retrieves. It receives a query and context passages, so it
can be evaluated in isolation with golden context:

    result = generator.generate(query, documents)   # GenerationResult
    for token in generator.stream(query, documents):  # streaming / TTFT
        ...
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable

from generation.helpers import (
    format_retrieved_context,
    message_text,
    usage_from_message,
)
from generation.prompts import GENERATION_PROMPT

Context = Sequence[Document | str]

NO_CONTEXT_ANSWER = (
    "I couldn't find relevant material in the indexed books, so I can't answer "
    "this from the available context. Try making the question more specific or "
    "using terminology from the books."
)


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    usage: dict[str, object] = field(default_factory=dict)


class Generator:
    """Generate a grounded teaching answer from query and context."""

    def __init__(
        self,
        model: BaseChatModel,
        book_titles: Mapping[str, str] | None = None,
    ) -> None:
        self.prompt = GENERATION_PROMPT
        self._book_titles = dict(book_titles or {})
        self._chain = GENERATION_PROMPT | model

    def build_inputs(self, query: str, context: Context) -> dict[str, str]:
        """Return the exact prompt variables used for a generation call."""
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty.")
        if not context:
            raise ValueError("Generation requires at least one context passage.")

        return {
            "question": query,
            "context": format_retrieved_context(context, self._book_titles),
        }

    @traceable(name="generate-answer", run_type="chain")
    def generate(self, query: str, context: Context) -> GenerationResult:
        message = self._chain.invoke(self.build_inputs(query, context))
        answer = message_text(message).strip()
        if not answer:
            raise ValueError("Generation returned an empty answer.")
        return GenerationResult(answer=answer, usage=usage_from_message(message))

    def stream(self, query: str, context: Context) -> Iterator[str]:
        """Yield the answer text as it is produced, skipping empty chunks."""
        for chunk in self._chain.stream(self.build_inputs(query, context)):
            text = message_text(chunk)
            if text:
                yield text
