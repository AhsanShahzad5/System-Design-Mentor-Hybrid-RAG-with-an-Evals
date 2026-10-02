"""Retrieval-augmented answer generation pipeline."""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from generation.helpers import format_retrieved_context
from generation.prompts import GENERATION_PROMPT


class GenerationPipeline:
    """Retrieve relevant book passages and generate a grounded teaching answer."""

    def __init__(
        self,
        retriever: Callable[[str], list[Document]],
        model: BaseChatModel,
    ) -> None:
        self.retriever = retriever
        self.chain = GENERATION_PROMPT | model | StrOutputParser()

    @traceable(name="book-mentor-generation", run_type="chain")
    def invoke(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        documents = self.retriever(question)
        if not documents:
            return (
                "I couldn't find relevant material in the indexed books, so I can't "
                "answer this from the available context. Try making the question more "
                "specific or using terminology from the books."
            )

        context = format_retrieved_context(documents)
        return self.chain.invoke({"question": question, "context": context})
