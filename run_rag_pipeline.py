"""Run one grounded question through retrieval and answer generation."""

from langchain_openai import ChatOpenAI
from langsmith import traceable

from generation import GenerationPipeline
from run_retrieval_pipeline import retrieve
from settings import get_settings

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
GENERATION_MODEL = settings.generation_model

# Change this value when manually testing the complete RAG pipeline.
QUESTION = "tell me names of all chapters of book ai engineering "


@traceable(name="book-mentor-rag-pipeline", run_type="chain")
def answer_question(question: str) -> str:
    model = ChatOpenAI(
        model=GENERATION_MODEL,
        verbosity="low",
        api_key=OPENAI_API_KEY,  # type: ignore
    )
    pipeline = GenerationPipeline(retriever=retrieve, model=model)
    return pipeline.invoke(question)


if __name__ == "__main__":
    print(answer_question(QUESTION))
