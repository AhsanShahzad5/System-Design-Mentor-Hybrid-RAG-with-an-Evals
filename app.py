"""Composition root: build every client and component exactly once.

Run scripts, evals and (later) an API all share this one construction path:

    from app import build_app
    app = build_app()
    result = app.invoke("How do LSM-trees work?")
    app.retriever.retrieve(...)       # component-level retriever evals
    app.generator.generate(q, ctx)    # component-level generator evals
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

from document_processing import BOOK_TITLES
from generation import Generator
from rag import ChatMessage, RAGGraph, RAGResult
from retrieval import RetrievalPipeline
from settings import Settings, get_settings


@dataclass(frozen=True)
class BookMentorApp:
    settings: Settings
    model: ChatOpenAI
    retriever: RetrievalPipeline
    generator: Generator
    graph: RAGGraph

    def invoke(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> RAGResult:
        return self.graph.invoke(query=query, history=history)


def build_retriever(settings: Settings) -> RetrievalPipeline:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
    )
    pinecone = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    return RetrievalPipeline(
        embeddings=embeddings,
        pinecone=pinecone,
        pinecone_index=pinecone.index(name=settings.pinecone_index_name),
        namespace=settings.pinecone_namespace,
        rerank_model=settings.rerank_model,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        fused_top_k=settings.fused_top_k,
        final_top_k=settings.final_top_k,
        rrf_k=settings.rrf_k,
    )


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.generation_model,
        verbosity="low",
        stream_usage=True,  # token usage is also reported when streaming
        api_key=settings.openai_api_key,
    )


def build_app(settings: Settings | None = None) -> BookMentorApp:
    settings = settings or get_settings()
    model = build_chat_model(settings)
    retriever = build_retriever(settings)
    generator = Generator(model=model, book_titles=BOOK_TITLES)
    graph = RAGGraph(
        model=model,
        retriever=retriever.retrieve,
        generator=generator,
        catalogue_path=settings.book_catalogue_path,
    )
    return BookMentorApp(
        settings=settings,
        model=model,
        retriever=retriever,
        generator=generator,
        graph=graph,
    )
