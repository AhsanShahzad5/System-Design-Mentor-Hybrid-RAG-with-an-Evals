"""Prompt and model composition for graph nodes."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag.prompts import (
    catalogue_answer_system_prompt,
    catalogue_answer_user_prompt,
    query_analysis_system_prompt,
    query_analysis_user_prompt,
)
from rag.schemas import QueryAnalysis


def build_query_analysis_chain(model: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_analysis_system_prompt),
            ("human", query_analysis_user_prompt),
        ]
    )
    return prompt | model.with_structured_output(QueryAnalysis)


def build_catalogue_answer_chain(model: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", catalogue_answer_system_prompt),
            ("human", catalogue_answer_user_prompt),
        ]
    )
    return prompt | model | StrOutputParser()
