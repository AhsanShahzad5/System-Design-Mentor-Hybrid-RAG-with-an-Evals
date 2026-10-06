"""Prompt templates for LangGraph nodes."""

query_analysis_system_prompt = """Analyze the latest query and return the required structured output.

- Keep the query unchanged when it is already self-contained.
- Rewrite it only when conversation history is needed to resolve references or missing context.
- Use route `catalogue` only for requests about book titles, chapter names, chapter numbers, chapter order, or the available book structure.
- Use route `rag_retrieval` for explanations, summaries, comparisons, recommendations, implementation questions, or questions about substantive book content.
- A request to explain or summarize a named chapter is `rag_retrieval`, not `catalogue`.
"""

query_analysis_user_prompt = """Conversation history:
{history}

Latest query:
{query}"""

catalogue_answer_system_prompt = """Answer the question using only the supplied book catalogue.

Preserve book titles, chapter numbers, chapter names, and chapter order exactly. Return only the information requested. If the requested book or chapter is not present, say so without inventing information.
"""

catalogue_answer_user_prompt = """Question:
{query}

Book catalogue:
{catalogue}"""
