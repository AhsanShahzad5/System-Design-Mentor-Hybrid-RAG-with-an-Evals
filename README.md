# RAG over System Design and AI Engineering Books

The project currently has two separate data pipelines: an expensive PDF
document-processing stage and a repeatable Pinecone ingestion stage.

## Implemented architecture

```text
PDF files
   ↓
DocumentProcessingPipeline
   ↓
One JSONL artifact per book
   ↓
IngestionPipeline
   ↓
Recursive chunks
   ↓
OpenAI dense embeddings + original text
   ↓
Pinecone dense and full-text index
```

The JSONL files are persistent checkpoints for LangChain `Document` objects.
Each line stores `page_content` and `metadata`; ingestion reconstructs normal
LangChain documents before chunking. Chunking and embedding settings can
therefore change without parsing the PDFs again.

## Document-processing pipeline

Run:

```powershell
uv run python run_document_processing.py
```

For every configured `book_id`, this pipeline:

1. Reads the PDF with PyMuPDF and `pymupdf4llm`, without OCR.
2. Extracts layout-aware Markdown while preserving page boundaries.
3. Detects numbered chapters from the embedded PDF table of contents.
4. Normalizes Markdown headings into chapter and section information.
5. Builds heading-aware LangChain `Document` objects.
6. Filters empty content and saves one JSONL artifact per book.

The documents preserve retrieval metadata:

```text
book_id
page_number
chapter_number
chapter_title
section_title
```

Artifacts are written atomically into `processed_documents/`. Existing complete
files are skipped, so interrupted processing can restart without repeating
finished books.

## Pinecone ingestion pipeline

Required environment variables:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
```

Run:

```powershell
uv run python run_ingestion_pipeline.py
```

The ingestion pipeline:

1. Reconstructs documents from all five JSONL files.
2. Merges the per-book collections.
3. Splits them with `RecursiveCharacterTextSplitter`.
4. Adds deterministic `chunk_id` and per-book `chunk_index` metadata.
5. Generates dense embeddings with `text-embedding-3-large`.
6. Stores each embedding, its original text, and metadata in Pinecone.

Application settings are centralized in `settings.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 100
PINECONE_UPSERT_BATCH_SIZE = 20
PINECONE_NAMESPACE = "books"
```

OpenAI processes 100 chunks per embedding request. Each result is divided into
Pinecone requests of 20 records to stay below the HTTP payload limit.
Deterministic IDs make reruns idempotent: the same records are overwritten
rather than duplicated.

### Pinecone schema

```text
embedding  dense vector, 3072 dimensions, cosine similarity
text       English full-text-search field for BM25
```

The runner creates this index when it is absent and rejects an incompatible
existing schema without deleting it. The `books` namespace is created on the
first upsert. The completed corpus currently contains 5,674 records.

### LangSmith tracing

With the standard LangSmith environment variables configured, ingestion creates
a parent trace named `book-ingestion-pipeline`. It records a compact run summary
instead of attaching thousands of chunks as trace output.

## Hybrid retrieval and reranking

Retrieval is implemented as a separate query-time pipeline:

```text
User query
├── OpenAI query embedding → Pinecone dense search
├── Raw query text         → Pinecone BM25 search
└── Reciprocal Rank Fusion (RRF)
    ↓
Final retrieved LangChain documents
    ↓
Optional reranker
```

Dense and BM25 searches produce independent candidate lists. RRF merges their
ranks without comparing incompatible raw scores, and Pinecone's
`bge-reranker-v2-m3` model selects the final results.

Set `QUERY` in `run_retrieval_pipeline.py`, then run:

```powershell
uv run python run_retrieval_pipeline.py
```

Current retrieval settings are defined in `settings.py`:

```python
DENSE_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 20
FINAL_TOP_K = 5
```

The reusable components live under `retrieval/`, while
`run_retrieval_pipeline.py` constructs the dependencies and runs a query.
Prompting and answer generation remain part of the future runtime RAG pipeline.

---
# Grounded answer generation

The complete query flow is:

```text
Query
  → query embedding
  → Pinecone dense + BM25 retrieval
  → RRF
  → reranking
  → top 5 Pinecone records
  → convert records into LangChain Documents
  → format their text and metadata into the prompt
  → LLM
  → grounded answer with section citations
```

After hybrid retrieval, RRF, and reranking, `run_rag_pipeline.py` sends the final
book passages to `gpt-5.6-terra`. The generation prompt is intentionally
context-only: it requires section-level citations, explicitly abstains when the
retrieved passages are insufficient, and teaches concepts using an
intuition-first mentor style.

Generation code is separated by responsibility:

- `generation/prompts.py` contains the mentor and grounding instructions.
- `generation/helpers.py` turns retrieved documents into labeled, citable sources.
- `generation/pipeline.py` composes retrieval, context formatting, and generation.
- `run_rag_pipeline.py` is the executable entry point for manual end-to-end tests.

Edit `QUESTION` in `run_rag_pipeline.py`, then run:

```bash
uv run run_rag_pipeline.py
```
