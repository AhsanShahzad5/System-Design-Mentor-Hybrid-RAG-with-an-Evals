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

Current settings are defined in `run_ingestion_pipeline.py`:

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

## Planned hybrid retriever

Retrieval is a separate query-time pipeline and is not part of ingestion:

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

Dense and BM25 searches will produce independent candidate lists. RRF will merge
their ranks without comparing incompatible raw scores. Retrieval, RRF,
reranking, prompting, and answer generation belong to the runtime RAG pipeline.

---
