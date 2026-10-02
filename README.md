## **Advanced RAG (the production default)**

Standard RAG plus the improvements you're already planning:

1. hybrid search
2. reranking,
3. query rewriting
4. better chunking
5. metadata filters.

## components

**1. vector db :**

Qdrant / pinecone

**2. Dataset :**

following 5 books

1. Alex xu system design book v1
2. Designing Machine Learning Systems — Chip Huyen
3. Machine Learning System Design Interview — Ali Aminian & Alex Xu
4. Generative AI System Design Interview — Ali Aminian & Hao Shen
5. Designing Data-Intensive Applications — Martin Kleppmann

**3. retreival :**

hybrid search dense + bm25 sparse retriveal , merged using Reciprocal Rank Fusion (RRF)

then we use a re ranker

![alt text](image-1.png)

This is what most real systems actually are.

## chunking

Generally in chunking , there are two levels

**1.document level**

breaking down the large document e.g a pdf , into pages / sections etc

basically breaking down into **units**

Each unit becomes a **Document (text + metadata).**

simplest is page lvl document

**2. Chunking**

Cut each Document into **chunks of a target size** with overlap, using a recursive splitter for e.g

Size is a trade-off: small chunks give precision, large chunks give context.

![alt text](image.png)

**what we do in our project**
Structure-aware Split on headings/sections (Markdown, HTML) and keep the heading as metadata Books, docs, blogs, your use case

# Limitations

Images and diagrams are not indexed; questions depending on figures are out of scope.

because to make it a multi modal RAG , we would have to process all images in these pdfs , individually convert every image into text , in order to store it in the vector db , which can be very costly.
