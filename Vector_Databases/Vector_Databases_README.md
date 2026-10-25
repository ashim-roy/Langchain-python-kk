# Vector Databases — Semantic Search Lab

A hands-on Python lab exploring text embeddings, document chunking, ChromaDB, and semantic search using LangChain and Hugging Face sentence-transformer embeddings.

## Learning objectives

- Convert text into numerical embeddings.
- Compare embeddings using cosine similarity.
- Split longer text into chunks with overlap.
- Represent text and metadata with LangChain `Document` objects.
- Store document chunks and embeddings in ChromaDB.
- Retrieve top-k results using semantic similarity.
- Interpret Chroma distance scores and thresholds.
- Understand persistent versus temporary vector storage.

## Project structure

```text
Vector_Databases/
├── README.md
├── task_1_understanding_embeddings.py
├── task_2_document_processing.py
├── task_2_document_processing-kk.py
├── task_3_vector_store.py
├── task_3_vector_store-kk.py
├── task_4_semantic_search.py
└── chroma_db_kk/                  # Created by Task 3 KodeKloud version
```

Notes:

- `task_2_document_processing.py` is the first, simplified chunking exercise.
- `task_2_document_processing-kk.py` follows the longer KodeKloud TechDocs handbook exercise.
- `task_3_vector_store-kk.py` creates the separate persistent `chroma_db_kk` database.
- `task_4_semantic_search.py` creates a separate temporary vector store.
- The existing `langchain-lab/chroma_db/` belongs to other work and is kept separate from `Vector_Databases/chroma_db_kk/`.

## Prerequisites

Use the existing project virtual environment. Required packages include:

- `sentence-transformers`
- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `langchain-text-splitters`
- `chromadb`
- `numpy`
- `scikit-learn`

From the project root, activate the virtual environment and install any missing packages:

```bash
pip install sentence-transformers langchain langchain-community langchain-huggingface langchain-text-splitters chromadb numpy scikit-learn
```

The model used is `all-MiniLM-L6-v2`. The first run may download it. Later scripts use `local_files_only=True`, so the model must be cached locally. No OpenAI API key is required.

## Run the exercises

From the project root:

```bash
cd Vector_Databases
```

Run each exercise:

```bash
python task_1_understanding_embeddings.py
python task_2_document_processing.py
python task_2_document_processing-kk.py
python task_3_vector_store-kk.py
python task_4_semantic_search.py
```

`task_3_vector_store.py` is also present as the non-KodeKloud version and can be run separately for comparison.

## Task 1 — Understanding embeddings

**File:** `task_1_understanding_embeddings.py`

This exercise uses `all-MiniLM-L6-v2` to convert a query and five sample text strings into embeddings. It uses scikit-learn's `cosine_similarity` to compare the query with each document and sorts the results by similarity.

Concepts covered:

- Query and document embeddings.
- Embedding dimensions.
- Cosine similarity.
- Ranking documents by similarity.

The model produces 384-dimensional embeddings. In the recorded run, the query vector had shape `(1, 384)` and the five document vectors had shape `(5, 384)`.

Example query: `forgot password`

The highest-ranked result was `How to reset your password`, with a cosine similarity of `0.8046` in the recorded run. Similarity scores are not probabilities and may vary with model versions, input text, and configuration.

## Task 2 — Document processing and chunking

### First implementation

**File:** `task_2_document_processing.py`

This exercise uses a short sample employee handbook and `RecursiveCharacterTextSplitter`.

Configuration:

- `chunk_size=500`
- `chunk_overlap=200`
- Separators that prefer paragraph breaks, newlines, and spaces.

The recorded run processed a 984-character sample and created three chunks.

### KodeKloud-style implementation

**File:** `task_2_document_processing-kk.py`

This version uses a longer TechDocs Employee Handbook covering Remote Work Policy, Communication Guidelines, and Performance Management.

Configuration:

- `chunk_size=500`
- `chunk_overlap=200`
- `length_function=len`
- `separators=[" "]`

The recorded run processed 1,607 characters and created five chunks. It displayed previews of the first three chunks and detected shared text between the first two chunks.

The KodeKloud version uses a space as its configured separator. Other separator configurations, such as `["\n\n", "\n", " ", ""]`, may better respect paragraph and line boundaries.

Overlap can preserve context around chunk boundaries. The actual overlap depends on the text and splitter configuration; it should not be assumed that every pair shares exactly the configured number of characters.

## Task 3 — Build a ChromaDB vector store

**KodeKloud-style file:** `task_3_vector_store-kk.py`

This exercise creates five company-policy `Document` objects:

- Remote work policy
- Office benefits
- Vacation policy
- Security guidelines
- Dress code

Each `Document` contains `page_content` and metadata such as `source`, `type`, and `id`. Documents are split using `chunk_size=200` and `chunk_overlap=50`.

The script uses `HuggingFaceEmbeddings` to embed the chunks and `Chroma.from_documents` to store text, embeddings, and metadata.

### Database location and persistence

The KodeKloud-style implementation uses:

```text
Vector_Databases/chroma_db_kk/
```

It uses the Chroma collection named `techdocs_kk`, separate from the existing project-level `langchain-lab/chroma_db/` directory.

The successful run created five chunks and reported five stored vectors. These queries retrieved their expected policy sections in the recorded run:

- `Can I work from home?`
- `What's the dress code for Friday?`
- `How many vacation days do I get?`

**Rerun caution:** the current implementation calls `Chroma.from_documents` each time. Re-running it with the same persistent collection may add duplicate records. Stable IDs or an explicit collection reset/rebuild strategy should be added before relying on repeatable ingestion.

## Task 4 — Semantic search

**File:** `task_4_semantic_search.py`

This exercise builds a knowledge base of 12 short policy entries, creates a temporary ChromaDB vector store, and searches it with natural-language queries.

It demonstrates:

- `similarity_search(query, k=3)` for top-k retrieval.
- `similarity_search_with_score(...)` for retrieving distance scores.
- Filtering results using a distance threshold.
- Matching queries whose wording differs from the stored text.

The main query is `work from home policy`. In the recorded run, the remote-work policy ranked first. The next results included the work-hours policy and health-insurance policy.

### Understanding the score threshold

Chroma's `similarity_search_with_score` returns distance scores for this integration. Lower distances generally indicate closer matches. The exercise keeps results where:

```python
score <= score_threshold
```

The configured threshold is `1.3`. This is a lab setting, not a universal relevance threshold. Evaluate thresholds against representative queries and documents before using them in an application.

### Search examples and limitations

The additional queries include:

- `Can I bring my dog to work?`
- `How many days off do I get?`
- `retirement savings`

The knowledge base does not include a dog policy, so the nearest result cannot be treated as a reliable answer to the dog question. In the recorded run, the days-off query retrieved the sick-leave entry, which is related to time off but does not fully answer the vacation question.

These examples show why retrieval quality must be evaluated and why an application should be able to identify when retrieved context does not answer a question.

**Storage note:** Task 4 uses a temporary directory. Its vector store is deleted when the script finishes; it does not reuse the persistent Task 3 database.

## Overall architecture

```text
Text documents
     |
     v
Document chunking
     |
     v
Embedding model
     |
     v
Vector embeddings + text + metadata
     |
     v
ChromaDB vector store
     |
     v
Embed a user's query
     |
     v
Similarity search / distance ranking
     |
     v
Retrieve relevant text chunks
```

Task 1 explores embeddings and cosine similarity directly. Task 2 focuses on chunking. Task 3 stores embedded chunks in a persistent ChromaDB collection. Task 4 demonstrates semantic retrieval using a separate temporary collection.

Together, these are foundational components for a Retrieval-Augmented Generation (RAG) application. The exercises do not yet implement a complete RAG answer-generation pipeline.

## Troubleshooting

### The embedding model cannot be found

Some scripts use `local_files_only=True`. Run Task 1 first so the model is downloaded and cached. If it is still unavailable, check the Hugging Face cache and model-loading error.

### `ModuleNotFoundError`

Activate the existing project virtual environment and install the missing package using the command in Prerequisites.

### ChromaDB data appears duplicated

Task 3's persistent implementation can add duplicate records when rerun. Do not delete or overwrite a database directory unless you are sure it contains only disposable lab data. Use a deliberate rebuild strategy or stable document IDs to make ingestion repeatable.

### Warnings on macOS

The PyTorch distributed-elastic message about redirects not being supported on macOS did not prevent the recorded exercises from running. The `langchain-community` deprecation warning also did not prevent the ChromaDB exercises from succeeding. Review package migration guidance when upgrading dependencies.

## Key takeaways

- Embeddings represent text as numerical vectors.
- Cosine similarity compares vector directions; higher values generally indicate closer matches.
- Chunking controls the size of text pieces that are embedded and retrieved.
- Metadata connects retrieved chunks to their source and other attributes.
- ChromaDB stores vectors alongside text and metadata.
- Semantic search matches meaning rather than requiring exact keyword matches.
- Distance thresholds must be evaluated for the selected model and dataset.
- A nearest match is not necessarily a correct answer.
- Persistent and temporary storage serve different purposes.
- These components provide the retrieval foundation for RAG, but retrieval and answer generation are separate stages.
