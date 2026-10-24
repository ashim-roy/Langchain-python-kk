
#!/usr/bin/env python3

"""
Ingest documents into Chroma so the RAG chatbot can retrieve them.

Flow:
Documents -> Text Splitting -> Embeddings -> Chroma Vector Store
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load the API key and other settings from .env.
load_dotenv()

# Load text files as LangChain Document objects.
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Split long documents into smaller, overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Generate embeddings using an OpenAI embedding model.
from langchain_openai import OpenAIEmbeddings

# Store vectors and their associated documents in Chroma.
from langchain_chroma import Chroma


# Resolve paths relative to this Python file, not the terminal directory.
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = PROJECT_DIR / "data" / "docs"
VECTOR_DIR = PROJECT_DIR / "chroma_db"


def main():
    # Fail early if the embedding API key is missing.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")

    # Find all .txt files under data/docs.
    # Each loaded document contains its text and source metadata.
    loader = DirectoryLoader(
        str(DOCUMENT_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No .txt documents found in {DOCUMENT_DIR}")

    print(f"Loaded {len(documents)} document(s).")

    # Split documents into chunks so retrieval can return relevant sections
    # instead of passing an entire long document to the LLM.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunk(s).")

    # An embedding model converts each chunk into a numerical vector.
    # This embedding model is separate from the chat model used to answer.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Chroma stores each chunk, its embedding, and its metadata on disk.
    # Re-running this basic ingestion script recreates the collection,
    # avoiding duplicate inserts when you rerun it.
    vector_store = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    # Delete the old collection contents before re-indexing the files.
    # This keeps repeated runs from accumulating duplicate documents.
    existing = vector_store.get()
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    # Generate embeddings for the chunks and save them to Chroma.
    vector_store.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunk(s) in {VECTOR_DIR}")
    print("✅ Document ingestion completed.")


if __name__ == "__main__":
    main()
