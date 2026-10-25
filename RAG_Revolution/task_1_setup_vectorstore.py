#!/usr/bin/env python3

"""
Task 1: Environment & Vector Store Setup

Goal:
Bridge the knowledge from the Vector Databases Lab to create
the foundation for our Retrieval-Augmented Generation (RAG) application.

In this task, we will:
1. Initialize ChromaDB with persistent storage.
2. Create or retrieve a collection named "techcorp_rag".
3. Load a sentence-transformer embedding model.
4. Generate a test embedding and verify its dimensions.
"""

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


print("🔧 Task 1: Setting up Vector Store for RAG")
print("=" * 50)


# ---------------------------------------------------------
# TODO 1: Initialize ChromaDB client for persistent storage
# ---------------------------------------------------------

# PersistentClient creates a ChromaDB database that saves its
# data on disk rather than keeping everything only in memory.
#
# "./chroma_db" means the database directory will be created
# inside the current RAG_Revolution folder.
#
# This allows other scripts, such as Task 2 and Task 5, to
# reuse the same database after this script finishes.

client = chromadb.PersistentClient(path="./chroma_db")

print("✅ ChromaDB client initialized")


# ---------------------------------------------------------
# TODO 2: Create or retrieve the collection
# ---------------------------------------------------------

# A collection in ChromaDB is similar to a table in a database.
# It stores documents, their embeddings, IDs, and optional metadata.
#
# get_or_create_collection() creates the collection if it doesn't
# exist. If it already exists, ChromaDB returns the existing one.
#
# We use the name "techcorp_rag" consistently across all tasks.

collection = client.get_or_create_collection(
    name="techcorp_rag"
)

print(f"✅ Collection '{collection.name}' ready")


# ---------------------------------------------------------
# TODO 3: Initialize the embedding model
# ---------------------------------------------------------

# SentenceTransformer loads a pretrained model that converts
# text into numerical vectors called embeddings.
#
# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
#
# These vectors represent semantic information about the text.
# Later, we will compare a user's question embedding with
# document embeddings to retrieve relevant documents.

model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Embedding model loaded")


# ---------------------------------------------------------
# Test the embedding model
# ---------------------------------------------------------

# Generate an embedding for a sample sentence.
# This confirms that the model can convert text into a vector.

test_text = "Testing RAG setup"

# encode() returns a numerical vector representing the text.
test_embedding = model.encode(test_text)

# len(test_embedding) returns the number of dimensions in the vector.
print(f"✅ Test embedding created: {len(test_embedding)} dimensions")


# ---------------------------------------------------------
# Verify that setup completed successfully
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("🎉 SUCCESS! Your vector store is ready for RAG!")
print("   - ChromaDB initialized")
print(f"   - Collection: {collection.name}")
print("   - Embedding model: all-MiniLM-L6-v2")
print(f"   - Vector dimensions: {len(test_embedding)}")
print("=" * 50)


# ---------------------------------------------------------
# Create a completion marker file
# ---------------------------------------------------------

# The original lab uses /root/markers, which is a Linux lab path.
# On your Mac, we create a local markers directory inside this
# RAG_Revolution project instead.
#
# Path(__file__).resolve().parent gives the directory containing
# this Python script, regardless of the terminal's working directory.

script_dir = Path(__file__).resolve().parent
markers_dir = script_dir / "markers"

# Create the directory if it doesn't already exist.
# exist_ok=True prevents an error if it already exists.
markers_dir.mkdir(parents=True, exist_ok=True)

# Write a marker indicating that Task 1 completed.
marker_file = markers_dir / "task1_setup_complete.txt"

with marker_file.open("w", encoding="utf-8") as file:
    file.write("TASK1_COMPLETE")

print(f"\n📄 Completion marker created: {marker_file}")

print(
    "\n💡 Remember: You learned this in the Vector Databases Lab "
    "- now applying it for RAG!"
)

print("\n✅ Task 1 completed!")
