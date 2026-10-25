#!/usr/bin/env python3

"""
Task 2: Smart Document Processing

Goal:
Implement paragraph-based chunking for better RAG context.

In this task, we:
1. Load the ChromaDB collection and embedding model from Task 1.
2. Read Markdown documents from techcorp-docs/.
3. Split documents into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store the chunks, embeddings, and metadata in ChromaDB.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


print("📄 Task 2: Smart Document Processing")
print("=" * 50)


# ---------------------------------------------------------
# Step 1: Initialize components from Task 1
# ---------------------------------------------------------

# Resolve the directory containing this script.
# This makes the document and database paths independent of
# whichever directory the terminal happens to be in.
SCRIPT_DIR = Path(__file__).resolve().parent

# Connect to the same persistent database created in Task 1.
# Do not change this path, or this task may use a different database.
client = chromadb.PersistentClient(
    path=str(SCRIPT_DIR / "chroma_db")
)

# Retrieve the existing collection or create it if it is missing.
# Task 1 and Task 2 must use the same collection name.
collection = client.get_or_create_collection(
    name="techcorp_rag"
)

# Load the same embedding model used in Task 1.
# Using the same model for documents and queries is important
# because their vectors need to be comparable.
model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Loaded vector store and embedding model")


# ---------------------------------------------------------
# Step 2: Implement smart paragraph-based chunking
# ---------------------------------------------------------

def smart_chunk_document(text, overlap_ratio=0.2):
    """
    Split a document into chunks using paragraphs and overlap.

    The lab's algorithm:
    - Split the document at double newlines.
    - Include the current paragraph in each chunk.
    - Include the next paragraph when one exists.
    - For paragraphs after the first, prepend a tail of the
      previous paragraph to preserve some context.

    overlap_ratio=0.2 means 20% of the previous paragraph's
    character count is used as overlapping text.
    """

    # TODO 1: Split the document into paragraphs.
    # A blank line is represented by two newline characters.
    paragraphs = text.split("\n\n")

    # Remove surrounding whitespace and ignore empty paragraphs.
    # This is useful when a Markdown file contains extra blank lines.
    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []

    # Process each paragraph to build a chunk.
    for i in range(len(paragraphs)):

        # This list collects the pieces that will form the current chunk.
        chunk_parts = []

        # Add the current paragraph to the chunk.
        chunk_parts.append(paragraphs[i])

        # Add the next paragraph if it exists.
        # This helps preserve context between adjacent paragraphs.
        if i + 1 < len(paragraphs):
            chunk_parts.append(paragraphs[i + 1])

        # TODO 2: Calculate the overlap character count.
        # For every paragraph after the first, take the final
        # 20% (by character count) of the previous paragraph.
        if i > 0 and overlap_ratio > 0:

            overlap_chars = int(
                len(paragraphs[i - 1]) * overlap_ratio
            )

            # Only add an overlap when at least one character exists.
            if overlap_chars > 0:

                # Extract the tail of the previous paragraph.
                overlap_text = paragraphs[i - 1][-overlap_chars:]

                # Insert that text at the beginning of the chunk.
                chunk_parts.insert(0, overlap_text)

        # Join the collected pieces into one chunk.
        # A space separates the pieces so words do not run together.
        chunk = " ".join(chunk_parts)

        # Store the completed chunk in our list.
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# Step 3: Find and process the TechCorp documents
# ---------------------------------------------------------

# The original lab uses /root/techcorp-docs, which is a Linux path.
# On your Mac, our documents are inside RAG_Revolution/techcorp-docs.
doc_dir = SCRIPT_DIR / "techcorp-docs"

# Counters used for the final summary.
total_chunks = 0
docs_processed = 0

# Check that the document directory exists before processing.
if not doc_dir.exists():
    raise FileNotFoundError(
        f"Document directory not found: {doc_dir}"
    )

# Find each category directory, such as policies, products, and support.
for category_dir in sorted(doc_dir.iterdir()):

    if category_dir.is_dir():

        print(f"\n📂 Processing {category_dir.name}:")

        # Read Markdown files inside this category directory.
        for doc_file in sorted(category_dir.glob("*.md")):

            # TODO 3: Create metadata to identify the source document.
            # Metadata is stored with every chunk, making it possible
            # to trace a retrieved chunk back to its original file.
            metadata = {
                "source": doc_file.name,
                "section": category_dir.name
            }


            # Read the complete Markdown document as UTF-8 text.
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Split the document into smaller chunks.
            chunks = smart_chunk_document(content)

            # Process each chunk separately.
            for i, chunk in enumerate(chunks):

                # Create a stable ID using the category, filename,
                # and chunk index. This identifies each stored chunk.
                chunk_id = (
                    f"{category_dir.name}_"
                    f"{doc_file.stem}_chunk_{i}"
                )

                # Convert the chunk into a 384-dimensional embedding.
                # .tolist() converts the NumPy vector to a regular list
                # that ChromaDB accepts.
                embedding = model.encode(chunk).tolist()

                # Store the chunk and its associated information.
                # upsert inserts a new ID or updates an existing ID
                # if the script is run again.
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata]
                )

                total_chunks += 1

            docs_processed += 1

            print(
                f"   ✅ {doc_file.name}: "
                f"{len(chunks)} chunks"
            )



# ---------------------------------------------------------
# Step 4: Print the final processing summary
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("🎉 Document Processing Complete!")
print(f"   - Documents processed: {docs_processed}")
print(f"   - Total chunks created: {total_chunks}")
print(f"   - Collection size: {collection.count()}")
print("=" * 50)


# ---------------------------------------------------------
# Step 5: Create a completion marker
# ---------------------------------------------------------

# The original lab writes to /root/markers.
# Create the equivalent marker directory locally on your Mac.
markers_dir = SCRIPT_DIR / "markers"
markers_dir.mkdir(parents=True, exist_ok=True)

# Write a small file indicating that Task 2 completed.
marker_file = markers_dir / "task2_processing_complete.txt"

with marker_file.open("w", encoding="utf-8") as f:
    f.write(
        f"TASK2_COMPLETE:DOCS={docs_processed},"
        f"CHUNKS={total_chunks}"
    )

print(f"\n📄 Completion marker created: {marker_file}")

print("\n💡 Smart chunking preserves context for better generation!")
print("\n✅ Task 2 completed!")