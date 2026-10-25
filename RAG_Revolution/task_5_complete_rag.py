#!/usr/bin/env python3

"""
Task 5: Complete RAG Pipeline

Goal:
Wire everything together:
Retrieve -> Augment -> Generate.

In this task, we:
1. Convert the user's question into an embedding.
2. Retrieve the three most relevant chunks from ChromaDB.
3. Add those chunks to a context-aware prompt.
4. Send the prompt to the LLM.
5. Return the answer with source citations.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI


print("🚀 Task 5: Complete RAG Pipeline")
print("=" * 50)


# ---------------------------------------------------------
# Step 1: Load configuration and initialize components
# ---------------------------------------------------------

# Find the absolute directory containing this script.
SCRIPT_DIR = Path(__file__).resolve().parent

# Locate the .env file living one level above the current working directory.
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load the API configuration environment variables safely without printing secrets.
load_dotenv(ENV_FILE)

api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

# Ensure that necessary credential variables exist before proceeding.
if not api_key:
    raise EnvironmentError(
        f"OPENAI_API_KEY was not found in {ENV_FILE}"
    )

if not api_base:
    raise EnvironmentError(
        f"OPENAI_API_BASE was not found in {ENV_FILE}"
    )


# Connect to the persistent ChromaDB database populated previously in Task 2.
# The path and collection name must precisely match prior tasks.
client_db = chromadb.PersistentClient(
    path=str(SCRIPT_DIR / "chroma_db")
)

collection = client_db.get_or_create_collection(
    name="techcorp_rag"
)

# Load the identical embedding model used when documents were first indexed/stored.
# Essential for ensuring query and document vectors exist in a compatible vector space.
model = SentenceTransformer("all-MiniLM-L6-v2")


# Initialize LangChain's chat model integration.
# Configured to use custom base endpoints and specified model architecture parameters.
client_llm = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model="gpt-5.6-luna",
    max_tokens=500
)

print("✅ All components loaded")


# ---------------------------------------------------------
# Step 2: Implement the complete RAG pipeline
# ---------------------------------------------------------

def rag_pipeline(user_question):
    """
    Complete RAG pipeline: Retrieve -> Augment -> Generate.

    Args:
        user_question: The question submitted by the user.

    Returns:
        The generated answer followed by the retrieved source names.
    """

    print(f"\n📝 Question: '{user_question}'")
    print("-" * 50)


    # =====================================================
    # STEP 1: RETRIEVE
    # =====================================================

    print("1️⃣ RETRIEVE: Converting to embedding...")

    # Convert the raw text question into a numerical vector embedding.
    # ChromaDB will compare this vector against stored document vector chunks.
    query_embedding = model.encode(
        user_question
    ).tolist()

    # Perform semantic search on the vector database collection.
    # - query_embeddings expects a list of query vectors (we pass a single-item list).
    # - n_results specifies how many top chunks to fetch (caps at 3 or total collection size).
    # - include pulls text chunks, file metadata, and distance metrics back.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns nested lists to support batching multiple queries.
    # Since we only passed one query, we extract index [0].
    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Gracefully handle edge cases where no context text matches or database is empty.
    if not retrieved_chunks:
        return (
            "I don't have that information in the provided documents."
        )

    print(f"   ✅ Retrieved {len(retrieved_chunks)} relevant chunks")

    # Log sources, sections, and vector distances for debugging.
    # Lower distance values represent closer semantic matches.
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "unknown")
        section = meta.get("section", "unknown")
        print(
            f"      - {source} ({section}), "
            f"distance={distances[i]:.4f}"
        )


    # =====================================================
    # STEP 2: AUGMENT
    # =====================================================

    print("\n2️⃣ AUGMENT: Building context...")

    # Define system instructions to guide model persona and restrict hallucinations.
    # Forces the LLM to rely strictly on context or return a designated safe fallback string.
    system_prompt = """You are TechCorp's helpful AI assistant.
Answer ONLY based on the provided context.
If the answer is not in the context, say:
'I don't have that information in the provided documents.'
Be concise and accurate."""

    # Initialize a context block header string to hold our retrieved text blocks.
    context_text = "Context from TechCorp documents:\n\n"

    # Append each retrieved document chunk, labeling them sequentially [Document 1], [Document 2], etc.
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_text += (
            f"[Document {i}]\n"
            f"{chunk}\n\n"
        )

    # Structure the final user prompt combining the retrieved context blocks and the user question.
    user_prompt = (
        f"{context_text}\n"
        f"Question: {user_question}\n\n"
        "Answer:"
    )

    print("   ✅ Context prepared with retrieved documents")


    # =====================================================
    # STEP 3: GENERATE
    # =====================================================

    print("\n3️⃣ GENERATE: Creating answer...")

    # Package prompts into standard role-based chat message dictionaries for LangChain/OpenAI.
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Invoke the language model with the structured prompt context messages.
    response = client_llm.invoke(messages)
    answer = response.content


    # =====================================================
    # STEP 4: FORMAT THE ANSWER WITH SOURCES
    # =====================================================

    # Extract original source filenames from metadata dictionary results.
    sources = [
        meta.get("source", "unknown")
        for meta in metadatas
    ]

    # De-duplicate file names while preserving natural sorting/retrieval order.
    unique_sources = list(dict.fromkeys(sources))

    # Append clear source citations to the bottom of the final response output.
    final_response = (
        f"{answer}\n\n"
        f"📎 Sources: {', '.join(unique_sources)}"
    )

    return final_response


# ---------------------------------------------------------
# Step 3: Test the complete pipeline
# ---------------------------------------------------------

def test_rag_pipeline():
    """Test the RAG pipeline with several different questions."""

    test_questions = [
        # Test case 1: Out-of-bounds query (information not in documents).
        "Can I bring my dog to the office?",

        # Test case 2: Standard internal documentation query (PTO policy).
        "How many vacation days do I get?",

        # Test case 3: Policy-based rules query (remote work thresholds).
        "What is the remote work policy?"
    ]

    # Loop through test prompts and execute the complete pipeline for each.
    for question in test_questions:
        answer = rag_pipeline(question)

        print("\n" + "=" * 50)
        print("💬 ANSWER:")
        print(answer)
        print("=" * 50)


# ---------------------------------------------------------
# Step 4: Run the pipeline and create a marker
# ---------------------------------------------------------

try:
    # Ensure that ChromaDB was populated successfully by previous setup tasks.
    if collection.count() == 0:
        raise RuntimeError(
            "No documents in ChromaDB. Please run Task 2 first."
        )

    print(
        f"\n📚 Database has {collection.count()} chunks ready"
    )

    # Run the test suite through the complete RAG pipeline.
    test_rag_pipeline()

    print("\n" + "=" * 50)
    print("🎉 RAG Pipeline Complete!")
    print("   - Retrieval: Semantic search working")
    print("   - Augmentation: Context injection ready")
    print("   - Generation: LLM producing answers")
    print("   - Citations: Sources included")
    print("=" * 50)

    # Ensure local directory structure exists and write out a task completion marker file.
    markers_dir = SCRIPT_DIR / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)

    marker_file = markers_dir / "task5_rag_complete.txt"

    with marker_file.open("w", encoding="utf-8") as file:
        file.write("TASK5_COMPLETE:RAG_PIPELINE_READY")

    print(f"\n📄 Completion marker created: {marker_file}")

except Exception as e:
    # Capture runtime pipeline exceptions cleanly and fail execution visibly.
    print(f"\n❌ Error: {e}")
    raise


print("\n🎯 You've built a complete RAG system - from search to answers!")
print("\n✅ Task 5 completed!")