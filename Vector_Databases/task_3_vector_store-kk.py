
#!/usr/bin/env python3

"""
Task 3: Build a Vector Store with ChromaDB — KodeKloud version.

Workflow:
1. Load a Hugging Face embedding model.
2. Create LangChain Document objects with metadata.
3. Split documents into smaller chunks.
4. Store chunks and embeddings in a dedicated ChromaDB directory.
5. Search the stored documents using natural-language queries.

The database is persistent and separate from the existing project database.
"""

# Path helps construct a database directory relative to this Python file.
from pathlib import Path

# Chroma is LangChain's integration with the Chroma vector database.
from langchain_community.vectorstores import Chroma

# HuggingFaceEmbeddings converts text into numerical embedding vectors.
from langchain_huggingface import HuggingFaceEmbeddings

# RecursiveCharacterTextSplitter divides longer text into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document stores text together with metadata.
from langchain_core.documents import Document


def main():

    # ------------------------------------------------------------
    # STEP 1: DISPLAY THE TASK HEADING
    # ------------------------------------------------------------

    print("🗄️ Task 3: Building Vector Store with ChromaDB")
    print("=" * 55)

    # ------------------------------------------------------------
    # STEP 2: INITIALIZE THE EMBEDDING MODEL
    # ------------------------------------------------------------

    # Load the same embedding model used in Task 1.
    #
    # all-MiniLM-L6-v2 generates 384-dimensional vectors.
    #
    # device='cpu' runs inference on the CPU.
    #
    # local_files_only=True requires the model files to be cached
    # locally. Task 1 downloaded this model, so it should be available.
    #
    # normalize_embeddings=True normalizes the vectors generated
    # for documents and queries.
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print("✅ Embedding model loaded")

    # ------------------------------------------------------------
    # STEP 3: DEFINE THE FIVE SAMPLE POLICY DOCUMENTS
    # ------------------------------------------------------------

    # Each string represents one section of a company handbook.
    # These are sample strings in Python, not files read from disk.
    documents_text = [

        # Policy 1: Remote work requirements.
        """Remote Work Policy: Employees can work from home up to 3 days per week.
        VPN access required. Core hours 10 AM - 3 PM local time. Manager approval needed.""",

        # Policy 2: Office benefits.
        """Office Benefits: Free lunch on Tuesdays and Thursdays. Gym membership reimbursement
        up to $50/month. Annual learning budget of $2000 for courses and conferences.""",

        # Policy 3: Vacation and sick leave.
        """Vacation Policy: 15 days PTO for first year, 20 days after 2 years.
        Sick leave separate - 10 days per year. Holidays follow local calendar.""",

        # Policy 4: Security requirements.
        """Security Guidelines: Two-factor authentication required for all accounts.
        Password changes every 90 days. No sharing of credentials. Report incidents immediately.""",

        # Policy 5: Dress code.
        """Dress Code: Business casual Monday-Thursday. Casual Fridays allow jeans.
        Client meetings require business formal. Work from home has no dress requirements."""
    ]

    # ------------------------------------------------------------
    # STEP 4: CREATE DOCUMENT OBJECTS WITH METADATA
    # ------------------------------------------------------------

    # This list will hold LangChain Document objects.
    documents = []

    # Loop over each text string.
    # enumerate() supplies its index, starting at zero.
    for i, text in enumerate(documents_text):

        # A Document stores the text and metadata together.
        doc = Document(

            # The actual policy content.
            page_content=text,

            # Metadata helps us identify the origin and type of a chunk.
            metadata={
                # i + 1 makes section numbering start at 1.
                "source": f"handbook_section_{i + 1}",

                # Identifies these documents as policy content.
                "type": "policy",

                # Stores the original zero-based index.
                "id": i,
            },
        )

        # Add this Document object to the list.
        documents.append(doc)

    # Display the number of original documents.
    print(f"📚 Processing {len(documents)} documents...")

    # ------------------------------------------------------------
    # STEP 5: SPLIT DOCUMENTS INTO CHUNKS
    # ------------------------------------------------------------

    # Configure the splitter to target 200 characters per chunk.
    #
    # The 50-character overlap helps preserve context between
    # neighboring chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    # split_documents() splits each Document's text into chunks.
    #
    # The chunks retain the original document metadata.
    splits = text_splitter.split_documents(documents)

    # Display the total number of chunks.
    print(f"✂️ Created {len(splits)} chunks")

    # ------------------------------------------------------------
    # STEP 6: CHOOSE A DEDICATED DATABASE DIRECTORY
    # ------------------------------------------------------------

    # __file__ is the path of this Python script.
    #
    # .resolve().parent gives us the directory containing this file.
    #
    # We create the database inside Vector_Databases/chroma_db_kk.
    #
    # This does NOT point to the existing langchain-lab/chroma_db.
    database_directory = (
        Path(__file__).resolve().parent / "chroma_db_kk"
    )

    # Print the exact directory being used.
    print(f"\n📁 Database directory: {database_directory}")

    # ------------------------------------------------------------
    # STEP 7: CREATE OR OPEN THE CHROMA VECTOR STORE
    # ------------------------------------------------------------

    # Chroma.from_documents() generates embeddings for the chunks
    # and stores their text, metadata, and vectors in ChromaDB.
    #
    # persist_directory specifies where ChromaDB saves its data.
    #
    # collection_name identifies the collection within the database.
    #
    # On the first run, the collection is created.
    # On later runs, this named collection can be reopened and reused.
    #
    # IMPORTANT:
    # Running this script again with the same documents may add
    # duplicate records. We will handle repeatable ingestion separately.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(database_directory),
        collection_name="techdocs_kk",
    )

    # Count the records stored in this Chroma collection.
    vector_count = vectorstore._collection.count()

    # Display the stored vector count.
    print(f"✅ Vector store created with {vector_count} vectors")

    # ------------------------------------------------------------
    # STEP 8: TEST THE VECTOR STORE WITH SAMPLE QUERIES
    # ------------------------------------------------------------

    # These are natural-language questions a user might ask.
    test_queries = [
        "Can I work from home?",
        "What's the dress code for Friday?",
        "How many vacation days do I get?",
    ]

    print("\n🔍 Testing Vector Store:")
    print("-" * 40)

    # Search once for each question.
    for query in test_queries:

        # Display the question.
        print(f"\n📝 Query: '{query}'")

        # Retrieve the single closest matching chunk.
        #
        # Chroma embeds the query and compares it with stored vectors.
        # k=1 means we request one result.
        results = vectorstore.similarity_search(
            query,
            k=1,
        )

        # Check whether the search returned a result.
        if results:

            # Store the best-matching Document in a variable.
            best_match = results[0]

            # Display the first 100 characters of the matched text.
            print(
                f"✨ Best match: "
                f"{best_match.page_content[:100]}..."
            )

            # Display the source metadata carried by the chunk.
            print(
                "   Source: "
                f"{best_match.metadata.get('source', 'unknown')}"
            )

        else:

            # This message appears if the search returns no documents.
            print("No matching documents found.")

    # ------------------------------------------------------------
    # STEP 9: SUMMARIZE THE RESULT
    # ------------------------------------------------------------

    print("\n💡 What We Built:")
    print("-" * 40)
    print("• A ChromaDB vector store")
    print("• Semantic search over policy documents")
    print("• Metadata associated with document chunks")
    print("• Persistent local storage in a dedicated directory")
    print("• A retrieval component that can support RAG applications")

    print("\n✅ Task 3 completed! Vector store built successfully.")


# Run main() only when the file is executed directly.
if __name__ == "__main__":
    main()
