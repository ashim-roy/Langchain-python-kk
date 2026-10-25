
#!/usr/bin/env python3

"""
Task 3: Build Vector Store with ChromaDB

Goal:
1. Load a local embedding model.
2. Represent company policy documents as LangChain Document objects.
3. Split documents into smaller chunks.
4. Store the chunks and their embeddings in ChromaDB.
5. Perform semantic searches against the stored content.

Important:
The vector store in this exercise uses temporary storage.
It will be deleted when the program finishes.
"""


# Import tempfile to create a temporary directory for ChromaDB.
# The temporary directory will be automatically cleaned up later.
import tempfile

# Import Chroma, LangChain's integration with the Chroma vector database.
# Chroma stores document embeddings and supports similarity searches.
from langchain_community.vectorstores import Chroma

# Import HuggingFaceEmbeddings, which converts text into numerical vectors
# using a pretrained embedding model from Hugging Face.
from langchain_huggingface import HuggingFaceEmbeddings

# Import RecursiveCharacterTextSplitter to divide longer documents
# into smaller chunks before storing them.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Document, LangChain's standard structure for holding text
# together with metadata about that text.
from langchain_core.documents import Document


def main():

    # ------------------------------------------------------------
    # STEP 1: DISPLAY THE TASK HEADING
    # ------------------------------------------------------------

    # Print the task name.
    print("🗄️ Task 3: Building Vector Store with ChromaDB")

    # Print a separator to make the terminal output easier to read.
    print("=" * 55)

    # ------------------------------------------------------------
    # STEP 2: INITIALIZE THE EMBEDDING MODEL
    # ------------------------------------------------------------

    # Create an embedding model using Hugging Face.
    #
    # model_name:
    # all-MiniLM-L6-v2 converts text into 384-dimensional vectors.
    #
    # model_kwargs:
    # device='cpu' tells the model to run on the CPU.
    #
    # local_files_only=True tells Hugging Face to use locally cached
    # model files only. It will not download missing model files.
    #
    # encode_kwargs:
    # normalize_embeddings=True normalizes each generated vector.
    #
    # This model was downloaded during Task 1, but this embedding
    # integration uses its own model-loading configuration.
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

    # Confirm that the embedding model object has been initialized.
    print("✅ Embedding model loaded")

    # ------------------------------------------------------------
    # STEP 3: DEFINE THE SAMPLE TECHDOCS DOCUMENTS
    # ------------------------------------------------------------

    # Create a list containing five company policy documents.
    #
    # Each triple-quoted string represents a separate policy section.
    #
    # These are strings inside the Python file, not files read from
    # your computer.
    documents_text = [

        # Document 1: Remote work rules.
        # Contains the number of remote days, VPN requirement,
        # core working hours, and manager approval requirement.
        """Remote Work Policy: Employees can work from home up to 3 days per week.
        VPN access required. Core hours 10 AM - 3 PM local time. Manager approval needed.""",

        # Document 2: Office benefits.
        # Contains information about lunch, gym reimbursement,
        # and the annual learning budget.
        """Office Benefits: Free lunch on Tuesdays and Thursdays. Gym membership reimbursement
        up to $50/month. Annual learning budget of $2000 for courses and conferences.""",

        # Document 3: Vacation and sick leave.
        # Contains paid-time-off and sick-leave allowances.
        """Vacation Policy: 15 days PTO for first year, 20 days after 2 years.
        Sick leave separate - 10 days per year. Holidays follow local calendar.""",

        # Document 4: Security requirements.
        # Contains authentication, password, and incident-reporting rules.
        """Security Guidelines: Two-factor authentication required for all accounts.
        Password changes every 90 days. No sharing of credentials. Report incidents immediately.""",

        # Document 5: Dress code.
        # Contains dress expectations for different days and situations.
        """Dress Code: Business casual Monday-Thursday. Casual Fridays allow jeans.
        Client meetings require business formal. Work from home has no dress requirements."""
    ]

    # ------------------------------------------------------------
    # STEP 4: CONVERT TEXT STRINGS INTO DOCUMENT OBJECTS
    # ------------------------------------------------------------

    # Create an empty list to store LangChain Document objects.
    #
    # We do this because a Document can contain both:
    # 1. page_content: the actual text.
    # 2. metadata: additional information about the text.
    documents = []

    # Loop through the text strings.
    #
    # enumerate() returns both the index and the text.
    # The index starts at 0 by default.
    for i, text in enumerate(documents_text):

        # Create a LangChain Document for the current text.
        doc = Document(

            # Store the actual policy text in page_content.
            page_content=text,

            # Attach metadata describing the document.
            metadata={

                # Identify the original handbook section.
                # i + 1 makes the displayed section number start at 1.
                "source": f"handbook_section_{i+1}",

                # Classify this document as a policy.
                "type": "policy",

                # Store the original zero-based index.
                # This is metadata; it does not determine the search rank.
                "id": i,
            },
        )

        # Add the newly created Document to our list.
        documents.append(doc)

    # Print the number of original Document objects created.
    print(f"📚 Processing {len(documents)} documents...")

    # ------------------------------------------------------------
    # STEP 5: SPLIT DOCUMENTS INTO SMALLER CHUNKS
    # ------------------------------------------------------------

    # Configure the text splitter.
    #
    # chunk_size=200:
    # Aim to keep each chunk at or below 200 characters.
    #
    # chunk_overlap=50:
    # Allow adjacent chunks to share up to approximately 50
    # characters, helping preserve context near chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    # split_documents() takes a list of Document objects rather
    # than a list of plain strings.
    #
    # It splits the page_content of each document.
    # The resulting chunks retain the associated metadata.
    #
    # A single original document may produce multiple chunks.
    splits = text_splitter.split_documents(documents)

    # Print the total number of chunks created from all documents.
    print(f"✂️ Created {len(splits)} chunks")

    # ------------------------------------------------------------
    # STEP 6: CREATE THE CHROMADB VECTOR STORE
    # ------------------------------------------------------------

    # Print a heading to show that vector-store creation is starting.
    print("\n🔨 Building ChromaDB vector store...")

    # Create a temporary directory for ChromaDB's local files.
    #
    # The "with" statement manages the directory's lifetime.
    # When execution leaves this block, the temporary directory
    # and its stored data are cleaned up.
    #
    # This matches the temporary-storage behavior of the original lab.
    with tempfile.TemporaryDirectory() as temp_dir:

        # Create a Chroma vector store from the split documents.
        vectorstore = Chroma.from_documents(

            # Supply the text chunks and their metadata.
            documents=splits,

            # Supply the embedding model.
            # Chroma uses it to generate vectors for the chunks.
            embedding=embeddings,

            # Tell Chroma where to store its local database files.
            persist_directory=temp_dir,

            # Give this Chroma collection a name.
            collection_name="techdocs",
        )

        # Count the vectors stored in the Chroma collection.
        #
        # Each split document is stored as an individual record
        # with its content, metadata, and embedding.
        #
        # _collection is an internal-style attribute of the
        # Chroma integration. It is used here to follow the lab.
        vector_count = vectorstore._collection.count()

        # Display the number of stored vectors.
        print(
            f"✅ Vector store created with "
            f"{vector_count} vectors"
        )

        # --------------------------------------------------------
        # STEP 7: SEARCH THE VECTOR STORE
        # --------------------------------------------------------

        # Display the heading for our sample semantic searches.
        print("\n🔍 Testing Vector Store:")

        # Print a separator.
        print("-" * 40)

        # Define three questions a user might ask about company policy.
        test_queries = [

            # Should retrieve the remote-work policy.
            "Can I work from home?",

            # Should retrieve the dress-code policy.
            "What's the dress code for Friday?",

            # Should retrieve the vacation policy.
            "How many vacation days do I get?"
        ]

        # Process each query separately.
        for query in test_queries:

            # Print the current query.
            print(f"\n📝 Query: '{query}'")

            # Search the vector store for the single most similar
            # chunk, because k=1 requests one result.
            #
            # LangChain embeds the query and compares its vector
            # with the vectors stored in ChromaDB.
            #
            # This is semantic search: it matches meaning, not
            # just exact keyword occurrences.
            results = vectorstore.similarity_search(
                query,
                k=1,
            )

            # Check that at least one result was returned.
            if results:

                # Display the content of the best-matching chunk.
                #
                # [:100] shows only the first 100 characters.
                # The three dots indicate that the preview may
                # have been shortened for display.
                print(
                    f"✨ Best match: "
                    f"{results[0].page_content[:100]}..."
                )

                # Retrieve the source metadata from the matched chunk.
                #
                # .get("source", "unknown") returns the source value
                # if it exists, or "unknown" if it does not.
                print(
                    f"   Source: "
                    f"{results[0].metadata.get('source', 'unknown')}"
                )

    # ------------------------------------------------------------
    # STEP 8: SUMMARIZE WHAT WE BUILT
    # ------------------------------------------------------------

    # The temporary directory has now been cleaned up.
    # We can still print a summary of the exercise.
    print("\n💡 What We Built:")

    # Print a separator.
    print("-" * 40)

    # We stored document chunks together with their vector embeddings.
    print("• A ChromaDB vector store")

    # We searched the stored chunks using natural-language questions.
    print("• Semantic search capability")

    # We attached source, type, and ID metadata to the documents.
    print("• Metadata associated with stored chunks")

    # Chroma supports persistent storage, but this particular
    # exercise uses a temporary directory.
    print("• Temporary storage in this exercise")

    # A vector store can serve as a retrieval component in a RAG system.
    print("• A foundation for RAG applications")

    # Print the completion message.
    print("\n✅ Task 3 completed! Vector store built successfully.")


# Run main() only when this file is executed directly.
if __name__ == "__main__":

    # Start the complete workflow.
    main()
