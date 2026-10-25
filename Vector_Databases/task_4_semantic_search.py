
#!/usr/bin/env python3

"""
Task 4: Semantic Search Implementation

Goal:
1. Load a local embedding model.
2. Create a knowledge base of company policies.
3. Store policy embeddings in ChromaDB.
4. Search for relevant documents using natural-language queries.
5. Retrieve the top-k results.
6. Filter results using a distance threshold.
7. Demonstrate semantic search with different queries.

This exercise uses temporary ChromaDB storage.
The database is removed when the temporary directory is cleaned up.
"""

# Import Chroma, which provides LangChain integration with ChromaDB.
# Chroma stores document text, metadata, and embedding vectors.
from langchain_community.vectorstores import Chroma

# Import HuggingFaceEmbeddings to convert text into numerical vectors.
from langchain_huggingface import HuggingFaceEmbeddings

# Import Document, which stores text and its associated metadata.
from langchain_core.documents import Document


def build_search_engine():

    # ------------------------------------------------------------
    # STEP 1: DISPLAY THE TASK HEADING
    # ------------------------------------------------------------

    # Print the task name.
    print("🔍 Task 4: Semantic Search Implementation")

    # Print a separator below the heading.
    print("=" * 55)

    # ------------------------------------------------------------
    # STEP 2: INITIALIZE THE EMBEDDING MODEL
    # ------------------------------------------------------------

    # Load the pretrained sentence embedding model.
    #
    # all-MiniLM-L6-v2 converts text into 384-dimensional vectors.
    #
    # device='cpu' means the model runs on the CPU.
    #
    # local_files_only=True means the model must already exist
    # in the local cache; it will not be downloaded if missing.
    #
    # normalize_embeddings=True normalizes generated vectors.
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

    # ------------------------------------------------------------
    # STEP 3: DEFINE THE KNOWLEDGE BASE
    # ------------------------------------------------------------

    # Create a list containing 12 short company-policy documents.
    #
    # Each string is a separate knowledge-base entry.
    #
    # These are strings defined inside this Python program.
    # We are not reading them from external files.
    knowledge_base = [

        # Entry 1: Remote work rules.
        "Remote work policy allows employees to work from home up to 3 days per week with manager approval.",

        # Entry 2: Dress code for different days.
        "Dress code is business casual Monday-Thursday. Jeans are permitted on Fridays only.",

        # Entry 3: Annual and mid-year performance reviews.
        "Annual performance reviews happen in December with mid-year check-ins in June.",

        # Entry 4: Health insurance coverage.
        "Health insurance covers employee and dependents with company paying 80% of premiums.",

        # Entry 5: Retirement savings and employer matching.
        "401k retirement plan includes company match up to 6% of salary.",

        # Entry 6: Vacation and paid-time-off policy.
        "Vacation policy provides 15 days PTO first year, increasing to 20 days after 2 years.",

        # Entry 7: Sick leave allowance.
        "Sick leave is separate from vacation with 10 days allocated annually.",

        # Entry 8: Professional development budget.
        "Training budget of $2000 per employee for professional development courses.",

        # Entry 9: Free meals at the office.
        "Office provides free lunch on Tuesdays and Thursdays for all employees.",

        # Entry 10: Employee parking.
        "Parking is free for all employees in the company garage.",

        # Entry 11: Flexible hours and required core hours.
        "Work hours are flexible but core hours 10 AM to 3 PM are required.",

        # Entry 12: Password and authentication requirements.
        "Security policy requires password changes every 90 days and two-factor authentication.",
    ]

    # ------------------------------------------------------------
    # STEP 4: CONVERT TEXT INTO DOCUMENT OBJECTS
    # ------------------------------------------------------------

    # Create one LangChain Document object for each knowledge-base entry.
    #
    # enumerate() provides an index starting at zero.
    #
    # Each Document contains:
    # - page_content: the original policy text.
    # - metadata: additional information about that entry.
    documents = [
        Document(
            page_content=text,
            metadata={"id": i},
        )
        for i, text in enumerate(knowledge_base)
    ]

    # Display the number of documents to be stored.
    print(
        f"📚 Loading {len(documents)} documents "
        f"into vector store..."
    )

    # ------------------------------------------------------------
    # STEP 5: CREATE A TEMPORARY CHROMADB VECTOR STORE
    # ------------------------------------------------------------

    # Create a temporary directory for the ChromaDB database.
    #
    # The "with" statement automatically cleans up the temporary
    # directory when execution leaves the block.
    #
    # This follows the original KodeKloud Task 4 design.
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:

        # Create a Chroma vector store from the Document objects.
        vectorstore = Chroma.from_documents(

            # Supply the 12 policy documents.
            documents=documents,

            # Supply the model used to generate document embeddings.
            embedding=embeddings,

            # Store ChromaDB's files in the temporary directory.
            persist_directory=temp_dir,
        )

        # Confirm that the vector store has been created.
        print("✅ Vector store ready!\n")

        # --------------------------------------------------------
        # STEP 6: CONFIGURE THE MAIN SEARCH QUERY
        # --------------------------------------------------------

        # TODO 1 completed:
        # Define the user's natural-language search query.
        #
        # The user wants to find information about working remotely.
        # The knowledge base uses the phrase "remote work policy",
        # so semantic search can match related wording.
        search_query = "work from home policy"

        # --------------------------------------------------------
        # STEP 7: CONFIGURE HOW MANY RESULTS TO RETURN
        # --------------------------------------------------------

        # TODO 2 completed:
        # k determines the maximum number of results requested.
        #
        # k=3 means we want the three most similar documents.
        k = 3

        # Display the query being searched.
        print(f"🔎 Searching for: '{search_query}'")

        # Display how many results we requested.
        print(f"   Returning top {k} results")

        # Print a separator.
        print("-" * 40)

        # --------------------------------------------------------
        # STEP 8: PERFORM BASIC SIMILARITY SEARCH
        # --------------------------------------------------------

        # Search the vector store using the natural-language query.
        #
        # Internally, LangChain:
        # 1. Converts the query into an embedding vector.
        # 2. Compares it against the stored document vectors.
        # 3. Retrieves the k closest documents.
        #
        # similarity_search() returns Document objects without
        # returning their distance scores.
        results = vectorstore.similarity_search(
            search_query,
            k=k,
        )

        # Display the results heading.
        print("\n📊 Search Results:")

        # Loop through the returned documents.
        #
        # enumerate(..., 1) numbers the results starting at 1.
        for i, doc in enumerate(results, 1):

            # Display the rank and full content of each result.
            print(f"\n{i}. {doc.page_content}")

        # --------------------------------------------------------
        # STEP 9: CONFIGURE A DISTANCE THRESHOLD
        # --------------------------------------------------------

        # TODO 3 completed:
        # Set the maximum accepted distance score.
        #
        # In this Chroma setup, lower distance scores generally
        # indicate closer matches.
        #
        # Therefore, we keep results whose score is <= 1.3.
        score_threshold = 1.3

        # Display the threshold.
        print(
            f"\n🎯 Filtered Search "
            f"(distance <= {score_threshold}):"
        )

        # Print a separator.
        print("-" * 40)

        # --------------------------------------------------------
        # STEP 10: SEARCH AND RETRIEVE DISTANCE SCORES
        # --------------------------------------------------------

        # Search for the five closest documents and include their
        # distance scores.
        #
        # Each returned item contains:
        # (Document, distance_score)
        #
        # The distance score is not a probability or percentage.
        results_with_scores = (
            vectorstore.similarity_search_with_score(
                search_query,
                k=5,
            )
        )

        # Keep only results whose distance is at or below the
        # configured threshold.
        #
        # This is a list comprehension:
        # - doc represents the matched Document.
        # - score represents its distance score.
        relevant_results = [
            (doc, score)
            for doc, score in results_with_scores
            if score <= score_threshold
        ]

        # --------------------------------------------------------
        # STEP 11: DISPLAY THE FILTERED RESULTS
        # --------------------------------------------------------

        # Check whether any results passed the threshold.
        if relevant_results:

            # Display at most the first three qualifying results.
            for doc, score in relevant_results[:3]:

                # Display the distance rounded to three decimals.
                print(f"\n📄 Distance score: {score:.3f}")

                # Display a preview of the matching document.
                print(f"   {doc.page_content[:100]}...")

        else:

            # This message appears if every returned distance
            # exceeds the threshold.
            print("No results met the distance threshold.")

        # --------------------------------------------------------
        # STEP 12: DEMONSTRATE SEMANTIC SEARCH
        # --------------------------------------------------------

        # Define additional queries with expected topics.
        #
        # The original KodeKloud example expects the dog question
        # to find the dress code document. That expectation is not
        # supported by the knowledge base: it contains no dog policy.
        #
        # We preserve the original test query but mark its expected
        # topic as None so we can inspect the actual best match
        # without asserting an unsupported result.
        test_searches = [
            ("Can I bring my dog to work?", None),
            ("How many days off do I get?", "vacation"),
            ("retirement savings", "401k"),
        ]

        # Print the heading for the additional examples.
        print("\n✨ Semantic Search Examples:")

        # Print a separator.
        print("-" * 40)

        # Run each example query.
        for query, expected_topic in test_searches:

            # Retrieve the single closest document.
            results = vectorstore.similarity_search(
                query,
                k=1,
            )

            # Check that at least one result was returned.
            if results:

                # Store the text of the best match.
                best_match = results[0].page_content

                # If the expected topic is None, there is no
                # grounded expectation to validate.
                if expected_topic is None:

                    # Show the actual match for manual inspection.
                    print(f"🔎 Query: '{query}'")
                    print(f"   Best match: {best_match}")

                else:

                    # Check whether the expected topic appears
                    # in the retrieved document text.
                    found_topic = (
                        expected_topic.lower()
                        in best_match.lower()
                    )

                    # Display whether the expected topic was found.
                    status = "✅" if found_topic else "❌"

                    print(
                        f"{status} '{query}' "
                        f"→ Expected topic: {expected_topic}"
                    )

                    # Print the matching text to help us understand
                    # why the search passed or failed.
                    print(f"   Best match: {best_match}")

            else:

                # Handle the unlikely case of an empty result.
                print(f"⚠️ No results returned for: '{query}'")

    # ------------------------------------------------------------
    # STEP 13: SUMMARIZE THE LEARNING OUTCOMES
    # ------------------------------------------------------------

    # The temporary database has been cleaned up after leaving
    # the "with" block.
    #
    # Print a summary of what this exercise demonstrated.
    print("\n🏆 What You've Achieved:")

    # Print a separator.
    print("-" * 40)

    # Summarize the semantic-search implementation.
    print("• Built a semantic search engine")
    print("• Searched by meaning, not just exact keywords")
    print("• Retrieved top-k documents")
    print("• Filtered results using distance scores")
    print("• Explored how natural-language queries match policy text")

    # Print the completion message.
    print("\n✅ Task 4 completed! Semantic search exercise finished.")
    print("🎉 You've completed the Vector Databases lab!")


# Execute the function only when this file is run directly.
if __name__ == "__main__":

    # Start the complete semantic-search workflow.
    build_search_engine()
