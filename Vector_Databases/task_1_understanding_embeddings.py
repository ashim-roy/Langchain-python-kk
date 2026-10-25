
"""
TASK 1: UNDERSTANDING EMBEDDINGS

This program demonstrates the basic workflow of semantic search.

We will:
1. Load a pretrained sentence embedding model.
2. Convert a search query into a numerical vector.
3. Convert documents into numerical vectors.
4. Calculate the similarity between the query and each document.
5. Sort the documents from the most similar to the least similar.

Important:
- Embeddings represent the meaning of text as numbers.
- Similar meanings tend to have similar vector representations.
- This program runs locally after the model has been downloaded.
- No OpenAI API key is required.
"""


# Import SentenceTransformer, which loads a pretrained model and
# converts text into numerical vectors called embeddings.
from sentence_transformers import SentenceTransformer

# Import cosine_similarity, which calculates how similar two vectors
# are by comparing the angle between their vector directions.
from sklearn.metrics.pairwise import cosine_similarity


# Define the main function.
# Putting the program logic inside a function makes the code easier
# to organize, understand, reuse, and test.
def main():

    # Print a line containing 60 equals signs.
    # This is purely for making the terminal output easier to read.
    print("=" * 60)

    # Print the heading for this exercise.
    print("TASK 1: UNDERSTANDING EMBEDDINGS")

    # Print another separator below the heading.
    print("=" * 60)

    # ------------------------------------------------------------
    # STEP 1: LOAD THE PRETRAINED EMBEDDING MODEL
    # ------------------------------------------------------------

    # Load the pretrained SentenceTransformer model named
    # "all-MiniLM-L6-v2".
    #
    # This model has learned to represent sentences as vectors.
    #
    # If the model is not already available in the local cache,
    # SentenceTransformer may download it on the first run.
    #
    # The variable "model" stores the loaded model so that we can
    # use it to convert both the query and documents into vectors.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------------------------------------------
    # STEP 2: DEFINE THE USER'S SEARCH QUERY
    # ------------------------------------------------------------

    # This is the text that a user wants to search for.
    #
    # The user is looking for information related to a forgotten
    # password.
    #
    # We will convert this text into an embedding and compare it
    # with the embeddings of our documents.
    query = "forgot password"

    # ------------------------------------------------------------
    # STEP 3: DEFINE THE DOCUMENTS TO SEARCH
    # ------------------------------------------------------------

    # Create a Python list containing five sample documents.
    #
    # In a real application, these documents could come from:
    # - Company knowledge-base articles
    # - Employee handbooks
    # - Technical documentation
    # - Frequently asked questions
    # - Internal support articles
    #
    # Each list element is a separate piece of text that we want
    # to compare with the user's search query.
    documents = [

        # This document directly discusses resetting a password.
        "How to reset your password",

        # This document discusses recovering a password.
        "Password recovery instructions",

        # This document discusses vacation leave.
        # It is probably unrelated to the password query.
        "How to request vacation leave",

        # This document discusses remote work policies.
        # It is also probably unrelated to the password query.
        "Remote work policy and home office rules",

        # This document discusses changing an account password.
        # It is another likely match for the query.
        "Change your account password",
    ]

    # Print a blank line followed by the original search query.
    # The \n character inserts a new line before the text.
    print(f"\nSearch query: {query}")

    # Print the number of documents in the list.
    #
    # len(documents) returns the number of elements in the list.
    # In this example, the result should be 5.
    print(f"Number of documents: {len(documents)}")

    # ------------------------------------------------------------
    # STEP 4: CONVERT THE QUERY INTO AN EMBEDDING
    # ------------------------------------------------------------

    # Convert the query into a numerical vector using the model.
    #
    # Why do we use [query] instead of query?
    #
    # query is a single string:
    #     "forgot password"
    #
    # [query] is a list containing one string:
    #     ["forgot password"]
    #
    # model.encode() accepts a string or a collection of texts.
    # Passing a list makes the output a batch of embeddings.
    #
    # normalize_embeddings=True normalizes the resulting vector
    # to have a length (L2 norm) of 1.
    #
    # Normalized vectors are useful for similarity calculations.
    #
    # The output is stored in query_embedding.
    query_embedding = model.encode(
        [query],                     # The query as a one-item batch.
        normalize_embeddings=True,   # Normalize the query vector.
    )

    # ------------------------------------------------------------
    # STEP 5: CONVERT ALL DOCUMENTS INTO EMBEDDINGS
    # ------------------------------------------------------------

    # Convert every document into its own numerical vector.
    #
    # The model processes the list of documents and returns a
    # collection of embedding vectors.
    #
    # If there are five documents and each embedding has 384
    # dimensions, the resulting array has the shape (5, 384).
    #
    # Each row represents one document.
    # Each column represents one numerical dimension of its vector.
    #
    # normalize_embeddings=True normalizes each document vector.
    document_embeddings = model.encode(
        documents,                    # All five input documents.
        normalize_embeddings=True,    # Normalize every vector.
    )

    # Print a blank line and confirm that the embeddings were created.
    print("\nEmbeddings generated successfully!")

    # Print the shape of the query embedding.
    #
    # query_embedding.shape tells us the dimensions of the array.
    #
    # Expected shape:
    #     (1, 384)
    #
    # 1 means we encoded one query.
    # 384 means the model represents each text using 384 numbers.
    print(f"Query vector shape: {query_embedding.shape}")

    # Print the shape of the document embeddings.
    #
    # Expected shape:
    #     (5, 384)
    #
    # 5 means we encoded five documents.
    # 384 means each document is represented by 384 numbers.
    print(f"Document vectors shape: {document_embeddings.shape}")

    # ------------------------------------------------------------
    # STEP 6: CALCULATE COSINE SIMILARITY
    # ------------------------------------------------------------

    # Compare the query vector with each document vector.
    #
    # query_embedding has shape (1, 384).
    # document_embeddings has shape (5, 384).
    #
    # cosine_similarity compares the query against each document
    # and returns one similarity score per document.
    #
    # Because the vectors are normalized, cosine similarity is
    # equivalent to their dot product, apart from numerical
    # floating-point effects.
    #
    # Scores closer to 1 generally indicate a closer match.
    # A score near 0 indicates little directional similarity.
    # A negative score indicates opposing directions.
    #
    # These scores are model-dependent; they are not probabilities.
    similarities = cosine_similarity(
        query_embedding,       # The one query vector.
        document_embeddings,   # The five document vectors.
    )[0]                       # Select the first (and only) query row.

    # Why do we use [0]?
    #
    # cosine_similarity returns a two-dimensional array.
    # Its shape here is (1, 5), because we have one query and
    # five documents.
    #
    # [0] selects the first row, producing five scores in a
    # one-dimensional array with shape (5,).

    # ------------------------------------------------------------
    # STEP 7: PAIR EACH DOCUMENT WITH ITS SIMILARITY SCORE
    # ------------------------------------------------------------

    # Use zip() to pair each document with its corresponding score.
    #
    # For example, the pairs conceptually look like:
    # (
    #     "How to reset your password",
    #     0.75
    # )
    #
    # The actual scores are calculated by the model and may differ.
    #
    # list() converts the zip object into a list so that we can
    # sort and iterate over the resulting pairs.
    results = list(zip(documents, similarities))

    # ------------------------------------------------------------
    # STEP 8: SORT RESULTS BY SIMILARITY
    # ------------------------------------------------------------

    # Sort the document-score pairs using the similarity score.
    #
    # key=lambda result: result[1]
    #     result is a pair: (document, score).
    #     result[0] is the document text.
    #     result[1] is its similarity score.
    #
    # reverse=True sorts from highest score to lowest score.
    #
    # This means the most similar document appears first.
    results.sort(
        key=lambda result: result[1],
        reverse=True,
    )

    # ------------------------------------------------------------
    # STEP 9: DISPLAY THE RANKED SEARCH RESULTS
    # ------------------------------------------------------------

    # Print a blank line followed by the results heading.
    print("\nRanked search results:")

    # Print a separator to make the output easier to read.
    print("-" * 60)

    # Loop through the sorted results.
    #
    # enumerate() provides both:
    # - rank: the position of the result
    # - (document, score): the text and its similarity score
    #
    # start=1 makes the ranking begin at 1 instead of 0.
    for rank, (document, score) in enumerate(results, start=1):

        # Print the ranking number and document text.
        #
        # Example:
        # 1. How to reset your password
        print(f"{rank}. {document}")

        # Print the similarity score rounded to four decimal places.
        #
        # :.4f formats a floating-point number with four digits
        # after the decimal point.
        #
        # Example:
        # Cosine similarity: 0.7523
        print(f"   Cosine similarity: {score:.4f}")

    # Print a final message indicating that the program has finished.
    print("\nTask 1 completed!")


# ------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------

# Python sets the special variable __name__ to "__main__" when
# this file is executed directly.
#
# This condition calls main() when we run:
#     python task_1_understanding_embeddings.py
#
# If another Python file imports this module, main() will not
# automatically execute.
if __name__ == "__main__":
    main()
