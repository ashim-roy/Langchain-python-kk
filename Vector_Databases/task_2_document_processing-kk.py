
#!/usr/bin/env python3

"""
Task 2: Document Processing with Smart Chunking

Learn how to split a large document into smaller pieces called chunks.

These chunks can later be converted into embeddings and stored in a
vector database for semantic search.
"""

# Import os because the original KodeKloud exercise uses it to create
# a completion marker. We will not need it for our local Mac version.
# import os

# Import List for optional type annotations.
# The original lab imports it, although this implementation does not
# require it, so we can omit it without changing the program's behavior.
# from typing import List


# TODO 1 completed:
# Import RecursiveCharacterTextSplitter from LangChain.
#
# This class splits text into smaller chunks according to the
# configured chunk size, overlap, and separators.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_documents():

    # Print the task heading.
    # The emoji makes the output easier to identify in the terminal.
    print("📄 Task 2: Smart Document Chunking")

    # Print a separator containing 55 equals signs.
    print("=" * 55)

    # ------------------------------------------------------------
    # STEP 1: DEFINE THE SAMPLE COMPANY DOCUMENTATION
    # ------------------------------------------------------------

    # Create a multiline string representing the TechDocs Employee
    # Handbook.
    #
    # IMPORTANT:
    # This is text written directly inside the Python program.
    # We are not reading a PDF or TXT file from your computer here.
    #
    # The string contains three chapters:
    # 1. Remote Work Policy
    # 2. Communication Guidelines
    # 3. Performance Management
    #
    # In a later exercise, we can extract text from an actual file
    # and process that text in the same way.
    long_document = """
    TechDocs Employee Handbook

    Chapter 1: Remote Work Policy

    Our company embraces flexible work arrangements to support work-life balance.
    Employees are permitted to work remotely up to 3 days per week, provided they
    maintain regular communication with their team and meet all performance expectations.
    Remote work requires manager approval and must not impact team collaboration or
    customer service quality.

    To work remotely, employees must have a suitable home office setup with reliable
    internet connection, appropriate workspace, and necessary equipment. The company
    provides a one-time stipend of $500 for home office setup. VPN access is mandatory
    for accessing company systems remotely.

    Chapter 2: Communication Guidelines

    Effective communication is essential for remote work success. All employees must
    be available during core hours (10 AM - 3 PM in their local timezone) for meetings
    and collaboration. Slack is our primary communication tool, with response times
    expected within 2 hours during work hours.

    Video calls are encouraged for team meetings to maintain personal connections.
    Camera use is optional but recommended. All meetings should have clear agendas
    and action items documented in our project management system.

    Chapter 3: Performance Management

    Remote work does not change performance expectations. Managers will evaluate
    employees based on deliverables, quality of work, and contribution to team goals
    rather than hours logged. Regular 1-on-1 meetings should continue virtually.
    """

    # ------------------------------------------------------------
    # STEP 2: INSPECT THE ORIGINAL DOCUMENT
    # ------------------------------------------------------------

    # len(long_document) calculates the number of characters in
    # the entire string, including spaces and newline characters.
    #
    # We print this length to compare the original document size
    # with the size of the chunks created later.
    print(
        f"📋 Original Document Length: "
        f"{len(long_document)} characters"
    )

    # Print a separator to make the output easier to read.
    print("-" * 40)

    # ------------------------------------------------------------
    # STEP 3: CONFIGURE THE TEXT SPLITTER
    # ------------------------------------------------------------

    # Create an instance of RecursiveCharacterTextSplitter.
    #
    # chunk_size=500:
    # Aim to keep each chunk at or below 500 characters.
    #
    # chunk_overlap=200:
    # Allow adjacent chunks to share up to approximately 200
    # characters of text. This can help preserve context.
    #
    # length_function=len:
    # Tell the splitter to measure chunk length using Python's
    # len() function. In this exercise, the unit is characters,
    # not tokens.
    #
    # separators=[" "]:
    # Use a space as the splitting separator.
    #
    # IMPORTANT:
    # This preserves the separator configuration from KodeKloud.
    # It means the splitter primarily considers spaces as split
    # boundaries instead of preferring paragraphs or newlines.
    # A more flexible configuration could use ["\n\n", "\n", " ", ""].
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Target maximum size: 500 characters.
        chunk_overlap=200,       # Target overlap: 200 characters.
        length_function=len,     # Measure lengths in characters.
        separators=[" "],        # Use spaces as split boundaries.
    )

    # ------------------------------------------------------------
    # STEP 4: SPLIT THE DOCUMENT INTO CHUNKS
    # ------------------------------------------------------------

    # Pass the entire handbook string into split_text().
    #
    # split_text() returns a list of smaller strings.
    #
    # The original document is not replaced or modified.
    # The resulting strings are stored in the variable "chunks".
    chunks = splitter.split_text(long_document)

    # ------------------------------------------------------------
    # STEP 5: DISPLAY THE CHUNKING RESULTS
    # ------------------------------------------------------------

    # Print a heading for the chunking results.
    print("\n✂️ Chunking Results:")

    # Print the total number of chunks produced by the splitter.
    #
    # len(chunks) returns the number of elements in the list.
    print(f"• Created {len(chunks)} chunks")

    # Display the configured target chunk size.
    print("• Chunk size: approximately 500 characters")

    # Display the configured overlap.
    #
    # The original KodeKloud code printed 100 characters here,
    # but its actual configuration specifies 200.
    # We print 200 to match the configured value.
    print("• Configured overlap: 200 characters\n")

    # ------------------------------------------------------------
    # STEP 6: DISPLAY THE FIRST THREE CHUNKS
    # ------------------------------------------------------------

    # Enumerate the first three chunks.
    #
    # chunks[:3] selects at most the first three elements.
    #
    # enumerate(..., 1) gives each chunk a number starting at 1.
    for i, chunk in enumerate(chunks[:3], 1):

        # Print the chunk number and its actual character count.
        print(f"📄 Chunk {i} ({len(chunk)} chars):")

        # Print a separator above the chunk content.
        print("-" * 40)

        # Display only the first 200 characters when a chunk is
        # longer than 200 characters.
        #
        # This keeps terminal output manageable.
        #
        # If the chunk has 200 characters or fewer, display the
        # entire chunk without truncating it.
        #
        # The "..." indicates that some text was omitted from
        # the displayed preview; it is not part of the chunk.
        print(
            chunk[:200] + "..."
            if len(chunk) > 200
            else chunk
        )

        # Print a blank line between chunks.
        print()

    # ------------------------------------------------------------
    # STEP 7: CHECK FOR OVERLAP BETWEEN THE FIRST TWO CHUNKS
    # ------------------------------------------------------------

    # Overlap is relevant only when at least two chunks exist.
    if len(chunks) > 1:

        # Extract the first 100 characters from the second chunk.
        #
        # We use 100 characters for this check because the original
        # KodeKloud exercise checks for a 100-character text segment.
        #
        # This check does not change the configured overlap of 200.
        overlap_start = chunks[1][:100]

        # Check whether that 100-character segment also occurs
        # somewhere inside the first chunk.
        #
        # If it does, we have evidence of shared text between
        # these two chunks.
        if overlap_start in chunks[0]:

            # Print the overlap detection heading.
            print("🔄 Overlap Detection:")

            # Print a separator.
            print("-" * 40)

            # Display the first 50 characters of the shared text.
            print(f"Found overlap: '{overlap_start[:50]}...'")

            # Confirm that shared text was detected.
            print("✅ Context preserved between chunks!")

        else:

            # If this condition occurs, our simple check did not
            # find the selected text inside the first chunk.
            #
            # This does not necessarily prove that no overlap exists:
            # the selected 100-character segment may not match
            # exactly, depending on how the splitter created chunks.
            print("ℹ️ The selected overlap segment was not found.")

    # ------------------------------------------------------------
    # STEP 8: PRINT THE KEY LEARNING POINTS
    # ------------------------------------------------------------

    # Display a summary of why chunking is useful.
    print("\n💡 Why Smart Chunking Matters:")

    # Print a separator below the heading.
    print("-" * 40)

    # Smaller chunks can make it easier to retrieve relevant text.
    print("• Makes smaller sections available for retrieval")

    # Shared text between chunks can preserve nearby context.
    print("• Can maintain context with overlap")

    # Chunks can be passed to embedding models individually.
    print("• Prepares text for embedding models")

    # Retrieval quality depends on many factors, including chunk
    # size, overlap, the embedding model, and the search strategy.
    # The original lab claims a 40% improvement, but this script
    # does not measure or establish that percentage.
    print("• Retrieval quality should be evaluated experimentally")

    # ------------------------------------------------------------
    # STEP 9: LOCAL COMPLETION
    # ------------------------------------------------------------

    # KodeKloud creates a completion marker under /root/markers.
    # That is a Linux lab-specific path and is not appropriate for
    # this local Mac exercise, so we omit that filesystem operation.
    #
    # Print the completion message instead.
    print("\n✅ Task 2 completed! Document chunking executed.")

    # Return the chunks so another Python function could reuse them.
    #
    # For example:
    # chunks = process_documents()
    #
    # A later step could convert these chunks into embeddings.
    return chunks


# This condition checks whether this file is being run directly.
#
# When you run:
# python task_2_document_processing.py
#
# Python sets __name__ to "__main__", so process_documents()
# is called and the task runs.
if __name__ == "__main__":

    # Execute the document-processing function and store its
    # returned list of chunks in the variable "chunks".
    chunks = process_documents()
