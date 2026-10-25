
"""
TASK 2: DOCUMENT PROCESSING AND CHUNKING

Goal:
1. Create a sample employee handbook.
2. Split the handbook into smaller text chunks.
3. Configure the chunk size and chunk overlap.
4. Inspect the resulting chunks.

Why do we need chunking?
Real documents can contain thousands of words.
Splitting them into smaller pieces makes it easier to retrieve
the specific information relevant to a search query.
"""

# Import RecursiveCharacterTextSplitter.
# This LangChain utility splits long text into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    # Print a heading so we can identify this task in the terminal.
    print("=" * 60)
    print("TASK 2: DOCUMENT PROCESSING AND CHUNKING")
    print("=" * 60)

    # ------------------------------------------------------------
    # STEP 1: CREATE SAMPLE DOCUMENT TEXT
    # ------------------------------------------------------------

    # This multiline string represents the content of an employee
    # handbook. It is sample text, not a file loaded from disk.
    #
    # In a later step, we can extract text from actual PDF or TXT
    # files and pass that extracted text to the splitter.
    handbook = """
    Employee Handbook

    Compensation and Payroll:
    Employees receive their salary monthly. Salary payments are
    deposited directly into the employee's registered bank account.
    Payroll questions should be directed to the HR department.
    Employees can access their payslips through the company portal.

    Leave Policy:
    Full-time employees receive annual paid leave. Leave requests
    must be submitted through the HR portal and approved by the
    employee's manager. Sick leave must be reported to the manager
    as soon as possible.

    Remote Work Policy:
    Employees may work remotely when permitted by their team policy.
    They must remain available during agreed working hours and
    protect confidential company information when working remotely.

    Security Policy:
    Employees must use strong passwords and enable multi-factor
    authentication where required. Company devices must be protected
    and security incidents reported to the IT team.
    """

    # Remove whitespace from the beginning and end of the handbook.
    #
    # strip() removes leading and trailing spaces and newline
    # characters. It does not remove spaces inside the text.
    handbook = handbook.strip()

    # Print the total number of characters in the sample document.
    #
    # len() returns the number of characters in the string.
    print(f"\nOriginal document length: {len(handbook)} characters")

    # ------------------------------------------------------------
    # STEP 2: CONFIGURE THE TEXT SPLITTER
    # ------------------------------------------------------------

    # Create a RecursiveCharacterTextSplitter instance.
    #
    # chunk_size=500:
    # Each chunk aims to contain at most 500 characters.
    #
    # chunk_overlap=200:
    # Adjacent chunks can share up to 200 characters.
    # This helps preserve context near chunk boundaries.
    #
    # separators:
    # The splitter tries these separators in order when deciding
    # where to split text. It prefers paragraph breaks, then
    # line breaks, then spaces, then individual characters.
    #
    # This is a character-based splitter, not a token-based splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    # ------------------------------------------------------------
    # STEP 3: SPLIT THE DOCUMENT INTO CHUNKS
    # ------------------------------------------------------------

    # Pass the entire handbook string to the splitter.
    #
    # split_text() returns a Python list containing the chunks.
    #
    # The original handbook remains unchanged; the chunks are
    # separate strings stored in the variable "chunks".
    chunks = text_splitter.split_text(handbook)

    # Print how many chunks were produced.
    print(f"Number of chunks created: {len(chunks)}")

    # ------------------------------------------------------------
    # STEP 4: DISPLAY EACH CHUNK
    # ------------------------------------------------------------

    # Loop through the list of chunks.
    #
    # enumerate(..., start=1) numbers the chunks starting from 1,
    # making the output easier to read.
    for index, chunk in enumerate(chunks, start=1):

        # Print the chunk number.
        print(f"\n{'-' * 60}")
        print(f"CHUNK {index}")

        # Print the length of the current chunk in characters.
        print(f"Length: {len(chunk)} characters")

        # Print the actual text contained in this chunk.
        print(chunk)

    # ------------------------------------------------------------
    # STEP 5: FINISH
    # ------------------------------------------------------------

    # Confirm that the program completed successfully.
    print(f"\n{'=' * 60}")
    print("Task 2 completed!")
    print("=" * 60)


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()
