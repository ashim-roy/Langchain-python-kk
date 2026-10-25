#!/usr/bin/env python3

"""
Task 4: Prompt Engineering

Goal:
Build a RAG prompt template that encourages accurate,
context-based answers.

In this task, we:
1. Initialize the LangChain ChatOpenAI client.
2. Create a system prompt with grounding instructions.
3. Format retrieved document chunks into a context section.
4. Combine the context and user's question into a user prompt.
5. Send both prompts to the LLM and inspect the answer.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


print("📝 Task 4: Prompt Engineering")
print("=" * 50)


# ---------------------------------------------------------
# Step 1: Load API configuration
# ---------------------------------------------------------

# Find the folder containing this script.
SCRIPT_DIR = Path(__file__).resolve().parent

# The project's .env file is one directory above RAG_Revolution.
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables without exposing the API key.
load_dotenv(ENV_FILE)

api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        f"OPENAI_API_KEY was not found in {ENV_FILE}"
    )

if not api_base:
    raise EnvironmentError(
        f"OPENAI_API_BASE was not found in {ENV_FILE}"
    )


# ---------------------------------------------------------
# Step 2: Initialize the LangChain ChatOpenAI client
# ---------------------------------------------------------

# ChatOpenAI sends messages to the configured language model.
#
# We use the same model and API configuration as Task 3.
# This model supports only its default temperature of 1,
# so we do not pass temperature=0.3.
#
# max_tokens=200 limits the response length for this exercise.

client = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model="gpt-5.6-luna",
    max_tokens=200
)

print("✅ OpenAI client ready")


# ---------------------------------------------------------
# Step 3: Build the RAG prompt
# ---------------------------------------------------------

def create_rag_prompt(context_chunks, user_question):
    """
    Create a system prompt and user prompt for RAG.

    Parameters:
        context_chunks: List of relevant text chunks retrieved
                        from the document collection.
        user_question: The question asked by the user.

    Returns:
        system_prompt: Instructions for how the LLM should answer.
        user_prompt: The context and question sent to the LLM.
    """

    # TODO 1: Define the system prompt.
    #
    # The system prompt establishes the rules the model should follow.
    # "ONLY" tells the model to use the supplied context rather than
    # inventing facts or answering from unrelated outside knowledge.
    #
    # The fallback instruction tells the model what to say when
    # the provided context does not contain the answer.

    system_prompt = """You are TechCorp's helpful AI assistant.
Answer ONLY based on the provided context.
If the answer is not in the context, say:
'I don't have that information in the provided documents.'
Be concise and accurate."""

    # TODO 2: Build the context section from the retrieved chunks.
    #
    # Start with a heading that explains where the context comes from.
    context_text = "Context from TechCorp documents:\n\n"

    # enumerate(..., 1) numbers the documents starting at 1
    # instead of 0, making the context easier to read.
    for i, chunk in enumerate(context_chunks, 1):

        # Label every chunk so its boundary is clear.
        # The actual document text is inserted after the label.
        context_text += f"[Document {i}]\n{chunk}\n\n"

    # TODO 3: Create the user prompt.
    #
    # Include both the retrieved context and the user's question.
    # The model needs the context as evidence and the question to know
    # what information it should extract from that evidence.
    user_prompt = f"""
{context_text}

Question: {user_question}

Answer:"""

    return system_prompt, user_prompt


# ---------------------------------------------------------
# Step 4: Test the prompt template with sample data
# ---------------------------------------------------------

def test_prompt_engineering():
    """Test prompt construction and generate an answer with the LLM."""

    # These are sample retrieved chunks.
    # In Task 5, we will retrieve relevant chunks from ChromaDB instead.
    test_chunks = [
        (
            "TechCorp allows employees to work remotely up to 3 days "
            "per week. Core hours are 10 AM to 3 PM."
        ),
        (
            "Remote work arrangements must be approved by your manager "
            "and documented with HR."
        ),
        (
            "VPN is mandatory when accessing company resources from home."
        )
    ]

    # This question can be answered directly from the first chunk.
    test_question = "How many days can I work from home?"

    # Build the prompts using our function.
    system_prompt, user_prompt = create_rag_prompt(
        test_chunks,
        test_question
    )

    # Print the system prompt so we can verify the grounding rules.
    print("\n📋 System Prompt:")
    print("-" * 40)
    print(system_prompt)

    # Print the user prompt to inspect how the chunks and question
    # have been combined.
    print("\n📋 User Prompt (with context):")
    print("-" * 40)

    # Limit the preview to 500 characters to avoid excessive output.
    # The complete prompt is still sent to the model below.
    print(
        user_prompt[:500] + "..."
        if len(user_prompt) > 500
        else user_prompt
    )

    # LangChain accepts messages with a role and content.
    # The system message contains instructions.
    # The user message contains the context and the question.
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

    # Send the messages to the LLM.
    # We do not pass a custom temperature because this model
    # requires its default temperature setting.
    response = client.invoke(messages)

    # response.content contains the generated natural-language answer.
    answer = response.content

    print("\n🤖 Generated Answer:")
    print("-" * 40)
    print(answer)

    return True


# ---------------------------------------------------------
# Step 5: Run the test and create a completion marker
# ---------------------------------------------------------

try:
    success = test_prompt_engineering()

    if success:
        print("\n" + "=" * 50)
        print("🎉 Prompt Engineering Complete!")
        print("   - System prompt: Context-aware")
        print("   - User prompt: Structured with context")
        print("   - Answer: Based on provided documents")
        print("   - Ready for complete RAG pipeline!")
        print("=" * 50)

        # Use a local marker directory instead of the Linux-specific
        # /root/markers path from the original lab.
        markers_dir = SCRIPT_DIR / "markers"
        markers_dir.mkdir(parents=True, exist_ok=True)

        marker_file = markers_dir / "task4_prompt_complete.txt"

        with marker_file.open("w", encoding="utf-8") as file:
            file.write("TASK4_COMPLETE:PROMPT_READY")

        print(f"\n📄 Completion marker created: {marker_file}")

except Exception as e:
    # Print the error and re-raise it so a failed API request does
    # not appear to be a successful task execution.
    print(f"\n❌ Error: {e}")
    raise

print("\n💡 The RAG prompt formula encourages context-based answers!")
print("\n✅ Task 4 completed!")
