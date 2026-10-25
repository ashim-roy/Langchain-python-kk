#!/usr/bin/env python3

"""
Task 3: LLM Integration

Goal:
Configure the AI generation engine using the pre-configured
OpenAI-compatible API.

In this task, we:
1. Load the API configuration from the project .env file.
2. Initialize LangChain's ChatOpenAI client.
3. Configure temperature and maximum output tokens.
4. Send a test prompt to the language model.
5. Save a completion marker if the test succeeds.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


print("🤖 Task 3: LLM Integration")
print("=" * 50)


# ---------------------------------------------------------
# Step 1: Load environment variables
# ---------------------------------------------------------

# Find the directory containing this script.
SCRIPT_DIR = Path(__file__).resolve().parent

# Your .env file is one directory above RAG_Revolution,
# in the main langchain-lab project directory.
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load variables from .env into the process environment.
# This does not print or expose the API key.
load_dotenv(ENV_FILE)

# Read the configured API endpoint and secret API key.
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

# Check the configuration before attempting an API call.
if not api_key:
    raise EnvironmentError(
        f"OPENAI_API_KEY was not found. Check {ENV_FILE}"
    )

if not api_base:
    raise EnvironmentError(
        f"OPENAI_API_BASE was not found. Check {ENV_FILE}"
    )


# ---------------------------------------------------------
# TODO 1: Initialize the LangChain ChatOpenAI client
# ---------------------------------------------------------

# ChatOpenAI is LangChain's integration for OpenAI-compatible
# chat-completion APIs.
#
# api_key: authenticates your API request.
# base_url: specifies which API endpoint receives the request.
# model: identifies the language model to use.
#
# We keep the model name specified by the lab.

client = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model="gpt-5.6-luna"
)

print("✅ OpenAI client initialized")
print(f"   Using API: {api_base}")
print("   API key: configured (hidden)")


# ---------------------------------------------------------
# Step 2: Test basic LLM generation
# ---------------------------------------------------------

def test_generation():
    """Test basic LLM generation and return whether it succeeds."""

    # TODO 2: Set temperature for focused answers.
    #
    # Temperature controls response randomness.
    # Lower values generally produce more focused and predictable
    # responses, although they do not guarantee identical answers.
    temperature = 0.3

    # TODO 3: Set the maximum number of output tokens.
    #
    # Tokens are pieces of text processed by the model.
    # This limit helps keep the response concise.
    max_tokens = 500

    # Configure generation parameters when making the request.
    #
    # Passing these options to invoke is explicit and avoids relying
    # on assigning arbitrary attributes to the client object.
    # print(
    #     f"\n📝 Testing gpt-5.6-luna "
    #     f"with temperature={temperature}"
    # )

    print("\n📝 Testing gpt-5.6-luna with model-default temperature")

    # The system message defines the assistant's behavior.
    # The user message contains the question to answer.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is RAG in AI? Answer in one sentence."
        }
    ]

    # invoke() sends the messages to the configured model.
    # The generation parameters apply to this individual request.
    # This model supports only its default temperature of 1.
    # Keep temperature=0.3 as a lab concept, but don't send it
    # to the API because the model rejects that value.
    #
    # max_tokens limits the response length. We retain the lab's
    # limit of 500 tokens.
    response = client.invoke(
        messages,
        max_tokens=max_tokens
    )

    #print("\n📝 Testing gpt-5.6-luna with model-default temperature")
    print("   - Temperature: model default (1)")

    # response.content contains the generated text.
    answer = response.content

    print(f"\n🤖 Test Response: {answer}")

    # Return True only after a response has been received.
    return True


# ---------------------------------------------------------
# Step 3: Run the test and handle errors
# ---------------------------------------------------------

try:
    success = test_generation()

    if success:
        print("\n" + "=" * 50)
        print("🎉 LLM Integration Successful!")
        print("   - Model: gpt-5.6-luna")
        print("   - Temperature: model default (1)")
        print("   - Max tokens: 500 (concise)")
        print("   - LangChain integration ready for RAG!")
        print("=" * 50)

        # Store the marker locally instead of using /root/markers,
        # which is a Linux lab path and not appropriate on your Mac.
        markers_dir = SCRIPT_DIR / "markers"
        markers_dir.mkdir(parents=True, exist_ok=True)

        marker_file = markers_dir / "task3_llm_complete.txt"

        with marker_file.open("w", encoding="utf-8") as file:
            file.write("TASK3_COMPLETE:LLM_READY")

        print(f"\n📄 Completion marker created: {marker_file}")

except Exception as e:
    # Display the error so we can troubleshoot configuration,
    # connectivity, model availability, or API compatibility.
    print(f"\n❌ Error: {e}")
    print("\n💡 Check that OPENAI_API_BASE and OPENAI_API_KEY are set.")
    print("   Also verify that the configured model is available.")

    # Re-raise the exception so Python exits with a failure status.
    # This prevents an API failure from appearing as a successful run.
    raise

print("\n💡 LLM ready to generate answers from retrieved context!")
print("\n✅ Task 3 completed!")