
#!/usr/bin/env python3
"""Compare the raw OpenAI SDK with LangChain."""

import os
from dotenv import load_dotenv

# Load API credentials from the project's .env file.
load_dotenv()


def raw_openai_approach():
    """Call the model directly using the OpenAI SDK."""
    import openai

    print("\n🔴 RAW OPENAI SDK APPROACH")

    # Initialize the OpenAI client with our API configuration.
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    # Send a chat completion request to the model.
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {
                "role": "user",
                "content": "Explain machine learning in one sentence"
            }
        ]
    )

    # Extract the generated text from the response object.
    text = response.choices[0].message.content
    print(f"Response: {text[:100]}...")
    return text


def langchain_approach():
    """Make the same request using LangChain."""
    from langchain_openai import ChatOpenAI

    print("\n🟢 LANGCHAIN APPROACH")

    # Initialize LangChain's chat model wrapper.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    # Invoke the model directly with a prompt.
    response = llm.invoke(
        "Explain machine learning in one sentence"
    )

    # LangChain returns an AI message; its text is in .content.
    print(f"Response: {response.content[:100]}...")
    return response.content


def main():
    print("🎯 Task 1: OpenAI vs LangChain")
    print("=" * 50)

    # Run both approaches to compare their results.
    raw_result = raw_openai_approach()
    langchain_result = langchain_approach()

    # Create the completion marker only after both calls succeed.
    if raw_result and langchain_result:
        os.makedirs("markers", exist_ok=True)

        with open("markers/task1_complete.txt", "w") as file:
            file.write("COMPLETED")

        print("\n✅ Task 1 completed!")
        print("Both approaches worked successfully.")


if __name__ == "__main__":
    main()
