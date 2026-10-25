
#!/usr/bin/env python3

"""
Task 1: Zero-Shot Prompting

Zero-shot prompting means asking the model to perform a task
without providing any examples of the expected answer.

We compare:
1. A vague prompt.
2. A specific prompt with context and constraints.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load OPENAI_API_KEY and OPENAI_API_BASE from the existing project .env.
# Because this script is inside a subdirectory, ../.env points to the
# environment file in the langchain-lab project root.
load_dotenv("../.env")


def main():
    print("Task 1: Zero-Shot Prompting")
    print("=" * 50)

    # Initialize the chat model.
    #
    # model: The model identifier that already works in your environment.
    # api_key: Credentials read from the environment, not hardcoded.
    # base_url: The configured API endpoint.
    # temperature: Controls response variability; 0.7 allows some variation.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7,
    )

    # ZERO-SHOT EXAMPLE 1: A vague instruction.
    #
    # No examples are provided. The model has to infer what kind of
    # policy, length, audience, and requirements we want.
    vague_prompt = "Write a data privacy policy"

    print("\n1. Vague prompt:")
    print(vague_prompt)

    # invoke() sends the prompt to the model.
    # response.content contains the generated text.
    vague_response = llm.invoke(vague_prompt)

    print("\nVague response:")
    print(vague_response.content)



    # ZERO-SHOT EXAMPLE 2: A more specific instruction.
    #
    # We still provide no examples, so this is also zero-shot prompting.
    # We improve the prompt by specifying the audience, length, and topics.
    specific_prompt = (
        "Draft an approximately 200-word data privacy policy for a "
        "fictional company serving European customers. Address data "
        "collection, purpose limitation, security, retention, and user "
        "rights. Use clear language and numbered sections. Treat this "
        "as an educational example, not legal advice, and do not invent "
        "specific legal guarantees."
    )

    print("\n\n2. Specific prompt:")
    print(specific_prompt)

    specific_response = llm.invoke(specific_prompt)

    print("\nSpecific response:")
    print(specific_response.content)

    # Compare output lengths as a simple observation.
    #
    # Output length does not measure correctness or quality by itself.
    print("\nComparison:")
    print(
        f"Vague response length: "
        f"{len(vague_response.content)} characters"
    )
    print(
        f"Specific response length: "
        f"{len(specific_response.content)} characters"
    )

    print("\nKey lesson:")
    print("Clear instructions help the model understand the task.")
    print("More words do not automatically mean a better answer.")


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()
