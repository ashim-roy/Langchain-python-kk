
#!/usr/bin/env python3

"""
Task 3: Few-Shot Prompting

Few-shot prompting provides multiple examples to guide the model.

Learning goal:
Use examples to encourage a consistent tone, structure, and
response pattern for customer-support messages.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)

# Load the .env file from the project root.
load_dotenv("../.env")


def main():
    print("Task 3: Few-Shot Prompting")
    print("=" * 50)

    # Use the same model configuration as the previous tasks.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7,
    )

    # TODO 1: Define multiple input-output examples.
    #
    # Each dictionary contains:
    # input  -> a customer issue
    # output -> an example of the response style we want
    #
    # The model sees all these examples inside its prompt.
    # It can use them as patterns when answering a new issue.
    examples = [
        {
            "input": "refund request",
            "output": (
                "I understand you'd like a refund. "
                "Let me check your order details. "
                "Our refund policy allows returns within 30 days. "
                "I'll help you with the next steps."
            ),
        },
        {
            "input": "shipping delay",
            "output": (
                "I'm sorry your delivery is delayed. "
                "Let me help you check the shipment status. "
                "Once we have the latest tracking information, "
                "we can determine the next steps."
            ),
        },
        {
            "input": "password reset",
            "output": (
                "I'll help you reset your password. "
                "Please use the password-reset option on our sign-in page. "
                "For your security, use the reset link sent to your "
                "registered email and never share your password."
            ),
        },
    ]

    print("\nExamples loaded:")
    for number, example in enumerate(examples, start=1):
        print(f"{number}. Issue: {example['input']}")
        print(f"   Response: {example['output']}\n")

    # TODO 2: Define the format for EACH example.
    #
    # {input} and {output} are placeholders populated from each
    # dictionary in the examples list.
    #
    # This template determines how individual examples appear
    # inside the final prompt.
    example_prompt = PromptTemplate(
        template=(
            "Customer Issue: {input}\n"
            "Support Response: {output}"
        ),
        input_variables=["input", "output"],
    )

    # TODO 3: Build the few-shot prompt.
    #
    # examples:
    #   The three input-output examples above.
    #
    # example_prompt:
    #   Defines how each example is formatted.
    #
    # prefix:
    #   Instructions shown before the examples.
    #
    # suffix:
    #   The final instruction and placeholder for the new issue.
    #
    # input_variables:
    #   Variables that must be supplied when formatting the final prompt.
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            "You are a helpful customer-support agent. "
            "Use the following examples to guide your tone and response style. "
            "Be empathetic, explain the next action, and avoid promising "
            "actions that have not actually been performed.\n\n"
        ),
        suffix=(
            "\nCustomer Issue: {input}\n"
            "Support Response:"
        ),
        input_variables=["input"],
    )

    # TODO 4: Test the prompt with an issue not shown in the examples.
    test_input = "account locked"

    # Format the final prompt.
    #
    # This inserts the examples, prefix, and new issue into one prompt.
    formatted_prompt = few_shot_prompt.format(input=test_input)

    print("New customer issue:", test_input)
    print("\nPrompt sent to the model:")
    print(formatted_prompt)

    # Send the complete few-shot prompt to the LLM.
    response = llm.invoke(formatted_prompt)

    print("\nGenerated support response:")
    print(response.content)

    print("\nKey takeaway:")
    print("Multiple examples demonstrate a repeatable response pattern.")
    print("The model uses the examples as context; no model training occurs.")


if __name__ == "__main__":
    main()
