
#!/usr/bin/env python3

"""
Task 2: One-Shot Prompting

One-shot prompting means giving the LLM one example of the
expected output before asking it to perform a new task.

Learning goal:
Teach the model a format using one example, then ask it to
generate a different policy following that format.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# Load environment variables from the project root.
# This file is inside Prompt_Engineering_with_LangChain/,
# so ../.env points to langchain-lab/.env.
load_dotenv("../.env")


def main():
    print("Task 2: One-Shot Prompting")
    print("=" * 50)

    # Initialize the same chat model used in Task 1.
    # Keeping the model and configuration consistent helps us
    # focus on the prompting technique rather than model changes.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7,
    )

    # TODO 1: Provide ONE example policy.
    #
    # This example teaches the model the structure we want:
    # title followed by five numbered sections.
    #
    # The model is not being trained or having its weights changed.
    # The example is simply included in the prompt for this request.
    example_policy = """
REFUND POLICY

1. Eligibility: Within 30 days of purchase
2. Conditions: Product unused and in original packaging
3. Process: Submit a request through customer support
4. Timeline: Refund processed within 5-7 business days
5. Exceptions: Some digital products may be non-refundable
"""

    print("\nExample policy:")
    print(example_policy)

    # TODO 2: Create a reusable prompt template.
    #
    # {example} is a placeholder for the example policy.
    # {policy_type} is a placeholder for the new policy we want.
    #
    # PromptTemplate lets us reuse the same prompt structure
    # for different policy types without rewriting the whole prompt.
    one_shot_template = PromptTemplate(
        template="""
Here is an example of our policy format:

{example}

Now write a {policy_type} policy following the same structure.
Use a title and five numbered sections.
Keep the sections relevant to the new policy.
Do not copy refund-specific rules into the new policy.
""",
        input_variables=["example", "policy_type"],
    )

    # TODO 3: Fill the template with actual values.
    #
    # .format() substitutes values into the placeholders.
    # It returns the completed prompt as a string.
    formatted_prompt = one_shot_template.format(
        example=example_policy,
        policy_type="remote work",
    )

    print("\nSending one-shot prompt to the LLM...")

    # Send the completed prompt to the chat model.
    response = llm.invoke(formatted_prompt)

    # .content extracts the generated text from the AIMessage.
    print("\nGenerated remote-work policy:")
    print(response.content)

    print("\nKey takeaway:")
    print("One example can teach the model the desired output format.")
    print("The model generates a new policy based on the example.")

    
# The main guard runs main() only when this file is executed directly.
if __name__ == "__main__":
    main()
