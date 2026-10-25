
#!/usr/bin/env python3

"""
Task 4: Structured Reasoning Prompts

We guide the model through a sequence of steps for a complex task.

Learning goal:
Understand how structured instructions can improve the organization
and completeness of a response.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# Load environment variables from the project root.
load_dotenv("../.env")


def main():
    print("Task 4: Structured Reasoning Prompts")
    print("=" * 50)

    # Use the same configured chat model as in previous tasks.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7,
    )

    # A fictional policy with several potential weaknesses.
    # We will use exactly the same problem in both experiments.
    problem = """
A fictional company has the following data-retention policy:

- Customer personal data is stored indefinitely.
- There is no documented deletion or review process.
- Backups containing personal data are retained forever.
- Active and inactive users are treated the same.

Review this policy and suggest improvements.
This is an educational exercise, not legal advice.
"""

    # PART 1: DIRECT PROMPTING
    #
    # We ask the model to review the policy without specifying
    # a sequence of steps.
    direct_prompt = (
        "Review this fictional data-retention policy and suggest "
        "improvements:\n" + problem
    )

    print("\n1. Direct prompt")
    print("-" * 40)

    direct_response = llm.invoke(direct_prompt)
    print(direct_response.content)

    # TODO 1: Define an explicit sequence of tasks.
    #
    # These steps guide the organization of the response.
    # They are instructions for the model, not executable Python steps.
    reasoning_steps = """
Step 1: Identify the main data-retention risks in the policy.
Step 2: Explain why indefinite retention and missing deletion
        procedures may create compliance and security concerns.
Step 3: Suggest proportionate retention and deletion controls.
Step 4: Address backups and inactive-user data.
Step 5: Summarize practical implementation recommendations.
"""

    # TODO 2: Create a reusable prompt template.
    #
    # {steps} is replaced with the structured instructions.
    # {problem} is replaced with the policy being reviewed.
    cot_template = PromptTemplate(
        template="""
Review the problem using the following structured approach:

{steps}

Problem:
{problem}

Provide a concise, numbered assessment with practical recommendations.
Distinguish general suggestions from legal requirements that would
need confirmation by qualified professionals.
""",
        input_variables=["steps", "problem"],
    )

    # TODO 3: Populate the template.
    #
    # .format() creates the final prompt text from the two variables.
    cot_prompt = cot_template.format(
        steps=reasoning_steps,
        problem=problem,
    )

    print("\n\n2. Structured prompt")
    print("-" * 40)

    cot_response = llm.invoke(cot_prompt)
    print(cot_response.content)

    # Compare response lengths for observation only.
    # A longer answer is not necessarily more accurate or better.
    print("\nComparison")
    print("-" * 40)
    print(f"Direct response: {len(direct_response.content)} characters")
    print(f"Structured response: {len(cot_response.content)} characters")

    print("\nKey takeaway:")
    print("Explicit steps can help organize complex responses.")
    print("Evaluate correctness and completeness, not length alone.")


if __name__ == "__main__":
    main()
