#!/usr/bin/env python3

"""
Task 5: Compare Prompt Engineering Techniques

We will give the same problem to the model using four approaches:

1. Zero-shot prompting
2. One-shot prompting
3. Few-shot prompting
4. Structured reasoning prompting

Learning goal:
Understand how examples and explicit instructions affect
the structure and content of model responses.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)


# Load environment variables from the project root.
load_dotenv("../.env")


def main():
    print("Task 5: Comparing Prompt Engineering Techniques")
    print("=" * 60)

    # Initialize the same model used in Tasks 1–4.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7,
    )

    # IMPORTANT:
    # All four techniques will solve this same problem.
    # This helps us compare their responses more fairly.
    problem = """
Create a remote work policy for a fictional technology company.

The policy should explain employee eligibility, working hours,
communication expectations, equipment, and data security.

Keep the policy practical and easy for employees to understand.
"""

    # --------------------------------------------------
    # 1. ZERO-SHOT PROMPTING
    # --------------------------------------------------
    # We provide the task but no examples.
    # The model must generate the policy from the instructions
    # and its existing knowledge.
    zero_shot_prompt = (
        "Create a remote work policy for a fictional technology "
        "company. Cover employee eligibility, working hours, "
        "communication, equipment, and data security. "
        "Use clear headings and practical rules."
    )

    print("\n\n1. ZERO-SHOT PROMPTING")
    print("-" * 60)

    zero_shot_response = llm.invoke(zero_shot_prompt)
    print(zero_shot_response.content)

    # --------------------------------------------------
    # 2. ONE-SHOT PROMPTING
    # --------------------------------------------------
    # We provide ONE example policy to demonstrate the desired
    # format. The model should use the format, not copy rules
    # that only apply to the example.
    example_policy = """
Example: Office Attendance Policy

1. Eligibility
Employees assigned to office-based roles must follow their
team's agreed office attendance schedule.

2. Working Hours
Employees must follow their agreed working hours and notify
their manager about schedule changes.

3. Communication
Employees should keep their team informed about availability
and important work updates.
"""

    # PromptTemplate allows us to reuse a prompt with variables.
    one_shot_template = PromptTemplate(
        template="""
Use the following example to understand the desired policy format.

{example}

Now create a remote work policy based on this request:

{problem}

Use a title and numbered sections.
Adapt the content to remote work rather than copying the example's rules.
""",
        input_variables=["example", "problem"],
    )

    # Fill in the two placeholders in the template.
    one_shot_prompt = one_shot_template.format(
        example=example_policy,
        problem=problem,
    )

    print("\n\n2. ONE-SHOT PROMPTING")
    print("-" * 60)

    one_shot_response = llm.invoke(one_shot_prompt)
    print(one_shot_response.content)

    # --------------------------------------------------
    # 3. FEW-SHOT PROMPTING
    # --------------------------------------------------
    # We provide TWO examples of policies with a consistent
    # structure. The model can use both examples as patterns.
    examples = [
        {
            "policy_type": "Sick Leave Policy",
            "sections": """
1. Eligibility
Employees may request sick leave when illness prevents them
from performing their duties.

2. Notification
Employees should inform their manager as soon as reasonably
possible when they cannot work.

3. Documentation
Any required medical documentation must follow company policy
and applicable requirements.
""",
        },
        {
            "policy_type": "Training Policy",
            "sections": """
1. Eligibility
Employees may participate in approved training relevant to
their role or professional development.

2. Approval
Employees should obtain the required manager approval before
registering for paid training.

3. Responsibilities
Employees should complete agreed training activities and
share relevant learnings with their team.
""",
        },
    ]

    # This template defines how each example will appear.
    example_prompt = PromptTemplate(
        template="""
Policy Type: {policy_type}
{sections}
""",
        input_variables=["policy_type", "sections"],
    )

    # FewShotPromptTemplate inserts every example into the prompt.
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
You write clear, practical company policies.

Here are examples of the expected style and structure:
""",
        suffix="""
Now create a remote work policy using this request:

{problem}

Use a title and numbered sections.
Adapt the examples to remote work; do not copy unrelated rules.
""",
        input_variables=["problem"],
    )

    few_shot_prompt = few_shot_template.format(problem=problem)

    print("\n\n3. FEW-SHOT PROMPTING")
    print("-" * 60)

    few_shot_response = llm.invoke(few_shot_prompt)
    print(few_shot_response.content)

    # --------------------------------------------------
    # 4. STRUCTURED REASONING PROMPTING
    # --------------------------------------------------
    # Here, we do not provide example policies.
    # Instead, we specify the order of tasks the model should
    # cover when preparing the policy.
    reasoning_steps = """
Step 1: Identify the main topics a remote work policy needs.
Step 2: Define eligibility and expectations for working hours.
Step 3: Explain communication and availability requirements.
Step 4: Cover equipment, confidentiality, and data security.
Step 5: Present the final policy in clear, numbered sections.
"""

    structured_template = PromptTemplate(
        template="""
Prepare a remote work policy by following this structured approach:

{steps}

Request:
{problem}

Provide the final policy with a title and numbered sections.
Focus on actionable rules employees can understand.
""",
        input_variables=["steps", "problem"],
    )

    structured_prompt = structured_template.format(
        steps=reasoning_steps,
        problem=problem,
    )

    print("\n\n4. STRUCTURED REASONING PROMPTING")
    print("-" * 60)

    structured_response = llm.invoke(structured_prompt)
    print(structured_response.content)

    # --------------------------------------------------
    # 5. COMPARE THE RESPONSES
    # --------------------------------------------------
    # Store the responses so we can compare their lengths.
    # Length is only a basic observation, NOT a quality score.
    responses = {
        "Zero-shot": zero_shot_response.content,
        "One-shot": one_shot_response.content,
        "Few-shot": few_shot_response.content,
        "Structured reasoning": structured_response.content,
    }

    print("\n\n5. RESPONSE COMPARISON")
    print("=" * 60)

    for technique, response in responses.items():
        print(f"{technique}: {len(response)} characters")

    print("\nCompare the responses manually:")
    print("- Did each policy cover all five requested topics?")
    print("- Which response followed a consistent structure?")
    print("- Which response gave the clearest practical rules?")
    print("- Did any response invent unnecessary requirements?")

    print("\nKey takeaway:")
    print(
        "Examples guide the desired format, while explicit steps "
        "guide how the response is organized."
    )
    print(
        "The best technique depends on the task. Response length "
        "alone does not establish quality or correctness."
    )


if __name__ == "__main__":
    main()