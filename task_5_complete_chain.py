
#!/usr/bin/env python3

"""
Task 5: Complete Chain — Combining LangChain Components

Learning goals:
1. Build a chain using PromptTemplate, an LLM, and an output parser.
2. Build multiple chains for different purposes.
3. Reuse chains with different inputs.
4. Understand how the pipe operator connects components.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
)

# Load API credentials and configuration from .env.
load_dotenv()


def main():
    print("🎯 Task 5: Complete Chain Composition")
    print("=" * 50)

    # ---------------------------------------------------------
    # STEP 1: INITIALIZE THE MODEL
    # ---------------------------------------------------------

    # Initialize the model once and reuse it across multiple chains.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.3,
    )

    # ---------------------------------------------------------
    # CHAIN 1: ANALYZE A TECHNOLOGY
    # ---------------------------------------------------------

    print("\n⛓️ Chain 1: Technology Analysis")

    # {technology} is a placeholder filled when the chain is invoked.
    analysis_prompt = PromptTemplate(
        template="Analyze {technology} and provide its pros and cons in 2-3 sentences.",
        input_variables=["technology"],
    )

    # StrOutputParser extracts plain text from the AI message.
    str_parser = StrOutputParser()

    # The pipe operator connects:
    # PromptTemplate -> LLM -> StrOutputParser.
    # Each component receives the output of the previous component.
    analysis_chain = analysis_prompt | llm | str_parser

    # Pass the value for {technology} into the chain.
    # The result is a plain Python string.
    analysis = analysis_chain.invoke({"technology": "blockchain"})

    print(f"Analysis:\n{analysis}")
    print(f"Output type: {type(analysis)}")

    # ---------------------------------------------------------
    # CHAIN 2: GENERATE A LIST OF USE CASES
    # ---------------------------------------------------------

    print("\n⛓️ Chain 2: Use Case Generation")

    list_prompt = PromptTemplate(
        template=(
            "List 3 use cases for {technology}. "
            "Return only the use cases, separated by commas."
        ),
        input_variables=["technology"],
    )

    # This parser converts comma-separated model output into a list.
    list_parser = CommaSeparatedListOutputParser()

    # This chain has the same structure but produces a different
    # output type because it uses a different parser.
    list_chain = list_prompt | llm | list_parser

    use_cases = list_chain.invoke({"technology": "blockchain"})

    print(f"Use cases: {use_cases}")
    print(f"Output type: {type(use_cases)}")

    # ---------------------------------------------------------
    # STEP 3: REUSE BOTH CHAINS WITH ANOTHER INPUT
    # ---------------------------------------------------------

    print("\n🔄 Reusing the same chains")
    test_technology = "artificial intelligence"

    # No need to create new templates or chains.
    # We simply provide a different input value.
    analysis = analysis_chain.invoke({"technology": test_technology})
    use_cases = list_chain.invoke({"technology": test_technology})

    print(f"\nTechnology: {test_technology}")
    print(f"Analysis: {analysis}")

    print("Use cases:")
    for index, use_case in enumerate(use_cases, start=1):
        # enumerate(..., start=1) gives each item a human-friendly number.
        print(f"  {index}. {use_case}")

    # ---------------------------------------------------------
    # STEP 4: SAVE THE COMPLETION MARKER
    # ---------------------------------------------------------

    # Use a project-local marker directory on the Mac.
    os.makedirs("markers", exist_ok=True)

    with open("markers/task5_complete.txt", "w") as file:
        file.write("COMPLETED")

    print("\n✅ Task 5 completed!")
    print("🏆 You've built reusable LangChain pipelines!")


if __name__ == "__main__":
    main()
