
#!/usr/bin/env python3

"""
Task 4: Output Parsers — Transform AI Text into Structured Data

Learning goals:
1. Extract plain text from a model response.
2. Convert comma-separated output into a Python list.
3. Convert JSON output into a Python dictionary.
4. Connect prompts, models, and parsers into a chain.
"""

import os
from dotenv import load_dotenv

# ChatOpenAI connects to our configured chat model.
from langchain_openai import ChatOpenAI

# StrOutputParser extracts text from a model message.
# CommaSeparatedListOutputParser converts comma-separated text to a list.
# JsonOutputParser parses JSON text into Python objects, usually a dict.
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)

# PromptTemplate lets us create reusable prompts with placeholders.
from langchain_core.prompts import PromptTemplate


# Load API credentials and configuration from the local .env file.
load_dotenv()


def main():
    print("🎯 Task 4: Output Parsers")
    print("=" * 50)

    # Initialize the model once; both examples reuse it.
    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0,
    )

    # temperature=0 generally makes output more predictable.
    # It does not guarantee identical responses or valid JSON.

    # ---------------------------------------------------------
    # EXAMPLE 1: CONVERT TEXT INTO A PYTHON LIST
    # ---------------------------------------------------------

    print("\n📋 Example 1: List Output Parser")

    # The parser converts a comma-separated string such as:
    # "Lower cost, Faster deployment, Better scalability"
    # into a Python list containing three strings.
    list_parser = CommaSeparatedListOutputParser()

    # This prompt asks the model to return three comma-separated items.
    # {technology} is a placeholder filled in when the chain runs.
    list_prompt = PromptTemplate(
        template=(
            "List 3 benefits of {technology}. "
            "Return only the benefits, separated by commas."
        ),
        input_variables=["technology"],
    )

    # The pipe operator connects three components:
    # 1. list_prompt formats the input.
    # 2. llm sends the prompt to the model.
    # 3. list_parser converts the model's text into a Python list.
    #
    # This composition is called a LangChain Runnable sequence.
    list_chain = list_prompt | llm | list_parser

    # invoke() passes a dictionary to the chain.
    # The key "technology" supplies the {technology} placeholder.
    list_result = list_chain.invoke({
        "technology": "cloud computing"
    })

    print(f"Parsed list: {list_result}")
    print(f"Result type: {type(list_result)}")

    # Since the result is a list, we can access an individual item.
    if list_result:
        print(f"First benefit: {list_result[0]}")

    # ---------------------------------------------------------
    # EXAMPLE 2: CONVERT JSON TEXT INTO A PYTHON DICTIONARY
    # ---------------------------------------------------------

    print("\n📦 Example 2: JSON Output Parser")

    # JSON is a structured data format that represents values such as
    # strings, numbers, arrays, objects, and booleans.
    #
    # JsonOutputParser parses JSON text into corresponding Python
    # objects. A JSON object becomes a Python dictionary.
    json_parser = JsonOutputParser()

    # get_format_instructions() returns instructions describing the
    # JSON format expected by the parser.
    # Injecting these instructions helps the model produce parseable JSON.
    json_prompt = PromptTemplate(
        template="""
Analyze {technology} and respond with a JSON object containing:
- benefits: an array of 2 benefits
- complexity: "low", "medium", or "high"
- use_case: one main use case

Technology: {technology}

{format_instructions}
""",
        input_variables=["technology"],

        # partial_variables supplies a value that does not need to be
        # passed every time invoke() is called.
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    # Connect prompt -> model -> JSON parser.
    # The parser converts the model's JSON text into a Python object.
    json_chain = json_prompt | llm | json_parser

    # Run the chain with the requested technology.
    json_result = json_chain.invoke({
        "technology": "machine learning"
    })

    print(f"Parsed JSON: {json_result}")
    print(f"Result type: {type(json_result)}")

    # json_result is already a Python dictionary after parsing.
    # We do not need json.loads() here.
    print(f"Benefits: {json_result.get('benefits', [])}")
    print(f"Complexity: {json_result.get('complexity', 'N/A')}")
    print(f"Use case: {json_result.get('use_case', 'N/A')}")

    # ---------------------------------------------------------
    # SAVE THE COMPLETION MARKER
    # ---------------------------------------------------------

    # Store the marker locally because this project runs on the Mac.
    os.makedirs("markers", exist_ok=True)

    with open("markers/task4_complete.txt", "w") as file:
        file.write("COMPLETED")

    print("\n✅ Task 4 completed!")


# This guard runs main() only when the file is executed directly.
if __name__ == "__main__":
    main()
