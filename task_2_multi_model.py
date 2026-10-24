
#!/usr/bin/env python3

"""
Task 2: Multi-Model Support — One Interface, Different Models

Learning goals:
1. Understand how LangChain initializes a chat model.
2. Learn how model configuration works.
3. Understand how to compare models using the same prompt.
4. Learn the difference between a model interface and a model provider.
"""

import os
from dotenv import load_dotenv

# ChatOpenAI is LangChain's integration for chat models that expose
# an OpenAI-compatible API.
#
# It provides a common interface such as .invoke(), so our application
# code doesn't need to manually build raw HTTP requests for each call.
from langchain_openai import ChatOpenAI


# Load environment variables from the project's .env file.
# This keeps API credentials and endpoint configuration out of source code.
load_dotenv()


def main():
    print("🎯 Task 2: Multi-Model Support")
    print("=" * 50)

    # ---------------------------------------------------------
    # STEP 1: CONFIGURE THE MODEL
    # ---------------------------------------------------------

    print("\n🌐 Initializing the model...")

    llm = ChatOpenAI(
        # This is the model identifier accepted by our configured endpoint.
        # We use the same model that worked in Task 1.
        model="gpt-5.6-luna",

        # The API key authenticates our request.
        # Reading it from the environment avoids hardcoding a secret.
        api_key=os.getenv("OPENAI_API_KEY"),

        # The base URL tells LangChain which API endpoint to contact.
        # Our .env currently points to https://api.openai.com/v1.
        base_url=os.getenv("OPENAI_API_BASE"),

        # temperature controls generation variability.
        # Lower values generally produce more predictable output;
        # higher values generally allow more varied output.
        # It does not guarantee accuracy or a particular response length.
        temperature=0.3
    )

    # Important: creating the llm object configures the model client.
    # It does not, by itself, send a prompt or generate a response.
    print("✅ Model initialized!")

    # ---------------------------------------------------------
    # STEP 2: DEFINE A TEST PROMPT
    # ---------------------------------------------------------

    # To compare models fairly, give each model the same prompt.
    # Otherwise, differences could come from the prompts rather than
    # from the models themselves.
    test_prompt = "Explain cloud computing in one sentence"

    print("\n📝 Model Comparison")
    print("=" * 50)
    print(f"Prompt: {test_prompt}")

    # ---------------------------------------------------------
    # STEP 3: INVOKE THE MODEL
    # ---------------------------------------------------------

    # .invoke() sends the prompt to the configured model and waits
    # for the response. This is where the API call happens.
    response = llm.invoke(test_prompt)

    # LangChain returns an AIMessage object.
    # Its .content property contains the generated text.
    print(f"\n🤖 Model: gpt-5.6-luna")
    print(f"Response: {response.content}")

    # ---------------------------------------------------------
    # STEP 4: UNDERSTAND PROVIDER FLEXIBILITY
    # ---------------------------------------------------------

    # We can use a similar LangChain interface with different providers,
    # but changing only the model name is NOT always enough.
    #
    # For example:
    # - OpenAI models can use ChatOpenAI with the OpenAI endpoint.
    # - Google models normally use Google's LangChain integration.
    # - xAI models may use an xAI integration or a compatible gateway.
    #
    # Each provider may require a different API key, endpoint, model ID,
    # or integration package. Our current endpoint is the official
    # OpenAI endpoint, so we test only the model configured above.
    print("\n💡 Same invocation pattern; provider configuration may differ.")

    # ---------------------------------------------------------
    # STEP 5: CREATE THE COMPLETION MARKER
    # ---------------------------------------------------------

    # Create a local markers directory if it doesn't already exist.
    # We use a project-local path because this lab runs on our Mac,
    # not in the Linux KodeKloud environment.
    os.makedirs("markers", exist_ok=True)

    # Record completion only after the model request succeeds.
    with open("markers/task2_complete.txt", "w") as file:
        file.write("COMPLETED")

    print("\n✅ Task 2 completed!")


# Run main() only when this script is executed directly.
# Importing this file from another Python module won't automatically
# trigger the API request.
if __name__ == "__main__":
    main()
