
#!/usr/bin/env python3
"""Verify that the local LangChain lab environment is ready."""

import os
import sys
import importlib.util
from dotenv import load_dotenv

# Load API settings from this project's .env file.
load_dotenv()

def main():
    # Confirm that Python is new enough for the lab.
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("FAIL: Python 3.8 or newer is required.")
        sys.exit(1)

    # Check that required packages are installed.
    packages = ["openai", "langchain", "langchain_openai", "pydantic", "dotenv"]
    missing = []

    for package in packages:
        if importlib.util.find_spec(package):
            print(f"PASS: {package} is installed")
        else:
            print(f"FAIL: {package} is missing")
            missing.append(package)

    # Check configuration without displaying the secret API key.
    if os.getenv("OPENAI_API_KEY"):
        print("PASS: OPENAI_API_KEY is configured")
    else:
        print("FAIL: OPENAI_API_KEY is missing")

    if os.getenv("OPENAI_API_BASE"):
        print("PASS: OPENAI_API_BASE is configured")
    else:
        print("INFO: OPENAI_API_BASE is not set")

    # Create a local marker only when required packages and API key exist.
    if not missing and os.getenv("OPENAI_API_KEY"):
        os.makedirs("markers", exist_ok=True)
        with open("markers/environment_verified.txt", "w") as file:
            file.write("VERIFIED")
        print("\nEnvironment verification passed!")
    else:
        print("\nPlease fix the failed checks above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
