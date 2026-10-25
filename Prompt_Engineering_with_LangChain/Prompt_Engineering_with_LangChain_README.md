# Prompt Engineering with LangChain

A hands-on learning subproject demonstrating common prompt-engineering techniques using LangChain and a chat model.

## Learning objectives

- Understand zero-shot, one-shot, and few-shot prompting.
- Build reusable prompts with `PromptTemplate`.
- Build example-driven prompts with `FewShotPromptTemplate`.
- Use explicit task steps to structure responses to complex requests.
- Compare prompting approaches on the same task.
- Understand why response length alone is not a reliable quality measure.

## Project structure

```text
Prompt_Engineering_with_LangChain/
├── task_1_zero_shot.py
├── task_2_one_shot.py
├── task_3_few_shot.py
├── task_4_chain_of_thought.py
├── task_5_comparison.py
└── README.md
```

## Prerequisites and configuration

- Python and a virtual environment with the required dependencies installed.
- An API key and compatible chat-model endpoint configured in the parent project's `.env` file.

The scripts load environment variables with:

```python
load_dotenv("../.env")
```

This assumes you run the scripts from inside `Prompt_Engineering_with_LangChain/` and `.env` is in the parent directory.

Expected environment variables:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
```

**Never commit `.env` or publish your API key.** Ensure `.env` is covered by `.gitignore`.

## Setup

From the root of the `langchain-lab` project, activate your virtual environment:

```bash
source .venv/bin/activate
```

Then enter the subproject:

```bash
cd Prompt_Engineering_with_LangChain
```

If the dependencies are not already installed in the parent environment:

```bash
pip install langchain-core langchain-openai python-dotenv
```

If the repository has a `requirements.txt` or another dependency-management file, prefer installing from that file.

## Run the exercises

Run each script from inside `Prompt_Engineering_with_LangChain/`:

```bash
python task_1_zero_shot.py
python task_2_one_shot.py
python task_3_few_shot.py
python task_4_chain_of_thought.py
python task_5_comparison.py
```

Complete them in order, inspect the generated response, and compare the prompt with the output.

## Exercises

### Task 1 — Zero-shot prompting

**Goal:** Compare a vague request with a more specific request without providing examples.

The exercise demonstrates how audience, scope, content requirements, and output format can help focus a response.

**Key idea:** Zero-shot means the model receives instructions but no worked examples in the prompt.

### Task 2 — One-shot prompting

**Goal:** Provide one example policy and ask the model to create a different policy using a similar structure.

The exercise uses `PromptTemplate` to define placeholders and `.format()` to insert the example and the new task.

**Key idea:** One example can demonstrate a desired style or format. The model should adapt the pattern rather than copy rules that do not apply.

### Task 3 — Few-shot prompting

**Goal:** Provide multiple customer-support examples and ask the model to respond to a new customer issue.

The exercise uses `PromptTemplate` to format each example and `FewShotPromptTemplate` to assemble the examples with a prefix and suffix.

**Key idea:** Multiple examples can demonstrate a pattern, but their relevance and quality matter more than the number of examples alone.

### Task 4 — Structured reasoning prompting

**Goal:** Compare a direct request with a prompt that explicitly lists tasks to cover when reviewing a fictional data-retention policy.

The structured prompt asks the model to identify risks, explain why they matter, recommend controls, address backups and inactive users, and summarize implementation actions.

**Key idea:** Explicit steps can guide the organization and coverage of a response. Python does not execute these steps as separate reasoning stages; they are instructions in the prompt.

> The script is named `task_4_chain_of_thought.py` to match the exercise, but the demonstrated technique is more precisely described as structured reasoning prompting. You can request concise explanations and conclusions without asking a model to expose private internal reasoning.

### Task 5 — Compare the techniques

**Goal:** Apply zero-shot, one-shot, few-shot, and structured reasoning prompts to the same remote-work-policy task.

The script prints each response and its character count.

When reviewing results, ask:

- Does the response cover every requested topic?
- Is the structure easy to follow?
- Are the rules practical and clear?
- Does it introduce unsupported assumptions?
- Is it concise without omitting important information?

**Important:** Character count is only a basic observation, not a quality score. The longest response is not necessarily the most accurate or useful. A rigorous comparison should use consistent evaluation criteria and control other differences between prompts.

## Core LangChain concepts

| Concept | Purpose |
|---|---|
| `ChatOpenAI` | Configures a chat model compatible with the OpenAI API interface |
| `PromptTemplate` | Defines a reusable prompt containing variables |
| `FewShotPromptTemplate` | Builds a prompt with formatted examples and task instructions |
| `input_variables` | Declares variables expected by a prompt template |
| `.format()` | Fills template variables and returns the formatted prompt text |
| `llm.invoke()` | Sends a prompt or message input to the model and returns a response |

## Key takeaways

1. **Specific instructions help focus outputs.** State the task, audience, required topics, constraints, and expected format.
2. **Examples demonstrate patterns.** Use relevant, high-quality examples and clarify whether the model should follow their format, tone, or decision pattern.
3. **More examples do not guarantee better results.** Irrelevant or inconsistent examples can confuse the intended pattern.
4. **Explicit steps can improve organization.** They help define the aspects a response should cover, but do not guarantee factual correctness.
5. **Evaluate quality, not just length.** Check correctness, completeness, relevance, clarity, consistency, and unsupported assumptions.
6. **Prompt examples do not fine-tune the model.** They are context supplied for that request; they do not permanently change model weights.

## Troubleshooting

### Environment variables are not loaded

- Run the script from `Prompt_Engineering_with_LangChain/`.
- Confirm `.env` exists in the parent directory.
- Confirm the variable names match those used in the scripts.
- Never paste an API key into source code or commit it to Git.

### A Python import statement fails in the terminal

Python statements such as `from dotenv import load_dotenv` belong in a `.py` file or Python interpreter, not directly in the shell. Run a script with `python task_1_zero_shot.py`.

### The model request fails

- Verify that the virtual environment is activated and required packages are installed.
- Check the API key, endpoint, model name, and network access.
- Review the complete error message before changing the code.

## Repository

This subproject belongs to [Langchain-python-kk](https://github.com/ashim-roy/Langchain-python-kk).
