# Contributing

Thanks for wanting to help! This is a small project, so the process is simple.

## Setup

```bash
git clone https://github.com/<you>/langchain-gpt5
cd langchain-gpt5
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

You don't need an OpenAI key for any of the tests. They use `RecordingChat` from `tests/conftest.py`, a fake chat model that answers from a list you give it and records every prompt. It can also stream and return tool calls (`tool_call("calculator", expression="1+1")`), so most features can be tested without the network.

## Before you open a PR

- `pytest`, `ruff check .` and `mypy gptwrap` pass (CI runs the same).
- New behaviour has a test. Bug fixes have a test that fails without the fix.
- Public functions have a short docstring, and the README is updated if the change is user-facing.
- Keep PRs focused: one feature or fix per PR is much easier to review.

## Adding things

- **A persona**: add it to `BUILT_IN` in `gptwrap/personas.py`. `test_every_built_in_renders_with_example_values` checks it renders.
- **A tool**: write a LangChain `@tool` with a clear docstring (that's what the model reads), return errors as text or raise, and add tests for both the tool and the tool loop. Tools that touch the outside world (files, network) should be opt-in, like `file_tools`.
- **A new error message**: `_explain` in `gptwrap/assistant.py` maps SDK exceptions to one readable line.

## Reporting bugs

Open an issue with what you ran, what you expected and what happened, plus your Python and `langchain-openai` versions. Please don't paste API keys.

By taking part you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
