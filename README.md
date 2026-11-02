# gptwrap (langchain-gpt5)

[![CI](https://github.com/adityarajsingh1/langchain-gpt5/actions/workflows/ci.yml/badge.svg)](https://github.com/adityarajsingh1/langchain-gpt5/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A small wrapper around GPT-5 built on LangChain. Every time I started a project with an LLM in it, I rewrote the same glue: config and key handling, a system prompt, chat memory, a tool loop, getting JSON back instead of prose, readable errors. This is that glue, written once and tested properly.

- **Assistant** with a system prompt and **memory per session**, with a turn limit so long chats don't get expensive
- **Personas**: reusable system prompts with variables (`tutor` for `{subject}` at a `{level}`)
- **Streaming** answers, plus a terminal chat with `/save`, `/load` and `--resume`
- **Tools** with a proper tool-calling loop: a calculator that never uses `eval()`, date/time, and file reading locked to one folder
- **Structured output**: any Pydantic model back from the model, with a retry when the output doesn't fit
- **Errors you can act on**: "check OPENAI_API_KEY" instead of a 40-line traceback
- **Tested without an API key**: every feature is tested against a fake chat model that records exactly what it was sent

## Install

```bash
git clone https://github.com/adityarajsingh1/langchain-gpt5
cd langchain-gpt5
pip install -e .          # or: pip install -e ".[dev]" for tests and linters
cp .env.example .env      # and put your OpenAI key in it
gptwrap                   # shows which settings will be used (doesn't call the API)
```

| variable | default | |
|---|---|---|
| `OPENAI_API_KEY` | (required) | |
| `OPENAI_MODEL` | `gpt-5` | e.g. `gpt-5-mini` to save money |
| `REASONING_EFFORT` | model default | `minimal`, `low`, `medium` or `high` |
| `OPENAI_TIMEOUT` | `60` | seconds |
| `OPENAI_BASE_URL` | | only for a proxy or a compatible server |

Temperature isn't set anywhere on purpose, because GPT-5 only accepts the default.

## Quick start

```python
from gptwrap.assistant import Assistant
from gptwrap.tools import DEFAULT_TOOLS

bot = Assistant(tools=DEFAULT_TOOLS)
print(bot.ask("how many days until christmas?"))   # uses current_datetime + days_between
print(bot.ask("and in weeks?"))                    # remembers the previous answer
```

From the terminal:

```bash
python chat.py                                                   # chat, answers stream in
python chat.py --persona tutor --set subject=physics --set level="high school"
python chat.py --files ~/notes                                   # it can read files in ~/notes, nothing else
python chat.py --resume chats/physics.json                       # carry on a saved chat
python summarize.py meeting-notes.txt                            # title, key points, to-dos
```

## Guide

### Assistant and memory

`Assistant` takes any LangChain chat model as its first argument and uses GPT-5 if you don't pass one. It remembers the conversation separately for each session id, so one assistant can serve several people:

```python
bot.ask("my name is Aditya", session="me")
bot.ask("what's my name?", session="me")       # knows it
bot.ask("what's my name?", session="someone")  # doesn't
bot.reset("me")                                # forget one conversation (reset(None) forgets all)
```

Only the last 20 question/answer pairs are sent along (`Assistant(max_turns=...)`, `None` for no limit). If a call fails, nothing is saved, so you can just ask again. `bot.stream("question")` yields the answer piece by piece; the turn is saved once the whole answer has arrived.

### Saving conversations

```python
from gptwrap.storage import load_conversation, save_conversation

save_conversation(bot, "chats/physics.json")
load_conversation(other_bot, "chats/physics.json")   # brings back the persona and system prompt too
```

The file is plain JSON, so other tools can read it:

```json
{"version": 1, "saved_at": "...", "persona": "tutor", "system": "...",
 "messages": [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}]}
```

It's written to a temp file and renamed, so a crash never leaves half a file, and a damaged or hand-edited file gives a clear error.

### Personas

```python
tutor = Assistant.from_persona("tutor", subject="chemistry", level="high school")
reviewer = Assistant.from_persona("reviewer", language="Go")
```

`gptwrap --personas` lists the built-in ones:

```
default      general helpful assistant
tutor        explains a subject step by step at the right level  [level='beginner', subject]
reviewer     code reviewer for one language  [language='Python']
translator   translates into another language, nothing else  [target_language]
interviewer  mock job interviewer  [role='junior software engineer']
```

A missing or misspelled variable raises an error before anything is sent, instead of a prompt with `{subject}` still in it. Make your own with `Persona(name, description, template, defaults)`.

### Tools

| tool | what it does | safety |
|---|---|---|
| `calculator` | exact arithmetic, `sqrt`, `log`, `pi`... | walks the syntax tree, no `eval()`; `__import__('os')` and `9**9**9` are refused |
| `current_datetime` | date, time and weekday in any timezone | |
| `days_between` | days between two dates | |
| `list_files`, `read_file` | read text files (opt in with `file_tools(folder)`) | paths are resolved first, so `../`, absolute paths and symlinks out of the folder are refused; 20k character cap |

When the model asks for a tool, the tool runs, the result goes back, and that repeats until it answers, at most 5 rounds (`max_tool_steps`). If a tool call goes wrong (unknown tool, wrong arguments, the tool crashes), the error goes back to the model as the tool's result instead of crashing the chat. Only the question and the final answer go into the history.

Any LangChain `@tool` works, so adding your own is a few lines:

```python
from langchain_core.tools import tool

@tool
def weather(city: str) -> str:
    """Current weather in a city."""
    ...

bot = Assistant(tools=[*DEFAULT_TOOLS, weather])
```

### Structured output

```python
from pydantic import BaseModel
from gptwrap.models import chat_model
from gptwrap.structured import extract, summarize

s = summarize(chat_model(), open("meeting-notes.txt").read())
s.action_items          # a real list, not text to parse
s.to_markdown()

class Contact(BaseModel):
    name: str
    email: str | None = None

extract(chat_model(), "write to Aditya at a@example.com", Contact)
```

If the output doesn't fit the schema, the model gets the validation error and one more try.

### Errors

API failures come back as `AssistantError` with one line on what to check: the key, the model name, the rate limit, the connection or the timeout.

## Examples

[`examples/`](examples/) has small scripts that use the library the way I would in a real project:

- `code_review.py`: review a file with the `reviewer` persona and folder-locked file tools
- `meeting_actions.py`: pull a who-does-what checklist out of messy meeting notes
- `support_desk.py`: one assistant, separate memory per customer
- `study_buddy.py`: a tutor that saves the chat after every answer so you can continue tomorrow

## Project layout

```
gptwrap/
  config.py      settings from env / .env, readable errors
  models.py      builds the ChatOpenAI model
  assistant.py   Assistant: prompt, memory, streaming, tool loop, error messages
  personas.py    persona templates
  tools.py       calculator, date/time tools
  files.py       folder-locked file tools
  structured.py  extract() / summarize() into Pydantic models
  storage.py     save / load conversations as JSON
chat.py          terminal chat
summarize.py     summaries from the terminal
examples/        example scripts
tests/           pytest, all against a fake chat model
```

## Development

```bash
pip install -e ".[dev]"
pytest          # no API key or network needed
ruff check .
mypy gptwrap
```

CI runs the same three on Python 3.10, 3.11 and 3.12 for every push and pull request. The fake chat model in `tests/conftest.py` records every prompt it gets and can stream, call tools and return structured output, so new features can be tested exactly.

## Contributing

Contributions are welcome, from typo fixes to new tools and personas. [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup and what makes a PR easy to merge. Some ideas if you're looking for something to work on:

- more personas (with a test that they render)
- an async version of `Assistant` (`aask`, `astream`)
- persistent memory backends (SQLite, Redis) behind the same `history` / `set_history` interface
- a token/cost counter per session
- support for other providers through any LangChain chat model, with docs

## License

MIT, see [LICENSE](LICENSE).
