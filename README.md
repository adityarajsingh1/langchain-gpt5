# langchain-gpt5

A small wrapper around GPT-5 built on LangChain: an assistant with personas, memory, tools and structured output, that I can drop into other projects instead of rewriting the same glue every time.

**Work in progress.** I'm building it step by step. So far: settings come from environment variables (or a `.env` file), the LangChain `ChatOpenAI` model is built from them, there's an `Assistant` class on top, and a few personas to start it with.

```bash
pip install -r requirements.txt
cp .env.example .env   # and put your OpenAI key in it
python -m gptwrap      # shows which settings would be used, doesn't call the API
```

| variable | default | |
|---|---|---|
| `OPENAI_API_KEY` | (required) | |
| `OPENAI_MODEL` | `gpt-5` | e.g. `gpt-5-mini` to save money |
| `REASONING_EFFORT` | model default | `minimal`, `low`, `medium` or `high` |
| `OPENAI_TIMEOUT` | `60` | seconds |
| `OPENAI_BASE_URL` | | only for a proxy or compatible server |

If the key is missing you get a message saying where to get one, not a stack trace. Temperature isn't set anywhere on purpose, because GPT-5 only accepts the default.

```python
from gptwrap.assistant import Assistant

bot = Assistant(system="You explain things to a 10 year old.")
print(bot.ask("why is the sky blue?"))
```

`Assistant` takes any LangChain chat model as its first argument and uses GPT-5 if you don't pass one. It remembers the conversation, separately for each session id:

```python
bot.ask("my name is Aditya", session="me")
bot.ask("what's my name?", session="me")      # knows it
bot.ask("what's my name?", session="someone") # doesn't
bot.reset("me")                               # forget that conversation (reset(None) forgets all)
```

Only the last 20 question/answer pairs are sent along (`Assistant(max_turns=...)`, `None` for no limit), so a long chat doesn't get more expensive with every message. History lives in memory for now, so it's gone when the program exits. If the API call fails, you get an `AssistantError` with one line on what to check (key, model name, rate limit, connection) instead of the raw SDK exception.

### Chatting in the terminal

```bash
python chat.py
python chat.py --persona tutor --set subject=physics --set level="high school"
```

Answers stream in as GPT-5 writes them. While chatting, `/reset` forgets the conversation, `/history` shows it, `/system` shows the system prompt and `/quit` (or ctrl+d) leaves. Ctrl+c stops an answer halfway without ending the chat, and the stopped answer isn't kept in the history. API errors are printed as one line and you can keep going.

From code it's `bot.stream("question")`, which yields the pieces of the answer.

### Tools

The model can use a few tools when it needs them:

- `calculator`: exact arithmetic (`17 * 23`, `sqrt(2) * 10`, `(1 + 0.07) ** 10`). It doesn't use `eval()`: it walks the expression's syntax tree and only allows numbers, operators and a short list of math functions, so `__import__('os')` just gets an error back. Huge powers like `9 ** 9 ** 9` are refused too.
- `current_datetime`: the date, time and weekday in any timezone (the model has no idea what day it is otherwise).
- `days_between`: days between two dates.
- `list_files` / `read_file` (opt in with `file_tools(folder)` or `chat.py --files FOLDER`): read text files, but only inside that one folder. Paths are fully resolved first, so `../`, absolute paths and symlinks pointing outside are all refused, and long files are cut off at 20k characters.

```python
from gptwrap.tools import DEFAULT_TOOLS

bot = Assistant(tools=DEFAULT_TOOLS)
bot.ask("how many days until christmas?")
```

When the model asks for a tool, the tool runs, the result goes back to the model, and that repeats until it answers. It stops after 5 rounds (`max_tool_steps`) so a confused model can't loop forever. Only the question and the final answer are kept in the history.

If a tool call goes wrong (the model asks for a tool that doesn't exist, passes the wrong arguments, or the tool itself crashes), the error is sent back to the model as the tool's result instead of crashing the chat. The model usually fixes its call or answers without the tool. `chat.py` turns the tools on by default, `--no-tools` turns them off.

### Structured output

Sometimes I don't want prose, I want data I can use in code. `extract()` asks the model to fill in any Pydantic model, and `summarize()` uses a built-in `Summary` (title, one-line tl;dr, key points, action items, tone):

```python
from gptwrap.models import chat_model
from gptwrap.structured import extract, summarize
from pydantic import BaseModel

s = summarize(chat_model(), open("meeting-notes.txt").read())
print(s.action_items)     # a real list, not text to parse
print(s.to_markdown())

class Contact(BaseModel):
    name: str
    email: str | None = None

extract(chat_model(), "write to Aditya at a@example.com", Contact)
```

If the model returns something that doesn't fit the schema (a missing field, a tone that isn't one of the allowed ones), it gets the validation error back and one more try. From the terminal:

```bash
python summarize.py meeting-notes.txt
cat article.txt | python summarize.py --json
```

### Personas

A persona is a system prompt with blanks, so I don't keep rewriting the same "you are a tutor for..." prompt:

```python
tutor = Assistant.from_persona("tutor", subject="chemistry", level="high school")
reviewer = Assistant.from_persona("reviewer", language="Go")
```

`python -m gptwrap --personas` lists them:

```
default      general helpful assistant
tutor        explains a subject step by step at the right level  [level='beginner', subject]
reviewer     code reviewer for one language  [language='Python']
translator   translates into another language, nothing else  [target_language]
interviewer  mock job interviewer  [role='junior software engineer']
```

Variables with a value in brackets have a default. A missing variable, or a misspelled one, raises an error straight away instead of sending a prompt with `{subject}` still in it. You can also make your own with `Persona(name, description, template)` and pass it to `from_persona`.

Tests: `python -m pytest` (no API key or network needed, they use a fake chat model that records the prompts it gets).
