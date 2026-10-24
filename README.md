# langchain-gpt5

A small wrapper around GPT-5 built on LangChain: an assistant with personas, memory, tools and structured output, that I can drop into other projects instead of rewriting the same glue every time.

**Work in progress.** I'm building it step by step. So far: settings come from environment variables (or a `.env` file), the LangChain `ChatOpenAI` model is built from them, and there's an `Assistant` class on top.

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

`Assistant` takes any LangChain chat model as its first argument and uses GPT-5 if you don't pass one. Each question is answered on its own for now (no memory yet). If the API call fails, you get an `AssistantError` with one line on what to check (key, model name, rate limit, connection) instead of the raw SDK exception.

Tests: `python -m pytest` (no API key or network needed, they use a fake chat model that records the prompts it gets).
