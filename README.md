# langchain-gpt5

A small wrapper around GPT-5 built on LangChain: an assistant with personas, memory, tools and structured output, that I can drop into other projects instead of rewriting the same glue every time.

**Work in progress.** I'm building it step by step. So far there's only the setup: settings come from environment variables (or a `.env` file), and the LangChain `ChatOpenAI` model is built from them.

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

Tests: `python -m pytest` (no API key or network needed).
