"""
Save conversations to JSON and load them back, so a chat can be picked up
later (or read by other tools, it's just JSON).

File format, version 1:

    {
      "version": 1,
      "saved_at": "2026-10-30T08:47:00+00:00",
      "persona": "tutor",
      "system": "You are a patient physics tutor...",
      "messages": [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}]
    }
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

VERSION = 1
_ROLES = {"human": HumanMessage, "ai": AIMessage}


class ConversationFileError(ValueError):
    pass


def save_conversation(bot, path, session="default"):
    """Write one session to `path`. Written to a temp file first, so a crash never leaves half a file."""
    data = {
        "version": VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "persona": bot.persona,
        "system": bot.system,
        "messages": [{"role": m.type, "content": m.content} for m in bot.history(session)],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    return path


def read_conversation(path):
    """Load and check a saved file. Returns (data, messages)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConversationFileError(f"{path} doesn't exist") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConversationFileError(f"{path} isn't valid JSON: {e}") from None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise ConversationFileError(f"{path} isn't a conversation file this version can read")

    messages = []
    for i, m in enumerate(data.get("messages", [])):
        if not isinstance(m, dict) or m.get("role") not in _ROLES or not isinstance(m.get("content"), str):
            raise ConversationFileError(f"message {i} in {path} should look like {{\"role\": \"human\"|\"ai\", \"content\": \"...\"}}")
        messages.append(_ROLES[m["role"]](m["content"]))
    return data, messages


def load_conversation(bot, path, session="default", use_system=True):
    """Put a saved conversation into `bot`'s session. By default the saved system prompt comes back too."""
    data, messages = read_conversation(path)
    bot.set_history(messages, session)
    if use_system and data.get("system"):
        bot.system = data["system"]
        bot.persona = data.get("persona")
    return len(messages)
