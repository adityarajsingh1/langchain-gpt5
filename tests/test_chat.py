import subprocess
import sys
from pathlib import Path

import httpx
import openai
import pytest

import chat
from gptwrap.assistant import Assistant, AssistantError
from gptwrap.personas import PersonaError

ROOT = Path(__file__).resolve().parent.parent


class Terminal:
    """Feeds lines to the chat loop and collects what it prints."""

    def __init__(self, *lines):
        self.lines = list(lines)
        self.out = []

    def read(self, prompt=""):
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def write(self, text="", end="\n", flush=False):
        self.out.append(text + end)

    @property
    def text(self):
        return "".join(self.out)


def test_stream_yields_pieces_and_saves_the_turn(fake_chat):
    bot = Assistant(fake_chat("The sky is blue because of scattering."))
    pieces = list(bot.stream("why is the sky blue?"))
    assert len(pieces) > 1
    assert "".join(pieces) == "The sky is blue because of scattering."
    assert [m.content for m in bot.history()] == ["why is the sky blue?", "The sky is blue because of scattering."]


def test_stopping_a_stream_halfway_keeps_nothing(fake_chat):
    bot = Assistant(fake_chat("one two three four"))
    gen = bot.stream("count")
    next(gen)
    gen.close()
    assert bot.history() == []


def test_stream_errors_are_readable(fake_chat):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    bot = Assistant(fake_chat(error=openai.AuthenticationError("no", response=httpx.Response(401, request=req), body=None)))
    with pytest.raises(AssistantError, match="OPENAI_API_KEY"):
        list(bot.stream("hi"))


def test_chat_loop(fake_chat):
    model = fake_chat("Hi Aditya!", "You said your name is Aditya.")
    term = Terminal("hello, I'm Aditya", "what did I say?", "/history", "/reset", "/history", "/quit", "never read")
    chat.chat(Assistant(model), term.read, term.write)
    assert "gpt> Hi Aditya!" in term.text
    assert "you: hello, I'm Aditya" in term.text and "gpt: Hi Aditya!" in term.text
    assert "starting fresh" in term.text and "(nothing yet)" in term.text
    assert len(model.seen) == 2


def test_chat_loop_commands_and_eof(fake_chat):
    term = Terminal("", "/system", "/nope", "/help")  # then EOF, like ctrl+d
    chat.chat(Assistant(fake_chat(), system="be brief"), term.read, term.write)
    assert "be brief" in term.text and "unknown command /nope" in term.text and "/reset" in term.text


def test_errors_dont_end_the_chat(fake_chat):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    model = fake_chat(error=openai.RateLimitError("slow", response=httpx.Response(429, request=req), body=None))
    term = Terminal("hi", "hi again")
    chat.chat(Assistant(model), term.read, term.write)
    assert term.text.count("error: Hit OpenAI's rate limit") == 2


def test_parse_vars():
    assert chat.parse_vars(["subject=physics", "level=high school"]) == {"subject": "physics", "level": "high school"}
    with pytest.raises(PersonaError):
        chat.parse_vars(["subject"])


def test_cli_help_and_missing_key(tmp_path):
    env = {"PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(ROOT / "chat.py"), "--help"], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--persona" in out.stdout
    out = subprocess.run([sys.executable, str(ROOT / "chat.py")], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert out.returncode != 0 and "OPENAI_API_KEY isn't set" in out.stderr


def test_save_and_load_commands(fake_chat, tmp_path):
    path = tmp_path / "chat.json"
    model = fake_chat("Hi Aditya!")
    term = Terminal("hello, I'm Aditya", f"/save {path}", "/reset", f"/load {path}", "/history", "/load", f"/load {tmp_path}/nope.json")
    chat.chat(Assistant(model), term.read, term.write)
    assert f"saved to {path}" in term.text
    assert "loaded 2 messages" in term.text
    assert "you: hello, I'm Aditya" in term.text
    assert "usage: /load FILE" in term.text
    assert "couldn't load:" in term.text and "doesn't exist" in term.text
