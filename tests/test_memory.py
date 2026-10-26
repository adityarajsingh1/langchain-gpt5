import httpx
import openai
import pytest

from gptwrap.assistant import Assistant, AssistantError


def contents(messages):
    return [(m.type, m.content) for m in messages]


def test_remembers_the_conversation(fake_chat):
    model = fake_chat("Nice to meet you, Aditya.", "Your name is Aditya.")
    bot = Assistant(model, system="sys")
    bot.ask("my name is Aditya")
    bot.ask("what's my name?")
    assert contents(model.seen[1]) == [
        ("system", "sys"),
        ("human", "my name is Aditya"),
        ("ai", "Nice to meet you, Aditya."),
        ("human", "what's my name?"),
    ]
    assert len(bot.history()) == 4


def test_sessions_are_kept_apart(fake_chat):
    model = fake_chat()
    bot = Assistant(model)
    bot.ask("I'm alice", session="alice")
    bot.ask("I'm bob", session="bob")
    bot.ask("who am I?", session="alice")
    last = [m.content for m in model.seen[-1]]
    assert "I'm alice" in last and "I'm bob" not in last
    assert sorted(bot.sessions()) == ["alice", "bob"]


def test_reset_one_or_all(fake_chat):
    bot = Assistant(fake_chat())
    bot.ask("a", session="one")
    bot.ask("b", session="two")
    bot.reset("one")
    assert bot.history("one") == [] and len(bot.history("two")) == 2
    bot.reset(None)
    assert bot.sessions() == []


def test_after_reset_the_model_starts_fresh(fake_chat):
    model = fake_chat()
    bot = Assistant(model)
    bot.ask("remember 42")
    bot.reset()
    bot.ask("what number?")
    assert len(model.seen[-1]) == 2  # just system + question


def test_only_the_last_turns_are_kept(fake_chat):
    model = fake_chat(*[f"answer {i}" for i in range(5)])
    bot = Assistant(model, max_turns=2)
    for i in range(5):
        bot.ask(f"question {i}")
    assert [m.content for m in bot.history()] == ["question 3", "answer 3", "question 4", "answer 4"]
    # the last call saw the two turns before it, not all four
    assert [m.content for m in model.seen[-1]][1:] == ["question 2", "answer 2", "question 3", "answer 3", "question 4"]


def test_failed_call_is_not_saved(fake_chat):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    model = fake_chat(error=openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None))
    bot = Assistant(model)
    with pytest.raises(AssistantError):
        bot.ask("hello?")
    assert bot.history() == []
