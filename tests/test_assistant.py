import httpx
import openai
import pytest

from gptwrap.assistant import DEFAULT_SYSTEM, Assistant, AssistantError


def test_sends_system_prompt_then_question(fake_chat):
    model = fake_chat("  Paris.  ")
    bot = Assistant(model, system="Answer in one word.")
    assert bot.ask("Capital of France?") == "Paris."

    (messages,) = model.seen
    assert [m.type for m in messages] == ["system", "human"]
    assert messages[0].content == "Answer in one word."
    assert messages[1].content == "Capital of France?"


def test_default_system_prompt(fake_chat):
    model = fake_chat()
    Assistant(model).ask("hi")
    assert model.seen[0][0].content == DEFAULT_SYSTEM


def test_follow_up_questions_see_earlier_turns(fake_chat):
    model = fake_chat("one", "two")
    bot = Assistant(model)
    assert bot.ask("first") == "one"
    assert bot.ask("second") == "two"
    assert [m.content for m in model.seen[1]][1:] == ["first", "one", "second"]


def test_braces_in_questions_are_not_template_variables(fake_chat):
    model = fake_chat()
    Assistant(model, system="Reply with {json}").ask("what does {name} mean in python?")
    assert model.seen[0][0].content == "Reply with {json}"
    assert model.seen[0][1].content == "what does {name} mean in python?"


def test_empty_question(fake_chat):
    with pytest.raises(ValueError):
        Assistant(fake_chat()).ask("   ")


def _status_error(cls, status):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


@pytest.mark.parametrize("err, hint", [
    (_status_error(openai.AuthenticationError, 401), "OPENAI_API_KEY"),
    (_status_error(openai.NotFoundError, 404), "OPENAI_MODEL"),
    (_status_error(openai.RateLimitError, 429), "rate limit"),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")), "Couldn't reach OpenAI"),
])
def test_openai_errors_become_readable(fake_chat, err, hint):
    with pytest.raises(AssistantError, match=hint):
        Assistant(fake_chat(error=err)).ask("hi")


def test_defaults_to_gpt5_from_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert Assistant().model.model_name == "gpt-5"
