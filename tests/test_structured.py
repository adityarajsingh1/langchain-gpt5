import pytest
from pydantic import BaseModel

from conftest import tool_call
from gptwrap.assistant import AssistantError
from gptwrap.structured import Summary, extract, summarize

GOOD = dict(
    title="Team sync",
    tldr="The launch moves to Friday and Sam fixes the login bug first.",
    key_points=["Launch moved to Friday", "Login bug blocks the release"],
    action_items=["Sam: fix the login bug"],
    sentiment="neutral",
)


def test_summary_comes_back_as_an_object(fake_chat):
    model = fake_chat(tool_call("Summary", **GOOD))
    s = summarize(model, "meeting notes ...")
    assert isinstance(s, Summary)
    assert s.action_items == ["Sam: fix the login bug"]
    assert model.bound_tools == ["Summary"]
    assert model.seen[0][-1].content == "meeting notes ..."


def test_invalid_output_gets_one_retry(fake_chat):
    bad = {**GOOD, "sentiment": "angry", "key_points": []}
    model = fake_chat(tool_call("Summary", **bad), tool_call("Summary", **GOOD))
    assert summarize(model, "notes").sentiment == "neutral"
    retry_prompt = model.seen[1][-1].content
    assert "didn't fit the structure" in retry_prompt and "sentiment" in retry_prompt


def test_gives_up_after_the_retries(fake_chat):
    model = fake_chat(tool_call("Summary", title="only a title"))
    with pytest.raises(AssistantError, match="couldn't get a valid Summary"):
        summarize(model, "notes")
    assert len(model.seen) == 2


def test_any_pydantic_model_works(fake_chat):
    class Contact(BaseModel):
        name: str
        email: str | None = None

    model = fake_chat(tool_call("Contact", name="Aditya", email="a@example.com"))
    contact = extract(model, "write to Aditya at a@example.com", Contact)
    assert contact == Contact(name="Aditya", email="a@example.com")


def test_markdown_rendering():
    md = Summary(**GOOD).to_markdown()
    assert md.startswith("## Team sync") and "- [ ] Sam: fix the login bug" in md and "_tone: neutral_" in md


def test_empty_text():
    with pytest.raises(ValueError):
        summarize(None, "  ")
