import json

import pytest

from gptwrap.assistant import Assistant
from gptwrap.storage import ConversationFileError, load_conversation, read_conversation, save_conversation


def test_round_trip(fake_chat, tmp_path):
    bot = Assistant.from_persona("tutor", fake_chat("F = ma.", "Mass times acceleration."), subject="physics")
    bot.ask("newton's second law?")
    bot.ask("in words?")
    path = save_conversation(bot, tmp_path / "chats" / "physics.json")

    fresh = Assistant(fake_chat("It's about force."))
    assert load_conversation(fresh, path) == 4
    assert fresh.persona == "tutor" and "physics tutor" in fresh.system
    assert [m.content for m in fresh.history()] == [m.content for m in bot.history()]

    # the next question sees the loaded conversation
    fresh.ask("and the first law?")
    sent = [m.content for m in fresh.model.seen[-1]]
    assert sent[1:4] == ["newton's second law?", "F = ma.", "in words?"]


def test_file_is_plain_readable_json(fake_chat, tmp_path):
    bot = Assistant(fake_chat("héllo 👋"))
    bot.ask("hi")
    data = json.loads(save_conversation(bot, tmp_path / "c.json").read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["messages"][1] == {"role": "ai", "content": "héllo 👋"}
    assert list(tmp_path.iterdir()) == [tmp_path / "c.json"]  # no temp files left behind


def test_sessions_and_keeping_the_current_system_prompt(fake_chat, tmp_path):
    bot = Assistant(fake_chat("b"), system="original")
    bot.ask("a", session="work")
    path = save_conversation(bot, tmp_path / "work.json", session="work")
    other = Assistant(fake_chat(), system="mine")
    load_conversation(other, path, session="restored", use_system=False)
    assert other.system == "mine" and len(other.history("restored")) == 2 and other.history() == []


@pytest.mark.parametrize("content, message", [
    (None, "doesn't exist"),
    ("{not json", "isn't valid JSON"),
    ('{"version": 99, "messages": []}', "this version can read"),
    ('{"version": 1, "messages": [{"role": "system", "content": "x"}]}', "message 0"),
    ('{"version": 1, "messages": [{"role": "ai"}]}', "message 0"),
])
def test_bad_files_give_clear_errors(tmp_path, content, message):
    path = tmp_path / "c.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(ConversationFileError, match=message):
        read_conversation(path)


def test_set_history_rejects_other_message_types(fake_chat):
    from langchain_core.messages import SystemMessage
    with pytest.raises(ValueError):
        Assistant(fake_chat()).set_history([SystemMessage("x")])
