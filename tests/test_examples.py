import sys
from pathlib import Path

from conftest import tool_call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import code_review  # noqa: E402
import meeting_actions  # noqa: E402
import study_buddy  # noqa: E402
import support_desk  # noqa: E402


def test_code_review_reads_the_file(fake_chat, tmp_path, capsys):
    f = tmp_path / "app.py"
    f.write_text("def add(a, b):\n    return a - b\n")
    model = fake_chat(tool_call("read_file", path="app.py"), "Bug: add() subtracts.")
    assert code_review.main([str(f)], model=model) == "Bug: add() subtracts."
    assert "return a - b" in model.seen[-1][-1].content
    assert "senior Python developer" in model.seen[0][0].content


def test_meeting_actions(fake_chat, tmp_path, capsys):
    notes = tmp_path / "notes.txt"
    notes.write_text("sam fixes login by friday, someone should update the docs")
    tasks = [{"owner": "Sam", "task": "fix login", "due": "Friday"}, {"owner": "unassigned", "task": "update docs"}]
    meeting_actions.main([str(notes)], model=fake_chat(tool_call("Tasks", tasks=tasks)))
    out = capsys.readouterr().out
    assert "- [ ] Sam: fix login (by Friday)" in out and "- [ ] unassigned: update docs" in out


def test_support_desk_keeps_customers_apart(fake_chat, capsys):
    bot, replies = support_desk.main(model=fake_chat("No, closed Sundays.", "2-3 days.", "Yes, 9-18."))
    assert [c for c, _ in replies] == ["anna", "ben", "anna"]
    anna_last_prompt = [m.content for m in bot.model.seen[-1]]
    assert "Hi, are you open on Sunday?" in anna_last_prompt
    assert not any("chain" in c for c in anna_last_prompt)


def reader(*lines):
    """Gives the chat loop these lines, then behaves like ctrl+d."""
    pending = list(lines)

    def read(prompt=""):
        if not pending:
            raise EOFError
        return pending.pop(0)
    return read


def test_study_buddy_saves_and_resumes(fake_chat, tmp_path):
    path = tmp_path / "study.json"
    study_buddy.main(["chemistry", "--file", str(path)], model=fake_chat("6.022e23 particles."),
                     read=reader("what is a mole?"), write=lambda *a, **k: None)
    assert path.exists()

    out = []
    bot = study_buddy.main(["chemistry", "--file", str(path)], model=fake_chat(),
                           read=reader(), write=lambda *a, **k: out.append(a))
    assert any("loaded 2 messages" in str(a) for a in out)
    assert len(bot.history()) == 2
