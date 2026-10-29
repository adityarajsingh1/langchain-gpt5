import json

import pytest

import summarize
from conftest import tool_call

SUMMARY = dict(title="Notes", tldr="Short.", key_points=["one"], action_items=[], sentiment="positive")


def test_markdown_from_a_file(fake_chat, tmp_path, capsys):
    f = tmp_path / "notes.txt"
    f.write_text("we shipped it")
    summarize.main([str(f)], model=fake_chat(tool_call("Summary", **SUMMARY)))
    assert capsys.readouterr().out.startswith("## Notes")


def test_json_from_stdin(fake_chat, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("we shipped it"))
    summarize.main(["--json"], model=fake_chat(tool_call("Summary", **SUMMARY)))
    assert json.loads(capsys.readouterr().out)["key_points"] == ["one"]


def test_missing_file_and_empty_text(fake_chat, tmp_path):
    with pytest.raises(SystemExit, match="couldn't read"):
        summarize.main([str(tmp_path / "nope.txt")], model=fake_chat())
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    with pytest.raises(SystemExit, match="no text"):
        summarize.main([str(empty)], model=fake_chat())
