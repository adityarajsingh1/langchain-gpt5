import os

import pytest

from conftest import tool_call
from gptwrap.assistant import Assistant
from gptwrap.files import OutsideFolderError, file_tools


@pytest.fixture
def shared(tmp_path):
    root = tmp_path / "shared"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "todo.md").write_text("- buy milk\n")
    (root / "readme.txt").write_text("hello")
    (root / ".secret").write_text("hidden")
    (root / "image.bin").write_bytes(b"\xff\xfe\x00\x81")
    (tmp_path / "passwords.txt").write_text("hunter2")
    return root


def tools_by_name(folder, **kw):
    return {t.name: t for t in file_tools(folder, **kw)}


def test_list_and_read(shared):
    t = tools_by_name(shared)
    assert t["list_files"].invoke({}) == "notes/\nimage.bin\nreadme.txt"
    assert ".secret" not in t["list_files"].invoke({})
    assert t["list_files"].invoke({"subfolder": "notes"}) == "todo.md"
    assert t["read_file"].invoke({"path": "notes/todo.md"}) == "- buy milk\n"


@pytest.mark.parametrize("path", ["../passwords.txt", "notes/../../passwords.txt", "/etc/passwd"])
def test_cant_leave_the_folder(shared, path):
    with pytest.raises(OutsideFolderError):
        tools_by_name(shared)["read_file"].invoke({"path": path})


def test_symlinks_out_are_refused(shared):
    os.symlink(shared.parent / "passwords.txt", shared / "link.txt")
    with pytest.raises(OutsideFolderError):
        tools_by_name(shared)["read_file"].invoke({"path": "link.txt"})


def test_binary_missing_and_long_files(shared):
    t = tools_by_name(shared, max_chars=3)
    with pytest.raises(ValueError, match="isn't a text file"):
        t["read_file"].invoke({"path": "image.bin"})
    with pytest.raises(ValueError, match="doesn't exist"):
        t["read_file"].invoke({"path": "nope.txt"})
    assert t["read_file"].invoke({"path": "readme.txt"}).startswith("hel\n\n[cut off: the file has 5 characters")


def test_folder_must_exist(tmp_path):
    with pytest.raises(ValueError, match="isn't a folder"):
        file_tools(tmp_path / "missing")


def test_escape_attempt_reaches_the_model_as_an_error(fake_chat, shared):
    model = fake_chat(tool_call("read_file", path="../passwords.txt"), "I can't read files outside that folder.")
    bot = Assistant(model, tools=file_tools(shared))
    assert bot.ask("read ../passwords.txt") == "I can't read files outside that folder."
    result = model.seen[-1][-1]
    assert result.status == "error" and "outside the folder" in result.content
    assert "hunter2" not in result.content
