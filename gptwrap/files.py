"""
Tools that let the model read files, but only inside one folder.

Paths are resolved (so "..", absolute paths and symlinks all end up as real
locations) and anything that lands outside the folder is refused.
"""

from pathlib import Path

from langchain_core.tools import StructuredTool

MAX_CHARS = 20_000  # roughly 5k tokens, enough for most source files without blowing the context


class OutsideFolderError(ValueError):
    pass


def _inside(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise OutsideFolderError(f"'{relative}' is outside the folder I'm allowed to read")
    return target


def file_tools(folder, max_chars=MAX_CHARS):
    """list_files and read_file, both locked to `folder`."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{folder} isn't a folder")

    def list_files(subfolder: str = ".") -> str:
        """List the files and folders in a folder (relative to the shared folder). Folders end with /."""
        target = _inside(root, subfolder)
        if not target.is_dir():
            raise ValueError(f"'{subfolder}' isn't a folder")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [p.name + ("/" if p.is_dir() else "") for p in entries if not p.name.startswith(".")]
        return "\n".join(lines) or "(empty folder)"

    def read_file(path: str) -> str:
        """Read a text file (path relative to the shared folder). Long files are cut off."""
        target = _inside(root, path)
        if not target.is_file():
            raise ValueError(f"'{path}' doesn't exist or isn't a file")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"'{path}' isn't a text file") from None
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[cut off: the file has {len(text)} characters, showing the first {max_chars}]"
        return text

    return [StructuredTool.from_function(list_files), StructuredTool.from_function(read_file)]
