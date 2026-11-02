"""Review a source file with the `reviewer` persona. The model can only read files in that file's folder."""

import argparse
import sys
from pathlib import Path

# so the examples run straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptwrap.assistant import Assistant
from gptwrap.files import file_tools

LANGUAGES = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".go": "Go", ".rs": "Rust", ".java": "Java"}


def main(argv=None, model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    args = parser.parse_args(argv)

    path = Path(args.file).resolve()
    language = LANGUAGES.get(path.suffix, "Python")
    bot = Assistant.from_persona("reviewer", model, tools=file_tools(path.parent), language=language)
    review = bot.ask(f"Review {path.name}. Read it with read_file first; you may open other files it imports.")
    print(review)
    return review


if __name__ == "__main__":
    main()
