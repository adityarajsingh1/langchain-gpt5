"""Turn messy meeting notes into a checklist of who does what."""

import argparse
import sys
from pathlib import Path

# so the examples run straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from gptwrap.models import chat_model
from gptwrap.structured import extract


class Task(BaseModel):
    owner: str = Field(description="who does it, 'unassigned' if nobody was named")
    task: str
    due: str | None = Field(default=None, description="deadline as written in the notes, if any")


class Tasks(BaseModel):
    tasks: list[Task]


def main(argv=None, model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", help="a text file with the meeting notes")
    args = parser.parse_args(argv)

    with open(args.notes, encoding="utf-8") as f:
        result = extract(model or chat_model(), f.read(), Tasks)
    for t in sorted(result.tasks, key=lambda t: t.owner.lower()):
        due = f" (by {t.due})" if t.due else ""
        print(f"- [ ] {t.owner}: {t.task}{due}")
    return result


if __name__ == "__main__":
    main()
