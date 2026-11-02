"""A tutor that remembers where you left off: the chat is saved to JSON after every answer."""

import argparse
import sys
from pathlib import Path

# so the examples run straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import chat
from gptwrap.assistant import Assistant
from gptwrap.storage import load_conversation, save_conversation


class SavingAssistant(Assistant):
    """Saves the conversation after every answer, so closing the terminal loses nothing."""

    save_to: Path

    def stream(self, question, session="default"):
        yield from super().stream(question, session)
        save_conversation(self, self.save_to, session)


def main(argv=None, model=None, read=input, write=print):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="e.g. 'organic chemistry'")
    parser.add_argument("--level", default="beginner")
    parser.add_argument("--file", default=None, help="where to keep the chat (default: study-<subject>.json)")
    args = parser.parse_args(argv)

    bot = SavingAssistant.from_persona("tutor", model, subject=args.subject, level=args.level)
    bot.save_to = Path(args.file or f"study-{args.subject.replace(' ', '-')}.json")
    if bot.save_to.exists():
        n = load_conversation(bot, bot.save_to)
        write(f"welcome back, loaded {n} messages from {bot.save_to}")
    chat(bot, read, write)
    return bot


if __name__ == "__main__":
    main()
