"""One assistant answering several customers, each with their own conversation memory."""

import sys
from pathlib import Path

# so the examples run straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptwrap.assistant import Assistant

SYSTEM = ("You are the support assistant for a small bike shop. Opening hours: Mon-Sat 9-18. "
          "Repairs take 2-3 days. Be friendly and brief. If you don't know something, say a human will follow up.")

CONVERSATIONS = [
    ("anna", "Hi, are you open on Sunday?"),
    ("ben", "My chain keeps slipping, how long would a repair take?"),
    ("anna", "OK, and what about Saturday?"),  # Anna's follow-up, Ben's question is not in her context
]


def main(model=None):
    bot = Assistant(model, system=SYSTEM)
    replies = []
    for customer, message in CONVERSATIONS:
        reply = bot.ask(message, session=customer)
        replies.append((customer, reply))
        print(f"[{customer}] {message}\n  -> {reply}\n")
    return bot, replies


if __name__ == "__main__":
    main()
