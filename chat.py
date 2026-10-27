"""
Chat with GPT-5 in the terminal. Answers stream in as they're written.

    python chat.py
    python chat.py --persona tutor --set subject=physics --set level="high school"

Commands while chatting: /reset, /history, /system, /help, /quit
"""

import argparse
import sys

from gptwrap.assistant import Assistant, AssistantError
from gptwrap.config import ConfigError
from gptwrap.personas import BUILT_IN, PersonaError

HELP = """commands:
  /reset     forget the conversation so far
  /history   show the conversation so far
  /system    show the system prompt
  /quit      exit (ctrl+d works too, ctrl+c stops an answer halfway)"""


def parse_vars(pairs):
    values = {}
    for pair in pairs:
        if "=" not in pair:
            raise PersonaError(f"--set wants name=value, got '{pair}'")
        name, value = pair.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def chat(bot, read=input, write=print):
    """The chat loop. read/write are swappable so the tests can drive it."""
    write(f"chatting as '{bot.persona or 'default'}'. /help for commands, /quit to leave.")
    while True:
        try:
            line = read("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return
        if not line:
            continue

        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd in ("/quit", "/exit"):
                return
            elif cmd == "/reset":
                bot.reset()
                write("ok, starting fresh.")
            elif cmd == "/history":
                for m in bot.history() or []:
                    write(f"{'you' if m.type == 'human' else 'gpt'}: {m.content}")
                if not bot.history():
                    write("(nothing yet)")
            elif cmd == "/system":
                write(bot.system)
            elif cmd == "/help":
                write(HELP)
            else:
                write(f"unknown command {cmd}. /help lists them.")
            continue

        write("\ngpt> ", end="", flush=True)
        try:
            for piece in bot.stream(line):
                write(piece, end="", flush=True)
            write("")
        except KeyboardInterrupt:
            write("\n(stopped. that answer isn't kept in the history)")
        except AssistantError as e:
            write(f"\nerror: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chat with GPT-5 in the terminal.")
    parser.add_argument("--persona", default="default", choices=sorted(BUILT_IN),
                        help="which built-in persona to use (see python -m gptwrap --personas)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="fill in a persona variable, e.g. --set subject=physics (repeatable)")
    parser.add_argument("--max-turns", type=int, default=20, help="earlier turns sent along with each question")
    args = parser.parse_args()

    try:
        bot = Assistant.from_persona(args.persona, **parse_vars(args.set))
    except (ConfigError, PersonaError) as e:
        sys.exit(f"error: {e}")
    bot.max_turns = args.max_turns
    chat(bot)


if __name__ == "__main__":
    main()
