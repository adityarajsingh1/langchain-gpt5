"""
Chat with GPT-5 in the terminal. Answers stream in as they're written.

    python chat.py
    python chat.py --persona tutor --set subject=physics --set level="high school"
    python chat.py --no-tools
    python chat.py --resume chats/physics.json
    python chat.py --files ~/notes      # it can read files in ~/notes, nothing else

Commands while chatting: /reset, /history, /system, /save, /load, /help, /quit
"""

import argparse
import sys

from gptwrap.assistant import Assistant, AssistantError
from gptwrap.config import ConfigError
from gptwrap.files import file_tools
from gptwrap.personas import BUILT_IN, PersonaError
from gptwrap.storage import ConversationFileError, load_conversation, save_conversation
from gptwrap.tools import DEFAULT_TOOLS

HELP = """commands:
  /reset     forget the conversation so far
  /history   show the conversation so far
  /system    show the system prompt
  /save FILE save the conversation as JSON
  /load FILE load a saved conversation (replaces the current one)
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
            elif cmd in ("/save", "/load"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    write(f"usage: {cmd} FILE")
                elif cmd == "/save":
                    try:
                        write(f"saved to {save_conversation(bot, parts[1])}")
                    except OSError as e:
                        write(f"couldn't save: {e}")
                else:
                    try:
                        n = load_conversation(bot, parts[1])
                        write(f"loaded {n} messages, as '{bot.persona or 'default'}'.")
                    except ConversationFileError as e:
                        write(f"couldn't load: {e}")
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
    parser.add_argument("--no-tools", action="store_true", help="don't give the model the calculator and date tools")
    parser.add_argument("--files", metavar="FOLDER", help="let the model list and read files in this folder (and only there)")
    parser.add_argument("--resume", metavar="FILE", help="carry on a conversation saved with /save")
    parser.add_argument("--max-turns", type=int, default=20, help="earlier turns sent along with each question")
    args = parser.parse_args()

    try:
        tools = [] if args.no_tools else list(DEFAULT_TOOLS)
        if args.files:
            tools += file_tools(args.files)
        bot = Assistant.from_persona(args.persona, tools=tools, **parse_vars(args.set))
    except (ConfigError, PersonaError, ValueError) as e:
        sys.exit(f"error: {e}")
    bot.max_turns = args.max_turns
    if args.resume:
        try:
            load_conversation(bot, args.resume)
        except ConversationFileError as e:
            sys.exit(f"error: {e}")
    chat(bot)


if __name__ == "__main__":
    main()
