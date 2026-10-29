"""
Summarise a text file (or stdin) into a structured summary.

    python summarize.py meeting-notes.txt
    cat article.txt | python summarize.py --json
"""

import argparse
import sys

from gptwrap.assistant import AssistantError
from gptwrap.config import ConfigError
from gptwrap.models import chat_model
from gptwrap.structured import summarize


def main(argv=None, model=None):
    parser = argparse.ArgumentParser(description="Summarise text into title, key points and to-dos with GPT-5.")
    parser.add_argument("file", nargs="?", help="text file to summarise (reads stdin if left out)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args(argv)

    try:
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()
        summary = summarize(model or chat_model(), text)
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"error: couldn't read {args.file}: {e}")
    except (ConfigError, AssistantError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(summary.model_dump_json(indent=2) if args.json else summary.to_markdown())


if __name__ == "__main__":
    main()
