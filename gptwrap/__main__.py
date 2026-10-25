"""`python -m gptwrap` shows which settings would be used, without calling the API.
`python -m gptwrap --personas` lists the built-in personas."""

import argparse
import sys

from gptwrap.config import ConfigError, load_settings
from gptwrap.personas import BUILT_IN


def main():
    parser = argparse.ArgumentParser(prog="gptwrap", description="Check the GPT-5 settings (no API call is made).")
    parser.add_argument("--personas", action="store_true", help="list the built-in personas and their variables")
    args = parser.parse_args()
    if args.personas:
        for p in BUILT_IN.values():
            variables = ", ".join(f"{v}={p.defaults[v]!r}" if v in p.defaults else v for v in p.variables)
            print(f"{p.name:<12} {p.description}" + (f"  [{variables}]" if variables else ""))
        return
    try:
        s = load_settings()
    except ConfigError as e:
        sys.exit(f"error: {e}")
    print(f"model:            {s.model}")
    print(f"api key:          {s.masked_key()}")
    print(f"reasoning effort: {s.reasoning_effort or 'model default'}")
    print(f"timeout:          {s.timeout:g}s")
    if s.base_url:
        print(f"base url:         {s.base_url}")


if __name__ == "__main__":
    main()
