"""`python -m gptwrap` shows which settings would be used, without calling the API."""

import argparse
import sys

from gptwrap.config import ConfigError, load_settings


def main():
    parser = argparse.ArgumentParser(prog="gptwrap", description="Check the GPT-5 settings (no API call is made).")
    parser.parse_args()
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
