"""
Settings, read from environment variables (or a .env file next to where you run it).

    OPENAI_API_KEY     required
    OPENAI_MODEL       which model to talk to       (default gpt-5)
    REASONING_EFFORT   minimal / low / medium / high (default: the model's own default)
    OPENAI_TIMEOUT     seconds before giving up      (default 60)
    OPENAI_BASE_URL    only if you go through a proxy or a compatible server
"""

import os
from dataclasses import dataclass
from pathlib import Path

EFFORTS = ("minimal", "low", "medium", "high")


class ConfigError(RuntimeError):
    pass


def load_dotenv(path=".env"):
    # KEY=value per line, # for comments. variables already set in the shell win
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = "gpt-5"
    reasoning_effort: str | None = None
    timeout: float = 60
    base_url: str | None = None

    def masked_key(self):
        return self.api_key[:3] + "..." + self.api_key[-4:] if len(self.api_key) > 10 else "***"


def load_settings(env=None):
    """Build Settings from the environment. Raises ConfigError with a readable message if something's off."""
    if env is None:
        load_dotenv()
        env = os.environ

    key = env.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY isn't set. Get a key at https://platform.openai.com/api-keys, then either "
            "`export OPENAI_API_KEY=sk-...` or put it in a .env file (see .env.example)."
        )

    effort = env.get("REASONING_EFFORT", "").strip().lower() or None
    if effort and effort not in EFFORTS:
        raise ConfigError(f"REASONING_EFFORT must be one of {', '.join(EFFORTS)}, got '{effort}'.")

    try:
        timeout = float(env.get("OPENAI_TIMEOUT", "60"))
    except ValueError:
        raise ConfigError(f"OPENAI_TIMEOUT should be a number of seconds, got '{env['OPENAI_TIMEOUT']}'.") from None

    return Settings(
        api_key=key,
        model=env.get("OPENAI_MODEL", "").strip() or "gpt-5",
        reasoning_effort=effort,
        timeout=timeout,
        base_url=env.get("OPENAI_BASE_URL", "").strip() or None,
    )
