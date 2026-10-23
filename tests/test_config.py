import subprocess
import sys
from pathlib import Path

import pytest

from gptwrap.config import ConfigError, load_settings
from gptwrap.models import chat_model

ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-test-1234567890abcdef"


def test_defaults():
    s = load_settings({"OPENAI_API_KEY": KEY})
    assert s.model == "gpt-5"
    assert s.reasoning_effort is None
    assert s.timeout == 60
    assert s.masked_key() == "sk-...cdef"


def test_missing_key_explains_what_to_do():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY isn't set"):
        load_settings({"OPENAI_API_KEY": "  "})


def test_bad_values_are_caught_early():
    with pytest.raises(ConfigError, match="REASONING_EFFORT"):
        load_settings({"OPENAI_API_KEY": KEY, "REASONING_EFFORT": "max"})
    with pytest.raises(ConfigError, match="OPENAI_TIMEOUT"):
        load_settings({"OPENAI_API_KEY": KEY, "OPENAI_TIMEOUT": "soon"})


def test_chat_model_uses_the_settings():
    s = load_settings({"OPENAI_API_KEY": KEY, "OPENAI_MODEL": "gpt-5-mini", "REASONING_EFFORT": "Low"})
    model = chat_model(s)  # building it doesn't call the API
    assert model.model_name == "gpt-5-mini"
    assert model.reasoning_effort == "low"
    assert model.openai_api_key.get_secret_value() == KEY


def test_cli_without_key(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "gptwrap"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "OPENAI_API_KEY isn't set" in out.stderr


def test_cli_reads_dotenv(tmp_path):
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={KEY}\nOPENAI_MODEL=gpt-5-nano\n")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "gptwrap"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "gpt-5-nano" in out.stdout and "sk-...cdef" in out.stdout and KEY not in out.stdout
