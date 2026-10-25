import subprocess
import sys
from pathlib import Path

import pytest

from gptwrap.assistant import DEFAULT_SYSTEM, Assistant
from gptwrap.personas import BUILT_IN, Persona, PersonaError, get_persona

ROOT = Path(__file__).resolve().parent.parent


def test_fills_in_variables_and_defaults():
    tutor = get_persona("tutor")
    assert tutor.variables == ["level", "subject"]
    text = tutor.render(subject="chemistry")
    assert "chemistry tutor for a beginner student" in text
    assert "advanced student" in tutor.render(subject="maths", level="advanced")


def test_missing_and_unknown_variables():
    with pytest.raises(PersonaError, match="needs a value for: target_language"):
        get_persona("translator").render()
    with pytest.raises(PersonaError, match="has no variable"):
        get_persona("reviewer").render(langauge="Go")  # typo should not be silently ignored


def test_unknown_persona_lists_the_real_ones():
    with pytest.raises(PersonaError, match="available: default, tutor"):
        get_persona("pirate")


def test_every_built_in_renders_with_example_values():
    for p in BUILT_IN.values():
        values = {v: "x" for v in p.variables if v not in p.defaults}
        assert "{" not in p.render(**values)


def test_assistant_from_persona_uses_it_as_system_prompt(fake_chat):
    model = fake_chat("Bonjour")
    bot = Assistant.from_persona("translator", model, target_language="French")
    assert bot.persona == "translator"
    assert bot.ask("Hello") == "Bonjour"
    system, human = model.seen[0]
    assert "into French" in system.content and human.content == "Hello"


def test_custom_persona(fake_chat):
    model = fake_chat()
    pirate = Persona("pirate", "talks like a pirate", "Talk like a pirate from {place}.")
    Assistant.from_persona(pirate, model, place="Cornwall").ask("hi")
    assert model.seen[0][0].content == "Talk like a pirate from Cornwall."


def test_plain_assistant_still_has_the_default_prompt(fake_chat):
    bot = Assistant(fake_chat())
    assert bot.system == DEFAULT_SYSTEM and bot.persona is None


def test_cli_lists_personas_without_a_key(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "gptwrap", "--personas"], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "tutor" in out.stdout and "level='beginner'" in out.stdout and "target_language" in out.stdout
