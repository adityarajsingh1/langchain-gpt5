from datetime import datetime

import pytest

from conftest import tool_call
from gptwrap import tools
from gptwrap.assistant import Assistant, AssistantError
from gptwrap.tools import DEFAULT_TOOLS, calculate, calculator, current_datetime, days_between


@pytest.mark.parametrize("expr, result", [
    ("17 * 23", 391),
    ("2^10", 1024),
    ("(1 + 0.07) ** 10", 1.9671513572895665),
    ("sqrt(16) + abs(-3)", 7),
    ("10 // 3 + 10 % 3", 4),
    ("round(pi, 2)", 3.14),
    ("-5 + +2", -3),
])
def test_calculate(expr, result):
    assert calculate(expr) == result


@pytest.mark.parametrize("expr, message", [
    ("__import__('os').system('ls')", "isn't allowed"),
    ("open('secrets.txt')", "isn't allowed"),
    ("x + 1", "isn't allowed"),
    ("9 ** 9 ** 9", "too big"),
    ("1 / 0", "division by zero"),
    ("2 +", "couldn't read"),
    ("1" * 300, "too long"),
])
def test_calculator_refuses_bad_input(expr, message):
    with pytest.raises(ValueError, match=message):
        calculate(expr)
    assert calculator.invoke({"expression": expr}).startswith("error:")


def test_datetime_tools(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda tz: datetime(2026, 10, 28, 8, 47, tzinfo=tz))
    assert current_datetime.invoke({"timezone": "Europe/Berlin"}) == "2026-10-28 08:47 (Wednesday) in Europe/Berlin"
    assert "unknown timezone" in current_datetime.invoke({"timezone": "Mars/Base"})
    assert days_between.invoke({"start": "2026-10-28", "end": "2026-12-25"}) == "58 days"
    assert days_between.invoke({"start": "2026-10-28", "end": "soon"}).startswith("error:")


def test_tool_loop_runs_the_tool_and_sends_the_result_back(fake_chat):
    model = fake_chat(tool_call("calculator", expression="17 * 23"), "17 times 23 is 391.")
    bot = Assistant(model, tools=DEFAULT_TOOLS)
    assert model.bound_tools == ["calculator", "current_datetime", "days_between"]

    assert bot.ask("what's 17 * 23?") == "17 times 23 is 391."
    second_call = model.seen[1]
    assert second_call[-2].tool_calls[0]["name"] == "calculator"
    assert second_call[-1].type == "tool" and second_call[-1].content == "391"
    # only the question and the final answer go into the history, not the tool chatter
    assert [m.content for m in bot.history()] == ["what's 17 * 23?", "17 times 23 is 391."]


def test_several_tool_rounds_and_streaming(fake_chat, monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda tz: datetime(2026, 10, 28, 8, 47, tzinfo=tz))
    model = fake_chat(
        tool_call("current_datetime", timezone="Asia/Kolkata"),
        tool_call("days_between", start="2026-10-28", end="2026-12-25"),
        "It's 58 days until Christmas.",
    )
    bot = Assistant(model, tools=DEFAULT_TOOLS)
    pieces = list(bot.stream("how many days until christmas?"))
    assert "".join(pieces) == "It's 58 days until Christmas." and len(pieces) > 1
    assert model.seen[2][-1].content == "58 days"
    assert "Wednesday" in model.seen[1][-1].content


def test_gives_up_when_the_model_keeps_calling_tools(fake_chat):
    model = fake_chat(tool_call("calculator", expression="1 + 1"))  # repeats forever
    bot = Assistant(model, tools=DEFAULT_TOOLS, max_tool_steps=3)
    with pytest.raises(AssistantError, match="gave up after 3"):
        bot.ask("loop please")
    assert len(model.seen) == 4 and bot.history() == []


def test_no_tools_means_nothing_is_bound(fake_chat):
    model = fake_chat("hi")
    Assistant(model).ask("hello")
    assert model.bound_tools == []
