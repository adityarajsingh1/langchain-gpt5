from langchain_core.tools import tool

from conftest import tool_call
from gptwrap.assistant import Assistant
from gptwrap.tools import DEFAULT_TOOLS


@tool
def flaky(x: int) -> str:
    """A tool that always breaks."""
    raise RuntimeError("disk on fire")


def last_tool_result(model):
    return model.seen[-1][-1]


def test_unknown_tool_goes_back_to_the_model(fake_chat):
    model = fake_chat(tool_call("weather", city="Berlin"), "Sorry, I can't check the weather.")
    bot = Assistant(model, tools=DEFAULT_TOOLS)
    assert bot.ask("weather in Berlin?") == "Sorry, I can't check the weather."
    result = last_tool_result(model)
    assert result.status == "error"
    assert "no tool called 'weather'" in result.content and "calculator" in result.content


def test_a_crashing_tool_doesnt_end_the_chat(fake_chat):
    model = fake_chat(tool_call("flaky", x=1), "That tool is broken right now.")
    bot = Assistant(model, tools=[flaky])
    assert bot.ask("use flaky") == "That tool is broken right now."
    assert "flaky failed: RuntimeError: disk on fire" in last_tool_result(model).content


def test_wrong_arguments_are_explained(fake_chat):
    model = fake_chat(tool_call("calculator", expr="1+1"), "ok")
    Assistant(model, tools=DEFAULT_TOOLS).ask("1+1")
    result = last_tool_result(model)
    assert result.status == "error"
    assert "bad arguments for calculator" in result.content and "expression" in result.content


def test_the_tool_call_id_is_kept(fake_chat):
    model = fake_chat(tool_call("nope"), "fine")
    Assistant(model, tools=DEFAULT_TOOLS).ask("x")
    assert last_tool_result(model).tool_call_id == "call_nope"
