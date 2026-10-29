import sys
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
import json
import re

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RecordingChat(BaseChatModel):
    """Fake chat model: answers from a list (or raises), and remembers every prompt it got."""

    replies: list = []
    error: Exception | None = None
    seen: list = []
    bound_tools: list = []

    @property
    def _llm_type(self):
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        if self.error:
            raise self.error
        reply = self._next_reply()
        message = reply if isinstance(reply, AIMessage) else AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # streams the reply word by word (keeping the spaces), like a real model would
        self.seen.append(messages)
        if self.error:
            raise self.error
        reply = self._next_reply()
        if isinstance(reply, AIMessage):  # a tool call comes in one chunk
            chunks = [{"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                      for i, c in enumerate(reply.tool_calls)]
            yield ChatGenerationChunk(message=AIMessageChunk(content=reply.content, tool_call_chunks=chunks))
            return
        for piece in re.findall(r"\S+\s*|\s+", reply):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    def bind_tools(self, tools, **kwargs):
        # tools can be LangChain tools or plain Pydantic classes (structured output)
        self.bound_tools = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self

    def _next_reply(self):
        return self.replies[min(len(self.seen), len(self.replies)) - 1]


@pytest.fixture
def fake_chat():
    def make(*replies, error=None):
        return RecordingChat(replies=list(replies) or ["ok"], error=error, seen=[])
    return make


def tool_call(tool_name, /, **args):
    """An AIMessage where the model asks for one tool call instead of answering."""
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": f"call_{tool_name}"}])
