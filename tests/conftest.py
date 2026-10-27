import sys
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
import re

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RecordingChat(BaseChatModel):
    """Fake chat model: answers from a list (or raises), and remembers every prompt it got."""

    replies: list = []
    error: Exception | None = None
    seen: list = []

    @property
    def _llm_type(self):
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._next_reply()))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # streams the reply word by word (keeping the spaces), like a real model would
        self.seen.append(messages)
        if self.error:
            raise self.error
        for piece in re.findall(r"\S+\s*|\s+", self._next_reply()):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    def _next_reply(self):
        return self.replies[min(len(self.seen), len(self.replies)) - 1]


@pytest.fixture
def fake_chat():
    def make(*replies, error=None):
        return RecordingChat(replies=list(replies) or ["ok"], error=error, seen=[])
    return make
