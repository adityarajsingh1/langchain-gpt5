import sys
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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
        text = self.replies[min(len(self.seen), len(self.replies)) - 1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@pytest.fixture
def fake_chat():
    def make(*replies, error=None):
        return RecordingChat(replies=list(replies) or ["ok"], error=error, seen=[])
    return make
