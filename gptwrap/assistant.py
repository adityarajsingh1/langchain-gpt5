"""
The Assistant: one system prompt plus whatever chat model you give it.
Takes a GPT-5 ChatOpenAI by default, but any LangChain chat model works,
which is how the tests run without an API key.
"""

import openai
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from gptwrap.personas import BUILT_IN, Persona, get_persona

DEFAULT_SYSTEM = BUILT_IN["default"].render()


class AssistantError(RuntimeError):
    pass


def _explain(err):
    # turn the openai SDK's exceptions into one line that says what to do about it
    if isinstance(err, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY."
    if isinstance(err, openai.NotFoundError):
        return "OpenAI doesn't know that model, or your key can't use it. Check OPENAI_MODEL."
    if isinstance(err, openai.RateLimitError):
        return "Hit OpenAI's rate limit (or ran out of credit). Wait a bit and try again."
    if isinstance(err, openai.APITimeoutError):
        return "OpenAI took too long to answer. Try again, or raise OPENAI_TIMEOUT."
    if isinstance(err, openai.APIConnectionError):
        return "Couldn't reach OpenAI. Check your internet connection (or OPENAI_BASE_URL)."
    return f"OpenAI returned an error: {err}"


class Assistant:
    """
    Remembers the conversation per session id, so one Assistant can talk to
    several people (or keep several topics apart) at once.
    max_turns limits how many earlier question/answer pairs are sent along,
    so long chats don't get slower and more expensive forever. None = keep all.

    With tools, the model can ask for a tool call instead of answering. The
    tool runs, its result goes back to the model, and that repeats until the
    model answers (or max_tool_steps is used up).
    """

    def __init__(self, model=None, system=DEFAULT_SYSTEM, max_turns=20, tools=None, max_tool_steps=5):
        if model is None:
            from gptwrap.models import chat_model
            model = chat_model()
        self.model = model
        self.system = system
        self.persona = None
        self.max_turns = max_turns
        self.tools = {t.name: t for t in tools or []}
        self.max_tool_steps = max_tool_steps
        # plain lists of messages per session. LangChain's own history classes
        # are deprecated in favour of LangGraph, which is overkill for this
        self._histories = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        self.llm = model.bind_tools(list(self.tools.values())) if self.tools else model

    @classmethod
    def from_persona(cls, persona, model=None, tools=None, **values):
        """Assistant.from_persona("tutor", subject="chemistry", level="high school")"""
        if not isinstance(persona, Persona):
            persona = get_persona(persona)
        bot = cls(model, system=persona.render(**values), tools=tools)
        bot.persona = persona.name
        return bot

    def ask(self, question, session="default"):
        question = question.strip()
        if not question:
            raise ValueError("ask() needs a question")
        # nothing gets saved when the call fails, so you can just ask again
        reply = "".join(self._turn(question, session, streaming=False)).strip()
        self._remember(session, question, reply)
        return reply

    def stream(self, question, session="default"):
        """Like ask(), but yields the answer piece by piece as the model writes it.
        The turn is only saved once the whole answer has arrived."""
        question = question.strip()
        if not question:
            raise ValueError("stream() needs a question")
        pieces = []
        for piece in self._turn(question, session, streaming=True):
            pieces.append(piece)
            yield piece
        self._remember(session, question, "".join(pieces).strip())

    def _turn(self, question, session, streaming):
        """One question: call the model, run any tools it asks for, repeat. Yields the answer text."""
        messages = self.prompt.invoke(self._inputs(question, session)).to_messages()
        for _ in range(self.max_tool_steps + 1):
            try:
                if streaming:
                    msg = None
                    for chunk in self.llm.stream(messages):
                        msg = chunk if msg is None else msg + chunk
                        if chunk.text:
                            yield chunk.text
                else:
                    msg = self.llm.invoke(messages)
            except openai.OpenAIError as e:
                raise AssistantError(_explain(e)) from e
            if not msg.tool_calls:
                if not streaming:
                    yield msg.text
                return
            messages.append(msg)
            messages.extend(self._run_tool(call) for call in msg.tool_calls)
        raise AssistantError(f"gave up after {self.max_tool_steps} rounds of tool calls without an answer")

    def _run_tool(self, call):
        """Run one tool call. Problems go back to the model as an error result instead of
        crashing the chat, so it can fix its arguments or answer without the tool."""
        tool = self.tools.get(call["name"])
        if tool is None:
            return self._tool_error(call, f"there's no tool called '{call['name']}'. "
                                          f"available: {', '.join(self.tools)}")
        try:
            return tool.invoke(call)  # given the whole tool call, a LangChain tool returns a ToolMessage
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return self._tool_error(call, f"bad arguments for {call['name']}: {problems}")
        except Exception as e:  # a broken tool shouldn't take the whole conversation down
            return self._tool_error(call, f"{call['name']} failed: {type(e).__name__}: {e}")

    @staticmethod
    def _tool_error(call, text):
        return ToolMessage(f"error: {text}", tool_call_id=call["id"], name=call["name"], status="error")

    def _inputs(self, question, session):
        return {"system": self.system, "history": self._histories.get(session, []), "question": question}

    def _remember(self, session, question, reply):
        history = self._histories.get(session, []) + [HumanMessage(question), AIMessage(reply)]
        if self.max_turns is not None:
            history = history[-2 * self.max_turns:]  # a turn is one question + one answer
        self._histories[session] = history

    def history(self, session="default"):
        """The saved messages of one session, oldest first."""
        return list(self._histories.get(session, []))

    def sessions(self):
        return [s for s, h in self._histories.items() if h]

    def reset(self, session="default"):
        """Forget one session. reset(None) forgets all of them."""
        if session is None:
            self._histories.clear()
        else:
            self._histories.pop(session, None)
